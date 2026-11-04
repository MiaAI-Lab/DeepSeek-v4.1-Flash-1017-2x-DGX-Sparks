#!/usr/bin/env python3
"""Tests for TP=3's Python tools (docs/tp3.md), offline, on small fake checkpoints: no Spark, GPU, Docker or network.

  Plan          scripts/tp3_plan.py: a fitting checkpoint gives a plan of every tensor from one source (both of the
                exporter's source names), the wo_a slice.8 alias and an F32 ffn.gate.bias are accepted; a missing keyed
                tensor, another shape, another dtype or a BF16 Engram projection refuse and write nothing; a tensor in
                two shards must be byte-identical there; the same inputs give the same bytes.
  Manifest      scripts/tp3_inputs.py manifest: each shard's size and sha256, hashed, or the sha256 its Hugging Face blob
                is named by when it is the same file as the snapshot's (a hard link); a symbolic link refuses.
  RopeCut       rope-cut: the first rows of the reference tables, checked against the reference's own hashes and
                against the pinned hashes of a pool of that size.
  EngramSource  engram-source: SOURCE.json with the rank, the world, each shard's size and source sha256.
  ReceiptAndModel (only with TP3_TOOLS=<LakyFx's tools/dsv41-tp3>, e.g. from the image's /opt/tp3/tools): receipt-ok
                against a receipt LakyFx's source_verify.py writes, before and after a shard is touched.
Run: python3 scripts/test_tp3_inputs.py
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE / "tp3_plan.py"
INPUTS = HERE / "tp3_inputs.py"
TOOLS = os.environ.get("TP3_TOOLS", "")


def shard(path, tensors):
    """A safetensors file: {name: (dtype, shape, payload bytes)}."""
    header, body = {}, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(body), len(body) + len(data)]}
        body += data
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + body)


def run(*args):
    r = subprocess.run([sys.executable, "-I", *map(str, args)], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


class Tmp(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.tmp = Path(d.name)


class Plan(Tmp):
    # the reference plan's tensors (LakyFx's TF29.json form) and three ranks' native keys over them
    REF = {"layers.0.attn.wo_a.slice.0.trellis": ("I16", [8, 4, 64]), "layers.0.attn.wo_a.slice.0.suh": ("F16", [128]),
           "layers.0.ffn.gate.bias": ("F16", [4]), "layers.1.engram.k_weight": ("F32", [4, 8]),
           "embed.weight": ("BF16", [8, 8])}
    KEYS = ["layers.0.attn.wo_a.slice.8|None|(0, 128)|tr", "layers.0.attn.wo_a.slice.8|None|(0, 128)|suh",
            "layers.0.ffn.gate.bias|torch.float32", "layers.1.engram.k_weight|None", "embed.weight|None"]

    def setUp(self):
        super().setUp()
        self.ckpt = self.tmp / "ckpt"
        self.ckpt.mkdir()
        (self.tmp / "keys").mkdir()
        for r in range(3):
            (self.tmp / "keys" / f"rank{r}.keys").write_text("\n".join(self.KEYS[r::3]) + "\n")
        ref = {n: {"dtype": d, "shape": s} for n, (d, s) in self.REF.items()}
        (self.tmp / "ref.json").write_text(json.dumps({"tensors": ref}))

    def write(self, overrides=None, extra=None, drop=()):
        t = {n: (d, s) for n, (d, s) in self.REF.items() if n not in drop}
        t["layers.0.attn.wo_a.slice.0.trellis"] = ("I16", [8, 4, 48])   # another bit width: the last axis only
        t["layers.0.ffn.gate.bias"] = ("F32", [4])
        t.update(overrides or {})
        size = {"I16": 2, "F16": 2, "BF16": 2, "F32": 4}

        def nbytes(d, s):
            n = size[d]
            for x in s:
                n *= x
            return n
        a = {n: (d, s, bytes(nbytes(d, s))) for n, (d, s) in t.items() if not n.startswith("embed")}
        b = {n: (d, s, bytes(nbytes(d, s))) for n, (d, s) in t.items() if n.startswith("embed")}
        b.update(extra or {})
        shard(self.ckpt / "model-00001-of-00002.safetensors", a)
        shard(self.ckpt / "model-00002-of-00002.safetensors", b)
        wm = {n: "model-00001-of-00002.safetensors" for n in a} | {n: "model-00002-of-00002.safetensors" for n in b}
        (self.ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))

    def plan(self, out="plan.json"):
        return run(PLAN, self.ckpt, "EAQ29", "org/repo", "abc123", self.tmp / "ref.json", self.tmp / "keys", self.tmp / out)

    def test_a_fitting_checkpoint(self):
        self.write()
        code, out = self.plan()
        self.assertEqual(code, 0, out)
        self.assertIn('"keyed_tensors": 5', out)
        doc = json.loads((self.tmp / "plan.json").read_bytes())
        self.assertEqual(doc["schema"], "ds41f.tensorfold.checkpoint-plan.v1")
        self.assertEqual(doc["arm"], "EAQ29")
        self.assertEqual(doc["sources"]["mia"], doc["sources"]["pollard"])
        self.assertEqual(doc["sources"]["mia"]["repo"], "org/repo")
        self.assertEqual({t["source"] for t in doc["tensors"].values()}, {"mia"})
        self.assertEqual(doc["tensors"]["embed.weight"]["shard"], "model-00002-of-00002.safetensors")
        self.assertEqual(doc["stored_bytes_by_source"]["mia"], sum(t["bytes"] for t in doc["tensors"].values()))
        code, out = self.plan("again.json")
        self.assertEqual((self.tmp / "plan.json").read_bytes(), (self.tmp / "again.json").read_bytes())
        self.assertIn("WROTE", out)
        self.assertIn(hashlib.sha256((self.tmp / "plan.json").read_bytes()).hexdigest(), out)

    def test_what_does_not_fit_is_refused_and_nothing_written(self):
        for name, over in {"missing": {"embed.weight": None}, "shape": {"embed.weight": ("BF16", [8, 9])},
                           "dtype": {"layers.0.attn.wo_a.slice.0.suh": ("F32", [128])},
                           "BF16 Engram projection": {"layers.1.engram.k_weight": ("BF16", [4, 8])},
                           "trellis rows": {"layers.0.attn.wo_a.slice.0.trellis": ("I16", [9, 4, 64])}}.items():
            with self.subTest(case=name):
                for p in self.ckpt.iterdir():
                    p.unlink()
                self.write({k: v for k, v in over.items() if v is not None}, drop={k for k, v in over.items() if v is None})
                out_name = f"bad-{len(name)}.json"
                code, out = self.plan(out_name)
                self.assertEqual(code, 1, out)
                self.assertIn("NOT WRITTEN", out)
                self.assertFalse((self.tmp / out_name).exists())

    def test_a_tensor_in_two_shards_must_be_the_same_bytes(self):
        dup = {"layers.1.engram.k_weight": ("F32", [4, 8], bytes(128))}
        self.write(extra=dup)
        code, out = self.plan()
        self.assertEqual(code, 0, out)
        self.assertIn('"identical_duplicates": 1', out)
        for p in self.ckpt.iterdir():
            p.unlink()
        self.write(extra={"layers.1.engram.k_weight": ("F32", [4, 8], b"\x01" * 128)})
        code, out = self.plan("other.json")
        self.assertEqual(code, 1, out)
        self.assertIn("two shards with different bytes", out)


class Manifest(Tmp):
    def test_hashes_or_reuses_the_blob_name(self):
        blobs, snap, ck = self.tmp / "blobs", self.tmp / "snap", self.tmp / "ck"
        for d in (blobs, snap, ck):
            d.mkdir()
        same = b"unchanged shard"
        digest = hashlib.sha256(same).hexdigest()
        (blobs / digest).write_bytes(same)
        (snap / "model-00001.safetensors").symlink_to(blobs / digest)
        os.link(blobs / digest, ck / "model-00001.safetensors")            # the hard link the TP=3 folder holds
        (ck / "model-00002.safetensors").write_bytes(b"grafted shard")   # a shard the overlay wrote
        (blobs / ("0" * 64)).write_bytes(b"stock shard 2")
        (snap / "model-00002.safetensors").symlink_to(blobs / ("0" * 64))
        (ck / "config.json").write_text("{}")
        code, out = run(INPUTS, "manifest", ck, "org/repo+overlay", "abc+def", self.tmp / "m.json", "--snapshot", snap)
        self.assertEqual(code, 0, out)
        self.assertIn("2 shards listed (1 by their Hugging Face blob's sha256, 1 hashed)", out)
        doc = json.loads((self.tmp / "m.json").read_text())
        self.assertEqual((doc["id"], doc["sha"]), ("org/repo+overlay", "abc+def"))
        self.assertEqual(doc["siblings"], [
            {"rfilename": "model-00001.safetensors", "size": len(same), "lfs": {"sha256": digest}},
            {"rfilename": "model-00002.safetensors", "size": 13, "lfs": {"sha256": hashlib.sha256(b"grafted shard").hexdigest()}}])
        code, out = run(INPUTS, "manifest", ck, "r", "v", self.tmp / "m.json")
        self.assertNotEqual(code, 0, "an existing output is never overwritten")

    def test_a_symbolic_link_refuses(self):
        ck = self.tmp / "ck"
        ck.mkdir()
        (self.tmp / "x").write_bytes(b"x")
        (ck / "model-00001.safetensors").symlink_to(self.tmp / "x")
        code, out = run(INPUTS, "manifest", ck, "r", "v", self.tmp / "m.json")
        self.assertEqual(code, 1, out)
        self.assertIn("is a symbolic link", out)
        self.assertFalse((self.tmp / "m.json").exists())


class RopeCut(Tmp):
    def setUp(self):
        super().setUp()
        self.ref, self.kit = self.tmp / "ref", self.tmp / "kit"
        self.ref.mkdir()
        self.kit.mkdir()
        rows, half = 8192 + 4096, 2
        doc = {"rows": rows, "half": half}
        for kind in ("plain", "compressed"):
            for part in ("cos", "sin"):
                name = f"rope-{kind}-{part}.f32"
                data = bytes((i * 7 + len(name)) % 251 for i in range(rows * half * 4))
                (self.ref / name).write_bytes(data)
                doc[name] = hashlib.sha256(data).hexdigest()
        (self.ref / "rope.json").write_text(json.dumps(doc))
        self.cut = {n: hashlib.sha256((self.ref / n).read_bytes()[:8192 * half * 4]).hexdigest() for n in doc if n.endswith(".f32")}

    def hashes(self, pins):
        (self.tmp / "h.json").write_text(json.dumps({"pool": dict(rows=8192, half=2, **pins)}))
        return self.tmp / "h.json"

    def test_the_first_rows_with_the_pinned_hashes(self):
        code, out = run(INPUTS, "rope-cut", self.ref, self.kit, 8000, self.hashes(self.cut))
        self.assertEqual(code, 0, out)
        self.assertIn("cut to 8192 rows for a 8000-position pool (LakyFx's qualified hashes match)", out)
        doc = json.loads((self.kit / "rope.json").read_text())
        self.assertEqual(doc, {"rows": 8192, "half": 2, **self.cut})
        self.assertEqual((self.kit / "rope-plain-cos.f32").stat().st_size, 8192 * 2 * 4)

    def test_a_cut_unlike_the_pin_refuses(self):
        pins = dict(self.cut, **{"rope-plain-sin.f32": "0" * 64})
        code, out = run(INPUTS, "rope-cut", self.ref, self.kit, 8000, self.hashes(pins))
        self.assertEqual(code, 1, out)
        self.assertIn("rope-plain-sin.f32: the 8192-row cut differs", out)
        self.assertFalse((self.kit / "rope.json").exists())

    def test_a_changed_reference_or_a_pool_too_large_refuses(self):
        code, out = run(INPUTS, "rope-cut", self.ref, self.kit, 20000, self.hashes(self.cut))
        self.assertEqual(code, 1, out)
        self.assertIn("needs 20480", out)
        with open(self.ref / "rope-compressed-cos.f32", "r+b") as f:
            f.write(b"\xff")
        code, out = run(INPUTS, "rope-cut", self.ref, self.kit, 8000, self.hashes(self.cut))
        self.assertEqual(code, 1, out)
        self.assertIn("changed since make_rope.py wrote it", out)


class EngramSource(Tmp):
    def test_source_json(self):
        d = self.tmp / "engram-r2"
        d.mkdir()
        for n, size in (("model-00047-of-00048.safetensors", 10), ("model-00048-of-00048.safetensors", 20)):
            with open(d / n, "wb") as f:
                f.truncate(size)                                         # sparse, as prepare_engram.py writes them
        (d / "engram-local.json").write_text(json.dumps({"layers": {"1": [16, 24], "14": [16, 24]}}))
        code, out = run(INPUTS, "engram-source", d, 2, "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e10",
                        "model-00047-of-00048.safetensors=aa", "model-00048-of-00048.safetensors=bb")
        self.assertEqual(code, 0, out)
        doc = json.loads((d / "SOURCE.json").read_text())
        self.assertEqual((doc["rank"], doc["world"], doc["original_revision"]), (2, 3, "1bae0e10"))
        self.assertEqual(doc["files"]["model-00048-of-00048.safetensors"], {"size": 20, "source_sha256": "bb"})
        self.assertEqual(doc["owned_rows"]["layers"]["14"], [16, 24])
        code, out = run(INPUTS, "engram-source", d, 2, "r", "v", "model-00047-of-00048.safetensors=aa")
        self.assertEqual(code, 1, out)


@unittest.skipUnless(TOOLS and Path(TOOLS, "source_verify.py").exists(), "TP3_TOOLS=<LakyFx's tools/dsv41-tp3> not set")
class ReceiptAndModel(Tmp):
    def test_receipt_ok_until_a_shard_is_touched(self):
        ck = self.tmp / "ck"
        ck.mkdir()
        shard(ck / "model-00001.safetensors", {"a": ("F16", [2], b"\0" * 4)})
        code, out = run(INPUTS, "manifest", ck, "org/repo", "abc", self.tmp / "m.json")
        self.assertEqual(code, 0, out)
        r = subprocess.run([sys.executable, "-E", "-s", f"{TOOLS}/source_verify.py", "--manifest", self.tmp / "m.json",
                            "--root", ck, "--out", self.tmp / "receipt.json", "--weights-only"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        ok = lambda rev="abc": run(INPUTS, "receipt-ok", "--tools", TOOLS, self.tmp / "receipt.json", ck, "org/repo", rev)
        self.assertEqual(ok()[0], 0, ok()[1])
        self.assertEqual(ok("other")[0], 1)
        os.link(ck / "model-00001.safetensors", self.tmp / "another-link")   # a hard link more: its ctime moves
        code, out = ok()
        self.assertEqual(code, 1, out)
        self.assertIn("receipt does not hold", out)


if __name__ == "__main__":
    unittest.main()
