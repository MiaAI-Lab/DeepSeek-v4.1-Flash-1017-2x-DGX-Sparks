"""Small binary fixtures exercise graft integrity without PyTorch or model downloads."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import ablit_validate as validator


def save(path, tensors):
    h, payload = {}, b""
    for name, (dtype, shape, data) in tensors.items():
        h[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    raw = json.dumps(h).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class IntegrityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.stock, self.graft, self.side = root / "stock", root / "graft", root / "side.safetensors"
        self.stock.mkdir()
        self.graft.mkdir()
        self.target = "layers.10.attn.wo_b.suh"
        self.bf16 = "layers.14.attn.indexer.wk.weight"
        self.mtp = "mtp.2.confidence_head.proj.weight"
        self.tensors = {
            self.target: ("F16", [2], b"\x00\x3c\x00\x40"),
            self.bf16: ("BF16", [1], b"\x80\x3f"),
            self.mtp: ("F16", [1], b"\x00\x3c"),
        }
        self.changed = dict(self.tensors)
        self.changed[self.target] = ("F16", [2], b"\x00\x40\x00\x42")
        self.shard = "model-00001-of-00001.safetensors"
        for d in (self.stock, self.graft):
            (d / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {k: self.shard for k in self.tensors}}))
            (d / "config.json").write_text(json.dumps({
                "quantization_config": {"quant_method": "exl3", "codebook": "mul1", "out_scales": "always"}}))
            (d / "tokenizer.json").write_text('{"fixture":true}')
        save(self.stock / self.shard, self.tensors)
        save(self.graft / self.shard, self.changed)
        save(self.side, {self.target: self.changed[self.target]})
        p = patch.object(validator, "TARGETS", {self.target: ("F16", [2])})
        p.start()
        self.addCleanup(p.stop)

    def run_check(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            validator.validate(self.stock, self.side, self.graft)
        return out.getvalue()

    def test_graft_preserves_bf16_mtp_and_receipt_survives_directory_move(self):
        before = (self.stock / self.shard).read_bytes()
        self.run_check()
        moved = self.graft.with_name("checkpoint")
        self.graft.rename(moved)
        self.graft = moved
        self.assertIn("receipt unchanged", self.run_check())
        self.assertEqual(before, (self.stock / self.shard).read_bytes())

    def test_same_shape_wrong_graft_bytes_rejected_after_receipt(self):
        self.run_check()
        self.changed[self.target] = self.tensors[self.target]
        save(self.graft / self.shard, self.changed)
        with self.assertRaisesRegex(ValueError, "tensor bytes differ"):
            self.run_check()

    def test_non_target_bf16_and_mtp_changes_rejected(self):
        for name in (self.bf16, self.mtp):
            with self.subTest(name=name):
                bad = dict(self.changed)
                dtype, shape, _ = bad[name]
                bad[name] = dtype, shape, b"\0\0"
                save(self.graft / self.shard, bad)
                with self.assertRaisesRegex(ValueError, "tensor bytes differ"):
                    self.run_check()

    def test_dtype_mismatch_rejected(self):
        self.changed[self.bf16] = "F16", [1], b"\x80\x3f"
        save(self.graft / self.shard, self.changed)
        with self.assertRaisesRegex(ValueError, "dtype or shape"):
            self.run_check()

    def test_missing_and_extra_sidecar_keys_rejected(self):
        for tensors in ({}, {**self.changed}):
            save(self.side, tensors)
            with self.assertRaisesRegex(ValueError, "exactly the 104"):
                self.run_check()

    def test_wrong_target_shape_rejected(self):
        save(self.side, {self.target: ("F16", [1, 2], b"\0" * 4)})
        with self.assertRaisesRegex(ValueError, "incompatible target"):
            self.run_check()

    def test_truncated_shard_rejected(self):
        p = self.graft / self.shard
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.run_check()

    def test_tokenizer_change_rejected(self):
        (self.graft / "tokenizer.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed asset"):
            self.run_check()

    def test_hardlink_cannot_substitute_for_graft(self):
        (self.graft / self.shard).unlink()
        os.link(self.stock / self.shard, self.graft / self.shard)
        with self.assertRaisesRegex(ValueError, "tensor bytes differ"):
            self.run_check()


if __name__ == "__main__":
    unittest.main()
