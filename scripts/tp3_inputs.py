#!/usr/bin/env python3
"""tp3_inputs.py STEP [args]: TP=3's inputs beside the plan (scripts/prepare-tp3.sh runs each step in the image; CPU
only, never a GPU). Standard library, plus LakyFx's preparation tools for `model` and `receipt-ok` (--tools: the image's
/opt/tp3/tools). Every output must be new.

  manifest CKPT REPO REV OUT [--snapshot DIR]
        OUT: a source manifest of CKPT's shards in the form LakyFx's source_verify.py reads (the Hub's model API form:
        {"id": REPO, "sha": REV, "siblings": [{"rfilename", "size", "lfs": {"sha256"}}]}), from the files themselves,
        so that it also serves the ABLIT=1 checkpoint, which is on no Hub: the grafted shards are hashed here, and are
        never fetched or "repaired" from the Hub. A shard that is the same file (a hard link) as DIR's shard of that
        name, where DIR is a Hugging Face snapshot, takes the sha256 its blob is named by (the Hub's LFS sha256) instead
        of a second read. source_verify.py then hashes every shard against it either way.
  receipt-ok --tools T RECEIPT ROOT REPO REV
        exit 0 when LakyFx's receipt (source_verify.py's output) still holds for every shard of ROOT: the same files,
        not touched since (check_receipt: size, times, inode); 1 otherwise (prepare-tp3.sh then verifies again).
  model --tools T CKPT PLAN RECEIPT TEMPLATE ORIG_REV OUT ARM
        OUT: the model folder LakyFx's engine serves at TP=3: CKPT's config.json for three ranks (LakyFx's
        tp3_layout.patch_config: 72 heads in 9 groups and a padded vocabulary, with the arm, the plan's sha256 and its
        sources, and the base model's revision ORIG_REV), its tokenizer, the recipe's chat template TEMPLATE, and the
        vision tower: its 266 BF16 tensors in vision.safetensors, every byte compared with CKPT's (LakyFx's
        vision_export.compare_payload), and their index.
  rope-cut REF KIT POOL HASHES
        KIT's RoPE tables for a POOL-position pool: the first ROWS rows (POOL + 49, in whole 4,096-row blocks) of REF's
        tables, which LakyFx's make_rope.py wrote at its 10,000,000-position reference pool (checked against its
        qualified hashes then; their own hashes are checked again here). make_rope computes rows in fixed blocks, so
        the cut equals a table made for the smaller pool: a cut to a row count HASHES (LakyFx's rope-hashes.json) lists,
        3,301,376 for the 3,300,000 pool, must match its hashes.
  engram-source DIR RANK ORIG_REPO ORIG_REV NAME=SHA256...
        DIR/SOURCE.json for a rank's Engram rows (DIR: LakyFx's prepare_engram.py output for RANK, a sparse copy of that
        rank's 8 of the 24 hash columns of layers 1 and 14): the rank, the world, the source revision, and each shard's
        size and source sha256 (the base model card's, which prepare.sh checks when VERIFY_SHA=1).
Adapted from easthambuilds' tools/make_arm.py, tools/rope_cut.py and tools/engram_source.py
(easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold, Apache-2.0).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

CHUNK = 64 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def tools_path(args: list[str]) -> list[str]:
    """--tools T at the front: put T (LakyFx's tools/dsv41-tp3) on the import path."""
    if args[:1] != ["--tools"] or len(args) < 2:
        sys.exit("--tools <LakyFx's tools/dsv41-tp3> comes first")
    sys.path.insert(0, args[1])
    return args[2:]


def manifest(args: list[str]) -> int:
    snapshot = None
    if "--snapshot" in args:
        i = args.index("--snapshot")
        snapshot = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    ckpt, repo, rev, out = Path(args[0]), args[1], args[2], Path(args[3])
    siblings, reused = [], 0
    for shard in sorted(p.name for p in ckpt.glob("*.safetensors")):
        path = ckpt / shard
        if path.is_symlink():
            sys.exit(f"{path} is a symbolic link: LakyFx's tools take the checkpoint's own files (hard links)")
        digest = None
        if snapshot is not None and (snapshot / shard).exists() and os.path.samefile(path, snapshot / shard):
            blob = os.path.basename(os.path.realpath(snapshot / shard))
            if re.fullmatch(r"[0-9a-f]{64}", blob):
                digest, reused = blob, reused + 1
        if digest is None:
            digest = sha256_file(path)
        siblings.append({"rfilename": shard, "size": path.stat().st_size, "lfs": {"sha256": digest}})
    if not siblings:
        sys.exit(f"{ckpt} holds no .safetensors shards")
    with open(out, "x") as f:
        json.dump({"id": repo, "sha": rev, "siblings": siblings}, f, indent=1)
    print(f"{len(siblings)} shards listed ({reused} by their Hugging Face blob's sha256, {len(siblings) - reused} hashed)")
    return 0


def receipt_ok(args: list[str]) -> int:
    args = tools_path(args)
    from source_verify import check_receipt
    receipt, root, repo, rev = json.loads(Path(args[0]).read_bytes()), Path(args[1]), args[2], args[3]
    shards = {p.name for p in root.glob("*.safetensors")}
    try:
        if set(receipt.get("files", {})) != shards:
            raise ValueError("the receipt lists other shards than the checkpoint holds")
        check_receipt(receipt, root, {"repo": repo, "revision": rev}, shards)
    except (ValueError, OSError, KeyError) as e:
        print(f"receipt does not hold: {e}")
        return 1
    print(f"receipt holds for {len(shards)} shards")
    return 0


def model(args: list[str]) -> int:
    args = tools_path(args)
    from materialize_checkpoint import write_shard
    from source_verify import check_receipt
    from tp3_layout import patch_config
    from vision_export import compare_payload, selected
    ckpt, plan_path, receipt_path, template = Path(args[0]), Path(args[1]), Path(args[2]), Path(args[3])
    orig, out, arm = args[4], Path(args[5]), args[6]
    plan = json.loads(plan_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    items = selected(plan)
    shard = next(iter(items.values()))["shard"]
    check_receipt(receipt, ckpt, plan["sources"]["mia"], {shard})
    out.mkdir(parents=True)
    cfg = patch_config(json.loads((ckpt / "config.json").read_text()))
    cfg["tensorfold_tp3"].update(arm=arm, weight_plan_sha256=sha256_file(plan_path), sources=plan["sources"],
                                 original_revision=orig)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(ckpt / name, out / name)
    shutil.copyfile(template, out / "chat_template.jinja")
    target = out / "vision.safetensors"
    header, payload = write_shard(ckpt / shard, target, items, read_mib_s=0)
    compared = compare_payload(ckpt / shard, target, items, read_mib_s=0)
    check_receipt(receipt, ckpt, plan["sources"]["mia"], {shard})
    with open(out / "model.safetensors.index.json", "x") as f:
        json.dump({"metadata": {"total_size": payload}, "weight_map": {n: "vision.safetensors" for n in header}}, f)
    print(json.dumps({"vision_entries": len(items), "payload": payload, "compared": compared,
                      "vision_sha256": sha256_file(target), "config_sha256": sha256_file(out / "config.json")}))
    return 0


def rope_cut(args: list[str]) -> int:
    ref, kit, pool, hashes = Path(args[0]), Path(args[1]), int(args[2]), json.loads(Path(args[3]).read_text())
    rows = (pool + 49 + 4095) // 4096 * 4096
    full = json.loads((ref / "rope.json").read_text())
    if full["rows"] < rows:
        sys.exit(f"the reference tables have {full['rows']} rows; a {pool}-position pool needs {rows}")
    pins = next((h for h in hashes.values() if h["rows"] == rows), None)
    result = {"rows": rows, "half": full["half"]}
    for kind in ("plain", "compressed"):
        for part in ("cos", "sin"):
            name = f"rope-{kind}-{part}.f32"
            if sha256_file(ref / name) != full[name]:
                sys.exit(f"{name}: the reference table changed since make_rope.py wrote it")
            left, h = rows * full["half"] * 4, hashlib.sha256()
            with open(ref / name, "rb") as src, open(kit / name, "xb") as dst:
                while left:
                    block = src.read(min(CHUNK, left))
                    if not block:
                        sys.exit(f"{name}: the reference table is short")
                    dst.write(block)
                    h.update(block)
                    left -= len(block)
                dst.flush()
                os.fsync(dst.fileno())
            result[name] = h.hexdigest()
            if pins is not None and pins[name] != result[name]:
                sys.exit(f"{name}: the {rows}-row cut differs from LakyFx's qualified table of that size")
    (kit / "rope.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"RoPE tables cut to {rows} rows for a {pool}-position pool"
          + (" (LakyFx's qualified hashes match)" if pins is not None else ""))
    return 0


def engram_source(args: list[str]) -> int:
    d, rank, repo, rev = Path(args[0]), int(args[1]), args[2], args[3]
    pinned = dict(a.split("=", 1) for a in args[4:])
    files = {p.name: {"size": p.stat().st_size, "source_sha256": pinned.get(p.name) or None}
             for p in sorted(d.glob("*.safetensors"))}
    if sorted(files) != sorted(pinned):
        sys.exit(f"{d}: expected the Engram shards {sorted(pinned)}, found {sorted(files)}")
    doc = {"rank": rank, "world": 3, "original_repo": repo, "original_revision": rev, "files": files,
           "owned_rows": json.loads((d / "engram-local.json").read_text()), "note": "sparse rank copy; unowned rows are holes"}
    with open(d / "SOURCE.json", "x") as f:
        json.dump(doc, f, indent=1)
    print(f"rank {rank}: SOURCE.json written")
    return 0


STEPS = {"manifest": manifest, "receipt-ok": receipt_ok, "model": model, "rope-cut": rope_cut,
         "engram-source": engram_source}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(STEPS[sys.argv[1]](sys.argv[2:]))
