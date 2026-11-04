#!/usr/bin/env python3
"""tp3_plan.py CKPT ARM REPO REVISION REF_PLAN KEYS_DIR OUT: the checkpoint as a LakyFx TP3 checkpoint plan.

TP=3 (docs/tp3.md). LakyFx's rank-file exporter (rank_cache.py, in the image's /opt/tp3/tools) cuts each rank's weights
from a plan (schema ds41f.tensorfold.checkpoint-plan.v1): every tensor of the checkpoint with its shard and byte range.
This writes one for CKPT, every tensor from the one source (LakyFx's TF29 arm is Mia's 2.9 bpw export alone too). The
plan names it under both sources the exporter takes (--pollard and --mia), and scripts/prepare-tp3.sh points both at
CKPT. Before it writes anything it checks, from the safetensors headers only:
  - every native key of rank0.keys, rank1.keys and rank2.keys (KEYS_DIR: LakyFx's frozen key order) is a tensor of CKPT
    (with their wo_a slice.8 -> slice.0 alias);
  - each keyed tensor has REF_PLAN's dtype and shape (LakyFx's TF29.json: Mia's earlier 2.9 bpw export), except the last
    axis of a trellis (an EXL3 tensor's bit width) and ffn.gate.bias, which may be F32 where the reference is F16 (the
    native keys read it as float32 either way);
  - a tensor stored in two shards (exllamav3 writes Engram's projections of layers 1 and 14 twice) is byte-identical in
    both, and the index's shard is the one used.
Mia's EAQ v1 2.9 bpw @ 45ee5d0f passes: 144,286 keyed tensors, none missing, no other geometry or dtype than TF29's
(its 40 ffn.gate.bias are F32). Unlike easthambuilds' make_plan.py this does not accept BF16 Engram k/q projections:
the image's exporter has no BF16 to float32 conversion, and this recipe's checkpoints store them as F32.
Prints a summary as JSON; exits 1 (and writes nothing) when the checkpoint does not fit.
Adapted from easthambuilds' tools/make_plan.py (easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold,
Apache-2.0).
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from pathlib import Path

PART = {"tr": "trellis", "suh": "suh", "svh": "svh"}


def headers(root: Path) -> dict:
    """Each shard's safetensors header (without __metadata__), by shard name, in name order."""
    doc = {}
    for shard in sorted(p.name for p in root.glob("*.safetensors")):
        with open(root / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        doc[shard] = h
    return doc


def payload_sha(root: Path, shard: str, spec: dict) -> str:
    with open(root / shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        lo, hi = spec["data_offsets"]
        f.seek(8 + n + lo)
        return hashlib.sha256(f.read(hi - lo)).hexdigest()


def keyed_names(keys_dir: Path, ref: dict, tensors: dict) -> set:
    """The checkpoint tensor each native key reads (rank_cache.py's layout(): name|cast, or name|cols|rows|part)."""
    out = set()
    for r in range(3):
        for key in (keys_dir / f"rank{r}.keys").read_text().splitlines():
            f = key.split("|")
            name = f[0] + ("." + PART[f[3]] if len(f) == 4 else "")
            if name not in ref and name not in tensors:
                name = name.replace(".attn.wo_a.slice.8.", ".attn.wo_a.slice.0.")
            out.add(name)
    return out


def fits(t: dict, w: dict, name: str) -> bool:
    if name.endswith(".trellis"):
        return t["dtype"] == w["dtype"] and t["shape"][:-1] == w["shape"][:-1]
    if name.endswith(".ffn.gate.bias"):
        return t["shape"] == w["shape"] and t["dtype"] in ("F16", "F32") and w["dtype"] in ("F16", "F32")
    return t["dtype"] == w["dtype"] and t["shape"] == w["shape"]


def plan(ckpt: Path, arm: str, repo: str, rev: str, ref_path: Path, keys_dir: Path) -> tuple[dict, dict]:
    hdr = headers(ckpt)
    raw = json.dumps({"repo": repo, "revision": rev, "headers": hdr}, sort_keys=True, separators=(",", ":")).encode()
    index = json.loads((ckpt / "model.safetensors.index.json").read_bytes())["weight_map"]
    tensors, dups = {}, []
    for shard, h in hdr.items():
        for name, spec in h.items():
            lo, hi = spec["data_offsets"]
            entry = dict(spec, shard=shard, bytes=hi - lo, source="mia")
            if name in tensors:
                u = tensors[name]
                if (u["dtype"], u["shape"]) != (spec["dtype"], spec["shape"]) or \
                        payload_sha(ckpt, u["shard"], u) != payload_sha(ckpt, shard, spec):
                    sys.exit(f"tensor {name} is in two shards with different bytes")
                dups.append(name)
                if index.get(name) != shard:
                    continue
            tensors[name] = entry
    off = [n for n in tensors if index.get(n) != tensors[n]["shard"]]
    if off:
        sys.exit(f"{len(off)} tensors are not in the shard the index names (first: {off[0]})")
    ref = json.loads(ref_path.read_bytes())["tensors"]
    keyed = keyed_names(keys_dir, ref, tensors)
    missing = sorted(n for n in keyed if n not in tensors)
    bad = [(n, tensors[n]["dtype"], tensors[n]["shape"], ref[n]["dtype"], ref[n]["shape"])
           for n in sorted(keyed) if n in tensors and n in ref and not fits(tensors[n], ref[n], n)]
    only_ours, only_ref = sorted(set(tensors) - set(ref)), sorted(set(ref) - set(tensors))
    summary = {"keyed_tensors": len(keyed), "identical_duplicates": len(dups), "n_missing": len(missing),
               "missing": missing[:20], "n_bad": len(bad), "geometry_mismatch": bad[:20],
               "only_ours": len(only_ours), "only_ours_sample": only_ours[:12],
               "only_ref": len(only_ref), "only_ref_sample": only_ref[:12]}
    src = {"repo": repo, "revision": rev, "headers_sha256": hashlib.sha256(raw).hexdigest()}
    doc = {"schema": "ds41f.tensorfold.checkpoint-plan.v1", "arm": arm, "stage": "PRE_PADDING_SOURCE_SELECTION",
           "serving_ready": False, "sources": {"pollard": src, "mia": src},
           "stored_bytes_by_source": {"pollard": 0, "mia": sum(v["bytes"] for v in tensors.values())},
           "tensors": tensors}
    return summary, doc


def main(argv: list[str]) -> int:
    if len(argv) != 8:
        print(__doc__.split("\n")[0], file=sys.stderr)
        return 2
    ckpt, arm, repo, rev, ref_path, keys_dir, out = argv[1:]
    summary, doc = plan(Path(ckpt), arm, repo, rev, Path(ref_path), Path(keys_dir))
    print(json.dumps(summary, indent=1))
    if summary["n_missing"] or summary["n_bad"]:
        print("NOT WRITTEN: the checkpoint does not fit LakyFx's TP3 layout", file=sys.stderr)
        return 1
    data = json.dumps(doc, separators=(",", ":")).encode()
    with open(out, "xb") as f:
        f.write(data)
    print("WROTE", out, hashlib.sha256(data).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
