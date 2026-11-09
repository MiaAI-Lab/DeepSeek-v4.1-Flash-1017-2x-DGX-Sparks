#!/usr/bin/env python3
"""Validate Keys' Mia graft before preparation and verify its output before cache generation.

Usage: ablit_validate.py STOCK SIDECAR [GRAFT]
Standard library only; streams tensor bytes with bounded memory. Headers must describe the
same tensors and dtypes, targets must equal the sidecar, everything else must equal stock.
A receipt avoids rereading rewritten shards when file identities and this validator match.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

SCHEMA = {
    "trellis": ("I16", [512, 320, 80]),
    "suh": ("F16", [8192]),
    "svh": ("F16", [5120]),
    "mul1": ("I32", []),
}
TARGETS = {f"layers.{i}.attn.wo_b.{s}": v for i in range(10, 36) for s, v in SCHEMA.items()}
CHUNK = 8 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def header(path):
    with path.open("rb") as f:
        raw = f.read(8)
        require(len(raw) == 8, f"short safetensors header: {path}")
        size = struct.unpack("<Q", raw)[0]
        require(0 < size <= 64 * 1024 * 1024, f"invalid header size: {path}")
        h = json.loads(f.read(size))
    h.pop("__metadata__", None)
    end = 8 + size
    for name, t in sorted(h.items(), key=lambda item: item[1]["data_offsets"]):
        a, b = t["data_offsets"]
        require(8 + size + a == end and b >= a, f"invalid tensor offsets: {path}: {name}")
        end = 8 + size + b
    require(end == path.stat().st_size, f"truncated or extra tensor data: {path}")
    return h, 8 + size


def index(root):
    wm = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    require(all(Path(s).name == s for s in wm.values()), "invalid shard filename in index")
    headers = {s: header(root / s) for s in set(wm.values())}
    actual = {k: s for s, (h, _) in headers.items() for k in h}
    require(actual == wm and sum(len(h) for h, _ in headers.values()) == len(wm),
            f"index and shard tensors differ: {root}")
    return wm, headers


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def compare_bytes(left, lo, right, ro, size, name):
    left.seek(lo)
    right.seek(ro)
    while size:
        n = min(size, CHUNK)
        a, b = left.read(n), right.read(n)
        require(len(a) == n and a == b, f"tensor bytes differ: {name}")
        size -= n


def validate(stock, sidecar, graft=None):
    wm, hs = index(stock)
    hside, base_side = header(sidecar)
    require(set(hside) == set(TARGETS), "sidecar must contain exactly the 104 L10-35 wo_b tensors")
    q = json.loads((stock / "config.json").read_text())["quantization_config"]
    require(q.get("quant_method") == "exl3" and q.get("codebook") == "mul1"
            and q.get("out_scales") == "always", "requires EXL3 mul1 with output scales")
    for k, (dtype, shape) in TARGETS.items():
        require(k in wm, f"missing stock target: {k}")
        for h in (hside, hs[wm[k]][0]):
            t = h[k]
            require((t["dtype"], t["shape"]) == (dtype, shape), f"incompatible target: {k}")
            width = 4 if dtype == "I32" else 2
            require(t["data_offsets"][1] - t["data_offsets"][0] == math.prod(shape) * width,
                    f"invalid target byte size: {k}")
    print("Compatible: 104 EXL3 mul1 targets, identical shapes and dtypes", flush=True)
    if graft is None:
        return
    wa, ha = index(graft)
    require(wm == wa, "graft changed the stock weight map")
    for s, (h, _) in hs.items():
        require(set(h) == set(ha[s][0]), f"graft changed tensor names: {s}")
        for k, t in h.items():
            a = ha[s][0][k]
            require((t["dtype"], t["shape"]) == (a["dtype"], a["shape"]),
                    f"graft changed stock dtype or shape: {k}")
            require(t["data_offsets"][1] - t["data_offsets"][0]
                    == a["data_offsets"][1] - a["data_offsets"][0], f"graft changed tensor length: {k}")
    # The upstream applicator copies these files; check config, index, tokenizer and all other visible assets.
    assets = sorted(p.name for p in stock.iterdir()
                    if p.is_file() and not p.name.startswith(".") and p.suffix != ".safetensors")
    for name in assets:
        require((stock / name).read_bytes() == (graft / name).read_bytes(), f"graft changed asset: {name}")
    identity = {
        "validator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sidecar": signature(sidecar),
        "stock": {s: signature(stock / s) for s in sorted(set(wm.values()) | set(assets))},
        "graft": {s: signature(graft / s) for s in sorted(set(wm.values()) | set(assets))},
    }
    receipt = graft / ".ablit-verified.json"
    if receipt.is_file():
        try:
            previous = json.loads(receipt.read_text())
        except (ValueError, OSError):
            previous = {}
        if previous.get("identity") == identity:
            print("Verified receipt unchanged; tensor byte scan already passed", flush=True)
            return
    preserved = 0
    with sidecar.open("rb") as side:
        for shard in sorted(hs):
            h, bs = hs[shard]
            a, ba = ha[shard]
            targets = set(h) & set(TARGETS)
            if not targets and os.path.samefile(stock / shard, graft / shard):
                preserved += len(h)
                continue
            print(f"Verifying {shard}: {len(targets)} graft targets; remaining tensors equal stock", flush=True)
            with (stock / shard).open("rb") as src, (graft / shard).open("rb") as dst:
                for k in sorted(h, key=lambda k: a[k]["data_offsets"][0]):
                    t = a[k]
                    n = t["data_offsets"][1] - t["data_offsets"][0]
                    if k in TARGETS:
                        compare_bytes(side, base_side + hside[k]["data_offsets"][0],
                                      dst, ba + t["data_offsets"][0], n, k)
                    else:
                        compare_bytes(src, bs + h[k]["data_offsets"][0],
                                      dst, ba + t["data_offsets"][0], n, k)
                        preserved += 1
    # Reject concurrent writes rather than signing a receipt for unverified bytes.
    require(signature(sidecar) == identity["sidecar"], "sidecar changed during verification")
    for role, root in (("stock", stock), ("graft", graft)):
        require(all(signature(root / p) == sig for p, sig in identity[role].items()),
                f"{role} changed during verification")
    result = {"identity": identity, "targets_verified": 104, "stock_tensors_preserved": preserved}
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, sort_keys=True) + "\n")
    os.replace(tmp, receipt)
    print(f"Verified: 104 graft tensors; {preserved} other tensors retain stock bytes and dtypes", flush=True)


if __name__ == "__main__":
    try:
        require(len(sys.argv) in (3, 4), "usage: ablit_validate.py STOCK SIDECAR [GRAFT]")
        validate(*(Path(arg) for arg in sys.argv[1:]))
    except (ValueError, OSError, KeyError, struct.error) as exc:
        print(f"ABLIT validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
