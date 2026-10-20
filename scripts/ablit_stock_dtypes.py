#!/usr/bin/env python3
"""ablit_stock_dtypes.py STOCK_DIR ABLIT_DIR: give the ABLIT checkpoint's tensors their stock dtypes back.

prepare.sh runs it in the image after the Keys overlay, and on a checkpoint prepared before it existed. The overlay's
apply_mia29_wo_b_graft.py also casts four MTP tensors from F16 to F32 for another engine (Jay Leaton's Zig serve, see
the overlay's TENSORFOLD-ZIG-MIA.md). This recipe's engine reads three of them as F32 whatever the checkpoint holds
(mtp.*.ffn.gate.bias), but it uploads mtp.2.confidence_head.proj.weight as stored and reads it as F16 (draft.zig:
f16ToF32 over D + rank values). From the F32 copy the drafter's confidence head computes garbage: the drafts it keeps
are still accepted, but it drafts 0.2-0.8 tokens a round and decode falls 25-45% below stock.

A tensor whose dtype differs from stock's must equal the stock tensor cast to its dtype (a pure cast, no new values);
it is then written back as the stock tensor, byte for byte. The shard is written to a new file that replaces it, so a
stock shard hardlinked into ABLIT_DIR is never written. Only headers are read, and nothing is written, when every
dtype already matches stock. Prints each tensor it gives back; exits 1 when a tensor is not a pure cast of stock's.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ablit_stock_dtypes.py STOCK_DIR ABLIT_DIR", file=sys.stderr)
        return 2
    stock, ablit = Path(sys.argv[1]), Path(sys.argv[2])
    weight_map = json.loads((ablit / "model.safetensors.index.json").read_text())["weight_map"]
    given = 0
    for name in sorted(set(weight_map.values())):
        s, a = stock / name, ablit / name
        if os.path.samefile(s, a):
            continue  # the stock shard, hardlinked
        hs, ha = header(s), header(a)
        cast = sorted(k for k in ha if k in hs and ha[k]["dtype"] != hs[k]["dtype"])
        if not cast:
            continue
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        with safe_open(str(a), framework="pt") as f:
            meta = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        with safe_open(str(s), framework="pt") as f:
            for k in cast:
                t = f.get_tensor(k)
                if t.shape != tensors[k].shape or not torch.equal(t.to(tensors[k].dtype), tensors[k]):
                    print(f"{name}: {k} is {ha[k]['dtype']} {ha[k]['shape']}, stock {hs[k]['dtype']} {hs[k]['shape']},"
                          " and not a cast of stock's values", file=sys.stderr)
                    return 1
                tensors[k] = t
                print(f"{k}: {ha[k]['dtype']} -> {hs[k]['dtype']} (stock's bytes)", flush=True)
        tmp = a.with_name(a.name + ".tmp")
        save_file(tensors, str(tmp), metadata=meta)
        os.chmod(tmp, 0o644)
        os.replace(tmp, a)
        given += len(cast)
    if not given:
        print("every tensor already has its stock dtype")
    return 0


if __name__ == "__main__":
    sys.exit(main())
