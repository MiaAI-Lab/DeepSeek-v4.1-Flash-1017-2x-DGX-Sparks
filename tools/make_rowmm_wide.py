#!/usr/bin/env python3
"""Compile the _rowmm_parts cubins a round wider than 16 rows needs, and merge them into a kit.

Kit abe07c7 records _rowmm_parts for rows 1 through 16 only. That kernel's RB constexpr is the
row count, so ROUND_ROWS above 16 dies while the CUDA graph is captured (MissingTritonVariant,
then CudaFailed). Recapturing the graph does not compile the cubin. The engine allows 48 rows.
At 12 live streams a 48-row round is 4 rows each (3 drafts). At 16 live streams it is 3 rows each.

compile runs in the serving image, on a Spark whose model is stopped (the compile wants the GPU).
The image does not install TensorFold's Python package, so mount kernels.py from branch
deepseek-v41-tp2 (src/tensorfold/families/deepseek_v41/cuda/kernels.py):

    docker run --rm --gpus all --network none \
      -v "$PWD/tools/make_rowmm_wide.py:/work/make_rowmm_wide.py:ro" \
      -v /path/to/kernels.py:/work/kernels.py:ro \
      -v /tmp/rowmm-out:/out \
      --entrypoint python3 IMAGE -u /work/make_rowmm_wide.py compile \
        --kernels /work/kernels.py --out /out

merge is stdlib only. Point it at the kit directory prepare.sh installed, then copy the updated
aot/aot.json, MANIFEST, and the new cubins to the other Spark before the next start:

    python3 tools/make_rowmm_wide.py merge --out /tmp/rowmm-out --kit /path/to/kit

A rows=16 launch is compiled first and checked against the kit's BN=8 variant (shared 32, 2 warps).
It is not installed. Rows 17 through 48 are compiled for both gate widths, N=128 and N=384.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

K = 5120
BK = 256
BN = 8
CPG = 1
NW = 2
NS = (128, 384)


def load_kernels(path: Path):
    spec = importlib.util.spec_from_file_location("dsv41_kernels", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ptx_entry(kernel) -> str:
    ptx = kernel.asm.get("ptx")
    if ptx is None:
        raise RuntimeError(f"no ptx in asm keys {list(kernel.asm)}")
    if isinstance(ptx, bytes):
        ptx = ptx.decode()
    names = re.findall(r"\.visible\s+\.entry\s+([A-Za-z_][A-Za-z0-9_]*)", ptx)
    if len(names) != 1:
        raise RuntimeError(f"ptx entries {names}")
    if names[0] != "_rowmm_parts":
        raise RuntimeError(f"export {names[0]!r}")
    return names[0]


def specialization(fn, args, kwargs):
    import triton

    device = triton.runtime.driver.active.get_current_device()
    binder = fn.device_caches[device][4]
    _bound, spec, _options = binder(*args, **kwargs)
    out = []
    for param, item in zip(fn.params, spec):
        kind, desc = item[0], item[1]
        div16 = kind != "constexpr" and isinstance(desc, str) and "D" in desc
        out.append((param.name, kind, div16))
    return out


def compile_one(fn, x, w, s, rows: int, n: int, out: Path):
    import triton

    xs = x[:rows].stride(0)
    if xs % 16 != 0:
        raise RuntimeError(f"xs {xs} is not divisible by 16")
    grid = (triton.cdiv(n, BN), (K // BK) // CPG)
    args = (x[:rows], xs, w[:n], s[:rows], rows)
    kwargs = dict(K=K, N=n, BN=BN, BK=BK, CPG=CPG, RB=rows, num_warps=NW, PDL=True, launch_pdl=True)
    t0 = time.time()
    kernel = fn[grid](*args, **kwargs)
    if kernel is None:
        raise RuntimeError("launch returned None")
    md = kernel.metadata
    for need in ("name", "num_warps", "shared"):
        if need not in md._fields:
            raise RuntimeError(f"metadata missing {need}: {sorted(md._fields)}")
    if md.name != "_rowmm_parts":
        raise RuntimeError(f"metadata.name {md.name!r}")
    ptx_entry(kernel)
    spec = specialization(fn, args, kwargs)
    runtime = [(name, kind, div16) for name, kind, div16 in spec if kind != "constexpr"]
    got = [name for name, _kind, _div16 in runtime]
    if got != ["X", "xs", "W", "S", "rows"]:
        raise RuntimeError(f"runtime params {got}")
    div = {name: div16 for name, _kind, div16 in runtime}
    expect = {"X": True, "xs": True, "W": True, "S": True, "rows": rows % 16 == 0}
    if div != expect:
        raise RuntimeError(f"div16 {div} != {expect}")
    cubin = kernel.kernel
    if not isinstance(cubin, (bytes, bytearray)) or len(cubin) < 64:
        raise RuntimeError("cubin is empty")
    digest = hashlib.sha256(cubin).hexdigest()
    shared = int(md.shared or 0)
    warps = int(md.num_warps)
    if not bool(getattr(md, "launch_pdl", False)):
        raise RuntimeError("launch_pdl is false; the kernel waits on PDL")
    entry = {
        "fn": "_rowmm_parts",
        "hash": digest,
        "name": "_rowmm_parts",
        "num_warps": warps,
        "num_ctas": int(getattr(md, "num_ctas", 1) or 1),
        "shared": shared,
        "global_scratch": int(getattr(md, "global_scratch_size", 0) or 0),
        "global_align": int(getattr(md, "global_scratch_align", 1) or 1),
        "profile_scratch": int(getattr(md, "profile_scratch_size", 0) or 0),
        "pdl": True,
        "params": [
            {"name": "X", "type": "*bf16", "div16": True, "nospec": False},
            {"name": "xs", "type": "i32", "div16": True, "nospec": False},
            {"name": "W", "type": "*fp16", "div16": True, "nospec": False},
            {"name": "S", "type": "*fp32", "div16": True, "nospec": False},
            {"name": "rows", "type": "i32", "div16": rows % 16 == 0, "nospec": False},
        ],
        "consts": {
            "K": {"int": K},
            "N": {"int": n},
            "BN": {"int": BN},
            "BK": {"int": BK},
            "CPG": {"int": CPG},
            "RB": {"int": rows},
            "PDL": {"int": 1},
        },
    }
    if entry["global_scratch"] or entry["profile_scratch"]:
        raise RuntimeError(f"rows={rows} n={n} needs scratch memory")
    (out / "cubins" / f"{digest}.cubin").write_bytes(cubin)
    print(
        f"OK rows={rows} n={n} shared={shared} warps={warps} bytes={len(cubin)} "
        f"secs={time.time() - t0:.1f} hash={digest[:12]}",
        flush=True,
    )
    return entry


def compile_cmd(a) -> int:
    import torch
    import triton

    if not torch.cuda.is_available():
        print("NO_CUDA", flush=True)
        return 3
    out = Path(a.out)
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    print("device", torch.cuda.get_device_name(0), "triton", triton.__version__, flush=True)
    fn = load_kernels(Path(a.kernels))._rowmm_parts
    x = torch.empty((48, K), dtype=torch.bfloat16, device="cuda")
    w = torch.empty((max(NS), K), dtype=torch.float16, device="cuda")
    s = torch.empty((48, K // BK, max(NS)), dtype=torch.float32, device="cuda")
    print("CANARY rows=16 n=384", flush=True)
    canary = compile_one(fn, x, w, s, 16, 384, out)
    (out / "cubins" / f"{canary['hash']}.cubin").unlink()
    print(
        f"CANARY_META shared={canary['shared']} warps={canary['num_warps']} ctas={canary['num_ctas']}",
        flush=True,
    )
    if canary["shared"] != 32 or canary["num_warps"] != 2 or canary["num_ctas"] != 1:
        print("CANARY_MISMATCH expected shared 32 warps 2 ctas 1", flush=True)
        return 4
    entries = []
    failed = []
    for n in NS:
        for rows in range(17, 49):
            try:
                entries.append(compile_one(fn, x, w, s, rows, n, out))
            except Exception:
                failed.append((rows, n))
                print(f"FAIL rows={rows} n={n}", flush=True)
                traceback.print_exc()
    (out / "entries.json").write_text(json.dumps(entries, indent=1) + "\n")
    want = 64
    print(f"COMPILE_DONE {len(entries)} FAIL {len(failed)} {failed}", flush=True)
    return 0 if len(entries) == want and not failed else 2


def variant_key(e: dict) -> tuple:
    consts = tuple(sorted((n, json.dumps(v, sort_keys=True)) for n, v in e["consts"].items()))
    return (e["fn"], consts)


def check_entry(e: dict) -> None:
    if e["fn"] != "_rowmm_parts" or e["name"] != "_rowmm_parts":
        raise SystemExit(f"bad name {e['fn']} {e['name']}")
    if e["shared"] != 32 or e["num_warps"] != 2 or e["pdl"] is not True:
        raise SystemExit(f"unexpected launch facts {e['hash'][:12]}")
    if e["global_scratch"] or e["profile_scratch"]:
        raise SystemExit(f"{e['hash'][:12]} needs scratch")
    rb = e["consts"]["RB"]["int"]
    n = e["consts"]["N"]["int"]
    if n not in NS or e["consts"]["BN"]["int"] != 8 or e["consts"]["K"]["int"] != K:
        raise SystemExit(f"unexpected shape N={n} RB={rb}")
    rows = next(p for p in e["params"] if p["name"] == "rows")
    if rows["div16"] != (rb % 16 == 0) or rows["nospec"] is not False:
        raise SystemExit(f"rows div16 wrong for RB={rb}")


def merge_cmd(a) -> int:
    out, kit = Path(a.out), Path(a.kit)
    entries = json.loads((out / "entries.json").read_text())
    if len(entries) != 64:
        raise SystemExit(f"{len(entries)} entries, want 64 (rows 17-48, N=128 and N=384)")
    seen = set()
    for e in entries:
        check_entry(e)
        blob = (out / "cubins" / f"{e['hash']}.cubin").read_bytes()
        if hashlib.sha256(blob).hexdigest() != e["hash"]:
            raise SystemExit(f"cubin hash mismatch {e['hash'][:12]}")
        if not (17 <= e["consts"]["RB"]["int"] <= 48):
            raise SystemExit(f"RB {e['consts']['RB']['int']} is outside 17-48")
        if variant_key(e) in seen:
            raise SystemExit(f"duplicate {variant_key(e)}")
        seen.add(variant_key(e))
    aot_path = kit / "aot" / "aot.json"
    man_path = kit / "MANIFEST"
    text = aot_path.read_text()
    meta = json.loads(text)
    if json.dumps(meta, indent=1) + "\n" != text:
        raise SystemExit(f"{aot_path} is not aot_pack.py's formatting: not rewriting it")
    have_hash = {e["hash"] for e in meta["kernels"]}
    have_key = {variant_key(e) for e in meta["kernels"]}
    clash = [e for e in entries if variant_key(e) in have_key or e["hash"] in have_hash]
    if clash:
        raise SystemExit(f"{len(clash)} entries already in the kit")
    for src in (aot_path, man_path):
        dest = src.with_name(src.name + ".bak-rowmm-wide")
        if not dest.exists():
            shutil.copy2(src, dest)
    for e in entries:
        dest = kit / "aot" / "cubins" / f"{e['hash']}.cubin"
        if dest.exists():
            raise SystemExit(f"{dest.name} already exists")
        shutil.copy2(out / "cubins" / dest.name, dest)
    meta["kernels"] = sorted(meta["kernels"] + entries, key=lambda x: (x["fn"], x["hash"]))
    aot_path.unlink()
    aot_path.write_text(json.dumps(meta, indent=1) + "\n")
    lines = [ln.split("  ", 1) for ln in man_path.read_text().splitlines()]
    if any(len(x) != 2 for x in lines):
        raise SystemExit("MANIFEST separator is not two spaces")
    sums = {p: s for s, p in lines}
    sums["aot/aot.json"] = hashlib.sha256(aot_path.read_bytes()).hexdigest()
    for e in entries:
        rel = f"aot/cubins/{e['hash']}.cubin"
        sums[rel] = hashlib.sha256((kit / rel).read_bytes()).hexdigest()
    paths = [p for _, p in lines]
    order = sorted(sums) if paths == sorted(paths) else paths + sorted(set(sums) - set(paths))
    man_path.unlink()
    man_path.write_text("".join(f"{sums[p]}  {p}\n" for p in order))
    print(f"merged {len(entries)} variants into {kit}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="compile rows 17-48 inside the serving image")
    c.add_argument("--kernels", required=True, help="kernels.py from TensorFold branch deepseek-v41-tp2")
    c.add_argument("--out", required=True)
    c.set_defaults(func=compile_cmd)
    m = sub.add_parser("merge", help="add a finished --out directory to a kit on disk")
    m.add_argument("--out", required=True)
    m.add_argument("--kit", required=True, help="the kit root (aot/aot.json and MANIFEST)")
    m.set_defaults(func=merge_cmd)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
