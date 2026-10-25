#!/usr/bin/env python3
"""Check the mixed-pack patches: their content and that they apply to the pinned TensorFold commits.

The serving patch (zig/patches/0024-mixed-pack.patch) is against TF_REF with the patches before it applied (it
comes after the port's series, 0011-0023); the rank-cache planner patch (zig/patches-inputs/0001-mixed-pack-rank-
cache.patch) is against INPUTS_REF. Both pins are read from scripts/config.sh, so a re-pin moves the check with it.
0025/0002 widen the EAQ v1.1 indexer's BF16 `weights_proj` to the f32 entry the loaders plan for it, in the engine's
checkpoint reader and in the rank-cache writer; 0026 pads a mixed pack's plain linear to a fixed row count
(`TF_DS_PLAIN_ROWS`), so cuBLAS's shape does not decide a token's bits.

The application check needs a TensorFold clone with the pinned commits: set TF_SRC, or keep one at /tmp/tf-fork.
Without one the content checks still run and the application check is skipped (the recipe's Dockerfile applies the
patches when it builds the image).
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / "zig" / "patches" / "0024-mixed-pack.patch"
SERVING_BF16 = ROOT / "zig" / "patches" / "0025-widen-bf16.patch"
INPUTS = ROOT / "zig" / "patches-inputs" / "0001-mixed-pack-rank-cache.patch"
INPUTS_BF16 = ROOT / "zig" / "patches-inputs" / "0002-widen-bf16.patch"
SERVING_ROWS = ROOT / "zig" / "patches" / "0026-plain-rows.patch"


def pinned(name):
    text = (ROOT / "scripts" / "config.sh").read_text()
    m = re.search(rf'^{name}="\$\{{{name}:-([0-9a-f]{{40}})\}}"', text, re.M)
    if not m:
        sys.exit(f"config.sh: no {name} pin")
    return m.group(1)


def check_content(path, needles):
    text = path.read_text()
    if not text.strip():
        sys.exit(f"{path.relative_to(ROOT)}: empty")
    for needle in needles:
        if needle not in text:
            sys.exit(f"{path.relative_to(ROOT)}: missing {needle!r}")
    print(f"ok  {path.relative_to(ROOT)}: {len(text.splitlines())} lines, {len(needles)} markers")


def source():
    env = os.environ.get("TF_SRC")
    if env:
        return Path(env)
    fallback = Path("/tmp/tf-fork")
    return fallback if fallback.exists() else None


def run(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{' '.join(args)}: {p.stderr.strip() or p.stdout.strip()}")


def apply_check(src, sha, patch):
    with tempfile.TemporaryDirectory() as td:
        wt = Path(td) / "wt"
        run(["git", "-C", str(src), "worktree", "add", "-q", "--detach", str(wt), sha])
        try:
            run(["git", "-C", str(wt), "apply", "--check", str(patch)])
        finally:
            subprocess.run(["git", "-C", str(src), "worktree", "remove", "--force", str(wt)],
                           capture_output=True, text=True)
    print(f"ok  {patch.relative_to(ROOT)} applies to {sha[:12]}")


def apply_chain(src, sha, patches, what="zig/patches"):
    """The Dockerfile's order: git apply each patch in filename order, so a later one may build on an earlier one."""
    with tempfile.TemporaryDirectory() as td:
        wt = Path(td) / "wt"
        run(["git", "-C", str(src), "worktree", "add", "-q", "--detach", str(wt), sha])
        try:
            for patch in patches:
                run(["git", "-C", str(wt), "apply", str(patch)])
        finally:
            subprocess.run(["git", "-C", str(src), "worktree", "remove", "--force", str(wt)],
                           capture_output=True, text=True)
    print(f"ok  all {len(patches)} {what} apply in filename order to {sha[:12]}")


def main():
    tf_ref = pinned("TF_REF")
    inputs_ref = pinned("INPUTS_REF")

    check_content(SERVING, [
        "mixed pack: {d} float linears",
        "plain: u64 = 0",
        "plain_tmp",
        "e.blas.linearBf16",
        "groupedRotated",
    ])
    check_content(SERVING_BF16, [
        "widenBFloat",
        "from == .bf16 and k.dtype.? == .f32",
        "a bf16 tensor read as f32 widens exactly",
    ])
    check_content(INPUTS, [
        "wantsShards",
        "plain = true",
        "plainKey",
        "t.shape[1]",
    ])
    check_content(INPUTS_BF16, [
        "widenBFloat",
        "from == .bf16 and k.dtype.? == .f32",
        "a bf16 tensor read as f32 widens exactly",
    ])
    check_content(SERVING_ROWS, [
        "plainRows",
        "TF_DS_PLAIN_ROWS",
        "e.plain_rows <= m",
        "e.round_rows",
    ])

    src = source()
    if src is None:
        print("skip the application check: no TensorFold clone (set TF_SRC)")
        return 0
    for sha in (tf_ref, inputs_ref):
        if subprocess.run(["git", "-C", str(src), "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True).returncode != 0:
            print(f"skip the application check: {src} has no {sha[:12]} (set TF_SRC)")
            return 0
    # the serving patches build on the port's series: checked in the Dockerfile's order, with every patch before them
    apply_check(src, inputs_ref, INPUTS)
    apply_check(src, inputs_ref, INPUTS_BF16)
    apply_chain(src, inputs_ref, sorted((ROOT / "zig" / "patches-inputs").glob("*.patch")), "zig/patches-inputs")
    apply_chain(src, tf_ref, sorted((ROOT / "zig" / "patches").glob("*.patch")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
