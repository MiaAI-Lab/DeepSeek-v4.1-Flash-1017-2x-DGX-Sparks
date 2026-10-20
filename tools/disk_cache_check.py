#!/usr/bin/env python3
"""Disk cache check (zig/patches/0009): do turns resumed from disk give the same replies as recomputing them?

  python3 -I tools/disk_cache_check.py record --out a.json [--n 6] [--words 48000] [--url URL]
  python3 -I tools/disk_cache_check.py record --out c.json --second-only --previous a.json   # after a restart
  python3 -I tools/disk_cache_check.py compare a.json b.json                                   # exit 1 on any mismatch

record sends N distinct conversations (each its own word seed, no shared prefix) a first turn each, then each one's
second turn (its reply and "Go on."), in order. Run it once with DISK_CACHE=0 and once with DISK_CACHE=1 on a pool too
small for all N (CONTEXT=262144 with 6 x 48000 words, about 52K tokens each): the first conversations lose their kept
state before their second turn, so with the cache off they run again from 0 and with it on they come back from disk.
Per request: prompt tokens, cached tokens, seconds, and the sha256 of the reply's tokens (greedy, fixed seed, thinking
off). --second-only sends only the second turns of an earlier record (after ./stop.sh and ./start.sh: from disk).
compare: same token hashes in both records, cached tokens and seconds side by side. Uses tools/exact.py's helpers.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact  # noqa: E402

SEED = 41017


def ask(a, messages, max_tokens=12):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 7,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    r = exact.post(a.url, body, a.timeout)
    if "error" in r:
        return {"error": r["error"]}
    u = r.get("usage") or {}
    tf = r.get("tensorfold") or {}
    text = r["choices"][0]["message"].get("content") or ""
    return {"prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", tf.get("cached_tokens")),
            "token_sha": tf.get("token_sha") or exact.sha(text), "seconds": round(r["_seconds"], 2), "text": text}


def conversation(a, i):
    return [{"role": "user", "content": f"Conversation {i}. Read this and answer with one word.\n\n" + exact.words(a.words, SEED + i)}]


def record(a):
    out = {"n": a.n, "words": a.words, "cases": {}}
    previous = {}
    if a.second_only:
        with open(a.previous) as f:
            previous = json.load(f)["cases"]
    else:
        for i in range(a.n):
            r1 = ask(a, conversation(a, i))
            out["cases"][f"turn1_{i}"] = r1
            exact.log(f"  turn1_{i}: prompt={r1.get('prompt_tokens')} cached={r1.get('cached_tokens')} {r1.get('seconds')}s {r1.get('error', '')}")
    for i in range(a.n):
        r1 = previous.get(f"turn1_{i}") or out["cases"][f"turn1_{i}"]
        m2 = conversation(a, i) + [{"role": "assistant", "content": r1.get("text", "")}, {"role": "user", "content": "Go on."}]
        r2 = ask(a, m2)
        out["cases"][f"turn2_{i}"] = r2
        exact.log(f"  turn2_{i}: prompt={r2.get('prompt_tokens')} cached={r2.get('cached_tokens')} {r2.get('seconds')}s {r2.get('error', '')}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    return 1 if any("error" in c for c in out["cases"].values()) else 0


def compare(pa, pb):
    with open(pa) as f:
        A = json.load(f)["cases"]
    with open(pb) as f:
        B = json.load(f)["cases"]
    if set(A) != set(B):
        print(f"MISMATCH: the records hold different cases ({sorted(set(A) ^ set(B))})")
        return 1
    bad = 0
    for k in sorted(A):
        x, y = A[k], B[k]
        same = x.get("token_sha") == y.get("token_sha") and x.get("prompt_tokens") == y.get("prompt_tokens")
        bad += not same
        print(f"{'ok      ' if same else 'MISMATCH'} {k}: prompt={x.get('prompt_tokens')} cached A/B={x.get('cached_tokens')}/{y.get('cached_tokens')} "
              f"seconds A/B={x.get('seconds')}/{y.get('seconds')}")
    print(f"{bad} mismatches in {len(A)}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=6)
    r.add_argument("--words", type=int, default=48000)
    r.add_argument("--url", default="http://127.0.0.1:8888")
    r.add_argument("--model", default=exact.DEFAULT_MODEL)
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--second-only", action="store_true")
    r.add_argument("--previous", help="the record with the first turns (with --second-only)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "record" and a.second_only and not a.previous:
        ap.error("--second-only needs --previous")
    return record(a) if a.cmd == "record" else compare(a.a, a.b)


if __name__ == "__main__":
    sys.exit(main())
