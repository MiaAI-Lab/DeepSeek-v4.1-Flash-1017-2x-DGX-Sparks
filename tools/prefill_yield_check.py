#!/usr/bin/env python3
"""A short request that arrives while a long prompt is being read: how long does it wait, and do the long reply's bits change?

  python3 tools/prefill_yield_check.py [--url http://127.0.0.1:8888] [--long-words 64000] [--delay 8] [--salt yieldtest]

Sends one long COLD prompt (greedy, thinking off, 24 tokens out), and `--delay` seconds later a short request (16 tokens
out). Prints each request's seconds and the long reply's token sha. With PREFILL_YIELD=0 the short request is read after
the whole long prompt; with PREFILL_YIELD=1 it is read after one chunk, so its seconds drop from about the long prompt's
prefill time to a few seconds. The long prompt is byte-identical for the same --salt, so its token sha can be compared
between a run before and a run after a restart (a restart empties the prompt cache; a second run on the same server is
not cold and the script says so). The long reply must not depend on the setting: the same sha is the check.
Stdlib only. API key as tools/exact.py (API_KEY, else API_KEY_FILE).
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact  # noqa: E402  (the same prose, post and key handling as the bit-exactness harness)

SUFFIX = "Summarize the text above in one sentence."


def ask(url, model, content, max_tokens, seed, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "seed": seed, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.monotonic()
    r = exact.post(url, body, timeout)
    if "error" in r:
        return {"error": r["error"], "seconds": round(time.monotonic() - t0, 2)}
    ch = r["choices"][0]
    u = r.get("usage") or {}
    tf = r.get("tensorfold") or {}
    text = ch["message"].get("content") or ""
    return {"seconds": round(r["_seconds"], 2), "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "finish": ch.get("finish_reason"), "token_sha": (tf.get("token_sha") or exact.sha(text))[:16], "text": text[:60]}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8888")
    p.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    p.add_argument("--long-words", type=int, default=64000, help="words of the long prompt (about 1.1 tokens a word)")
    p.add_argument("--delay", type=float, default=8.0, help="seconds after the long request starts that the short one is sent")
    p.add_argument("--salt", default="yieldtest")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    long_text = f"[ref {a.salt}]\n" + exact.words(a.long_words, a.seed) + "\n\n" + SUFFIX
    short_text = f"[ref short-{int(time.time())}] What is the capital of France? Answer in one word."
    out = {}

    def run_long():
        out["long"] = ask(a.url, a.model, long_text, 24, a.seed, a.timeout)

    t0 = time.monotonic()
    th = threading.Thread(target=run_long)
    th.start()
    time.sleep(a.delay)
    sent_at = time.monotonic() - t0
    out["short"] = ask(a.url, a.model, short_text, 16, a.seed, a.timeout)
    out["short"]["sent_after_s"] = round(sent_at, 1)
    th.join()
    out["long"]["total_wall_s"] = round(time.monotonic() - t0, 1)
    print(json.dumps(out, indent=2))
    if "error" in out["long"] or "error" in out["short"]:
        return 1
    if out["long"].get("cached_tokens"):
        print(f"NOTE: the long prompt was not cold (cached_tokens={out['long']['cached_tokens']}): its timing says nothing; "
              "restart the server or use another --salt", file=sys.stderr)
    if out["short"]["seconds"] > out["long"]["seconds"]:
        print("NOTE: the short request finished after the long one: it did not run beside it", file=sys.stderr)
    print(f"short request: {out['short']['seconds']} s (sent {out['short']['sent_after_s']} s in); "
          f"long: {out['long']['seconds']} s, {out['long']['prompt_tokens']} tokens, sha {out['long']['token_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
