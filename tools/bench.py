#!/usr/bin/env python3
"""One benchmark for the served model, measured the same way every time, with JSON output so runs can be diffed.

Suites (--suite, comma separated, default prefill,decode,multi,needle):
  prefill  prompts of --prefill-sizes tokens (thinking off, max_tokens 1): TTFT, prefill tok/s = uncached prompt
           tokens / TTFT, cached tokens as the server reports them.
  decode   a short prompt, --decode-tokens tokens out: TTFT and decode tok/s = (completion - 1) / (end - first token).
  multi    --streams concurrent decode requests released together: each stream's decode tok/s, and the aggregate
           (all completion tokens / the window from the first token anywhere to the last token anywhere).
  needle   a passphrase hidden in prose of --needle-sizes tokens at each --depths: right or wrong, TTFT, prefill tok/s.

    python3 -I tools/bench.py --out runs/base.json --label base
    python3 -I tools/bench.py --suite decode,multi --streams 1,2,4 --reps 3 --out runs/x.json
    python3 -I tools/bench.py --compare runs/base.json runs/x.json        # per-metric table, no server needed

Every request is streamed with stream_options.include_usage, temperature 0, a fixed seed and thinking OFF (--thinking
turns it on). Prompts are built from --seed and a salt: the salt is random per run (so a repeated run does not read
the prefix cache; it is recorded in the JSON), or fix it with --salt to repeat the very same prompts.
Each reply's `tensorfold` block (when the server sends one) is kept as is under "server", and any numeric field
whose name has "draft" or "accept" is also summed into the run's "draft" figures, with accepted/drafted when both
exist. Optionally --health / --metrics snapshots are taken before and after (raw, not interpreted). Exit code 1 if a
needle is missed or a request fails.

Stdlib only. Endpoint: --url, else API_URL, else PORT from the environment or from scripts/local.sh (the file the
README tells you to put settings in; 8888 when neither sets it); model: --model or MODEL.
New for this recipe; its measurements follow the GLM-5.3-Flash recipe's tools (Mia's AI Lab, Apache-2.0: client.py's
random prose, needle.py, prompt_reuse.py, hermes_benchmark.py's per-request figures) and the lead's streaming
bench.py (TTFT, usage-based prefill and decode rates). No code is copied from them beyond the prose generator below.
"""
import argparse
import json
import os
import platform
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

SCHEMA = 1
DEFAULT_MODEL = "DeepSeek-V4.1-Flash-1017"
SECRET = "violet-harbor-7291"
# Same word list as tools/client.py, so prompt token counts match the other checks (~0.82 tokens per prose "size").
WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue "
         "side kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine garden theory market winter method bridge letter window voice paper field").split()
TOK_PER_SIZE = 0.82
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through a proxy


def _api_key():
    """The server's API key (start.sh with API_KEY=1): API_KEY, else the first key in API_KEY_FILE, else none (as client.py)."""
    key = os.environ.get("API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get("API_KEY_FILE", os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key"))
    try:
        with open(path) as f:
            line = next((x.strip() for x in f if x.strip() and not x.startswith("#")), "")
    except OSError:
        return ""
    return line.split(":", 1)[1].strip() if ":" in line else line


if _api_key():
    OPENER.addheaders = [("Authorization", "Bearer " + _api_key())]


def prose(size: int, seed) -> str:
    rng = random.Random(seed)
    out = []
    while len(out) < size * 0.72:
        sentence = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
        out += sentence[:-1] + [sentence[-1] + "."]
    return " ".join(out)


def parse_list(text: str) -> list:
    return [int(x.replace("k", "000").replace("K", "000")) if x.lower().endswith("k") and x[:-1].isdigit() else int(x)
            for x in text.split(",") if x.strip()]


class Ctx:
    pass


def numeric_leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_leaves(v, f"{prefix}{k}.")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix[:-1], obj


def draft_figures(server: dict) -> dict:
    d = {k: v for k, v in numeric_leaves(server or {}) if any(s in k.lower() for s in ("draft", "accept"))}
    drafted = next((v for k, v in d.items() if "draft" in k.lower() and "accept" not in k.lower() and "rate" not in k.lower()), None)
    accepted = next((v for k, v in d.items() if "accept" in k.lower() and "rate" not in k.lower()), None)
    if drafted and accepted is not None:
        d["accepted_over_drafted"] = accepted / drafted
    return d


def stream_chat(c: Ctx, messages: list, max_tokens: int, start_gate=None) -> dict:
    """One streamed request; returns its figures. Never raises: a failure comes back as {"error": ...}."""
    body = {"model": c.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": c.seed,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": c.thinking}}
    req = urllib.request.Request(c.url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    if start_gate is not None:
        start_gate.wait()
    t0 = time.monotonic()
    first = last = None
    usage = server = None
    text, events, finish = [], 0, None
    try:
        with OPENER.open(req, timeout=c.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                if ev.get("tensorfold"):
                    server = ev["tensorfold"]
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece or delta.get("reasoning_content") or delta.get("reasoning"):
                        now = time.monotonic()
                        first = first or now
                        last = now
                        events += 1
                        text.append(piece)
                    finish = ch.get("finish_reason") or finish
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"}
    except Exception as exc:  # noqa: BLE001 - a benchmark reports, it does not crash
        return {"error": f"{type(exc).__name__}: {exc}"}
    end = time.monotonic()
    if first is None or usage is None:
        return {"error": f"no tokens or no usage chunk (events={events}, usage={usage})"}
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None and isinstance(server, dict):
        cached = next((v for k, v in server.items() if "cached" in k.lower() and isinstance(v, int)), None)
    ttft = first - t0
    uncached = pt - (cached or 0)
    r = {"prompt_tokens": pt, "completion_tokens": ct, "cached_tokens": cached, "ttft_s": ttft,
         "prefill_tok_s": uncached / ttft if ttft > 0 else None,
         "decode_tok_s": (ct - 1) / (last - first) if ct > 1 and last > first else None,
         "total_s": end - t0, "sse_events": events, "finish_reason": finish,
         "t_first": first, "t_last": last, "text": "".join(text)}
    if server:
        r["server"] = server
        r["draft"] = draft_figures(server)
    return r


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def suite_prefill(c: Ctx) -> list:
    out = []
    for size in c.prefill_sizes:
        for rep in range(c.reps):
            hay = prose(int(size / TOK_PER_SIZE), f"{c.seed}-p-{size}-{rep}-{c.salt}")
            r = stream_chat(c, [{"role": "user", "content": hay + "\n\nReply with the single word: ok"}], 1)
            r.update(suite="prefill", target_tokens=size, rep=rep)
            out.append(r)
            show(r)
    return out


DECODE_PROMPT = ("Write a long, detailed essay about the history of the printing press, with many paragraphs. "
                 "Do not stop early. Variation tag: ")


def suite_decode(c: Ctx) -> list:
    out = []
    for rep in range(c.reps):
        r = stream_chat(c, [{"role": "user", "content": DECODE_PROMPT + f"{c.salt}-{rep}"}], c.decode_tokens)
        r.update(suite="decode", rep=rep)
        out.append(r)
        show(r)
    return out


def suite_multi(c: Ctx) -> list:
    out = []
    for n in c.streams:
        for rep in range(c.reps):
            gate = threading.Barrier(n)
            res = [None] * n

            def work(i):
                res[i] = stream_chat(c, [{"role": "user", "content": DECODE_PROMPT + f"{c.salt}-m{n}-{rep}-{i}"}],
                                     c.decode_tokens, gate)
            ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            ok = [r for r in res if "error" not in r]
            row = {"suite": "multi", "streams": n, "rep": rep, "errors": [r["error"] for r in res if "error" in r],
                   "per_stream_decode_tok_s": [r["decode_tok_s"] for r in ok], "ttft_s": [r["ttft_s"] for r in ok]}
            if ok:
                window = max(r["t_last"] for r in ok) - min(r["t_first"] for r in ok)
                total = sum(r["completion_tokens"] for r in ok)
                row.update(completion_tokens=total, window_s=window,
                           aggregate_tok_s=(total - len(ok)) / window if window > 0 else None,
                           mean_stream_decode_tok_s=statistics.mean(x for x in row["per_stream_decode_tok_s"] if x))
                row["draft"] = {}
                for r in ok:
                    for k, v in (r.get("draft") or {}).items():
                        if k != "accepted_over_drafted":
                            row["draft"][k] = row["draft"].get(k, 0) + v
            out.append(row)
            print(f"  multi x{n} rep{rep}: aggregate {fmt(row.get('aggregate_tok_s'))} tok/s, mean per-stream "
                  f"{fmt(row.get('mean_stream_decode_tok_s'))}, errors {len(row['errors'])}", flush=True)
    return out


def suite_needle(c: Ctx) -> list:
    out = []
    for size in c.needle_sizes:
        for depth in c.depths:
            hay = prose(int(size / TOK_PER_SIZE), f"{c.seed}-n-{size}-{c.salt}").split(". ")
            hay.insert(int(len(hay) * depth), f"Remember this: the secret passphrase is {SECRET}")
            prompt = (". ".join(hay) + "\n\nWhat is the secret passphrase mentioned in the text above? "
                      "Reply with the passphrase only.")
            r = stream_chat(c, [{"role": "user", "content": prompt}], 64)
            r.update(suite="needle", target_tokens=size, depth=depth, correct="error" not in r and SECRET in r["text"])
            out.append(r)
            show(r)
    return out


def fmt(x):
    return "n/a" if x is None else f"{x:.1f}"


def show(r: dict) -> None:
    if "error" in r:
        print(f"  {r.get('suite')} ERROR {r['error']}", flush=True)
        return
    extra = f" depth {r['depth']} {'CORRECT' if r['correct'] else 'WRONG'}" if r["suite"] == "needle" else ""
    d = r.get("draft") or {}
    dr = f" accepted/drafted {d['accepted_over_drafted']:.2f}" if "accepted_over_drafted" in d else ""
    print(f"  {r['suite']:7s} prompt {r['prompt_tokens']:7d} (cached {r['cached_tokens']}) ttft {r['ttft_s']:7.2f}s "
          f"prefill {fmt(r['prefill_tok_s']):>8} tok/s | out {r['completion_tokens']:4d} decode "
          f"{fmt(r['decode_tok_s']):>6} tok/s{extra}{dr}", flush=True)


def summarize(runs: list) -> dict:
    s = {}
    for size in sorted({r["target_tokens"] for r in runs if r.get("suite") == "prefill" and "error" not in r}):
        rs = [r for r in runs if r.get("suite") == "prefill" and r.get("target_tokens") == size and "error" not in r]
        s[f"prefill.{size}.ttft_s"] = med([r["ttft_s"] for r in rs])
        s[f"prefill.{size}.tok_s"] = med([r["prefill_tok_s"] for r in rs])
    rs = [r for r in runs if r.get("suite") == "decode" and "error" not in r]
    if rs:
        s["decode.tok_s"] = med([r["decode_tok_s"] for r in rs])
        s["decode.ttft_s"] = med([r["ttft_s"] for r in rs])
        acc = [r["draft"]["accepted_over_drafted"] for r in rs if "accepted_over_drafted" in (r.get("draft") or {})]
        s["decode.accepted_over_drafted"] = med(acc)
    for n in sorted({r["streams"] for r in runs if r.get("suite") == "multi"}):
        rs = [r for r in runs if r.get("suite") == "multi" and r["streams"] == n and not r["errors"]]
        s[f"multi.{n}.aggregate_tok_s"] = med([r.get("aggregate_tok_s") for r in rs])
        s[f"multi.{n}.per_stream_tok_s"] = med([r.get("mean_stream_decode_tok_s") for r in rs])
    for size in sorted({r["target_tokens"] for r in runs if r.get("suite") == "needle"}):
        rs = [r for r in runs if r.get("suite") == "needle" and r["target_tokens"] == size]
        s[f"needle.{size}.correct"] = f"{sum(bool(r['correct']) for r in rs)}/{len(rs)}"
        s[f"needle.{size}.actual_prompt_tokens"] = med([r.get("prompt_tokens") for r in rs])
        s[f"needle.{size}.ttft_s"] = med([r.get("ttft_s") for r in rs if "error" not in r])
    s["errors"] = sum(1 for r in runs if "error" in r or r.get("errors"))
    return s


def get_json_or_text(c: Ctx, path: str):
    try:
        with OPENER.open(c.url + path, timeout=15) as r:
            raw = r.read().decode("utf-8", "replace")
        try:
            return json.loads(raw)
        except ValueError:
            return raw[:20000]
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def compare(a_path: str, b_path: str) -> int:
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    print(f"A = {a.get('label')} ({a_path})\nB = {b.get('label')} ({b_path})")
    print(f"{'metric':36s} {'A':>12s} {'B':>12s} {'B vs A':>9s}")
    sa, sb = a["summary"], b["summary"]
    for k in sorted(set(sa) | set(sb)):
        x, y = sa.get(k), sb.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x:
            delta = f"{(y - x) / x:+.1%}"
        else:
            delta = "" if x == y else "differs"
        f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
        print(f"{k:36s} {f(x):>12s} {f(y):>12s} {delta:>9s}")
    for k in ("model", "salt", "seed"):
        if a.get(k) != b.get(k):
            print(f"note: {k} differs: {a.get(k)!r} vs {b.get(k)!r}")
    return 0


def _port_from_local_sh():
    """The kit's PORT, from scripts/local.sh beside this tool: start.sh serves on it, so bench.py must talk to it."""
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "local.sh")
    try:
        with open(local, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"""\s*(?:export\s+)?PORT\s*=\s*["']?(\d{1,5})["']?\s*(?:#.*)?$""", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def default_base() -> str:
    """--url's default: API_URL, else the port from PORT or scripts/local.sh, else the recipe's 8888.

    local.sh is the file the README says to put settings in, and it is where most pairs change PORT; reading it here
    keeps the tool on the port the server actually listens on instead of a refused 8888.
    """
    if os.environ.get("API_URL"):
        return os.environ["API_URL"]
    return "http://127.0.0.1:" + (os.environ.get("PORT") or _port_from_local_sh() or "8888")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=default_base())
    ap.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    ap.add_argument("--suite", default="prefill,decode,multi,needle")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", help="write the JSON here (always printed as a summary table too)")
    ap.add_argument("--reps", type=int, default=3, help="repetitions per prefill size / decode / multi cell")
    ap.add_argument("--prefill-sizes", default="2048,8192,32768", help="prompt tokens, comma separated (k suffix ok)")
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--streams", default="1,2,4")
    ap.add_argument("--needle-sizes", default="8k,32k,128k")
    ap.add_argument("--depths", default="0.6", help="needle depths as fractions, e.g. 0.1,0.5,0.9")
    ap.add_argument("--thinking", action="store_true", help="thinking ON (default off)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--salt", default=None, help="fixed salt to repeat the same prompts (default random per run)")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--health", action="store_true", help="snapshot GET /health before and after (raw)")
    ap.add_argument("--metrics", action="store_true", help="snapshot GET /metrics before and after (raw)")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="diff two saved runs and exit")
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    c = Ctx()
    c.url, c.model, c.reps, c.seed = a.url.rstrip("/"), a.model, a.reps, a.seed
    c.thinking, c.timeout, c.decode_tokens = a.thinking, a.timeout, a.decode_tokens
    c.salt = a.salt if a.salt is not None else f"{random.SystemRandom().randrange(16**8):08x}"
    c.prefill_sizes, c.needle_sizes = parse_list(a.prefill_sizes), parse_list(a.needle_sizes)
    c.streams, c.depths = parse_list(a.streams), [float(x) for x in a.depths.split(",")]
    suites = [s.strip() for s in a.suite.split(",") if s.strip()]
    table = {"prefill": suite_prefill, "decode": suite_decode, "multi": suite_multi, "needle": suite_needle}
    bad = [s for s in suites if s not in table]
    if bad:
        sys.exit(f"unknown suite {bad}; choose from {sorted(table)}")
    doc = {"schema": SCHEMA, "label": a.label, "model": c.model, "url": c.url, "salt": c.salt, "seed": c.seed,
           "thinking": c.thinking, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
           "python": platform.python_version(), "argv": sys.argv[1:], "snapshots": {}}
    for name, on in (("health", a.health), ("metrics", a.metrics)):
        if on:
            doc["snapshots"][name + "_before"] = get_json_or_text(c, "/" + name)
    runs = []
    for s in suites:
        print(f"[{s}]", flush=True)
        runs += table[s](c)
    for name, on in (("health", a.health), ("metrics", a.metrics)):
        if on:
            doc["snapshots"][name + "_after"] = get_json_or_text(c, "/" + name)
    for r in runs:
        r.pop("t_first", None) if r.get("suite") != "multi" else None
    doc["runs"], doc["summary"] = runs, summarize(runs)
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print("\nsummary (medians):")
    for k, v in doc["summary"].items():
        print(f"  {k:36s} {v if not isinstance(v, float) else round(v, 2)}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"wrote {a.out}")
    wrong = any(r.get("suite") == "needle" and not r.get("correct") for r in runs)
    return 1 if wrong or doc["summary"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
