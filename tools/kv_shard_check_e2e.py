#!/usr/bin/env python3
"""End-to-end checks of KV sharding (wojo's GLM 0118: owner-only compressed KV, owner decode, borrowed twins,
cross-owner carry, live moves; TF_DS_KV_SHARD=1 with TF_DS_POOL) against a running server. Every reply is compared by
strong token hash with a baseline record made on the unchanged (unsharded) image; the engine's KV-ownership counters
prove which path ran.

    python3 -I tools/kv_shard_check_e2e.py record --out kv-base.json                # BASELINE server (sharding off)
    python3 -I tools/kv_shard_check_e2e.py run --baseline kv-base.json --receipt r-kv.json \\
        [--expect-window 1048576] [--expect-pool 2101248] [--needle-sizes 524288,max]   # CANDIDATE server
        [--startup-log FILE]      # rank 0's `dsv41: pool ... a conversation may use up to ...` line (docker logs)
        [--kill-cmd CMD]          # kill-rank1: kills rank 1's engine (tools/gpu_validate.sh kill-ranks --rank 1)

Counters (shard-lanes, merged on port/integration: server/status_routes.zig putKvParts, metrics.zig kvPartFamilies):
/health `kv_shard{<lanes.kv_parts.Stats>, per_part: [{part, free, live_rows, kept_rows, twin_rows, hold_rows, live}]}`
or /metrics `tensorfold:kv_shard_<name>`, `tensorfold:kv_shard_part_<row>{part="N"}`; read as kv_shard.<name> and
kv_shard.part<N>.<row> (checklib KV_FIELDS / KV_PART_FIELDS). Per-part gauges are sampled (/health, else /metrics) every
--sample-ms while a check's requests run (the most live streams each part held, and samples with both parts live).

Checks (PASS / FAIL / SKIP = not exercised or not reported, with why; exit 1 on any FAIL, or SKIP with --strict).
The numbers are shard-lanes.md's "GPU validation needed" items:
  limits          the single-conversation limit and the total pool, judged apart:
                    limits: window      /tokenize max_model_len (the engine's only endpoint for it; /v1/models
                                        max_model_len, /health context_window and the start-up line agree when given);
                                        with --expect-window it is at least that
                    limits: pool        /health kv_shard parts x part_rows, else the start-up line (--startup-log, or
                                        --reported-pool); every source that reports it agrees; window <= pool;
                                        == --expect-pool. SKIP when nothing reports it (FAIL with --strict)
                    limits: over-window a prompt of window + --over-window tokens (sized with /tokenize) is refused
                                        with HTTP 400 context_length; the admitted half is the `max` needle
  owners      (2) two long decodes at once (one per owner): replies == baseline; both parts held a live stream
                  (per_part live sampled) and decode rounds ran sharded (fetch_rounds + owner_rounds grew)
  mixed       (2) eight requests at once, varied sizes: replies == baseline; samples with both parts live (rounds
                  mixing both owners) and sharded rounds grew
  burst       (3) --burst requests at once on one long prompt (same-part copies): replies == baseline; no transfer
                  failed (transfers_failed unchanged); cross-part copies are reported, not judged
  carry       (4) a kept conversation, fillers, and a hit after each (--carry-rounds): every reply == baseline and a
                  carry happened (kv_shard.carries +1); moved_tokens / moved_bytes reported (720 B a token expected);
                  SKIP when no carry was provoked (try a smaller TF_DS_POOL)
  twins       (6) a long fresh prompt borrows the other owner's free rows: reply == baseline, twins_made +1, and its
                  pass fetched nothing (prompt_fetches unchanged; when twins_revoked grew, a revoke then a fetch is
                  reported, the reply still judged); every part's twin_rows back to 0 after it
  move        (5) two decodes (one per part), then a fresh prompt (--move-tokens) that fits in no part beside one of
                  them: every reply == baseline (the moved stream decodes on exactly) and a live move happened
                  (kv_shard.moves +1); SKIP when none did
  cancel-transfer the carry provoked again with another conversation, its client leaving while it runs (carries +1
                  or a part's hold_rows seen): the server stays healthy, every part's hold_rows and live back to 0, the
                  request again == baseline. SKIP when the request had already ended (or was not cut) at the cancel
  kill-rank1  (7) the carry provoked once more and --kill-cmd run while it runs: the request in flight must END
                  (an error, or the server gone) within --kill-deadline; a hang FAILs. The caller restarts the pair
                  afterwards. SKIP without --kill-cmd, without a provoked carry, or when the request had already ended
                  before the kill (then --kill-cmd is not run). Run it last (the pair is gone)
  idle            once nothing runs: two parts; every part's live, live_rows, twin_rows and hold_rows 0;
                  free + kept_rows <= part_rows; transfers_failed did not grow over the run
  needle          three hidden values at 10/50/90% of a prompt per --needle-sizes (tokens; `max` = the largest the
                  reported window admits), sized with /tokenize, all three found (no baseline needed)
Every numeric kv_shard field is kept in the receipt (before and after each check). Stdlib only.
"""
import argparse
import concurrent.futures
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checklib as C  # noqa: E402

Q = ["Summarize the text above in one sentence.", "Which word comes last in the text above?",
     "Give a title for the text above.", "List three nouns from the text above.",
     "Name two words that repeat in the text above.", "Reply with a number between one and nine.",
     "Say one short sentence about the text above.", "What is the first topic in the text above?",
     "Write one question a reader could ask about the text above.", "Describe the text above in five words.",
     "Pick one word from the text above and define it.", "How long is the text above, roughly?"]
Q_AFTER = "After the interruption: summarize the text above in one sentence."
MIXED = [1200, 2600, 4100, 5300, 6800, 8200, 9900, 12000]
SECRETS = {"ALPHA": "384912", "BETA": "725063", "GAMMA": "491857"}
FILLER = "Archive note: The old stone bridge crosses a quiet river beside the northern forest.\n"
POOL_LINE = re.compile(r"dsv41: pool (\d+) positions, (\w+) (\d+) a part; a conversation may use up to (\d+) tokens")


def startup_line(path):
    """(pool, mode, part, window) from the last start-up line in a saved log, or None."""
    try:
        with open(path, errors="replace") as f:
            found = POOL_LINE.findall(f.read())
    except OSError:
        return None
    if not found:
        return None
    pool, mode, part, window = found[-1]
    return int(pool), mode, int(part), int(window)


def doc(salt, name, tokens, seed):
    return f"[ref {salt}-{name}]\n" + C.words(C.words_for(tokens), seed + sum(map(ord, name)) * 104729)


def m(text, q):
    return [{"role": "user", "content": text + ("\n\n" + q if q else "")}]


class Plan:
    """Every request this tool makes, from salt + seed + sizes (the record and the run build the same prompts)."""

    def __init__(self, meta):
        self.meta = meta
        s, seed = meta["salt"], meta["seed"]
        self.owners = {f"O{i}": m(doc(s, f"O{i}", meta["owner_tokens"], seed),
                                  "Write a long, detailed story inspired by the text above. Do not stop early.")
                       for i in (1, 2)}
        self.mixed = {f"M{i + 1}": m(doc(s, f"M{i + 1}", t, seed), Q[i % len(Q)]) for i, t in enumerate(MIXED)}
        ca = doc(s, "CA", meta["carry_tokens"], seed)
        self.carry_seed = m(ca, Q[0])
        self.carry_hits = [m(ca, Q[(i + 1) % len(Q)]) for i in range(meta["carry_rounds"])]
        cb = doc(s, "CB", meta["carry_tokens"], seed)
        self.cancel_seed, self.cancel_after = m(cb, Q[0]), m(cb, Q_AFTER)
        ck = doc(s, "CK", meta["carry_tokens"], seed)
        self.kill_seed, self.kill_after = m(ck, Q[0]), m(ck, Q_AFTER)
        self.twin = m(doc(s, "TW", meta["twin_tokens"], seed), Q[2])
        bu = doc(s, "BU", meta.get("burst_tokens", 0), seed)
        self.burst = {f"BU{i + 1}": m(bu, Q[i % len(Q)]) for i in range(meta.get("burst", 0))}
        self.move_side = {f"MV_{x}": m(doc(s, f"MV_{x}", meta.get("move_side_tokens", 0), seed),
                                       "Write a long, detailed story inspired by the text above. Do not stop early.")
                          for x in ("A", "B")} if meta.get("move_tokens") else {}
        self.move_big = m(doc(s, "MV_C", meta["move_tokens"], seed), Q[3]) if meta.get("move_tokens") else None

    def filler(self, salt):
        return m(doc(self.meta["salt"] + "-f", salt, self.meta["filler_tokens"], self.meta["seed"]), Q[0])


def record(a) -> int:
    srv = C.Server(a.url, a.model, a.timeout, a.seed)
    if not srv.alive():
        C.log(f"record: {srv.url} is not healthy: {srv.health()}")
        return 2
    meta = {"tool": "kv_shard_check_e2e", "schema": 2, "salt": a.salt or secrets.token_hex(6), "seed": a.seed,
            "model": a.model, "max_tokens": a.max_tokens, "decode_tokens": a.decode_tokens,
            "owner_tokens": a.owner_tokens, "carry_tokens": a.carry_tokens, "carry_rounds": a.carry_rounds,
            "twin_tokens": a.twin_tokens, "filler_tokens": a.filler_tokens, "burst": a.burst,
            "burst_tokens": a.burst_tokens, "move_side_tokens": a.move_side_tokens, "move_tokens": a.move_tokens,
            "label": a.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "health": srv.health()}
    p = Plan(meta)
    rows = {}

    def put(name, msgs, mt):
        r = srv.chat(msgs, mt)
        rows[name] = r
        C.log(f"  {name}: " + (r.get("error") or f"prompt={r['prompt_tokens']} out={r['completion_tokens']} "
                                                  f"sha={r['token_sha'][:12]}({r['token_sha_source']})"))
    for n, msgs in p.owners.items():
        put(n, msgs, a.decode_tokens)
    for n, msgs in p.mixed.items():
        put(n, msgs, a.max_tokens)
    for n, msgs in p.burst.items():
        put(n, msgs, a.max_tokens)
    put("CA", p.carry_seed, a.max_tokens)
    for i, msgs in enumerate(p.carry_hits):
        put(f"CA{i + 1}", msgs, a.max_tokens)
    put("CB", p.cancel_seed, a.max_tokens)
    put("CB_after", p.cancel_after, a.max_tokens)
    put("CK", p.kill_seed, a.max_tokens)
    put("TW", p.twin, a.max_tokens)
    for n, msgs in p.move_side.items():
        put(n, msgs, a.decode_tokens)
    if p.move_big:
        put("MV_C", p.move_big, a.max_tokens)
    errors = [n for n, r in rows.items() if "error" in r]
    meta["weak"] = [n for n, r in rows.items() if "error" not in r and r.get("token_sha_source") != "server"]
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1, sort_keys=True)
    C.log(f"wrote {a.out}: {len(rows)} rows, errors {errors}" + (f", WEAK {meta['weak']}" if meta["weak"] else ""))
    return 1 if errors else 0


class Sampler:
    """Polls /health while a check's requests run: the most live streams each part held at once, how many samples
    had both parts live, and the largest hold_rows / twin_rows seen. `parts` stays 0 when no kv_shard is reported."""

    def __init__(self, srv, every):
        self.srv, self.every = srv, every
        self.samples = self.both = self.parts = 0
        self.max_live, self.max_hold, self.max_twin = [], 0, 0
        self.stop = threading.Event()
        self.th = threading.Thread(target=self.loop, daemon=True)

    def take(self):
        ps = C.kv_parts(self.srv.counters())     # /health, or /metrics when an API key hides it
        if not ps:
            return
        self.samples += 1
        self.parts = max(self.parts, len(ps))
        lives = [int(p.get("live", 0)) for p in ps]
        self.max_live = [max(a, b) for a, b in zip(self.max_live + [0] * len(lives), lives)]
        if len(lives) >= 2 and lives[0] > 0 and lives[1] > 0:
            self.both += 1
        self.max_hold = max(self.max_hold, *(int(p.get("hold_rows", 0)) for p in ps))
        self.max_twin = max(self.max_twin, *(int(p.get("twin_rows", 0)) for p in ps))

    def loop(self):
        while not self.stop.is_set():
            try:
                self.take()
            except Exception:  # noqa: BLE001 - a sample is best effort
                pass
            self.stop.wait(self.every)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.th.join(10)

    def summary(self):
        return {"samples": self.samples, "parts": self.parts, "max_live": self.max_live, "both_live": self.both,
                "max_hold_rows": self.max_hold, "max_twin_rows": self.max_twin}


class Run:
    def __init__(self, a):
        self.a = a
        self.base = json.load(open(a.baseline))
        self.meta = self.base["meta"]
        self.p = Plan(self.meta)
        self.srv = C.Server(a.url, a.model or self.meta["model"], a.timeout, self.meta["seed"])
        self.rc = C.Receipt("kv_shard_check_e2e", {"url": self.srv.url, "baseline": a.baseline, "label": a.label,
                                                   "salt": self.meta["salt"]})
        self.mt, self.dt = self.meta["max_tokens"], self.meta["decode_tokens"]
        self.rounds_to_carry = None
        self.c_start = None

    def same(self, name, row):
        return C.same_reply(self.base["rows"].get(name), row, self.a.allow_weak)

    def c(self):
        return self.srv.counters()

    def kv(self, c):
        return {k: v for k, v in c.items() if k.startswith("kv_shard.")}

    def sampler(self):
        return Sampler(self.srv, self.a.sample_ms / 1000)

    def sharded(self, c):
        return "kv_shard.parts" in c

    def rounds(self, c0, c1):
        """Sharded decode rounds between two readings (fetch + owner), None when not reported."""
        d = C.deltas(c0, c1, ["kv_shard.fetch_rounds", "kv_shard.owner_rounds"])
        return None if any(v is None for v in d.values()) else sum(d.values())

    def missing(self, *names):
        return [n for n in names if n not in self.base["rows"]]

    # ------------------------------------------------------------------ checks
    def limits(self):
        models = self.srv.get("/v1/models")
        c = self.c()
        h = self.srv.health() or {}
        mml = None
        try:
            mml = models["data"][0].get("max_model_len")
        except (KeyError, IndexError, TypeError):
            pass
        count, tok_limit, _ = self.srv.tokenize([{"role": "user", "content": "hi"}])
        line = startup_line(self.a.startup_log) if self.a.startup_log else None
        lp, lmode, lpart, lw = line or (None, None, None, None)
        hpool = C.kv_pool(c)
        window = tok_limit or mml or h.get("context_window") or self.a.reported_window or lw
        pool = hpool or self.a.reported_pool or lp
        part = c.get("kv_shard.part_rows") or lpart
        info = {"tokenize_max_model_len": tok_limit, "models_max_model_len": mml,
                "health_context_window": h.get("context_window"),
                "health_kv_shard": {"parts": c.get("kv_shard.parts"), "part_rows": c.get("kv_shard.part_rows")},
                "startup_line": {"pool": lp, "mode": lmode, "part": lpart, "window": lw} if line else None,
                "startup_log": self.a.startup_log, "reported_window": self.a.reported_window or None,
                "reported_pool": self.a.reported_pool or None, "pool": pool, "part": part}
        self.window = window
        # the window: one conversation's limit, from /tokenize (and every other source that reports it)
        why = []
        if window is None:
            why.append("no single-conversation limit reported (/tokenize max_model_len)")
        lims = {w for w in (tok_limit, mml, h.get("context_window"), lw, self.a.reported_window or None) if w is not None}
        if len(lims) > 1:
            why.append(f"the window differs between sources: {sorted(lims)}")
        if self.a.expect_window and (window or 0) < self.a.expect_window:
            why.append(f"window {window} < --expect-window {self.a.expect_window}")
        self.rc.check("limits: window", not why, "; ".join(why) or f"one conversation up to {window} tokens", **info)
        # the pool: all positions both parts hold, reported apart from the window
        if pool is None:
            self.rc.add("limits: pool", "SKIP", "no total pool reported (no /health kv_shard parts x part_rows, "
                        "--startup-log line or --reported-pool)" + (f"; --expect-pool {self.a.expect_pool} not judged"
                                                                     if self.a.expect_pool else ""), **info)
        else:
            why = []
            pools = {x for x in (hpool, self.a.reported_pool or None, lp) if x is not None}
            if len(pools) > 1:
                why.append(f"the pool differs between sources: {sorted(pools)}")
            if lpart is not None and c.get("kv_shard.part_rows") is not None and lpart != c["kv_shard.part_rows"]:
                why.append(f"part {c['kv_shard.part_rows']} (/health) != {lpart} (start-up line)")
            if window is not None and window > pool:
                why.append(f"window {window} > pool {pool}")
            if self.a.expect_pool and pool != self.a.expect_pool:
                why.append(f"pool {pool} != --expect-pool {self.a.expect_pool}")
            self.rc.check("limits: pool", not why, "; ".join(why) or f"pool {pool} ({lmode or '?'}, part "
                          f"{part}), a conversation up to {window}", **info)
        if self.a.over_window and window:
            self.over_window(window)

    def over_window(self, window):
        """A prompt past the window is refused (HTTP 400 context_length), while the pool may be twice it."""
        target = window + self.a.over_window
        n, got = self.size(lambda k: FILLER * k, target, lo=target, hi=target + 2 * self.a.over_window)
        if got is None:
            return self.rc.add("limits: over-window", "SKIP", "the server has no /tokenize to size the prompt")
        if got <= window:
            return self.rc.add("limits: over-window", "SKIP", f"could not size a prompt past the window ({got} <= {window})")
        r = self.srv.post_json("/v1/chat/completions", self.srv.body([{"role": "user", "content": FILLER * n}], 16),
                               timeout=600)
        err = str(r.get("error", ""))
        refused = r.get("http_status") == 400 and "context" in err.lower()
        self.rc.check("limits: over-window", refused, f"a {got}-token prompt (window {window}): "
                      + (f"refused, {err[:160]}" if refused else
                         f"NOT refused as context_length (status {r.get('http_status')}, {err[:160] or 'answered'})"),
                      prompt_tokens=got, window=window, http_status=r.get("http_status"), error=err[:300])

    def size(self, content, target, lo=None, hi=None):
        """(n, tokens): n units of content(n) whose rendered prompt is within [lo, hi] (default [target - 256,
        target]), found with /tokenize; (n, None) without /tokenize."""
        lo = target - 256 if lo is None else lo
        hi = target if hi is None else hi

        def count(k):
            return self.srv.tokenize([{"role": "user", "content": content(k)}])[0]
        ca, cb = count(256), count(512)
        if ca is None or cb is None:
            return 0, None
        per = (cb - ca) / 256
        n = max(1, int(256 + ((lo + hi) / 2 - ca) / per))
        got = None
        for _ in range(8):
            got = count(n)
            if lo <= got <= hi:
                break
            adj = int(((lo + hi) / 2 - got) / per) or (-1 if got > hi else 1)
            n = max(1, n + adj)
        return n, got

    def at_once(self, named, mt):
        """Every (name, messages) at once -> {name: row}, with a Sampler over them."""
        with self.sampler() as smp:
            with concurrent.futures.ThreadPoolExecutor(len(named)) as ex:
                futs = {n: ex.submit(self.srv.chat, msgs, mt) for n, msgs in named.items()}
                rows = {n: f.result() for n, f in futs.items()}
        return rows, smp

    def owners(self):
        c0 = self.c()
        rows, smp = self.at_once(self.p.owners, self.dt)
        c1 = self.c()
        bad = [f"{n}: {d}" for n in rows for ok, d in [self.same(n, rows[n])] if not ok]
        self.rc.check("owners: replies == baseline", not bad, "; ".join(bad) or "both equal", rows=rows)
        r = self.rounds(c0, c1)
        if not smp.parts or r is None:
            return self.rc.add("owners: both parts decoded", "SKIP", f"no kv_shard per_part / rounds reported "
                               f"(kv_shard fields: {sorted(self.kv(c1))[:12]})", sampler=smp.summary())
        live = smp.max_live + [0, 0]
        ok = live[0] >= 1 and live[1] >= 1 and r > 0
        self.rc.check("owners: both parts decoded", ok, f"most live streams per part {smp.max_live} over "
                      f"{smp.samples} samples, sharded decode rounds +{r}", sampler=smp.summary(), rounds=r)

    def mixed(self):
        c0 = self.c()
        rows, smp = self.at_once(self.p.mixed, self.mt)
        c1 = self.c()
        bad = [f"{n}: {d}" for n in rows for ok, d in [self.same(n, rows[n])] if not ok]
        self.rc.check("mixed: 8 at once == baseline", not bad, "; ".join(bad) or "8 of 8 equal", rows=rows)
        r = self.rounds(c0, c1)
        if not smp.parts or r is None:
            return self.rc.add("mixed: rounds spanned both owners", "SKIP", "no kv_shard per_part / rounds reported",
                               sampler=smp.summary())
        self.rc.check("mixed: rounds spanned both owners", smp.both > 0 and r > 0,
                      f"{smp.both} of {smp.samples} samples had both parts live, sharded decode rounds +{r}",
                      sampler=smp.summary(), rounds=r)

    def burst(self):
        if not self.p.burst:
            return self.rc.add("burst", "SKIP", "the baseline was recorded without burst rows (--burst 0, or an "
                               "older record)")
        c0 = self.c()
        rows, smp = self.at_once(self.p.burst, self.mt)
        c1 = self.c()
        bad = [f"{n}: {d}" for n in rows for ok, d in [self.same(n, rows[n])] if not ok]
        self.rc.check(f"burst: {len(rows)} on one prompt == baseline", not bad, "; ".join(bad) or
                      f"{len(rows)} of {len(rows)} equal", rows=rows)
        d = C.deltas(c0, c1, ["kv_shard.transfers_failed", "kv_shard.copies", "kv_shard.carries"])
        if d["kv_shard.transfers_failed"] is None:
            return self.rc.add("burst: no transfer failed", "SKIP", "kv_shard.transfers_failed not reported", deltas=d)
        self.rc.check("burst: no transfer failed", d["kv_shard.transfers_failed"] == 0,
                      f"{d} (same-part copies run on the owner and count no cross-part copy)", deltas=d,
                      sampler=smp.summary())

    def carry(self):
        c0 = self.c()
        r = self.srv.chat(self.p.carry_seed, self.mt)
        ok, d = self.same("CA", r)
        bad = [] if ok else [f"CA: {d}"]
        log = []
        carried_at = None
        if "kv_shard.carries" not in c0:
            C.log("  carry: the server reports no kv_shard.carries; hashes are still compared")
        for i, msgs in enumerate(self.p.carry_hits):
            f = self.srv.chat(self.p.filler(f"carry-{i}-{secrets.token_hex(3)}"), 8)
            cb = self.c()
            t0 = time.monotonic()
            h = self.srv.chat(msgs, self.mt)
            secs = time.monotonic() - t0
            ca = self.c()
            ok, d = self.same(f"CA{i + 1}", h)
            dc = C.delta(cb, ca, "kv_shard.carries")
            mv = C.deltas(cb, ca, ["kv_shard.moved_tokens", "kv_shard.moved_bytes", "kv_shard.copy_bytes"])
            log.append({"round": i + 1, "filler_error": f.get("error"), "equal": ok, "carries": dc, "moved": mv,
                        "seconds": round(secs, 3), "cached_tokens": h.get("cached_tokens"), "kv": self.kv(ca)})
            if not ok:
                bad.append(f"CA{i + 1}: {d}")
            if dc:
                carried_at = i + 1
                break
        self.rounds_to_carry = carried_at
        if bad:
            return self.rc.add("carry", "FAIL", "; ".join(bad), rounds=log)
        if carried_at is None:
            return self.rc.add("carry", "SKIP", f"no carry after {len(log)} filler rounds (replies all equal); a smaller "
                               "TF_DS_POOL or larger --filler-tokens makes the pressure", rounds=log)
        mv = log[-1]["moved"]
        per = (mv["kv_shard.moved_bytes"] / mv["kv_shard.moved_tokens"]) if mv.get("kv_shard.moved_tokens") else None
        self.rc.add("carry", "PASS", f"carried at round {carried_at} in {log[-1]['seconds']}s; every reply equal; moved "
                    f"{mv.get('kv_shard.moved_tokens')} tokens, {mv.get('kv_shard.moved_bytes')} B"
                    + (f" ({per:.0f} B a token)" if per else ""), rounds=log, bytes_per_token=per)

    def twins(self):
        c0 = self.c()
        with self.sampler() as smp:
            r = self.srv.chat(self.p.twin, self.mt)
        c1 = self.c()
        ok, d = self.same("TW", r)
        self.rc.check("twins: reply == baseline", ok, d, row=r)
        dd = C.deltas(c0, c1, ["kv_shard.twins_made", "kv_shard.prompt_fetches", "kv_shard.twins_revoked"])
        if dd["kv_shard.twins_made"] is None:
            return self.rc.add("twins: borrowed", "SKIP", f"kv_shard.twins_made not reported (kv_shard fields: "
                               f"{sorted(self.kv(c1))[:12]})", deltas=dd)
        self.rc.check("twins: borrowed", dd["kv_shard.twins_made"] >= 1, f"{dd}", deltas=dd, sampler=smp.summary())
        if dd["kv_shard.twins_made"] >= 1 and dd["kv_shard.prompt_fetches"] is not None:
            if dd["kv_shard.twins_revoked"]:
                # K6: a twin revoked between chunks goes on by fetching, with the same bits (the reply check above)
                self.rc.add("twins: the twinned pass fetched nothing", "PASS",
                            f"twins_revoked +{dd['kv_shard.twins_revoked']}: revoked, then fetched (prompt_fetches "
                            f"+{dd['kv_shard.prompt_fetches']}), reply equal", deltas=dd)
            else:
                self.rc.check("twins: the twinned pass fetched nothing", dd["kv_shard.prompt_fetches"] == 0,
                              f"prompt_fetches +{dd['kv_shard.prompt_fetches']}, twins_revoked +0", deltas=dd)
        left = [p.get("twin_rows") for p in C.kv_parts(c1)]
        if left:
            self.rc.check("twins: released", not any(left), f"twin_rows per part after the reply: {left}")

    def move(self):
        if not self.p.move_big or self.missing("MV_A", "MV_B", "MV_C"):
            return self.rc.add("move", "SKIP", "the baseline was recorded without move rows (--move-tokens 0, or an "
                               "older record)")
        c0 = self.c()
        rows, firsts, ths = {}, {}, []
        with self.sampler() as smp:
            for n, msgs in self.p.move_side.items():
                firsts[n] = threading.Event()
                th = threading.Thread(target=lambda n=n, msgs=msgs: rows.update(
                    {n: self.srv.stream(msgs, self.dt, first=firsts[n])}))
                th.start()
                ths.append(th)
                end = time.monotonic() + self.a.move_wait   # one at a time: each takes the part with fewer live streams
                while not firsts[n].wait(0.05) and th.is_alive() and time.monotonic() < end:
                    pass
            rows["MV_C"] = self.srv.chat(self.p.move_big, self.mt)
            for th in ths:
                th.join(self.a.timeout)
        c1 = self.c()
        bad = [f"{n}: {d}" for n in ("MV_A", "MV_B", "MV_C") for ok, d in [self.same(n, rows.get(n) or {"error": "no reply"})]
               if not ok]
        self.rc.check("move: replies == baseline", not bad, "; ".join(bad) or "3 of 3 equal (the moved stream too)",
                      rows=rows, sampler=smp.summary())
        dm = C.delta(c0, c1, "kv_shard.moves")
        if dm is None:
            return self.rc.add("move: a live stream moved", "SKIP", "kv_shard.moves not reported")
        if not dm:
            return self.rc.add("move: a live stream moved", "SKIP", f"no move (moves +0, no_part "
                               f"+{C.delta(c0, c1, 'kv_shard.no_part')}): the big prompt fitted, or waited; a larger "
                               "--move-tokens (near the window) or longer side prompts make the pressure",
                               kv_after=self.kv(c1))
        self.rc.add("move: a live stream moved", "PASS", f"moves +{dm}, moved_tokens "
                    f"+{C.delta(c0, c1, 'kv_shard.moved_tokens')}", kv_after=self.kv(c1))

    def provoke(self, seed_msgs, seed_name, tag):
        """Seed a conversation and push it out with rounds_to_carry fillers, so its next hit carries."""
        r = self.srv.chat(seed_msgs, self.mt)
        ok0, d0 = self.same(seed_name, r)
        for i in range(self.rounds_to_carry):
            self.srv.chat(self.p.filler(f"{tag}-{i}-{secrets.token_hex(3)}"), 8)
        return ok0, d0

    def carry_running(self, c0, th):
        """True once a carry shows (carries grew, or a part holds rows), or once the request has ended (which only
        stops the wait: the callers judge th.is_alive() again right before the fault, and SKIP when it had ended)."""
        c = self.c()
        holds = [p.get("hold_rows", 0) for p in C.kv_parts(c)]
        return (C.delta(c0, c, "kv_shard.carries") or 0) > 0 or any(holds) or not th.is_alive()

    def cancel_transfer(self):
        if self.rounds_to_carry is None:
            return self.rc.add("cancel-transfer", "SKIP", "the carry check provoked no carry to repeat")
        ok0, d0 = self.provoke(self.p.cancel_seed, "CB", "cancel")
        c0 = self.c()
        ev = threading.Event()
        box = {}
        th = threading.Thread(target=lambda: box.update(row=self.srv.stream(self.p.cancel_after, self.mt, cancel=ev)))
        th.start()
        seen = C.wait_until(lambda: self.carry_running(c0, th), self.a.cancel_wait, 0.02)
        if not seen:
            time.sleep(self.a.cancel_after)
        in_flight = th.is_alive()       # just before the cancel: a request already answered tests nothing
        ev.set()
        th.join(60)

        def quiet():
            ps = C.kv_parts(self.c())
            return bool(ps) and not any(p.get("hold_rows", 0) or p.get("live", 0) for p in ps)
        settled = C.wait_until(quiet, self.a.settle_wait, 0.2)
        alive = self.srv.alive()
        c1 = self.c()
        again = self.srv.chat(self.p.cancel_after, self.mt)
        ok, d = self.same("CB_after", again)
        dc = C.delta(c0, c1, "kv_shard.carries")
        data = {"cancelled_row": box.get("row"), "again": again, "kv_before": self.kv(c0), "kv_after": self.kv(c1),
                "parts_after": C.kv_parts(c1), "in_flight_at_cancel": in_flight}
        if not (alive and settled and ok and ok0):
            return self.rc.add("cancel-transfer", "FAIL", f"alive={alive} settled={bool(settled)} (hold_rows/live "
                               f"{[(p.get('hold_rows'), p.get('live')) for p in C.kv_parts(c1)]}) seed: {d0}; "
                               f"again: {d}", **data)
        cut = bool((box.get("row") or {}).get("cancelled"))
        if not in_flight or not cut:
            return self.rc.add("cancel-transfer", "SKIP", "the request finished before the cancel (" +
                               ("it had ended" if not in_flight else "its row shows no cut before the end") +
                               f"): no client left during a carry; carries +{dc}, the request again: {d}", **data)
        if not dc:
            return self.rc.add("cancel-transfer", "SKIP", f"no carry began this time (carries +{dc}); the request "
                               f"again: {d}", **data)
        self.rc.add("cancel-transfer", "PASS", f"carry begun (+{dc}) and its client left; nothing held after, "
                    f"healthy, the request again: {d}", **data)

    def kill_rank1(self):
        if not self.a.kill_cmd:
            return self.rc.add("kill-rank1", "SKIP", "no --kill-cmd")
        if self.rounds_to_carry is None:
            return self.rc.add("kill-rank1", "SKIP", "the carry check provoked no carry to repeat")
        self.provoke(self.p.kill_seed, "CK", "kill")
        c0 = self.c()
        box = {}
        th = threading.Thread(target=lambda: box.update(row=self.srv.stream(self.p.kill_after, self.mt)), daemon=True)
        th.start()
        seen = C.wait_until(lambda: self.carry_running(c0, th), self.a.cancel_wait, 0.02)
        if not th.is_alive():           # just before the kill: a request already answered tests no hang
            th.join(1)
            return self.rc.add("kill-rank1", "SKIP", "the request finished before the kill: --kill-cmd not run (the "
                               "pair is still up)", row=box.get("row"), kv_after=self.kv(self.c()))
        t0 = time.monotonic()
        k = subprocess.run(self.a.kill_cmd, shell=True, capture_output=True, text=True, timeout=120)
        th.join(self.a.kill_deadline)
        ended = not th.is_alive()
        secs = time.monotonic() - t0
        alive = self.srv.alive()
        row = box.get("row") or {}
        data = {"carry_seen": bool(seen), "kill_exit": k.returncode, "kill_output": (k.stdout + k.stderr)[-500:],
                "seconds_to_end": round(secs, 2), "row": row, "server_healthy_after": alive}
        if k.returncode != 0:
            return self.rc.add("kill-rank1", "FAIL", f"--kill-cmd failed (exit {k.returncode})", **data)
        if not ended:
            return self.rc.add("kill-rank1", "FAIL", f"the request in flight did not end within --kill-deadline "
                               f"{self.a.kill_deadline}s after rank 1 was killed: rank 0 hangs", **data)
        self.rc.add("kill-rank1", "PASS", f"the request ended {secs:.1f}s after the kill "
                    f"({row.get('error') or 'answered'}); server healthy after: {alive}"
                    + ("" if seen else " (no carry was seen running at the kill)"), **data)

    def idle(self):
        def quiet():
            ps = C.kv_parts(self.c())
            return bool(ps) and not any(p.get("live", 0) for p in ps)
        C.wait_until(quiet, self.a.settle_wait, 0.5)
        c = self.c()
        ps = C.kv_parts(c)
        if not ps:
            return self.rc.add("idle", "SKIP", "no kv_shard per_part reported")
        rows = int(c.get("kv_shard.part_rows") or 0)
        why = []
        if c.get("kv_shard.parts") != 2 or len(ps) != 2:
            why.append(f"parts {c.get('kv_shard.parts')} ({len(ps)} rows), not 2")
        for i, p in enumerate(ps):
            held = {k: p.get(k) for k in ("live", "live_rows", "twin_rows", "hold_rows") if p.get(k)}
            if held:
                why.append(f"part {i} still holds {held}")
            if rows and p.get("free", 0) + p.get("kept_rows", 0) > rows:
                why.append(f"part {i}: free {p.get('free')} + kept_rows {p.get('kept_rows')} > part_rows {rows}")
        df = C.delta(self.c_start or {}, c, "kv_shard.transfers_failed") if self.c_start else None
        if df:
            why.append(f"transfers_failed +{df} over the run")
        self.rc.check("idle", not why, "; ".join(why) or f"2 parts of {rows}, nothing held, no transfer failed",
                      parts=ps, transfers_failed_delta=df)

    def needle(self):
        sizes = [x.strip() for x in self.a.needle_sizes.split(",") if x.strip()]
        if not sizes:
            return
        window = getattr(self, "window", None)
        for s in sizes:
            target = (window - self.a.needle_answer - 64) if s == "max" and window else int(s) if s != "max" else None
            name = f"needle {s}"
            if target is None:
                self.rc.add(name, "SKIP", "no window reported")
                continue
            if window and target + self.a.needle_answer > window:
                self.rc.add(name, "SKIP", f"{target} tokens is over the window ({window})")
                continue
            self.needle_one(name, target)

    def needle_one(self, name, target):
        intro = ("Read this synthetic archive. Three special entries give the secret values for ALPHA, BETA and "
                 "GAMMA. Remember their exact values and answer the final question.\n")
        question = ("\nWhat are the secret values for ALPHA, BETA and GAMMA? Return only one JSON object with these "
                    "three keys and their six-digit string values.\n")

        def content(n):
            a, b, c = n // 10, n * 4 // 10, n * 4 // 10
            return "".join([intro, FILLER * a, f"\nSPECIAL ENTRY: ALPHA secret value is {SECRETS['ALPHA']}.\n",
                            FILLER * b, f"\nSPECIAL ENTRY: BETA secret value is {SECRETS['BETA']}.\n",
                            FILLER * c, f"\nSPECIAL ENTRY: GAMMA secret value is {SECRETS['GAMMA']}.\n",
                            FILLER * (n - a - b - c), question])

        def count(n):
            return self.srv.tokenize([{"role": "user", "content": content(n)}])[0]
        ca, cb = count(256), count(512)
        if ca is None or cb is None:
            return self.rc.add(name, "SKIP", "the server has no /tokenize to size the prompt")
        per = (cb - ca) / 256
        n = max(1, int(256 + (target - ca) / per))
        got = None
        for _ in range(6):
            got = count(n)
            if target - 256 <= got <= target:
                break
            adj = int((target - got) / per) or (-1 if got > target else 1)
            n = max(1, n + adj)
        t0 = time.monotonic()
        row = self.srv.stream([{"role": "user", "content": content(n)}], self.a.needle_answer, keep_text=True)
        text = row.pop("text", "")
        found = {k: v in text for k, v in SECRETS.items()}
        ok = row.get("error") is None and all(found.values())
        self.rc.check(name, ok, f"prompt {row.get('prompt_tokens')} tokens (tokenize {got}, target {target}), ttft "
                      f"{row.get('ttft')}s, total {time.monotonic() - t0:.1f}s, found {found}"
                      + (f", error {row.get('error')}" if row.get("error") else ""),
                      row={k: v for k, v in row.items() if k != "text"}, answer=text[-300:])

    def go(self) -> int:
        a = self.a
        if not self.srv.wait_healthy(a.health_timeout):
            self.rc.add("health", "FAIL", f"{self.srv.url} not healthy: {self.srv.health()}")
            return self.rc.finish(a.receipt)
        self.c_start = self.c()
        self.rc.doc["counters_before"] = self.c_start
        checks = a.checks.split(",") if a.checks else ["limits", "owners", "mixed", "burst", "carry", "twins", "move",
                                                        "cancel-transfer", "needle", "idle"]
        for chk in checks:
            try:
                getattr(self, re.sub(r"\W", "_", chk))()
            except Exception as e:  # noqa: BLE001
                self.rc.add(chk, "FAIL", f"{type(e).__name__}: {e}")
        self.rc.doc["counters_after"] = self.c() if "kill-rank1" not in checks else {}
        return self.rc.finish(a.receipt, a.strict)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for p in (sub.add_parser("record"), sub.add_parser("run")):
        p.add_argument("--url", default=C.default_url())
        p.add_argument("--model", default=None)
        p.add_argument("--timeout", type=float, default=3600)
        p.add_argument("--label", default="")
    r = sub.choices["record"]
    r.add_argument("--out", required=True)
    r.add_argument("--salt")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--max-tokens", type=int, default=48)
    r.add_argument("--decode-tokens", type=int, default=300, help="the two owners' and the move's side decodes")
    r.add_argument("--owner-tokens", type=int, default=6000)
    r.add_argument("--carry-tokens", type=int, default=64000)
    r.add_argument("--carry-rounds", type=int, default=10)
    r.add_argument("--twin-tokens", type=int, default=96000)
    r.add_argument("--filler-tokens", type=int, default=64000)
    r.add_argument("--burst", type=int, default=4, help="requests at once on one prompt (0: no burst rows)")
    r.add_argument("--burst-tokens", type=int, default=24000)
    r.add_argument("--move-side-tokens", type=int, default=8000, help="the two decodes the big prompt must move")
    r.add_argument("--move-tokens", type=int, default=256000, help="the big fresh prompt: near the window of the "
                                                                   "configuration that runs `move` (0: no move rows)")
    u = sub.choices["run"]
    u.add_argument("--baseline", required=True)
    u.add_argument("--receipt")
    u.add_argument("--checks", help="comma list (default: limits,owners,mixed,burst,carry,twins,move,cancel-transfer,"
                                    "needle,idle; kill-rank1 only when named, last)")
    u.add_argument("--expect-window", type=int, default=0)
    u.add_argument("--expect-pool", type=int, default=0)
    u.add_argument("--startup-log", help="a saved rank-0 log (docker logs) holding the `dsv41: pool ...` start-up line")
    u.add_argument("--reported-pool", type=int, default=0, help="the pool, when read elsewhere")
    u.add_argument("--reported-window", type=int, default=0, help="the window, when read elsewhere")
    u.add_argument("--over-window", type=int, default=1024, help="limits: a prompt this far past the window must be "
                                                                 "refused (0: not sent)")
    u.add_argument("--needle-sizes", default="524288,max", help="tokens, or max (the largest the window admits); "
                                                                 "empty: none")
    u.add_argument("--needle-answer", type=int, default=128)
    u.add_argument("--cancel-wait", type=float, default=120)
    u.add_argument("--cancel-after", type=float, default=0.5, help="leave after this when no transfer is seen")
    u.add_argument("--settle-wait", type=float, default=120, help="cancel-transfer, idle: how long nothing may stay "
                                                                    "held or live after the requests end")
    u.add_argument("--move-wait", type=float, default=600, help="move: wait for each side decode's first token")
    u.add_argument("--kill-cmd", help="kill-rank1: a shell command that kills rank 1's engine")
    u.add_argument("--kill-deadline", type=float, default=90, help="kill-rank1: the request in flight must end "
                                                                    "within this many seconds of the kill")
    u.add_argument("--sample-ms", type=float, default=100, help="/health sampling period while a check runs")
    u.add_argument("--health-timeout", type=float, default=900)
    u.add_argument("--allow-weak", action="store_true")
    u.add_argument("--strict", action="store_true")
    a = C.use_api_key_args(C.add_api_key_arg(ap).parse_args())
    if a.mode == "record":
        a.model = a.model or os.environ.get("MODEL", C.DEFAULT_MODEL)
        return record(a)
    return Run(a).go()


if __name__ == "__main__":
    sys.exit(main())
