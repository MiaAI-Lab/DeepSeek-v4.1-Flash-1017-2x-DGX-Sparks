#!/usr/bin/env python3
"""Self-test of the port's validation tools against tools/fake_server.py, on any machine (no GPU, no model):

    python3 -I tools/selftest.py [--only compile,counters,exact,spill,kv,prefix,runbook] [--keep]

Each tool must PASS against a correct fake and FAIL against a fake with the fault it exists to catch (--mutate), so a
check that cannot fail is caught too. Exit 1 when any expectation is not met. Logs and receipts go to a temporary folder
(kept with --keep, or on a failure).
"""
import argparse
import os
import py_compile
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable or "python3"
FAST = ["--prefill-us-per-token", "3", "--decode-ms-per-token", "1"]
results = []


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Fake:
    def __init__(self, work, name, *args):
        self.work, self.name, self.args = work, name, list(args)
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.start()

    def start(self, *extra):
        self.log = open(os.path.join(self.work, f"fake-{self.name}.log"), "a")
        self.p = subprocess.Popen([PY, "-I", os.path.join(HERE, "fake_server.py"), "--port", str(self.port)]
                                  + self.args + list(extra), stdout=self.log, stderr=subprocess.STDOUT)
        end = time.monotonic() + 20
        while time.monotonic() < end:
            try:
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(self.url + "/health", timeout=2).read()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError(f"fake {self.name} did not start")

    def stop(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()
        self.log.close()

    def restart(self, *extra):
        self.stop()
        self.start(*extra)


def run(work, label, args, expect_ok=True, env=None):
    """Run a tool; it must exit 0 (expect_ok) or non-zero (a planted fault caught)."""
    log = os.path.join(work, label.replace(" ", "_").replace("/", "_") + ".log")
    t0 = time.monotonic()
    with open(log, "w") as f:
        p = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})),
                           timeout=900)
    ok = (p.returncode == 0) == expect_ok
    results.append((label, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {label}: exit {p.returncode} (expected {'0' if expect_ok else 'non-zero'}), "
          f"{time.monotonic() - t0:.1f}s, log {log}", flush=True)
    if not ok:
        with open(log) as f:
            print("      " + "      ".join(f.readlines()[-15:]), flush=True)
    return ok


def tool(name):
    return [PY, "-I", os.path.join(HERE, name)]


def t_exact(work):
    base = Fake(work, "exact-base", *FAST)
    cand = Fake(work, "exact-cand", *FAST)
    bad = Fake(work, "exact-conc-wrong", "--mutate", "conc-wrong", *FAST)
    weak = Fake(work, "exact-weak", "--mutate", "no-token-sha", *FAST)
    j = lambda n: os.path.join(work, n)  # noqa: E731
    try:
        old = j("exact_old.py")
        g = subprocess.run(["git", "-C", ROOT, "show", "e65785c:tools/exact.py"], capture_output=True, text=True)
        if g.returncode == 0:
            open(old, "w").write(g.stdout)
            run(work, "exact: old 4-stream tool records", [PY, "-I", old, "record", "--url", base.url, "--out", j("x-old.json")])
            run(work, "exact: 8-stream record --like a 4-stream record", tool("exact.py") + [
                "record", "--url", cand.url, "--out", j("x-new-like-old.json"), "--like", j("x-old.json")])
            run(work, "exact: compare 4-stream vs 8-stream record", tool("exact.py") + [
                "compare", j("x-old.json"), j("x-new-like-old.json")])
        else:
            print("SKIP  exact: no git history for the old exact.py")
        run(work, "exact: record baseline (8 streams)", tool("exact.py") + ["record", "--url", base.url, "--out", j("x-base.json")])
        run(work, "exact: record candidate --like", tool("exact.py") + [
            "record", "--url", cand.url, "--out", j("x-cand.json"), "--like", j("x-base.json")])
        run(work, "exact: compare identical", tool("exact.py") + ["compare", j("x-base.json"), j("x-cand.json")])
        run(work, "exact: 4-stream record still works", tool("exact.py") + [
            "record", "--url", cand.url, "--out", j("x-cand4.json"), "--like", j("x-base.json"), "--streams", "4"])
        run(work, "exact: compare 8-stream baseline vs 4-stream candidate (extra cases are notes)", tool("exact.py") + [
            "compare", j("x-base.json"), j("x-cand4.json")])
        run(work, "exact: fault conc-wrong caught by record (solo != conc_5..8)", tool("exact.py") + [
            "record", "--url", bad.url, "--out", j("x-bad.json"), "--like", j("x-base.json")], expect_ok=False)
        run(work, "exact: fault conc-wrong caught by compare", tool("exact.py") + [
            "compare", j("x-base.json"), j("x-bad.json")], expect_ok=False)
        run(work, "exact: weak record (no token_sha) refused by compare", tool("exact.py") + [
            "record", "--url", weak.url, "--out", j("x-weak.json"), "--like", j("x-base.json")])
        run(work, "exact: compare refuses weak", tool("exact.py") + ["compare", j("x-base.json"), j("x-weak.json")],
            expect_ok=False)
    finally:
        for f in (base, cand, bad, weak):
            f.stop()


def t_counters(work):
    """checklib reads the engine's shapes: /health prompt_cache.spill as an array of {rank, ...}, /metrics
    spill_<name>{rank="N"}; every TierCounts field per rank and over the ranks, from the fake and from literal text."""
    sys.path.insert(0, HERE)
    import checklib as C
    h = {"status": "ok", "prompt_cache": {"disk_lookups": 3, "spill": [
        {"rank": 0, "writes_committed": 2, "disk_free_bytes": 100, "no_space": 1},
        {"rank": 1, "writes_committed": 3, "disk_free_bytes": 50, "no_space": 0}]}}
    m = ('# HELP tensorfold:spill_writes_committed x\n# TYPE tensorfold:spill_writes_committed counter\n'
         'tensorfold:spill_writes_committed{rank="0"} 2\ntensorfold:spill_writes_committed{rank="1"} 3\n'
         'tensorfold:spill_disk_free_bytes{rank="0"} 100\ntensorfold:spill_disk_free_bytes{rank="1"} 50\n'
         'tensorfold:prompt_cache_disk_lookups 3\ntensorfold:kv_pool_usage_ratio{lane="0"} 0.5\n')
    want = {"spill.writes_committed": 5, "spill.writes_committed.rank0": 2, "spill.writes_committed.rank1": 3,
            "spill.disk_free_bytes": 50, "prompt_cache.disk_lookups": 3}
    for label, got in (("checklib: engine-shaped /health (spill array per rank)", C.health_counters(h)),
                       ("checklib: engine-shaped /metrics (spill_*{rank=N})", C.metrics_counters(C.parse_metrics(m)))):
        bad = {k: got.get(k) for k, v in want.items() if got.get(k) != v}
        results.append((label, not bad))
        print(f"{'PASS' if not bad else 'FAIL'}  {label}{': ' + str(bad) if bad else ''}", flush=True)
    # KV sharding's shapes as shard-lanes merged them (status_routes.zig putKvParts; metrics.zig's own test text)
    hk = {"status": "ok", "kv_shard": {"parts": 2, "part_rows": 1050624, "carries": 3, "moved_bytes": 720,
                                       "per_part": [{"part": 0, "free": 10, "live_rows": 0, "kept_rows": 0,
                                                     "twin_rows": 0, "hold_rows": 0, "live": 1},
                                                    {"part": 1, "free": 20, "live_rows": 0, "kept_rows": 0,
                                                     "twin_rows": 4096, "hold_rows": 0, "live": 0}]}}
    mk = ('# TYPE tensorfold:kv_shard_parts gauge\ntensorfold:kv_shard_parts 2\n'
          '# TYPE tensorfold:kv_shard_part_rows gauge\ntensorfold:kv_shard_part_rows 1050624\n'
          '# TYPE tensorfold:kv_shard_carries counter\ntensorfold:kv_shard_carries 3\n'
          'tensorfold:kv_shard_moved_bytes 720\n# TYPE tensorfold:kv_shard_part_free gauge\n'
          'tensorfold:kv_shard_part_free{part="0"} 10\ntensorfold:kv_shard_part_free{part="1"} 20\n'
          'tensorfold:kv_shard_part_twin_rows{part="1"} 4096\ntensorfold:kv_shard_part_live{part="0"} 1\n')
    wk = {"kv_shard.parts": 2, "kv_shard.part_rows": 1050624, "kv_shard.carries": 3, "kv_shard.moved_bytes": 720,
          "kv_shard.part0.free": 10, "kv_shard.part1.free": 20, "kv_shard.part1.twin_rows": 4096,
          "kv_shard.part0.live": 1}
    for label, got in (("checklib: engine-shaped /health kv_shard (per_part rows)", C.health_counters(hk)),
                       ("checklib: engine-shaped /metrics kv_shard_* and kv_shard_part_*{part=N}",
                        C.metrics_counters(C.parse_metrics(mk)))):
        bad = {k: got.get(k) for k, v in wk.items() if got.get(k) != v}
        if C.kv_pool(got) != 2 * 1050624 or len(C.kv_parts(got)) != 2:
            bad["kv_pool/kv_parts"] = (C.kv_pool(got), C.kv_parts(got))
        results.append((label, not bad))
        print(f"{'PASS' if not bad else 'FAIL'}  {label}{': ' + str(bad) if bad else ''}", flush=True)
    f = Fake(work, "counters-kv", "--kv-shard", "--pool", "131072", *FAST)
    try:
        srv = C.Server(f.url)
        keys = [f"kv_shard.{n}" for n in C.KV_FIELDS] + [f"kv_shard.part{p}.{r}" for p in (0, 1) for r in C.KV_PART_FIELDS]
        for label, got in (("checklib: every kv_shard counter and part row from the fake's /health + /metrics",
                            srv.counters()),
                           ("checklib: every kv_shard counter and part row from the fake's /metrics alone",
                            C.metrics_counters(C.parse_metrics(srv.get("/metrics", raw=True) or "")))):
            miss = [k for k in keys if k not in got]
            results.append((label, not miss))
            print(f"{'PASS' if not miss else 'FAIL'}  {label}{': missing ' + str(miss[:6]) if miss else ''}", flush=True)
    finally:
        f.stop()
    d0, d1 = os.path.join(work, "cnt-r0"), os.path.join(work, "cnt-r1")
    f = Fake(work, "counters", "--spill-dir", d0, "--spill-dir-rank1", d1, *FAST)
    try:
        srv = C.Server(f.url)
        keys = [f"spill.{n}{r}" for n in C.SPILL_FIELDS for r in ("", ".rank0", ".rank1")]
        for label, got in (("checklib: every spill.* counter from the fake's /health + /metrics", srv.counters()),
                           ("checklib: every spill.* counter from the fake's /metrics alone",
                            C.metrics_counters(C.parse_metrics(srv.get("/metrics", raw=True) or "")))):
            miss = [k for k in keys if k not in got]
            results.append((label, not miss))
            print(f"{'PASS' if not miss else 'FAIL'}  {label}{': missing ' + str(miss[:6]) if miss else ''}", flush=True)
    finally:
        f.stop()


def t_spill(work):
    j = lambda n: os.path.join(work, n)  # noqa: E731
    d0, d1 = j("spill-r0"), j("spill-r1")
    base = Fake(work, "spill-base", *FAST)
    sp = ["--spill-dir", d0, "--spill-dir-rank1", d1, "--mem-points", "6", "--write-ms", "150"] + FAST
    cand = Fake(work, "spill-cand", *sp)
    rec, state = j("spill-base.json"), j("spill-state.json")
    dirs = ["--rank0-dir", d0, "--rank1-dir", d1, "--filler-tokens", "3000", "--commit-timeout", "30"]
    S = tool("spill_check.py")
    fakes = [base, cand]
    try:
        run(work, "spill: record baseline", S + ["record", "--url", base.url, "--out", rec, "--sizes", "16000,40000",
                                                 "--cancel-tokens", "60000", "--dual-tokens", "30000",
                                                 "--kill-tokens", "30000", "--floor-tokens", "20000"])
        run(work, "spill: main phase", S + ["run", "--url", cand.url, "--baseline", rec, "--phase", "main",
                                            "--receipt", j("r-spill-main.json")] + dirs)
        run(work, "spill: pictures (a point with its picture kept on both ranks, restored from disk)", S + [
            "run", "--url", cand.url, "--baseline", rec, "--phase", "main", "--checks", "pictures,settled", "--strict",
            "--receipt", j("r-spill-pictures.json")] + dirs)
        run(work, "spill: prerestart (with a picture conversation)", S + [
            "run", "--url", cand.url, "--baseline", rec, "--phase", "prerestart", "--state", state, "--pictures"] + dirs)
        cand.stop()
        run(work, "spill: drop rank 1's manifest while stopped", S + ["drop", "--state", state, "--rank1-dir", d1])
        cand.start()
        run(work, "spill: postrestart (missing-rank1 judged, the picture point restored, --strict)", S + [
            "run", "--url", cand.url, "--baseline", rec, "--phase", "postrestart", "--state", state, "--strict",
            "--pictures",
            "--receipt", j("r-spill-restart.json")] + dirs)
        kill = f"{PY} -c 'import os; os.kill({cand.p.pid}, 9)'"
        run(work, "spill: prekill (PIDs resolved first, kill -9 while K2's write is pending)", S + [
            "run", "--url", cand.url, "--baseline", rec, "--phase", "prekill", "--state", state,
            "--resolve-cmd", "true", "--kill-cmd", kill] + dirs)
        cand.restart()
        run(work, "spill: postkill (the in-flight case exercised, --strict)", S + [
            "run", "--url", cand.url, "--baseline", rec, "--phase", "postkill", "--state", state, "--strict",
            "--receipt", j("r-spill-kill.json")] + dirs)
        cand.stop()
        # the floor: the store's rule (floor + need + queued > free) on a simulated 1 GB disk, an empty store each time
        free = 10 ** 9
        for label, margin, extra, ok in (
                ("spill: free-disk floor just under free disk (refused on both ranks)", 10000,
                 ["--floor-margin-bytes", "10000"], True),
                ("spill: floor sizing guard (margin above F1's estimated write) caught", 10000,
                 ["--floor-margin-bytes", "200000"], False),
                ("spill: floor far under free disk (F1 admitted) caught", 400000, [], False)):
            f0, f1 = j(f"floor-{margin}-{len(extra)}-r0"), j(f"floor-{margin}-{len(extra)}-r1")
            ff = Fake(work, "spill-floor", "--spill-dir", f0, "--spill-dir-rank1", f1, "--disk-free-bytes", str(free),
                      "--spill-min-free-gib", repr((free - margin) / 2 ** 30), *FAST)
            fakes.append(ff)
            try:
                run(work, label, S + ["run", "--url", ff.url, "--baseline", rec, "--phase", "floor",
                                      "--rank0-dir", f0, "--rank1-dir", f1, "--commit-timeout", "10",
                                      "--bytes-per-position", "2", "--small-state-bytes", "1024",
                                      "--receipt", j(f"r-spill-floor-{margin}-{len(extra)}.json")] + extra, expect_ok=ok)
            finally:
                ff.stop()
        run(work, "spill: estimate prints F1's write", S + ["estimate", "--baseline", rec, "--bytes"])
        for mut, checks in (("restore-wrong", "evict"), ("no-disk", "evict"), ("ignore-crc", "corrupt-block,corrupt-small"),
                            ("no-pictures", "pictures")):
            m0, m1 = j(f"spill-{mut}-r0"), j(f"spill-{mut}-r1")
            f = Fake(work, f"spill-{mut}", "--spill-dir", m0, "--spill-dir-rank1", m1, "--mem-points", "6",
                     "--mutate", mut, *FAST)
            try:
                run(work, f"spill: fault {mut} caught ({checks})", S + [
                    "run", "--url", f.url, "--baseline", rec, "--phase", "main", "--checks", checks,
                    "--rank0-dir", m0, "--rank1-dir", m1, "--filler-tokens", "3000", "--commit-timeout", "30"],
                    expect_ok=False)
            finally:
                f.stop()
        # a store that restores without rank 1's presence: missing-rank1 must catch it
        m0, m1, mst = j("spill-ip-r0"), j("spill-ip-r1"), j("spill-ip-state.json")
        f = Fake(work, "spill-ignore-presence", "--spill-dir", m0, "--spill-dir-rank1", m1, "--mem-points", "6",
                 "--mutate", "ignore-presence", *FAST)
        fakes.append(f)
        mdirs = ["--rank0-dir", m0, "--rank1-dir", m1, "--commit-timeout", "30", "--state", mst]
        run(work, "spill: ignore-presence prerestart", S + ["run", "--url", f.url, "--baseline", rec, "--phase",
                                                            "prerestart"] + mdirs)
        f.stop()
        run(work, "spill: ignore-presence drop", S + ["drop", "--state", mst, "--rank1-dir", m1])
        f.start()
        run(work, "spill: fault ignore-presence caught (missing-rank1)", S + [
            "run", "--url", f.url, "--baseline", rec, "--phase", "postrestart"] + mdirs, expect_ok=False)
    finally:
        for f in fakes:
            if f.p.poll() is None:
                f.stop()
    t_parked(work)


def t_pair(work):
    """pair_check against a fake that a kill ends as a broken pair ends rank 0 (every stream ends, /health goes, the
    container shows `exited 75`, the 75 line is logged), and against the faults it exists to catch: rank 0 ending
    with another code, and a kill that changes nothing (a pair that does not exit)."""
    import json
    j = lambda n: os.path.join(work, n)  # noqa: E731
    P = tool("pair_check.py")
    f = Fake(work, "pair", *FAST, "--decode-ms-per-token", "300")   # 12-31 tokens: 4-9 s a reply
    st, rlog = j("pair-state"), j("pair-rank0.log")

    def killer(code):     # rank 0's last line and its container's state, then the process ends
        k = j(f"pair-kill-{code}.py")
        with open(k, "w") as fh:
            fh.write(f"import os\nopen({rlog!r}, 'w').write('tensorfold: the rank pair broke; exiting with status {code}\\n')\n"
                     f"open({st!r}, 'w').write('exited {code}\\n')\nos.kill({f.p.pid}, 9)\n")
        return f"{PY} {k}"

    def running():
        with open(st, "w") as fh:
            fh.write("running 0\n")
    common = ["--url", f.url, "--streams", "3", "--max-tokens", "2000", "--decode-s", "0.5", "--exit-cmd", f"cat {st}"]
    try:
        running()
        run(work, "pair: a kill ends every stream, /health goes, rank 0's container exits 75, the 75 line is logged",
            P + common + ["--kill-cmd", killer(75), "--end-deadline", "15", "--receipt", j("r-pair.json"),
                          "--expect-log", f"cat {rlog}::exiting with status 75"])
        f.restart()
        running()
        run(work, "pair: rank 0 ending with another code (137) is caught", P + common + [
            "--kill-cmd", killer(137), "--end-deadline", "15", "--receipt", j("r-pair-137.json")], expect_ok=False)
        failed = json.load(open(j("r-pair-137.json"))).get("failed")
        expect("pair: ... by the exit check alone", failed == ["exited"], f"failed {failed}")
        f.restart()
        running()
        run(work, "pair: a pair that does not exit is caught", P + common + [
            "--kill-cmd", "true", "--end-deadline", "3"], expect_ok=False)
    finally:
        f.stop()


def expect(label, ok, detail):
    results.append((label, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}", flush=True)


def t_kinds(work):
    """checklib.kind_of: the store's files by kind, its temporaries as temps, and the recipe's marker at the spill root
    (start.sh's) as neither: spill_check's "no temporaries" checks must not count it."""
    sys.path.insert(0, HERE)
    import checklib as C
    cases = {"blocks/ab12.blk": "block", "points/k.st": "small", "points/k.man": "manifest", "points/k.ids": "ids",
             "points/k.man.tmp": "temp", "points/.k": "temp", ".lost": "temp", "LOCK": "other",
             C.SPILL_MARKER: "marker", "points/" + C.SPILL_MARKER: "temp"}
    got = {p: C.kind_of(p) for p in cases}
    bad = {p: (got[p], k) for p, k in cases.items() if got[p] != k}
    expect("checklib: kind_of names the store's files, its temporaries and only the exact root marker as the marker",
           not bad, bad or f"{len(cases)} names")
    listing = {C.SPILL_MARKER: [0, 0.0], "points/k.man": [10, 0.0], "points/k.man.tmp": [3, 0.0]}
    temps = [p for p in listing if C.kind_of(p) == "temp"]
    expect("checklib: a clean spill root with the marker has no temporaries but its real one", temps == ["points/k.man.tmp"], temps)


def t_parked(work):
    """The parked gauge (requests waiting on a restore: the one reading it and every coalesced one) is non-zero while a
    coalesced restore is in flight and back to 0 after, which spill_check's settled check waits for."""
    import json
    import threading
    j = lambda n: os.path.join(work, n)  # noqa: E731
    d0, d1 = j("parked-r0"), j("parked-r1")
    f = Fake(work, "spill-parked", "--spill-dir", d0, "--spill-dir-rank1", d1, "--write-ms", "10",
             "--restore-ms-per-block", "150", *FAST)
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health():
        return json.loads(op.open(f.url + "/health", timeout=5).read()).get("prompt_cache") or {}

    def ask():
        body = json.dumps({"messages": [{"role": "user", "content": " ".join(f"w{i % 97}" for i in range(12000))}],
                          "max_tokens": 8}).encode()
        req = urllib.request.Request(f.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        return json.loads(op.open(req, timeout=120).read())

    try:
        ask()                                   # the seed: its point (10,240 tokens) goes to both ranks' disks
        end = time.monotonic() + 30
        while time.monotonic() < end:
            sp = health().get("spill") or []
            if sp and all(r["points"] >= 1 and not r["writes_pending"] for r in sp):
                break
            time.sleep(0.1)
        f.restart()                             # memory empty: the point is on disk only
        seen, out = [], []
        ts = [threading.Thread(target=lambda: out.append(ask())) for _ in range(2)]
        for t in ts:
            t.start()
        while any(t.is_alive() for t in ts):
            seen.append(health().get("parked") or 0)
            time.sleep(0.02)
        for t in ts:
            t.join()
        pc = health()
        shared, committed = pc.get("restores_shared") or 0, pc.get("disk_hits_committed") or 0
        expect("spill: a coalesced restore shows parked > 0 while in flight",
               len(out) == 2 and shared >= 1 and committed >= 1 and max(seen + [0]) >= 2,
               f"restores_shared={shared} disk_hits_committed={committed} parked max={max(seen + [0])} "
               f"over {len(seen)} readings, replies={len(out)}")
        expect("spill: parked back to 0 after (spill_check's settled reading)",
               pc.get("parked") == 0 and not pc.get("restores_active"),
               f"parked={pc.get('parked')} restores_active={pc.get('restores_active')}")
    finally:
        f.stop()


def t_kv_race(work):
    """kill-rank1 / cancel-transfer when the request ends before any carry shows (a carry that completes between two
    samples): SKIP, never PASS, and --kill-cmd is not run; twins: a revoke then a fetch is not a FAIL. Stubs, no server:
    the order of events is fixed."""
    sys.path.insert(0, HERE)
    import argparse as ap_
    import checklib as C
    import kv_shard_check_e2e as K

    class Srv:
        def __init__(self, cut=False):
            self.done, self.cut, self.t_end = False, cut, 0.0
            self.k = {"kv_shard.carries": 0, "kv_shard.twins_made": 0, "kv_shard.prompt_fetches": 0,
                      "kv_shard.twins_revoked": 0, "kv_shard.parts": 2, "kv_shard.part_rows": 100}

        def counters(self):
            c = dict(self.k)
            if self.done and time.monotonic() - self.t_end > 0.1:
                c["kv_shard.carries"] = 1      # the carry shows only once the request has ended (and its thread)
            for i in (0, 1):
                c.update({f"kv_shard.part{i}.hold_rows": 0, f"kv_shard.part{i}.live": 0,
                          f"kv_shard.part{i}.twin_rows": 0})
            return c

        def stream(self, msgs, mt, cancel=None, **kw):
            self.done, self.t_end = True, time.monotonic()
            return {"token_sha": "x", "cancelled": self.cut, "finish_reason": "stop"}

        def chat(self, msgs, mt):
            return {"token_sha": "x"}

        def alive(self):
            time.sleep(0.2)                    # so the late carry is counted in the check's after-reading
            return True

    def mk(srv, kill_cmd=None):
        r = object.__new__(K.Run)
        r.a = ap_.Namespace(kill_cmd=kill_cmd, cancel_wait=2, cancel_after=0.01, kill_deadline=2, settle_wait=1,
                            sample_ms=20, allow_weak=False)
        r.srv, r.rounds_to_carry, r.mt, r.c_start = srv, 0, 8, None
        r.rc = C.Receipt("t", {})
        r.p = ap_.Namespace(kill_seed=[], kill_after=[], cancel_seed=[], cancel_after=[], twin=[],
                            filler=lambda tag: [])
        r.same = lambda name, row: (True, "equal")
        return r

    def status(r, name):
        return [c["status"] for c in r.rc.doc["checks"] if c["check"] == name]
    flag = os.path.join(work, "kill-ran")
    r = mk(Srv(), f"touch '{flag}'")
    r.kill_rank1()
    results.append(("kv: kill-rank1 whose request ended before the kill is a SKIP (--kill-cmd not run)",
                    status(r, "kill-rank1") == ["SKIP"] and not os.path.exists(flag)))
    for cut in (False, True):
        r = mk(Srv(cut))
        r.cancel_transfer()
        results.append((f"kv: cancel-transfer whose request ended before the cancel is a SKIP (cancelled={cut})",
                        status(r, "cancel-transfer") == ["SKIP"]))
    srv = Srv()
    r = mk(srv)
    r.c = lambda: dict(srv.counters(), **({"kv_shard.twins_made": 1, "kv_shard.prompt_fetches": 3,
                                            "kv_shard.twins_revoked": 1} if srv.done else {}))
    srv.chat = lambda m, mt: (setattr(srv, "done", True), {"token_sha": "x"})[1]
    r.sampler = lambda: K.Sampler(srv, 1)
    r.twins()
    results.append(("kv: twins revoked then fetched (K6) is not a FAIL", "FAIL" not in
                    [c["status"] for c in r.rc.doc["checks"]] and status(r, "twins: the twinned pass fetched nothing")
                    == ["PASS"]))
    for label, ok in results[-4:]:
        print(f"{'PASS' if ok else 'FAIL'}  {label}", flush=True)


def t_kv(work):
    t_kv_race(work)
    j = lambda n: os.path.join(work, n)  # noqa: E731
    base = Fake(work, "kv-base", "--context", "65518", *FAST)
    sl = j("kv-startup.log")
    # decodes slow enough (40 ms a token) for the /health sampler to see both parts live and for a move to find them
    sh = ["--kv-shard", "--pool", "131072", "--mem-points", "16"] + FAST + ["--decode-ms-per-token", "40"]
    cand = Fake(work, "kv-cand", *sh, "--startup-log", sl)
    bad = Fake(work, "kv-shard-wrong", *sh, "--mutate", "shard-wrong")
    over = Fake(work, "kv-admit-over-window", *sh, "--mutate", "admit-over-window")
    leak = Fake(work, "kv-leak-hold", *sh, "--mutate", "leak-hold")
    tfetch = Fake(work, "kv-twin-fetch", *sh, "--mutate", "twin-fetch")
    kill = Fake(work, "kv-kill", *sh)
    hang = Fake(work, "kv-kill-hang", *sh, "--mutate", "kill-hang")
    rec = j("kv-base.json")
    K = tool("kv_shard_check_e2e.py")
    S = ["--sample-ms", "20", "--settle-wait", "5"]

    def kill_cmd(f):
        return (f"{PY} -I -c \"import urllib.request as u; u.build_opener(u.ProxyHandler({{}})).open(u.Request("
                f"'{f.url}/fake/kill-rank1', data=b'{{}}'), timeout=10).read()\"")
    try:
        run(work, "kv: record baseline", K + ["record", "--url", base.url, "--out", rec, "--carry-tokens", "20000",
                                              "--twin-tokens", "20000", "--filler-tokens", "12000",
                                              "--owner-tokens", "2000", "--decode-tokens", "100", "--burst-tokens",
                                              "6000", "--move-side-tokens", "8000", "--move-tokens", "60000"])
        run(work, "kv: run (all checks, pool from /health and the start-up line, over-window refused, needles 20000 "
                  "and max, burst, move, idle)", K + [
            "run", "--url", cand.url, "--baseline", rec, "--receipt", j("r-kv.json"), "--needle-sizes", "20000,max",
            "--startup-log", sl, "--expect-pool", "131072", "--expect-window", "65518", "--strict"] + S)
        run(work, "kv: fault shard-wrong caught", K + ["run", "--url", bad.url, "--baseline", rec, "--checks",
                                                       "owners,mixed"] + S, expect_ok=False)
        run(work, "kv: wrong --expect-pool caught", K + ["run", "--url", cand.url, "--baseline", rec, "--checks",
                                                         "limits", "--startup-log", sl, "--expect-pool", "2097152"],
            expect_ok=False)
        run(work, "kv: a start-up line that disagrees with /health caught", K + [
            "run", "--url", cand.url, "--baseline", rec, "--checks", "limits", "--reported-pool", "262144"],
            expect_ok=False)
        run(work, "kv: the pool from /health alone (no start-up line)", K + [
            "run", "--url", cand.url, "--baseline", rec, "--checks", "limits", "--expect-pool", "131072", "--strict"])
        run(work, "kv: no pool reported (unsharded) is a SKIP, not a FAIL", K + [
            "run", "--url", base.url, "--baseline", rec, "--checks", "limits", "--expect-pool", "131072"])
        run(work, "kv: ... and fails with --strict", K + ["run", "--url", base.url, "--baseline", rec, "--checks",
                                                          "limits", "--expect-pool", "131072", "--strict"],
            expect_ok=False)
        run(work, "kv: unsharded owners/mixed evidence is a SKIP, failing with --strict", K + [
            "run", "--url", base.url, "--baseline", rec, "--checks", "owners", "--strict"] + S, expect_ok=False)
        run(work, "kv: fault admit-over-window caught", K + ["run", "--url", over.url, "--baseline", rec, "--checks",
                                                             "limits"], expect_ok=False)
        run(work, "kv: fault leak-hold caught (cancel-transfer, idle)", K + [
            "run", "--url", leak.url, "--baseline", rec, "--checks", "carry,cancel-transfer"] + S, expect_ok=False)
        run(work, "kv: fault twin-fetch caught (a twinned pass that fetches)", K + [
            "run", "--url", tfetch.url, "--baseline", rec, "--checks", "twins"] + S, expect_ok=False)
        run(work, "kv: kill-rank1: the request in flight ends, rank 0 goes", K + [
            "run", "--url", kill.url, "--baseline", rec, "--checks", "carry,kill-rank1", "--kill-cmd", kill_cmd(kill),
            "--kill-deadline", "10", "--strict"] + S)
        run(work, "kv: fault kill-hang caught (rank 0 never notices)", K + [
            "run", "--url", hang.url, "--baseline", rec, "--checks", "carry,kill-rank1", "--kill-cmd", kill_cmd(hang),
            "--kill-deadline", "3"] + S, expect_ok=False)
        run(work, "kv: kill-rank1 without --kill-cmd is a SKIP", K + [
            "run", "--url", cand.url, "--baseline", rec, "--checks", "kill-rank1"])
    finally:
        for f in (base, cand, bad, over, leak, tfetch, kill, hang):
            f.stop()


def t_prefix(work):
    j = lambda n: os.path.join(work, n)  # noqa: E731
    P = tool("prefix_fanout.py")
    # --serial-prefill is the native admission (one prompt pass at a time: a shared prefix is computed once); without
    # it the fake's prefills overlap and compute the prefix again, which --expect-one-fill must catch
    cases = [("prefix: record on a serial server, one fill at every level", ["--serial-prefill"],
              ["--out", j("fan-base.json"), "--system-tokens", "30000", "--expect-one-fill",
               "--receipt", j("r-fan-serial.json")], True),
             ("prefix: baseline replies again on a fresh serial server", ["--serial-prefill"],
              ["--baseline", j("fan-base.json"), "--expect-one-fill", "--strict-cold"], True),
             ("prefix: measure-only on a server whose prefills overlap", [],
              ["--baseline", j("fan-base.json"), "--receipt", j("r-fan-overlap.json")], True),
             ("prefix: duplicate prefill caught (--expect-one-fill, prefills overlap)", [],
              ["--baseline", j("fan-base.json"), "--expect-one-fill"], False),
             ("prefix: a warm server is not judged cold (--strict-cold)", ["--serial-prefill"], None, False)]
    for label, fake_args, args, ok in cases:
        f = Fake(work, "prefix", "--mem-points", "32", *fake_args, *FAST)
        try:
            if args is None:     # the same prompts twice on one server: the second pass is warm
                warm = ["--baseline", j("fan-base.json"), "--strict-cold"]
                run(work, label + " (first pass)", P + ["--url", f.url] + warm)
                args = warm
            run(work, label, P + ["--url", f.url] + args, expect_ok=ok)
        finally:
            f.stop()


STUBS = {
    # sudo -n CMD...: run it (the stubs below log what matters)
    "sudo": 'while [[ "${1:-}" == -* ]]; do shift; done; exec "$@"',
    # systemctl is-active / start / restart / stop, against $STUB/active; start and restart bring the daily up
    "systemctl": 'echo "systemctl $*" >>"$STUB/calls"; case "$1" in\n'
                 '  is-active) [[ -f "$STUB/active" ]] ;;\n'
                 '  start|restart) touch "$STUB/active" "$STUB/health"; echo daily >"$STUB/container" ;;\n'
                 '  stop) rm -f "$STUB/active" "$STUB/health" "$STUB/container" ;; esac',
    # docker ps -a / inspect (the tf.recipe.port label: $STUB/label, empty for the daily's container)
    "docker": 'echo "docker $*" >>"$STUB/calls"; case "$1" in\n'
              '  ps) [[ -f "$STUB/container" ]] && echo dsv41-1017-exl3-tf; true ;;\n'
              '  inspect) [[ -f "$STUB/container" ]] || exit 1; [[ "$*" == *tf.recipe.port* ]] && { cat "$STUB/label" 2>/dev/null || echo "<no value>"; }; [[ "$*" == *State.Pid* ]] && echo 4242; true ;; esac',
    # ssh [opts] HOST CMD: run CMD here (the worker shares the stub state)
    "ssh": 'while [[ "${1:-}" == -* ]]; do [[ "$1" == -o ]] && shift; shift; done; shift; exec bash -c "$*"',
    "curl": '[[ -f "$STUB/health" ]] && echo \'{"status": "ok"}\' || exit 7',
    "hostname": 'echo gx10-a',
    # kill: logged, never sent (the stubbed PIDs are fictitious)
    "kill": 'echo "kill $*" >>"$STUB/calls"',
}


def stub_env(work, name, state):
    d = os.path.join(work, f"stub-{name}")
    os.makedirs(os.path.join(d, "bin"), exist_ok=True)
    for k, body in STUBS.items():
        f = os.path.join(d, "bin", k)
        with open(f, "w") as fh:
            fh.write("#!/usr/bin/env bash\n" + body + "\n")
        os.chmod(f, 0o755)
    port = os.path.join(d, "port")
    os.makedirs(port, exist_ok=True)
    with open(os.path.join(port, "stop.sh"), "w") as fh:     # the port's stop.sh: removes the container
        fh.write('#!/usr/bin/env bash\necho "port stop.sh" >>"$STUB/calls"; rm -f "$STUB/container" "$STUB/health" "$STUB/label"\n')
    os.chmod(os.path.join(port, "stop.sh"), 0o755)
    for k, v in state.items():
        with open(os.path.join(d, k), "w") as fh:
            fh.write(v)
    hold = os.path.join(d, "hold")
    open(hold, "w").close()
    return d, hold, {"PATH": os.path.join(d, "bin") + ":" + os.environ["PATH"], "STUB": d, "PORT_DIR": port,
                     "DAILY_DIR": os.path.join(d, "daily"), "SESS": os.path.join(d, "sess"), "HOLD": hold,
                     "WORKER": "gx10-b.invalid", "GPU_VALIDATE_HOST_OK": "1", "VERIFY_RESTORE": "0"}


def t_runbook(work):
    sh = os.path.join(HERE, "gpu_validate.sh")
    run(work, "runbook: bash -n", ["bash", "-n", sh])
    env = {"PORT_DIR": os.path.join(work, "port-checkout"), "DAILY_DIR": os.path.join(work, "daily"),
           "SESS": os.path.join(work, "sess"), "WORKER": "gx10-b.invalid", "GPU_VALIDATE_HOST_OK": "1"}
    run(work, "runbook: --dry-run all phases", ["bash", sh, "--dry-run"], env=env)
    run(work, "runbook: --dry-run selected phases", ["bash", sh, "--dry-run", "--phases", "hold,stop,restore"], env=env)
    run(work, "runbook: unknown phase refused", ["bash", sh, "--dry-run", "--phases", "nope"], expect_ok=False, env=env)
    # the refusal patterns: the engine's texts (port/integration 98370ef) match; another failure that says "features"
    # does not
    import re
    pat = dict(re.findall(r'^(REFUSED_[A-Z]+)="([^"]*)"$', open(sh).read(), re.M))
    feat, shard = re.compile(pat.get("REFUSED_FEATURES", "^$")), re.compile(pat.get("REFUSED_SHARD", "^$"))
    hits = [bool(feat.search("error: rank 1's configuration differs from rank 0's in features")),
            bool(shard.search("dsv41: error.KvShardMismatch: the ranks disagree on TF_DS_KV_SHARD, the pool")),
            bool(shard.search("TF_DS_KV_SHARD=1: the ranks cut the pool into different parts (another TF_DS_POOL")),
            not feat.search("NoEngineId: the spill tier needs TF_DS_ENGINE_ID (features: spill)"),
            not feat.search("rank 1's configuration differs from rank 0's in engine_commit"),
            not shard.search("out of memory while opening the features")]
    results.append(("runbook: the refusal patterns match the engine's texts only", len(pat) == 2 and all(hits)))

    def said(label, args, env, want, unwanted=()):
        p = subprocess.run(["bash", sh] + args, capture_output=True, text=True, env=dict(os.environ, **env), timeout=120)
        out = p.stdout + p.stderr
        calls = open(os.path.join(env["STUB"], "calls")).read() if "STUB" in env and \
            os.path.exists(os.path.join(env["STUB"], "calls")) else ""
        hay = out + calls
        plain = hay.replace("\\", "")      # printf %q's escapes differ between bash versions
        bad = [w for w in want if w not in hay and w not in plain] + [u for u in unwanted if u in hay or u in plain]
        results.append((label, not bad))
        print(f"{'PASS' if not bad else 'FAIL'}  {label}{': ' + str(bad) if bad else ''}", flush=True)
        if bad:
            print("      " + "      ".join((out + calls).splitlines(True)[-20:]), flush=True)
        return out, calls
    said("runbook: an early exit before the stop removes only the hold (dry run)",
         ["--dry-run", "--phases", "hold,baseline"], env, ["only the hold goes"], ["systemctl start"])
    said("runbook: after the stop, restore starts the service (dry run)",
         ["--dry-run", "--phases", "hold,stop"], env, ["systemctl start deepseek1017.service"])
    # restore against stubbed systemctl/docker/ssh/curl: what it does to a daily service in each state
    d, hold, e = stub_env(work, "healthy", {"active": "", "health": "", "container": "daily"})
    said("runbook: restore leaves a healthy daily service alone (no port stop, no start/restart)", ["restore"], e,
         ["is active and healthy"], ["port stop.sh", "systemctl start", "systemctl restart"])
    results.append(("runbook: ... and removes the hold", not os.path.exists(hold)))
    d, hold, e = stub_env(work, "down", {"active": ""})
    said("runbook: restore restarts an active oneshot unit whose containers are gone", ["restore"], e,
         ["systemctl restart deepseek1017.service"], ["port stop.sh", "systemctl start"])
    d, hold, e = stub_env(work, "port", {"container": "port", "label": "off", "health": ""})
    said("runbook: restore stops a PORT container (labelled) before starting the service", ["restore"], e,
         ["port stop.sh", "systemctl start deepseek1017.service"])
    d, hold, e = stub_env(work, "daily-not-port", {"active": "", "container": "daily", "health": ""})
    said("runbook: restore never runs the port stop.sh on the daily's own container", ["restore"], e, [],
         ["port stop.sh"])
    d, hold, e = stub_env(work, "early", {"active": "", "container": "daily"})
    said("runbook: a die in baseline (daily not healthy) only removes the hold", ["--phases", "hold,baseline"], e,
         ["only the hold goes"], ["systemctl start", "systemctl restart", "port stop.sh"])
    results.append(("runbook: ... the hold is gone", not os.path.exists(hold)))
    d, hold, e = stub_env(work, "resolve", {"container": "port", "label": "off"})
    pids = os.path.join(d, "pids")
    said("runbook: kill-ranks --resolve writes both PIDs (and kills nothing)", ["kill-ranks", "--resolve", pids], e,
         ["rank 0 PID 4242, rank 1 PID 4242"], ["kill -9"])
    results.append(("runbook: ... the PID file", os.path.exists(pids) and open(pids).read().split() == ["4242", "4242"]))
    d, hold, e = stub_env(work, "kill1", {"container": "port", "label": "off"})
    out, calls = said("runbook: kill-ranks --rank 1 kills rank 1 alone", ["kill-ranks", "--rank", "1"], e,
                      ["killed rank 1 PID 4242"], ["rank 0 PID"])
    results.append(("runbook: ... one kill sent", calls.count("kill -9 4242") == 1))
    d, hold, e = stub_env(work, "stop1", {"container": "port", "label": "off"})
    out, calls = said("runbook: kill-ranks --rank 1 --signal STOP stops rank 1 alone", ["kill-ranks", "--rank", "1",
                      "--signal", "STOP"], e, ["stopped rank 1 PID 4242"], ["rank 0 PID", "kill -9"])
    results.append(("runbook: ... one SIGSTOP sent", calls.count("kill -STOP 4242") == 1))
    said("runbook: --dry-run of every configuration (refusals, kill, bench)", ["--dry-run", "--configs", ",".join(
        ["off", "spill", "spill-mismatch", "shard", "shard-p2", "shard-p4", "shard-nodraft", "shard-copies",
         "shard-pressure", "shard-check", "shard-mismatch", "shard-pool-mismatch", "shard-kill", "shard-perf",
         "shard-perf-fetch", "spill-shard", "spill-shard-pressure", "spill-pictures", "spill-pictures-mismatch",
         "pair-kill", "pair-stop"])], env,
         ["expecting a refusal: spill-mismatch", "RANK1_EXTRA_ENV=TF_DS_KV_SHARD_CARRY=0",
          "expecting a refusal: spill-pictures-mismatch", "RANK1_EXTRA_ENV=TF_DS_IMAGE_CACHE=0",
          "--checks pictures,settled", "--phase prerestart", "--pictures", "spill-shard-main.json",
          "pair_check.py", "kill-ranks --rank 1 --signal 9", "kill-ranks --rank 1 --signal STOP", "--exit-code 75",
          "start.sh restart", "saved_log_has", "pair-stop-after-exact.json",
          "expecting a refusal: shard-pool-mismatch", "RANK1_EXTRA_ENV=TF_DS_WINDOW=1044480", "kill-ranks --rank 1",
          "--checks limits,carry,twins,move,cancel-transfer,idle", "bench.py", "KV_SHARD_CHECK=1",
          "off-fanout.json --strict-cold --expect-one-fill", "suite off-startup",
          "a broken rank pair ends this server with status 75", "start the port server: pair-kill (SPILL_GIB=0 KV_SHARD=0 "])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="compile,kinds,counters,exact,spill,kv,prefix,pair,runbook")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="tf-tools-selftest-")
    print(f"selftest: work folder {work}", flush=True)
    only = a.only.split(",")
    if "compile" in only:
        for f in sorted(os.listdir(HERE)):
            if f.endswith(".py"):
                try:
                    py_compile.compile(os.path.join(HERE, f), cfile=os.path.join(work, f + "c"), doraise=True)
                    results.append((f"py_compile {f}", True))
                except py_compile.PyCompileError as e:
                    results.append((f"py_compile {f}", False))
                    print(f"FAIL  py_compile {f}: {e}")
        print(f"PASS  py_compile of {sum(1 for n, _ in results if n.startswith('py_compile'))} tools"
              if all(ok for _, ok in results) else "FAIL  py_compile", flush=True)
    for name, fn in (("kinds", t_kinds), ("counters", t_counters), ("exact", t_exact), ("spill", t_spill), ("kv", t_kv), ("prefix", t_prefix), ("pair", t_pair), ("runbook", t_runbook)):
        if name in only:
            try:
                fn(work)
            except Exception as e:  # noqa: BLE001
                results.append((name, False))
                print(f"FAIL  {name}: {type(e).__name__}: {e}", flush=True)
    bad = [n for n, ok in results if not ok]
    print(f"\nselftest: {'PASS' if not bad else 'FAIL'} ({len(results)} expectations, {len(bad)} unmet){': ' + ', '.join(bad) if bad else ''}")
    if not bad and not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"selftest: logs kept in {work}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
