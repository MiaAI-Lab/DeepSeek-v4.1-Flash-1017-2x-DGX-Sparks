#!/usr/bin/env python3
"""The broken-pair policy end to end (tensorfold docs/port/robustness.md, GPU validation 2-3): with the spill tier or
KV sharding on, rank 0's engine ends with status 75 when the rank pair breaks, and the recipe's pair-restart guard
(scripts/pairwatch.sh) restarts both ranks. Against a running server:

    python3 -I tools/pair_check.py --url http://127.0.0.1:8888 --streams 8 --kill-cmd "K" \\
        --began-cmd "grep -q 'exited with status 75' pairwatch.log" --end-deadline 60 --began-deadline 60 \\
        --expect-log "zcat \\$(ls -t logs/*-rank0.log.gz | head -1)::exiting with status 75" --receipt r.json

Checks (PASS / FAIL / SKIP; exit 1 on a FAIL):
  decoding       every stream had its first token before the kill and none had ended (the pair was busy: the stall
                 monitor counts only then)
  kill           --kill-cmd exits 0 (kill -9, or SIGSTOP, of rank 1's engine by PID)
  streams end    every stream ends (an error, a cut, or a normal finish inside the drain) within --end-deadline of
                 the kill: nothing hangs
  down           /health stops answering ok within --end-deadline (rank 0's process ended)
  restart began  --began-cmd succeeds (polled each second) within --began-deadline of the kill, and (with
                 --restart-within S) within S seconds of /health going down: the guard saw the 75 and acted
  recovered      /health answers ok again within --recover-timeout of the kill (the load included)
  logs           each --expect-log CMD::PATTERN: CMD's output (after the recovery) matches the regex PATTERN
Times (seconds after the kill) are in the receipt: streams_ended, down, began, recovered. Stdlib only.
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python3 -I does not add the script's folder
import checklib as C  # noqa: E402


def prompt(i, salt):
    return [{"role": "user", "content": f"[pair {salt}-{i}] Write a long, detailed story about a lighthouse keeper "
                                        "who finds a map. Do not stop early."}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=C.default_url())
    ap.add_argument("--model", default=os.environ.get("MODEL", C.DEFAULT_MODEL))
    ap.add_argument("--label", default="")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8000, help="long enough that no stream ends before the kill")
    ap.add_argument("--decode-s", type=float, default=10, help="seconds of decoding (after every first token) before "
                                                               "the kill")
    ap.add_argument("--kill-cmd", required=True)
    ap.add_argument("--began-cmd", help="a shell test that succeeds once the restart began (pairwatch's log line)")
    ap.add_argument("--end-deadline", type=float, default=60)
    ap.add_argument("--began-deadline", type=float, default=60)
    ap.add_argument("--restart-within", type=float, default=0, help="restart began at most this long after /health "
                                                                    "went down (0: not judged)")
    ap.add_argument("--recover-timeout", type=float, default=2400)
    ap.add_argument("--expect-log", action="append", default=[], help="CMD::PATTERN, repeatable")
    ap.add_argument("--receipt")
    a = ap.parse_args()
    srv = C.Server(a.url, a.model, a.end_deadline + a.recover_timeout)
    rc = C.Receipt("pair_check", {"url": srv.url, "label": a.label, "streams": a.streams})
    if not srv.wait_healthy(900):
        rc.add("health", "FAIL", f"{srv.url} not healthy before the kill: {srv.health()}")
        return rc.finish(a.receipt)
    salt = str(int(time.time()))
    rows, firsts, ended = [None] * a.streams, [threading.Event() for _ in range(a.streams)], [None] * a.streams

    def one(i):
        rows[i] = srv.stream(prompt(i, salt), a.max_tokens, first=firsts[i])
        ended[i] = time.monotonic()
    ths = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(a.streams)]
    for t in ths:
        t.start()
    for e in firsts:
        e.wait(600)
    time.sleep(a.decode_s)
    early = [i for i in range(a.streams) if ended[i] is not None]
    rc.check("decoding", not early, f"{a.streams} streams decoding at the kill" if not early else
             f"streams {early} had already ended before the kill: {[rows[i] for i in early][:2]}")
    t_kill = time.monotonic()
    k = subprocess.run(a.kill_cmd, shell=True, capture_output=True, text=True, timeout=120)
    rc.check("kill", k.returncode == 0, f"{a.kill_cmd!r} exit {k.returncode} {(k.stdout + k.stderr).strip()[-200:]}")
    times = {}

    # down, began and the streams' ends are watched together until the end deadline
    began = None
    down = None
    end = t_kill + max(a.end_deadline, a.began_deadline)
    while time.monotonic() < end:
        now = time.monotonic()
        if down is None and not srv.alive():
            down = now - t_kill
        if began is None and a.began_cmd and \
                subprocess.run(a.began_cmd, shell=True, capture_output=True, timeout=30).returncode == 0:
            began = time.monotonic() - t_kill
        if all(e is not None for e in ended) and down is not None and (began is not None or not a.began_cmd):
            break
        time.sleep(0.5)
    hung = [i for i in range(a.streams) if ended[i] is None]
    last = max((e for e in ended if e is not None), default=None)
    times["streams_ended"] = round(last - t_kill, 2) if last and not hung else None
    times["down"] = round(down, 2) if down is not None else None
    times["began"] = round(began, 2) if began is not None else None
    finished = [i for i in range(a.streams) if rows[i] and "error" not in rows[i] and rows[i].get("finish_reason")]
    rc.check("streams end", not hung and (last is None or last - t_kill <= a.end_deadline),
             f"streams {hung} still open {a.end_deadline}s after the kill: rank 0 hangs" if hung else
             f"all {a.streams} ended {times['streams_ended']}s after the kill ({len(finished)} finished normally inside "
             "the drain, the rest failed)", rows=[r for r in rows if r])
    rc.check("down", down is not None and down <= a.end_deadline,
             f"/health stopped answering {times['down']}s after the kill" if down is not None else
             f"/health still answered ok {a.end_deadline}s after the kill: rank 0 did not exit")
    if a.began_cmd:
        lag = round(began - down, 2) if began is not None and down is not None else None
        times["began_after_down"] = lag
        late = a.restart_within and (lag is None or lag > a.restart_within)
        rc.check("restart began", began is not None and began <= a.began_deadline and not late,
                 (f"{a.began_cmd!r} true {times['began']}s after the kill, {lag}s after /health went down"
                  + (f" (more than --restart-within {a.restart_within}s)" if late else "")) if began is not None else
                 f"{a.began_cmd!r} not true within {a.began_deadline}s of the kill")
    else:
        rc.add("restart began", "SKIP", "no --began-cmd")
    ok = srv.wait_healthy(max(1, a.recover_timeout - (time.monotonic() - t_kill)))
    times["recovered"] = round(time.monotonic() - t_kill, 1) if ok else None
    rc.check("recovered", ok, f"/health ok again {times['recovered']}s after the kill" if ok else
             f"/health not ok within {a.recover_timeout}s of the kill: {srv.health()}")
    for spec in a.expect_log:
        cmd, _, pat = spec.rpartition("::")
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        m = re.search(pat, p.stdout + p.stderr)
        rc.check(f"log /{pat}/", bool(m), m.group(0)[:200] if m else f"{cmd!r} (exit {p.returncode}) has no /{pat}/: "
                 f"{(p.stdout + p.stderr).strip()[-300:]}")
    rc.doc["times"] = times
    return rc.finish(a.receipt)


if __name__ == "__main__":
    sys.exit(main())
