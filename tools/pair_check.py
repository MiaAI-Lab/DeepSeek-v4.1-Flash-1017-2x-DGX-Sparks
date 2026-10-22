#!/usr/bin/env python3
"""The broken-pair policy end to end (tensorfold docs/port/robustness.md, GPU validation 2-3): with two ranks, rank 0's
engine ends with status 75 when the rank pair breaks (TF_DS_PAIR_BROKEN unset; `degrade` opts out) instead of hanging.
Restarting the pair is not its job (a supervisor's, docs/systemd.md, or the caller's). Against a running server:

    python3 -I tools/pair_check.py --url http://127.0.0.1:8888 --streams 8 --kill-cmd "K" --end-deadline 60 \\
        --exit-cmd "docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' dsv41-1017-exl3-tf" \\
        --expect-log "docker logs dsv41-1017-exl3-tf::exiting with status 75" --receipt r.json

Checks (PASS / FAIL / SKIP; exit 1 on a FAIL):
  decoding       every stream had its first token before the kill and none had ended (the pair was busy: the stall
                 monitor counts only then)
  kill           --kill-cmd exits 0 (kill -9, or SIGSTOP, of rank 1's engine by PID)
  streams end    every stream ends (an error, a cut, or a normal finish inside the drain) within --end-deadline of
                 the kill: nothing hangs
  down           /health stops answering ok (gone, or 503 for a stalled or broken pair) within --end-deadline
  exited         --exit-cmd (polled) prints `exited <--exit-code>` (75) within --end-deadline: rank 0's container
                 ended with the policy's status, not another code (a memguard kill is 137); SKIP without --exit-cmd
  logs           each --expect-log CMD::PATTERN: CMD's output (after the exit) matches the regex PATTERN
Times (seconds after the kill) are in the receipt: streams_ended, down, exited. Stdlib only.
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


def state(cmd):
    """--exit-cmd's output as (status, code): ("exited", 75), ("running", 0), or (None, the text) when unreadable."""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    words = p.stdout.split()
    if p.returncode == 0 and len(words) == 2 and words[1].lstrip("-").isdigit():
        return words[0], int(words[1])
    return None, (p.stdout + p.stderr).strip()[-200:]


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
    ap.add_argument("--end-deadline", type=float, default=60)
    ap.add_argument("--exit-cmd", help="prints rank 0's container as '<status> <exit code>' (docker inspect -f "
                                       "'{{.State.Status}} {{.State.ExitCode}}' NAME)")
    ap.add_argument("--exit-code", type=int, default=75, help="the exit code --exit-cmd must show (the policy's 75)")
    ap.add_argument("--expect-log", action="append", default=[], help="CMD::PATTERN, repeatable")
    ap.add_argument("--receipt")
    a = C.use_api_key_args(C.add_api_key_arg(ap).parse_args())
    srv = C.Server(a.url, a.model)
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

    # the streams' ends, /health going down and rank 0's exit are watched together until the end deadline
    down = exited = None
    said, last_state = None, None
    end = t_kill + a.end_deadline
    while time.monotonic() < end:
        now = time.monotonic()
        if down is None:
            h = srv.health()
            if not (isinstance(h, dict) and h.get("status") == "ok"):
                down, said = now - t_kill, h
        if exited is None and a.exit_cmd:
            last_state = state(a.exit_cmd)
            if last_state[0] not in (None, "running", "restarting"):
                exited = time.monotonic() - t_kill
        if all(e is not None for e in ended) and down is not None and (exited is not None or not a.exit_cmd):
            break
        time.sleep(0.5)
    hung = [i for i in range(a.streams) if ended[i] is None]
    last = max((e for e in ended if e is not None), default=None)
    times["streams_ended"] = round(last - t_kill, 2) if last and not hung else None
    times["down"] = round(down, 2) if down is not None else None
    times["exited"] = round(exited, 2) if exited is not None else None
    finished = [i for i in range(a.streams) if rows[i] and "error" not in rows[i] and rows[i].get("finish_reason")]
    rc.check("streams end", not hung and (last is None or last - t_kill <= a.end_deadline),
             f"streams {hung} still open {a.end_deadline}s after the kill: rank 0 hangs" if hung else
             f"all {a.streams} ended {times['streams_ended']}s after the kill ({len(finished)} finished normally inside "
             "the drain, the rest failed)", rows=[r for r in rows if r])
    rc.check("down", down is not None,
             f"/health stopped answering ok {times['down']}s after the kill: {str(said)[:200]}" if down is not None else
             f"/health still answered ok {a.end_deadline}s after the kill: rank 0 did not exit")
    if a.exit_cmd:
        st, code = last_state if last_state else (None, "not read")
        rc.check("exited", exited is not None and code == a.exit_code,
                 f"rank 0's container {st} {code}, {times['exited']}s after the kill" if exited is not None else
                 f"rank 0's container still {st or 'unreadable'} ({code}) {a.end_deadline}s after the kill",
                 state=st, code=code)
    else:
        rc.add("exited", "SKIP", "no --exit-cmd")
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
