"""Tests for the worker list (scripts/nodes.sh's worker_ids, WORKER2): the scripts reach their workers as a list, so
that three Sparks (TP=3) can follow, and at TP=2 they do exactly what they did with the one worker.

  TwoSparksUnchanged  start.sh (DRY_RUN=1, and a real start, restart and failed load with fakes) and stop.sh of this
                      tree against the scripts before the worker list (TPN_BASE, default 31457d0): the same output, the
                      same ssh commands (the functions sent over ssh included) and the same docker calls.
  CheckWorkers        TP=3 is refused with its own message, any other TP as before; WORKER's message is unchanged;
                      a WORKER2 set at TP=2 is left alone; with the TP check taken out, every worker must be set and
                      distinct.
  ThreeRanks          with check_workers and the links stubbed (TP=3 itself comes later): start.sh launches rank i on
                      worker i with `<i> 3` and its own link, address and devices, waits for and stops every rank, and
                      refuses on the worker that is short of memory.

Offline, as scripts/test_scripts.py (whose fakes these use): no Spark, GPU, Docker or network. The comparison with
TPN_BASE needs git and that commit in this checkout (else it is skipped). The scripts need bash 4 or later: BASH4=<path>.
Run: python3 scripts/test_nodes_tpn.py
"""
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_scripts as ts  # noqa: E402  (its fakes and sandbox; its own tests are not collected here)

BASE = os.environ.get("TPN_BASE", "31457d0")
NAME = ts.NAME

# the link of two Sparks, fixed (as scripts/port_test.sh's golden test has it): the routes and sysfs of a pair are not
# what is compared
LINK_TP2 = """
detect_link() {
  HEAD_ADDR=10.9.0.1; HEAD_DEV=enp1s0f0np0; HEAD_HCA=rocep1s0f0; HEAD_GID=3
  WORKER_ADDR=10.9.0.2; WORKER_DEV=enp1s0f0np0; WORKER_HCA=rocep1s0f0; WORKER_GID=3
  HEAD_HCAS=rocep1s0f0,roceP2p1s0f0; WORKER_HCAS=rocep1s0f0,roceP2p1s0f0
  NODE_DEV=("$HEAD_DEV" "$WORKER_DEV"); NODE_HCAS=("$HEAD_HCAS" "$WORKER_HCAS"); NODE_GID=("$HEAD_GID" "$WORKER_GID")
  LINK_HEAD_ADDR[1]=$HEAD_ADDR; LINK_WORKER_ADDR[1]=$WORKER_ADDR
  MASTER_ADDR=${MASTER_ADDR:-$HEAD_ADDR}
}
"""
# three ranks: each worker on its own link to the head (a triangle's two cables from the head)
LINK_TP3 = """
check_workers() { :; }
detect_links() {
  MASTER_ADDR=${MASTER_ADDR:-10.9.1.1}
  NODE_DEV=(dev0 dev1 dev2); NODE_HCAS=(hca0a,hca0b hca1 hca2); NODE_GID=(3 4 5)
  LINK_HEAD_ADDR=(- 10.9.1.1 10.9.2.1); LINK_WORKER_ADDR=(- 10.9.1.2 10.9.2.2)
}
"""
# each Spark's memory reading (TPN_HEAD_GB here, TPN_WORKER_GB on a worker, TPN_W2_GB on w2); it reaches the workers
# through await_memory's declare -f
MEM = """
mem_available_gib() {
  case "${SIM_HOST:-head}" in head) echo "${TPN_HEAD_GB:-119.0}" ;; w2) echo "${TPN_W2_GB:-${TPN_WORKER_GB:-119.0}}" ;; *) echo "${TPN_WORKER_GB:-119.0}" ;; esac
}
"""
FAKE_CURL = """#!/bin/sh
[ -e "$SIM/curl_fail" ] && exit 7
case "$*" in *completions*) echo '{"choices": [{"message": {"content": "OK"}}], "usage": {"completion_tokens": 1}}' ;;
  *) echo '{"requests_running": 0}' ;; esac
"""
FAKES = {**ts.StartRestart.FAKES, "curl": FAKE_CURL}


class Base(ts.ScriptTest):
    def setUp(self):
        super().setUp()
        bin_ = self.tmp / "bin"
        for name, body in FAKES.items():
            (bin_ / name).write_text(body)
            (bin_ / name).chmod(0o755)
        self.env["MIN_AVAIL_GIB"] = "0"

    def patch(self, tree, nodes_extra):
        nodes = tree / "scripts" / "nodes.sh"
        nodes.write_text(nodes.read_text() + nodes_extra)
        (tree / "scripts" / "zig_flags.py").write_text("import sys\nsys.stdin.read()\n")
        (tree / "scripts" / "prepare.sh").write_text("#!/usr/bin/env bash\necho prepared >> \"$SIM/events\"\n")

    def fresh_sim(self):
        shutil.rmtree(self.sim, ignore_errors=True)
        (self.sim / "hosts").mkdir(parents=True)

    def up(self, *hosts):
        (self.sim / "image").write_text("")
        for h in hosts:
            self.containers(h, NAME)

    def run_in(self, tree, args, extra_env=None, timeout=60):
        env = {**self.env, **(extra_env or {})}
        try:
            r = subprocess.run([ts.BASH, *args], cwd=tree, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            out = lambda b: b.decode(errors="replace") if isinstance(b, bytes) else (b or "")
            return 124, f"timed out after {timeout} s\n" + out(e.stdout) + out(e.stderr)
        return r.returncode, r.stdout + r.stderr

    def launched(self):
        log = (self.sim / "docker.log").read_text() if (self.sim / "docker.log").exists() else ""
        return [l.split()[0] for l in log.splitlines() if " docker run -d " in l]


class TwoSparksUnchanged(Base):
    def setUp(self):
        super().setUp()
        if shutil.which("git") is None or subprocess.run(["git", "-C", str(ts.ROOT), "cat-file", "-e", BASE + "^{commit}"],
                                                         capture_output=True).returncode != 0:
            self.skipTest(f"no commit {BASE} in {ts.ROOT} (TPN_BASE=<commit>)")
        self.base = self.tmp / "base"
        self.base.mkdir()
        arc = subprocess.run(["git", "-C", str(ts.ROOT), "archive", BASE, *ts.COPY], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(self.base)], input=arc, check=True)
        for tree in (self.base, self.repo):
            self.patch(tree, LINK_TP2 + MEM)

    def observe(self, tree, setup, args, extra_env):
        """One run in a fresh sandbox: (exit code and output, the ssh commands, the docker calls), paths and times out."""
        self.fresh_sim()
        setup()
        code, out = self.run_in(tree, args, extra_env)
        def norm(text):
            text = text.replace(str(tree), "<repo>").replace(str(self.tmp), "<tmp>")
            text = re.sub(r"\d{8}-\d{6}", "<stamp>", text)                  # saved logs' names
            text = re.sub(r"answered after \d+s", "answered after <n>s", text)
            text = re.sub(r"\(\d+ s\)", "(<n> s)", text)
            # bash's own notice when the load's log follower is killed: its line number and pids are the script's
            return "\n".join(l for l in text.splitlines() if not re.search(r"Broken pipe|Terminated", l))
        read = lambda f: (self.sim / f).read_text() if (self.sim / f).exists() else ""
        # docker calls sorted: stop.sh stops the ranks at once
        return f"exit {code}\n{norm(out)}", norm(read("ssh.log")), "\n".join(sorted(norm(read("docker.log")).splitlines()))

    def assertSame(self, setup, args, extra_env=None):
        before = self.observe(self.base, setup, args, extra_env)
        after = self.observe(self.repo, setup, args, extra_env)
        for what, b, a in zip(["output", "ssh commands", "docker calls"], before, after):
            self.assertEqual(b, a, f"{what} differ from {BASE}'s")
        return after

    def test_dry_run(self):
        cases = {
            "defaults": {},
            "RDMA=0 API_KEY=1 RANK1_PROFILE=2 POOL": {"RDMA": "0", "API_KEY": "1", "RANK1_PROFILE": "2", "POOL": "2097152"},
            "KV_SHARD=1": {"KV_SHARD": "1"},
            "NCCL_* and TF_DS_* exports": {"NCCL_DEBUG": "INFO", "NCCL_MAX_NCHANNELS": "4", "TF_DS_CONNECT_S": "900"},
            "WORKER2 set at TP=2": {"WORKER2": "w2"},
            "no marker: prepare.sh would run": {"PREPARE": "auto"},
        }
        for name, extra in cases.items():
            with self.subTest(case=name):
                out, _, _ = self.assertSame(lambda: None, ["start.sh"], {"DRY_RUN": "1", "PREPARE": "0", **extra})
                self.assertIn("[dry-run] rank 1 on w1:", out)
                self.assertIn(" /rank-cache 1 2 10.9.0.1 ", out)
                self.assertIn("2 x DGX Spark", out)

    def test_dry_run_worker_out_of_reach(self):
        out, _, _ = self.assertSame(lambda: self.unreachable("w1"), ["start.sh"], {"DRY_RUN": "1", "PREPARE": "0"})
        self.assertIn("cannot ssh to w1", out)

    def test_start_restart_and_a_failed_load(self):
        cases = [
            ("start", lambda: self.up(), ["start.sh"], {"PREPARE": "0"}),
            ("start, memory checked on both", lambda: self.up(), ["start.sh"], {"PREPARE": "0", "MIN_AVAIL_GIB": "112", "MIN_AVAIL_WAIT_S": "0"}),
            ("start, the worker short of memory", lambda: self.up(), ["start.sh"],
             {"PREPARE": "0", "MIN_AVAIL_GIB": "112", "MIN_AVAIL_WAIT_S": "0", "TPN_WORKER_GB": "100.0"}),
            ("already running", lambda: self.up("head", "w1"), ["start.sh"], {"PREPARE": "0"}),
            ("restart", lambda: self.up("head", "w1"), ["start.sh", "restart"], {"PREPARE": "0"}),
            ("restart, API_KEY=1", lambda: self.up("head", "w1"), ["start.sh", "restart"], {"PREPARE": "0", "API_KEY": "1"}),
            ("the load fails", lambda: (self.up(), (self.sim / "curl_fail").write_text("")), ["start.sh"],
             {"PREPARE": "0", "WAIT_TIMEOUT": "0"}),
            ("FOREGROUND=1", lambda: self.up(), ["start.sh"], {"PREPARE": "0", "FOREGROUND": "1"}),
        ]
        for name, setup, args, extra in cases:
            with self.subTest(case=name):
                self.assertSame(setup, args, extra)

    def test_stop(self):
        for name, setup, extra in [("both up", lambda: self.up("head", "w1"), {}),
                                   ("DRY_RUN", lambda: self.up("head", "w1"), {"DRY_RUN": "1"}),
                                   ("nothing up", lambda: None, {}),
                                   ("the worker out of reach", lambda: (self.up("head", "w1"), self.unreachable("w1")), {})]:
            with self.subTest(case=name):
                self.assertSame(setup, ["stop.sh"], extra)


class CheckWorkers(ts.ScriptTest):
    def test_tp3_is_refused_with_its_own_message(self):
        code, out = self.bash("check_workers", {"TP": "3", "WORKER2": "w2"})
        self.assertEqual(code, 1, out)
        self.assertIn("TP=3 (three Sparks) is not served yet", out)
        self.assertNoTraceback(out)

    def test_other_tp_is_refused_as_before(self):
        for tp in ["1", "4", "two"]:
            with self.subTest(tp=tp):
                code, out = self.bash("check_workers", {"TP": tp})
                self.assertEqual(code, 1, out)
                self.assertIn(f"TP is 2 (two Sparks): the dsv41 family runs tensor-parallel over two nodes here, not {tp}", out)

    def test_worker_unset_message_unchanged(self):
        code, out = self.bash("check_workers", {"WORKER": ""})
        self.assertEqual(code, 1, out)
        self.assertIn("WORKER is not set: put WORKER=user@<address of the second Spark> in scripts/local.sh "
                      "(see scripts/local.sh.example)", out)

    def test_worker_ids(self):
        for extra, want in [({}, "1"), ({"WORKER2": "w2"}, "1"), ({"TP": "3", "WORKER2": "w2"}, "1 2")]:
            with self.subTest(extra=extra):
                code, out = self.bash('echo $(worker_ids)', extra)
                self.assertEqual((code, out.strip()), (0, want), out)

    def test_worker2_at_tp2_is_left_alone(self):
        code, out = self.bash("check_workers; need_workers; echo ok", {"WORKER2": "w2"})
        self.assertEqual(code, 0, out)
        log = (self.sim / "ssh.log").read_text()
        self.assertNotIn("w2", log)

    def test_every_worker_set_and_distinct(self):
        # the loop alone: the TP refusals taken out of a copy of check_workers (TP=3 itself comes later)
        loop = ('eval "$(declare -f check_workers | sed -e \'/TP=3 (three Sparks)/d\' '
                '-e \'s/\\[\\[ "\\$TP" == 2 \\]\\]/[[ "$TP" == [23] ]]/\')"\nTP=3\ncheck_workers\necho ok')
        for extra, want in [({"WORKER2": "w2"}, "ok"),
                            ({"WORKER2": ""}, "WORKER2 is not set: put WORKER2=user@<address of rank 2's Spark>"),
                            ({"WORKER2": "w1"}, "WORKER2 and WORKER are the same node (w1)")]:
            with self.subTest(extra=extra):
                code, out = self.bash(loop, extra)
                self.assertIn(want, out)
                self.assertEqual(code, 0 if want == "ok" else 1, out)


class ThreeRanks(Base):
    def setUp(self):
        super().setUp()
        self.patch(self.repo, LINK_TP3 + MEM)
        self.env.update({"TP": "3", "WORKER2": "w2"})

    def test_dry_run_launches_each_rank_on_its_worker(self):
        code, out = self.run_in(self.repo, ["start.sh"], {"DRY_RUN": "1", "PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertNoTraceback(out)
        self.assertIn("3 x DGX Spark", out)
        self.assertIn("Launch: container %s, ranks 1 to 2 on the workers, then rank 0 here" % NAME, out)
        cmds = {}
        lines = out.splitlines()
        for k, l in enumerate(lines):
            m = re.match(r"\[dry-run\] rank (\d) ", l)
            if m:
                cmds[int(m.group(1))] = lines[k + 1]
        self.assertEqual(sorted(cmds), [0, 1, 2], out)
        for i, (addr, hca, gid) in {1: ("10.9.1.1", "hca1", "4"), 2: ("10.9.2.1", "hca2", "5")}.items():
            self.assertIn(f" /rank-cache {i} 3 {addr} 29620 ", cmds[i])
            self.assertIn(f" --rdma {hca}", cmds[i])
            self.assertIn(f"-e TF_RDMA_GID_INDEX={gid} ", cmds[i])
            self.assertIn(f"-e NCCL_IB_HCA={hca} ", cmds[i])
        self.assertIn(" -e TF_TP_WORLD=3 ", cmds[0])
        self.assertIn(" -e TF_RDMA_DEVICES=hca0a,hca0b ", cmds[0])
        self.assertIn("[dry-run] rank 2 on w2:", out)
        self.assertIn("rank 2 (w2): RoCE hca2 (GID 5), NFS from 10.9.2.1", out)

    def test_rank1_extra_env_stays_rank_1s(self):
        code, out = self.run_in(self.repo, ["start.sh"], {"DRY_RUN": "1", "PREPARE": "0", "RANK1_EXTRA_ENV": "TF_DS_X=1"})
        self.assertEqual(code, 0, out)
        r1 = out.split("[dry-run] rank 1 on w1:")[1].splitlines()[1]
        r2 = out.split("[dry-run] rank 2 on w2:")[1].splitlines()[1]
        self.assertIn("TF_DS_X=1", r1)
        self.assertNotIn("TF_DS_X=1", r2)

    def test_restart_stops_and_starts_every_rank(self):
        self.up("head", "w1", "w2")
        code, out = self.run_in(self.repo, ["start.sh", "restart"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertNoTraceback(out)
        self.assertEqual(sorted(self.launched()), ["head", "w1", "w2"], out)
        self.assertIn("Stopped and removed rank 2 on w2", out)
        self.assertIn("(rank 1: ssh w1 docker logs -f %s)   (rank 2: ssh w2 docker logs -f %s)" % (NAME, NAME), out)
        self.assertIn("is now LIVE", out)

    def test_a_worker_short_of_memory_is_named(self):
        self.up()
        code, out = self.run_in(self.repo, ["start.sh"],
                                {"PREPARE": "0", "MIN_AVAIL_GIB": "112", "MIN_AVAIL_WAIT_S": "0", "TPN_W2_GB": "100.0"})
        self.assertEqual(code, 1, out)
        self.assertIn("only 100.0 GiB memory available on w2", out)
        self.assertEqual(self.launched(), [], out)

    def test_a_rank_that_exits_during_the_load_is_named(self):
        self.up()
        (self.sim / "curl_fail").write_text("")
        bin_ = self.tmp / "bin"
        # w2's container is gone as soon as it was started: rank 2 exited
        old = 'echo "$n" >> "$d/containers"; echo container-id ;;'
        docker = (bin_ / "docker").read_text()
        self.assertIn(old, docker)
        (bin_ / "docker").write_text(docker.replace(old, '[[ "$h" == w2 ]] || ' + old))
        # (the fake docker names a container by the last --name it is given: rank 0's serve flags have one too)
        code, out = self.run_in(self.repo, ["start.sh"], {"PREPARE": "0", "SERVED_NAME": NAME})
        self.assertEqual(code, 1, out)
        self.assertIn("rank 2 on w2 exited before the server was ready", out)
        self.assertIn("rank 1 (w1): last log lines", out)
        self.assertIn("rank 2 (w2): last log lines", out)


if __name__ == "__main__":
    unittest.main()
