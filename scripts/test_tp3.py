"""Tests for TP=3 (three Sparks, experimental; docs/tp3.md) and for TP=2 being left exactly as it was.

  Settings      scripts/config.sh at TP=3: LakyFx's TF29 profile as the defaults, the TP=3 image tag, and tp3_check
                refusing what only the two-Spark engine has (the port's spill and KV sharding, EAQ v1.1, the graph cap,
                the RDMA nap, the prefill interleave, the connect window, rank 0's own Engram folder) and a pool the
                RoPE tables or the engine cannot hold; at TP=2 none of it is read.
  Triangle      scripts/nodes.sh's pair_triangle and detect_triangle on three fake nodes cabled as a triangle (one
                subnet a PCIe function of a cable, documentation addresses): each rank's four rails in port order, its
                peer masks, NCCL devices (the first of each cable), GID, socket, the master, the address range, and the
                routes; a node with two rails, a missing cable or rails in another order are refused; a missing route
                is printed as the command that adds it.
  Launch        start.sh with TP=3 (DRY_RUN=1, and a real start, restart and failed load with fakes): the three rank
                commands (LakyFx's variables, TF_DS_EXPERIMENTAL_TP3, the peer masks, the four switchless NCCL settings,
                each worker's own folder, the master for both followers, none of zig/patches' settings), workers first,
                every rank stopped, the one short of memory or the one that exits named, and the smoke test through all
                three.
  TwoSparksAsBefore  start.sh (dry runs, starts, restarts, a failed load) and stop.sh at TP=2 against the tree before
                TP=3 (TP3_BASE, default 9912367: the worker list): the same output, ssh commands and docker calls
                (scripts/test_nodes_tpn.py's comparison, with this base).

Offline, as scripts/test_scripts.py (whose fakes these use): no Spark, GPU, Docker or network. The comparison needs git
and TP3_BASE in this checkout (else it is skipped). The scripts need bash 4 or later: BASH4=<path>.
Run: python3 scripts/test_tp3.py
"""
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_scripts as ts  # noqa: E402  (their fakes and sandbox; their own tests are not collected here)
import test_nodes_tpn as tpn  # noqa: E402

BASE = os.environ.get("TP3_BASE", "9912367")
NAME = ts.NAME

# Three nodes cabled as a ring: head f0 - w1 f1, w1 f0 - w2 f1, w2 f0 - head f1; each cable's two PCIe functions in
# a /29 of their own (six subnets of 192.0.2.0/26), one RoCE v2 GID index (3) everywhere.
INVENTORY = {
    "head": ["enp1s0f0np0 rocep1s0f0 192.0.2.1/29 3", "enP2p1s0f0np0 roceP2p1s0f0 192.0.2.9/29 3",
             "enp1s0f1np1 rocep1s0f1 192.0.2.34/29 3", "enP2p1s0f1np1 roceP2p1s0f1 192.0.2.42/29 3"],
    "w1": ["enp1s0f0np0 rocep1s0f0 192.0.2.17/29 3", "enP2p1s0f0np0 roceP2p1s0f0 192.0.2.25/29 3",
           "enp1s0f1np1 rocep1s0f1 192.0.2.2/29 3", "enP2p1s0f1np1 roceP2p1s0f1 192.0.2.10/29 3"],
    "w2": ["enp1s0f0np0 rocep1s0f0 192.0.2.33/29 3", "enP2p1s0f0np0 roceP2p1s0f0 192.0.2.41/29 3",
           "enp1s0f1np1 rocep1s0f1 192.0.2.18/29 3", "enP2p1s0f1np1 roceP2p1s0f1 192.0.2.26/29 3"],
}
RAILS = "rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1"


def inventory_lines(inv):
    """pair_triangle's input: each node's node_inventory lines, prefixed with its rank."""
    return "".join(f"{r} {line}\n" for r, h in enumerate(["head", "w1", "w2"]) for line in ["default eno1"] + inv[h])


# node_inventory as each fake Spark answers it (TP3_INV_<host>, ;-separated lines, overrides INVENTORY), and the
# routes: none missing unless TP3_MISS_HOST names the host and TP3_MISS the line (both reach the workers through the
# functions' declare -f and the environment the fake ssh keeps); the rank folders are ready (tp3_files_check)
def stubs(inv=INVENTORY):
    cases = "".join(f'    {h}) printf "%s\\n" "default eno1" {" ".join(repr(l) for l in lines)} ;;\n' for h, lines in inv.items())
    return f"""
node_inventory() {{
  local v="TP3_INV_${{SIM_HOST:-head}}"
  if [[ -n "${{!v:-}}" ]]; then tr ';' '\\n' <<<"${{!v}}"; return; fi
  case "${{SIM_HOST:-head}}" in
{cases}  esac
}}
tp3_route_missing() {{ [[ "${{SIM_HOST:-head}}" != "${{TP3_MISS_HOST:-none}}" ]] || echo "$TP3_MISS"; }}
tp3_files_check() {{ echo "files of rank $1: checked" >> "$SIM/events"; }}
"""


class Settings(ts.ScriptTest):
    def test_tp3_defaults_and_tag(self):
        code, out = self.bash('tp3_check; echo "$CONTEXT $POOL $PARALLEL $ROUND_ROWS $MIN_AVAIL_GIB $PROMPT_CACHE_GIB $IMAGE $(tp3_rope_rows)"',
                              {"TP": "3", "WORKER2": "w2"})
        self.assertEqual(code, 0, out)
        self.assertRegex(out.strip().splitlines()[-1],
                         r"^401408 3300000 8 16 80 1 tensorfold-dsv41:exl3-tp3-[0-9a-f]{12} 3301376$")
        code, out = self.bash('echo "$CONTEXT [$POOL] $PARALLEL [$ROUND_ROWS] $MIN_AVAIL_GIB $IMAGE"')
        self.assertRegex(out.strip().splitlines()[-1], r"^1048576 \[\] 4 \[\] 112 tensorfold-dsv41:exl3-tp2-[0-9a-f]{12}$")

    def test_two_spark_engine_settings_refused_at_tp3(self):
        for extra, want in [({"GRAPH_CAP": "64"}, "GRAPH_CAP=64"), ({"RDMA_IDLE_S": "0"}, "RDMA_IDLE_S=0"),
                            ({"RDMA_NAP_US": "100"}, "RDMA_NAP_US=100"), ({"PREFILL_INTERLEAVE_MS": "0"}, "PREFILL_INTERLEAVE_MS=0"),
                            ({"TF_DS_CONNECT_S": "60"}, "TF_DS_CONNECT_S=60"), ({"HEAD_ENGRAM_DIR": "/x"}, "HEAD_ENGRAM_DIR=/x"),
                            ({"KV_SHARD": "1"}, "KV_SHARD=1"), ({"SPILL_GIB": "64"}, "SPILL_GIB=64"),
                            ({"EAQ_V1_1": "1"}, "EAQ_V1_1=1")]:
            with self.subTest(extra=extra):
                code, out = self.bash("tp3_check; echo ok", {"TP": "3", "WORKER2": "w2", **extra})
                self.assertEqual(code, 1, out)
                self.assertIn("TP=3 runs LakyFx's TP3 engine, which has none of: " + want, out)
                self.assertNoTraceback(out)
                code, out = self.bash("tp3_check; echo ok", extra if "EAQ" not in want else {})
                self.assertEqual((code, out.strip().splitlines()[-1]), (0, "ok"), "TP=2 takes it: " + out)

    def test_pool_and_rows_at_tp3(self):
        for extra, ok in [({"POOL": "10000000"}, True), ({"POOL": "10000001"}, False), ({"POOL": "401408"}, False),
                          ({"POOL": "403456"}, True), ({"CONTEXT": "262144", "POOL": "264192"}, True),
                          ({"ROUND_ROWS": "32"}, False), ({"PROMPT_CACHE_GIB": "x"}, False)]:
            with self.subTest(extra=extra):
                code, out = self.bash("tp3_check; echo ok", {"TP": "3", "WORKER2": "w2", **extra})
                self.assertEqual(code, 0 if ok else 1, out)

    def test_prepared_state_at_tp3_only(self):
        code, two = self.bash("prepared_state")
        self.assertEqual(code, 0, two)
        self.assertNotIn("tp=3", two)
        code, three = self.bash("prepared_state", {"TP": "3", "WORKER2": "w2"})
        self.assertIn("tp=3 tp3=14a6d49c7cb99b8ce25f9755541da7e7d57cb1d1 arm=EAQ29", three)
        self.assertIn("rope=3301376", three)
        self.assertIn("tp3template=7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e", three)
        self.assertIn(r"wtp3=\$HOME/.cache/dsv41-tensorfold/tp3-45ee5d0f", three)


class Triangle(ts.ScriptTest):
    def pair(self, inv):
        p = subprocess.run([ts.BASH, "-c", f"source {self.repo}/scripts/nodes.sh; pair_triangle"],
                           input=inventory_lines(inv), capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout.splitlines()

    def test_pairing(self):
        out = self.pair(INVENTORY)
        self.assertEqual(out[:7], [
            f"rank 0 {RAILS} 0,3,12 rocep1s0f0,rocep1s0f1 3 enp1s0f0np0 192.0.2.1",
            f"rank 1 {RAILS} 12,0,3 rocep1s0f0,rocep1s0f1 3 enp1s0f1np1 192.0.2.2",
            f"rank 2 {RAILS} 3,12,0 rocep1s0f0,rocep1s0f1 3 enp1s0f0np0 192.0.2.33",
            "master 192.0.2.1", "peer 1 192.0.2.2 192.0.2.1", "peer 2 192.0.2.33 192.0.2.34", "range 192.0.2.0/26"])
        routes = [l for l in out if l.startswith("route ")]
        # the far end of each rail (12), and the three routes through the cable both ranks share
        self.assertEqual(len(routes), 15, routes)
        self.assertIn("route 0 192.0.2.2 - enp1s0f0np0", routes)
        self.assertIn("route 2 192.0.2.42 - enP2p1s0f0np0", routes)
        via = [l for l in routes if " - " not in l]
        self.assertEqual(via, ["route 1 192.0.2.33 192.0.2.18 enp1s0f0np0", "route 2 192.0.2.1 192.0.2.34 enp1s0f0np0",
                               "route 2 192.0.2.2 192.0.2.17 enp1s0f1np1"])

    def test_the_rails_sorted_whatever_the_listing_order(self):
        inv = dict(INVENTORY, w1=list(reversed(INVENTORY["w1"])))
        self.assertEqual(self.pair(inv), self.pair(INVENTORY))

    def test_what_does_not_fit(self):
        two = dict(INVENTORY, w2=INVENTORY["w2"][::2])
        out = self.pair(two)
        self.assertTrue(out and all(l.startswith("error ") for l in out), out)
        self.assertIn("rank 2 has 2 RoCE devices up with an IPv4 address (rocep1s0f0,rocep1s0f1), not 4", out[0])
        # w2's f1 port cabled nowhere near w1 (a subnet of its own): ranks 1 and 2 share no subnet
        apart = dict(INVENTORY, w2=INVENTORY["w2"][:2] + ["enp1s0f1np1 rocep1s0f1 198.51.100.1/29 3",
                                                          "enP2p1s0f1np1 roceP2p1s0f1 198.51.100.9/29 3"])
        self.assertIn("error ranks 1 and 2 share no subnet: three Sparks need a cable between every pair (a triangle)",
                      self.pair(apart))
        # w1's twins addressed crosswise (its PCIe function 0 in the twin's subnet): the rails pair in another order
        cross = dict(INVENTORY, w1=["enp1s0f0np0 rocep1s0f0 192.0.2.17/29 3", "enP2p1s0f0np0 roceP2p1s0f0 192.0.2.25/29 3",
                                    "enp1s0f1np1 rocep1s0f1 192.0.2.10/29 3", "enP2p1s0f1np1 roceP2p1s0f1 192.0.2.2/29 3"])
        self.assertTrue(any("ranks 0 and 1 do not list their rails to each other in one order" in l for l in self.pair(cross)))
        gid = dict(INVENTORY, head=INVENTORY["head"][:3] + ["enP2p1s0f1np1 roceP2p1s0f1 192.0.2.42/29 5"])
        self.assertTrue(any("rank 0: its rails have RoCE v2 IPv4 GID indexes 3,5" in l for l in self.pair(gid)))

    def detect(self, extra=None, body="detect_links; declare -p NODE_HCAS NODE_MASKS NODE_NCCL_HCAS NODE_GID NODE_DEV LINK_HEAD_ADDR LINK_WORKER_ADDR; echo \"$MASTER_ADDR $NCCL_ADDR_RANGE\""):
        nodes = self.repo / "scripts" / "nodes.sh"
        nodes.write_text(nodes.read_text() + stubs())
        return self.bash(f"need_workers\n{body}", {"TP": "3", "WORKER2": "w2", **(extra or {})})

    def test_detect_triangle(self):
        code, out = self.detect()
        self.assertEqual(code, 0, out)
        self.assertIn('NODE_MASKS=([0]="0,3,12" [1]="12,0,3" [2]="3,12,0")', out)
        self.assertIn('NODE_DEV=([0]="enp1s0f0np0" [1]="enp1s0f1np1" [2]="enp1s0f0np0")', out)
        self.assertIn('LINK_HEAD_ADDR=([1]="192.0.2.1" [2]="192.0.2.34")', out)
        self.assertIn("192.0.2.1 192.0.2.0/26", out)
        code, out = self.detect({"MASTER_ADDR": "198.51.100.7", "SOCKET_IFNAME": "eno1"})
        self.assertIn('NODE_DEV=([0]="eno1" [1]="eno1" [2]="eno1")', out)
        self.assertIn("198.51.100.7 192.0.2.0/26", out)

    def test_a_missing_route_is_the_command_that_adds_it(self):
        miss = {"TP3_MISS_HOST": "w2", "TP3_MISS": "192.0.2.1 192.0.2.34 enp1s0f0np0"}
        code, out = self.detect(miss)
        self.assertEqual(code, 1, out)
        self.assertIn("routes over the cables are missing", out)
        self.assertIn("rank 2 (w2): sudo ip route replace 192.0.2.1/32 via 192.0.2.34 dev enp1s0f0np0", out)
        code, out = self.detect({**miss, "DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertIn("WARN: DRY_RUN: routes over the cables are missing", out)
        self.assertIn("192.0.2.1,192.0.2.34,enp1s0f0np0", (self.sim / "ssh.log").read_text())

    def test_a_cable_that_does_not_fit_refuses(self):
        code, out = self.detect({"TP3_INV_w2": ";".join(INVENTORY["w2"][:2])})
        self.assertEqual(code, 1, out)
        self.assertIn("the three Sparks' cables do not fit TP=3", out)
        self.assertIn("rank 2 has 2 RoCE devices", out)

    def test_a_worker_out_of_reach_in_a_dry_run(self):
        self.unreachable("w2")
        code, out = self.detect({"DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertIn("DRY_RUN: WORKER2 cannot be reached: the triangle is not paired", out)
        self.assertIn('NODE_MASKS=([0]="<head:masks>" [1]="<WORKER:masks>" [2]="<WORKER2:masks>")', out)


class Launch(tpn.Base):
    def setUp(self):
        super().setUp()
        self.patch(self.repo, stubs() + tpn.MEM)
        self.env.update({"TP": "3", "WORKER2": "w2"})

    def ranks(self, out):
        cmds, lines = {}, out.splitlines()
        for k, l in enumerate(lines):
            m = re.match(r"\[dry-run\] rank (\d) ", l)
            if m:
                cmds[int(m.group(1))] = lines[k + 1].replace("\\", "")   # (bash 5.2's %q escapes commas)
        return cmds

    def test_dry_run_rank_commands(self):
        code, out = self.run_in(self.repo, ["start.sh"], {"DRY_RUN": "1", "PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertNoTraceback(out)
        self.assertIn("3 x DGX Spark", out)
        self.assertIn(" · pool 3300000", out)
        self.assertIn("rank 2 (w2): RoCE %s (GID 3), peer masks 3,12,0, NCCL rocep1s0f0,rocep1s0f1 over enp1s0f0np0, "
                      "NFS from 192.0.2.34" % RAILS, out)
        cmds = self.ranks(out)
        self.assertEqual(sorted(cmds), [0, 1, 2], out)
        for i, (masks, dev) in {1: ("12,0,3", "enp1s0f1np1"), 2: ("3,12,0", "enp1s0f0np0")}.items():
            c = cmds[i]
            self.assertIn(f"tf-dsv41-lanes /tp3/model /tp3/rank-cache {i} 3 192.0.2.1 29620 /tp3/kit --engram /tp3/engram "
                          "--token-map /tp3/kit/engram-token-map.json --pool 3300000 --streams 8 --drafts 1 --graphs 1 "
                          f"--side 1 --prefetch 1 --aio 1 --round-rows 16 --rdma {RAILS}", c)
            self.assertIn(r"-v $HOME/.cache/dsv41-tensorfold/tp3-45ee5d0f:/tp3:ro ", c)
            self.assertIn(f"-e TF_RDMA_PEER_MASKS={masks} ", c)
            self.assertIn(f"-e NCCL_SOCKET_IFNAME=={dev} -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 ", c)
        for c in cmds.values():
            for want in ["TF_DS_EXPERIMENTAL_TP3=1", "TF_DS_START_TIMEOUT=1800", "TF_DS_POOL_TOKENS=3300000", "TF_DS_STREAMS=8",
                         "TF_DS_ROUND_ROWS=16", "TF_DS_CACHE_SHORT_CONTINUATION=1", "TF_DS_TOOL_SEPARATOR_ROUNDTRIP=1",
                         "TF_RDMA_GID_INDEX=3", "TF_RDMA_TRACE=0", "NCCL_SWITCHLESS_RING_ONLY=1", "NCCL_ALGO=Ring",
                         "NCCL_IB_SUBNET_AWARE_ROUTING=1", "NCCL_IB_MERGE_NICS=0", "NCCL_IB_ADDR_RANGE=192.0.2.0/26",
                         "TENSORFOLD_NO_UPDATE_CHECK=1"]:
                self.assertIn(want, c)
            for never in ["TF_DS_POOL=", "TF_DS_WINDOW", "TF_RDMA_IDLE_S", "TF_RDMA_NAP_US", "PREFILL_INTERLEAVE",
                          "GRAPH_CAP", "--graph-cap", "NCCL_IB_GID_INDEX", "/rank-cache:ro", "dsv41_token_map"]:
                self.assertNotIn(never, c)
        c = cmds[0]
        for want in ["-e TF_TP_WORLD=3 ", "-e TF_TP_MASTER=192.0.2.1 ", "-e TF_DS_RANK_CACHE=/hf/tensorfold-dsv41/45ee5d0f/tp3/rank-cache ",
                     "-e TF_DS_KIT=/hf/tensorfold-dsv41/45ee5d0f/tp3/kit ", "-e TF_DS_ENGRAM=/hf/tensorfold-dsv41/45ee5d0f/tp3/engram ",
                     "-e TF_DS_TOKEN_MAP=/hf/tensorfold-dsv41/45ee5d0f/tp3/kit/engram-token-map.json ",
                     f"-e TF_RDMA_DEVICES={RAILS} ", "-e TF_RDMA_PEER_MASKS=0,3,12 ",
                     "-e NCCL_SOCKET_IFNAME==enp1s0f0np0 ",
                     "tensorfold-native serve /hf/tensorfold-dsv41/45ee5d0f/tp3/model --backend cuda",
                     "--context 401408 --parallel 8 ", "--prompt-cache-gib 1 --no-update-check"]:
            self.assertIn(want, c)

    def test_dry_run_ablit_and_a_worker_out_of_reach(self):
        code, out = self.run_in(self.repo, ["start.sh"], {"DRY_RUN": "1", "PREPARE": "0", "ABLIT": "1"})
        self.assertEqual(code, 0, out)
        cmds = self.ranks(out)
        self.assertIn("/tp3-45ee5d0f-ablit-0bdc2036:/tp3:ro", cmds[2])
        self.assertIn("serve /hf/tensorfold-dsv41/45ee5d0f-ablit-0bdc2036/tp3/model ", cmds[0])
        self.unreachable("w2")
        code, out = self.run_in(self.repo, ["start.sh"], {"DRY_RUN": "1", "PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertIn("cannot ssh to w2", out)
        self.assertIn("-e TF_RDMA_PEER_MASKS=<WORKER2:masks>", out.replace("\\", ""))

    def test_refused_before_anything_is_stopped(self):
        self.up("head", "w1", "w2")
        code, out = self.run_in(self.repo, ["start.sh", "restart"], {"PREPARE": "0", "GRAPH_CAP": "64"})
        self.assertEqual(code, 1, out)
        self.assertIn("which has none of: GRAPH_CAP=64", out)
        self.assertEqual([self.left(h) for h in ("head", "w1", "w2")], [[NAME]] * 3)

    def test_restart_stops_and_starts_every_rank_workers_first(self):
        self.up("head", "w1", "w2")
        code, out = self.run_in(self.repo, ["start.sh", "restart"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertNoTraceback(out)
        self.assertEqual(self.launched(), ["w1", "w2", "head"], out)
        self.assertIn("Stopped and removed rank 2 on w2", out)
        self.assertIn("Smoke test: one chat completion through all 3 ranks", out)
        self.assertIn("Loading: ~67 GiB of weights on each Spark", out)
        self.assertIn("is now LIVE", out)
        events = (self.sim / "events").read_text()
        for r in (0, 1, 2):
            self.assertIn(f"files of rank {r}: checked", events)

    def test_a_worker_short_of_memory_is_named(self):
        self.up()
        code, out = self.run_in(self.repo, ["start.sh"], {"PREPARE": "0", "MIN_AVAIL_GIB": "80", "MIN_AVAIL_WAIT_S": "0",
                                                          "TPN_W2_GB": "70.0"})
        self.assertEqual(code, 1, out)
        self.assertIn("only 70.0 GiB memory available on w2, 80 GiB needed", out)
        self.assertEqual(self.launched(), [], out)

    def test_a_rank_that_exits_during_the_load_is_named(self):
        self.up()
        (self.sim / "curl_fail").write_text("")
        bin_ = self.tmp / "bin"
        old = 'echo "$n" >> "$d/containers"; echo container-id ;;'
        docker = (bin_ / "docker").read_text()
        self.assertIn(old, docker)
        (bin_ / "docker").write_text(docker.replace(old, '[[ "$h" == w2 ]] || ' + old))
        code, out = self.run_in(self.repo, ["start.sh"], {"PREPARE": "0", "SERVED_NAME": NAME})
        self.assertEqual(code, 1, out)
        self.assertIn("rank 2 on w2 exited before the server was ready", out)
        self.assertIn("rank 2 (w2): last log lines", out)

    def test_stop_stops_all_three(self):
        self.up("head", "w1", "w2")
        code, out = self.run_in(self.repo, ["stop.sh"])
        self.assertEqual(code, 0, out)
        self.assertIn("Stopped and removed rank 2 on w2", out)
        self.assertEqual([self.left(h) for h in ("head", "w1", "w2")], [[], [], []])


class TwoSparksAsBefore(tpn.TwoSparksUnchanged):
    """scripts/test_nodes_tpn.py's comparison of TP=2, against the tree before TP=3 (TP3_BASE)."""

    def setUp(self):
        tpn.Base.setUp(self)
        if shutil.which("git") is None or subprocess.run(["git", "-C", str(ts.ROOT), "cat-file", "-e", BASE + "^{commit}"],
                                                         capture_output=True).returncode != 0:
            self.skipTest(f"no commit {BASE} in {ts.ROOT} (TP3_BASE=<commit>)")
        self.base = self.tmp / "base"
        self.base.mkdir()
        arc = subprocess.run(["git", "-C", str(ts.ROOT), "archive", BASE, *ts.COPY], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(self.base)], input=arc, check=True)
        for tree in (self.base, self.repo):
            self.patch(tree, tpn.LINK_TP2 + tpn.MEM)


if __name__ == "__main__":
    unittest.main()
