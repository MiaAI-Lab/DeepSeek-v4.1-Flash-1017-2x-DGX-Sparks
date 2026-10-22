"""CPUSET and its per-node overrides reach both docker runs, and only when set."""
from pathlib import Path
import os
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


def resolve(**env):
    """The three CPUSET lines of config.sh, run on their own."""
    lines = [line for line in (ROOT / "scripts" / "config.sh").read_text().splitlines()
             if line.startswith(("CPUSET=", "HEAD_CPUSET=", "WORKER_CPUSET="))]
    assert len(lines) == 3, lines
    script = "\n".join(lines) + '\nprintf "%s|%s|%s" "$CPUSET" "$HEAD_CPUSET" "$WORKER_CPUSET"\n'
    environment = {"PATH": "/usr/bin:/bin"}
    environment.update(env)
    result = subprocess.run(["bash", "-c", script], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


class CpusetTest(unittest.TestCase):
    def test_defaults_leave_every_cpu(self):
        self.assertEqual(resolve(), "||")

    def test_cpuset_reaches_both_nodes(self):
        self.assertEqual(resolve(CPUSET="0-4,10-14"), "0-4,10-14|0-4,10-14|0-4,10-14")

    def test_per_node_overrides(self):
        self.assertEqual(resolve(CPUSET="0-4,10-14", WORKER_CPUSET="2-3"), "0-4,10-14|0-4,10-14|2-3")
        self.assertEqual(resolve(HEAD_CPUSET="1"), "|1|")
        self.assertEqual(resolve(HEAD_CPUSET="1", WORKER_CPUSET="2"), "|1|2")

    def test_both_docker_runs_carry_the_flag(self):
        start = (ROOT / "start.sh").read_text()
        self.assertIn("${WORKER_CPUSET:+--cpuset-cpus=$WORKER_CPUSET}", start)
        self.assertIn("${HEAD_CPUSET:+--cpuset-cpus=\"$HEAD_CPUSET\"}", start)


if __name__ == "__main__":
    unittest.main()
