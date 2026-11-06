"""Offline tests for scripts/runtime.sh: the one place that chooses docker or podman.

Run: python3 scripts/test_runtime.py      (or python3 -m unittest discover -s scripts -p 'test_*.py')
No Spark, GPU, Docker or Podman: each case sources config.sh in a bash with the variables set, then asks ctr, rt_save
and rt_build_net what they would run (the runtime CLI itself is a shell function that prints its arguments).
"""
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(os.environ.get("SCRIPTS_DIR") or Path(__file__).resolve().parent.parent)


def bash(script, env=None, args=()):
    # only `env` sets the settings under test: "unset" has to mean unset, as it is on a host with no local.sh
    e = {k: v for k, v in os.environ.items() if k not in ("CONTAINER_RT", "CONTAINER_RT_SUDO", "CGROUP_PARENT")}
    e.update(env or {})
    out = subprocess.run(["bash", "-c", script, "test-runtime", *args], cwd=str(ROOT), env=e,
                         capture_output=True, text=True)
    return out.returncode, out.stdout.strip(), out.stderr.strip()


PROBE = r"""
set -euo pipefail
# the runtime CLIs are stubs: this test asks which command the recipe would run, not whether one is installed
docker() { printf 'docker %s\n' "$*"; }
podman() { printf 'podman %s\n' "$*"; }
sudo()   { printf 'sudo %s\n' "$*"; }
source ./scripts/config.sh
case "$1" in
  words)  printf '%s\n' "$RT_WORDS" ;;
  ctr)    ctr version ;;
  save)   rt_save some-image ;;
  build)  printf '[%s]\n' "$(rt_build_net)" ;;
esac
"""


class Runtime(unittest.TestCase):
    def probe(self, what, **env):
        code, out, err = bash(PROBE, env, [what])
        self.assertEqual(code, 0, err)
        return out

    def test_default_is_docker(self):
        self.assertEqual(self.probe("words", CONTAINER_RT="", CONTAINER_RT_SUDO=""), "docker")
        self.assertEqual(self.probe("ctr", CONTAINER_RT="", CONTAINER_RT_SUDO=""), "docker version")

    def test_podman_rootless(self):
        env = {"CONTAINER_RT": "podman", "CONTAINER_RT_SUDO": "0"}
        self.assertEqual(self.probe("words", **env), "podman")

    def test_podman_rootful(self):
        env = {"CONTAINER_RT": "podman", "CONTAINER_RT_SUDO": "1"}
        self.assertEqual(self.probe("words", **env), "sudo -n podman")

    def test_save_keeps_the_docker_archive_format_under_podman(self):
        self.assertIn("--format docker-archive", self.probe("save", CONTAINER_RT="podman", CONTAINER_RT_SUDO="1"))
        self.assertNotIn("--format", self.probe("save", CONTAINER_RT="docker", CONTAINER_RT_SUDO="0"))

    def test_builds_use_the_host_network_only_under_podman(self):
        self.assertEqual(self.probe("build", CONTAINER_RT="podman", CONTAINER_RT_SUDO="1"), "[--network host]")
        self.assertEqual(self.probe("build", CONTAINER_RT="docker", CONTAINER_RT_SUDO="0"), "[]")

    def test_a_bad_runtime_is_refused(self):
        code, _, err = bash(PROBE, {"CONTAINER_RT": "containerd", "CONTAINER_RT_SUDO": "0"}, ["words"])
        self.assertNotEqual(code, 0)
        self.assertIn("CONTAINER_RT is docker or podman", err)

    def test_sudo_with_docker_is_refused(self):
        code, _, err = bash(PROBE, {"CONTAINER_RT": "docker", "CONTAINER_RT_SUDO": "1"}, ["words"])
        self.assertNotEqual(code, 0)
        self.assertIn("CONTAINER_RT_SUDO is for podman", err)

    def test_cgroup_parent_is_optional(self):
        # start.sh reads it under set -u (its --cgroup-parent line); runtime.sh owns the empty default
        self.assertEqual(self.probe("words", CONTAINER_RT="podman", CONTAINER_RT_SUDO="1"), "sudo -n podman")

    def test_cgroup_parent_must_be_a_slice_name(self):
        # it travels to the worker inside an ssh command line (start.sh's RUN_ARGS), so no shell syntax and no path
        for bad in ("tensorfold", "-rf.slice", "$(id).slice", "tensorfold.slice; id", "../../etc/passwd.slice"):
            with self.subTest(value=bad):
                code, _, err = bash(PROBE, {"CONTAINER_RT": "podman", "CGROUP_PARENT": bad}, ["words"])
                self.assertNotEqual(code, 0)
                self.assertIn("CGROUP_PARENT is a slice name", err)
        self.assertEqual(self.probe("words", CONTAINER_RT="podman", CGROUP_PARENT="tensorfold.slice"), "podman")


class NoBareDocker(unittest.TestCase):
    """The scripts must call ctr, not docker: a bare `docker` word is how a call slips past the switch."""

    FILES = ["start.sh", "stop.sh", "scripts/prepare.sh", "scripts/nodes.sh", "scripts/port.sh",
             "scripts/publish-image.sh", "scripts/memguard.sh"]

    def test_no_docker_command_outside_the_runtime_file(self):
        offenders = []
        for name in self.FILES:
            for no, line in enumerate((ROOT / name).read_text().splitlines(), 1):
                s = line.lstrip()
                if s.startswith("#"):
                    continue
                for word in ("docker", "podman"):
                    if f" {word} " in f" {line} " and "CONTAINER_RT" not in line and "docker group" not in line:
                        offenders.append(f"{name}:{no}: {line.strip()}")
        self.assertEqual(offenders, [], "calls that bypass ctr:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main(verbosity=2)
