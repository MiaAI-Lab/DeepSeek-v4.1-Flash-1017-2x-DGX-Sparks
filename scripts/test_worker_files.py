#!/usr/bin/env python3
"""CPU-only NFS preflight regression tests. Run: python3 scripts/test_worker_files.py."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkerFilesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.fixture = self.base / "hf"
        self.paths = ["checkpoint/config.json", "kit-test/kit/MANIFEST", "data-test/dsv41_token_map.json"]
        for name in self.paths:
            path = self.fixture / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        docker = self.base / "docker"
        docker.write_text("""#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ['DOCKER_LOG']).write_text(json.dumps(args))
if os.environ.get('DOCKER_ERROR'):
    print(os.environ['DOCKER_ERROR'], file=sys.stderr)
    sys.exit(125)
assert args[:4] == ['run', '--rm', '--entrypoint', 'sh'], args
assert args[4:6] == ['-v', os.environ['VOLUME'] + ':/hf:ro'], args
assert args[6] == 'test-image', args
command = ['sh'] + args[7:]
command = [os.environ['FIXTURE'] + arg[3:] if arg.startswith('/hf/') else arg for arg in command]
sys.exit(subprocess.call(command))
""")
        docker.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.base) + os.pathsep + os.environ["PATH"],
                        FIXTURE=str(self.fixture), DOCKER_LOG=str(self.base / "docker.json"), VOLUME="test-volume")

    def run_check(self, volume="test-volume", model="/hf/checkpoint"):
        self.env["VOLUME"] = volume
        script = r'''
set -euo pipefail
source "$1/scripts/nodes.sh"
worker() { shift; eval "$*"; }
worker_host() { echo worker.example; }
nfs_server() { echo 192.0.2.1; }
model_in_container() { printf '%s\n' "$MODEL_PATH"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
NFS_VOLUME=$VOLUME; NFS_PATH=/; IMAGE=test-image; KIT_SUB=kit-test; DATA_SUB=data-test
worker_files_check 1
'''
        return subprocess.run(["bash", "-c", script, "test", str(ROOT)],
                              env=dict(self.env, MODEL_PATH=model), text=True, capture_output=True)

    def test_valid_named_volume(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        args = json.loads((self.base / "docker.json").read_text())
        self.assertNotIn("--gpus", args)

    def test_each_required_file_is_checked(self):
        for name in self.paths:
            with self.subTest(name=name):
                path = self.fixture / name
                path.unlink()
                result = self.run_check()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("/hf/" + name, result.stderr)
                self.assertIn("NFSv4 root", result.stderr)
                path.write_text("fixture\n")

    def test_required_paths_must_be_regular_files(self):
        for name in self.paths:
            with self.subTest(name=name):
                path = self.fixture / name
                path.unlink()
                path.mkdir()
                result = self.run_check()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("/hf/" + name, result.stderr)
                path.rmdir()
                path.write_text("fixture\n")

    def test_empty_file(self):
        (self.fixture / self.paths[1]).write_text("")
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing, empty or unreadable", result.stderr)

    def test_wrong_root_and_dangling_symlink(self):
        config = self.fixture / self.paths[0]
        config.unlink()
        config.symlink_to("missing-blob")
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("config.json", result.stderr)
        self.assertIn("competing host/container NFS servers", result.stderr)

    def test_docker_and_stale_handle_errors_are_preserved(self):
        for error in ["failed to mount local volume: no such file or directory\nRun 'docker run --help' for more information",
                      "Stale file handle"]:
            with self.subTest(error=error):
                self.env["DOCKER_ERROR"] = error
                result = self.run_check()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertIn("test-volume", result.stderr)
                self.assertIn("192.0.2.1", result.stderr)

    def test_host_path_mount_and_shell_quoting(self):
        special = "checkpoint with spaces;literal"
        (self.fixture / special).mkdir()
        (self.fixture / special / "config.json").write_text("fixture\n")
        result = self.run_check(volume="/local/cache with spaces", model="/hf/" + special)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_call_sites(self):
        start = (ROOT / "start.sh").read_text()
        self.assertIn("if (( ! DRY )); then\n  # Preparation's marker", start)
        check = start.index("worker_files_check 1")
        self.assertLess(start.index("./scripts/prepare.sh\n"), check)
        self.assertLess(check, start.index("if exists_here || exists_worker 1; then ./stop.sh; fi"))
        self.assertLess(check, start.index("# ---------------------------------------------------------------- 3. launch"))
        prepare = (ROOT / "scripts/prepare.sh").read_text()
        self.assertIn("ensure_nfs_volume 1\nworker_files_check 1\n", prepare)


if __name__ == "__main__":
    unittest.main()
