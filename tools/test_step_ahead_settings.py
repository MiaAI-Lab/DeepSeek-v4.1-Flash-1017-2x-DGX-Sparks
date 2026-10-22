#!/usr/bin/env python3
"""STEP_AHEAD (zig/patches/0011): 0 or 1 before either Spark is contacted, and only 1 reaches the ranks."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StepAheadSettingsTest(unittest.TestCase):
    def test_bounds_before_worker_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            for name in ("start.sh", "scripts/config.sh", "scripts/nodes.sh"):
                shutil.copy2(ROOT / name, root / name)
            env = {"PATH": os.environ["PATH"], "HOME": folder, "ABLIT": "0", "IMAGE": "step-ahead-settings-test"}
            for value, message in (("0", "WORKER is not set"), ("1", "WORKER is not set"), ("2", "STEP_AHEAD is 0 or 1"),
                                   ("on", "STEP_AHEAD is 0 or 1"), ("", "WORKER is not set")):  # empty: the default, 0
                with self.subTest(value=value):
                    result = subprocess.run(["bash", str(root / "start.sh")], env=env | {"STEP_AHEAD": value},
                                            capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)

    def test_only_on_reaches_the_ranks_and_a_raw_export_cannot(self):
        source = (ROOT / "start.sh").read_text()
        engine = source[source.index("ENGINE_ENV=("):source.index("envs() {")]
        base = dict(PATH=os.environ["PATH"], CONTEXT="65536", POOL="", REPLAY_FLOOR="1", PARALLEL="4", HF_HUB_OFFLINE="1",
                    ROUND_ROWS="", PREFILL_INTERLEAVE_MS="250", RDMA_IDLE_S="2", RDMA_NAP_US="500")
        for value, raw, want in (("0", None, []), ("1", None, ["TF_DS_STEP_AHEAD=1"]), ("0", "1", []), ("1", "0", ["TF_DS_STEP_AHEAD=1"])):
            with self.subTest(step_ahead=value, raw=raw):
                env = base | {"STEP_AHEAD": value} | ({"TF_DS_STEP_AHEAD": raw} if raw is not None else {})
                result = subprocess.run(["bash", "-euc", engine + '\nprintf "%s\\n" "${ENGINE_ENV[@]}"'], env=env,
                                        capture_output=True, text=True, check=True)
                self.assertEqual([line for line in result.stdout.splitlines() if line.startswith("TF_DS_STEP_AHEAD=")], want)


if __name__ == "__main__":
    unittest.main()
