#!/usr/bin/env python3
"""Exercise launcher RDMA limits before it can contact either Spark."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RdmaSettingsTest(unittest.TestCase):
    def test_bounds_before_worker_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            for name in ("start.sh", "scripts/config.sh", "scripts/nodes.sh", "scripts/port.sh", "scripts/memavail.sh"):
                shutil.copy2(ROOT / name, root / name)
            env = {"PATH": os.environ["PATH"], "HOME": folder, "ABLIT": "0", "IMAGE": "rdma-settings-test"}
            cases = [
                ({"RDMA_IDLE_S": "0", "RDMA_NAP_US": "1"}, "WORKER is not set"),
                ({"RDMA_IDLE_S": "3600", "RDMA_NAP_US": "1000000"}, "WORKER is not set"),
                ({"RDMA_IDLE_S": "3600.1"}, "RDMA_IDLE_S is 0 to 3600"),
                ({"RDMA_IDLE_S": "99999999999999999999999999"}, "RDMA_IDLE_S is 0 to 3600"),
                ({"RDMA_NAP_US": "0"}, "RDMA_NAP_US is 1 to 1000000"),
                ({"RDMA_NAP_US": "1000001"}, "RDMA_NAP_US is 1 to 1000000"),
                ({"RDMA_NAP_US": "99999999999999999999999999"}, "RDMA_NAP_US is 1 to 1000000"),
            ]
            for settings, message in cases:
                with self.subTest(settings=settings):
                    result = subprocess.run(
                        ["bash", str(root / "start.sh")], env=env | settings,
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
