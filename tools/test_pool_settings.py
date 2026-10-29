"""Both ranks must derive pool/window sizing from the same recipe settings."""
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PoolSettingsTest(unittest.TestCase):
    def test_raw_engine_exports_cannot_override_rank_mapping(self):
        source = (ROOT / "start.sh").read_text()
        engine = source[source.index("ENGINE_ENV=("):source.index("envs() {")]
        follow = next(line for line in source.splitlines() if line.startswith('FOLLOW="'))
        for pool in ("", "131072"):
            with self.subTest(pool=pool):
                env = dict(PATH=os.environ["PATH"], CONTEXT="65536", POOL=pool, REPLAY_FLOOR="0", PARALLEL="4",
                           HF_HUB_OFFLINE="1", ROUND_ROWS="", PREFILL_INTERLEAVE_MS="0", RDMA_IDLE_S="2",
                           RDMA_NAP_US="500", DRAFTS="1", GRAPHS="1", SIDE="0", PREFETCH="0", ENGRAM_AIO="0",
                           TF_DS_POOL="999999", TF_DS_WINDOW="1")
                result = subprocess.run(["bash", "-euc", engine + follow + '\nprintf "%s\\n" "${ENGINE_ENV[@]}" "$FOLLOW"'],
                                        env=env, capture_output=True, text=True, check=True)
                lines = result.stdout.splitlines()
                self.assertNotIn("TF_DS_POOL=999999", lines)
                self.assertNotIn("TF_DS_WINDOW=1", lines)
                self.assertEqual([line for line in lines if line.startswith("TF_DS_POOL=")],
                                 ["TF_DS_POOL=" + pool] if pool else [])
                self.assertEqual([line for line in lines if line.startswith("TF_DS_WINDOW=")],
                                 ["TF_DS_WINDOW=65536"] if pool else [])
                self.assertTrue(lines[-1].startswith("--pool " + (pool or "65536") + " --streams 4"))


if __name__ == "__main__":
    unittest.main()
