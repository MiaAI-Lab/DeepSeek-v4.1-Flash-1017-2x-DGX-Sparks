#!/usr/bin/env python3
"""CPU regression: worker cleanup must work with unmatched-glob errors enabled.

Usage: python3 scripts/test_worker_rank_cache.py [repository-root]
Bash failglob reproduces zsh's default error for unmatched globs without zsh.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = (Path(sys.argv.pop(1)).resolve() if __name__ == '__main__' and len(sys.argv) > 1
        else Path(__file__).resolve().parents[1])


class WorkerRankCacheTest(unittest.TestCase):
    def cleanup_command(self, cache, rank=1, tp=2):
        lines = (ROOT / 'scripts/prepare.sh').read_text().splitlines()
        matches = [line.strip() for line in lines if line.strip().startswith('worker "$i" "mkdir -p $WORKER_CACHE_DIR &&')]
        self.assertEqual(len(matches), 1)
        # Only substitute the trusted temporary path and worker i's rank and TP (prepare.sh's step 7 runs it for each
        # worker); the extracted worker command is the exact cleanup command sent to the worker's login shell.
        return (matches[0][len('worker "$i" "'):-1].replace('$WORKER_CACHE_DIR', str(cache))
                .replace('${i}', str(rank)).replace('$TP', str(tp)))

    def test_cleanup_empty_cache_without_matching_glob(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'rank-cache'
            result = subprocess.run(['bash', '-O', 'failglob', '-c', self.cleanup_command(cache)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(cache.is_dir())

    def test_cleanup_removes_stale_weights_and_partial_only(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'rank-cache'
            cache.mkdir()
            for name in ['rank1of2-old.bin', '.rank1of2.partial', 'keep.bin']:
                (cache / name).write_text('fixture')
            result = subprocess.run(['bash', '-O', 'failglob', '-c', self.cleanup_command(cache)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(path.name for path in cache.iterdir()), ['keep.bin'])

    def test_cleanup_of_rank_2_of_3_leaves_rank_1s(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'rank-cache'
            cache.mkdir()
            for name in ['rank2of3-old.bin', '.rank2of3.partial', 'rank1of3-x.bin', 'rank1of2-y.bin']:
                (cache / name).write_text('fixture')
            result = subprocess.run(['bash', '-O', 'failglob', '-c', self.cleanup_command(cache, 2, 3)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(path.name for path in cache.iterdir()), ['rank1of2-y.bin', 'rank1of3-x.bin'])


if __name__ == '__main__':
    unittest.main()
