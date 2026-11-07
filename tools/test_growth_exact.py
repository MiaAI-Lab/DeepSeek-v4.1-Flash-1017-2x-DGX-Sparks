"""The pressure comparison must require both actual reuse and token identity."""
import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

PATH = pathlib.Path(__file__).parent / 'claims/0063-growth-exact.py'
SPEC = importlib.util.spec_from_file_location('growth_exact', PATH)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class GrowthExact(unittest.TestCase):
    def test_pressure_requires_equal_token_hash(self):
        def run(cached, digest):
            with tempfile.TemporaryDirectory() as tmp:
                ref = pathlib.Path(tmp) / 'cold.json'
                out = pathlib.Path(tmp) / 'pressure.json'
                row = {'prompt': 282826, 'cached': 0, 'completion_tokens': 2,
                       'message': {'role': 'assistant', 'content': 'test'},
                       'finish_reason': 'stop', 'token_sha': 'abc'}
                ref.write_text(json.dumps({'phase': 'cold', 'seed': 1, 'results': [row]}))
                responses = []
                for value, sha in [(0, 'x'), (0, 'x'), (cached, digest)]:
                    responses.append(io.BytesIO(json.dumps({'usage': {'prompt_tokens': 282826,
                        'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': value}},
                        'choices': [{'message': row['message'], 'finish_reason': 'stop'}],
                        'tensorfold': {'token_sha': sha}}).encode()))
                with patch.object(sys, 'argv', [str(PATH), '--phase', 'pressure', '--seed', '1',
                                               '--out', str(out), '--reference', str(ref)]), \
                     patch.object(CHECK, 'prose', return_value='text'), \
                     patch.object(CHECK, 'drain_writes'), patch.object(CHECK, 'open_url', side_effect=responses), \
                     contextlib.redirect_stdout(io.StringIO()):
                    try:
                        CHECK.main()
                    except SystemExit as exc:
                        return str(exc)
                    return 'pass'

        self.assertEqual(run(282624, 'abc'), 'pass')
        self.assertIn('did not resume', run(0, 'abc'))
        self.assertIn('hashes differ', run(282624, 'different'))
        self.assertIn('unavailable', run(282624, None))

    def test_fork_requires_recovery_of_the_longer_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = pathlib.Path(tmp) / 'cold.json'
            row = {'prompt': 282826, 'completion_tokens': 2,
                   'message': {'role': 'assistant', 'content': 'test'},
                   'finish_reason': 'stop', 'token_sha': 'abc'}
            ref.write_text(json.dumps({'phase': 'cold', 'scenario': 'fork', 'seed': 1, 'results': [row]}))

            def run(restored):
                responses = []
                for prompt, cached in [(282826, 0), (298524, 282624), (282826, 282624), (298524, restored)]:
                    responses.append(io.BytesIO(json.dumps({'usage': {'prompt_tokens': prompt,
                        'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': cached}},
                        'choices': [{'message': row['message'], 'finish_reason': 'stop'}],
                        'tensorfold': {'token_sha': 'abc'}}).encode()))
                with patch.object(sys, 'argv', [str(PATH), '--phase', 'pressure', '--scenario', 'fork',
                                               '--seed', '1', '--out', str(pathlib.Path(tmp) / 'out.json'),
                                               '--reference', str(ref)]), \
                     patch.object(CHECK, 'prose', return_value='text'), \
                     patch.object(CHECK, 'drain_writes'), patch.object(CHECK, 'open_url', side_effect=responses), \
                     contextlib.redirect_stdout(io.StringIO()):
                    try:
                        CHECK.main()
                    except SystemExit as exc:
                        return str(exc)
                    return 'pass'

            self.assertEqual(run(296960), 'pass')
            self.assertIn('did not recover', run(282624))

    def test_quick_uses_small_prompts_and_pool(self):
        seen = []

        def prose(size, seed):
            seen.append(size)
            return 'text'

        def response(request, timeout):
            if isinstance(request, str):
                return io.BytesIO(b'{"prompt_cache": {"spill": [{"rank": 0, "writes_pending": 0}, {"rank": 1, "writes_pending": 0}]}}')
            return io.BytesIO(json.dumps({'usage': {'prompt_tokens': 4800, 'completion_tokens': 2,
                'prompt_tokens_details': {'cached_tokens': 0}},
                'choices': [{'message': {'role': 'assistant', 'content': 'OK'}, 'finish_reason': 'stop'}],
                'tensorfold': {'token_sha': 'abc'}}).encode())

        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / 'cold.json'
            with patch.object(sys, 'argv', [str(PATH), '--quick', '--phase', 'cold', '--scenario', 'fork',
                                           '--seed', '1', '--out', str(out)]), \
                 patch.object(CHECK, 'prose', side_effect=prose), \
                 patch.object(CHECK, 'open_url', side_effect=response), \
                 contextlib.redirect_stdout(io.StringIO()):
                CHECK.main()
            result = json.loads(out.read_text())
        self.assertEqual(result['context'], 8192)
        self.assertIn(6000, seen)
        self.assertIn(1000, seen)


if __name__ == '__main__':
    unittest.main()
