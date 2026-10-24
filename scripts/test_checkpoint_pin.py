"""Checkpoint pin checks: the EAQ v1 default, and (RUN_HF_PIN_CHECK=1) both pins on the public Hub."""
import json
import os
from pathlib import Path
import re
import unittest
import urllib.request

CONFIG = (Path(__file__).resolve().parent / "config.sh").read_text()
PINS = dict(re.findall(r"(Mia-AiLab/DeepSeek-V4\.1-Flash-1017-EXL3-EAQ-v1(?:\.1)?-2\.9-bpw)\) _rev=([0-9a-f]{40})", CONFIG))


class DefaultCheckpointTest(unittest.TestCase):
    def test_eaq_v1_is_the_default(self):
        self.assertIn('EAQ_V1_1="${EAQ_V1_1:-0}"', CONFIG)
        self.assertIn("_default_id=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw", CONFIG)
        self.assertIn("_default_rev=45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b", CONFIG)
        self.assertIn("_default_id=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw", CONFIG)
        self.assertIn("_default_rev=aee640e2afba93be683bb5b9b420115fe7fbac20", CONFIG)


@unittest.skipUnless(os.environ.get("RUN_HF_PIN_CHECK") == "1", "requires public Hugging Face access")
class CheckpointPinTest(unittest.TestCase):
    def test_pins_resolve_with_checkpoint_files(self):
        self.assertEqual(len(PINS), 2)
        for repo, revision in PINS.items():
            url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
            with urllib.request.urlopen(url, timeout=30) as response:
                metadata = json.loads(response.read(16 * 1024 * 1024))
            self.assertEqual(metadata["sha"], revision)
            names = {item["rfilename"] for item in metadata["siblings"]}
            self.assertTrue({"config.json", "tokenizer_config.json", "model.safetensors.index.json"} <= names)
            self.assertEqual(len([name for name in names if re.fullmatch(r"model-\d{5}-of-00039\.safetensors", name)]), 39)


if __name__ == "__main__":
    unittest.main()
