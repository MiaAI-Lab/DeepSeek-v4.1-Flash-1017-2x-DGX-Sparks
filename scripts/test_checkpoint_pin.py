"""Opt-in public Hub integration regression: RUN_HF_PIN_CHECK=1 python3 scripts/test_checkpoint_pin.py."""
import json
import os
from pathlib import Path
import re
import unittest
import urllib.request


@unittest.skipUnless(os.environ.get("RUN_HF_PIN_CHECK") == "1", "requires public Hugging Face access")
class CheckpointPinTest(unittest.TestCase):
    def test_default_revision_resolves_with_checkpoint_files(self):
        config = (Path(__file__).resolve().parent / "config.sh").read_text()
        revision = re.search(r"_rev=([0-9a-f]{40})", config).group(1)
        url = "https://huggingface.co/api/models/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw/revision/" + revision
        with urllib.request.urlopen(url, timeout=30) as response:
            metadata = json.loads(response.read(16 * 1024 * 1024))
        self.assertEqual(metadata["sha"], revision)
        names = {item["rfilename"] for item in metadata["siblings"]}
        self.assertTrue({"config.json", "tokenizer_config.json", "model.safetensors.index.json"} <= names)
        self.assertEqual(len([name for name in names if re.fullmatch(r"model-\d{5}-of-00039\.safetensors", name)]), 39)


if __name__ == "__main__":
    unittest.main()
