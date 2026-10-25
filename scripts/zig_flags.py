#!/usr/bin/env python3
"""ENGINE=zig: check the serve arguments against `tensorfold-native capabilities --json` (on stdin).

Every --flag must be one the binary serves, and a flag with a fixed set of values must get one of them. The binary
refuses the rest at start (an unsupported flag, --tp 3), after the container is up; this says so before anything is
stopped. Usage: tensorfold-native capabilities --json | zig_flags.py <serve arguments after the model>
"""
import json
import sys

flags = json.load(sys.stdin)["serve"]["flags"]
args = sys.argv[1:]
errors = []
i = 0
while i < len(args):
    a = args[i]
    i += 1
    if not a.startswith("--"):
        continue
    name, eq, value = a.partition("=")
    if name not in flags:
        errors.append(f"{name} is not served by this tensorfold-native yet")
        continue
    values = flags[name].get("values")
    takes_value = bool(eq) or (values is not None) or name in {
        "--host", "--port", "--name", "--context", "--parallel", "--max-tokens", "--drafter", "--tp", "--rank",
        "--master", "--master-port", "--prompt-cache-gib", "--backend", "--temperature", "--top-p", "--top-k",
        "--min-p", "--reasoning-effort", "--alias", "--api-key", "--api-key-file", "--chat-template", "--device",
        # the prompt cache's disk tier (scripts/port.sh)
        "--spill-dir", "--spill-gib", "--spill-min-free-gib", "--spill-min-tokens", "--spill-quota",
    }
    if takes_value and not eq:
        value = args[i] if i < len(args) else ""
        i += 1
    if values is not None and value not in values:
        errors.append(f"{name} takes one of {', '.join(values)}, not {value}")
if errors:
    sys.exit("tensorfold serve: error: " + "; ".join(errors))
