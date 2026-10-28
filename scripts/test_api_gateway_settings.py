#!/usr/bin/env python3
"""Regression test for API_GATEWAY in scripts/config.sh and start.sh.

`API_GATEWAY=0` (the default) must print exactly what the recipe printed before the gateway existed: the recorded
fixture is the output of `DRY_RUN=1 ./start.sh` on the branch without the feature. The image tag is the one derived
value that necessarily changes when patch 0035 joins zig/patches, so it is normalized on both sides; every other byte
is compared. `API_GATEWAY=1` must move the engine to 127.0.0.1:ENGINE_PORT, enable the token route, and print the
gateway container with its flags and mounts. Invalid values must be rejected with their message.

The container has no Docker, no ssh and no RoCE devices, so the test supplies small stubs for docker, ssh, ip, ls and
cat; ssh runs its command locally, and the fake sysfs answers one RoCE device. The output is deterministic. Run:
    python3 -I scripts/test_api_gateway_settings.py
"""
import json
import os
import re
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "scripts" / "fixtures" / "start-api-gateway-0.txt"
HOME = "/tmp/ds41-home"
IMAGE_TAG = re.compile(r"tensorfold-dsv41:exl3-tp2-[0-9a-f]{12}")

STUBS = {
    # FAKE_UID makes `id -u` answer it (the refusal to run a keyed gateway as root); otherwise the real id.
    "id": """#!/usr/bin/env bash
if [[ -n "${FAKE_UID:-}" && "$1" == "-u" ]]; then echo "$FAKE_UID"; else exec /usr/bin/id "$@"; fi
""",
    "docker": """#!/usr/bin/env bash
exit 1
""",
    "ssh": """#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
[[ $# -gt 0 ]] && shift
exec bash -c "$*"
""",
    "ip": """#!/usr/bin/env bash
args=("$@"); i=0
while [[ $i -lt ${#args[@]} ]]; do
  case "${args[$i]}" in
    -o|-4|-4o|-o4) i=$((i+1)) ;;
    *) break ;;
  esac
done
cmd=${args[$i]:-}; sub=${args[$((i+1))]:-}
case "$cmd $sub" in
  "route get") echo "127.0.0.1 dev lo src 127.0.0.1 uid 0" ;;
  "addr show") echo "1: lo    inet 127.0.0.1/8 scope host lo" ;;
  "route show") echo "127.0.0.0/8 dev lo scope host src 127.0.0.1" ;;
  *) exit 1 ;;
esac
""",
    "ls": """#!/usr/bin/env bash
case "$1" in
  /sys/class/net/*/device/infiniband) echo mlx5_0 ;;
  /sys/class/infiniband/*) echo mlx5_0 ;;
  *) exec /bin/ls "$@" ;;
esac
""",
    "cat": """#!/usr/bin/env bash
case "$1" in
  /sys/class/infiniband/*/ports/1/gids/*) echo "0000:0000:0000:0000:0000:ffff:7f00:0001" ;;
  /sys/class/infiniband/*/ports/1/gid_attrs/types/*) echo "RoCE v2" ;;
  /sys/class/net/*/operstate) echo up ;;
  *) exec /bin/cat "$@" ;;
esac
""",
}


def normalized(text):
    return IMAGE_TAG.sub("tensorfold-dsv41:exl3-tp2-<hash>", text)


class ApiGatewaySettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile

        cls.tmp = tempfile.TemporaryDirectory(prefix="ds41-api-gateway-")
        cls.bin = Path(cls.tmp.name) / "bin"
        cls.bin.mkdir()
        for name, body in STUBS.items():
            path = cls.bin / name
            path.write_text(body)
            path.chmod(0o755)
        Path(HOME).mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_start(self, **extra):
        env = {
            "PATH": f"{self.bin}:/usr/local/bin:/usr/bin:/bin",
            "HOME": HOME,
            "WORKER": "dummy@127.0.0.1",
            "FABRIC_PEER": "127.0.0.1",
            "DRY_RUN": "1",
            # The feature's settings, explicit: an exported API_GATEWAY=1 or GATEWAY_* must not change the regression.
            "API_GATEWAY": "0",
            "TOKEN_ROUTE": "0",
            "TF_DS_API_EFFORTS": "",
            "GATEWAY_IMAGE": "ds41-gateway:test",
            "GATEWAY_MEMORY": "512m",
            "GATEWAY_CPUS": "2",
            "GATEWAY_STRICT": "0",
            "GATEWAY_BLOCK_PRIVATE_URLS": "1",
            "GATEWAY_API_SAMPLING": "1",
            "GATEWAY_IMAGE_TOKEN": "<\uff5cdeepseek_image\uff5c>",
            "GATEWAY_MAX_IMAGES": "600",
            "GATEWAY_MAX_IMAGE_MB": "64",
            "GATEWAY_LOG": "info",
        }
        env.update({key: str(value) for key, value in extra.items()})
        return subprocess.run(
            [str(ROOT / "start.sh")], cwd=ROOT, env=env, text=True, capture_output=True, timeout=60
        )

    def _gateway_tokens(self, out):
        """Decode the [dry-run] gateway docker line back into argv (bash %q escapes the image token)."""
        line = next(l for l in out.splitlines() if l.strip().startswith("docker run -d --name ds41-gateway"))
        split = subprocess.run(
            ["bash", "-c", 'eval "a=($(cat))"; printf "%s\\0" "${a[@]}"'],
            input=line.strip(), text=True, capture_output=True, env={"PATH": f"{self.bin}:/usr/bin:/bin"}, timeout=10,
        )
        self.assertEqual(split.returncode, 0, split.stderr)
        return [token for token in split.stdout.split("\0") if token]

    def test_api_gateway_0_matches_the_recorded_output(self):
        result = self.run_start()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(normalized(result.stdout), normalized(FIXTURE.read_text()))
        self.assertNotIn("ds41-gateway", result.stdout)
        self.assertNotIn("TENSORFOLD_TOKEN_ROUTE", result.stdout)

    def test_api_gateway_1_moves_the_engine_and_prints_the_gateway(self):
        result = self.run_start(
            API_GATEWAY=1,
            TF_DS_API_EFFORTS=1,
            GATEWAY_STRICT=1,
            GATEWAY_BLOCK_PRIVATE_URLS=0,
            GATEWAY_API_SAMPLING=0,
            GATEWAY_MEMORY="768m",
            GATEWAY_CPUS="1.5",
            GATEWAY_LOG="debug",
            API_KEY=1,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        out = result.stdout
        self.assertIn("--host 127.0.0.1 --port 8887", out)
        self.assertIn("TENSORFOLD_TOKEN_ROUTE=1", out)
        self.assertIn("TF_DS_API_EFFORTS=1", out)
        self.assertIn("[dry-run] gateway:", out)
        self.assertIn(
            "--network host --memory 768m --cpus 1.5 --restart on-failure:3 --read-only --tmpfs /tmp",
            out,
        )
        # The gateway carries the digest of its settings, which a later start compares.
        self.assertRegex(out, r"docker run -d --name ds41-gateway --label tf\.recipe\.gateway=[0-9a-f]{16} --network host")
        self.assertIn("-v /tmp/ds41-home/.cache/huggingface/tensorfold-dsv41/45ee5d0f/model:/served:ro", out)
        self.assertIn("-v /tmp/ds41-home/.local/state/dsv41-tensorfold/api-key:/run/secrets/key:ro", out)
        self.assertIn("-e API_KEY_FILE=/run/secrets/key", out)
        self.assertIn("--cap-drop ALL --security-opt no-new-privileges", out)
        # The key file is the invoking user's, mode 600: the gateway must run as that user to read it.
        self.assertIn(f"--user {os.getuid()}:{os.getgid()} -v /tmp/ds41-home/.local/state/dsv41-tensorfold/api-key", out)
        self.assertIn(
            "--listen 0.0.0.0:8888 --engine http://127.0.0.1:8887 "
            "--tokenizer /served/tokenizer.json --served-name DeepSeek-V4.1-Flash-1017 --image-token",
            out,
        )
        gateway_tokens = self._gateway_tokens(out)
        self.assertEqual(gateway_tokens[gateway_tokens.index("--image-token") + 1], "<\uff5cdeepseek_image\uff5c>")
        self.assertIn(
            "--max-model-len 1048576 --max-images 600 --max-image-mb 64 --max-concurrent 3 --max-connections 256 "
            "--max-connections-per-peer 32 --header-timeout 30 --body-timeout 60 --write-timeout 60 --max-prompt-mb 8 --log debug "
            "--default-thinking --strict --block-private-urls=false --api-sampling=false --require-key",
            out,
        )
        self.assertNotIn("-e API_KEY=0", out)
        self.assertNotIn("--host 0.0.0.0 --port 8888", out)

    def test_a_keyed_gateway_refuses_to_run_as_root(self):
        result = self.run_start(API_GATEWAY=1, API_KEY=1, FAKE_UID=0)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("run it as the key's non-root owner", result.stderr)
        # Control: the same settings as a non-root user start.
        result = self.run_start(API_GATEWAY=1, API_KEY=1)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gateway_mode_waits_for_the_engine_on_loopback(self):
        """The engine binds 127.0.0.1 in gateway mode, whatever HOST is: readiness and its smoke test go there."""
        source = (ROOT / "start.sh").read_text()
        self.assertIn('SERVE_HOST=127.0.0.1; SERVE_PORT=$ENGINE_PORT', source)
        self.assertIn('if [[ "$API_GATEWAY" == 1 ]]; then URL="http://127.0.0.1:$ENGINE_PORT"; fi', source)
        self.assertNotIn('URL="http://$API_HOST:$ENGINE_PORT"', source)

    def test_foreground_with_the_gateway_attaches_after_the_gateway_is_up(self):
        """FOREGROUND=1 used to attach right after the launch, so a gateway was never started."""
        source = (ROOT / "start.sh").read_text()
        early = source.index('[[ "${FOREGROUND:-0}" != 1 || "$API_GATEWAY" == 1 ]] || foreground')
        answered = source.index('log "Gateway answered on $API_URL"')
        late = source.index("    foreground\n", answered)
        self.assertLess(early, answered)
        self.assertLess(answered, late)

    def test_without_a_key_the_gateway_keeps_the_image_user(self):
        result = self.run_start(API_GATEWAY=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--user", result.stdout)
        self.assertNotIn("/run/secrets/key", result.stdout)

    def test_gateway_argv_has_no_duplicate_binary_and_flags_are_accepted(self):
        """Docker runs ENTRYPOINT + the arguments after the image; argv[1] must be --listen, not the binary again."""
        result = self.run_start(API_GATEWAY=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        at = lines.index("[dry-run] gateway:")
        line = lines[at + 1].strip()
        split = subprocess.run(
            ["bash", "-c", 'eval "a=($(cat))"; printf "%s\\0" "${a[@]}"'],
            input=line, text=True, capture_output=True, env={"PATH": f"{self.bin}:/usr/bin:/bin"}, timeout=10,
        )
        self.assertEqual(split.returncode, 0, split.stderr)
        tokens = [token for token in split.stdout.split("\0") if token]
        image = next(i for i, token in enumerate(tokens) if token.startswith("ds41-gateway:"))
        args = tokens[image + 1:]
        dockerfile = (ROOT / "gateway" / "Dockerfile").read_text()
        entrypoint = json.loads(re.search(r"^ENTRYPOINT\s+(\[.*\])\s*$", dockerfile, re.M).group(1))
        argv = entrypoint + args
        self.assertEqual(argv[0], "ds41-gateway")
        self.assertEqual(argv[1], "--listen", f"argv={argv[:4]}")
        self.assertNotIn("ds41-gateway", args)

        flags = [token.split("=", 1)[0] for token in args if token.startswith("--")]
        binary = Path(os.environ.get("DS41_GATEWAY_BIN", ROOT / "gateway" / "target" / "release" / "ds41-gateway"))
        if binary.is_file():
            checked = subprocess.run([str(binary), *args, "--help"], text=True, capture_output=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            help_text = checked.stdout + checked.stderr
            for flag in flags:
                self.assertIn(flag, help_text)
        else:
            source = (ROOT / "gateway" / "src" / "main.rs").read_text()
            accepted = set()
            for match in re.finditer(r"#\[arg\(([^)]*)\)\]\s*\n\s*(\w+)\s*:", source):
                attrs, field = match.groups()
                if "long" not in attrs:
                    continue
                long_name = re.search(r'long\s*=\s*"([^"]+)"', attrs)
                accepted.add("--" + (long_name.group(1) if long_name else field.replace("_", "-")))
            for flag in flags:
                self.assertIn(flag, accepted)

    def test_api_gateway_banner_is_the_contract_phrase(self):
        """DRY_RUN stops before the banner, so pin its exact phrase in the source (the contract's wording)."""
        source = (ROOT / "start.sh").read_text()
        self.assertIn(
            "(gateway: DeepSeek API parity on; engine on 127.0.0.1:$ENGINE_PORT)",
            source,
        )
        self.assertIn('|| echo "(model: $SERVED_NAME)"', source)

    def test_token_route_without_the_gateway_only_sets_the_engine_variable(self):
        result = self.run_start(TOKEN_ROUTE=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TENSORFOLD_TOKEN_ROUTE=1", result.stdout)
        self.assertIn("--host 0.0.0.0 --port 8888", result.stdout)
        self.assertNotIn("ds41-gateway", result.stdout)

    def test_invalid_values_are_rejected(self):
        cases = [
            ({"API_GATEWAY": 2}, "API_GATEWAY is 0 or 1"),
            ({"TOKEN_ROUTE": 2}, "TOKEN_ROUTE is 0 or 1"),
            ({"GATEWAY_STRICT": 2}, "GATEWAY_STRICT is 0 or 1"),
            ({"GATEWAY_BLOCK_PRIVATE_URLS": 2}, "GATEWAY_BLOCK_PRIVATE_URLS is 0 or 1"),
            ({"GATEWAY_API_SAMPLING": 2}, "GATEWAY_API_SAMPLING is 0 or 1"),
            ({"TF_DS_API_EFFORTS": 2}, "TF_DS_API_EFFORTS is empty, 0 or 1"),
            ({"ENGINE_PORT": 0}, "ENGINE_PORT is 1 to 65535"),
            ({"ENGINE_PORT": 70000}, "ENGINE_PORT is 1 to 65535"),
            ({"ENGINE_PORT": 8888}, "ENGINE_PORT and PORT are both 8888"),
            ({"GATEWAY_MEMORY": "512"}, "GATEWAY_MEMORY is a size"),
            ({"GATEWAY_MEMORY": "1.5x"}, "GATEWAY_MEMORY is a size"),
            ({"GATEWAY_MEMORY": "0m"}, "GATEWAY_MEMORY is a size over zero"),
            ({"GATEWAY_MEMORY": "0.0g"}, "GATEWAY_MEMORY is a size over zero"),
            ({"GATEWAY_CPUS": 0}, "GATEWAY_CPUS is a positive number"),
            ({"GATEWAY_LOG": "verbose"}, "GATEWAY_LOG is debug, info, warn or error"),
            ({"GATEWAY_IMAGE_TOKEN": ""}, "GATEWAY_IMAGE_TOKEN is empty"),
            ({"GATEWAY_MAX_IMAGES": 0}, "GATEWAY_MAX_IMAGES is a positive count"),
            ({"GATEWAY_MAX_IMAGE_MB": 0}, "GATEWAY_MAX_IMAGE_MB is a positive count"),
            ({"GATEWAY_MAX_CONNECTIONS": 0}, "GATEWAY_MAX_CONNECTIONS is a positive count"),
            ({"GATEWAY_BODY_TIMEOUT": "1.5"}, "GATEWAY_BODY_TIMEOUT is a positive number of seconds"),
            ({"GATEWAY_MAX_CONNECTIONS_PER_PEER": 0}, "GATEWAY_MAX_CONNECTIONS_PER_PEER is a positive count"),
            ({"GATEWAY_HEADER_TIMEOUT": "x"}, "GATEWAY_HEADER_TIMEOUT is a positive number of seconds"),
            ({"GATEWAY_WRITE_TIMEOUT": 0}, "GATEWAY_WRITE_TIMEOUT is a positive number of seconds"),
            ({"GATEWAY_MAX_PROMPT_MB": 0}, "GATEWAY_MAX_PROMPT_MB is a positive count"),
        ]
        for extra, message in cases:
            with self.subTest(extra=extra):
                env = {"API_GATEWAY": 1}
                env.update(extra)
                result = self.run_start(**env)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(message, result.stderr)


    def test_gateway_recover_starts_a_stopped_gateway_or_fails(self):
        """gateway_recover (the already-running check and FOREGROUND's watcher): start a stopped gateway and wait for
        its health; a removed gateway, one docker cannot start or one that never gets healthy is a failure."""
        import tempfile

        text = (ROOT / "start.sh").read_text()
        function = re.search(r"^gateway_recover\(\) \{.*?^\}", text, re.S | re.M).group(0)
        with tempfile.TemporaryDirectory(prefix="ds41-recover-") as tmp:
            tmp = Path(tmp)
            stubs = tmp / "bin"
            stubs.mkdir()
            calls = tmp / "calls"
            bodies = {
                "docker": f"""#!/usr/bin/env bash
echo "docker $*" >> {calls}
case "$1" in
  inspect) [[ -e {tmp}/exists ]] ;;
  start) [[ -e {tmp}/startable ]] ;;
  *) exit 1 ;;
esac
""",
                "curl": f"#!/usr/bin/env bash\necho curl >> {calls}\n[[ -e {tmp}/healthy ]]\n",
                "sleep": "#!/usr/bin/env bash\nexit 0\n",
            }
            for name, body in bodies.items():
                (stubs / name).write_text(body)
                (stubs / name).chmod(0o755)

            def recover(*flags):
                for flag in ("exists", "startable", "healthy"):
                    (tmp / flag).unlink(missing_ok=True)
                for flag in flags:
                    (tmp / flag).touch()
                calls.write_text("")
                result = subprocess.run(
                    ["bash", "-c", function + "\ngateway_recover http://127.0.0.1:8888"],
                    env={"PATH": f"{stubs}:/usr/bin:/bin"},
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.returncode, calls.read_text().splitlines()

            code, seen = recover("exists", "startable", "healthy")
            self.assertEqual(code, 0, seen)
            self.assertIn("docker start ds41-gateway", seen)
            code, seen = recover()
            self.assertEqual(code, 1, seen)
            self.assertNotIn("docker start ds41-gateway", seen)
            code, seen = recover("exists")
            self.assertEqual(code, 1, seen)
            self.assertNotIn("curl", seen)
            code, seen = recover("exists", "startable")
            self.assertEqual(code, 1, seen)
            self.assertEqual(seen.count("curl"), 30)


    def test_foreground_exits_with_a_failure_when_the_watcher_stops_the_server(self):
        """FOREGROUND's watcher stops rank 0 when the gateway cannot be started again or rank 1 is gone; the exit code
        must say so even though stopping rank 0 ends `docker logs -f` and the watcher is killed before it exits."""
        import tempfile

        text = (ROOT / "start.sh").read_text()
        functions = "\n".join(
            re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M).group(0)
            for name in ("gateway_remove_keeping_log", "gateway_recover", "foreground")
        )
        functions += "\n" + "\n".join(
            line for line in text.splitlines() if re.match(r"^(running_here|running_worker|running_gateway)\(\)", line)
        )
        with tempfile.TemporaryDirectory(prefix="ds41-foreground-") as tmp:
            tmp = Path(tmp)
            stubs = tmp / "bin"
            stubs.mkdir()
            bodies = {
                "docker": f"""#!/usr/bin/env bash
t={tmp}
rank=rank0; [[ -n "${{WORKER_SIDE:-}}" ]] && rank=rank1
case "$1" in
  inspect)
    if [[ "$2" == -f ]]; then
      case "$3" in
        *Running*) if [[ "$4" == ds41-gateway ]]; then [[ -e $t/gw_running ]] && echo true || echo false
                   else [[ -e $t/$rank ]] && echo true || echo false; fi ;;
        *ExitCode*) echo 0 ;;
      esac
    else [[ -e $t/gw_exists ]]; fi ;;
  start) [[ -e $t/gw_startable ]] && touch $t/gw_running ;;
  stop) rm -f $t/$rank ;;
  rm) rm -f $t/gw_exists $t/gw_running ;;
  logs) while [[ -e $t/rank0 ]]; do /bin/sleep 0.05; done ;;
esac
""",
                "curl": f"#!/usr/bin/env bash\n[[ -e {tmp}/gw_running ]]\n",
                "sleep": "#!/usr/bin/env bash\nexec /bin/sleep 0.05\n",
            }
            for name, body in bodies.items():
                (stubs / name).write_text(body)
                (stubs / name).chmod(0o755)
            harness = f"""
warn() {{ :; }}
worker_host() {{ echo worker; }}
worker() {{ shift; WORKER_SIDE=1 bash -c "$*"; }}
save_log() {{ /bin/sleep 0.5; }}   # saving the gateway's log takes a while, as docker logs does
API_GATEWAY=1 API_URL=http://127.0.0.1:8888 CONTAINER_NAME=rank STATE_DIR={tmp} LOG_DIR={tmp} LOG_KEEP=1
{functions}
"""

            def run(state, extra=""):
                for flag in ("rank0", "rank1", "gw_exists", "gw_running", "gw_startable"):
                    (tmp / flag).unlink(missing_ok=True)
                for flag in state:
                    (tmp / flag).touch()
                result = subprocess.run(
                    ["bash", "-c", harness + extra + "\nforeground"],
                    env={"PATH": f"{stubs}:/usr/bin:/bin"},
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                return result.returncode

            # The gateway exited and cannot be started again: the server stops with a failure.
            self.assertEqual(run(["rank0", "rank1", "gw_exists"]), 1)
            self.assertFalse((tmp / "foreground.failed").exists())
            # Rank 1 is gone: the same.
            self.assertEqual(run(["rank0", "gw_exists", "gw_running"]), 1)
            # The gateway exited and is started again: the server goes on until rank 0 ends cleanly, then 0.
            self.assertEqual(run(["rank0", "rank1", "gw_exists", "gw_startable"], f"( /bin/sleep 1; rm -f {tmp}/rank0 ) &"), 0)
            # Control: all up, rank 0 ends cleanly by itself: 0.
            self.assertEqual(run(["rank0", "rank1", "gw_exists", "gw_running"], f"( /bin/sleep 0.5; rm -f {tmp}/rank0 ) &"), 0)


    def test_gateway_identity_changes_with_the_gateway_settings(self):
        """The gateway's label: the same settings give the same digest; another setting or another key file, another."""
        text = (ROOT / "start.sh").read_text()
        function = re.search(r"^gateway_identity\(\) \{.*?^\}", text, re.S | re.M).group(0)

        def identity(**extra):
            env = {"PATH": "/usr/bin:/bin", "GATEWAY_STRICT": "0", "GATEWAY_MAX_CONCURRENT": "3", "HOST": "0.0.0.0",
                   "PORT": "8888", "API_KEY": "0", "GATEWAY_RUN": "ignored"}
            env.update(extra)
            script = "".join(f"{k}={v!r}; " for k, v in env.items()) + "\n" + function + "\ngateway_identity"
            return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.strip()

        base = identity()
        self.assertRegex(base, r"^[0-9a-f]{16}$")
        self.assertEqual(base, identity())
        self.assertEqual(base, identity(GATEWAY_RUN="other"))  # the command arrays are left out
        self.assertNotEqual(base, identity(GATEWAY_STRICT="1"))
        self.assertNotEqual(base, identity(GATEWAY_MAX_CONCURRENT="4"))
        self.assertNotEqual(base, identity(API_KEY="1"))
        # The key the gateway loaded: rotating it in the same file is another gateway.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="ds41-identity-") as tmp:
            key = Path(tmp) / "key"
            key.write_text("old-key\n")
            before = identity(API_KEY="1", API_KEY_FILE=str(key))
            self.assertEqual(before, identity(API_KEY="1", API_KEY_FILE=str(key)))
            key.write_text("new-key\n")
            self.assertNotEqual(before, identity(API_KEY="1", API_KEY_FILE=str(key)))


if __name__ == "__main__":
    unittest.main()
