#!/usr/bin/env bash
# Tests for scripts/check-pins.sh: a local fixture of the Hugging Face and GitHub endpoints, so every failure has a case and
# no network is used. Run: bash scripts/test_check_pins.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
root=$(mktemp -d); srv=""
trap '[[ -n "$srv" ]] && kill "$srv" 2>/dev/null; rm -rf "$root"' EXIT
www="$root/www"

# fixture <path> <<< json: one served file (query strings are ignored by http.server, as the API ignores them)
fixture() { mkdir -p "$www/$(dirname "$1")"; cat > "$www/$1"; }

SHA_OK=1111111111111111111111111111111111111111
SHA_NOTOKEN=4444444444444444444444444444444444444444
SHA_BADINDEX=5555555555555555555555555555555555555555
SHA_DEAD=9999999999999999999999999999999999999999
SHA_ENGRAM=2222222222222222222222222222222222222222
SHA_ENGRAM_BAD=6666666666666666666666666666666666666666
SHA_GIT=3333333333333333333333333333333333333333
SHA_NOGIT=8888888888888888888888888888888888888888

# the checkpoint: a complete snapshot, one without tokenizer.json, one whose index names a shard the snapshot lacks
checkpoint() {  # checkpoint <sha> <has-tokenizer> <shard-in-index>
  local files='[{"rfilename":"config.json","size":900},{"rfilename":"model.safetensors.index.json","size":1200},'
  files+='{"rfilename":"model-00001-of-00002.safetensors","size":6500000000},'
  files+='{"rfilename":"model-00002-of-00002.safetensors","size":6400000000}'
  [[ "$2" == 1 ]] && files+=',{"rfilename":"tokenizer.json","size":6367257}'
  fixture "models/test-org/model/revision/$1" <<< "{\"sha\":\"$1\",\"siblings\":$files]}"
  fixture "test-org/model/resolve/$1/model.safetensors.index.json" \
    <<< "{\"weight_map\":{\"a\":\"model-00001-of-00002.safetensors\",\"b\":\"$3\"}}"
}
checkpoint "$SHA_OK" 1 model-00002-of-00002.safetensors
checkpoint "$SHA_NOTOKEN" 0 model-00002-of-00002.safetensors
checkpoint "$SHA_BADINDEX" 1 model-00009-of-00009.safetensors

# Engram: the two shards at the sizes scripts/config.sh records
fixture "models/test-org/engram/revision/$SHA_ENGRAM" <<< "{\"sha\":\"$SHA_ENGRAM\",\"siblings\":[
 {\"rfilename\":\"config.json\",\"size\":900},
 {\"rfilename\":\"model.safetensors.index.json\",\"size\":1200},
 {\"rfilename\":\"model-00047-of-00048.safetensors\",\"size\":101535150936},
 {\"rfilename\":\"model-00048-of-00048.safetensors\",\"size\":101537926640}]}"
# and one whose shard 47 is not the size the card lists
fixture "models/test-org/engram/revision/$SHA_ENGRAM_BAD" <<< "{\"sha\":\"$SHA_ENGRAM_BAD\",\"siblings\":[
 {\"rfilename\":\"config.json\",\"size\":900},
 {\"rfilename\":\"model.safetensors.index.json\",\"size\":1200},
 {\"rfilename\":\"model-00047-of-00048.safetensors\",\"size\":1},
 {\"rfilename\":\"model-00048-of-00048.safetensors\",\"size\":101537926640}]}"
# the git API: one commit that exists
fixture "gh/repos/test-org/repo/commits/$SHA_GIT" <<< "{\"sha\":\"$SHA_GIT\"}"

python3 - "$www" "$root/port" <<'PY' &
import http.server, socketserver, sys, threading, functools, pathlib
served = pathlib.Path(sys.argv[1])
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
handler = functools.partial(Quiet, directory=str(served))
with socketserver.TCPServer(("127.0.0.1", 0), handler) as srv:
    pathlib.Path(sys.argv[2]).write_text(str(srv.server_address[1]))
    srv.serve_forever()
PY
srv=$!
for _ in $(seq 50); do [[ -s "$root/port" ]] && break; sleep 0.1; done
port=$(cat "$root/port")
base="http://127.0.0.1:$port"

pass=0; fail=0
# run <name> <expected exit> <expected text> [VAR=value ...]
run() {
  local name=$1 want=$2 text=$3; shift 3
  local out status
  set +e
  out=$(env HF_API="$base/models" HF_RESOLVE="$base" GITHUB_API="$base/gh" \
        MODEL_ID=test-org/model MODEL_REVISION=$SHA_OK \
        ENGRAM_ID=test-org/engram ENGRAM_REVISION=$SHA_ENGRAM \
        TF_REPO=https://github.com/test-org/repo.git TF_REF=$SHA_GIT INPUTS_REF=$SHA_GIT \
        KIT_REPO=https://github.com/test-org/repo.git KIT_REF=$SHA_GIT "$@" \
        ./scripts/check-pins.sh 2>&1)
  status=$?
  set -e
  if [[ "$status" == "$want" && "$out" == *"$text"* ]]; then
    printf 'PASS %-34s exit %s, %s\n' "$name" "$status" "$text"; pass=$((pass + 1))
  else
    printf 'FAIL %-34s exit %s (wanted %s), wanted %s\n%s\n' "$name" "$status" "$want" "$text" "$out"; fail=$((fail + 1))
  fi
}

run "every pin resolves"                0 "every pin resolves"
run "a checkpoint revision that is gone" 1 "revision not found"          MODEL_REVISION=$SHA_DEAD
run "a snapshot without tokenizer.json"  1 "lacks tokenizer.json"        MODEL_REVISION=$SHA_NOTOKEN
run "an index naming an absent shard"    1 "shards but 1 missing"        MODEL_REVISION=$SHA_BADINDEX
run "an Engram shard at the wrong size"  1 "the card lists 101535150936" ENGRAM_REVISION=$SHA_ENGRAM_BAD
run "a git commit that is gone"          1 "commit not found"            TF_REF=$SHA_NOGIT

echo "$pass passed, $fail failed"
(( fail == 0 ))
