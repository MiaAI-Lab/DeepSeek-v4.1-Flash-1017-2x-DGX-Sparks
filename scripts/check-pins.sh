#!/usr/bin/env bash
# Resolve every pin in scripts/config.sh before prepare.sh builds an image or downloads hundreds of GB: the checkpoint's
# and Engram's revisions on Hugging Face, the small files prepare.sh reads inside them, the checkpoint's index against its
# shards, Engram's two shards against the sizes the card lists, and the three git commits the image and the kit come from.
# One HTTPS request per check; nothing is downloaded whole.
#
# A dead pin used to surface as an `hf download` failure minutes in, or as a tokenizer missing from a snapshot that does
# exist. This names the pin that is wrong, and why, before anything is fetched.
#
# PIN_CHECK=0 in scripts/local.sh (or the environment) skips it; prepare.sh calls it at the end of its preflight.
# GITHUB_TOKEN, when set, raises the api.github.com rate limit; a rate-limited git check warns rather than fails. HF_API, GITHUB_API and PIN_TIMEOUT_S override the
# endpoints and the per-request timeout.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

case "${1:-}" in
  -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
  "") ;;
  *) die "unknown argument: $1" ;;
esac

HF_API="${HF_API:-https://huggingface.co/api/models}"
HF_RESOLVE="${HF_RESOLVE:-https://huggingface.co}"
GH_API="${GITHUB_API:-https://api.github.com}"
TMO="${PIN_TIMEOUT_S:-20}"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/check-pins.XXXXXX"); trap 'rm -rf -- "$TMP"' EXIT
fails=0
warns=0
HF_REV=""   # the commit hf_revision resolved

# http_get <url> <out>: the status code, "000" when the request itself failed (offline, DNS, TLS).
http_get() {
  local url=$1 out=$2 auth=()
  [[ -n "${GITHUB_TOKEN:-}" && "$url" == "$GH_API"/* ]] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
  curl -sS --max-time "$TMO" -L -o "$out" -w '%{http_code}' "${auth[@]}" "$url" 2>"$TMP/curl.err" || true
}

# say <pin> <commit> <state> <detail>: one aligned line per pin; a FAIL increments $fails.
say() {
  printf '  %-10s %-10s %-4s %s\n' "$1" "${2:0:10}" "$3" "$4"
  [[ "$3" == FAIL ]] && fails=$((fails + 1)) || true
  [[ "$3" == WARN ]] && warns=$((warns + 1)) || true
}

# json_missing <file> <name>...: prints the names not in the revision's file list (empty when all are there).
json_missing() {
  python3 -I - "$@" <<'PY'
import json, sys
have = {s["rfilename"] for s in json.load(open(sys.argv[1])).get("siblings", [])}
print(" ".join(n for n in sys.argv[2:] if n not in have))
PY
}

# hf_revision <repo> <rev> <pin>: writes the revision's metadata (with sizes) to $TMP/<pin>.json and sets $HF_REV to the
# resolved commit; an empty <rev> follows the repo's main, as prepare.sh does. A missing revision is reported and returns 1.
# (The commit comes back in a variable, not on stdout: the lines above have to reach the terminal, not a $(...) capture.)
hf_revision() {
  local repo=$1 rev=$2 pin=$3 status
  HF_REV=""
  if [[ -n "$rev" ]]; then
    status=$(http_get "$HF_API/$repo/revision/$rev?blobs=true" "$TMP/$pin.json")
    if [[ "$status" != 200 ]]; then
      say "$pin" "$rev" FAIL "revision not found on Hugging Face (HTTP $status): the repo rewrote its history, or the pin is wrong"
      return 1
    fi
    HF_REV=$rev
  else
    status=$(http_get "$HF_API/$repo?blobs=true" "$TMP/$pin.json")
    if [[ "$status" != 200 ]]; then
      say "$pin" main FAIL "cannot read the repo (HTTP $status); is $HF_API reachable?"
      return 1
    fi
    say "$pin" main WARN "no revision pinned: prepare.sh follows main (a pin is what makes a run reproducible)"
    HF_REV=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha"])' "$TMP/$pin.json")
  fi
}

# git_pin <pin> <repo url> <sha>: the commit has to exist in the repo the image or the kit is built from.
git_pin() {
  local pin=$1 url=$2 sha=$3 repo status
  repo=${url#https://github.com/}; repo=${repo#git@github.com:}; repo=${repo%.git}
  status=$(http_get "$GH_API/repos/$repo/commits/$sha" "$TMP/git-$pin.json")
  case "$status" in
    200) say "$pin" "$sha" OK "$repo" ;;
    404|422) say "$pin" "$sha" FAIL "$repo: commit not found (HTTP $status)" ;;
    # a rate limit (60 an hour without a token) says nothing about the pin, and git clones from another endpoint with its
    # own limits, so this warns: GITHUB_TOKEN raises the limit, and the build resolves the commit anyway
    *) say "$pin" "$sha" WARN "$repo: could not ask api.github.com (HTTP $status); the build fetches it" ;;
  esac
}

log "the pins in scripts/config.sh (one request each; nothing is downloaded)"

# ---------------------------------------------------------------- the checkpoint: revision, small files, index vs shards
if hf_revision "$MODEL_ID" "$MODEL_REVISION" checkpoint; then
  rev=$HF_REV
  missing=$(json_missing "$TMP/checkpoint.json" config.json tokenizer.json model.safetensors.index.json)
  if [[ -n "$missing" ]]; then
    say checkpoint "$rev" FAIL "the snapshot lacks $missing (prepare.sh reads them)"
  else
    idx_status=$(http_get "$HF_RESOLVE/$MODEL_ID/resolve/$rev/model.safetensors.index.json" "$TMP/checkpoint.index.json")
    if [[ "$idx_status" != 200 ]]; then
      say checkpoint "$rev" FAIL "model.safetensors.index.json did not download (HTTP $idx_status)"
    elif line=$(python3 -I - "$TMP/checkpoint.index.json" "$TMP/checkpoint.json" <<'PY'
import json, sys
index = json.load(open(sys.argv[1]))
meta = json.load(open(sys.argv[2]))
want = set(index.get("weight_map", {}).values())
have = {s["rfilename"] for s in meta.get("siblings", [])}
missing = sorted(want - have)
print(f"{len(want)} shards, index matches" if not missing else f"{len(want)} shards but {len(missing)} missing: {', '.join(missing[:3])}")
sys.exit(1 if missing else 0)
PY
    ); then
      say checkpoint "$rev" OK "$line; tokenizer.json and config.json present"
    else
      say checkpoint "$rev" FAIL "$line"
    fi
  fi
fi

# ---------------------------------------------------------------- Engram: revision, the two shards, their sizes
if hf_revision "$ENGRAM_ID" "$ENGRAM_REVISION" engram; then
  rev=$HF_REV
  missing=$(json_missing "$TMP/engram.json" config.json model.safetensors.index.json $ENGRAM_SHARDS)
  if [[ -n "$missing" ]]; then
    say engram "$rev" FAIL "the snapshot lacks $missing"
  else
    want=(); for f in $ENGRAM_SHARDS; do want+=("$f=${ENGRAM_SHARD_SIZE[$f]}"); done
    if line=$(python3 -I - "$TMP/engram.json" "${want[@]}" <<'PY'
import json, sys
sizes = {s["rfilename"]: s.get("size") for s in json.load(open(sys.argv[1])).get("siblings", [])}
bad = []
for pair in sys.argv[2:]:
    name, want = pair.rsplit("=", 1)
    if sizes.get(name) != int(want):
        bad.append(f"{name}: {sizes.get(name)} B, the card lists {want} B")
print("both shards, sizes match the card" if not bad else "; ".join(bad))
sys.exit(1 if bad else 0)
PY
    ); then
      say engram "$rev" OK "$line"
    else
      say engram "$rev" FAIL "$line"
    fi
  fi
fi

# ---------------------------------------------------------------- the engine, the inputs' tools and the kit
git_pin engine "$TF_REPO" "$TF_REF"
git_pin inputs "$TF_REPO" "$INPUTS_REF"
git_pin kit "$KIT_REPO" "$KIT_REF"

if (( fails > 0 )); then
  die "$fails pin(s) do not resolve (above): nothing was built or downloaded. Fix the pin in scripts/config.sh or scripts/local.sh, or set PIN_CHECK=0 to skip this check"
fi
if (( warns > 0 )); then log "every pin resolves ($warns advisory check(s) could not run; see WARN above)"; else log "every pin resolves"; fi
