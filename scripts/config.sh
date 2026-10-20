# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`
#   2. scripts/local.sh (this setup's own values, above all WORKER; sourced as bash), then ./.env (KEY=value lines, read,
#      never run): both are yours, not the repository's; where both set a key, local.sh wins
#   3. the defaults below
# The loading of local.sh and .env is the GLM-5.3-Flash recipe's (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
_cfg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$_cfg_root/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _n; do _cfg_env[$_n]=${!_n}; done < <(compgen -e)
  source "$_cfg_root/scripts/local.sh"
  # the environment wins over local.sh: put back any variable it had that local.sh changed
  for _n in "${!_cfg_env[@]}"; do [[ "${!_n-}" == "${_cfg_env[$_n]}" ]] || export "$_n=${_cfg_env[$_n]}"; done
  unset _cfg_env
fi
if [[ -f "$_cfg_root/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$_cfg_root/.env"
fi
unset _n _line _key _value

# ---------------------------------------------------------------- the two Sparks
# This machine serves rank 0 and the API; WORKER (an ssh target, key-based) is rank 1. The two are cabled directly
# through their ConnectX-7 ports (scripts/nodes.sh finds the link).
TP=2
WORKER="${WORKER:-}"; FABRIC_PEER="${FABRIC_PEER:-}"
MASTER_PORT="${MASTER_PORT:-29620}"  # the ranks' rendezvous port (keep it on the private link)
# The rendezvous address: this node's address on the link to the worker (scripts/nodes.sh sets it from the link when it
# is empty).
MASTER_ADDR="${MASTER_ADDR:-}"
SOCKET_IFNAME="${SOCKET_IFNAME:-}"     # NCCL's bootstrap socket's netdev on both nodes (default: the link's)

# ---------------------------------------------------------------- the checkpoint, the engine, the kernel kit
# The weights: Mia's AI Lab's EAQ v1 EXL3 quantization of DeepSeek-V4.1-Flash-1017 at 2.9 bits a weight (39 shards, 196 GiB;
# the vision tower and the DSpark draft blocks are in it). Its Engram n-gram tables are not: they are read unquantized from
# the base model's shards 47 and 48 (ENGRAM_ID, ~203 GB), which this Spark downloads beside it.
MODEL_ID="${MODEL_ID:-Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw}"
_rev=""; [[ "$MODEL_ID" == Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw ]] && _rev=6d54be4fca338d79b17c3b03cc9a423bb3ac6718
MODEL_REVISION="${MODEL_REVISION-$_rev}"
# Keys' matching Mia EAQ 2.9 bpw overlay (not a standalone checkpoint).
export ABLIT="${ABLIT:-0}"
ABLIT_ID=drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated
ABLIT_REVISION="${ABLIT_REVISION:-0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0}"
ABLIT_SIDECAR=exl3_wo_b_l10_35_mia29.safetensors
ENGRAM_ID="${ENGRAM_ID:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
_erev=""; [[ "$ENGRAM_ID" == deepseek-ai/DeepSeek-V4.1-Flash-1017 ]] && _erev=1bae0e1088993b9b82de08f59de5be5a34e42170
ENGRAM_REVISION="${ENGRAM_REVISION-$_erev}"
# Only these files of the base model are downloaded (the card's sha256 and sizes: scripts/prepare.sh checks them)
ENGRAM_SHARDS="${ENGRAM_SHARDS:-model-00047-of-00048.safetensors model-00048-of-00048.safetensors}"
# The 1017 release's two shards' sizes (the model card's, as scripts/prepare.sh checks them): HEAD_ENGRAM_DIR is checked against them
declare -A ENGRAM_SHARD_SIZE=([model-00047-of-00048.safetensors]=101535150936 [model-00048-of-00048.safetensors]=101537926640)
# The engine: BertholomusAI's TensorFold fork, branch deepseek-v41-zig (the DeepSeek-V4.1-Flash family on TensorFold's
# native Zig engine, two nodes), at the head of that branch; it serves both ranks.
TF_REPO="${TF_REPO:-https://github.com/bertholomus/TensorFold.git}"
TF_REF="${TF_REF:-629c524a0ea57512a5342abb498d1b7c25353343}"   # the branch tip of 2026-10-19 (was 3ff3b2ac; docs/zig-engine.md, "The pin")
# The inputs' tools: the same fork's branch deepseek-v41-zig-tp4, at the commit of the four-node recipe v0.1.1. It adds
# what the two-node branch lacks: tf-dsv41-rank-cache (a rank's weight file from the checkpoint alone, no GPU, no
# Python) and the standalone token-map and RoPE-table scripts. Only those are taken; it does not serve here.
INPUTS_REF="${INPUTS_REF:-59402086381a42c74d53053443224eec2790b4c5}"
# The family's kernel kit (its Triton kernels as captured binaries, the EXL3 kernels, Engram's constants, the vision
# tower's attention kernel): the kit/ of BertholomusAI's two-node recipe, checked against its MANIFEST; the RoPE tables
# and the image routing bias are made from the checkpoint.
KIT_REPO="${KIT_REPO:-https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10.git}"
KIT_REF="${KIT_REF:-abe07c7d509a0199cc7fec15263be2aa52ac652f}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"   # zig-aarch64-linux-0.17.0.tar.xz
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
image_hash() {
  ( cd "$_cfg_root" || exit 1
    cat zig/patches/*.patch 2>/dev/null
    cat zig/Dockerfile
    echo "$TF_REPO $TF_REF $INPUTS_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE"
  ) | sha256sum | cut -c1-12
}
TAG_BASE="exl3-tp2"             # the image tag's prefix: <TAG_BASE>-<image hash>
IMAGE="${IMAGE:-tensorfold-dsv41:$TAG_BASE-$(image_hash)}"
# A published image of these pins, if any (scripts/publish-image.sh pushes it; the registry handling is the GLM-5.3-Flash
# recipe's): prepare.sh pulls $GHCR_IMAGE:<TAG_BASE>-<hash>, or, when IMAGE_TAG names that tag and IMAGE_DIGEST is set,
# the pinned digest (a tag can be moved, a digest cannot), and builds locally when there is none. None is published yet.
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/deepseek-v4.1-flash-1017-2x-dgx-sparks-tensorfold}"
IMAGE_TAG="${IMAGE_TAG:-}"
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
prebuilt_image() {
  local tag="${TAG_BASE}-$(image_hash)"
  if [[ "$tag" == "$IMAGE_TAG" && -n "$IMAGE_DIGEST" ]]; then echo "$GHCR_IMAGE@$IMAGE_DIGEST"; else echo "$GHCR_IMAGE:$tag"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-dsv41-1017-exl3-tf}"   # the same name on both Sparks

# ---------------------------------------------------------------- serving
SERVED_NAME="${SERVED_NAME:-DeepSeek-V4.1-Flash-1017}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# A request's positions at most (prompt and reply). The model's window is 1,048,576 tokens. The streams share a pool of
# CONTEXT positions (rank 1's --pool is rank 0's --context), so PARALLEL streams at the full window take turns.
CONTEXT="${CONTEXT:-1048576}"
PARALLEL="${PARALLEL:-4}"              # requests at once (the engine's streams, TF_DS_STREAMS; the family measured 4)
ROUND_ROWS="${ROUND_ROWS:-}"           # a decode round's rows at most, 16 to 48 (empty: the engine's 16)
MAX_TOKENS="${MAX_TOKENS:-32768}"      # a reply's budget when a request sets none
THINKING="${THINKING:-1}"              # think first unless a request turns it off
# The reasoning effort when a request names none: low, medium, high or xhigh (empty: the chat template's default)
REASONING_EFFORT="${REASONING_EFFORT:-}"
# The family's measured sampling (BertholomusAI's two-node launch): temperature 1.0, top_p 0.95, from the 20 likeliest
# tokens (the family's exact sampler needs a top-k). A request's own values win.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
DRAFTS="${DRAFTS:-1}"                  # the checkpoint's own DSpark draft blocks (exact: drafted replies equal plain)
# The engine's settings as BertholomusAI's served two-node launch has them: CUDA graphs, the mixes' side stream, the
# paced L2 prefetch, Engram reads by Linux AIO. RDMA: the decode-size exchanges between the nodes over RoCE instead of
# NCCL, on each node's RoCE devices (scripts/nodes.sh finds them).
RDMA="${RDMA:-1}"
GRAPHS="${GRAPHS:-1}"
SIDE="${SIDE:-1}"
PREFETCH="${PREFETCH:-1}"
ENGRAM_AIO="${ENGRAM_AIO:-1}"
# TF_DS_REPLAY_FLOOR=1 is the family's served setting: a last prompt chunk of 16 rows or fewer reads its window from the
# replay cut, not from the previous request's keys. It goes to both ranks.
REPLAY_FLOOR="${REPLAY_FLOOR:-1}"
# zig/patches/0005 (issue #3; pull request #5 by Coinupbtc): milliseconds a long prompt pass gives, between two of its chunks, to decode
# rounds for the streams already live (and to their cancels): TF_DS_PREFILL_INTERLEAVE_MS, read by rank 0 only. 0: the pass runs to
# its end first, as before. 250 is what PR #5 and this recipe measured. Each such stop costs the long prompt about the time given.
PREFILL_INTERLEAVE_MS="${PREFILL_INTERLEAVE_MS:-250}"

# ---------------------------------------------------------------- files
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# What the ranks read besides the checkpoint: rank 0's weight file, Engram's token map and the kernel kit, under HF_CACHE
# so that one NFS export serves the worker too.
DATA_SUB="tensorfold-dsv41/${MODEL_REVISION:0:8}"
[[ "$ABLIT" != 1 ]] || DATA_SUB+="-ablit-${ABLIT_REVISION:0:8}"
DATA_DIR="$HF_CACHE/$DATA_SUB"
KIT_SUB="tensorfold-dsv41/kit-${KIT_REF:0:8}"
# The worker reads this Spark's HF_CACHE over NFS (the recipe's only mode: one copy of the 196 GiB checkpoint, here): a
# read-only docker volume NFS_VOLUME on the worker that prepare.sh creates (no sudo there). This Spark must export
# NFS_PATH (default: HF_CACHE; "/" for an NFSv4 export with fsid=0) to the worker's address on its link. NFS_SERVER: the
# head's address the worker mounts from (default: the head's address on the link, found by scripts/nodes.sh).
NFS_PATH="${NFS_PATH:-$HF_CACHE}"
NFS_SERVER="${NFS_SERVER:-}"
NFS_VOLUME="${NFS_VOLUME:-dsv41-1017-exl3-hf}"
# On the worker's own disk (paths on the worker): Engram's tables (a decode round reads ~70 random 4 KiB rows from
# them; over NFS that took most of a round), and its weight file (~106 GB; written there by prepare.sh from the
# checkpoint over NFS, so it never crosses the network twice).
WORKER_ENGRAM_DIR="${WORKER_ENGRAM_DIR:-\$HOME/.cache/dsv41-tensorfold/engram-${ENGRAM_REVISION:0:8}}"
WORKER_CACHE_DIR="${WORKER_CACHE_DIR:-\$HOME/.cache/dsv41-tensorfold/rank-cache-${MODEL_REVISION:0:8}}"
# Keep stock and abliterated weights separate even when the user sets WORKER_CACHE_DIR.
if [[ "$ABLIT" == 1 && "$WORKER_CACHE_DIR" != *"-ablit-${ABLIT_REVISION:0:8}" ]]; then
  WORKER_CACHE_DIR+="-ablit-${ABLIT_REVISION:0:8}"
fi
STATE_DIR="${STATE_DIR:-$HOME/.local/state/dsv41-tensorfold}"   # locks and the prepared marker
LOG_DIR="${LOG_DIR:-$HOME/.cache/dsv41-tensorfold/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Rank 0 can read the checkpoint over NFS too, from HEAD_NFS_SERVER (an address of the NFS server that reaches this Spark),
# through a docker volume HEAD_NFS_VOLUME that start.sh creates here (the docker daemon mounts it: no root, and it survives a
# reboot). Empty: rank 0 reads HF_CACHE as a host path. Not sshfs: it reports every file as type "unknown", which the engine
# skips, so it finds no weight file (NoRankCache).
HEAD_NFS_SERVER="${HEAD_NFS_SERVER:-}"
# Engram's tables on this Spark's own disk (a folder with the base model's shards 47 and 48, as WORKER_ENGRAM_DIR is on the worker),
# mounted into rank 0 at /engram. Without it rank 0 reads them from HF_CACHE: over NFS a random 4 KiB read took 942 us here
# against 90 us from the worker's local copy, and a decode round or a prefill chunk makes many of them. Empty: HF_CACHE.
HEAD_ENGRAM_DIR="${HEAD_ENGRAM_DIR:-}"
HEAD_NFS_VOLUME="${HEAD_NFS_VOLUME:-dsv41-hf-head}"

# ---------------------------------------------------------------- memory
# A Spark's GPU shares its 121 GiB with the host: a rank that takes it all does not fail cleanly, the machine freezes and has
# to be power-cycled. Two guards:
#  - MIN_AVAIL_GIB: start.sh refuses to start unless each Spark has this much memory available (the weights, 98.9 GiB, plus
#    the engine's own 12.2 GiB reserve). 0 turns the check off.
#  - MEMGUARD_GIB: scripts/memguard.sh runs on both Sparks while the server is up and kills the rank's container when the
#    memory available stays under this floor for MEMGUARD_HOLD_S seconds without coming back over it (GiB, decimals allowed:
#    2.5; 0 turns the guard off), or at once when it falls under MEMGUARD_HARD_GIB. A long prompt's workspace takes a few GiB
#    for a moment and gives them back: that dip is logged, not killed.
# Do not lower TENSORFOLD_MEMORY_RESERVE_GIB to make the weights fit: the engine then takes the reserve too (a rank 0 with a
# 10 GiB reserve froze its Spark).
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-112}"
MEMGUARD_GIB="${MEMGUARD_GIB:-2.5}"
MEMGUARD_HOLD_S="${MEMGUARD_HOLD_S:-4}"
MEMGUARD_HARD_GIB="${MEMGUARD_HARD_GIB:-1}"

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
ablit_notice() {
  [[ "$ABLIT" =~ ^[01]$ ]] || die "ABLIT is 0 or 1, not $ABLIT"
  [[ "$ABLIT" == 1 ]] || return 0
  case "$MODEL_ID" in
    Mia-AiLab/DeepSeek-V4.1-Flash-1017-EAQ-v1-2.9-bpw|Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) ;;
    *) die "ABLIT=1 requires the Mia EAQ v1 2.9 bpw checkpoint and its matching $ABLIT_SIDECAR overlay" ;;
  esac
  [[ "$ABLIT_REVISION" =~ ^[0-9a-f]{40}$ ]] || die "ABLIT_REVISION must be a full Hugging Face commit hash"
  warn "ABLIT=1: $ABLIT_ID is gated. You must have access at https://huggingface.co/$ABLIT_ID (accept its terms while logged in) and use your Hugging Face read token: export HF_TOKEN=hf_... or hf auth login. The token's account must have access to this repository."
}
ablit_identity() { if [[ "$ABLIT" == 1 ]]; then echo "1@$ABLIT_REVISION"; else echo 0; fi; }
hub_dir()         { echo "$HF_CACHE/hub/models--${1//\//--}"; }
model_cache_dir() { hub_dir "$MODEL_ID"; }
engram_cache_dir() { hub_dir "$ENGRAM_ID"; }
# The snapshot this setup serves: the pin, else what refs/main names here.
snapshot_rev() { local rev=$MODEL_REVISION; [[ -n "$rev" ]] || rev=$(cat "$(model_cache_dir)/refs/main" 2>/dev/null); echo "$rev"; }
engram_rev()   { local rev=$ENGRAM_REVISION; [[ -n "$rev" ]] || rev=$(cat "$(engram_cache_dir)/refs/main" 2>/dev/null); echo "$rev"; }
# The checkpoint and the Engram folder as the containers see them (HF_CACHE mounted at /hf on both Sparks).
model_in_container()  {
  if [[ "$ABLIT" == 1 ]]; then echo "/hf/$DATA_SUB/checkpoint"
  else echo "/hf/hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"; fi
}
engram_in_container() { echo "/hf/hub/models--${ENGRAM_ID//\//--}/snapshots/$(engram_rev)"; }
# The folder rank 0 serves (prepare.sh): the checkpoint's files linked, with the recipe's chat template.
served_in_container() { echo "/hf/$DATA_SUB/model"; }

# What scripts/prepare.sh last left ready (start.sh runs it again when this line changes). "+stock-dtypes": an overlay
# checkpoint prepared before scripts/ablit_stock_dtypes.py gets that step on the next start.
PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  echo "model=$MODEL_ID@$(snapshot_rev) engram=$ENGRAM_ID@$(engram_rev) image=$(image_ident "$IMAGE") kit=$KIT_REF" \
       "w=$(worker_host 1) wengram=$WORKER_ENGRAM_DIR wcache=$WORKER_CACHE_DIR ablit=$ABLIT" \
       "overlay=$( [[ "$ABLIT" == 1 ]] && echo "$ABLIT_ID@$ABLIT_REVISION+stock-dtypes" || echo none)"
}

# ---------------------------------------------------------------- API key
# The API is open on the network by default: anyone who can reach PORT can use the GPUs. API_KEY=1 makes rank 0 require a key
# (Authorization: Bearer <key>, or x-api-key) on every request but /health, which then answers only {"status": "ok"}. The
# key lives in API_KEY_FILE (default: $STATE_DIR/api-key, mode 600; start.sh generates one when it is missing) and reaches
# the container as a read-only file for tensorfold-native's --api-key-file, never in the command line or the environment.
# The engine rereads that file when it changes, so a new key needs no restart; one key (or "label: key") a line. Clients
# send the key from `cat $API_KEY_FILE`; tools/ reads it from API_KEY or API_KEY_FILE.
API_KEY="${API_KEY:-0}"
API_KEY_FILE="${API_KEY_FILE:-$STATE_DIR/api-key}"
API_KEY_IN_CONTAINER=/run/secrets/tensorfold-api-key
# api_key_ensure: create API_KEY_FILE (mode 600, one random key) when it does not exist; never overwrites a file
api_key_ensure() {
  [[ -e "$API_KEY_FILE" ]] && { [[ "$(stat -c %a "$API_KEY_FILE")" =~ ^[0-7]00$ ]] || chmod 600 "$API_KEY_FILE"; return 0; }
  mkdir -p "$(dirname "$API_KEY_FILE")"
  ( umask 077; python3 -c 'import secrets; print("sk-tf-" + secrets.token_urlsafe(32))' > "$API_KEY_FILE" ) || die "could not write $API_KEY_FILE"
  log "Generated an API key in $API_KEY_FILE (mode 600); clients send it as Authorization: Bearer \$(cat $API_KEY_FILE)"
}
# api_header_file: a mode-600 file holding the Authorization header, for curl -H @file (the key stays out of ps)
api_header_file() {
  local h="$STATE_DIR/curl-auth-header"
  ( umask 077; printf 'Authorization: Bearer %s\n' "$(head -n 1 "$API_KEY_FILE" | sed 's/^[^:]*: *//')" > "$h" ) && echo "$h"
}
