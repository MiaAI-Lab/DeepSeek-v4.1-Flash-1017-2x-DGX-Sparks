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
# TP: the ranks, one a Spark: 2 (the default), or 3, experimental: three Sparks cabled as a triangle on another engine
# (below, and docs/tp3.md). Rank i runs on worker i: WORKER, then WORKER2 with its own NFS_SERVER2 (and FABRIC_PEER2 at
# TP=2), as the GLM-5.3 3x and the 4x recipes name them. WORKER2 and its settings are read only when TP is 3.
TP="${TP:-2}"
WORKER="${WORKER:-}"; FABRIC_PEER="${FABRIC_PEER:-}"
WORKER2="${WORKER2:-}"; FABRIC_PEER2="${FABRIC_PEER2:-}"
MASTER_PORT="${MASTER_PORT:-29620}"  # the ranks' rendezvous port (keep it on the private link)
# The rendezvous address: this node's address on the link to the worker (scripts/nodes.sh sets it from the link when it
# is empty).
MASTER_ADDR="${MASTER_ADDR:-}"
SOCKET_IFNAME="${SOCKET_IFNAME:-}"     # NCCL's bootstrap socket's netdev on both nodes (default: the link's)

# ---------------------------------------------------------------- three Sparks (TP=3, experimental)
# BertholomusAI's engine at TF_REF runs the dsv41 family over two or four ranks, not three. At TP=3 the image is
# LakyFx's TensorFold TP3 port instead (LakyFx/deepseek-v41-flash-exl3-tp3-tp4-spark, named ...-tp3-spark when this tag
# was made; GitHub sends the old name on): the TensorFold runtime snapshot its tag ships, checked by its extract.py,
# built with Zig ZIG_VERSION, and NCCL 2.30.7 with the switchless patches for direct cables (alexellis/switchless-nccl
# v0.0.1, sha256-checked), as easthambuilds' three-Spark recipe runs it
# (easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold). The checkpoint, Engram's tables, ABLIT=1 and
# model/chat_template.jinja stay this recipe's. scripts/prepare-tp3.sh makes every rank's files, zig/Dockerfile.tp3 the
# image; docs/tp3.md says what was and was not run on three Sparks.
TP3_REPO="${TP3_REPO:-https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-tp4-spark.git}"
TP3_TAG=v1.2.2TF                                                    # TP3_REV's tag (for the logs)
TP3_REV="${TP3_REV:-14a6d49c7cb99b8ce25f9755541da7e7d57cb1d1}"
TP3_SNAPSHOT="${TP3_SNAPSHOT:-61357debe730ac9ffe20e760edcddbc253b0e525}"   # the runtime snapshot extract.py must report
NCCL_TGZ_URL="${NCCL_TGZ_URL:-https://github.com/alexellis/switchless-nccl/releases/download/v0.0.1/nccl-2.30.7-switchless-hardened-sm121-linux-arm64.tar.gz}"
NCCL_TGZ_SHA256="${NCCL_TGZ_SHA256:-b4a686382a92e57b485ca1bf7cd0f9fde780a68f01ea902ac432b60505b2041f}"
NCCL_LIB_SHA256="${NCCL_LIB_SHA256:-78cb83871792ec57d763d142e4cae26fc754ae284bcc81dcb2a7d50e17d4fa57}"   # libnccl.so.2.30.7 in it
# The rank files' and the served config's name for this checkpoint in LakyFx's layout: Mia's EAQ v1 2.9 bpw, cut as
# LakyFx's TF29 arm (Mia's earlier 2.9 bpw export, which it matches tensor for tensor: scripts/tp3_plan.py checks it).
TP3_ARM=EAQ29
# The chat template rank 0 serves at TP=3: LakyFx's qualified 1017 template (its port of DeepSeek's encoding.py, the one
# its 1.2.0 profile and easthambuilds' recipe pin), not model/chat_template.jinja, which TP=2 keeps serving. LakyFx's
# model-file step checks it: neither the checkpoint's own template nor this recipe's passes. prepare.sh downloads it from
# LakyFx's 1017 model repository (public, MIT) at its pinned revision and checks its sha256.
TP3_TEMPLATE_ID=LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold
TP3_TEMPLATE_REV=8621801c1fc74aece1fa83e67c6ade3fde1e7894
TP3_TEMPLATE_SHA256=7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e
# The plan scripts/tp3_plan.py writes for the default checkpoint (ABLIT=0), computed from its 39 shards' published
# headers: prepare.sh refuses another. Not checked for another MODEL_ID or MODEL_REVISION, nor ABLIT=1. Empty: no check.
TP3_PLAN_FOR=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw@45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b
TP3_PLAN_SHA256="${TP3_PLAN_SHA256-709fe6c502be4b92fd33d35a6ef108b074e7744e27aa6835c93a69b7bc9257db}"
# The settings only the two-Spark engine has (its zig/patches): set at TP=3 they are refused (tp3_check), so they are
# read here, before their defaults below. TP=3 has its own defaults: LakyFx's qualified TF29 profile (a 3,300,000-token
# pool, 401,408 tokens a request, 8 at once, 16-row rounds, a 1 GiB prompt cache on the head).
TP3_REFUSED=()
if [[ "$TP" == 3 ]]; then
  for _n in GRAPH_CAP RDMA_IDLE_S RDMA_NAP_US PREFILL_INTERLEAVE_MS TF_DS_CONNECT_S HEAD_ENGRAM_DIR; do
    [[ -z "${!_n:-}" ]] || TP3_REFUSED+=("$_n=${!_n}")
  done
  CONTEXT="${CONTEXT:-401408}"
  POOL="${POOL:-3300000}"
  PARALLEL="${PARALLEL:-8}"
  ROUND_ROWS="${ROUND_ROWS:-16}"
  # the weights a Spark holds at TP=3: a third of the 215,248,585,443 bytes of LakyFx's three TF29 rank files, 66.8
  # GiB, and the engine's 12.2 GiB reserve (the memory notes below): 79 GiB, rounded up
  MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-80}"
fi
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-1}"   # TP=3: rank 0's --prompt-cache-gib (kept prompt states on the head)

# ---------------------------------------------------------------- the checkpoint, the engine, the kernel kit
# The weights: Mia's AI Lab's EAQ EXL3 quantization of DeepSeek-V4.1-Flash-1017 at 2.9 bits a weight (39 shards,
# ~196 GiB; the vision tower and the DSpark draft blocks are in it). EAQ v1 is the default: Mia's model card keeps it
# the recommended pack. EAQ v1.1 is experimental (no perplexity or KL measurement against the unquantized model): set
# EAQ_V1_1=1 for it. v1.1 keeps v1's widths but stores 27 indexer/compressor linears as BF16, so the engine must have
# the mixed-pack loader; the image has it. Neither pack holds Engram's n-gram tables: they are read unquantized from
# the base model's shards 47 and 48 (ENGRAM_ID, ~203 GB), which this Spark downloads beside it.
EAQ_V1_1="${EAQ_V1_1:-0}"   # 0: EAQ v1 (recommended); 1: EAQ v1.1 (experimental, BF16 mixed pack)
_default_id=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw
_default_rev=45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b
if [[ "$EAQ_V1_1" == 1 ]]; then
  _default_id=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw
  _default_rev=aee640e2afba93be683bb5b9b420115fe7fbac20
fi
MODEL_ID="${MODEL_ID:-$_default_id}"
_rev=""
case "$MODEL_ID" in
  Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) _rev=45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b ;;
  Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw) _rev=aee640e2afba93be683bb5b9b420115fe7fbac20 ;;
esac
MODEL_REVISION="${MODEL_REVISION-$_rev}"
# Keys' matching Mia EAQ 2.9 bpw overlay (not a standalone checkpoint). It matches EAQ v1, not v1.1: ABLIT=1 with
# EAQ v1.1 is refused below; leave EAQ_V1_1=0 (the default) for the overlay.
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
TF_REF="${TF_REF:-fbbc97132cd004237d9887b8d3185896efaedc5c}"   # the branch tip of 2026-10-23 (was 629c524a; docs/zig-engine.md, "The pin")
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
# Where zig/Dockerfile downloads the compiler: these community mirrors first (space-separated; each serves
# <base>/zig-aarch64-linux-<version>.tar.xz), then ziglang.org, each with bounded retries, the first that serves
# ZIG_SHA256 wins. Set but empty (ZIG_MIRRORS=): ziglang.org only. Not part of the image hash (the bytes are pinned).
ZIG_MIRRORS="${ZIG_MIRRORS-https://zig.linus.dev/zig}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
# (TP=3: zig/Dockerfile.tp3 and its pins; neither zig/patches nor TF_REF is in that image)
image_hash() {
  ( cd "$_cfg_root" || exit 1
    if [[ "$TP" == 3 ]]; then
      cat zig/Dockerfile.tp3
      echo "$TP3_REPO $TP3_REV $TP3_SNAPSHOT $NCCL_TGZ_URL $NCCL_TGZ_SHA256 $NCCL_LIB_SHA256 zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE"
    else
      cat zig/patches/*.patch 2>/dev/null
      cat zig/patches-inputs/*.patch 2>/dev/null
      cat zig/Dockerfile
      echo "$TF_REPO $TF_REF $INPUTS_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE"
    fi
  ) | sha256sum | cut -c1-12
}
TAG_BASE="exl3-tp2"             # the image tag's prefix: <TAG_BASE>-<image hash>
[[ "$TP" != 3 ]] || TAG_BASE="exl3-tp3"
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
# zig/patches/0008: the positions the streams' extents and kept prompts share (rank 0's and rank 1's --pool), when more than CONTEXT: more
# long prompts stay resident (each kept prompt holds its extent) and more long requests run at once. Empty (default): CONTEXT, as before.
# It may not be smaller than CONTEXT. Its KV caches grow with it (about 0.9 GiB a rank per 1,048,576 positions, in host memory: measure the
# head's headroom first, see docs/memory-and-safety.md); the RoPE tables stay the window's size. TF_DS_POOL, and TF_DS_WINDOW (= CONTEXT) on both ranks.
# With KV_SHARD=1 (scripts/port.sh) the same POOL is cut into one part a Spark, each holding a whole CONTEXT window, and each Spark
# keeps only its own part's compressed KV: about half the KV memory a Spark for the same POOL (measured at 2,101,248 positions:
# 1,062 MiB a rank against 1,783 MiB replicated; decode 1-4% slower). Empty with KV_SHARD=1: the pool it needs (2,101,248 at 1,048,576).
POOL="${POOL:-}"
PARALLEL="${PARALLEL:-4}"              # requests at once (the engine's streams, TF_DS_STREAMS; the family measured 4)
ROUND_ROWS="${ROUND_ROWS:-}"           # empty or 16: the pinned kit only has row kernels through 16 (issue #23)
MAX_TOKENS="${MAX_TOKENS:-32768}"      # a reply's budget when a request sets none
THINKING="${THINKING:-1}"              # think first unless a request turns it off
# The reasoning effort when a request names none: low, medium, high or xhigh (empty: the chat template's default)
REASONING_EFFORT="${REASONING_EFFORT:-}"
# Reasoning tokens a reply may spend before the engine closes the think block itself (tensorfold-native --thinking-budget; empty: no
# cap). A request's own thinking_budget wins. With no cap, a turn on a hard task can spend its whole MAX_TOKENS thinking and end
# with no text and no tool call (finish_reason length).
THINKING_BUDGET="${THINKING_BUDGET:-}"
# 1: the chat template sees no assistant turn's reasoning_content (tensorfold-native --drop-reasoning-history, patch 0057). A client
# that replays every turn's reasoning in a tool loop (omp does; the template keeps it all once tools are present) otherwise shows
# the model its own earlier thinking at every step, and long agent runs drifted into flat, then repeating reasoning (loops).
DROP_REASONING_HISTORY="${DROP_REASONING_HISTORY:-0}"
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
# zig/patches/0010: the RDMA proxy naps once quiet for RDMA_IDLE_S seconds, RDMA_NAP_US between looks. 0 keeps the
# always-spinning proxy (the stock engine's behavior). A decode round's gathers never see a nap.
RDMA_IDLE_S="${RDMA_IDLE_S:-2}"       # 0 to 3600 seconds
RDMA_NAP_US="${RDMA_NAP_US:-500}"    # 1 to 1000000 microseconds; bounds wake and shutdown delay
GRAPHS="${GRAPHS:-1}"
# zig/patches/0033: with GRAPHS=1, the captured CUDA graphs each rank keeps at most; past it the least recently replayed
# one is destroyed and captured again when it comes up. 0 keeps every one, as before 0033: they add up over days of
# growing conversations (rows x context buckets x stretches) and are never freed (issue #69).
GRAPH_CAP="${GRAPH_CAP:-128}"
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
# head's address the worker mounts from (default: the head's address on the link, found by scripts/nodes.sh);
# NFS_SERVER2 is WORKER2's.
NFS_PATH="${NFS_PATH:-$HF_CACHE}"
NFS_SERVER="${NFS_SERVER:-}"; NFS_SERVER2="${NFS_SERVER2:-}"
NFS_VOLUME="${NFS_VOLUME:-dsv41-1017-exl3-hf}"
# On the worker's own disk (paths on the worker, the same on every worker): Engram's tables (a decode round reads ~70
# random 4 KiB rows from them; over NFS that took most of a round), and its weight file (~106 GB; written there by
# prepare.sh from the checkpoint over NFS, so it never crosses the network twice).
WORKER_ENGRAM_DIR="${WORKER_ENGRAM_DIR:-\$HOME/.cache/dsv41-tensorfold/engram-${ENGRAM_REVISION:0:8}}"
WORKER_CACHE_DIR="${WORKER_CACHE_DIR:-\$HOME/.cache/dsv41-tensorfold/rank-cache-${MODEL_REVISION:0:8}}"
# Keep stock and abliterated weights separate even when the user sets WORKER_CACHE_DIR.
if [[ "$ABLIT" == 1 && "$WORKER_CACHE_DIR" != *"-ablit-${ABLIT_REVISION:0:8}" ]]; then
  WORKER_CACHE_DIR+="-ablit-${ABLIT_REVISION:0:8}"
fi
# TP=3: each worker's own folder (a path on the worker, the same on every worker): its kit, the served model folder
# (config, tokenizer, vision tower), its Engram rows and its weight file (~67 GiB), copied there by prepare.sh over NFS
# and checked by sha256. No rank reads the checkpoint over NFS at TP=3.
WORKER_TP3_DIR="${WORKER_TP3_DIR:-\$HOME/.cache/dsv41-tensorfold/tp3-${MODEL_REVISION:0:8}}"
if [[ "$ABLIT" == 1 && "$WORKER_TP3_DIR" != *"-ablit-${ABLIT_REVISION:0:8}" ]]; then
  WORKER_TP3_DIR+="-ablit-${ABLIT_REVISION:0:8}"
fi
# TP=3: rank 0's own folder (kit, model, rank-cache, engram, as each worker's WORKER_TP3_DIR) and prepare.sh's work
# files, under DATA_DIR, so that the workers copy theirs over the NFS export
TP3_SUB="$DATA_SUB/tp3"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/dsv41-tensorfold}"   # locks and the prepared marker
LOG_DIR="${LOG_DIR:-$HOME/.cache/dsv41-tensorfold/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
# Each rank's container log (docker's json-file driver) rotates once it reaches LOG_MAX_SIZE, keeping LOG_MAX_FILE files:
# the engine writes an access line for every response, so a monitoring poll grew it for as long as the container ran
# (issue #49). stop.sh saves what docker kept. LOG_MAX_SIZE= (empty) leaves the docker daemon's own log settings.
LOG_MAX_SIZE="${LOG_MAX_SIZE-100m}"
LOG_MAX_FILE="${LOG_MAX_FILE:-3}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Rank 0 can read the checkpoint over NFS too, from HEAD_NFS_SERVER (an address of the NFS server that reaches this Spark),
# through a docker volume HEAD_NFS_VOLUME that start.sh creates here (the docker daemon mounts it: no root, and it survives a
# reboot). Empty: rank 0 reads HF_CACHE as a host path. Not sshfs: it reports every file as type "unknown", which the engine
# skips, so it finds no weight file (NoRankCache).
HEAD_NFS_SERVER="${HEAD_NFS_SERVER:-}"
# Engram's tables on this Spark's own disk (a folder with the base model's shards 47 and 48, as WORKER_ENGRAM_DIR is on the worker),
# mounted into rank 0 at /engram. Without it rank 0 reads them from HF_CACHE: over NFS a random 4 KiB read took 942 us here
# against 90 us from the worker's local copy, and a decode round or a prefill chunk makes many of them. Empty: HF_CACHE.
# With the spill tier on, the engine's rank fingerprint compares this folder's files with the worker's (name and size,
# read through symlinks), so the HF snapshot rank 0 reads when this is empty lists the same files as WORKER_ENGRAM_DIR;
# start.sh's port_engram_check refuses folders that differ before anything is stopped (issue #61).
HEAD_ENGRAM_DIR="${HEAD_ENGRAM_DIR:-}"
HEAD_NFS_VOLUME="${HEAD_NFS_VOLUME:-dsv41-hf-head}"

# ---------------------------------------------------------------- memory
# A Spark's GPU shares its 121 GiB with the host: a rank that takes it all does not fail cleanly, the machine freezes and has
# to be power-cycled. Two guards:
#  - MIN_AVAIL_GIB: start.sh refuses to start unless each Spark has this much memory available (the weights, 98.9 GiB, plus
#    the engine's own 12.2 GiB reserve). 0 turns the check off. Available is MemAvailable plus the pages parked on the
#    kernel's per-CPU free lists beyond their resting level, as the engine's startup budget counts it (GLM 0119;
#    scripts/memavail.sh): a docker save | load or an image build leaves several GiB there for a minute or two.
#    MIN_AVAIL_WAIT_S: it re-reads the memory for up to this many seconds before refusing (0 reads once): the page cache a
#    build or a large copy just left is counted as available only as it is written back, and parked pages drain (120).
#  - MEMGUARD_GIB: scripts/memguard.sh runs on both Sparks while the server is up and kills the rank's container when the
#    memory available stays under this floor for MEMGUARD_HOLD_S seconds without coming back over it (GiB, decimals allowed:
#    2.5; 0 turns the guard off), or at once when it falls under MEMGUARD_HARD_GIB. A long prompt's workspace takes a few GiB
#    for a moment and gives them back: that dip is logged, not killed.
# Do not lower TENSORFOLD_MEMORY_RESERVE_GIB to make the weights fit: the engine then takes the reserve too (a rank 0 with a
# 10 GiB reserve froze its Spark).
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-112}"
MIN_AVAIL_WAIT_S="${MIN_AVAIL_WAIT_S:-120}"
MEMGUARD_GIB="${MEMGUARD_GIB:-2.5}"
MEMGUARD_HOLD_S="${MEMGUARD_HOLD_S:-4}"
MEMGUARD_HARD_GIB="${MEMGUARD_HARD_GIB:-1}"

# ---------------------------------------------------------------- the ported features (off by default: spill v2, KV sharding)
# Spill v2 (SPILL_GIB, SPILL_DIR, WORKER_SPILL_DIR, SPILL_FLOOR_GIB, SPILL_QUOTA_GIB), KV sharding (KV_SHARD,
# KV_SHARD_DECODE; it builds on POOL above): defaults, checks and their mapping to the engine's flags and variables in
# scripts/port.sh (docs/PORT-RECIPE.md).
source "$_cfg_root/scripts/port.sh"

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
ablit_notice() {
  [[ "$EAQ_V1_1" =~ ^[01]$ ]] || die "EAQ_V1_1 is 0 or 1, not $EAQ_V1_1"
  [[ "$ABLIT" =~ ^[01]$ ]] || die "ABLIT is 0 or 1, not $ABLIT"
  [[ "$ABLIT" == 1 ]] || return 0
  case "$MODEL_ID" in
    Mia-AiLab/DeepSeek-V4.1-Flash-1017-EAQ-v1-2.9-bpw|Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) ;;
    *) die "ABLIT=1 requires the Mia EAQ v1 2.9 bpw checkpoint and its matching $ABLIT_SIDECAR overlay (EAQ_V1_1=0)" ;;
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
# The folder rank 0 serves (prepare.sh): the checkpoint's files linked, with the recipe's chat template. TP=3: the
# model folder LakyFx's engine reads (its config for three ranks, the tokenizer, the template and the vision tower).
served_in_container() { if [[ "$TP" == 3 ]]; then echo "/hf/$TP3_SUB/model"; else echo "/hf/$DATA_SUB/model"; fi; }
# TP=3: the RoPE tables' rows the pool needs (LakyFx's make_rope.py: POOL + 49, in whole 4,096-row blocks)
tp3_rope_rows() { echo $(( (POOL + 49 + 4095) / 4096 * 4096 )); }

# tp3_check: at TP=3, refuse what LakyFx's engine does not have (the two-Spark engine's zig/patches and scripts/port.sh:
# spill v2, KV sharding, the EAQ v1.1 mixed-pack loader, the graph cap, the RDMA nap, the prefill interleave, the
# connect window, the reasoning-history drop; rank 0's own Engram folder), and a pool its RoPE tables and engine cannot
# hold. Nothing at TP=2.
tp3_check() {
  [[ "$TP" == 3 ]] || return 0
  local bad=(${TP3_REFUSED[@]+"${TP3_REFUSED[@]}"})
  [[ "$EAQ_V1_1" != 1 ]] || bad+=("EAQ_V1_1=1")
  [[ "$DROP_REASONING_HISTORY" != 1 ]] || bad+=("DROP_REASONING_HISTORY=1")
  [[ "$KV_SHARD" == 0 ]] || bad+=("KV_SHARD=$KV_SHARD")
  ! port_spill_on || bad+=("SPILL_GIB=$SPILL_GIB")
  (( ${#bad[@]} == 0 )) || die "TP=3 runs LakyFx's TP3 engine, which has none of: ${bad[*]} (they are the two-Spark engine's: zig/patches, scripts/port.sh). Unset them for TP=3 (docs/tp3.md)"
  [[ "$POOL" =~ ^[1-9][0-9]{0,7}$ ]] && (( POOL >= (CONTEXT + 50 + 2047) / 2048 * 2048 && POOL <= 10000000 )) ||
    die "POOL is $POOL: at TP=3 it is the pool every request shares (TF_DS_POOL_TOKENS), at least CONTEXT + 50 in whole 2,048-position blocks ($(( (CONTEXT + 50 + 2047) / 2048 * 2048 ))) and at most 10,000,000 (LakyFx's reference RoPE tables). 3300000 is the qualified one"
  [[ "$ROUND_ROWS" == 16 ]] || die "ROUND_ROWS is 16 at TP=3 (LakyFx's qualified TF29 profile), not '$ROUND_ROWS'"
  [[ "$PROMPT_CACHE_GIB" =~ ^[0-9]+$ ]] || die "PROMPT_CACHE_GIB is a whole number of GiB (rank 0's --prompt-cache-gib), not $PROMPT_CACHE_GIB"
}

# What scripts/prepare.sh last left ready (start.sh runs it again when this line changes). "+stock-dtypes": an overlay
# checkpoint prepared before scripts/ablit_stock_dtypes.py gets that step on the next start.
PREPARED_MARKER="$STATE_DIR/prepared"
# Every field is quoted (printf %q), so a value holding another field's text cannot shift into it, and the template's
# sha256 is part of it (a changed model/chat_template.jinja is served only after prepare.sh linked it again). It fails
# (status 1, a message) when the template cannot be hashed: a state that cannot be computed never reads as ready.
prepared_state() {
  local i tpl f
  tpl=$(sha256sum "$_cfg_root/model/chat_template.jinja" | cut -c1-64) && [[ -n "$tpl" ]] ||
    die "cannot hash model/chat_template.jinja (sha256sum)"
  f=("model=$MODEL_ID@$(snapshot_rev)" "engram=$ENGRAM_ID@$(engram_rev)" "image=$(image_ident "$IMAGE")" "kit=$KIT_REF"
     "w=$(worker_host 1)" "wengram=$WORKER_ENGRAM_DIR" "wcache=$WORKER_CACHE_DIR" "ablit=$ABLIT"
     "overlay=$( [[ "$ABLIT" == 1 ]] && echo "$ABLIT_ID@$ABLIT_REVISION+stock-dtypes" || echo none)"
     "hf=$HF_CACHE" "nfs_path=$NFS_PATH" "nfs_volume=$NFS_VOLUME" "template=$tpl")
  for i in 2 3; do [[ -z "$(worker_host "$i")" ]] || f+=("w$i=$(worker_host "$i")"); done
  for i in 1 2 3; do f+=("nfs$i=$(wval NFS_SERVER "$i")"); done
  # TP=3's own files: another engine, plan and folders, and RoPE tables cut to the pool (only then: TP=2's line is as it was)
  [[ "$TP" != 3 ]] || f+=("tp=3" "tp3=$TP3_REV" "arm=$TP3_ARM" "wtp3=$WORKER_TP3_DIR" "rope=$(tp3_rope_rows)" "tp3template=$TP3_TEMPLATE_SHA256")
  (export LC_ALL=C; printf '%q ' "${f[@]}"; echo)   # (a subshell: the caller's locale is left as it was)
}
# prepare.sh's last step: computed first and swapped in whole, so a failure leaves the previous marker as it was
write_prepared_marker() {
  local state
  state=$(prepared_state) || return 1
  printf '%s\n' "$state" > "$PREPARED_MARKER.new" && mv -f "$PREPARED_MARKER.new" "$PREPARED_MARKER"
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
