#!/usr/bin/env bash
# Serve Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw (an EXL3 quantization of DeepSeek-V4.1-Flash-1017) on two DGX
# Sparks with TensorFold's native engine, end to end: runs scripts/prepare.sh when the image, the checkpoint, the weight
# files, the kit or the worker's NFS volume are not ready yet (the first run, or after the patches change), starts rank 1
# on the worker and rank 0 here, which serves the API on port 8888, waits until the OpenAI API answers, then runs a smoke
# test. The worker reads the checkpoint from this Spark over NFS. Stop it with ./stop.sh. EAQ v1 is the default; the
# experimental EAQ v1.1 pack is opt-in with EAQ_V1_1=1.
# The shape of this script (setup, checks, launch, load, smoke test) is the GLM-5.3-Flash recipe's and the four-Spark
# DeepSeek-V4.1 recipe's (MiaAI-Lab); the engine's flags are BertholomusAI's two-node launch.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                  # scripts/config.sh's defaults (4 requests at once, a 1,048,576-token window, drafts,
#                               # images); if the server already runs, says so and leaves it alone
#   SPILL_GIB=64 KV_SHARD=1 ./start.sh restart   # the ported spill v2 and KV sharding on, as the pair was validated
#   ./start.sh restart          # stop both ranks (./stop.sh), then start again (e.g. after changing settings)
#   CONTEXT=262144 ./start.sh restart
#   EAQ_V1_1=1 ./start.sh restart # the experimental EAQ v1.1 mixed pack instead of the EAQ v1 default
#   ABLIT=1 ./start.sh restart   # Mia EAQ 2.9 bpw + Keys overlay; requires your HF token and gated repo access
#   API_KEY=1 ./start.sh restart  # require an API key (generated in $STATE_DIR/api-key, mode 600; see scripts/config.sh)
#   FOREGROUND=1 ./start.sh       # stay attached to rank 0's log and exit with its code (for systemd Restart=on-failure)
#   DRY_RUN=1 ./start.sh        # print the link found and both ranks' docker commands, change nothing and start nothing
# Extra arguments go to rank 0's `tensorfold-native serve` after the defaults, so they win.
# Setup: WORKER (an ssh target, key-based) in scripts/local.sh.
# Settings, from the environment, scripts/local.sh or ./.env (defaults and their notes in scripts/config.sh):
#   serving  CONTEXT, POOL, PARALLEL, ROUND_ROWS, MAX_TOKENS, THINKING, REASONING_EFFORT, THINKING_BUDGET, TEMPERATURE, TOP_P, TOP_K, DRAFTS,
#            SERVED_NAME, HOST, PORT, RDMA, RDMA_IDLE_S, RDMA_NAP_US, GRAPHS, GRAPH_CAP, SIDE, PREFETCH, ENGRAM_AIO, REPLAY_FLOOR
#   nodes    WORKER, FABRIC_PEER, MASTER_ADDR, MASTER_PORT, SOCKET_IFNAME, NCCL_RAILS, NCCL_DEBUG
#   files    EAQ_V1_1 (0 | 1), ABLIT (0 | 1), ABLIT_REVISION, MODEL_ID, MODEL_REVISION, ENGRAM_ID, ENGRAM_REVISION, HF_CACHE, NFS_PATH, NFS_SERVER, NFS_VOLUME,
#            WORKER_ENGRAM_DIR, WORKER_CACHE_DIR, STATE_DIR, LOG_DIR, LOG_KEEP, LOG_MAX_SIZE, LOG_MAX_FILE
#   engine   TF_REPO, TF_REF, INPUTS_REF, KIT_REPO, KIT_REF, ZIG_VERSION, BASE_IMAGE, IMAGE, CONTAINER_NAME
#   setup    PREPARE (auto | 1 | 0), WAIT_TIMEOUT (seconds, default 2400), DRY_RUN=1
#   memory   MIN_AVAIL_GIB (refuse to start below it; pages parked on the per-CPU free lists count, as the engine counts
#            them, and a start waits up to MIN_AVAIL_WAIT_S for them to drain), MEMGUARD_GIB (kill a rank's container
#            below it), HEAD_NFS_SERVER,
#            HEAD_NFS_VOLUME (rank 0 reads the checkpoint over NFS through a docker volume)
#   port     SPILL_GIB (default 0: off), SPILL_DIR, WORKER_SPILL_DIR, SPILL_FLOOR_GIB, SPILL_QUOTA_GIB, SPILL_MIN_TOKENS,
#            SPILL_STAGING_MIB, SPILL_TURNS, SPILL_FLUSH_MS, SPILL_WEIGHTS_SHA, KV_SHARD (default 0: off), KV_SHARD_DECODE,
#            KV_SHARD_CARRY, KV_SHARD_TWINS, KV_SHARD_CHECK (with KV_SHARD=1 an empty POOL is the pool it needs): the
#            ported spill v2 / KV sharding, OFF by default (scripts/port.sh, docs/PORT-RECIPE.md). Set explicitly, a
#            setting that cannot run (a Spark's spill folder fails its checks) refuses; a fork's default that cannot
#            starts without it and warns. Both ranks get the same values, the spill tier's identity
#            (derived from the image and the pins), and each its own local spill directory at /spill. Spill settings
#            are never extra arguments (refused: rank 1 would not get them)
#   ranks    TENSORFOLD_*, TF_DS_* and TF_SPILL_* go to both ranks, except values mapped from recipe settings (use POOL, not TF_DS_POOL)
#            every other exported NCCL_* variable too (values of A-Z a-z 0-9 _ . , : = / - only), RANK1_PROFILE=1|2: rank 1's --profile
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
source ./scripts/nodes.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help|-h|--help) usage; exit 0 ;;
esac
ablit_notice
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1
for v in CONTEXT PARALLEL MAX_TOKENS TOP_K; do [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v is a positive count, not ${!v}"; done
(( CONTEXT <= 1048576 )) || die "CONTEXT is at most 1048576 (the model's window), not $CONTEXT"
[[ -z "$POOL" ]] || { [[ "$POOL" =~ ^[1-9][0-9]*$ ]] && (( POOL >= CONTEXT )); } || die "POOL is empty (the pool is CONTEXT) or a count of positions of at least CONTEXT ($CONTEXT), not $POOL"
(( PARALLEL <= 16 )) || die "PARALLEL is 1 to 16 (the engine's streams), not $PARALLEL"
[[ -z "$ROUND_ROWS" ]] || { [[ "$ROUND_ROWS" =~ ^[0-9]+$ ]] && (( ROUND_ROWS == 16 )); } || die "ROUND_ROWS must be empty or 16: the pinned kit has no wider row kernels (issue #23), not $ROUND_ROWS"
[[ -z "$THINKING_BUDGET" || "$THINKING_BUDGET" =~ ^[1-9][0-9]*$ ]] || die "THINKING_BUDGET is a positive count of reasoning tokens (or empty for no cap), not $THINKING_BUDGET"
[[ -z "$REASONING_EFFORT" || "$REASONING_EFFORT" =~ ^(low|medium|high|xhigh)$ ]] || die "REASONING_EFFORT is low, medium, high or xhigh (or empty), not $REASONING_EFFORT"
[[ "$MIN_AVAIL_GIB" =~ ^[0-9]+$ ]] || die "MIN_AVAIL_GIB is a whole number of GiB (0 turns it off), not $MIN_AVAIL_GIB"
[[ "$MIN_AVAIL_WAIT_S" =~ ^[0-9]+$ ]] && (( MIN_AVAIL_WAIT_S <= 3600 )) || die "MIN_AVAIL_WAIT_S is 0 to 3600 seconds, not $MIN_AVAIL_WAIT_S"
[[ "$MEMGUARD_GIB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MEMGUARD_GIB is a number of GiB (2.5 is fine; 0 turns it off), not $MEMGUARD_GIB"
[[ "$MEMGUARD_HOLD_S" =~ ^[0-9]+(\.[0-9]+)?$ && "$MEMGUARD_HARD_GIB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MEMGUARD_HOLD_S is a number of seconds and MEMGUARD_HARD_GIB a number of GiB, not $MEMGUARD_HOLD_S and $MEMGUARD_HARD_GIB"
[[ "$PREFILL_INTERLEAVE_MS" =~ ^[0-9]+$ ]] && (( PREFILL_INTERLEAVE_MS <= 5000 )) || die "PREFILL_INTERLEAVE_MS is 0 to 5000 milliseconds, not $PREFILL_INTERLEAVE_MS"
[[ -z "$LOG_MAX_SIZE" || "$LOG_MAX_SIZE" =~ ^[1-9][0-9]*[kmg]$ ]] || die "LOG_MAX_SIZE is a size such as 100m (k, m or g), or empty for the docker daemon's own setting, not $LOG_MAX_SIZE"
[[ "$LOG_MAX_FILE" =~ ^[1-9][0-9]*$ ]] || die "LOG_MAX_FILE is a number of files, 1 or more, not $LOG_MAX_FILE"
for v in EAQ_V1_1 THINKING DRAFTS RDMA GRAPHS SIDE PREFETCH ENGRAM_AIO REPLAY_FLOOR API_KEY; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$GRAPH_CAP" =~ ^[0-9]+$ ]] || die "GRAPH_CAP is a count of CUDA graphs a rank keeps, 0 for all, not $GRAPH_CAP"
[[ "$RDMA_IDLE_S" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v value="$RDMA_IDLE_S" 'BEGIN { exit !(value <= 3600) }' || die "RDMA_IDLE_S is 0 to 3600 seconds (0 turns the nap off), not $RDMA_IDLE_S"
[[ "$RDMA_NAP_US" =~ ^[1-9][0-9]*$ ]] && awk -v value="$RDMA_NAP_US" 'BEGIN { exit !(value <= 1000000) }' || die "RDMA_NAP_US is 1 to 1000000 microseconds, not $RDMA_NAP_US"
[[ "${RANK1_PROFILE:-}" =~ ^[012]?$ ]] || die "RANK1_PROFILE is empty, 0, 1 or 2, not '${RANK1_PROFILE:-}'"
for v in API_GATEWAY TOKEN_ROUTE GATEWAY_STRICT GATEWAY_BLOCK_PRIVATE_URLS GATEWAY_API_SAMPLING; do
  [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"
done
[[ -z "${TF_DS_API_EFFORTS:-}" || "$TF_DS_API_EFFORTS" =~ ^[01]$ ]] || die "TF_DS_API_EFFORTS is empty, 0 or 1, not $TF_DS_API_EFFORTS"
[[ "$ENGINE_PORT" =~ ^[1-9][0-9]*$ ]] && (( ENGINE_PORT <= 65535 )) || die "ENGINE_PORT is 1 to 65535, not $ENGINE_PORT"
[[ "$API_GATEWAY" != 1 || "$ENGINE_PORT" != "$PORT" ]] || die "ENGINE_PORT and PORT are both $PORT: the engine and the gateway need different ports"
[[ "$GATEWAY_MEMORY" =~ ^([0-9]+([.][0-9]+)?)[mMgG]$ ]] && awk -v m="${BASH_REMATCH[1]}" 'BEGIN { exit !(m > 0) }' ||
  die "GATEWAY_MEMORY is a size over zero with an m or g suffix (512m, 1g), not $GATEWAY_MEMORY"
[[ "$GATEWAY_CPUS" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v c="$GATEWAY_CPUS" 'BEGIN { exit !(c > 0) }' || die "GATEWAY_CPUS is a positive number, not $GATEWAY_CPUS"
[[ "$GATEWAY_MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || die "GATEWAY_MAX_CONCURRENT is a positive count, not $GATEWAY_MAX_CONCURRENT"
[[ "$GATEWAY_MAX_CONNECTIONS" =~ ^[1-9][0-9]*$ ]] || die "GATEWAY_MAX_CONNECTIONS is a positive count, not $GATEWAY_MAX_CONNECTIONS"
[[ "$GATEWAY_BODY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "GATEWAY_BODY_TIMEOUT is a positive number of seconds, not $GATEWAY_BODY_TIMEOUT"
[[ "$GATEWAY_MAX_CONNECTIONS_PER_PEER" =~ ^[1-9][0-9]*$ ]] || die "GATEWAY_MAX_CONNECTIONS_PER_PEER is a positive count, not $GATEWAY_MAX_CONNECTIONS_PER_PEER"
[[ "$GATEWAY_HEADER_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "GATEWAY_HEADER_TIMEOUT is a positive number of seconds, not $GATEWAY_HEADER_TIMEOUT"
[[ "$GATEWAY_WRITE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "GATEWAY_WRITE_TIMEOUT is a positive number of seconds, not $GATEWAY_WRITE_TIMEOUT"
[[ "$GATEWAY_MAX_PROMPT_MB" =~ ^[1-9][0-9]*$ ]] || die "GATEWAY_MAX_PROMPT_MB is a positive count, not $GATEWAY_MAX_PROMPT_MB"
# With a key the gateway runs as the key's owner (below), so that owner may not be root: the network-facing process
# would then run as root. Run start.sh as the non-root user who owns API_KEY_FILE.
[[ "$API_GATEWAY" != 1 || "$API_KEY" != 1 || "$(id -u)" != 0 ]] ||
  die "API_GATEWAY=1 with API_KEY=1 runs the gateway as the user running start.sh, and that is root: run it as the key's non-root owner"
[[ "$GATEWAY_LOG" =~ ^(debug|info|warn|error)$ ]] || die "GATEWAY_LOG is debug, info, warn or error, not $GATEWAY_LOG"
[[ -n "$GATEWAY_IMAGE_TOKEN" ]] || die "GATEWAY_IMAGE_TOKEN is empty: give the served tokenizer's image-token spelling"
for v in GATEWAY_MAX_IMAGES GATEWAY_MAX_IMAGE_MB; do [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v is a positive count, not ${!v}"; done
[[ -n "$GATEWAY_IMAGE" ]] || die "GATEWAY_IMAGE is empty"
[[ "$API_GATEWAY" != 1 ]] || TOKEN_ROUTE=1
nccl_extra_collect
check_workers
need_workers
port_validate
# the spill tier's directory on each Spark (scripts/port.sh): local disk, yours, outside the export and the model folders,
# above SPILL_FLOOR_GIB; created when missing (DRY_RUN: not). Checked here, before the flags, the variables and the
# tf.recipe.port label are made and before a running pair is compared with them: a spill tier that is on by default
# only and fails here is off for this start (a warning), an explicit SPILL_GIB that fails refuses. Nothing with SPILL_GIB=0.
port_spill_check
# with the spill tier on, both ranks must list the same Engram files (the engine's rank fingerprint compares them; rank
# 0's HF snapshot is all symlinks, which the listing skips) — refused here, before anything is stopped (issue #61)
port_engram_check
port_map
port_check_extra "$@"

# rank 0's command (its serve flags need no link): built first, so the flags can be checked before anything is stopped
# API_GATEWAY=1: the engine moves to loopback and the gateway takes HOST:PORT. With API_GATEWAY=0 this is Mia's command;
# only the derived image tag changes when patch 0039 enters zig/patches/.
SERVE_HOST=$HOST; SERVE_PORT=$PORT
if [[ "$API_GATEWAY" == 1 ]]; then SERVE_HOST=127.0.0.1; SERVE_PORT=$ENGINE_PORT; fi
SERVE=(tensorfold-native serve "$(served_in_container)" --backend cuda --host "$SERVE_HOST" --port "$SERVE_PORT" --name "$SERVED_NAME"
       --context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE+=(--thinking); else SERVE+=(--no-thinking); fi
[[ -z "$REASONING_EFFORT" ]] || SERVE+=(--reasoning-effort "$REASONING_EFFORT")
[[ -z "$THINKING_BUDGET" ]] || SERVE+=(--thinking-budget "$THINKING_BUDGET")
# drafts off on rank 0 as on rank 1 (a follower without the drafter refuses a drafted round)
[[ "$DRAFTS" == 1 ]] || SERVE+=(--no-drafts)
[[ "$API_KEY" != 1 ]] || SERVE+=(--api-key-file "$API_KEY_IN_CONTAINER")
SERVE+=(${PORT_SERVE[@]+"${PORT_SERVE[@]}"})   # the ported features' flags (scripts/port.sh; none with all of them off)
SERVE+=("$@")

B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · 2 x DGX Spark · %s at once · %s-token window%s · EXL3 2.9 bpw · port %s%s\n\n' "$D" "$MODEL_ID" "$PARALLEL" "$CONTEXT" "$(port_summary)" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
(( DRY )) && log "DRY_RUN=1: printing the docker commands; nothing is stopped, started or prepared"

API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" != *:* ]] || API_HOST="[$API_HOST]"   # an IPv6 address goes in brackets in a URL
URL="http://$API_HOST:$PORT"
running_here()   { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
running_worker() { [[ "$(worker "$1" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
# a container of that name at all, running or exited (a rank that failed, or rank 0 ended by a broken pair's status 75)
exists_here()    { docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; }
exists_worker()  { worker "$1" docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; }
running_gateway() { [[ "$(docker inspect -f '{{.State.Running}}' ds41-gateway 2>/dev/null)" == true ]]; }
# A digest of the gateway's settings: the GATEWAY_* variables and what else goes into its container. Its label tells a
# gateway started with other settings, which must not be reused.
gateway_identity() {
  local name
  { for name in $(compgen -v GATEWAY_); do
      case "$name" in GATEWAY_RUN | GATEWAY_ARGS | GATEWAY_LISTEN | GATEWAY_TOKENIZER) continue ;; esac
      printf '%s=%s\n' "$name" "${!name}"
    done
    printf '%s\n' "$HOST" "$PORT" "$ENGINE_PORT" "$SERVED_NAME" "$CONTEXT" "$THINKING" "$REASONING_EFFORT" "$API_KEY" \
      "$API_KEY_FILE" "$DATA_DIR"
    # the key the gateway loaded at its start: a key rotated in the same file is another gateway (only a digest of it,
    # inside this digest)
    if [[ "$API_KEY" == 1 && -r "$API_KEY_FILE" ]]; then sha256sum < "$API_KEY_FILE"; else echo no-key-file; fi
  } | sha256sum | cut -c1-16
}
# Remove ds41-gateway after saving its log (FOREGROUND's stops: a gateway that exited keeps its last words).
gateway_remove_keeping_log() {
  save_log "$LOG_DIR" gateway ds41-gateway "$LOG_KEEP" >/dev/null 2>&1 || true
  docker rm -f ds41-gateway >/dev/null 2>&1 || true
}
# Start a stopped ds41-gateway again (its on-failure restarts used up) and wait for $1/health; a removed one cannot be
# rebuilt from here, and one that does not come up within 30 s is a failure.
gateway_recover() {
  local i
  docker inspect ds41-gateway >/dev/null 2>&1 || return 1
  docker start ds41-gateway >/dev/null 2>&1 || return 1
  for i in $(seq 1 30); do
    curl -sf --max-time 3 "$1/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}
if (( ! DRY )) && [[ "$MODE" == start ]] && running_here && running_worker 1; then
  # Preparation may have run separately while the old server stayed up: inspect the live ranks too.
  live_head=$(docker inspect -f '{{index .Config.Labels "tf.recipe.ablit"}}' "$CONTAINER_NAME")
  live_worker=$(worker 1 "docker inspect -f '{{index .Config.Labels \"tf.recipe.ablit\"}}' '$CONTAINER_NAME'")
  # Containers launched before this flag existed are stock.
  [[ -n "$live_head" && "$live_head" != '<no value>' ]] || live_head=0
  [[ -n "$live_worker" && "$live_worker" != '<no value>' ]] || live_worker=0
  [[ "$live_head" == "$(ablit_identity)" && "$live_worker" == "$(ablit_identity)" ]] || \
    die "the running ranks use a different ABLIT mode or overlay revision; run ./start.sh restart with ABLIT=$ABLIT"
  # the ported features (scripts/port.sh): containers started before them carry no label, and run with all of them off
  live_head=$(docker inspect -f '{{index .Config.Labels "tf.recipe.port"}}' "$CONTAINER_NAME")
  live_worker=$(worker 1 "docker inspect -f '{{index .Config.Labels \"tf.recipe.port\"}}' '$CONTAINER_NAME'")
  [[ -n "$live_head" && "$live_head" != '<no value>' ]] || live_head=off
  [[ -n "$live_worker" && "$live_worker" != '<no value>' ]] || live_worker=off
  [[ "$live_head" == "$live_worker" ]] || die "the two running ranks disagree on the ported features ($live_head here, $live_worker on $(worker_host 1)): run ./start.sh restart"
  [[ "$live_head" == "$(port_identity)" ]] || \
    die "the running ranks use other spill / KV-shard / pool settings ($live_head, asked: $(port_identity)); run ./start.sh restart"
  state=$(prepared_state) || die "cannot tell whether the setup is ready: see the error above"
  [[ "$state" == "$(cat "$PREPARED_MARKER" 2>/dev/null)" ]] || \
    die "the requested configuration differs from the prepared server; run ./start.sh restart with the same settings (including ABLIT)"
  if [[ "$API_GATEWAY" == 1 ]]; then
    # The gateway is reused (or started again) only when its label reads back equal to these settings' digest, key
    # included. Missing, unreadable, unlabelled or different: no reuse.
    live_gateway=$(docker inspect -f '{{index .Config.Labels "tf.recipe.gateway"}}' ds41-gateway 2>/dev/null || true)
    [[ "$live_gateway" == "$(gateway_identity)" ]] || \
      die "ds41-gateway is missing, or was started with other settings or another key than these: run ./start.sh restart"
    if ! running_gateway; then
      gateway_recover "$URL" || die "the ranks are running but ds41-gateway is not, and it could not be started again: run ./start.sh restart"
      log "ds41-gateway was not running: started it again on $HOST:$PORT"
    fi
  fi
  log "$CONTAINER_NAME is already running on both Sparks (port $PORT): nothing to do. ./start.sh restart restarts it, ./stop.sh stops it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
step 1 "Setup: image, checkpoint, Engram's tables, weight files and kit; the worker's NFS volume and disk"
# (a state that cannot be computed is an error here, not "equal to a missing marker": both read empty otherwise)
need_prepare=0
case "${PREPARE:-auto}" in
  1) need_prepare=1 ;;
  0) ;;
  *) state=$(prepared_state) || die "cannot tell whether the setup is ready: see the error above"
     [[ "$state" == "$(cat "$PREPARED_MARKER" 2>/dev/null)" ]] || need_prepare=1 ;;
esac
if (( need_prepare )); then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now (not ready yet)"
  else
    # setup builds the image and writes ~106 GB files: with ~104 GiB of a Spark's memory held by a running rank that can freeze it
    if running_here || running_worker 1; then
      log "Setup is needed and a server is running: stopping it first (the build needs the memory)"
      ./stop.sh
    fi
    log "Not ready yet: running scripts/prepare.sh (the first time this builds the image, downloads ~400 GB and writes two ~106 GB weight files)"
    ./scripts/prepare.sh
  fi
else
  log "Ready: $IMAGE, $MODEL_ID @ $(snapshot_rev | cut -c1-8), Engram's tables, the weight files and the kit"
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || (( DRY )) || die "image $IMAGE missing: scripts/prepare.sh did not build it"
MODEL_ARG=$(model_in_container)

# ---------------------------------------------------------------- 2. checks
step 2 "Checks: link, previous server, port, memory"
# HEAD_ENGRAM_DIR: every Engram shard there, whole, before anything is stopped or started (a missing or short shard would
# only show up as a failed or slow rank 0 after rank 1 is up)
if [[ -n "$HEAD_ENGRAM_DIR" ]] && (( ! DRY )); then
  for f in $ENGRAM_SHARDS; do
    [[ -f "$HEAD_ENGRAM_DIR/$f" ]] || die "HEAD_ENGRAM_DIR $HEAD_ENGRAM_DIR has no $f (needs: $ENGRAM_SHARDS)"
    if [[ "$ENGRAM_ID" == deepseek-ai/DeepSeek-V4.1-Flash-1017 && -n "${ENGRAM_SHARD_SIZE[$f]:-}" ]]; then
      [[ "$(stat -L -c %s "$HEAD_ENGRAM_DIR/$f")" == "${ENGRAM_SHARD_SIZE[$f]}" ]] ||
        die "$HEAD_ENGRAM_DIR/$f is $(stat -L -c %s "$HEAD_ENGRAM_DIR/$f") bytes, not the release's ${ENGRAM_SHARD_SIZE[$f]}: copy it again (or unset HEAD_ENGRAM_DIR)"
    fi
  done
fi
detect_links
log "Link: rendezvous $MASTER_ADDR:$MASTER_PORT"
for r in 0 1; do
  log "  rank $r ($( (( r == 0 )) && echo this node || worker_host "$r")): RoCE ${NODE_HCAS[r]} (GID ${NODE_GID[r]})$( (( r == 0 )) || echo ", NFS from ${LINK_HEAD_ADDR[r]:-?}")"
done
# the serve flags against what this build of tensorfold-native serves (scripts/zig_flags.py, from the GLM-5.3-Flash recipe):
# the binary refuses an unsupported flag at start, after the container is up and the old server stopped
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker run --rm --network none --entrypoint tensorfold-native "$IMAGE" capabilities --json |
    python3 scripts/zig_flags.py "${SERVE[@]:3}" || die "rank 0's serve arguments are not ones $IMAGE serves (above); nothing was stopped"
fi
# (the spill tier's directories were checked before the flags were made: port_spill_check, above)
# the spill tier's identity (scripts/port.sh): the image's patch hash and ref, the checkpoint's and Engram's revisions,
# each rank's weight-file digest (cached beside the file). Nothing with SPILL_GIB=0.
port_ident_map "$IMAGE"
port_weights_check
# A previous gateway (running or exited) is not reusable: preserve its log before replacing it.
if (( ! DRY )) && docker inspect ds41-gateway >/dev/null 2>&1; then
  gateway_saved=$(save_log "$LOG_DIR" gateway ds41-gateway "$LOG_KEEP") || warn "could not save the previous gateway's log"
  docker rm -f ds41-gateway >/dev/null 2>&1 || die "could not remove the previous ds41-gateway container"
fi
if (( ! DRY )); then
  # Preparation's marker cannot detect an export/root change since the last run.
  worker_files_check 1
  # stop.sh for any container left, an exited one too: it saves each rank's log before it removes the container, so a
  # restart (./start.sh restart, or a supervisor's start after rank 0 exited) keeps the failed run's last lines
  if exists_here || exists_worker 1; then ./stop.sh; fi
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  worker 1 "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1; true"
  if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"; fi
  # with the gateway the engine listens on ENGINE_PORT too (loopback): an occupied one would fail only after the ranks load
  if [[ "$API_GATEWAY" == 1 ]] && ss -ltn "sport = :$ENGINE_PORT" 2>/dev/null | grep -q LISTEN; then
    die "ENGINE_PORT $ENGINE_PORT is already in use: $(ss -ltnp "sport = :$ENGINE_PORT" 2>/dev/null | tail -n +2)"
  fi
  # memory: a rank that takes all of a Spark's memory freezes the machine, so refuse to start on one that cannot hold it.
  # Counted as the engine's startup budget counts it (0119): MemAvailable plus the pages parked on the per-CPU free
  # lists (scripts/memavail.sh). Right after an image build or copy those drain for a minute or two: wait up to
  # MIN_AVAIL_WAIT_S for both Sparks before refusing.
  if (( MIN_AVAIL_GIB > 0 )); then
    await_memory || true
    here_gb=$HERE_GB
    if awk -v a="$here_gb" -v m="$MIN_AVAIL_GIB" 'BEGIN {exit !(a < m)}'; then
      shm=$(du -sm /dev/shm 2>/dev/null | cut -f1); shm=${shm:-0}
      die "only $here_gb GiB memory available here (MemAvailable plus the pages parked on the per-CPU lists), $MIN_AVAIL_GIB GiB needed (the weights plus the engine's reserve): free memory first${shm:+ (/dev/shm holds $shm MiB)}; MIN_AVAIL_GIB=0 skips this check, and a rank that takes all the memory freezes the Spark"
    fi
    g=$WORKER_GB
    awk -v a="$g" -v m="$MIN_AVAIL_GIB" 'BEGIN {exit !(a < m)}' &&
      die "only $g GiB memory available on $(worker_host 1), $MIN_AVAIL_GIB GiB needed: stop other GPU work there (MIN_AVAIL_GIB=0 skips this check)"
  fi
  memguard_start
fi

# ---------------------------------------------------------------- 3. launch
# FOREGROUND=1: stay attached to rank 0's log and exit with its code, so that a supervisor (systemd's Restart=on-failure)
# can restart the server. Either rank ending takes the other one down: a lone rank waits for its peer forever. SIGINT and
# SIGTERM run ./stop.sh. Note the supervisor restarts a start that was killed by memguard too; give it a RestartSec.
# (From MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold's start.sh, for the one worker here.)
foreground() {
  local watch w code failed="$STATE_DIR/foreground.failed"
  trap './stop.sh; exit 130' INT TERM
  # The watcher marks a failure BEFORE it stops rank 0: stopping rank 0 ends `docker logs -f` below, and this shell then
  # kills the watcher, which may not live to exit 1. The mark survives it.
  rm -f "$failed"
  ( exec 8>&-                                        # not holding start.sh's lock once start.sh has exited
    while sleep 30; do
      if ! running_here; then
        [[ "$API_GATEWAY" != 1 ]] || gateway_remove_keeping_log
        exit 0
      fi
      # The gateway is the public API: one that exited is started again, and if that fails the server stops with a
      # failure, so that the supervisor restarts the whole of it instead of leaving it up without its API.
      if [[ "$API_GATEWAY" == 1 ]] && ! running_gateway; then
        if gateway_recover "$API_URL"; then
          warn "ds41-gateway had exited: started it again"
        else
          warn "ds41-gateway exited and could not be started again: stopping the server"
          : > "$failed"
          docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1
          gateway_remove_keeping_log
          exit 1
        fi
      fi
      worker 1 true 2>/dev/null || continue         # a worker out of reach for a moment says nothing about its rank
      running_worker 1 && continue
      warn "rank 1 on $(worker_host 1) exited: stopping rank 0"
      : > "$failed"
      docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1
      [[ "$API_GATEWAY" != 1 ]] || gateway_remove_keeping_log
      exit 1
    done ) &
  watch=$!
  docker logs -f "$CONTAINER_NAME" || true
  kill "$watch" 2>/dev/null || true
  w=0; wait "$watch" || w=$?
  code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)
  [[ "$w" != 1 || "$code" != 0 ]] || code=1         # the worker's rank failed, even if rank 0 then shut down cleanly
  if [[ -e "$failed" ]]; then [[ "$code" != 0 ]] || code=1; rm -f "$failed"; fi   # the same, when the watcher was killed first
  worker 1 "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME'" >/dev/null 2>&1 || true
  [[ "$API_GATEWAY" != 1 ]] || gateway_remove_keeping_log
  exit "$code"
}
# What both ranks' engine reads, as BertholomusAI's two-node launch has it (HF_CACHE at /hf: this Spark's own, the worker's over NFS)
# zig/patches/0037: how long rank 1 retries its connect to rank 0 (seconds, default 300); both ranks refuse anything else
[[ -z "${TF_DS_CONNECT_S:-}" ]] || { [[ "$TF_DS_CONNECT_S" =~ ^[1-9][0-9]{0,4}$ ]] && (( TF_DS_CONNECT_S <= 86400 )); } ||
  die "TF_DS_CONNECT_S is a count of seconds from 1 to 86400, not $TF_DS_CONNECT_S"
ENGINE_ENV=("TF_DS_REPLAY_FLOOR=$REPLAY_FLOOR" "TF_DS_STREAMS=$PARALLEL" "TF_RDMA_IDLE_S=$RDMA_IDLE_S" "TF_RDMA_NAP_US=$RDMA_NAP_US" "HF_HUB_OFFLINE=$HF_HUB_OFFLINE")
[[ "${TOKEN_ROUTE:-0}" != 1 ]] || ENGINE_ENV+=("TENSORFOLD_TOKEN_ROUTE=1")
[[ -z "${TF_DS_API_EFFORTS:-}" ]] || ENGINE_ENV+=("TF_DS_API_EFFORTS=$TF_DS_API_EFFORTS")
# API_GATEWAY=1: the effort names must render the API table (minimal 50, low 50, medium 75, high 75,
# xhigh 75, max 100); the engine's own table (25/50/62/75/88/100) is the native one. 0039 carries both.
[[ "${API_GATEWAY:-0}" != 1 || -n "${TF_DS_API_EFFORTS:-}" ]] || ENGINE_ENV+=("TF_DS_API_EFFORTS=1")
[[ -z "$POOL" ]] || ENGINE_ENV+=("TF_DS_POOL=$POOL" "TF_DS_WINDOW=$CONTEXT")   # zig/patches/0008: both ranks, alike
[[ -z "$ROUND_ROWS" ]] || ENGINE_ENV+=("TF_DS_ROUND_ROWS=$ROUND_ROWS")
(( PREFILL_INTERLEAVE_MS == 0 )) || ENGINE_ENV+=("TF_DS_PREFILL_INTERLEAVE_MS=$PREFILL_INTERLEAVE_MS")
ENGINE_ENV+=(${PORT_ENV[@]+"${PORT_ENV[@]}"})   # the ported features' variables, the same on both ranks (scripts/port.sh)
ENGINE_ENV+=(${PORT_IDENT_ENV[@]+"${PORT_IDENT_ENV[@]}"})   # the spill tier's identity, the same on both ranks
while IFS='=' read -r name _; do ENGINE_ENV+=("$name=${!name}"); done < <(env | grep -E '^(TENSORFOLD|TF_DS|TF_SPILL|TF_FILL)_[A-Z0-9_]+=' | grep -vE '^TF_DS_(REPLAY_FLOOR|STREAMS|POOL|WINDOW|ROUND_ROWS|GRAPHS|GRAPH_CAP|HC_SIDE|L2_PREFETCH|ENGRAM_AIO|RANK_CACHE|KIT|ENGRAM|TOKEN_MAP|API_EFFORTS)=' | grep -vE "^(${PORT_MAPPED_RE:-^$})=" | grep -v '^TENSORFOLD_TOKEN_ROUTE=' || true)
envs() { local x; for x in "$@"; do printf -- '-e %q ' "$x"; done; }
RUN_ARGS="--label tf.recipe.ablit=$(ablit_identity) --label tf.recipe.port=$(port_identity) --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband"
# each rank's container log rotates (LOG_MAX_SIZE, LOG_MAX_FILE; issue #49): the engine logs every response
[[ -z "$LOG_MAX_SIZE" ]] || RUN_ARGS+=" --log-driver json-file --log-opt max-size=$LOG_MAX_SIZE --log-opt max-file=$LOG_MAX_FILE"
# A gateway started for this run is removed before a failure exits, so a later start does not inherit a broken one.
gateway_remove_after_failure() {
  [[ "$API_GATEWAY" == 1 ]] || return 0
  docker inspect ds41-gateway >/dev/null 2>&1 || return 0
  if docker rm -f ds41-gateway >/dev/null 2>&1; then
    log "Removed ds41-gateway after the failure"
  else
    warn "could not remove ds41-gateway after the failure"
  fi
}
# Save and print the gateway's last log lines before removing it, so a failed start keeps its reason.
gateway_dump_logs() {
  [[ "$API_GATEWAY" == 1 ]] || return 0
  docker inspect ds41-gateway >/dev/null 2>&1 || return 0
  local saved
  saved=$(save_log "$LOG_DIR" gateway ds41-gateway "$LOG_KEEP") || true
  docker logs --tail 15 ds41-gateway 2>&1 | sed 's/^/  │ /'
  [[ -z "$saved" ]] || log "Gateway's log: $saved"
}
# ds41-gateway's command: the API's flags, the engine on loopback, the served tokenizer and the container's limits.
GATEWAY_LISTEN="$HOST:$PORT"; [[ "$HOST" != *:* ]] || GATEWAY_LISTEN="[$HOST]:$PORT"
GATEWAY_ARGS=(--listen "$GATEWAY_LISTEN" --engine "http://127.0.0.1:$ENGINE_PORT" --tokenizer /served/tokenizer.json
              --served-name "$SERVED_NAME" --image-token "$GATEWAY_IMAGE_TOKEN" --max-model-len "$CONTEXT"
              --max-images "$GATEWAY_MAX_IMAGES" --max-image-mb "$GATEWAY_MAX_IMAGE_MB"
              --max-concurrent "$GATEWAY_MAX_CONCURRENT" --max-connections "$GATEWAY_MAX_CONNECTIONS"
              --max-connections-per-peer "$GATEWAY_MAX_CONNECTIONS_PER_PEER" --header-timeout "$GATEWAY_HEADER_TIMEOUT"
              --body-timeout "$GATEWAY_BODY_TIMEOUT" --write-timeout "$GATEWAY_WRITE_TIMEOUT" --max-prompt-mb "$GATEWAY_MAX_PROMPT_MB" --log "$GATEWAY_LOG")
if [[ "$THINKING" == 1 ]]; then GATEWAY_ARGS+=(--default-thinking); else GATEWAY_ARGS+=(--default-thinking=false); fi
[[ -z "$REASONING_EFFORT" ]] || GATEWAY_ARGS+=(--default-effort "$REASONING_EFFORT")
[[ "$GATEWAY_STRICT" != 1 ]] || GATEWAY_ARGS+=(--strict)
if [[ "$GATEWAY_BLOCK_PRIVATE_URLS" == 1 ]]; then GATEWAY_ARGS+=(--block-private-urls); else GATEWAY_ARGS+=(--block-private-urls=false); fi
if [[ "$GATEWAY_API_SAMPLING" == 1 ]]; then GATEWAY_ARGS+=(--api-sampling); else GATEWAY_ARGS+=(--api-sampling=false); fi
[[ "$API_KEY" != 1 ]] || GATEWAY_ARGS+=(--require-key --api-key-file /run/secrets/key)
# The gateway reads one file of the served folder, its tokenizer. prepare.sh fills the folder with relative links into the
# HF cache (tokenizer.json -> ../hub/.../snapshots/REV/tokenizer.json -> ../../blobs/HASH), which a mount of the folder
# would leave dangling inside the container: the file the links end at is mounted itself. A dry run prints the commands
# without touching the filesystem (prepare.sh may not have downloaded the model yet).
gateway_tokenizer() {
  local file
  file=$(readlink -f "$1/tokenizer.json" 2>/dev/null) || return 1
  [[ -f "$file" && -r "$file" ]] || return 1
  printf '%s\n' "$file"
}
GATEWAY_TOKENIZER=$DATA_DIR/model/tokenizer.json
if [[ "$DRY" != 1 && "$API_GATEWAY" == 1 ]]; then   # only the gateway reads it
  GATEWAY_TOKENIZER=$(gateway_tokenizer "$DATA_DIR/model") ||
    die "$DATA_DIR/model/tokenizer.json does not lead to a readable file: prepare.sh links the served folder there"
fi
GATEWAY_RUN=(docker run -d --name ds41-gateway --network host --memory "$GATEWAY_MEMORY" --cpus "$GATEWAY_CPUS"
             --restart on-failure:3 --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges
             -v "$GATEWAY_TOKENIZER:/served/tokenizer.json:ro")
# The key file is mode 600 and belongs to the user running this script (api_key_ensure, and the smoke test reads it), so
# with a key the gateway runs as that uid:gid instead of the image's ds41 (10001), which could not read it; never as root
# (refused above). With Docker's userns-remap the uid inside is not the host's and the key stays unreadable: not supported.
if [[ "$API_KEY" == 1 ]]; then
  GATEWAY_RUN+=(--user "$(id -u):$(id -g)" -v "$API_KEY_FILE:/run/secrets/key:ro" -e API_KEY_FILE=/run/secrets/key)
fi
GATEWAY_RUN+=("$GATEWAY_IMAGE")
DATA=/hf/$DATA_SUB
KIT=/hf/$KIT_SUB/kit
# rank 1 takes the round's settings as flags only (tf-dsv41-lanes reads no TF_DS_GRAPHS & co.): without them it runs
# every round ungraphed, with no side stream, prefetch or AIO, and rank 0 waits on it each round. Its pool is rank 0's
# context (or POOL, which scripts/port.sh derives for KV_SHARD=1 when it is empty), and its streams rank 0's (TF_DS_STREAMS): with other values the two ranks wait on each other.
FOLLOW="--pool ${POOL:-$CONTEXT} --streams $PARALLEL --drafts $DRAFTS --graphs $GRAPHS --side $SIDE --prefetch $PREFETCH --aio $ENGRAM_AIO"
# zig/patches/0033: only with graphs on, so an image from before 0033 still starts with GRAPHS=0 (it refuses the flag)
[[ "$GRAPHS" == 0 ]] || FOLLOW+=" --graph-cap $GRAPH_CAP"
[[ -z "$ROUND_ROWS" ]] || FOLLOW+=" --round-rows $ROUND_ROWS"
# RANK1_PROFILE=1|2: rank 1's own profile (--profile; 2 adds prefill_phases_ms), AGGREGATED over the run and printed to its
# log only when rank 0 goes away first (docker stop rank 0, read rank 1's log, then ./stop.sh). It synchronizes the stream
# at every chunk and round: for attribution only, never for speed figures. Empty or 0: off.
[[ "${RANK1_PROFILE:-0}" == 0 ]] || FOLLOW+=" --profile $RANK1_PROFILE"
AUTH=()      # curl's Authorization header (a file, so the key stays out of ps) when API_KEY=1
if [[ "$API_KEY" == 1 ]]; then
  if (( DRY )); then log "DRY_RUN: API_KEY=1: the key would be read from $API_KEY_FILE (created, mode 600, when missing)"
  else api_key_ensure; AUTH=(-H "@$(api_header_file)"); fi
fi
# The gateway's settings digest takes in its key, which exists from here on (api_key_ensure creates it on a first start).
GATEWAY_RUN=("${GATEWAY_RUN[@]:0:5}" --label "tf.recipe.gateway=$(gateway_identity)" "${GATEWAY_RUN[@]:5}")
step 3 "Launch: container $CONTAINER_NAME, rank 1 on the worker, then rank 0 here"
# spill files a crashed run left as root go back to their directory's owner (scripts/port.sh; nothing with SPILL_GIB=0)
(( DRY )) || port_spill_fix_both
cmd="tf-dsv41-lanes $MODEL_ARG /rank-cache 1 2 ${LINK_HEAD_ADDR[1]:-$MASTER_ADDR} $MASTER_PORT $KIT --engram /engram --token-map $DATA/dsv41_token_map.json $FOLLOW"
[[ "$RDMA" == 0 ]] || cmd+=" --rdma ${NODE_HCAS[1]}"   # this node's RoCE devices
spill1=$(port_spill_args 1)   # rank 1's spill directory (scripts/port.sh; empty, and no extra space, with it off)
remote="mkdir -p \$HOME/.cache/dsv41-tensorfold && docker run -d --name $CONTAINER_NAME $RUN_ARGS -v $NFS_VOLUME:/hf:ro -v $WORKER_ENGRAM_DIR:/engram:ro -v $WORKER_CACHE_DIR:/rank-cache:ro ${spill1:+$spill1 }$(envs "${ENGINE_ENV[@]}" "TF_RDMA_GID_INDEX=${NODE_GID[1]}" ${PORT_RANK_SHA[1]:+"TF_DS_RANK_CACHE_SHA=${PORT_RANK_SHA[1]}"} ${PORT_RANK1_ENV[@]+"${PORT_RANK1_ENV[@]}"}) $(rank_nccl_env 1) $IMAGE $cmd"
if (( DRY )); then printf '[dry-run] rank 1 on %s:\n  %s\n' "$(worker_host 1)" "$remote"
else
  log "Rank 1 on $(worker_host 1): $cmd"
  worker 1 "$remote" >/dev/null || die "could not start rank 1 on $(worker_host 1)"
fi
ENV0=(TF_TP_WORLD=2 "TF_TP_MASTER=${LINK_HEAD_ADDR[1]:-$MASTER_ADDR}" "TF_TP_PORT=$MASTER_PORT" "TF_DS_RANK_CACHE=$DATA/rank-cache" "TF_DS_KIT=$KIT"
      "TENSORFOLD_CUDA_KERNELS=$KIT/aot" "TF_DS_ENGRAM=$([[ -n "$HEAD_ENGRAM_DIR" ]] && echo /engram || engram_in_container)" "TF_DS_TOKEN_MAP=$DATA/dsv41_token_map.json"
      "TF_DS_GRAPHS=$GRAPHS" "TF_DS_GRAPH_CAP=$GRAPH_CAP" "TF_DS_HC_SIDE=$SIDE" "TF_DS_L2_PREFETCH=$PREFETCH" "TF_DS_ENGRAM_AIO=$ENGRAM_AIO" "TF_RDMA_GID_INDEX=${NODE_GID[0]}")
if (( DRY )); then HEAD_HF=${HEAD_NFS_SERVER:+$HEAD_NFS_VOLUME}; HEAD_HF=${HEAD_HF:-$HF_CACHE}; else HEAD_HF=$(head_hf_mount); fi
here=(docker run -d --name "$CONTAINER_NAME" $RUN_ARGS -v "$HEAD_HF:/hf:ro")
read -ra port0 <<<"$(port_spill_args 0)"; here+=(${port0[@]+"${port0[@]}"})   # rank 0's spill directory (scripts/port.sh)
if [[ -n "$HEAD_ENGRAM_DIR" ]]; then
  here+=(-v "$HEAD_ENGRAM_DIR:/engram:ro")
fi
[[ "$RDMA" == 0 ]] || ENV0+=("TF_RDMA_DEVICES=${NODE_HCAS[0]}")
[[ -z "${PORT_RANK_SHA[0]:-}" ]] || ENV0+=("TF_DS_RANK_CACHE_SHA=${PORT_RANK_SHA[0]}")   # this rank's own weight file's
if [[ "$API_KEY" == 1 ]]; then here+=(-v "$API_KEY_FILE:$API_KEY_IN_CONTAINER:ro"); fi
for x in "${ENV0[@]}" "${ENGINE_ENV[@]}"; do here+=(-e "$x"); done
rank_nccl_args 0
here+=("${NCCL_ARGS[@]}" "$IMAGE" "${SERVE[@]}")
if (( DRY )); then
  printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${here[@]}")"
  if [[ "$API_GATEWAY" == 1 ]]; then
    printf '[dry-run] gateway:\n  %s\n' "$(printf '%q ' "${GATEWAY_RUN[@]}" "${GATEWAY_ARGS[@]}")"
  fi
  exit 0
fi
log "Rank 0 here: ${SERVE[*]}"
"${here[@]}" >/dev/null
# FOREGROUND=1 attaches here, as on main; with the gateway it needs the engine up first, so it attaches after step 4
[[ "${FOREGROUND:-0}" != 1 || "$API_GATEWAY" == 1 ]] || foreground

# ---------------------------------------------------------------- 4. load
API_URL=$URL
if [[ "$API_GATEWAY" == 1 ]]; then URL="http://127.0.0.1:$ENGINE_PORT"; fi   # the engine is on loopback (SERVE_HOST)
step 4 "Loading: ~98 GiB of weights on each Spark, then Engram and the caches (a few minutes)"
docker logs -f "$CONTAINER_NAME" 2>&1 | sed -u "s/^/  ${D}│${R} /" &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true; gateway_remove_after_failure' EXIT
fail() {
  kill $LOGS_PID 2>/dev/null || true
  printf '%s── rank 1 (%s): last log lines ──%s\n' "$D" "$(worker_host 1)" "$R"; worker 1 docker logs --tail 15 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  gateway_dump_logs
  die "$1"
}
start=$SECONDS
until curl -sf --max-time 5 "${AUTH[@]}" "$URL/v1/models" >/dev/null 2>&1; do
  running_here || fail "rank 0 exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  running_worker 1 || fail "rank 1 on $(worker_host 1) exited before the server was ready"
  (( SECONDS - start < WAIT_TIMEOUT )) || fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); the ranks are still running"
  sleep 5
done
kill $LOGS_PID 2>/dev/null || true
log "Server answered after $((SECONDS - start))s"

if [[ "$API_GATEWAY" == 1 ]]; then
  log "Starting ds41-gateway on $HOST:$PORT (engine on 127.0.0.1:$ENGINE_PORT)"
  docker rm -f ds41-gateway >/dev/null 2>&1 || true
  "${GATEWAY_RUN[@]}" "${GATEWAY_ARGS[@]}" >/dev/null || fail "could not start ds41-gateway"
  for _ in $(seq 1 24); do
    curl -sf --max-time 3 "$API_URL/health" >/dev/null 2>&1 && break
    if ! running_gateway; then gateway_dump_logs; gateway_remove_after_failure; fail "ds41-gateway exited during startup"; fi
    sleep 2
  done
  curl -sf --max-time 5 "$API_URL/health" >/dev/null 2>&1 || { gateway_dump_logs; fail "ds41-gateway did not become healthy"; }
  log "Gateway answered on $API_URL"
  if [[ "${FOREGROUND:-0}" == 1 ]]; then
    trap - EXIT                                        # foreground() removes the gateway itself when the ranks end
    foreground
  fi
fi

# ---------------------------------------------------------------- 5. smoke test
step 5 "Smoke test: one chat completion through both ranks"
if [[ "$API_GATEWAY" == 1 ]]; then
  if smoke=$(curl -s --max-time 300 "${AUTH[@]}" "$API_URL/v1/chat/completions" -H 'Content-Type: application/json' \
               -d "{\"model\": \"$SERVED_NAME\", \"max_tokens\": 16, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
             python3 -c 'import json,sys; r = json.load(sys.stdin); c = r["choices"][0]["message"].get("content") or ""; assert c.strip(); assert len(c.strip()) < 8 or len(set(c.strip())) > 2, "one repeated character"; print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
    log "OK through the gateway: $smoke"
  else
    fail "the gateway smoke test failed (no reply text, or one character repeated); the ranks and gateway are still running"
  fi
fi
# The engine's own API too: a gateway conversion bug must not be able to hide an engine problem. Here URL is the engine.
if smoke=$(curl -s --max-time 300 "${AUTH[@]}" "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED_NAME\", \"max_tokens\": 16, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); c = r["choices"][0]["message"].get("content") or ""; assert c.strip(); assert len(c.strip()) < 8 or len(set(c.strip())) > 2, "one repeated character"; print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
  if [[ "$API_GATEWAY" == 1 ]]; then log "OK at the engine: $smoke"; else log "OK: $smoke"; fi
else
  fail "the smoke test request failed (no reply text, or one character repeated); the ranks are still running"
fi
# Headroom: how far MemAvailable sits above memguard's floor once the weights, graphs and caches are loaded (issue #4). Nothing in
# the engine holds MemAvailable above that floor while it runs: a long prompt's workspace comes out of this headroom, and memguard
# kills the rank if it reaches the floor. A head that also runs a desktop and other containers has the least.
if awk -v g="$MEMGUARD_GIB" 'BEGIN {exit !(g > 0)}'; then
  mem_here=$(mem_available_gib)
  mem_there=$(worker 1 "awk '/^MemAvailable:/ {printf \"%.1f\", \$2 / 1048576}' /proc/meminfo" 2>/dev/null || echo "?")
  log "Memory after the load: ${mem_here} GiB available on this Spark, ${mem_there} GiB on $(worker_host 1); memguard's floor is $MEMGUARD_GIB GiB"
  for pair in "this Spark:$mem_here" "$(worker_host 1):$mem_there"; do
    v=${pair##*:}; [[ "$v" =~ ^[0-9.]+$ ]] || continue
    awk -v a="$v" -v g="$MEMGUARD_GIB" 'BEGIN {exit !(a < g + 3)}' &&
      warn "only $v GiB available on ${pair%:*}, within 3 GiB of memguard's floor ($MEMGUARD_GIB GiB): a long prompt can take that and memguard then kills the rank. Free memory there (desktop, other containers: README, Memory and safety) before relying on this server"
  done
fi
trap 'kill $LOGS_PID 2>/dev/null || true; save_log "$LOG_DIR" gateway ds41-gateway "$LOG_KEEP" >/dev/null 2>&1 || true' EXIT
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
[[ "$IP" != *:* ]] || IP="[$IP]"   # an IPv6 address goes in brackets in a URL
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED_NAME" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   $( [[ "$API_GATEWAY" == 1 ]] && echo "(gateway: DeepSeek API parity on; engine on 127.0.0.1:$ENGINE_PORT)" || echo "(model: $SERVED_NAME)" )
    Access   $( [[ "$API_KEY" == 1 ]] && echo "API key required: Authorization: Bearer \$(cat $API_KEY_FILE)" || echo "open: no key, anyone who reaches port $PORT can use it (API_KEY=1 requires one)")
    Window   $CONTEXT tokens · $PARALLEL at once · EXL3 2.9 bpw · drafts $( [[ "$DRAFTS" == 1 ]] && echo on || echo off) · images on$(port_summary)
    Logs     docker logs -f $CONTAINER_NAME   (rank 1: ssh $(worker_host 1) docker logs -f $CONTAINER_NAME)
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
