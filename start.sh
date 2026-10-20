#!/usr/bin/env bash
# Serve Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw (an EXL3 quantization of DeepSeek-V4.1-Flash-1017) on two DGX
# Sparks with TensorFold's native engine, end to end: runs scripts/prepare.sh when the image, the checkpoint, the weight
# files, the kit or the worker's NFS volume are not ready yet (the first run, or after the patches change), starts rank 1
# on the worker and rank 0 here, which serves the API on port 8888, waits until the OpenAI API answers, then runs a smoke
# test. The worker reads the checkpoint from this Spark over NFS. Stop it with ./stop.sh.
# The shape of this script (setup, checks, launch, load, smoke test) is the GLM-5.3-Flash recipe's and the four-Spark
# DeepSeek-V4.1 recipe's (MiaAI-Lab); the engine's flags are BertholomusAI's two-node launch.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                  # scripts/config.sh's defaults (4 requests at once, a 1,048,576-token window, drafts,
#                               # images); if the server already runs, says so and leaves it alone
#   ./start.sh restart          # stop both ranks (./stop.sh), then start again (e.g. after changing settings)
#   CONTEXT=262144 ./start.sh restart
#   ABLIT=1 ./start.sh restart   # Mia EAQ 2.9 bpw + Keys overlay; requires your HF token and gated repo access
#   API_KEY=1 ./start.sh restart  # require an API key (generated in $STATE_DIR/api-key, mode 600; see scripts/config.sh)
#   FOREGROUND=1 ./start.sh       # stay attached to rank 0's log and exit with its code (for systemd Restart=on-failure)
#   DRY_RUN=1 ./start.sh        # print the link found and both ranks' docker commands, change nothing and start nothing
# Extra arguments go to rank 0's `tensorfold-native serve` after the defaults, so they win.
# Setup: WORKER (an ssh target, key-based) in scripts/local.sh.
# Settings, from the environment, scripts/local.sh or ./.env (defaults and their notes in scripts/config.sh):
#   serving  CONTEXT, PARALLEL, ROUND_ROWS, MAX_TOKENS, THINKING, REASONING_EFFORT, THINKING_BUDGET, TEMPERATURE, TOP_P, TOP_K, DRAFTS,
#            SERVED_NAME, HOST, PORT, RDMA, GRAPHS, SIDE, PREFETCH, ENGRAM_AIO, REPLAY_FLOOR
#   nodes    WORKER, FABRIC_PEER, MASTER_ADDR, MASTER_PORT, SOCKET_IFNAME, NCCL_RAILS, NCCL_DEBUG
#   files    ABLIT (0 | 1), ABLIT_REVISION, MODEL_ID, MODEL_REVISION, ENGRAM_ID, ENGRAM_REVISION, HF_CACHE, NFS_PATH, NFS_SERVER, NFS_VOLUME,
#            WORKER_ENGRAM_DIR, WORKER_CACHE_DIR, STATE_DIR, LOG_DIR, LOG_KEEP
#   engine   TF_REPO, TF_REF, INPUTS_REF, KIT_REPO, KIT_REF, ZIG_VERSION, BASE_IMAGE, IMAGE, CONTAINER_NAME
#   setup    PREPARE (auto | 1 | 0), WAIT_TIMEOUT (seconds, default 2400), DRY_RUN=1
#   memory   MIN_AVAIL_GIB (refuse to start below it), MIN_AVAIL_WAIT_S (re-read for this long first), MEMGUARD_GIB (kill a rank's container below it), HEAD_NFS_SERVER,
#            HEAD_NFS_VOLUME (rank 0 reads the checkpoint over NFS through a docker volume)
#   ranks    every TENSORFOLD_* and TF_DS_* variable you set goes to both ranks
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
(( PARALLEL <= 16 )) || die "PARALLEL is 1 to 16 (the engine's streams), not $PARALLEL"
[[ -z "$ROUND_ROWS" ]] || { [[ "$ROUND_ROWS" =~ ^[0-9]+$ ]] && (( ROUND_ROWS >= 16 && ROUND_ROWS <= 48 )); } || die "ROUND_ROWS is empty or 16 to 48, not $ROUND_ROWS"
[[ -z "$THINKING_BUDGET" || "$THINKING_BUDGET" =~ ^[1-9][0-9]*$ ]] || die "THINKING_BUDGET is a positive count of reasoning tokens (or empty for no cap), not $THINKING_BUDGET"
[[ -z "$REASONING_EFFORT" || "$REASONING_EFFORT" =~ ^(low|medium|high|xhigh)$ ]] || die "REASONING_EFFORT is low, medium, high or xhigh (or empty), not $REASONING_EFFORT"
[[ "$MIN_AVAIL_GIB" =~ ^[0-9]+$ ]] || die "MIN_AVAIL_GIB is a whole number of GiB (0 turns it off), not $MIN_AVAIL_GIB"
[[ "$MIN_AVAIL_WAIT_S" =~ ^[0-9]+$ ]] || die "MIN_AVAIL_WAIT_S is a whole number of seconds (0 reads the memory once), not $MIN_AVAIL_WAIT_S"
[[ "$MEMGUARD_GIB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MEMGUARD_GIB is a number of GiB (2.5 is fine; 0 turns it off), not $MEMGUARD_GIB"
[[ "$MEMGUARD_HOLD_S" =~ ^[0-9]+(\.[0-9]+)?$ && "$MEMGUARD_HARD_GIB" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MEMGUARD_HOLD_S is a number of seconds and MEMGUARD_HARD_GIB a number of GiB, not $MEMGUARD_HOLD_S and $MEMGUARD_HARD_GIB"
[[ "$PREFILL_INTERLEAVE_MS" =~ ^[0-9]+$ ]] && (( PREFILL_INTERLEAVE_MS <= 5000 )) || die "PREFILL_INTERLEAVE_MS is 0 to 5000 milliseconds, not $PREFILL_INTERLEAVE_MS"
for v in THINKING DRAFTS RDMA GRAPHS SIDE PREFETCH ENGRAM_AIO REPLAY_FLOOR API_KEY; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "${RANK1_PROFILE:-}" =~ ^[012]?$ ]] || die "RANK1_PROFILE is empty, 0, 1 or 2, not '${RANK1_PROFILE:-}'"
nccl_extra_collect
check_workers

# rank 0's command (its serve flags need no link): built first, so the flags can be checked before anything is stopped
SERVE=(tensorfold-native serve "$(served_in_container)" --backend cuda --host "$HOST" --port "$PORT" --name "$SERVED_NAME"
       --context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE+=(--thinking); else SERVE+=(--no-thinking); fi
[[ -z "$REASONING_EFFORT" ]] || SERVE+=(--reasoning-effort "$REASONING_EFFORT")
[[ -z "$THINKING_BUDGET" ]] || SERVE+=(--thinking-budget "$THINKING_BUDGET")
# drafts off on rank 0 as on rank 1 (a follower without the drafter refuses a drafted round)
[[ "$DRAFTS" == 1 ]] || SERVE+=(--no-drafts)
[[ "$API_KEY" != 1 ]] || SERVE+=(--api-key-file "$API_KEY_IN_CONTAINER")
SERVE+=("$@")

B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · 2 x DGX Spark · %s at once · %s-token window · EXL3 2.9 bpw · port %s%s\n\n' "$D" "$MODEL_ID" "$PARALLEL" "$CONTEXT" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
need_workers
(( DRY )) && log "DRY_RUN=1: printing the docker commands; nothing is stopped, started or prepared"

API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
URL="http://$API_HOST:$PORT"
running_here()   { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
running_worker() { [[ "$(worker "$1" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
if (( ! DRY )) && [[ "$MODE" == start ]] && running_here && running_worker 1; then
  # Preparation may have run separately while the old server stayed up: inspect the live ranks too.
  live_head=$(docker inspect -f '{{index .Config.Labels "tf.recipe.ablit"}}' "$CONTAINER_NAME")
  live_worker=$(worker 1 "docker inspect -f '{{index .Config.Labels \"tf.recipe.ablit\"}}' '$CONTAINER_NAME'")
  # Containers launched before this flag existed are stock.
  [[ -n "$live_head" && "$live_head" != '<no value>' ]] || live_head=0
  [[ -n "$live_worker" && "$live_worker" != '<no value>' ]] || live_worker=0
  [[ "$live_head" == "$(ablit_identity)" && "$live_worker" == "$(ablit_identity)" ]] || \
    die "the running ranks use a different ABLIT mode or overlay revision; run ./start.sh restart with ABLIT=$ABLIT"
  [[ "$(prepared_state 2>/dev/null)" == "$(cat "$PREPARED_MARKER" 2>/dev/null)" ]] || \
    die "the requested configuration differs from the prepared server; run ./start.sh restart with the same settings (including ABLIT)"
  log "$CONTAINER_NAME is already running on both Sparks (port $PORT): nothing to do. ./start.sh restart restarts it, ./stop.sh stops it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
step 1 "Setup: image, checkpoint, Engram's tables, weight files and kit; the worker's NFS volume and disk"
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state 2>/dev/null)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
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
if (( ! DRY )); then
  # Preparation's marker cannot detect an export/root change since the last run.
  worker_files_check 1
  if running_here || running_worker 1; then ./stop.sh; fi
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  worker 1 "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1; true"
  if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"; fi
  # memory: a rank that takes all of a Spark's memory freezes the machine, so refuse to start on one that cannot hold it
  if (( MIN_AVAIL_GIB > 0 )); then
    await_memory || true
    here_gb=$HERE_GB
    if awk -v a="$here_gb" -v m="$MIN_AVAIL_GIB" 'BEGIN {exit !(a < m)}'; then
      shm=$(du -sm /dev/shm 2>/dev/null | cut -f1); shm=${shm:-0}
      die "only $here_gb GiB memory available here, $MIN_AVAIL_GIB GiB needed (the weights plus the engine's reserve): free memory first${shm:+ (/dev/shm holds $shm MiB)}; MIN_AVAIL_GIB=0 skips this check, and a rank that takes all the memory freezes the Spark"
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
  local watch w code
  trap './stop.sh; exit 130' INT TERM
  ( exec 8>&-                                        # not holding start.sh's lock once start.sh has exited
    while sleep 30; do
      running_here || exit 0
      worker 1 true 2>/dev/null || continue         # a worker out of reach for a moment says nothing about its rank
      running_worker 1 && continue
      warn "rank 1 on $(worker_host 1) exited: stopping rank 0"
      docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1
      exit 1
    done ) &
  watch=$!
  docker logs -f "$CONTAINER_NAME" || true
  kill "$watch" 2>/dev/null || true
  w=0; wait "$watch" || w=$?
  code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)
  [[ "$w" != 1 || "$code" != 0 ]] || code=1         # the worker's rank failed, even if rank 0 then shut down cleanly
  worker 1 "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME'" >/dev/null 2>&1 || true
  exit "$code"
}
# What both ranks' engine reads, as BertholomusAI's two-node launch has it (HF_CACHE at /hf: this Spark's own, the worker's over NFS)
ENGINE_ENV=("TF_DS_REPLAY_FLOOR=$REPLAY_FLOOR" "TF_DS_STREAMS=$PARALLEL" "HF_HUB_OFFLINE=$HF_HUB_OFFLINE")
[[ -z "$ROUND_ROWS" ]] || ENGINE_ENV+=("TF_DS_ROUND_ROWS=$ROUND_ROWS")
(( PREFILL_INTERLEAVE_MS == 0 )) || ENGINE_ENV+=("TF_DS_PREFILL_INTERLEAVE_MS=$PREFILL_INTERLEAVE_MS")
while IFS='=' read -r name _; do ENGINE_ENV+=("$name=${!name}"); done < <(env | grep -E '^(TENSORFOLD|TF_DS)_[A-Z0-9_]+=' | grep -vE '^TF_DS_(REPLAY_FLOOR|STREAMS|ROUND_ROWS|GRAPHS|HC_SIDE|L2_PREFETCH|ENGRAM_AIO|RANK_CACHE|KIT|ENGRAM|TOKEN_MAP)=' || true)
envs() { local x; for x in "$@"; do printf -- '-e %q ' "$x"; done; }
RUN_ARGS="--label tf.recipe.ablit=$(ablit_identity) --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband"
DATA=/hf/$DATA_SUB
KIT=/hf/$KIT_SUB/kit
# rank 1 takes the round's settings as flags only (tf-dsv41-lanes reads no TF_DS_GRAPHS & co.): without them it runs
# every round ungraphed, with no side stream, prefetch or AIO, and rank 0 waits on it each round. Its pool is rank 0's
# context, and its streams rank 0's (TF_DS_STREAMS): with other values the two ranks wait on each other.
FOLLOW="--pool $CONTEXT --streams $PARALLEL --drafts $DRAFTS --graphs $GRAPHS --side $SIDE --prefetch $PREFETCH --aio $ENGRAM_AIO"
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
step 3 "Launch: container $CONTAINER_NAME, rank 1 on the worker, then rank 0 here"
cmd="tf-dsv41-lanes $MODEL_ARG /rank-cache 1 2 ${LINK_HEAD_ADDR[1]:-$MASTER_ADDR} $MASTER_PORT $KIT --engram /engram --token-map $DATA/dsv41_token_map.json $FOLLOW"
[[ "$RDMA" == 0 ]] || cmd+=" --rdma ${NODE_HCAS[1]}"   # this node's RoCE devices
remote="mkdir -p \$HOME/.cache/dsv41-tensorfold && docker run -d --name $CONTAINER_NAME $RUN_ARGS -v $NFS_VOLUME:/hf:ro -v $WORKER_ENGRAM_DIR:/engram:ro -v $WORKER_CACHE_DIR:/rank-cache:ro $(envs "${ENGINE_ENV[@]}" "TF_RDMA_GID_INDEX=${NODE_GID[1]}") $(rank_nccl_env 1) $IMAGE $cmd"
if (( DRY )); then printf '[dry-run] rank 1 on %s:\n  %s\n' "$(worker_host 1)" "$remote"
else
  log "Rank 1 on $(worker_host 1): $cmd"
  worker 1 "$remote" >/dev/null || die "could not start rank 1 on $(worker_host 1)"
fi
ENV0=(TF_TP_WORLD=2 "TF_TP_MASTER=${LINK_HEAD_ADDR[1]:-$MASTER_ADDR}" "TF_TP_PORT=$MASTER_PORT" "TF_DS_RANK_CACHE=$DATA/rank-cache" "TF_DS_KIT=$KIT"
      "TENSORFOLD_CUDA_KERNELS=$KIT/aot" "TF_DS_ENGRAM=$([[ -n "$HEAD_ENGRAM_DIR" ]] && echo /engram || engram_in_container)" "TF_DS_TOKEN_MAP=$DATA/dsv41_token_map.json"
      "TF_DS_GRAPHS=$GRAPHS" "TF_DS_HC_SIDE=$SIDE" "TF_DS_L2_PREFETCH=$PREFETCH" "TF_DS_ENGRAM_AIO=$ENGRAM_AIO" "TF_RDMA_GID_INDEX=${NODE_GID[0]}")
if (( DRY )); then HEAD_HF=${HEAD_NFS_SERVER:+$HEAD_NFS_VOLUME}; HEAD_HF=${HEAD_HF:-$HF_CACHE}; else HEAD_HF=$(head_hf_mount); fi
here=(docker run -d --name "$CONTAINER_NAME" $RUN_ARGS -v "$HEAD_HF:/hf:ro")
if [[ -n "$HEAD_ENGRAM_DIR" ]]; then
  here+=(-v "$HEAD_ENGRAM_DIR:/engram:ro")
fi
[[ "$RDMA" == 0 ]] || ENV0+=("TF_RDMA_DEVICES=${NODE_HCAS[0]}")
if [[ "$API_KEY" == 1 ]]; then here+=(-v "$API_KEY_FILE:$API_KEY_IN_CONTAINER:ro"); fi
for x in "${ENV0[@]}" "${ENGINE_ENV[@]}"; do here+=(-e "$x"); done
rank_nccl_args 0
here+=("${NCCL_ARGS[@]}" "$IMAGE" "${SERVE[@]}")
if (( DRY )); then printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${here[@]}")"; exit 0; fi
log "Rank 0 here: ${SERVE[*]}"
"${here[@]}" >/dev/null
[[ "${FOREGROUND:-0}" != 1 ]] || foreground

# ---------------------------------------------------------------- 4. load
step 4 "Loading: ~98 GiB of weights on each Spark, then Engram and the caches (a few minutes)"
docker logs -f "$CONTAINER_NAME" 2>&1 | sed -u "s/^/  ${D}│${R} /" &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
fail() {
  kill $LOGS_PID 2>/dev/null || true
  printf '%s── rank 1 (%s): last log lines ──%s\n' "$D" "$(worker_host 1)" "$R"; worker 1 docker logs --tail 15 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
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

# ---------------------------------------------------------------- 5. smoke test
step 5 "Smoke test: one chat completion through both ranks"
if smoke=$(curl -s --max-time 300 "${AUTH[@]}" "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED_NAME\", \"max_tokens\": 16, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); c = r["choices"][0]["message"].get("content") or ""; assert c.strip(); assert len(c.strip()) < 8 or len(set(c.strip())) > 2, "one repeated character"; print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
  log "OK: $smoke"
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
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED_NAME" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED_NAME)
    Access   $( [[ "$API_KEY" == 1 ]] && echo "API key required: Authorization: Bearer \$(cat $API_KEY_FILE)" || echo "open: no key, anyone who reaches port $PORT can use it (API_KEY=1 requires one)")
    Window   $CONTEXT tokens · $PARALLEL at once · EXL3 2.9 bpw · drafts $( [[ "$DRAFTS" == 1 ]] && echo on || echo off) · images on
    Logs     docker logs -f $CONTAINER_NAME   (rank 1: ssh $(worker_host 1) docker logs -f $CONTAINER_NAME)
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
