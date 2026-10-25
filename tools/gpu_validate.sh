#!/usr/bin/env bash
# The port's GPU validation runbook as a script, for the coordinator to run on gx10-a (rank 0) later. Lean by design:
# targeted suites for what changed, each with a JSON receipt, not multi-hour reruns (tools/PORT-VALIDATION.md).
#
# Usage: tools/gpu_validate.sh [--dry-run] [--phases LIST] [--configs LIST] [--no-restore]
#        tools/gpu_validate.sh kill-ranks [--resolve FILE | --pids FILE] [--rank 1 [--signal STOP]]
#                                  # kill -9 both ranks' engine processes by PID (spill_check's prekill); --resolve only
#                                  # writes both PIDs to FILE, --pids kills the PIDs FILE holds (one quick step);
#                                  # --rank 1 kills rank 1's alone (kv_shard_check_e2e's kill-rank1, pair_check);
#                                  # --signal STOP stops it instead (a hung peer: the stall monitor's case)
#        tools/gpu_validate.sh restore   # stop a PORT container if one runs, put the daily service back, remove the hold
#
# Phases, in order (default: all):
#   preflight  this host, commands, the worker over ssh, sudo, directories (SPILL_DIR never under /tmp)
#   checkout   the port recipe in its own checkout PORT_DIR under ~/sess (git clone/fetch PORT_REPO at PORT_REF), its
#              scripts/local.sh copied from the daily checkout, its own STATE_DIR (the daily prepared marker untouched);
#              the tools (this script's branch, TOOLS_REF) in TOOLS_DIR, used for every suite
#   hold       touch ~/.deepseek1017-hold (the watchdog leaves the stack alone); from here a failure or ^C restores
#   baseline   on the RUNNING daily service: health, exact.py / spill_check / kv_shard_check_e2e / prefix_fanout records
#   stop       sudo systemctl stop deepseek1017.service; no container left on either Spark
#   build      the port image: scripts/prepare.sh in PORT_DIR (its own STATE_DIR); the daily's weight files on both
#              nodes and its served folder must be unchanged after it, or the run stops (and restores)
#   dropcache  sync; drop page caches on both nodes (also before every configuration's start)
#   suites     per configuration (--configs; default: off,spill,spill-mismatch,shard,shard-copies,
#              shard-pressure,shard-check,shard-mismatch,shard-pool-mismatch,shard-kill,spill-shard,spill-pictures,
#              spill-pictures-mismatch,pair-kill,pair-stop; opt-in: shard-p2, shard-p4, shard-nodraft, shard-perf,
#              shard-perf-fetch, spill-shard-pressure): start the port server with its switches (a fresh SPILL_DIR
#              for every spill configuration), run its suites, save receipts, health, metrics, the start-up lines
#              and logs, stop. A suite that ends a rank restarts the pair itself
#   restore    stop the port server (only a container carrying the port's tf.recipe.port label), then: the daily
#              service untouched -> only the hold goes; stopped -> systemctl start (restart when the oneshot unit is still
#              active), wait for /health, smoke, exact compare; remove the hold
#
#
# The configurations and the engine's GPU validation lists they cover (docs/port/spill-lanes.md "What needs GPU
# validation" = S1..S6, docs/port/shard-lanes.md "GPU validation needed" = K1..K9; KV checks are kv_shard_check_e2e's):
#   off             K1   switches unset: exact == daily; the start-up lines saved (the `arena ... used` figure; the
#                        daily engine prints none, so it is recorded, not compared); the shared-prefix fan-out
#                        (prefix_fanout: duplicate prefill tokens per level, replies == daily); BENCH=1 adds a bench run
#   spill           S1   both ranks agree and open their stores (rank 0's "the ranks agreed" line, staging
#                        SPILL_STAGING_MIB MiB); S2/S3 keep -> evict -> restore from disk == baseline with drafts on
#                        (spill_check main: the resumed replies are DSpark-drafted), restart over the same dirs
#                        (prerestart/drop/postrestart), S4 dual-restore's decoder gap and restore TTFT per size,
#                        S5 corrupt/truncate/remove on either rank, kill -9 mid-write (prekill/postkill), the floor;
#                        S6 the staging and CUDA-budget lines saved (staging before the pool: read the order there)
#   spill-mismatch  S1   rank 1 with another TF_DS_SPILL_MIN_TOKENS (RANK1_EXTRA_ENV): the start is refused and a
#                        rank's log has the engine's "rank 1's configuration differs from rank 0's in features"
#   shard           K2   KV_SHARD=1, CARRY=0, TWINS=0, POOL=SHARD_POOL, the default PARALLEL: exact; kv limits (window
#                        and pool from /health parts x part_rows and the start-up line), owners, mixed, needles, idle
#   shard-p2/-p4    K2   (opt-in) the same at PARALLEL=2 / 4: exact; kv owners, mixed, idle
#   shard-nodraft   K2   (opt-in) the same with DRAFTS=0: exact against the drafted daily baseline (drafts are exact)
#   shard-copies    K3   CARRY=1, TWINS=1, POOL=SHARD_POOL: kv burst (same-part copies), idle; exact
#   shard-pressure  K4-6 small parts (PRESSURE_POOL / PRESSURE_CONTEXT), CARRY=1, TWINS=1: kv carry (moved bytes per
#                        token in the receipt), twins (no fetch in a twinned pass, twins released), move (a live move
#                        decodes on exactly), cancel-transfer, idle
#   shard-check     K7   the pressure configuration with KV_SHARD_CHECK=1 (a planner digest check every round): kv
#                        carry, idle; exact
#   shard-mismatch  K7   rank 1 with TF_DS_KV_SHARD_CARRY=0 (RANK1_EXTRA_ENV): the start is refused (not a request),
#                        a rank's log has "differs from rank 0's in features" (or ShardLayoutMismatch / KvShardMismatch)
#   shard-pool-mismatch K7 rank 1 with another window (RANK1_EXTRA_ENV=TF_DS_WINDOW=SHARD_WINDOW - 4096: zig/patches/
#                        0008's window, which a follower reads; Model.open compares every rank's pool and window):
#                        refused, a rank's log has KvShardMismatch / ShardLayoutMismatch (or their explanations)
#   shard-kill      K7   the pressure configuration: kv kill-rank1 (rank 1 killed during a carry: the request in flight
#                        ends within --kill-deadline, no hang), then a clean restart: exact, kv idle
#   shard-perf, shard-perf-fetch  K8/K9  (opt-in) bench.py prefill/decode/multi at 1, 2, 4, 8 streams with TWINS=1 / TWINS=0
#                        (and the daily's, recorded at baseline), for the comparison
# docs/port/spill-shard.md "What needs GPU validation" = SS1..SS10, spill-pictures.md = SP1..SP3, robustness.md "GPU
# validation needed" = R1..R6:
#   off             R1   (above) every switch at its default: exact, and rank 0's start-up line "a broken rank pair
#                        ends this server with status 75" (off-startup)
#   spill-shard     SS1-5,9  the daily configuration (spill + KV_SHARD=1, POOL=SHARD_POOL): both ranks agree (rank 0's
#                        "the ranks agreed"); exact; kv limits, owners, mixed, idle; spill_check main (keep, evict,
#                        restore from disk, corrupt on rank 0 / truncate and remove on rank 1, cancel, dual-restore,
#                        sizes, settled); each rank's store size (du, recorded: SS3's 890 / 170 B a token is read
#                        there); restart over the same dirs (prerestart, drop, postrestart: recovered points, restore)
#   spill-shard-pressure SS6-8 (opt-in) small parts with spill, CARRY=1, TWINS=1: kv carry, twins, move, idle while
#                        the tier writes; spill_check evict, dual-restore, settled (kv_shard_staged, forced_staging,
#                        writes_failed are in the metrics snapshots)
#   spill-pictures  SP1  spill on: spill_check pictures (P1, a picture before its text, keeps a point with the picture:
#                        pictures_kept +1 on both ranks; restored from disk after the eviction == baseline), settled;
#                        restart over the same dirs with a picture conversation (P2) restored after it
#   spill-pictures-mismatch SP3 rank 1 with TF_DS_IMAGE_CACHE=0 (RANK1_EXTRA_ENV): refused, "differs ... in features"
#   pair-kill       R2   spill and KV sharding off (as off): PAIR_STREAMS streams decode, rank 1's engine is killed
#                        (kill -9 by PID): within PAIR_KILL_DEADLINE every stream ends (no hang), /health goes down (or
#                        503) and rank 0's container is `exited 75` (docker inspect), its log with the engine's "exiting
#                        with status 75"; then the suite restarts the pair (./start.sh restart, as a supervisor would:
#                        its stop.sh saves the exited rank 0's log, which must have that line) and runs exact
#   pair-stop       R3   the same with rank 1 SIGSTOPped (a hung peer, nothing errors), on the spill-shard
#                        configuration: the stall monitor (120 s) and the abort (15 s) end rank 0 with 75 within
#                        PAIR_STOP_DEADLINE; restart and exact as above
# Not covered here (no server-level handle): S2's tf-dsv41-lanes "drop_memory" runs (the server suites cover the same
# keep/restore path), K6's per-layer ch.o equality (kit tooling, --heads64), K8's per-transfer Dev.wait cost on its own
# (only as carry admission time in the carry receipt); SS4's drop_memory restores and the part a restore lands in,
# SS10's packing costs; SP2's first logits of a cut straddling a picture (kit tooling); R4 (a stop of both ranks: exit
# 0; every configuration ends with one, not judged), R5 (TF_DS_PAIR_BROKEN=degrade: no start-up line), R6 (the fence
# and restores_deferred: counters and logs of the spill suites).
#
# Settings (environment): DAILY_DIR, SESS (~/sess), PORT_DIR, PORT_REPO, PORT_REF (spill-kvshard), TOOLS_REF
# (empty: the tools of PORT_DIR's own checkout, which spill-kvshard carries; a ref: that ref's tools in TOOLS_DIR),
# TOOLS_DIR, PORT_STATE_DIR, PORT_LOG_DIR, WORKER, SPILL_DIR, SPILL_GIB (200), SPILL_FLOOR_GIB (50),
# FLOOR_MARGIN_BYTES (default: half of F1's estimated write), SHARD_POOL (2101248), SHARD_WINDOW (1048576),
# PRESSURE_POOL (524288), PRESSURE_CONTEXT (260096),
# RECEIPTS, API (http://127.0.0.1:8888), SERVICE, CONTAINER, HOLD, CFG_<NAME> (a configuration's VAR=value words,
# overriding the defaults below), EXACT_ARGS, SPILL_SIZES (16000,40000,100000,260000), NEEDLES (524288,max),
# PREFIX_FLAGS (off's fan-out; default --expect-one-fill: every level computes the shared prefix once; empty: measure
# only), VERIFY_RESTORE (1: exact compare after restore),
# BENCH (1: bench.py on the daily and on off too), KILL_DEADLINE (90 s), PAIR_STREAMS (8), PAIR_KILL_DEADLINE (60 s),
# PAIR_STOP_DEADLINE (200 s).
# Kills are by PID (docker inspect .State.Pid), never pkill -f. Nothing is written under /tmp on the Sparks.
set -uo pipefail
SELF=$(readlink -f "$0" 2>/dev/null || echo "$0")
HERE=$(cd "$(dirname "$SELF")" && pwd)

DRY=0; NO_RESTORE=0; PHASES=""; CONFIGS=""; CMD=run; RESOLVE_FILE=""; PIDS_FILE=""; KILL_RANK=""; KILL_SIG=9
ALL_PHASES="preflight checkout hold baseline stop build dropcache suites restore"
ALL_CONFIGS="off spill spill-mismatch shard shard-p2 shard-p4 shard-nodraft shard-copies shard-pressure shard-check shard-mismatch shard-pool-mismatch shard-kill shard-perf shard-perf-fetch spill-shard spill-shard-pressure spill-pictures spill-pictures-mismatch pair-kill pair-stop"
DEFAULT_CONFIGS="off spill spill-mismatch shard shard-copies shard-pressure shard-check shard-mismatch shard-pool-mismatch shard-kill spill-shard spill-pictures spill-pictures-mismatch pair-kill pair-stop"
while (( $# )); do
  case "$1" in
    --dry-run) DRY=1 ;;
    --phases) PHASES=${2//,/ }; shift ;;
    --configs) CONFIGS=${2//,/ }; shift ;;
    --no-restore) NO_RESTORE=1 ;;
    kill-ranks|restore) CMD=$1 ;;
    --resolve) RESOLVE_FILE=$2; shift ;;
    --pids) PIDS_FILE=$2; shift ;;
    --rank) KILL_RANK=$2; shift ;;
    --signal) KILL_SIG=$2; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$SELF"; exit 0 ;;
    *) echo "gpu_validate: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
PHASES=${PHASES:-$ALL_PHASES}; CONFIGS=${CONFIGS:-$DEFAULT_CONFIGS}
[[ -z "$KILL_RANK" || "$KILL_RANK" == 1 ]] || { echo "gpu_validate: --rank takes 1 (rank 1 alone)" >&2; exit 2; }
[[ "$KILL_SIG" == 9 || ( "$KILL_SIG" == STOP && "$KILL_RANK" == 1 ) ]] || { echo "gpu_validate: --signal takes STOP, with --rank 1" >&2; exit 2; }
for p in $PHASES; do [[ " $ALL_PHASES " == *" $p "* ]] || { echo "gpu_validate: unknown phase: $p (one of: $ALL_PHASES)" >&2; exit 2; }; done
for c in $CONFIGS; do [[ " $ALL_CONFIGS " == *" $c "* ]] || { echo "gpu_validate: unknown configuration: $c (one of: $ALL_CONFIGS)" >&2; exit 2; }; done

# ---------------------------------------------------------------- settings
DAILY_DIR=${DAILY_DIR:-$HOME/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks}
SESS=${SESS:-$HOME/sess}
PORT_DIR=${PORT_DIR:-$SESS/dsv41-port/recipe}
PORT_REPO=${PORT_REPO:-$DAILY_DIR}
PORT_REF=${PORT_REF:-spill-kvshard}                # recipe-int's branch (deepseek1017-recipe-port)
TOOLS_REF=${TOOLS_REF:-}                          # empty: PORT_DIR's own tools/ (spill-kvshard carries them)
TOOLS_DIR=${TOOLS_DIR:-$SESS/dsv41-port/tools-checkout}
PORT_STATE_DIR=${PORT_STATE_DIR:-$SESS/dsv41-port/state}
PORT_LOG_DIR=${PORT_LOG_DIR:-$SESS/dsv41-port/logs}  # the port's saved rank-0 logs (the daily's LOG_DIR untouched)
SPILL_DIR=${SPILL_DIR:-$SESS/dsv41-port/spill}
SPILL_GIB=${SPILL_GIB:-200}
SPILL_FLOOR_GIB=${SPILL_FLOOR_GIB:-50}
FLOOR_MARGIN_BYTES=${FLOOR_MARGIN_BYTES:-}   # the floor suite's floor: free space less this (default: F1's write / 2)
SHARD_POOL=${SHARD_POOL:-2101248}          # two whole 1,048,576-token owners (docs/port/shard-attn.md)
# the window the shard configuration must report: shard.plan serves min(CONTEXT, part - rows - 2), so 2,101,248 keeps
# the whole 1,048,576 (the unsharded daily pool reports 1,048,558)
SHARD_WINDOW=${SHARD_WINDOW:-1048576}
PRESSURE_POOL=${PRESSURE_POOL:-524288}     # small parts, so fillers can press one owner (carry, twins, cancel)
# the largest window two 262,144-position parts keep whole at 16-row rounds (port.sh's port_shard_pool of it is 524,288)
PRESSURE_CONTEXT=${PRESSURE_CONTEXT:-260096}
STAMP=$(date +%Y%m%d-%H%M%S)
RECEIPTS=${RECEIPTS:-$SESS/dsv41-port/receipts/$STAMP}
API=${API:-http://127.0.0.1:8888}
SERVICE=${SERVICE:-deepseek1017.service}
CONTAINER=${CONTAINER:-dsv41-1017-exl3-tf}
HOLD=${HOLD:-$HOME/.deepseek1017-hold}
SPILL_SIZES=${SPILL_SIZES:-16000,40000,100000,260000}
NEEDLES=${NEEDLES:-524288,max}
PREFIX_FLAGS=${PREFIX_FLAGS-"--expect-one-fill"}   # set but empty: measure only
EXACT_ARGS=${EXACT_ARGS:-}
VERIFY_RESTORE=${VERIFY_RESTORE:-1}
BENCH=${BENCH:-0}
BENCH_ARGS=${BENCH_ARGS:---suite prefill,decode,multi --streams 1,2,4,8 --salt gpu-validate}
KILL_DEADLINE=${KILL_DEADLINE:-90}
PAIR_STREAMS=${PAIR_STREAMS:-8}
PAIR_KILL_DEADLINE=${PAIR_KILL_DEADLINE:-60}     # kill -9: a ring or transfer error, then the 15 s deadline at most
PAIR_STOP_DEADLINE=${PAIR_STOP_DEADLINE:-200}    # SIGSTOP: the stall monitor's 120 s, the abort 15 s later, the grace
if [[ -z "${WORKER:-}" && -f "$DAILY_DIR/scripts/config.sh" ]]; then
  WORKER=$(cd "$DAILY_DIR" && bash -c 'source scripts/config.sh >/dev/null 2>&1; echo "${WORKER:-}"')
fi
WORKER=${WORKER:-gx10-b}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER")
TOOLS_GIVEN=${TOOLS:-}
TOOLS=${TOOLS:-$TOOLS_DIR/tools}
[[ -f "$TOOLS/spill_check.py" ]] || TOOLS=$HERE          # the tools beside this script until the checkout phase ran
PY=(python3 -I)

# The configurations: recipe settings for start.sh (names as spill-kvshard's scripts/port.sh and config.sh have them;
# the checkout phase refuses one they do not know). Engine variables (TF_DS_* / TENSORFOLD_*) may be added too: start.sh
# passes those to both ranks as they are. Each configuration names the switch it does not test as off (SPILL_GIB=0 /
# KV_SHARD=0), so it runs the same where a fork turns spill or KV sharding on by default (Robert's main does): off is
# both off, spill is spill alone, the shard ones are sharding alone.
CFG_off=${CFG_off:-SPILL_GIB=0 KV_SHARD=0}
CFG_spill=${CFG_spill:-KV_SHARD=0 SPILL_DIR=$SPILL_DIR WORKER_SPILL_DIR=$SPILL_DIR SPILL_GIB=$SPILL_GIB SPILL_FLOOR_GIB=$SPILL_FLOOR_GIB}
CFG_spill_mismatch=${CFG_spill_mismatch:-$CFG_spill RANK1_EXTRA_ENV=TF_DS_SPILL_MIN_TOKENS=4096}
CFG_shard=${CFG_shard:-SPILL_GIB=0 KV_SHARD=1 POOL=$SHARD_POOL KV_SHARD_CARRY=0 KV_SHARD_TWINS=0}
CFG_shard_p2=${CFG_shard_p2:-$CFG_shard PARALLEL=2}
CFG_shard_p4=${CFG_shard_p4:-$CFG_shard PARALLEL=4}
CFG_shard_nodraft=${CFG_shard_nodraft:-$CFG_shard DRAFTS=0}
CFG_shard_copies=${CFG_shard_copies:-SPILL_GIB=0 KV_SHARD=1 POOL=$SHARD_POOL KV_SHARD_CARRY=1 KV_SHARD_TWINS=1}
CFG_shard_pressure=${CFG_shard_pressure:-SPILL_GIB=0 KV_SHARD=1 POOL=$PRESSURE_POOL CONTEXT=$PRESSURE_CONTEXT KV_SHARD_CARRY=1 KV_SHARD_TWINS=1}
CFG_shard_check=${CFG_shard_check:-$CFG_shard_pressure KV_SHARD_CHECK=1}
CFG_shard_mismatch=${CFG_shard_mismatch:-SPILL_GIB=0 KV_SHARD=1 POOL=$SHARD_POOL RANK1_EXTRA_ENV=TF_DS_KV_SHARD_CARRY=0}
CFG_shard_pool_mismatch=${CFG_shard_pool_mismatch:-SPILL_GIB=0 KV_SHARD=1 POOL=$SHARD_POOL RANK1_EXTRA_ENV=TF_DS_WINDOW=$(( SHARD_WINDOW - 4096 ))}
CFG_shard_kill=${CFG_shard_kill:-$CFG_shard_pressure}
CFG_shard_perf=${CFG_shard_perf:-SPILL_GIB=0 KV_SHARD=1 POOL=$SHARD_POOL KV_SHARD_TWINS=1}
CFG_shard_perf_fetch=${CFG_shard_perf_fetch:-SPILL_GIB=0 KV_SHARD=1 POOL=$SHARD_POOL KV_SHARD_TWINS=0}
CFG_spill_shard=${CFG_spill_shard:-$CFG_spill KV_SHARD=1 POOL=$SHARD_POOL}
CFG_spill_shard_pressure=${CFG_spill_shard_pressure:-$CFG_shard_pressure $CFG_spill KV_SHARD=1}  # shard settings first: they set SPILL_GIB=0, which the spill settings then override
CFG_spill_pictures=${CFG_spill_pictures:-$CFG_spill}
CFG_spill_pictures_mismatch=${CFG_spill_pictures_mismatch:-$CFG_spill RANK1_EXTRA_ENV=TF_DS_IMAGE_CACHE=0}
CFG_pair_kill=${CFG_pair_kill:-$CFG_off}
CFG_pair_stop=${CFG_pair_stop:-$CFG_spill_shard}

# ---------------------------------------------------------------- helpers
log()  { printf '[gpu_validate %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[gpu_validate %s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[gpu_validate %s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
# run CMD...: execute, or print it under --dry-run
run() { if (( DRY )); then printf '[dry-run] %s\n' "$(printf '%q ' "$@")"; else "$@"; fi; }
# wrk CMD-STRING: on the worker
wrk() { if (( DRY )); then printf '[dry-run] %s %q\n' "${SSH[*]}" "$1"; else "${SSH[@]}" "$1"; fi; }
cfg_of() { local v="CFG_${1//-/_}"; echo "${!v}"; }
SUMMARY=()
# suite NAME CMD...: run one suite, its output in $RECEIPTS/NAME.log, its exit code in the summary (never aborts)
suite() {
  local name=$1; shift
  log "suite $name"
  if (( DRY )); then
    if [[ "$1" == T ]]; then run env $(tools_env) "${PY[@]}" "$TOOLS/$2" "${@:3}"; else run "$@"; fi
    SUMMARY+=("$name dry-run"); return 0
  fi
  mkdir -p "$RECEIPTS"
  "$@" >"$RECEIPTS/$name.log" 2>&1
  local rc=$?
  SUMMARY+=("$name $rc")
  if (( rc == 0 )); then log "  $name: PASS"; else warn "  $name: exit $rc (see $RECEIPTS/$name.log)"; tail -n 5 "$RECEIPTS/$name.log" >&2; fi
  return 0
}
health_ok() { curl -sf -m 10 "$API/health" 2>/dev/null | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("status") == "ok" else 1)' 2>/dev/null; }
wait_health() {
  local limit=${1:-2400} t=0
  (( DRY )) && { log "[dry-run] wait for $API/health"; return 0; }
  until health_ok; do
    (( t >= limit )) && return 1
    sleep 10; t=$((t + 10))
  done
}
snapshot() {   # snapshot DIR: health and metrics now
  (( DRY )) && return 0
  mkdir -p "$1"
  curl -s -m 15 "$API/health" >"$1/health-$(date +%H%M%S).json" 2>/dev/null || true
  curl -s -m 15 "$API/metrics" >"$1/metrics-$(date +%H%M%S).txt" 2>/dev/null || true
}
containers_left() {
  local n=0
  if (( DRY )); then return 1; fi
  docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER" && n=1
  "${SSH[@]}" "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER'" && n=1
  (( n == 1 ))
}
kill_ranks() {   # kill -9 both ranks' engine processes by PID, at once (--resolve FILE: only write them; --pids FILE: kill those)
  local p0 p1
  if (( DRY )); then
    [[ -n "$KILL_RANK" ]] || run sudo -n kill -9 "<docker inspect -f {{.State.Pid}} $CONTAINER>"
    wrk "sudo -n kill -$KILL_SIG \$(docker inspect -f '{{.State.Pid}}' $CONTAINER)"
    return 0
  fi
  if [[ "$KILL_RANK" == 1 ]]; then      # rank 1 alone: rank 0 must notice (PairBroken / TransferTimeout), not hang
    p1=$("${SSH[@]}" "docker inspect -f '{{.State.Pid}}' '$CONTAINER'" 2>/dev/null)
    [[ "$p1" =~ ^[1-9][0-9]*$ ]] || { echo "no running engine PID on rank 1: '$p1'" >&2; return 1; }
    "${SSH[@]}" "sudo -n kill -$KILL_SIG $p1"; local r1=$?
    if [[ "$KILL_SIG" == STOP ]]; then echo "stopped rank 1 PID $p1 (exit $r1)"; else echo "killed rank 1 PID $p1 (exit $r1)"; fi
    return $r1
  fi
  if [[ -n "$PIDS_FILE" ]]; then
    read -r p0 p1 <"$PIDS_FILE" || { echo "cannot read the PIDs from $PIDS_FILE" >&2; return 1; }
  else
    p0=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER" 2>/dev/null)
    p1=$("${SSH[@]}" "docker inspect -f '{{.State.Pid}}' '$CONTAINER'" 2>/dev/null)
  fi
  [[ "$p0" =~ ^[1-9][0-9]*$ && "$p1" =~ ^[1-9][0-9]*$ ]] || { echo "no running engine PID (rank 0: '$p0', rank 1: '$p1')" >&2; return 1; }
  if [[ -n "$RESOLVE_FILE" ]]; then
    echo "$p0 $p1" >"$RESOLVE_FILE" && echo "rank 0 PID $p0, rank 1 PID $p1 -> $RESOLVE_FILE"
    return
  fi
  "${SSH[@]}" "sudo -n kill -9 $p1" &
  sudo -n kill -9 "$p0"; local r0=$?
  wait $!; local r1=$?
  echo "killed rank 0 PID $p0 (exit $r0), rank 1 PID $p1 (exit $r1)"
  (( r0 == 0 && r1 == 0 ))
}
port_env() {   # the environment start.sh / stop.sh / tools get for the port checkout
  local key=""
  [[ -f "$PORT_STATE_DIR/api-key" ]] && key="API_KEY_FILE=$PORT_STATE_DIR/api-key"
  # its own LOG_DIR here, and LOG_KEEP high enough that its stops never rotate out the daily's logs on the worker
  # (stop.sh saves rank 1's log in the worker's fixed ~/.cache/dsv41-tensorfold/logs, keeping the newest LOG_KEEP)
  echo "STATE_DIR=$PORT_STATE_DIR LOG_DIR=$PORT_LOG_DIR LOG_KEEP=1000 $key"
}
# port_running: a container named CONTAINER that the PORT's start.sh made (it labels tf.recipe.port, "off" included;
# the daily recipe's containers carry no such label), here or on the worker
port_running() {
  (( DRY )) && return 1
  local l0 l1
  l0=$(docker inspect -f '{{index .Config.Labels "tf.recipe.port"}}' "$CONTAINER" 2>/dev/null || true)
  l1=$("${SSH[@]}" "docker inspect -f '{{index .Config.Labels \"tf.recipe.port\"}}' '$CONTAINER'" 2>/dev/null || true)
  [[ ( -n "$l0" && "$l0" != '<no value>' ) || ( -n "$l1" && "$l1" != '<no value>' ) ]]
}
start_port() {   # start_port [--restart] CFG-NAME [extra VAR=value...]: ./start.sh (--restart: ./start.sh restart)
  local mode=""
  [[ "$1" != --restart ]] || { mode=restart; shift; }
  local name=$1; shift
  local words; words="$(cfg_of "$name") $*"
  log "start the port server: $name ($words)${mode:+, $mode}"
  PORT_STARTED=1
  if (( DRY )); then run env $(port_env) $words "$PORT_DIR/start.sh" $mode; return 0; fi
  mkdir -p "$RECEIPTS/$name"
  (cd "$PORT_DIR" && env $(port_env) $words ./start.sh $mode) >>"$RECEIPTS/$name/start.log" 2>&1 || { warn "start.sh failed ($name)"; return 1; }
  wait_health 2400 || { warn "the port server did not become healthy ($name)"; return 1; }
}
stop_port() {
  log "stop the port server"
  if (( DRY )); then run env $(port_env) "$PORT_DIR/stop.sh"; return 0; fi
  [[ -x "$PORT_DIR/stop.sh" ]] || { warn "no $PORT_DIR/stop.sh"; return 0; }
  mkdir -p "$RECEIPTS"
  (cd "$PORT_DIR" && env $(port_env) ./stop.sh) >>"$RECEIPTS/stop.log" 2>&1 || warn "stop.sh failed"
  containers_left && warn "a $CONTAINER container is still present after stop.sh"
  return 0
}
drop_caches() {
  log "drop page caches on both nodes"
  run sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  wrk "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
}
fresh_spill() {   # an empty SPILL_DIR on both nodes (the engine's store directories are root-owned)
  case "$SPILL_DIR" in "$SESS"/*) ;; *) die "SPILL_DIR ($SPILL_DIR) must be under $SESS" ;; esac
  log "fresh spill directory $SPILL_DIR on both nodes"
  run sudo -n rm -rf -- "$SPILL_DIR"
  run mkdir -p "$SPILL_DIR"
  wrk "sudo -n rm -rf -- '$SPILL_DIR' && mkdir -p '$SPILL_DIR'"
}
BALLAST=${BALLAST:-$SPILL_DIR.ballast}   # beside SPILL_DIR (same file system, not inside the store)
free_of() {   # free_of 0|1: free bytes on SPILL_DIR's file system on that rank's node
  if [[ "$1" == 0 ]]; then df -B1 --output=avail "$SPILL_DIR" 2>/dev/null | tail -1 | tr -dc 0-9
  else "${SSH[@]}" "df -B1 --output=avail '$SPILL_DIR'" 2>/dev/null | tail -1 | tr -dc 0-9; fi
}
remove_ballast() {
  [[ -n "${BALLAST_ON:-}" ]] || return 0
  log "remove the floor ballast ($BALLAST)"
  if [[ "$BALLAST_ON" == 0 ]]; then run rm -f -- "$BALLAST"; else wrk "rm -f -- '$BALLAST'"; fi
  BALLAST_ON=""
}
# floor_setup NEED: on the empty store, make both nodes' free space equal (a ballast file on the roomier one, so one
# floor binds on both ranks), then FLOOR_GIB = free - MARGIN, with MARGIN = FLOOR_MARGIN_BYTES or NEED / 2. The store
# refuses F1's write when floor + need > free, i.e. when NEED > MARGIN (start.sh refuses a floor above free disk).
floor_setup() {
  local need=$1 f0 f1 diff m
  m=${FLOOR_MARGIN_BYTES:-$(( need / 2 ))}
  FLOOR_MARGIN=$m
  if (( DRY )); then FLOOR_GIB="<min-free-after-ballast-less-$m-B>"; log "[dry-run] floor: F1 needs ~$need B, margin $m B; ballast $BALLAST on the roomier node"; return 0; fi
  (( need > m )) || { warn "F1's write (~$need B) is not above the margin ($m B): the floor would admit it"; return 1; }
  f0=$(free_of 0); f1=$(free_of 1)
  [[ -n "$f0" && -n "$f1" ]] || { warn "could not read the free space of $SPILL_DIR"; return 1; }
  diff=$(( f0 > f1 ? f0 - f1 : f1 - f0 ))
  if (( diff > m / 4 )); then
    if (( f0 > f1 )); then BALLAST_ON=0; fallocate -l "$diff" -- "$BALLAST" || { warn "fallocate on rank 0 failed"; BALLAST_ON=""; return 1; }
    else BALLAST_ON=1; "${SSH[@]}" "fallocate -l $diff -- '$BALLAST'" || { warn "fallocate on the worker failed"; BALLAST_ON=""; return 1; }; fi
    sync; f0=$(free_of 0); f1=$(free_of 1)
  fi
  local lo=$(( f0 < f1 ? f0 : f1 ))
  printf 'need %s B, margin %s B, ballast %s B on rank %s, free after: rank 0 %s B, rank 1 %s B\n' "$need" "$m" "$diff" "${BALLAST_ON:-none}" "$f0" "$f1" >"$RECEIPTS/spill/floor-free.txt"
  FLOOR_GIB=$(awk -v f="$lo" -v m="$m" 'BEGIN { g = (f - m) / 1073741824; if (g <= 0) exit 1; printf "%.6f", g }') || return 1
}
# the daily stack's files a port prepare.sh could replace under it: both weight files, its served folder (links and
# the chat template). Compared before and after the build.
daily_files() {
  local dd wc
  read -r dd wc < <(cd "$DAILY_DIR" && bash -c 'source scripts/config.sh >/dev/null 2>&1; echo "$DATA_DIR $WORKER_CACHE_DIR"')
  [[ -n "$dd" ]] || { echo "no DATA_DIR from $DAILY_DIR/scripts/config.sh"; return 1; }
  echo "rank 0 weight files ($dd/rank-cache):"
  find "$dd/rank-cache" -maxdepth 1 -name 'rank0of2-*.bin' -printf '%f %s %T@\n' 2>/dev/null | sort
  echo "served folder ($dd/model):"
  find "$dd/model" -maxdepth 1 -printf '%f %l\n' 2>/dev/null | sort
  sha256sum "$dd/model/chat_template.jinja" 2>/dev/null | cut -d' ' -f1
  echo "rank 1 weight files ($wc):"
  "${SSH[@]}" "find $wc -maxdepth 1 -name 'rank1of2-*.bin' -printf '%f %s %T@\n' 2>/dev/null | sort"
}
tools_env() { if (( DRY )); then echo "API_URL=$API"; else echo "API_URL=$API $(port_env)"; fi; }
# a tool, with the port's API key file when there is one: the checklib tools take it only as --api-key-file (never a
# default path); exact.py, bench.py and the other upstream tools read API_KEY_FILE from tools_env
T() {
  local k=()
  case "$1" in spill_check.py|kv_shard_check_e2e.py|resume_bench.py|pair_check.py|prefix_fanout.py)
    (( DRY )) || [[ ! -f "$PORT_STATE_DIR/api-key" ]] || k=(--api-key-file "$PORT_STATE_DIR/api-key") ;;
  esac
  env $(tools_env) "${PY[@]}" "$TOOLS/$1" ${k[@]+"${k[@]}"} "${@:2}"
}

RESTORED=0; HELD=0; DAILY_STOPPED=0; PORT_STARTED=0; UNKNOWN_STATE=0
service_active() { systemctl is-active --quiet "$SERVICE"; }
restore() {
  (( RESTORED )) && return 0
  RESTORED=1
  (( DRY )) || mkdir -p "$RECEIPTS"
  remove_ballast
  log "restore: the port server off, $SERVICE back, the hold removed"
  if (( PORT_STARTED )) || port_running; then stop_port; fi
  if port_running; then
    warn "a port container ($CONTAINER, tf.recipe.port label) still runs after stop.sh: not starting $SERVICE over it; the hold stays ($HOLD)"
    return 1
  fi
  if (( ! DAILY_STOPPED && ! PORT_STARTED && ! UNKNOWN_STATE )); then
    log "the daily service was not stopped by this run: only the hold goes"
    run rm -f "$HOLD"; HELD=0
    return 0
  fi
  if (( DRY )); then
    log "[dry-run] if $SERVICE is active and /health is ok: nothing; if active (oneshot, RemainAfterExit) but down: restart; else: start"
    run sudo -n systemctl start "$SERVICE"
  elif service_active && health_ok && ! port_running; then
    log "$SERVICE is active and healthy"
  elif service_active; then
    log "$SERVICE is active (a oneshot unit stays active after its containers went) but not serving: restart it"
    sudo -n systemctl restart "$SERVICE"
  else
    sudo -n systemctl start "$SERVICE"
  fi
  if wait_health 3600; then
    log "daily service healthy"
    if (( ! DRY )); then
      env API_URL="$API" "${PY[@]}" "$TOOLS/smoke.py" --url "$API" --json "$RECEIPTS/restore-smoke.json" >"$RECEIPTS/restore-smoke.log" 2>&1 \
        && log "  restore smoke: PASS" || warn "  restore smoke failed (see $RECEIPTS/restore-smoke.log)"
      if [[ "$VERIFY_RESTORE" == 1 && -f "$RECEIPTS/base-exact.json" ]]; then
        suite restore-exact-record "${PY[@]}" "$TOOLS/exact.py" record --url "$API" --label daily-after --like "$RECEIPTS/base-exact.json" --out "$RECEIPTS/restore-exact.json"
        suite restore-exact-compare "${PY[@]}" "$TOOLS/exact.py" compare "$RECEIPTS/base-exact.json" "$RECEIPTS/restore-exact.json"
      fi
    fi
  else
    warn "the daily service did not become healthy: the hold stays ($HOLD); check journalctl -u $SERVICE"
    return 1
  fi
  run rm -f "$HOLD"
  HELD=0
}
on_exit() {
  local rc=$?
  if (( HELD && ! NO_RESTORE && ! RESTORED )); then warn "exiting early (status $rc): restoring the daily service"; restore; fi
  remove_ballast
  if (( ${#SUMMARY[@]} )); then
    log "summary:"; printf '  %s\n' ${SUMMARY[@]+"${SUMMARY[@]}"}
    if (( ! DRY )); then
      mkdir -p "$RECEIPTS"
      printf '%s\n' ${SUMMARY[@]+"${SUMMARY[@]}"} | python3 -c 'import json,sys; rows=[l.split() for l in sys.stdin if l.strip()]; json.dump({"suites": {n: (int(c) if c.isdigit() else c) for n, c in rows}, "failed": [n for n, c in rows if c not in ("0", "dry-run")]}, open(sys.argv[1], "w"), indent=1)' "$RECEIPTS/summary.json"
      log "receipts: $RECEIPTS"
    fi
  fi
}
trap on_exit EXIT
trap 'exit 130' INT TERM

case "$CMD" in
  kill-ranks) kill_ranks; exit $? ;;
  restore) HELD=1; UNKNOWN_STATE=1; restore; exit $? ;;
esac
has() { [[ " $PHASES " == *" $1 "* ]]; }
log "phases: $PHASES; configurations: $CONFIGS; receipts: $RECEIPTS$( (( DRY )) && echo ' (dry run: nothing is executed)')"

# ---------------------------------------------------------------- preflight
if has preflight; then
  [[ "$(hostname)" == *gx10-a* || "${GPU_VALIDATE_HOST_OK:-0}" == 1 ]] || die "run this on gx10-a (rank 0), not $(hostname) (GPU_VALIDATE_HOST_OK=1 overrides)"
  # (a dry run executes nothing: tools/selftest.py runs it from a temporary folder, /tmp on Linux)
  (( DRY )) || case "$SESS" in /tmp/*|/var/tmp/*) die "SESS must not be under /tmp on the Sparks: $SESS" ;; esac
  case "$SPILL_DIR" in "$SESS"/*) ;; *) die "SPILL_DIR must be under $SESS (it is wiped before the spill suites): $SPILL_DIR" ;; esac
  if (( ! DRY )); then
    for c in docker python3 curl git ssh sudo; do command -v "$c" >/dev/null || die "$c is not installed"; done
    sudo -n true 2>/dev/null || die "sudo needs a password here: the runbook needs sudo -n (systemctl, drop_caches, kill)"
    "${SSH[@]}" true || die "cannot ssh to the worker $WORKER"
    "${SSH[@]}" "sudo -n true" || die "sudo -n fails on the worker $WORKER"
    [[ -d "$DAILY_DIR" ]] || die "the daily checkout $DAILY_DIR is missing"
  fi
  run mkdir -p "$SESS/dsv41-port" "$RECEIPTS" "$PORT_STATE_DIR"
  log "preflight ok: worker $WORKER, daily $DAILY_DIR, port $PORT_DIR @ $PORT_REF, spill $SPILL_DIR"
fi

# ---------------------------------------------------------------- checkout
if has checkout; then
  if [[ -d "$PORT_DIR/.git" ]]; then
    run git -C "$PORT_DIR" fetch --quiet "$PORT_REPO" "$PORT_REF"
    run git -C "$PORT_DIR" checkout --quiet --detach FETCH_HEAD
  else
    run git clone --quiet "$PORT_REPO" "$PORT_DIR"
    run git -C "$PORT_DIR" fetch --quiet "$PORT_REPO" "$PORT_REF"
    run git -C "$PORT_DIR" checkout --quiet --detach FETCH_HEAD
  fi
  if [[ -f "$DAILY_DIR/scripts/local.sh" ]]; then run cp "$DAILY_DIR/scripts/local.sh" "$PORT_DIR/scripts/local.sh"; fi
  (( DRY )) || git -C "$PORT_DIR" log --oneline -1 >"$RECEIPTS/port-commit.txt"
  # every recipe setting a configuration names must be one the port's start.sh knows
  if (( ! DRY )); then
    for c in $CONFIGS; do
      for w in $(cfg_of "$c"); do
        v=${w%%=*}
        [[ "$v" == TF_DS_* || "$v" == TENSORFOLD_* ]] && continue
        grep -qw -- "$v" "$PORT_DIR/scripts/config.sh" "$PORT_DIR/scripts/port.sh" "$PORT_DIR/start.sh" 2>/dev/null || die "configuration $c sets $v, which the port's start.sh, scripts/config.sh and scripts/port.sh do not know: set CFG_${c//-/_} to its names"
      done
    done
  fi
  # the tools: PORT_DIR's own (spill-kvshard carries them since ec937bb), or TOOLS_REF's in their own checkout
  if [[ -z "${TOOLS_GIVEN:-}" && -z "$TOOLS_REF" ]]; then
    if (( DRY )) || [[ -f "$PORT_DIR/tools/spill_check.py" ]]; then TOOLS=$PORT_DIR/tools
    else die "$PORT_REF has no tools/spill_check.py ($PORT_DIR): set TOOLS_REF to a ref that has"; fi
  elif [[ -z "${TOOLS_GIVEN:-}" ]]; then
    if [[ -d "$TOOLS_DIR/.git" ]]; then run git -C "$TOOLS_DIR" fetch --quiet "$PORT_REPO" "$TOOLS_REF"
    else run git clone --quiet --no-checkout "$PORT_REPO" "$TOOLS_DIR"; run git -C "$TOOLS_DIR" fetch --quiet "$PORT_REPO" "$TOOLS_REF"; fi
    run git -C "$TOOLS_DIR" checkout --quiet --detach FETCH_HEAD
    (( DRY )) || git -C "$TOOLS_DIR" log --oneline -1 >"$RECEIPTS/tools-commit.txt"
    if (( DRY )) || [[ -f "$TOOLS_DIR/tools/spill_check.py" ]]; then TOOLS=$TOOLS_DIR/tools
    else die "$TOOLS_REF has no tools/spill_check.py ($TOOLS_DIR)"; fi
  fi
  log "tools: $TOOLS"
fi

# ---------------------------------------------------------------- hold
if has hold; then
  run touch "$HOLD"; HELD=1
  log "hold set ($HOLD): the watchdog leaves the stack alone until restore"
fi

# ---------------------------------------------------------------- baseline (the daily service, unchanged)
if has baseline; then
  if (( ! DRY )); then health_ok || die "the daily service is not healthy at $API: no baseline can be recorded"; fi
  snapshot "$RECEIPTS/baseline"
  suite base-exact env API_URL="$API" "${PY[@]}" "$TOOLS/exact.py" record --url "$API" --label daily --out "$RECEIPTS/base-exact.json" $EXACT_ARGS
  suite base-spill env API_URL="$API" "${PY[@]}" "$TOOLS/spill_check.py" record --url "$API" --label daily --sizes "$SPILL_SIZES" --out "$RECEIPTS/base-spill.json"
  suite base-kv env API_URL="$API" "${PY[@]}" "$TOOLS/kv_shard_check_e2e.py" record --url "$API" --label daily --out "$RECEIPTS/base-kv.json"
  suite base-fanout env API_URL="$API" "${PY[@]}" "$TOOLS/prefix_fanout.py" --url "$API" --label daily --out "$RECEIPTS/base-fanout.json" --receipt "$RECEIPTS/r-base-fanout.json"
  if [[ "$BENCH" == 1 || " $CONFIGS " == *" shard-perf"* ]]; then
    suite base-bench env API_URL="$API" "${PY[@]}" "$TOOLS/bench.py" --url "$API" --label daily $BENCH_ARGS --out "$RECEIPTS/base-bench.json"
  fi
fi

# ---------------------------------------------------------------- stop
if has stop; then
  DAILY_STOPPED=1
  run sudo -n systemctl stop "$SERVICE"
  if containers_left; then
    warn "$CONTAINER still present after stopping $SERVICE: running the daily stop.sh"
    (cd "$DAILY_DIR" && ./stop.sh) || die "could not stop the daily containers"
  fi
  log "daily service stopped"
fi

# ---------------------------------------------------------------- build
if has build; then
  log "build the port image: scripts/prepare.sh in $PORT_DIR (STATE_DIR $PORT_STATE_DIR)"
  # prepare.sh shares HF_CACHE/DATA_DIR and WORKER_CACHE_DIR with the daily stack, and replaces a weight file whose
  # index hash differs: the daily's files are compared before and after
  if (( DRY )); then run env STATE_DIR="$PORT_STATE_DIR" "$PORT_DIR/scripts/prepare.sh"; log "[dry-run] compare the daily's weight files and served folder before and after"
  else
    daily_files >"$RECEIPTS/daily-files-before.txt" 2>&1 || die "could not list the daily's weight files (see $RECEIPTS/daily-files-before.txt)"
    (cd "$PORT_DIR" && env STATE_DIR="$PORT_STATE_DIR" ./scripts/prepare.sh) >"$RECEIPTS/prepare.log" 2>&1 || die "prepare.sh failed (see $RECEIPTS/prepare.log)"
    daily_files >"$RECEIPTS/daily-files-after.txt" 2>&1
    cmp -s "$RECEIPTS/daily-files-before.txt" "$RECEIPTS/daily-files-after.txt" ||
      die "the port's prepare.sh changed the DAILY stack's weight files or served folder (diff $RECEIPTS/daily-files-before.txt $RECEIPTS/daily-files-after.txt): the daily service may need its own ./scripts/prepare.sh before it serves correctly"
    log "the daily's weight files and served folder are unchanged"
  fi
fi

if has dropcache; then drop_caches; fi

# ---------------------------------------------------------------- suites
exact_suite() {   # exact_suite CFG: the same prompts as the daily record, compared by strong token hash
  suite "$1-exact-record" T exact.py record --url "$API" --label "$1" --like "$RECEIPTS/base-exact.json" --out "$RECEIPTS/$1-exact.json" $EXACT_ARGS
  suite "$1-exact-compare" T exact.py compare "$RECEIPTS/base-exact.json" "$RECEIPTS/$1-exact.json"
}
spill_staging() { local v; v=$(cfg_of spill); [[ " $v " =~ \ SPILL_STAGING_MIB=([0-9]+)\  ]] && echo "${BASH_REMATCH[1]}" || echo 128; }
SPILL_STATE=$RECEIPTS/spill/state.json        # what the spill phases pass on (files, the drop, the kill)
SPILL_ARGS=(--baseline "$RECEIPTS/base-spill.json" --state "$SPILL_STATE" --rank0-dir "$SPILL_DIR" --rank1-dir "$SPILL_DIR"
            --rank1-shell "${SSH[*]}" --sudo --spill-min-tokens 8192)
KILL_PIDS=$RECEIPTS/spill/pids
startup_line() {   # startup_line CFG: rank 0's `dsv41: pool ... a conversation may use up to ...` line, for kv limits
  if (( DRY )); then run docker logs "$CONTAINER" "| grep 'dsv41: pool' > $RECEIPTS/$1/startup-pool.txt"; return 0; fi
  docker logs "$CONTAINER" 2>&1 | grep 'dsv41: pool' | tail -1 >"$RECEIPTS/$1/startup-pool.txt" || true
  [[ -s "$RECEIPTS/$1/startup-pool.txt" ]] || warn "no 'dsv41: pool' line in rank 0's log ($1): kv limits will SKIP the pool"
}
save_logs() {   # save_logs CFG T0: the ranks' logs stop.sh saved since T0 (the port's own, not the daily's)
  (( DRY )) && return 0
  find "$PORT_LOG_DIR" -maxdepth 1 -name '*rank0*' -newermt "@$2" -exec cp -f {} "$RECEIPTS/$1/" \; 2>/dev/null || true
  local f
  for f in $("${SSH[@]}" "find ~/.cache/dsv41-tensorfold/logs -maxdepth 1 -name '*rank1*' -newermt '@$2'" 2>/dev/null); do
    scp -q -o BatchMode=yes "$WORKER:$f" "$RECEIPTS/$1/" 2>/dev/null || warn "could not copy rank 1's $f"
  done
}
# startup_lines CFG: both ranks' start-up lines the receipts keep (the pool line, the spill agreement and staging, the
# CUDA budget's parked pages, refusals), from the containers' logs
startup_lines() {
  (( DRY )) && { run docker logs "$CONTAINER" "| grep -E 'dsv41|CUDA budget' > $RECEIPTS/$1/startup-rank0.txt"; return 0; }
  docker logs "$CONTAINER" 2>&1 | grep -E 'dsv41|CUDA budget|spill|kv_shard|error' | head -200 >"$RECEIPTS/$1/startup-rank0.txt" || true
  "${SSH[@]}" "docker logs '$CONTAINER' 2>&1 | grep -E 'dsv41|CUDA budget|spill|kv_shard|error' | head -200" >"$RECEIPTS/$1/startup-rank1.txt" 2>/dev/null || true
}
# log_has CFG RANK PATTERN...: every pattern (grep -E) in that rank's saved start-up lines (a suite: exit 1 when one is missing)
log_has() {
  local cfg=$1 r=$2 p f; shift 2
  f="$RECEIPTS/$cfg/startup-rank$r.txt"
  for p in "$@"; do grep -Eq -- "$p" "$f" || { echo "rank $r's start-up lines ($f) have no /$p/"; return 1; }; echo "rank $r: /$p/ found"; done
}
# expect_refused CFG PATTERN: the configuration's start must FAIL (the ranks refuse to agree at start, never at a
# request), and a RANK's log must have PATTERN (grep -E): rank 0's or rank 1's docker log, else rank 0's lines streamed
# into start.log (those without start.sh's own "[start.sh]" prefix). A start that failed for another reason (no
# identity, an OOM, a wrong image) is not a refusal.
expect_refused() {
  local name=$1 pat=$2 rc
  log "start the port server, expecting a refusal: $name ($(cfg_of "$name"))"
  PORT_STARTED=1
  if (( DRY )); then run env $(port_env) WAIT_TIMEOUT=2400 $(cfg_of "$name") "$PORT_DIR/start.sh"; SUMMARY+=("$name-refused dry-run"); return 0; fi
  mkdir -p "$RECEIPTS/$name"
  (cd "$PORT_DIR" && env $(port_env) WAIT_TIMEOUT=2400 $(cfg_of "$name") ./start.sh) >>"$RECEIPTS/$name/start.log" 2>&1; rc=$?
  docker logs "$CONTAINER" >"$RECEIPTS/$name/rank0.log" 2>&1 || true
  "${SSH[@]}" "docker logs '$CONTAINER'" >"$RECEIPTS/$name/rank1.log" 2>&1 || true
  if (( rc == 0 )); then
    SUMMARY+=("$name-refused 1"); warn "  $name: the pair STARTED although rank 1 runs other settings"
  elif grep -Eq -- "$pat" "$RECEIPTS/$name/rank0.log" "$RECEIPTS/$name/rank1.log" ||
       grep -v '^\[start\.sh\]' "$RECEIPTS/$name/start.log" | grep -Eq -- "$pat"; then
    SUMMARY+=("$name-refused 0"); log "  $name: refused at start, a log names /$pat/: PASS"
  else
    SUMMARY+=("$name-refused 1"); warn "  $name: refused, but no log names /$pat/ (see $RECEIPTS/$name/)"
  fi
}
# the engine's own refusal texts (port/integration 98370ef): rank_wire.describeBroken for a fingerprint field, and
# native.explain / the error names for the KV-shard agreement (model.zig KvShardMismatch, kv_lanes ShardLayoutMismatch)
REFUSED_FEATURES="configuration differs from rank 0.s in features"
REFUSED_SHARD="KvShardMismatch|ShardLayoutMismatch|the ranks disagree on TF_DS_KV_SHARD|the ranks cut the pool into different parts"
KV=(kv_shard_check_e2e.py run --url "$API" --baseline "$RECEIPTS/base-kv.json")
kv_suite() {   # kv_suite CFG NAME CHECKS [args...]
  local cfg=$1 name=$2 checks=$3; shift 3
  suite "$cfg-kv-$name" T "${KV[@]}" --receipt "$RECEIPTS/$cfg-kv-$name.json" --checks "$checks" --needle-sizes "" "$@"
}
# spill_restart CFG [spill_check args...]: prerestart, stop, drop rank 1's manifest of MM, start, postrestart (S3 / SS5)
spill_restart() {
  local cfg=$1; shift
  local st=(--state "$RECEIPTS/$cfg/state.json")
  suite "$cfg-prerestart" T spill_check.py run --url "$API" --phase prerestart --receipt "$RECEIPTS/$cfg-prerestart.json" "${SPILL_ARGS[@]}" "${st[@]}" "$@"
  stop_port
  suite "$cfg-drop" T spill_check.py drop "${st[@]}" --rank1-dir "$SPILL_DIR" --rank1-shell "${SSH[*]}" --sudo
  start_port "$cfg" || SUMMARY+=("$cfg-restart 1")
  suite "$cfg-postrestart" T spill_check.py run --url "$API" --phase postrestart --receipt "$RECEIPTS/$cfg-postrestart.json" "${SPILL_ARGS[@]}" "${st[@]}" "$@"
}
# store_sizes CFG: each rank's store on disk (du; SS3 reads 890 B a token for a rank's own part, 170 for the other's)
store_sizes() {
  local f="$RECEIPTS/$1/stores.txt"
  if (( DRY )); then run sudo -n du -sb "$SPILL_DIR/native-spill-*/rank*"; wrk "sudo -n du -sb $SPILL_DIR/native-spill-*/rank*"; return 0; fi
  { echo "rank 0:"; sudo -n sh -c "du -sb $SPILL_DIR/native-spill-*/rank* $SPILL_DIR/native-spill-*/rank*/*"
    echo "rank 1:"; "${SSH[@]}" "sudo -n sh -c 'du -sb $SPILL_DIR/native-spill-*/rank* $SPILL_DIR/native-spill-*/rank*/*'"; } >"$f" 2>&1
  log "  store sizes: $f"
}
# saved_log_has T0 PATTERN: the newest rank-0 log stop.sh saved into PORT_LOG_DIR since T0 has PATTERN (grep -E)
saved_log_has() {
  local f
  f=$(find "$PORT_LOG_DIR" -maxdepth 1 -name '*-rank0.log.gz' -newermt "@$1" 2>/dev/null | LC_ALL=C sort | tail -n 1)
  [[ -n "$f" ]] || { echo "no rank-0 log saved into $PORT_LOG_DIR since the break"; return 1; }
  if zcat "$f" | grep -Eq -- "$2"; then echo "$f: /$2/ found"; else echo "$f has no /$2/"; return 1; fi
}
# pair_suite CFG SIGNAL DEADLINE: streams decode, rank 1 is killed (9) or stopped (STOP); every stream must end, /health
# go down and rank 0's container end with 75, its log saying so (pair_check.py); then the suite restarts the pair with
# ./start.sh restart (its stop.sh saves the exited rank 0's log first), checks that saved log, and runs exact (R2, R3)
pair_suite() {
  local cfg=$1 sig=$2 dl=$3 t
  t=$(date +%s)
  suite "$cfg-break" T pair_check.py --url "$API" --label "$cfg" --streams "$PAIR_STREAMS" \
    --kill-cmd "WORKER=$WORKER CONTAINER=$CONTAINER bash $SELF kill-ranks --rank 1 --signal $sig" --end-deadline "$dl" \
    --exit-cmd "docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' $CONTAINER" --exit-code 75 \
    --expect-log "docker logs $CONTAINER::exiting with status 75" \
    --receipt "$RECEIPTS/$cfg-break.json"
  start_port --restart "$cfg" || SUMMARY+=("$cfg-restart 1")
  suite "$cfg-saved-log" saved_log_has "$t" 'exiting with status 75'
  startup_lines "$cfg"
  exact_suite "$cfg-after"
}
if has suites; then
  for c in $CONFIGS; do
    log "=== configuration $c"
    t0=$(date +%s)
    (( DRY )) || mkdir -p "$RECEIPTS/$c"
    [[ "$c" == spill* || "$c" == pair-stop ]] && fresh_spill
    drop_caches
    case "$c" in
      spill-mismatch) expect_refused "$c" "$REFUSED_FEATURES"; stop_port; save_logs "$c" "$t0"; continue ;;
      shard-mismatch) expect_refused "$c" "$REFUSED_FEATURES|$REFUSED_SHARD"; stop_port; save_logs "$c" "$t0"; continue ;;
      shard-pool-mismatch) expect_refused "$c" "$REFUSED_SHARD"; stop_port; save_logs "$c" "$t0"; continue ;;
      spill-pictures-mismatch) expect_refused "$c" "$REFUSED_FEATURES"; stop_port; save_logs "$c" "$t0"; continue ;;
    esac
    if ! start_port "$c"; then SUMMARY+=("$c-start 1"); stop_port; save_logs "$c" "$t0"; continue; fi
    snapshot "$RECEIPTS/$c"
    startup_lines "$c"
    case "$c" in
      off)
        suite off-startup log_has off 0 'a broken rank pair ends this server with status 75'
        exact_suite off
        suite off-smoke T smoke.py --url "$API" --json "$RECEIPTS/off-smoke.json"
        suite off-fanout T prefix_fanout.py --url "$API" --baseline "$RECEIPTS/base-fanout.json" --receipt "$RECEIPTS/off-fanout.json" --strict-cold $PREFIX_FLAGS
        suite off-toolcheck T toolcheck.py
        [[ "$BENCH" != 1 ]] || suite off-bench T bench.py --url "$API" --label off $BENCH_ARGS --out "$RECEIPTS/off-bench.json" ;;
      spill)
        suite spill-startup log_has spill 0 'the ranks agreed' "staging $(spill_staging) MiB"
        exact_suite spill
        suite spill-main T spill_check.py run --url "$API" --phase main --receipt "$RECEIPTS/spill-main.json" "${SPILL_ARGS[@]}"
        # clean restart; while stopped, rank 1 loses MM's manifest (missing-rank1 is judged in postrestart)
        suite spill-prerestart T spill_check.py run --url "$API" --phase prerestart --receipt "$RECEIPTS/spill-prerestart.json" "${SPILL_ARGS[@]}"
        stop_port
        suite spill-drop T spill_check.py drop --state "$SPILL_STATE" --rank1-dir "$SPILL_DIR" --rank1-shell "${SSH[*]}" --sudo
        start_port spill || SUMMARY+=("spill-restart 1")
        suite spill-postrestart T spill_check.py run --url "$API" --phase postrestart --receipt "$RECEIPTS/spill-postrestart.json" "${SPILL_ARGS[@]}"
        # forced kill: both PIDs resolved before K2, the kill fired while K2's write is pending
        suite spill-prekill T spill_check.py run --url "$API" --phase prekill \
          --resolve-cmd "WORKER=$WORKER CONTAINER=$CONTAINER bash $SELF kill-ranks --resolve $KILL_PIDS" \
          --kill-cmd "WORKER=$WORKER CONTAINER=$CONTAINER bash $SELF kill-ranks --pids $KILL_PIDS" \
          --receipt "$RECEIPTS/spill-prekill.json" "${SPILL_ARGS[@]}"
        stop_port; start_port spill || SUMMARY+=("spill-kill-restart 1")
        suite spill-postkill T spill_check.py run --url "$API" --phase postkill --receipt "$RECEIPTS/spill-postkill.json" "${SPILL_ARGS[@]}"
        stop_port
        # The floor refuses a write only when evicting every stored state could not make room, so the floor suite
        # starts on an empty store; both nodes' free space is made equal (a ballast beside SPILL_DIR on the roomier
        # one), and the floor is that free space less a margin below F1's estimated write (start.sh refuses a floor
        # above free disk): F1's write is refused on both ranks.
        fresh_spill
        if (( DRY )); then need=183488512
        else need=$(T spill_check.py estimate --baseline "$RECEIPTS/base-spill.json" --item F1 --bytes 2>/dev/null); fi
        if [[ "$need" =~ ^[0-9]+$ ]] && floor_setup "$need"; then
          log "floor suite: SPILL_FLOOR_GIB=$FLOOR_GIB (F1 ~$need B, margin $FLOOR_MARGIN B)"
          if start_port spill "SPILL_FLOOR_GIB=$FLOOR_GIB"; then
            suite spill-floor T spill_check.py run --url "$API" --phase floor --floor-margin-bytes "$FLOOR_MARGIN" \
              --receipt "$RECEIPTS/spill-floor.json" "${SPILL_ARGS[@]}"
          else SUMMARY+=("spill-floor-start 1"); fi
        else SUMMARY+=("spill-floor-setup 1"); warn "floor suite not run (F1 estimate '$need')"; fi ;;
      shard)
        exact_suite shard
        startup_line shard
        kv_suite shard main limits,owners,mixed,needle,idle --startup-log "$RECEIPTS/shard/startup-pool.txt" \
          --expect-window "$SHARD_WINDOW" --expect-pool "$SHARD_POOL" --needle-sizes "$NEEDLES" ;;
      shard-p2|shard-p4)
        exact_suite "$c"
        kv_suite "$c" streams owners,mixed,idle ;;
      shard-nodraft)
        exact_suite shard-nodraft ;;
      shard-copies)
        exact_suite shard-copies
        kv_suite shard-copies burst burst,idle ;;
      shard-pressure)
        startup_line shard-pressure
        kv_suite shard-pressure main limits,carry,twins,move,cancel-transfer,idle \
          --startup-log "$RECEIPTS/shard-pressure/startup-pool.txt" --expect-pool "$PRESSURE_POOL" ;;
      shard-check)
        kv_suite shard-check carry carry,idle
        exact_suite shard-check ;;
      shard-kill)
        # rank 1 killed during a carry: rank 0 must end the request (PairBroken / TransferTimeout, xfer_ns 30 s), not hang
        kv_suite shard-kill kill carry,kill-rank1 --kill-deadline "$KILL_DEADLINE" \
          --kill-cmd "WORKER=$WORKER CONTAINER=$CONTAINER bash $SELF kill-ranks --rank 1"
        startup_lines shard-kill
        stop_port
        start_port shard-kill || SUMMARY+=("shard-kill-restart 1")
        exact_suite shard-kill-after
        kv_suite shard-kill after idle ;;
      spill-shard)
        suite spill-shard-startup log_has spill-shard 0 'the ranks agreed'
        exact_suite spill-shard
        startup_line spill-shard
        kv_suite spill-shard main limits,owners,mixed,idle --startup-log "$RECEIPTS/spill-shard/startup-pool.txt" \
          --expect-window "$SHARD_WINDOW" --expect-pool "$SHARD_POOL"
        suite spill-shard-main T spill_check.py run --url "$API" --phase main --receipt "$RECEIPTS/spill-shard-main.json" "${SPILL_ARGS[@]}"
        store_sizes spill-shard
        spill_restart spill-shard ;;
      spill-shard-pressure)
        kv_suite spill-shard-pressure main carry,twins,move,idle
        suite spill-shard-pressure-spill T spill_check.py run --url "$API" --phase main --checks evict,dual-restore,settled \
          --receipt "$RECEIPTS/spill-shard-pressure-spill.json" "${SPILL_ARGS[@]}"
        store_sizes spill-shard-pressure ;;
      spill-pictures)
        suite spill-pictures-startup log_has spill-pictures 0 'the ranks agreed'
        suite spill-pictures-main T spill_check.py run --url "$API" --phase main --checks pictures,settled \
          --receipt "$RECEIPTS/spill-pictures-main.json" "${SPILL_ARGS[@]}"
        spill_restart spill-pictures --pictures ;;
      pair-kill) pair_suite pair-kill 9 "$PAIR_KILL_DEADLINE" ;;
      pair-stop) pair_suite pair-stop STOP "$PAIR_STOP_DEADLINE" ;;
      shard-perf|shard-perf-fetch)
        suite "$c-bench" T bench.py --url "$API" --label "$c" $BENCH_ARGS --out "$RECEIPTS/$c-bench.json"
        [[ ! -f "$RECEIPTS/base-bench.json" ]] || suite "$c-bench-compare" T bench.py --compare "$RECEIPTS/base-bench.json" "$RECEIPTS/$c-bench.json" ;;
    esac
    snapshot "$RECEIPTS/$c"
    stop_port
    remove_ballast
    save_logs "$c" "$t0"
  done
fi

# ---------------------------------------------------------------- restore
if has restore && (( ! NO_RESTORE )); then restore || exit 1; fi
fails=0
for s in ${SUMMARY[@]+"${SUMMARY[@]}"}; do [[ "${s##* }" == 0 || "${s##* }" == dry-run ]] || fails=$((fails + 1)); done
log "done: $fails suite(s) failed"
exit $(( fails > 0 ))
