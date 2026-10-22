# The ported engine features' settings (wojo's GLM patches 0088/0110/0111 spill v2, 0118 KV sharding, 0119 parked
# pages, carried into the native dsv41 engine as zig/patches/0011-... to 0022-...): their defaults, their checks, and the ONE place
# they map to the engine's flags and variables (port_map). Sourced by scripts/config.sh after the serving settings;
# start.sh and stop.sh call the functions. Every feature is OFF by default (PORT_SPILL_DEFAULT_GIB, PORT_SHARD_DEFAULT
# below: SPILL_GIB=0, KV_SHARD=0, POOL empty): with them off nothing below adds a flag, a variable, a mount or a label
# value that differs from the recipe before the port, and both ranks run as before. Set them in scripts/local.sh to turn
# them on. A fork may turn a default on by changing those two values; such a default that cannot run here is turned off
# with a warning (port_validate, port_spill_check: spill when a Spark's spill folder fails its checks, KV sharding when
# the setup is not two Sparks or an explicit POOL does not fit it), while the same setting given explicitly refuses.
# The pool is zig/patches/0008's single POOL setting (config.sh; start.sh gives both ranks TF_DS_POOL, TF_DS_WINDOW and
# rank 1's --pool from it): KV_SHARD=1 cuts that same pool into one part a Spark, and with POOL empty port_map derives
# the pool sharding needs (2,101,248 at the 1,048,576 window).
# docs/PORT-RECIPE.md lists each setting, its engine name (checked against the engine branch port/series-pair-370daf0),
# and the deploy and rollback steps; docs/FORK-DEPLOY.md how the fork checkout on gx10-a is updated and rolled back.
# Kept to bash 3.2 syntax so that scripts/port_test.sh runs on a Mac too.

# ---------------------------------------------------------------- the settings
# Spill tier (GLM 0088 + 0110 spill v2, 0111 continuation; BadAd84's block-tier design): kept prompt states written to
# each Spark's own local disk when they are kept, in shared 2,048-token blocks, and read back instead of a new prefill,
# also after a restart or a crash.
#   SPILL_GIB        the cap a Spark (GiB, decimals allowed; 64 is what the pair was validated with). Default 0: off.
#                    Set explicitly (SPILL_GIB or TF_DS_SPILL_GIB), a Spark whose spill folder fails the checks below
#                    refuses the start; a fork default above 0 starts the pair without the tier there (a loud warning)
#   SPILL_DIR        rank 0's local directory, mounted at /spill (default: beside HF_CACHE, never inside it: HF_CACHE is
#                    the NFS export the worker reads). Must be on a local disk, owned by you, outside the NFS export,
#                    the model, Engram and weight-file folders and any archive folder; start.sh checks all of that
#   WORKER_SPILL_DIR rank 1's (default: SPILL_DIR; a leading literal $HOME/ is the worker's home)
#   SPILL_FLOOR_GIB  free disk the tier leaves on SPILL_DIR's file system: past it writes are refused (default 50)
#   SPILL_QUOTA_GIB  what one conversation may hold before its own old turns and forks go first (0: no quota)
#   SPILL_MIN_TOKENS shorter kept states are not written (default 8192). Both ranks get it (rank 0's --spill-min-tokens,
#                    rank 1's TF_DS_SPILL_MIN_TOKENS): it is in the ranks' fingerprint, so a difference fails the start
#   SPILL_STAGING_MIB the pinned staging a rank's tier works in, taken before the pool (MiB, at least 16; default 128)
#   SPILL_TURNS      old turns a conversation keeps on disk past its newest (the engine's TF_SPILL_TURNS, default 4)
#   SPILL_FLUSH_MS   what a clean stop gives the tier to write what still waits (TF_SPILL_FLUSH_MS, default 60000)
#   SPILL_WEIGHTS_SHA=1 each rank's weight file's SHA-256 as TF_DS_RANK_CACHE_SHA (computed once a file, cached beside
#                    it; ~1 minute a Spark the first time). 0: the checkpoint's revision alone identifies the weights.
#                    It names the store: switching it (either way) opens another, empty store. With 1, a start that
#                    cannot get a rank's digest refuses (DRY_RUN only warns)
#   MODEL_QUANT      the quantization's name in TF_DS_MODEL_REVISION (default: from MODEL_ID, e.g. exl3-2.9bpw)
# The spill tier's identity (TF_DS_ENGINE_ID, TF_DS_RECIPE_PATCHES, TF_DS_MODEL_REVISION, TF_DS_ENGRAM_REVISION and
# each rank's TF_DS_RANK_CACHE_SHA, which the engine requires with the tier and refuses to start without) is derived by
# port_ident_map from the image, the checkpoint and Engram's pins: nothing to set. An exported value of the first four
# wins (the same on both ranks); TF_DS_BUILD_FLAGS, when exported, reaches both ranks as it is.
# KV_SHARD=1:    each conversation's compressed latents stored once, on the Spark whose part of the pool holds it
#                (GLM 0118): the same POOL costs about half the KV memory a Spark (measured at 2,101,248 positions:
#                1,062 MiB a rank against 1,783 MiB replicated), for 1-4% decode. Needs a pool of two owner parts each
#                holding a whole CONTEXT window. Default 0 (off). Set explicitly (KV_SHARD or TF_DS_KV_SHARD), a setup it
#                cannot run on refuses; a fork default of 1 is turned off with a warning there instead
# KV_SHARD_DECODE: fetch (the reference: the owner sends the rows decode picks) or owner (owner-computed decode);
#                empty: the engine's default (fetch)
# KV_SHARD_CARRY: 1 (default): a kept conversation whose part is full is carried into the other part, and live streams
#                move to make room (TF_DS_KV_SHARD_CARRY); 0: every hit stays in its part
# KV_SHARD_TWINS: 1 (default): a prompt that runs from 0 borrows the other part's free rows (TF_DS_KV_SHARD_TWINS);
#                0: every prompt chunk fetches
# KV_SHARD_CHECK: decode rounds between the ranks' planner digest checks (TF_DS_KV_SHARD_CHECK, default 1024; 0: at
#                start only). The three go to both ranks, explicitly, with KV_SHARD=1 only
# POOL:          zig/patches/0008's setting (scripts/config.sh): total pool positions, every stream's extent together;
#                CONTEXT stays the window one request may use. Empty: with KV_SHARD=1 the pool it needs (port_shard_pool:
#                2,101,248 at CONTEXT=1048576 and 16-row rounds, so another CONTEXT or ROUND_ROWS gets its own); with
#                KV_SHARD=0 CONTEXT (as before). TF_DS_POOL / TF_DS_WINDOW exported by hand are ignored (warned): POOL is
#                the one setting, so both ranks always get the same pool and window
# RANK1_EXTRA_ENV: TESTS ONLY (tools/gpu_validate.sh's mismatch cases): NAME=value words added last to rank 1's
#                environment, so they win there: a deliberately different rank 1. Empty (the default) for serving
# Each setting is also taken from its engine variable when only that one is set (TF_DS_KV_SHARD,
# TF_DS_SPILL_GIB, TF_DS_SPILL_MIN_FREE_GIB, TF_DS_SPILL_QUOTA as a share of the cap, TF_DS_SPILL_MIN_TOKENS, ...), so a
# variable exported by hand is not silently overridden by the default (start.sh passes the mapped names only through
# port_map). TF_DS_SPILL_DIR is the folder inside the container (/spill) and cannot name a Spark's folder: exported
# alone it is ignored with a warning. The round-1 placeholder names TF_SPILL_DIR, TF_SPILL_GIB, TF_SPILL_MIN_FREE_GIB
# and TF_SPILL_QUOTA are no engine's: exported, they are ignored with a warning and passed to no rank.
# Whether the two switches were set (here, scripts/local.sh or the environment) or are the recipe's defaults: a default
# that cannot run degrades, an explicit one refuses.
PORT_SPILL_EXPLICIT=0; [[ -z "${SPILL_GIB:-}${TF_DS_SPILL_GIB:-}" ]] || PORT_SPILL_EXPLICIT=1
PORT_SHARD_EXPLICIT=0; [[ -z "${KV_SHARD:-}${TF_DS_KV_SHARD:-}" ]] || PORT_SHARD_EXPLICIT=1
PORT_SPILL_DEFAULT_GIB=0   # the spill tier's cap a Spark when SPILL_GIB is not set (0: off)
PORT_SHARD_DEFAULT=0       # KV_SHARD when it is not set (0: off)
SPILL_GIB="${SPILL_GIB:-${TF_DS_SPILL_GIB:-$PORT_SPILL_DEFAULT_GIB}}"
SPILL_DIR="${SPILL_DIR:-$(dirname "$HF_CACHE")/dsv41-tensorfold-spill}"
WORKER_SPILL_DIR="${WORKER_SPILL_DIR:-$SPILL_DIR}"
SPILL_FLOOR_GIB="${SPILL_FLOOR_GIB:-${TF_DS_SPILL_MIN_FREE_GIB:-50}}"
# TF_DS_SPILL_QUOTA is the engine's share (0 to 1) of SPILL_GIB; SPILL_QUOTA_GIB is GiB. An invalid share is left for
# port_validate to refuse (PORT_BAD_QUOTA).
PORT_BAD_QUOTA=""
if [[ -z "${SPILL_QUOTA_GIB:-}" && -n "${TF_DS_SPILL_QUOTA:-}" ]]; then
  if [[ "$TF_DS_SPILL_QUOTA" =~ ^(0(\.[0-9]+)?|1(\.0+)?|\.[0-9]+)$ && "$SPILL_GIB" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    SPILL_QUOTA_GIB=$(awk -v q="$TF_DS_SPILL_QUOTA" -v g="$SPILL_GIB" 'BEGIN {printf "%.6g", q * g}')
  else PORT_BAD_QUOTA=$TF_DS_SPILL_QUOTA; fi
fi
SPILL_QUOTA_GIB="${SPILL_QUOTA_GIB:-0}"
SPILL_MIN_TOKENS="${SPILL_MIN_TOKENS:-${TF_DS_SPILL_MIN_TOKENS:-8192}}"
SPILL_STAGING_MIB="${SPILL_STAGING_MIB:-${TF_DS_SPILL_STAGING_MIB:-128}}"
SPILL_TURNS="${SPILL_TURNS:-${TF_SPILL_TURNS:-4}}"
SPILL_FLUSH_MS="${SPILL_FLUSH_MS:-${TF_SPILL_FLUSH_MS:-60000}}"
SPILL_WEIGHTS_SHA="${SPILL_WEIGHTS_SHA:-1}"
MODEL_QUANT="${MODEL_QUANT:-}"
KV_SHARD="${KV_SHARD:-${TF_DS_KV_SHARD:-$PORT_SHARD_DEFAULT}}"
KV_SHARD_DECODE="${KV_SHARD_DECODE:-${TF_DS_KV_SHARD_DECODE:-}}"
KV_SHARD_CARRY="${KV_SHARD_CARRY:-${TF_DS_KV_SHARD_CARRY:-1}}"
KV_SHARD_TWINS="${KV_SHARD_TWINS:-${TF_DS_KV_SHARD_TWINS:-1}}"
KV_SHARD_CHECK="${KV_SHARD_CHECK:-${TF_DS_KV_SHARD_CHECK:-1024}}"
RANK1_EXTRA_ENV="${RANK1_EXTRA_ENV:-}"
# Spill and KV sharding together: the engine supports it since round 3 (port/spill-shard: owner-layout identity, part
# salt, restores into the recorded part). 0 refuses the combination here again.
PORT_SHARD_SPILL_READY=1
# Where each rank sees its spill directory.
PORT_SPILL_MOUNT=/spill

port_num()   { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }
port_gt0()   { awk -v x="$1" 'BEGIN {exit !(x + 0 > 0)}'; }
port_spill_on() { port_num "$SPILL_GIB" && port_gt0 "$SPILL_GIB"; }
# port_pool_set: the pool positions both ranks are given apart from CONTEXT: POOL, else (KV_SHARD=1) the pool sharding
# needs, else empty (the pool is CONTEXT, as before the port). Computed when asked, so a KV_SHARD turned off by
# port_validate takes its derived pool with it.
port_pool_set() {
  if [[ -n "$POOL" ]]; then echo "$POOL"
  elif [[ "$KV_SHARD" == 1 && "$CONTEXT" =~ ^[1-9][0-9]{0,8}$ && "$(port_rows)" =~ ^[0-9]{1,3}$ ]]; then port_shard_pool; fi
}
port_pool()  { local p; p=$(port_pool_set); echo "${p:-$CONTEXT}"; }
port_rows()  { echo "${ROUND_ROWS:-16}"; }
# The pool KV_SHARD=1 needs so that each of the two owner parts holds a whole CONTEXT window, a round's rows and two
# (the engine's shard.plan rule), in whole 2,048-position blocks: 2,101,248 for 1,048,576 at 16 rows.
port_shard_pool() { local n=$(( CONTEXT + $(port_rows) + 2 )); echo $(( (n + 2047) / 2048 * 2048 * 2 )); }
# port_shard_fits <pool>: that pool is whole 4,096-position parts and each owner part keeps the whole CONTEXT window
port_shard_fits() {
  [[ "$1" =~ ^[1-9][0-9]{0,17}$ ]] || return 1
  (( $1 % 4096 == 0 && $1 / 2 - $(port_rows) - 2 >= CONTEXT ))
}
# A clean stop writes what still waits within the tier's flush bound (SPILL_FLUSH_MS, the engine's TF_SPILL_FLUSH_MS,
# default 60000): give the ranks that long, rounded up to whole seconds, and 30 s (the stop without the tier). 90 s by
# default; an invalid SPILL_FLUSH_MS counts as the default here and port_validate refuses it. Unchanged with the tier off.
port_flush_ms() { if [[ "$SPILL_FLUSH_MS" =~ ^[0-9]{1,9}$ ]]; then echo $(( 10#$SPILL_FLUSH_MS )); else echo 60000; fi; }
port_stop_timeout() { echo $(( ( $(port_flush_ms) + 999 ) / 1000 + 30 )); }
PORT_STOP_TIMEOUT_SET=0
if port_spill_on && [[ -z "${STOP_TIMEOUT:-}" ]]; then STOP_TIMEOUT=$(port_stop_timeout); PORT_STOP_TIMEOUT_SET=1; fi

# port_spill_default_off <reason>: the spill tier was on by default only and cannot run here: say so loudly and start the
# pair without it (both ranks: port_map then gives neither a spill flag, variable or mount). An explicit SPILL_GIB never
# comes here: its failures refuse (port_spill_refuse).
port_spill_default_off() {
  warn "SPILL TIER OFF for this start: $1. The spill tier is on by default (SPILL_GIB=$SPILL_GIB a Spark) and needs a usable local folder on BOTH Sparks (SPILL_DIR here, WORKER_SPILL_DIR on the worker); the server starts without it (kept prompt states stay in memory only, and are gone after a restart). Fix the folder and run ./start.sh restart, set SPILL_GIB=0 to turn the tier off without this warning, or set SPILL_GIB explicitly to make this a refusal"
  SPILL_GIB=0
  if (( PORT_STOP_TIMEOUT_SET )); then unset STOP_TIMEOUT; PORT_STOP_TIMEOUT_SET=0; fi
  PORT_SPILL_REAL=(); PORT_SPILL_UID=()
}
# port_spill_refuse <reason>: a spill folder check failed. Explicit SPILL_GIB: refuse (die). Default: PORT_SPILL_WHY
# keeps the reason and the caller returns 1 (port_spill_check then turns the tier off). Returns 0 when it does not die.
PORT_SPILL_WHY=""
port_spill_refuse() { (( PORT_SPILL_EXPLICIT )) || { PORT_SPILL_WHY=$1; return 0; }; die "$1"; }
# The pool's state on each Spark, estimated (not measured): ~890 bytes a position, the per-position layout of the
# spill tier's blocks (1,822,720 bytes a 2,048-position block). POOL above 4x CONTEXT warns, above 16x is refused.
PORT_POOL_BYTES_PER_POS=890
# KV_SHARD=1: each Spark keeps only its own part's compressed rows (~530 bytes a position of the pool; measured at
# 2,101,248 positions: 1,062 MiB of compressed planes a rank against 1,783 MiB replicated)
PORT_POOL_BYTES_PER_POS_SHARDED=530

# ---------------------------------------------------------------- checks (no ssh, nothing changed)
# port_validate: every setting's form, and the combinations the engine would refuse at load, before anything is stopped.
port_validate() {
  local pool part rows w need share est
  [[ "$KV_SHARD" =~ ^[01]$ ]] || die "KV_SHARD is 0 or 1, not $KV_SHARD"
  # the recipe's default KV_SHARD=1 where it cannot run: off, with a warning (explicit: the checks below refuse)
  if [[ "$KV_SHARD" == 1 ]] && (( ! PORT_SHARD_EXPLICIT )); then
    local why=""
    if [[ "$TP" != 2 ]]; then why="it is written for two Sparks (TP=2), not TP=$TP"
    elif [[ -n "$POOL" ]] && ! port_shard_fits "$POOL"; then
      why="POOL=$POOL (set explicitly) is not the pool it needs at CONTEXT=$CONTEXT (two whole-window owner parts: POOL=$(port_shard_pool), or leave POOL empty)"
    fi
    if [[ -n "$why" ]]; then
      warn "KV SHARDING OFF for this start: KV_SHARD=1 is the recipe's default, but $why. Set KV_SHARD=0 to silence this, or KV_SHARD=1 explicitly to make it a refusal"
      KV_SHARD=0
    fi
  fi
  [[ -z "$KV_SHARD_DECODE" || "$KV_SHARD_DECODE" =~ ^(fetch|owner)$ ]] || die "KV_SHARD_DECODE is fetch, owner or empty, not $KV_SHARD_DECODE"
  [[ -z "$KV_SHARD_DECODE" || "$KV_SHARD" == 1 ]] || warn "KV_SHARD_DECODE=$KV_SHARD_DECODE does nothing without KV_SHARD=1"
  [[ "$KV_SHARD_CARRY" =~ ^[01]$ ]] || die "KV_SHARD_CARRY is 0 or 1, not $KV_SHARD_CARRY"
  [[ "$KV_SHARD_TWINS" =~ ^[01]$ ]] || die "KV_SHARD_TWINS is 0 or 1, not $KV_SHARD_TWINS"
  [[ "$KV_SHARD_CHECK" =~ ^[0-9]{1,18}$ ]] || die "KV_SHARD_CHECK is a count of decode rounds (0: at start only), not $KV_SHARD_CHECK"
  [[ "$KV_SHARD" == 1 || "$KV_SHARD_CARRY$KV_SHARD_TWINS$KV_SHARD_CHECK" == 111024 ]] ||
    warn "KV_SHARD_CARRY / KV_SHARD_TWINS / KV_SHARD_CHECK do nothing without KV_SHARD=1"
  port_rank1_env_ok || die "RANK1_EXTRA_ENV is NAME=value words (NAME of A-Z 0-9 _, values of A-Z a-z 0-9 _ . , : = / + @ -), not '$RANK1_EXTRA_ENV'"
  [[ -z "$RANK1_EXTRA_ENV" ]] || warn "RANK1_EXTRA_ENV=$RANK1_EXTRA_ENV: rank 1 runs with other settings than rank 0 (a test of the ranks' agreement, never for serving)"
  local x
  for x in TF_SPILL_DIR TF_SPILL_GIB TF_SPILL_MIN_FREE_GIB TF_SPILL_QUOTA; do
    [[ -z "${!x:-}" ]] || warn "$x=${!x} is ignored and passed to no rank: it was a round-1 placeholder, not an engine name (SPILL_DIR / SPILL_GIB / SPILL_FLOOR_GIB / SPILL_QUOTA_GIB, or TF_DS_SPILL_GIB, TF_DS_SPILL_MIN_FREE_GIB, TF_DS_SPILL_QUOTA)"
  done
  for x in TF_DS_POOL TF_DS_WINDOW; do
    [[ -z "${!x:-}" ]] || warn "$x=${!x} is ignored and passed to no rank: the pool is the one POOL setting (start.sh gives both ranks TF_DS_POOL, TF_DS_WINDOW=CONTEXT and rank 1's --pool from it)"
  done
  [[ -z "${TF_DS_RANK_CACHE_SHA:-}" ]] || ! port_spill_on || warn "TF_DS_RANK_CACHE_SHA=$TF_DS_RANK_CACHE_SHA is ignored: each rank's is its own weight file's digest (SPILL_WEIGHTS_SHA=1 computes it, 0 leaves it out)"
  [[ -z "$POOL" || "$POOL" =~ ^[1-9][0-9]*$ ]] || die "POOL is empty (CONTEXT, or with KV_SHARD=1 the pool it needs) or a count of positions, not $POOL"
  pool=$(port_pool)
  (( pool >= CONTEXT )) || die "POOL ($pool) is below CONTEXT ($CONTEXT): the pool holds every stream's extent, a request's window must fit in it"
  local bpp=$PORT_POOL_BYTES_PER_POS; [[ "$KV_SHARD" != 1 ]] || bpp=$PORT_POOL_BYTES_PER_POS_SHARDED
  est=$(awk -v p="$pool" -v b="$bpp" 'BEGIN {printf "%.1f", p * b / 1073741824}')
  (( pool <= 16 * CONTEXT )) || die "POOL ($pool) is more than 16x CONTEXT ($CONTEXT): ~$est GiB a Spark for the pool's state (estimated, ~$bpp bytes a position), out of the memory the weights leave. Lower POOL"
  (( pool <= 4 * CONTEXT )) || warn "POOL ($pool) is more than 4x CONTEXT ($CONTEXT): ~$est GiB a Spark for the pool's state (estimated, ~$bpp bytes a position); watch MemAvailable against MEMGUARD_GIB"
  port_num "$SPILL_GIB" || die "SPILL_GIB is a number of GiB (0: off), not $SPILL_GIB"
  port_num "$SPILL_FLOOR_GIB" || die "SPILL_FLOOR_GIB is a number of GiB, not $SPILL_FLOOR_GIB"
  port_num "$SPILL_QUOTA_GIB" || die "SPILL_QUOTA_GIB is a number of GiB (0: no quota), not $SPILL_QUOTA_GIB"
  [[ -z "$PORT_BAD_QUOTA" ]] || die "TF_DS_SPILL_QUOTA is the engine's share of SPILL_GIB, 0 to 1 (with SPILL_GIB a number), not $PORT_BAD_QUOTA: set SPILL_QUOTA_GIB instead"
  [[ -z "${TF_DS_SPILL_DIR:-}" ]] || warn "TF_DS_SPILL_DIR=$TF_DS_SPILL_DIR is ignored: the ranks always see their spill folder at $PORT_SPILL_MOUNT; set SPILL_DIR / WORKER_SPILL_DIR (each Spark's folder) instead"
  # the default spill tier with a folder name it cannot use: off (port_spill_default_off); explicit: refused below
  if port_spill_on && (( ! PORT_SPILL_EXPLICIT )); then
    if [[ ! "$SPILL_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]]; then port_spill_default_off "SPILL_DIR '$SPILL_DIR' is not an absolute path of A-Z a-z 0-9 . _ / - only"
    elif [[ ! "$WORKER_SPILL_DIR" =~ ^(/|\$HOME/)[A-Za-z0-9._/-]+$ ]]; then port_spill_default_off "WORKER_SPILL_DIR '$WORKER_SPILL_DIR' is not an absolute path (or \$HOME/...) of A-Z a-z 0-9 . _ / - only"; fi
  fi
  if port_spill_on; then
    [[ "$SPILL_FLUSH_MS" =~ ^[0-9]{1,9}$ ]] || die "SPILL_FLUSH_MS (TF_SPILL_FLUSH_MS) is a whole number of milliseconds, not $SPILL_FLUSH_MS"
    [[ "$SPILL_MIN_TOKENS" =~ ^[0-9]{1,9}$ ]] || die "SPILL_MIN_TOKENS is a token count, not $SPILL_MIN_TOKENS"
    (( 10#$SPILL_MIN_TOKENS % 2048 == 0 )) || warn "SPILL_MIN_TOKENS=$SPILL_MIN_TOKENS: kept points sit at 2,048-token boundaries, so it acts as the next multiple of 2048"
    [[ "$SPILL_STAGING_MIB" =~ ^[0-9]{1,6}$ ]] && (( 10#$SPILL_STAGING_MIB >= 16 )) || die "SPILL_STAGING_MIB is a whole number of MiB, at least 16 (the engine's minimum), not $SPILL_STAGING_MIB"
    [[ "$SPILL_TURNS" =~ ^[0-9]{1,6}$ ]] || die "SPILL_TURNS (TF_SPILL_TURNS) is a count of turns, not $SPILL_TURNS"
    [[ "$SPILL_WEIGHTS_SHA" =~ ^[01]$ ]] || die "SPILL_WEIGHTS_SHA is 0 or 1, not $SPILL_WEIGHTS_SHA"
    [[ -z "$MODEL_QUANT" || "$MODEL_QUANT" =~ ^[A-Za-z0-9._+-]+$ ]] || die "MODEL_QUANT is a name of A-Z a-z 0-9 . _ + - only, not '$MODEL_QUANT'"
    [[ "$SPILL_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "SPILL_DIR is an absolute path of A-Z a-z 0-9 . _ / - only, not '$SPILL_DIR'"
    [[ "$WORKER_SPILL_DIR" =~ ^(/|\$HOME/)[A-Za-z0-9._/-]+$ ]] || die "WORKER_SPILL_DIR is an absolute path (or \$HOME/...) of A-Z a-z 0-9 . _ / - only, not '$WORKER_SPILL_DIR'"
    awk -v q="$SPILL_QUOTA_GIB" -v g="$SPILL_GIB" 'BEGIN {exit !(q + 0 <= g + 0)}' ||
      die "SPILL_QUOTA_GIB ($SPILL_QUOTA_GIB) is more than SPILL_GIB ($SPILL_GIB): one conversation's share of the cap"
  fi
  if [[ "$KV_SHARD" == 1 ]]; then
    (( TP == 2 )) || die "KV_SHARD=1 is written for two Sparks (TP=2), not $TP"
    rows=$(port_rows)
    (( pool % 4096 == 0 )) || die "KV_SHARD=1 splits the pool into two owner parts of whole 2,048-position blocks: POOL ($pool) is not a multiple of 4096 (try $(port_shard_pool))"
    part=$(( pool / 2 ))
    w=$(( part - rows - 2 ))
    need=$(port_shard_pool)
    (( w > 0 )) || die "KV_SHARD=1: an owner part of $part positions leaves no room after a round's $rows rows and two: set POOL=$need"
    (( w >= CONTEXT )) || die "KV_SHARD=1: with POOL=$pool a conversation fits in one owner part of $part positions, up to $w tokens, less than CONTEXT ($CONTEXT). Set POOL=$need to keep the whole window, or lower CONTEXT to $w"
    (( PARALLEL > 1 )) || warn "KV_SHARD=1 with PARALLEL=1: one stream uses one owner part, the other half of the pool holds only kept states"
    if port_spill_on && (( ! PORT_SHARD_SPILL_READY )); then
      die "KV_SHARD=1 and SPILL_GIB>0 do not run together yet: the spill identity does not carry the owner layout and part (the engine refuses it: SpillWithKvShard, until port/spill-shard lands). Turn one off"
    fi
  fi
  if port_spill_on; then
    share=$(awk -v q="$SPILL_QUOTA_GIB" -v g="$SPILL_GIB" 'BEGIN {printf "%.6g", q / g}')
    [[ "$share" != 1 ]] || [[ "$SPILL_QUOTA_GIB" == 0 ]] || warn "SPILL_QUOTA_GIB equals SPILL_GIB: no quota in effect"
  fi
  return 0
}

# ---------------------------------------------------------------- the mapping (the one place)
# port_map fills:
#   PORT_ENV     NAME=value pairs both ranks get, identical (docker -e)
#   PORT_SERVE   rank 0's extra `tensorfold-native serve` flags (checked by scripts/zig_flags.py before anything stops)
#   POOL         set to the pool KV_SHARD=1 needs when it is empty (port_pool_set); start.sh then gives both ranks
#                TF_DS_POOL and TF_DS_WINDOW, and rank 1 --pool, from it, as zig/patches/0008 does for any POOL
#   PORT_RANK1_ENV  RANK1_EXTRA_ENV's words (tests only; empty for serving), added last to rank 1's environment
#   PORT_MAPPED_RE  the engine variable names set here: an exported variable of these names is not passed a second time
# port_ident_map (called by start.sh once the image is there) fills PORT_IDENT_ENV, the spill tier's identity, and
# port_weights_check PORT_RANK_SHA (each rank's own TF_DS_RANK_CACHE_SHA).
# Engine names (docs/PORT-RECIPE.md has the table), checked 2026-10-22 against port/series-pair-370daf0:
#   rank 0 (server/cli.zig): --spill-dir, --spill-gib, --spill-min-free-gib, --spill-min-tokens, --spill-quota,
#     TF_SPILL_TURNS, TF_SPILL_FLUSH_MS; TF_DS_SPILL_STAGING_MIB (native.spillSettings)
#   rank 1 (tf-dsv41-lanes: spill_tier.Settings.follower): TF_DS_SPILL_DIR, TF_DS_SPILL_GIB, TF_DS_SPILL_MIN_FREE_GIB,
#     TF_DS_SPILL_MIN_TOKENS (must equal rank 0's --spill-min-tokens: the fingerprint's features),
#     TF_DS_SPILL_QUOTA, TF_DS_SPILL_STAGING_MIB, TF_SPILL_TURNS, TF_SPILL_FLUSH_MS. It also takes them as --spill-*
#     flags; the variables are used so that rank 1's command line is the one before the port
#   every rank (rank_identity.Extra.env, required with the tier): TF_DS_ENGINE_ID, TF_DS_RECIPE_PATCHES,
#     TF_DS_MODEL_REVISION and/or TF_DS_RANK_CACHE_SHA, TF_DS_ENGRAM_REVISION (Engram is always on here);
#     optional TF_DS_BUILD_FLAGS (exported by hand: it passes through as it is)
#   every rank's Model.open / kv_lanes.Settings: TF_DS_KV_SHARD, TF_DS_KV_SHARD_DECODE, TF_DS_KV_SHARD_CARRY,
#     TF_DS_KV_SHARD_TWINS, TF_DS_KV_SHARD_CHECK (the pool and window are 0008's: start.sh, from POOL)
PORT_MAPPED_BASE='TF_DS_POOL|TF_DS_WINDOW|TF_DS_KV_SHARD|TF_DS_KV_SHARD_DECODE|TF_SPILL_DIR|TF_SPILL_GIB|TF_SPILL_MIN_FREE_GIB|TF_SPILL_QUOTA'
PORT_MAPPED_SPILL='TF_DS_SPILL_DIR|TF_DS_SPILL_GIB|TF_DS_SPILL_MIN_FREE_GIB|TF_DS_SPILL_MIN_TOKENS|TF_DS_SPILL_QUOTA|TF_DS_SPILL_STAGING_MIB|TF_SPILL_TURNS|TF_SPILL_FLUSH_MS|TF_DS_ENGINE_ID|TF_DS_RECIPE_PATCHES|TF_DS_MODEL_REVISION|TF_DS_RANK_CACHE_SHA|TF_DS_ENGRAM_REVISION'
PORT_MAPPED_SHARD='TF_DS_KV_SHARD_CARRY|TF_DS_KV_SHARD_TWINS|TF_DS_KV_SHARD_CHECK'
PORT_MAPPED_RE=$PORT_MAPPED_BASE
port_map() {
  local share
  PORT_ENV=(); PORT_SERVE=(); PORT_RANK1_ENV=()
  PORT_MAPPED_RE=$PORT_MAPPED_BASE
  POOL=$(port_pool_set)   # (empty stays empty: the pool is CONTEXT)
  if [[ "$KV_SHARD" == 1 ]]; then
    PORT_ENV+=("TF_DS_KV_SHARD=1")
    [[ -z "$KV_SHARD_DECODE" ]] || PORT_ENV+=("TF_DS_KV_SHARD_DECODE=$KV_SHARD_DECODE")
    PORT_ENV+=("TF_DS_KV_SHARD_CARRY=$KV_SHARD_CARRY" "TF_DS_KV_SHARD_TWINS=$KV_SHARD_TWINS" "TF_DS_KV_SHARD_CHECK=$(( 10#$KV_SHARD_CHECK ))")
    PORT_MAPPED_RE+="|$PORT_MAPPED_SHARD"
  fi
  if port_spill_on; then
    share=$(awk -v q="$SPILL_QUOTA_GIB" -v g="$SPILL_GIB" 'BEGIN {printf "%.6g", q / g}')
    PORT_SERVE+=(--spill-dir "$PORT_SPILL_MOUNT" --spill-gib "$SPILL_GIB" --spill-min-free-gib "$SPILL_FLOOR_GIB"
                 --spill-min-tokens "$(( 10#$SPILL_MIN_TOKENS ))")
    port_gt0 "$SPILL_QUOTA_GIB" && PORT_SERVE+=(--spill-quota "$share")
    # rank 1's (the follower reads them; rank 0 reads only the last three, and its flags above for the rest)
    PORT_ENV+=("TF_DS_SPILL_DIR=$PORT_SPILL_MOUNT" "TF_DS_SPILL_GIB=$SPILL_GIB" "TF_DS_SPILL_MIN_FREE_GIB=$SPILL_FLOOR_GIB"
               "TF_DS_SPILL_MIN_TOKENS=$(( 10#$SPILL_MIN_TOKENS ))" "TF_DS_SPILL_QUOTA=$share"
               "TF_DS_SPILL_STAGING_MIB=$(( 10#$SPILL_STAGING_MIB ))" "TF_SPILL_TURNS=$(( 10#$SPILL_TURNS ))"
               "TF_SPILL_FLUSH_MS=$(port_flush_ms)")
    PORT_MAPPED_RE+="|$PORT_MAPPED_SPILL"
  fi
  [[ -z "$RANK1_EXTRA_ENV" ]] || read -ra PORT_RANK1_ENV <<<"$RANK1_EXTRA_ENV"
  return 0
}

# port_rank1_env_ok: RANK1_EXTRA_ENV is NAME=value words of safe characters (no shell or docker meaning)
port_rank1_env_ok() {
  local -a ws=(); local w
  [[ -z "$RANK1_EXTRA_ENV" ]] || read -ra ws <<<"$RANK1_EXTRA_ENV"
  for w in ${ws[@]+"${ws[@]}"}; do [[ "$w" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_.,:=/+@-]*$ ]] || return 1; done
  return 0
}

# port_check_extra ARGS...: rank 0's extra serve arguments (./start.sh [restart] ARGS) may not set the spill tier:
# its settings must reach both ranks (rank 1 would run with the defaults, and the fingerprint refuses the pair)
port_check_extra() {
  local a
  for a in "$@"; do
    case "$a" in
      --spill-*) die "rank 0's extra argument $a: the spill tier's settings reach both ranks only as settings (SPILL_GIB, SPILL_DIR, SPILL_FLOOR_GIB, SPILL_QUOTA_GIB, SPILL_MIN_TOKENS in scripts/local.sh or the environment), never as an extra argument" ;;
    esac
  done
  return 0
}

# ---------------------------------------------------------------- the spill tier's identity
# The engine refuses the tier unless every rank names the engine, the patch series and the weights' content
# (rank_identity.Extra.env: NoEngineId / NoRecipePatches / NoWeightsId / NoEngramId). They enter the ranks'
# fingerprint and the store's compatibility digest, so another image, checkpoint, quant or Engram revision reads none of
# the kept states the old one wrote (a new native-spill-<compat> folder). Derived here, never typed:
#   TF_DS_RECIPE_PATCHES  the image's tf.patches label: the hash of zig/patches, zig/Dockerfile and the pins that
#                         scripts/prepare.sh built it from (config.sh image_hash when the image is not there yet)
#   TF_DS_ENGINE_ID       tensorfold@<the image's tf.ref label (TF_REF), 12 hex>+<that patch hash>
#   TF_DS_MODEL_REVISION  <MODEL_ID>@<the checkpoint's revision>,quant=<MODEL_QUANT>,ablit=<0 | 1@ABLIT_REVISION>
#   TF_DS_ENGRAM_REVISION <ENGRAM_ID>@<Engram's revision>
#   TF_DS_RANK_CACHE_SHA  per rank: sha256:<its weight file's digest> (port_weights_check; SPILL_WEIGHTS_SHA=0: none)
# An exported TF_DS_ENGINE_ID / TF_DS_RECIPE_PATCHES / TF_DS_MODEL_REVISION / TF_DS_ENGRAM_REVISION wins.
# port_image_label <image> <label>: the label's value, empty when the image or the label is missing
port_image_label() {
  local v
  v=$(docker image inspect -f "{{index .Config.Labels \"$2\"}}" "$1" 2>/dev/null) || return 0
  [[ "$v" == '<no value>' ]] || echo "$v"
}
# port_model_quant: MODEL_QUANT, else read from MODEL_ID (".../X-EXL3-...-2.9-bpw" -> exl3-2.9bpw)
port_model_quant() {
  [[ -z "$MODEL_QUANT" ]] || { echo "$MODEL_QUANT"; return 0; }
  local id fmt="" bpw=""
  id=$(tr '[:upper:]' '[:lower:]' <<<"${MODEL_ID##*/}")
  [[ "$id" != *exl3* ]] || fmt=exl3
  [[ "$id" != *exl2* ]] || fmt=exl2
  [[ ! "$id" =~ ([0-9]+(\.[0-9]+)?)-?bpw ]] || bpw=${BASH_REMATCH[1]}bpw
  if [[ -n "$fmt$bpw" ]]; then echo "${fmt:-quant}${bpw:+-$bpw}"; else echo "$id"; fi
}
port_model_revision() { echo "$MODEL_ID@$(snapshot_rev),quant=$(port_model_quant),ablit=$(ablit_identity)"; }
# port_ident_map <image>: PORT_IDENT_ENV, the identity both ranks get (empty with the tier off)
port_ident_map() {
  PORT_IDENT_ENV=()
  port_spill_on || return 0
  local img=$1 patches ref
  # an empty revision would let two checkpoints (or Engram revisions) share one identity, and so one store
  [[ -n "${TF_DS_MODEL_REVISION:-}" || -n "$(snapshot_rev)" ]] ||
    die "the spill tier needs the checkpoint's revision: set MODEL_REVISION (scripts/local.sh), or fetch $MODEL_ID so $(model_cache_dir)/refs/main names one"
  [[ -n "${TF_DS_ENGRAM_REVISION:-}" || -n "$(engram_rev)" ]] ||
    die "the spill tier needs Engram's revision: set ENGRAM_REVISION (scripts/local.sh), or fetch $ENGRAM_ID so $(engram_cache_dir)/refs/main names one"
  patches=$(port_image_label "$img" tf.patches); [[ -n "$patches" ]] || patches=$(image_hash)
  ref=$(port_image_label "$img" tf.ref); [[ -n "$ref" ]] || ref=$TF_REF
  PORT_IDENT_ENV=("TF_DS_ENGINE_ID=${TF_DS_ENGINE_ID:-tensorfold@${ref:0:12}+$patches}"
                  "TF_DS_RECIPE_PATCHES=${TF_DS_RECIPE_PATCHES:-$patches}"
                  "TF_DS_MODEL_REVISION=${TF_DS_MODEL_REVISION:-$(port_model_revision)}"
                  "TF_DS_ENGRAM_REVISION=${TF_DS_ENGRAM_REVISION:-$ENGRAM_ID@$(engram_rev)}")
}
# rank_cache_sha <compute 0|1> <dir> <rank>: one line about the weight file rank<rank>of2-*.bin in <dir> on this node
# (self-contained: start.sh sends it to the worker; a leading literal $HOME/ is this node's home): sha256:<64 hex>
# (cached in <dir>/.<file>.sha256 with the file's size, mtime and inode; recomputed when any changed, with O_DIRECT
# reads so the page cache is not filled), PENDING (compute=0 and nothing cached), or ERR <reason>. The engine takes
# the one regular file of that name (rank_cache.find); so does this.
rank_cache_sha() {
  local compute=$1 d=$2 r=$3 f="" n=0 x st st2 a b c sum cache
  case "$d" in '$HOME/'*) d="$HOME/${d#\$HOME/}" ;; esac
  for x in "$d"/rank"$r"of2-*.bin; do
    if [ -f "$x" ] && [ ! -L "$x" ]; then f=$x; n=$((n + 1)); fi
  done
  [ "$n" = 1 ] || { echo "ERR $n weight files rank${r}of2-*.bin in $d"; return 0; }
  st=$(stat -c '%s %Y %i' -- "$f" 2>/dev/null) || { echo "ERR cannot stat $f"; return 0; }
  cache="$d/.$(basename -- "$f").sha256"
  if [ -r "$cache" ] && read -r a b c sum < "$cache" && [ "$a $b $c" = "$st" ] && [ "${#sum}" = 64 ] &&
     [ -z "$(printf '%s' "$sum" | tr -d 0-9a-f)" ]; then echo "sha256:$sum"; return 0; fi
  [ "$compute" = 1 ] || { echo PENDING; return 0; }
  sum=$(set -o pipefail; dd if="$f" bs=16M iflag=direct status=none 2>/dev/null | sha256sum | cut -c1-64) ||
    sum=$(sha256sum -- "$f" | cut -c1-64) || { echo "ERR cannot read $f"; return 0; }
  st2=$(stat -c '%s %Y %i' -- "$f" 2>/dev/null)
  [ "$st2" = "$st" ] || { echo "ERR $f changed while it was hashed"; return 0; }
  [ "${#sum}" = 64 ] && [ -z "$(printf '%s' "$sum" | tr -d 0-9a-f)" ] || { echo "ERR no digest of $f"; return 0; }
  { printf '%s %s\n' "$st" "$sum" > "$cache.$$" && mv -f -- "$cache.$$" "$cache"; } 2>/dev/null || rm -f -- "$cache.$$" 2>/dev/null
  echo "sha256:$sum"
}
# port_weights_check: PORT_RANK_SHA[0] and [1], both Sparks at once (start.sh, step 2; DRY_RUN computes nothing and
# uses a cached digest only). The digest enters the rank's fingerprint and so the store's folder (native-spill-<compat>):
# a rank that started once with it and once without would open another, empty store and every kept state would be
# lost. So a real start refuses when either rank's digest cannot be had (the worker's probe is tried twice); only
# DRY_RUN warns and goes on. Nothing with the tier off or SPILL_WEIGHTS_SHA=0 (which names another store, too).
declare -a PORT_RANK_SHA=()
port_weights_probe() {   # port_weights_probe <compute>: the worker's line
  local out
  [[ -z "${WORKER_DOWN[1]:-}" ]] || { echo "ERR the worker cannot be reached"; return 0; }
  out=$(worker 1 "$(declare -f rank_cache_sha); rank_cache_sha $1 '$WORKER_CACHE_DIR' 1" 2>/dev/null | head -n 1) || true
  echo "${out:-ERR no answer from the worker}"
}
port_weights_check() {
  PORT_RANK_SHA=()
  port_spill_on && [[ "$SPILL_WEIGHTS_SHA" == 1 ]] || return 0
  local compute=1 t0=$SECONDS l0 l1="" tmp r line bad=""
  [[ "${DRY_RUN:-0}" != 1 ]] || compute=0
  (( compute == 0 )) || log "Weights' digests (TF_DS_RANK_CACHE_SHA): a few seconds when cached, about a minute a Spark when a weight file is new"
  if tmp=$(mktemp "${STATE_DIR:-/tmp}/rank-sha.XXXXXX" 2>/dev/null); then
    port_weights_probe "$compute" > "$tmp" &
    l0=$(rank_cache_sha "$compute" "$DATA_DIR/rank-cache" 0)
    wait
    l1=$(head -n 1 "$tmp" 2>/dev/null); rm -f -- "$tmp"
  else
    l0=$(rank_cache_sha "$compute" "$DATA_DIR/rank-cache" 0)
  fi
  case "$l1" in sha256:*|PENDING) ;; *) l1=$(port_weights_probe "$compute") ;; esac   # a second try: an ssh glitch
  for r in 0 1; do
    if (( r == 0 )); then line=$l0; else line=$l1; fi
    case "$line" in
      sha256:*) PORT_RANK_SHA[r]=$line ;;
      PENDING) log "DRY_RUN: rank $r's weight-file digest is not cached yet: start.sh computes it" ;;
      *) if (( compute == 0 )); then warn "DRY_RUN: rank $r: no weight-file digest (${line#ERR }): a real start refuses"
         else bad+="${bad:+; }rank $r: ${line#ERR }"; fi ;;
    esac
  done
  [[ -z "$bad" ]] || die "no weight-file digest ($bad). It names the spill store, so a rank without it would open another, empty store and lose every kept state: fix that, or set SPILL_WEIGHTS_SHA=0 on purpose (a different store too). Nothing was stopped"
  log "Weights: rank 0 ${PORT_RANK_SHA[0]:-(revision only)}, rank 1 ${PORT_RANK_SHA[1]:-(revision only)} ($((SECONDS - t0)) s)"
}

# port_identity: the settings both ranks run with, as one word (the containers' tf.recipe.port label): "off" when every
# setting is off, as the containers started before the port (no label) read.
port_identity() {
  if ! port_spill_on && [[ "$KV_SHARD" != 1 && -z "$(port_pool_set)" ]]; then echo off; return 0; fi
  local s="pool=$(port_pool),shard=$KV_SHARD"
  [[ "$KV_SHARD" != 1 ]] || s+="/${KV_SHARD_DECODE:-fetch}/c${KV_SHARD_CARRY}t${KV_SHARD_TWINS}k$(( 10#$KV_SHARD_CHECK ))"
  # the folders as a checksum (one word, no paths in the label): another folder is another set of kept states
  if port_spill_on; then s+=",spill=$SPILL_GIB/$SPILL_FLOOR_GIB/$SPILL_QUOTA_GIB/$(( 10#$SPILL_MIN_TOKENS ))@$(port_dirs_sum)"; else s+=",spill=0"; fi
  echo "$s"
}

# port_dirs_sum: a checksum of SPILL_DIR and WORKER_SPILL_DIR (as configured; POSIX cksum)
port_dirs_sum() { printf '%s\n%s\n' "$SPILL_DIR" "$WORKER_SPILL_DIR" | cksum | awk '{print $1}'; }

# port_summary: the banner's words for what is on (empty when nothing is)
port_summary() {
  local s=""
  local pool; pool=$(port_pool_set)
  [[ -z "$pool" ]] || s+=" · pool $pool"
  [[ "$KV_SHARD" != 1 ]] || s+=" · KV shard (${KV_SHARD_DECODE:-fetch})"
  ! port_spill_on || s+=" · spill $SPILL_GIB GiB"
  echo "$s"
}

# ---------------------------------------------------------------- each rank's spill directory
# spill_dir_probe <create 0|1> <dir> [<path the directory must not overlap>...]: one line about <dir> on this node,
# self-contained (start.sh sends it to the worker over ssh). A leading literal $HOME/ is this node's home. With create=1
# a missing directory is made (as you, so docker never creates it as root); with 0 the nearest existing parent is
# described. Prints: OK|NEW <owner uid> <owner gid> <your uid> <fs type> <source device> <free bytes> <exported 0|1>
# <overlaps path or -> <home-or-root 0|1> <canonical path>, or ERR <reason>. Exported: under a path that the kernel's
# export table (/var/lib/nfs/etab) or /etc/exports or /etc/exports.d/*.exports lists.
spill_dir_probe() {
  local create=$1 d=$2 real p st=OK fs src avail exp=0 e over=- top=0 f fr x
  shift 2
  case "$d" in '$HOME/'*) d="$HOME/${d#\$HOME/}" ;; esac
  if [ ! -d "$d" ]; then
    if [ "$create" = 1 ]; then mkdir -p -- "$d" 2>/dev/null || { echo "ERR cannot create $d"; return 0; }
    else st=NEW; fi
  fi
  real=$(readlink -m -- "$d")
  p=$real; while [ ! -d "$p" ]; do p=$(dirname "$p"); done
  fs=$(findmnt -n -o FSTYPE -T "$p" 2>/dev/null | head -1); [ -n "$fs" ] || fs=$(stat -f -c %T "$p" 2>/dev/null)
  src=$(findmnt -n -o SOURCE -T "$p" 2>/dev/null | head -1 | tr -s ' \t' '__')
  avail=$(df -B1 --output=avail "$p" 2>/dev/null | tail -1 | tr -d ' ')
  for x in /var/lib/nfs/etab /etc/exports /etc/exports.d/*.exports; do
    [ -r "$x" ] || continue
    while read -r e _; do
      e=${e#\"}; e=${e%\"}
      case "$e" in /*) e=${e%/}; case "$real/" in "$e"/*) exp=1 ;; esac ;; esac
    done < "$x"
  done
  for f in "$@"; do
    [ -n "$f" ] || continue
    case "$f" in '$HOME/'*) f="$HOME/${f#\$HOME/}" ;; esac
    fr=$(readlink -m -- "$f")
    case "$real/" in "$fr"/*) over=$fr; break ;; esac
    case "$fr/" in "$real"/*) over=$fr; break ;; esac
  done
  [ "$real" != / ] && [ "$real" != "$(readlink -m -- "$HOME")" ] || top=1
  echo "$st $(stat -c '%u %g' "$p") $(id -u) ${fs:-unknown} ${src:--} ${avail:-0} $exp $over $top $real"
}

# port_spill_judge <rank> <where> <probe line>: refuse a spill directory that is not this rank's own local disk, or
# would hand files to root, or has less free than the floor. Sets PORT_SPILL_REAL[rank], PORT_SPILL_UID[rank]. Each
# refusal goes through port_spill_refuse: with SPILL_GIB explicit it dies; with the default it returns 1 and
# PORT_SPILL_WHY says why.
declare -a PORT_SPILL_REAL=() PORT_SPILL_UID=()
port_spill_judge() {
  local r=$1 where=$2 st uid gid me fs src avail exp over top real lc floor_b want_b
  read -r st uid gid me fs src avail exp over top real <<<"$3"
  local name=SPILL_DIR; (( r == 0 )) || name=WORKER_SPILL_DIR
  [[ "$st" == OK || "$st" == NEW ]] || { port_spill_refuse "$name on $where: ${3#ERR }"; return 1; }
  [[ "$real" =~ ^/[A-Za-z0-9._/-]+$ ]] || { port_spill_refuse "$name on $where resolves to '$real': a path of A-Z a-z 0-9 . _ / - only, please"; return 1; }
  case "$fs" in
    nfs|nfs4|cifs|smb|smb2|smb3|smbfs|fuse.sshfs|sshfs|9p|ceph|fuse.ceph*|glusterfs|fuse.glusterfs|lustre|afs|davfs|fuse.rclone|fuse.s3fs|fuse.juicefs)
      port_spill_refuse "$name $real on $where is on a network file system ($fs): the spill tier needs each Spark's own local disk (NVMe)"; return 1 ;;
    tmpfs|ramfs|devtmpfs|overlay|overlayfs|squashfs|zram)
      port_spill_refuse "$name $real on $where is on $fs (memory, a container layer or a read-only image): the spill tier needs each Spark's own local disk (NVMe)"; return 1 ;;
  esac
  case "$src" in
    /dev/zram*) port_spill_refuse "$name $real on $where is on $src, a compressed RAM disk: the spill tier needs each Spark's own local disk (NVMe)"; return 1 ;;
  esac
  [[ "$exp" == 0 ]] || { port_spill_refuse "$name $real on $where is inside an NFS export (/var/lib/nfs/etab, /etc/exports or /etc/exports.d): keep the spill tier out of the shared model export"; return 1; }
  [[ "$over" == - ]] || { port_spill_refuse "$name $real on $where overlaps $over (the model cache, NFS export, Engram or weight-file folder): give the spill tier a folder of its own"; return 1; }
  [[ "$top" == 0 ]] || { port_spill_refuse "$name on $where is / or your home folder itself: give the spill tier a folder of its own"; return 1; }
  lc=$(tr '[:upper:]' '[:lower:]' <<<"$real")
  [[ "$lc/" != */archive*/* ]] || { port_spill_refuse "$name $real on $where is under an archive folder: the spill tier writes and deletes there"; return 1; }
  if [[ "$st" == NEW ]]; then
    log "$name $real on $where does not exist yet: start.sh creates it (as you)"
  elif [[ "$uid" == 0 && "$me" != 0 ]]; then
    port_spill_refuse "$name $real on $where belongs to root: its files would too. Make it yours there: sudo chown $me $real"; return 1
  fi
  [[ "$uid" == "$me" || "$st" == NEW ]] || warn "$name $real on $where belongs to uid $uid, not to you ($me): its spill files stay root's (only a folder of your own is handed back: spill_fix_target)"
  floor_b=$(awk -v g="$SPILL_FLOOR_GIB" 'BEGIN {printf "%.0f", g * 1073741824}')
  want_b=$(awk -v g="$SPILL_GIB" -v f="$floor_b" 'BEGIN {printf "%.0f", g * 1073741824 + f}')
  awk -v a="$avail" -v f="$floor_b" 'BEGIN {exit !(a + 0 >= f + 0)}' ||
    { port_spill_refuse "$name $real on $where has $(( avail / 1073741824 )) GiB free, under SPILL_FLOOR_GIB ($SPILL_FLOOR_GIB): the tier would refuse every write. Free the disk or lower SPILL_GIB/SPILL_FLOOR_GIB"; return 1; }
  awk -v a="$avail" -v w="$want_b" 'BEGIN {exit !(a + 0 < w + 0)}' &&
    warn "$name $real on $where has $(( avail / 1073741824 )) GiB free, less than SPILL_GIB ($SPILL_GIB) + SPILL_FLOOR_GIB ($SPILL_FLOOR_GIB): the tier stops writing at the floor before it reaches its cap"
  PORT_SPILL_REAL[r]=$real; PORT_SPILL_UID[r]=$uid
  return 0
}

# port_spill_check: both ranks' spill directories, before anything is stopped (start.sh, step 2): created when missing
# (as you; DRY_RUN=1 creates nothing), on a local disk, outside the export and the model folders, above the floor.
# The two ranks run with the same settings by construction (port_map); this checks what each Spark has under them.
# start.sh runs it before port_map and before it compares a running pair's tf.recipe.port label, so a default tier
# that cannot run is off everywhere it matters: the flags, the variables, the mounts and the label. With SPILL_GIB
# explicit a failed check refuses the start; with the default it turns the tier off (port_spill_default_off), on both
# ranks, when either Spark's folder fails.
port_spill_check() {
  port_spill_on || return 0
  local create=1 line
  [[ "${DRY_RUN:-0}" != 1 ]] || create=0
  PORT_SPILL_WHY=""
  line=$(spill_dir_probe "$create" "$SPILL_DIR" "$HF_CACHE" "$(port_nfs_path)" "$HEAD_ENGRAM_DIR" "$DATA_DIR")
  if ! port_spill_judge 0 "this Spark" "$line"; then port_spill_default_off "$PORT_SPILL_WHY"; return 0; fi
  if [[ -n "${WORKER_DOWN[1]:-}" ]]; then
    warn "DRY_RUN: the worker cannot be reached: WORKER_SPILL_DIR ($WORKER_SPILL_DIR) is not checked"
    PORT_SPILL_REAL[1]="<WORKER:$WORKER_SPILL_DIR>"; return 0
  fi
  if ! line=$(worker 1 "$(declare -f spill_dir_probe); spill_dir_probe $create '$WORKER_SPILL_DIR' '$WORKER_ENGRAM_DIR' '$WORKER_CACHE_DIR' \"\${HF_HOME:-\$HOME/.cache/huggingface}\""); then
    port_spill_refuse "could not check WORKER_SPILL_DIR on $(worker_host 1)"
    port_spill_default_off "$PORT_SPILL_WHY"; return 0
  fi
  if ! port_spill_judge 1 "$(worker_host 1)" "$line"; then port_spill_default_off "$PORT_SPILL_WHY"; return 0; fi
  log "Spill tier: $SPILL_GIB GiB a Spark, floor $SPILL_FLOOR_GIB GiB, quota ${SPILL_QUOTA_GIB} GiB; rank 0 ${PORT_SPILL_REAL[0]}, rank 1 ${PORT_SPILL_REAL[1]}"
}
# NFS_PATH when it is a real path ("/" is an NFSv4 pseudo-root, not a folder of this Spark)
port_nfs_path() { [[ "$NFS_PATH" == / ]] || echo "$NFS_PATH"; }

# port_spill_args <rank>: that rank's docker arguments for the spill tier, one word each (no spaces): its directory at
# /spill and the tf.recipe.spill label (stop.sh gives the files back by it). Empty with the tier off. Every container
# also carries tf.recipe.port=$(port_identity) (start.sh's RUN_ARGS), which `./start.sh` compares on both ranks.
port_spill_args() {
  port_spill_on || return 0
  printf -- '--label tf.recipe.spill=%s -v %s:%s\n' "${PORT_SPILL_REAL[$1]}" "${PORT_SPILL_REAL[$1]}" "$PORT_SPILL_MOUNT"
}

# The ranks' containers run as root (the GPU, IPC_LOCK, /dev/infiniband), so the engine writes its spill files as root
# (mode 0600). These give those files back to the spill folder's owner, before a start (leftovers of a crash) and at
# every stop, so the person who owns the folder can read, move and delete them without sudo. Changing a file's owner
# away from root needs root (CAP_CHOWN), which is why a container does it, and why it is kept as narrow as it can be
# (maintainer review of the port, MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks#30):
#   - only a folder start.sh accepted and marked: an absolute, canonical path (no symlink anywhere on it), a real
#     directory (not a symlink), owned by the user running this (not root), holding the marker file
#     .dsv41-tensorfold-spill that start.sh created there as that user after port_spill_check passed; anything else is
#     refused and left untouched (a label that names another folder, a folder swapped for a symlink, a root-owned one)
#   - only that folder is mounted (at /spill), nothing else of the host
#   - only root-owned regular files and directories under it, never the folder itself, never a symlink (find without
#     -L does not follow them, `-type f -o -type d` leaves them out, chown --no-dereference), never across a file system
#     (-xdev); they become the invoking user's (id -u:id -g), not whatever the folder says
#   - the container has no network, a read-only root, no-new-privileges, and of root's capabilities only CHOWN and
#     DAC_READ_SEARCH (to walk the engine's 0700 folders); it runs the serving image's find (no other image is pulled)
# Self-contained (bash or sh, GNU or BSD stat): sent to the worker over ssh.
# spill_fix_target <dir>: 0 when <dir> may be handed back (the rules above; it prints why not, and returns 1, otherwise)
spill_fix_target() {
  local dir=$1 me u real m=.dsv41-tensorfold-spill
  me=$(id -u)
  case "$dir" in /*) ;; *) echo "spill folder '$dir' is not an absolute path: not touched"; return 1 ;; esac
  if [ -L "$dir" ] || [ ! -d "$dir" ]; then echo "spill folder $dir is a symlink or not a directory: not touched"; return 1; fi
  real=$(cd -P -- "$dir" 2>/dev/null && pwd -P) || { echo "spill folder $dir cannot be entered: not touched"; return 1; }
  [ "$real" = "${dir%/}" ] || { echo "spill folder $dir resolves to $real (a symlink on its path): not touched"; return 1; }
  u=$(stat -c %u -- "$dir" 2>/dev/null || stat -f %u -- "$dir" 2>/dev/null)
  [ "$u" = "$me" ] || { echo "spill folder $dir belongs to uid ${u:-?}, not to you ($me): not touched"; return 1; }
  if [ -L "$dir/$m" ] || [ ! -f "$dir/$m" ]; then echo "spill folder $dir has no $m (start.sh did not accept it): not touched"; return 1; fi
  u=$(stat -c %u -- "$dir/$m" 2>/dev/null || stat -f %u -- "$dir/$m" 2>/dev/null)
  [ "$u" = "$me" ] || { echo "spill folder $dir: its $m is not yours: not touched"; return 1; }
  return 0
}
# spill_fix_owner <dir> <image> [mark 0|1]: hand the root-owned entries under <dir> back (mark=1, start.sh only: first
# create the marker, as you, in a folder that passes every other rule)
spill_fix_owner() {
  local dir=$1 img=$2 mark=${3:-0} why m=.dsv41-tensorfold-spill
  [ -e "$dir" ] || [ -L "$dir" ] || return 0      # nothing there yet
  [ "$(id -u)" != 0 ] || return 0                  # run as root: the files are already the runner's
  if [ "$mark" = 1 ] && [ -d "$dir" ] && [ ! -L "$dir" ] && [ ! -e "$dir/$m" ] && [ ! -L "$dir/$m" ]; then
    : > "$dir/$m" 2>/dev/null || { echo "spill folder $dir: cannot create $m as you: not touched" >&2; return 1; }
  fi
  why=$(spill_fix_target "$dir") || { echo "$why" >&2; return 1; }
  docker run --rm --network none --read-only --security-opt no-new-privileges --cap-drop ALL --cap-add CHOWN \
    --cap-add DAC_READ_SEARCH --user 0:0 -v "$dir:/spill" --entrypoint find "$img" \
    /spill -xdev -mindepth 1 -uid 0 \( -type f -o -type d \) -exec chown --no-dereference "$(id -u):$(id -g)" {} + >/dev/null
}
# spill_fix_container <container>: the same for the spill directory a (stopped) container of start.sh's had, by its label
# (the label only names the folder: spill_fix_target still decides whether it is one start.sh accepted)
spill_fix_container() {
  local c=$1 dir img
  dir=$(docker inspect -f '{{index .Config.Labels "tf.recipe.spill"}}' "$c" 2>/dev/null) || return 0
  [ -n "$dir" ] && [ "$dir" != "<no value>" ] || return 0
  img=$(docker inspect -f '{{.Image}}' "$c") || return 1
  spill_fix_owner "$dir" "$img"
}
# port_spill_fix_both: before a start, on both Sparks, once port_spill_check passed (warnings only: the engine runs as
# root and reads them anyway); it marks each accepted folder, so later stops may hand its files back
port_spill_fix_both() {
  port_spill_on || return 0
  spill_fix_owner "${PORT_SPILL_REAL[0]}" "$IMAGE" 1 || warn "could not give rank 0's spill files back to you in ${PORT_SPILL_REAL[0]}"
  worker 1 "$(declare -f spill_fix_target spill_fix_owner); spill_fix_owner '${PORT_SPILL_REAL[1]}' '$IMAGE' 1" ||
    warn "could not give rank 1's spill files back to you in ${PORT_SPILL_REAL[1]} on $(worker_host 1)"
}
