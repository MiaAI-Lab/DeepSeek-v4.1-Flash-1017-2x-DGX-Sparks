#!/usr/bin/env bash
# Checks of scripts/port.sh without the Sparks: the defaults (everything off: no flag, variable or mount), a fork's
# defaults turned on (PT_FORK=1: spill 64 GiB, KV sharding, the derived pool) degrading where an explicit setting
# refuses, each turned off, the mapping to the engine's flags and variables, the refusals (port_validate) and the spill directory rules (port_spill_judge, on
# synthetic probe lines), and on Linux spill_dir_probe on a temporary folder. bash 3.2 or later; no docker, no ssh.
# Usage: scripts/port_test.sh      (exits 1 on the first failed check)
set -u
here=$(cd "$(dirname "$0")" && pwd)
fails=0; n=0
# PT_FORK=1 in a case's first script: port.sh with a fork's defaults on (PORT_SPILL_DEFAULT_GIB=64, PORT_SHARD_DEFAULT=1),
# the configuration the degrade paths exist for
fork_sh=$(mktemp "${TMPDIR:-/tmp}/port-fork.XXXXXX")
trap 'rm -f -- "$fork_sh"' EXIT
sed -e 's/^PORT_SPILL_DEFAULT_GIB=[0-9.]*/PORT_SPILL_DEFAULT_GIB=64/' -e 's/^PORT_SHARD_DEFAULT=[01]/PORT_SHARD_DEFAULT=1/' \
  "$here/port.sh" > "$fork_sh"

# case <name> <expected exit 0|1> <script...>: runs the script in a subshell over a fresh port.sh with the serving
# settings of a two-Spark default (CONTEXT 1048576, PARALLEL 8, 16-row rounds), die/warn/log stubbed
case_() {
  local name=$1 want=$2 got out
  shift 2
  n=$((n + 1))
  out=$( (
    TP=2 CONTEXT=1048576 PARALLEL=8 ROUND_ROWS=16 HF_CACHE=/home/u/.cache/deepseek-1017 NFS_PATH=/home/u/.cache/deepseek-1017
    die()  { echo "DIE: $*"; exit 1; }
    warn() { echo "WARN: $*"; }
    log()  { echo "LOG: $*"; }
    eval "$1"
    : "${POOL=}"   # scripts/config.sh's (zig/patches/0008's setting), before port.sh
    # shellcheck source=scripts/port.sh
    if [[ "${PT_FORK:-0}" == 1 ]]; then source "$fork_sh"; else source "$here/port.sh"; fi
    eval "$2"
  ) 2>&1 ); got=$?
  if [[ "$got" == "$want" ]]; then printf 'ok   %s\n' "$name"
  else fails=$((fails + 1)); printf 'FAIL %s (exit %s, wanted %s)\n%s\n' "$name" "$got" "$want" "$(sed 's/^/     /' <<<"$out")"; fi
}
# expect <a> <b>: inside a case, exit 1 when they differ
expect='expect() { [[ "$1" == "$2" ]] || { echo "got  [$1]"; echo "want [$2]"; exit 1; }; }'

# ---- defaults: everything off (the recipe before the port)
case_ "defaults: everything off, nothing added, POOL left empty" 0 "unset SPILL_GIB KV_SHARD POOL STOP_TIMEOUT" "$expect
  expect \"\$PORT_SPILL_EXPLICIT \$PORT_SHARD_EXPLICIT \$SPILL_GIB \$KV_SHARD\" '0 0 0 0'
  port_validate; port_map
  expect \"\${#PORT_ENV[@]} \${#PORT_SERVE[@]} [\$POOL]\" '0 0 []'
  expect \"\$(port_identity)\" off
  expect \"\$(port_summary)\" ''
  expect \"\${STOP_TIMEOUT:-unset}\" unset
  expect \"\$PORT_MAPPED_RE\" \"\$PORT_MAPPED_BASE\""
# ---- a fork's defaults on (Robert's: spill 64 GiB a Spark, KV sharding, the pool from CONTEXT)
case_ "fork defaults: spill 64 GiB, KV shard, the derived 2101248 pool" 0 "PT_FORK=1; unset SPILL_GIB KV_SHARD POOL STOP_TIMEOUT" "$expect
  expect \"\$PORT_SPILL_EXPLICIT \$PORT_SHARD_EXPLICIT\" '0 0'
  expect \"[\$POOL]\" '[]'
  port_validate; port_map
  expect \"\${PORT_SERVE[*]}\" '--spill-dir /spill --spill-gib 64 --spill-min-free-gib 50 --spill-min-tokens 8192'
  expect \"\${PORT_ENV[*]}\" 'TF_DS_KV_SHARD=1 TF_DS_KV_SHARD_CARRY=1 TF_DS_KV_SHARD_TWINS=1 TF_DS_KV_SHARD_CHECK=1024 TF_DS_SPILL_DIR=/spill TF_DS_SPILL_GIB=64 TF_DS_SPILL_MIN_FREE_GIB=50 TF_DS_SPILL_MIN_TOKENS=8192 TF_DS_SPILL_QUOTA=0 TF_DS_SPILL_STAGING_MIB=128 TF_SPILL_TURNS=4 TF_SPILL_FLUSH_MS=60000'
  expect \"\$POOL\" 2101248
  expect \"\$(port_identity)\" \"pool=2101248,shard=1/fetch/c1t1k1024,spill=64/50/0/8192@\$(port_dirs_sum)\"
  expect \"\$(port_summary)\" ' · pool 2101248 · KV shard (fetch) · spill 64 GiB'
  expect \"\$STOP_TIMEOUT\" 90
  expect \"\$SPILL_DIR\" /home/u/.cache/dsv41-tensorfold-spill
  expect \"\$WORKER_SPILL_DIR\" /home/u/.cache/dsv41-tensorfold-spill"
case_ "the derived pool follows CONTEXT and ROUND_ROWS" 0 "KV_SHARD=1 CONTEXT=262144" "$expect
  port_validate; expect \"\$(port_pool)\" 528384
  ROUND_ROWS=48; expect \"\$(port_pool)\" 528384; CONTEXT=526336; expect \"\$(port_pool)\" 1056768"
case_ "all off (SPILL_GIB=0 KV_SHARD=0): no flags, no variables, the pool is CONTEXT" 0 "PT_FORK=1; unset STOP_TIMEOUT; SPILL_GIB=0 KV_SHARD=0" "$expect
  port_validate; port_map
  expect \"\${#PORT_ENV[@]} \${#PORT_SERVE[@]} [\$POOL]\" '0 0 []'
  expect \"\$(port_identity)\" off
  expect \"\$(port_summary)\" ''
  expect \"\$(port_spill_args 0)\" ''
  expect \"\${STOP_TIMEOUT:-unset}\" unset
  expect \"\$PORT_MAPPED_RE\" \"\$PORT_MAPPED_BASE\""
case_ "SPILL_GIB=0.0 KV_SHARD=0 is off" 0 "PT_FORK=1; SPILL_GIB=0.0 KV_SHARD=0" "$expect; port_validate; port_map; expect \"\${#PORT_SERVE[@]}\" 0; expect \"\$(port_identity)\" off"
case_ "each off alone: SPILL_GIB=0 keeps KV shard, KV_SHARD=0 keeps spill and the CONTEXT pool" 0 "PT_FORK=1" "$expect
  ( SPILL_GIB=0; port_validate; port_map; expect \"\$(port_identity)\" 'pool=2101248,shard=1/fetch/c1t1k1024,spill=0'; expect \"\${#PORT_SERVE[@]}\" 0 ) || exit 1
  KV_SHARD=0; port_validate; port_map; expect \"[\$POOL]\" '[]'
  expect \"\$(port_identity)\" \"pool=1048576,shard=0,spill=64/50/0/8192@\$(port_dirs_sum)\""
case_ "POOL=CONTEXT with a fork's default KV shard: shard off with a warning, the pool as before" 0 "PT_FORK=1 SPILL_GIB=0 POOL=1048576" "$expect
  out=\$(port_validate); grep -q 'WARN: KV SHARDING OFF' <<<\"\$out\" || { echo \"\$out\"; exit 1; }
  port_validate >/dev/null; port_map; expect \"\$KV_SHARD \${#PORT_ENV[@]} \$POOL\" '0 0 1048576'"
case_ "refuse: POOL=CONTEXT with KV_SHARD=1 set explicitly" 1 "KV_SHARD=1 POOL=1048576" "port_validate"
case_ "a fork's default KV shard on another TP: off with a warning; explicit refuses" 0 "PT_FORK=1 SPILL_GIB=0 TP=1" "$expect
  out=\$(port_validate); grep -q 'WARN: KV SHARDING OFF.*TP=1' <<<\"\$out\" || { echo \"\$out\"; exit 1; }
  port_validate >/dev/null; port_map; expect \"\$(port_identity)\" off
  PORT_SHARD_EXPLICIT=1; KV_SHARD=1; ( port_validate ) && exit 1; exit 0"
case_ "TF_DS_SPILL_GIB / TF_DS_KV_SHARD exported count as explicit" 0 "export TF_DS_SPILL_GIB=8 TF_DS_KV_SHARD=0" "$expect
  expect \"\$PORT_SPILL_EXPLICIT \$PORT_SHARD_EXPLICIT \$SPILL_GIB \$KV_SHARD\" '1 1 8 0'"
case_ "a fork's default spill with an unusable SPILL_DIR name: off with a warning" 0 "PT_FORK=1 SPILL_DIR=relative" "$expect
  out=\$(port_validate); grep -q 'WARN: SPILL TIER OFF for this start: SPILL_DIR .relative.' <<<\"\$out\" || { echo \"\$out\"; exit 1; }
  port_validate >/dev/null; port_map; expect \"\${#PORT_SERVE[@]} \${STOP_TIMEOUT:-unset}\" '0 unset'
  expect \"\$(port_identity)\" 'pool=2101248,shard=1/fetch/c1t1k1024,spill=0'"
case_ "refuse: an unusable SPILL_DIR name with SPILL_GIB set" 1 "SPILL_GIB=64 SPILL_DIR=relative" "port_validate"

# ---- the mapping
case_ "spill on: rank 0 flags, both ranks' variables, quota as a share" 0 \
  "KV_SHARD=0 SPILL_GIB=64 SPILL_QUOTA_GIB=16 SPILL_FLOOR_GIB=40 SPILL_DIR=/nvme/spill" "$expect
  port_validate; port_map
  expect \"\${PORT_SERVE[*]}\" '--spill-dir /spill --spill-gib 64 --spill-min-free-gib 40 --spill-min-tokens 8192 --spill-quota 0.25'
  expect \"\${PORT_ENV[*]}\" 'TF_DS_SPILL_DIR=/spill TF_DS_SPILL_GIB=64 TF_DS_SPILL_MIN_FREE_GIB=40 TF_DS_SPILL_MIN_TOKENS=8192 TF_DS_SPILL_QUOTA=0.25 TF_DS_SPILL_STAGING_MIB=128 TF_SPILL_TURNS=4 TF_SPILL_FLUSH_MS=60000'
  expect \"\$(port_identity)\" \"pool=1048576,shard=0,spill=64/40/16/8192@\$(printf '/nvme/spill\\n/nvme/spill\\n' | cksum | awk '{print \$1}')\"
  expect \"\$STOP_TIMEOUT\" 90
  expect \"\$WORKER_SPILL_DIR\" /nvme/spill"
case_ "spill on without a quota: no --spill-quota" 0 "SPILL_GIB=8" "$expect
  port_validate; port_map
  expect \"\${PORT_SERVE[*]}\" '--spill-dir /spill --spill-gib 8 --spill-min-free-gib 50 --spill-min-tokens 8192'"
case_ "spill: min tokens, staging, turns and flush reach both ranks alike" 0 \
  "KV_SHARD=0 SPILL_GIB=8 SPILL_MIN_TOKENS=4096 SPILL_STAGING_MIB=256 SPILL_TURNS=2 SPILL_FLUSH_MS=1500" "$expect
  port_validate; port_map
  expect \"\${PORT_SERVE[*]}\" '--spill-dir /spill --spill-gib 8 --spill-min-free-gib 50 --spill-min-tokens 4096'
  expect \"\${PORT_ENV[*]}\" 'TF_DS_SPILL_DIR=/spill TF_DS_SPILL_GIB=8 TF_DS_SPILL_MIN_FREE_GIB=50 TF_DS_SPILL_MIN_TOKENS=4096 TF_DS_SPILL_QUOTA=0 TF_DS_SPILL_STAGING_MIB=256 TF_SPILL_TURNS=2 TF_SPILL_FLUSH_MS=1500'
  expect \"\$STOP_TIMEOUT\" 32"
case_ "spill: rank 0's --spill-min-tokens always equals rank 1's TF_DS_SPILL_MIN_TOKENS" 0 "export TF_DS_SPILL_GIB=8 TF_DS_SPILL_MIN_TOKENS=16384" "$expect
  port_validate; port_map
  a=\$(printf '%s\\n' \"\${PORT_SERVE[@]}\" | awk 'p { print; exit } \$0 == \"--spill-min-tokens\" { p = 1 }')
  b=\$(printf '%s\\n' \"\${PORT_ENV[@]}\" | sed -n 's/^TF_DS_SPILL_MIN_TOKENS=//p')
  expect \"\$a \$b\" '16384 16384'"
case_ "spill on: the engine's spill, identity and turn names are not passed through twice" 0 "SPILL_GIB=8" "$expect
  port_map
  for v in TF_DS_SPILL_DIR TF_DS_SPILL_GIB TF_DS_SPILL_MIN_TOKENS TF_DS_SPILL_STAGING_MIB TF_SPILL_TURNS TF_SPILL_FLUSH_MS \\
           TF_DS_ENGINE_ID TF_DS_RECIPE_PATCHES TF_DS_MODEL_REVISION TF_DS_RANK_CACHE_SHA TF_DS_ENGRAM_REVISION TF_SPILL_GIB; do
    [[ \"\$v=x\" =~ ^(\$PORT_MAPPED_RE)= ]] || { echo \"not mapped: \$v\"; exit 1; }
  done
  [[ ! 'TF_DS_BUILD_FLAGS=x' =~ ^(\$PORT_MAPPED_RE)= ]] || { echo 'TF_DS_BUILD_FLAGS must pass through'; exit 1; }"
case_ "spill off: only the base names are mapped (hand-exported spill variables pass as before)" 0 "SPILL_GIB=0 KV_SHARD=0" "$expect
  port_map; expect \"\$PORT_MAPPED_RE\" \"\$PORT_MAPPED_BASE\""
case_ "refuse: SPILL_STAGING_MIB under the engine's 16" 1 "SPILL_GIB=8 SPILL_STAGING_MIB=8" "port_validate"
case_ "refuse: SPILL_MIN_TOKENS not a count" 1 "SPILL_GIB=8 SPILL_MIN_TOKENS=8k" "port_validate"
case_ "SPILL_MIN_TOKENS off the 2048 grid warns" 0 "SPILL_GIB=8 SPILL_MIN_TOKENS=5000" "port_validate | grep -q 'WARN: SPILL_MIN_TOKENS=5000' || exit 1"
case_ "refuse: SPILL_TURNS not a count" 1 "SPILL_GIB=8 SPILL_TURNS=all" "port_validate"
case_ "refuse: SPILL_WEIGHTS_SHA=yes" 1 "SPILL_GIB=8 SPILL_WEIGHTS_SHA=yes" "port_validate"
case_ "a round-1 placeholder name exported warns and is not used" 0 "SPILL_GIB=0 KV_SHARD=0; export TF_SPILL_GIB=8" "$expect
  port_validate | grep -q 'WARN: TF_SPILL_GIB=8 is ignored' || exit 1
  port_map; expect \"\${#PORT_SERVE[@]} \$(port_identity)\" '0 off'"
case_ "extra serve arguments: --spill-* refused, others accepted" 0 "" "
  ( port_check_extra --reasoning-effort high ) || exit 1
  ( port_check_extra --spill-min-tokens 4096 ) && exit 1
  ( port_check_extra --spill-gib=8 ) && exit 1; exit 0"
case_ "RANK1_EXTRA_ENV: rank 1's own words, with a warning" 0 "SPILL_GIB=0 KV_SHARD=0 RANK1_EXTRA_ENV='TF_DS_SPILL_MIN_TOKENS=4096 TF_DS_KV_SHARD_CARRY=0'" "$expect
  port_validate | grep -q 'WARN: RANK1_EXTRA_ENV' || exit 1
  port_map; expect \"\${PORT_RANK1_ENV[*]}\" 'TF_DS_SPILL_MIN_TOKENS=4096 TF_DS_KV_SHARD_CARRY=0'; expect \"\${#PORT_ENV[@]}\" 0"
case_ "refuse: RANK1_EXTRA_ENV with a shell character" 1 "RANK1_EXTRA_ENV='X=\$(id)'" "port_validate"
case_ "refuse: RANK1_EXTRA_ENV not NAME=value" 1 "RANK1_EXTRA_ENV='lower=1'" "port_validate"

# ---- the spill tier's identity (port_ident_map): from the image's labels, else the pins
ident_stubs='image_hash() { echo 99f490fc5c18; }; snapshot_rev() { echo 6d54be4fca338d79b17c3b03cc9a423bb3ac6718; }
  engram_rev() { echo 1bae0e1088993b9b82de08f59de5be5a34e42170; }; ablit_identity() { echo 0; }
  TF_REF=629c524a0ea57512a5342abb498d1b7c25353343; MODEL_ID=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw
  ENGRAM_ID=deepseek-ai/DeepSeek-V4.1-Flash-1017'
case_ "identity: off with the tier off" 0 "SPILL_GIB=0" "$expect; $ident_stubs; port_ident_map img; expect \"\${#PORT_IDENT_ENV[@]}\" 0"
case_ "identity: no image yet -> the pins (image_hash, TF_REF)" 0 "SPILL_GIB=8" "$expect; $ident_stubs
  port_image_label() { :; }
  port_ident_map img
  expect \"\${PORT_IDENT_ENV[*]}\" 'TF_DS_ENGINE_ID=tensorfold@629c524a0ea5+99f490fc5c18 TF_DS_RECIPE_PATCHES=99f490fc5c18 TF_DS_MODEL_REVISION=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw@6d54be4fca338d79b17c3b03cc9a423bb3ac6718,quant=exl3-2.9bpw,ablit=0 TF_DS_ENGRAM_REVISION=deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170'"
case_ "identity: the image's labels win over the pins (IMAGE= an older tag)" 0 "SPILL_GIB=8" "$expect; $ident_stubs
  port_image_label() { case \$2 in tf.patches) echo 0123456789ab ;; tf.ref) echo ffffffffffff0000 ;; esac; }
  port_ident_map img
  expect \"\${PORT_IDENT_ENV[0]} \${PORT_IDENT_ENV[1]}\" 'TF_DS_ENGINE_ID=tensorfold@ffffffffffff+0123456789ab TF_DS_RECIPE_PATCHES=0123456789ab'"
case_ "identity: exported values win; ABLIT and MODEL_QUANT in the revision" 0 \
  "SPILL_GIB=8 TF_DS_ENGINE_ID=mine TF_DS_ENGRAM_REVISION=eng MODEL_QUANT=exl3-eaq-v1-2.9bpw" "$expect; $ident_stubs
  ablit_identity() { echo 1@0bdc2036; }; port_image_label() { :; }
  port_ident_map img
  expect \"\${PORT_IDENT_ENV[0]}\" TF_DS_ENGINE_ID=mine; expect \"\${PORT_IDENT_ENV[3]}\" TF_DS_ENGRAM_REVISION=eng
  expect \"\${PORT_IDENT_ENV[2]}\" 'TF_DS_MODEL_REVISION=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw@6d54be4fca338d79b17c3b03cc9a423bb3ac6718,quant=exl3-eaq-v1-2.9bpw,ablit=1@0bdc2036'"
case_ "identity: a quant name from other model ids" 0 "" "$expect
  MODEL_ID=x/Model-EXL2-4.0bpw; expect \"\$(port_model_quant)\" exl2-4.0bpw
  MODEL_ID=x/Plain-Model; expect \"\$(port_model_quant)\" plain-model"
case_ "identity: SPILL_WEIGHTS_SHA=0 computes no digest" 0 "SPILL_GIB=8 SPILL_WEIGHTS_SHA=0" "$expect
  rank_cache_sha() { echo called; exit 1; }; port_weights_check; expect \"\${#PORT_RANK_SHA[@]}\" 0"
case_ "identity: a real start refuses when a rank has no digest (it names the store)" 0 "SPILL_GIB=8" "
  STATE_DIR=\$(mktemp -d); DATA_DIR=/x; WORKER_CACHE_DIR=/y; worker() { echo 'ERR 0 weight files rank1of2-*.bin in /x'; }
  rank_cache_sha() { echo sha256:\$(printf 'a%.0s' \$(seq 64)); }
  trap 'rm -rf \"\$STATE_DIR\"' EXIT
  out=\$( (port_weights_check) 2>&1) && { echo \"\$out\"; exit 1; }
  grep -q 'DIE: no weight-file digest (rank 1: 0 weight files' <<<\"\$out\" || { echo \"\$out\"; exit 1; }"
case_ "identity: an unreachable worker refuses too, and so does a rank 0 without a digest" 0 "SPILL_GIB=8" "
  STATE_DIR=\$(mktemp -d); DATA_DIR=/x; WORKER_CACHE_DIR=/y; trap 'rm -rf \"\$STATE_DIR\"' EXIT
  worker() { return 255; }; rank_cache_sha() { echo sha256:\$(printf 'a%.0s' \$(seq 64)); }
  ( port_weights_check ) && exit 1
  ( port_weights_check ) 2>&1 | grep -q 'rank 1: no answer from the worker' || exit 1
  worker() { echo sha256:\$(printf 'b%.0s' \$(seq 64)); }; rank_cache_sha() { echo 'ERR cannot read /x/f'; }
  ( port_weights_check ) 2>&1 | grep -q 'DIE: no weight-file digest (rank 0: cannot read /x/f)' || exit 1
  STATE_DIR=/nonexistent/dir; worker() { return 255; }; rank_cache_sha() { echo sha256:\$(printf 'a%.0s' \$(seq 64)); }
  ( port_weights_check ) && exit 1; exit 0"
case_ "identity: the worker's probe is tried twice (one ssh glitch does not stop the start)" 0 "SPILL_GIB=8" "$expect
  STATE_DIR=\$(mktemp -d); DATA_DIR=/x; WORKER_CACHE_DIR=/y; trap 'rm -rf \"\$STATE_DIR\"' EXIT
  worker() { if [[ -e \"\$STATE_DIR/tried\" ]]; then echo sha256:\$(printf 'b%.0s' \$(seq 64)); else : > \"\$STATE_DIR/tried\"; return 255; fi; }
  rank_cache_sha() { echo sha256:\$(printf 'a%.0s' \$(seq 64)); }
  port_weights_check >/dev/null
  [[ \"\${PORT_RANK_SHA[0]}\" == sha256:aaaa* && \"\${PORT_RANK_SHA[1]}\" == sha256:bbbb* ]] || { echo \"[\${PORT_RANK_SHA[*]}]\"; exit 1; }
  STATE_DIR=/nonexistent/dir
  worker() { echo sha256:\$(printf 'c%.0s' \$(seq 64)); }; port_weights_check >/dev/null
  [[ \"\${PORT_RANK_SHA[1]}\" == sha256:cccc* ]] || { echo 'no mktemp: the probe is not run'; exit 1; }"
case_ "identity: DRY_RUN without a digest only warns" 0 "SPILL_GIB=8 DRY_RUN=1" "$expect
  STATE_DIR=\$(mktemp -d); DATA_DIR=/x; WORKER_CACHE_DIR=/y; trap 'rm -rf \"\$STATE_DIR\"' EXIT
  worker() { echo 'ERR 0 weight files rank1of2-*.bin in /x'; }; rank_cache_sha() { echo PENDING; }
  out=\$(port_weights_check 2>&1)
  grep -q 'WARN: DRY_RUN: rank 1: no weight-file digest (0 weight files' <<<\"\$out\" || { echo \"\$out\"; exit 1; }
  port_weights_check >/dev/null 2>&1; expect \"\${#PORT_RANK_SHA[@]}\" 0"
case_ "refuse: the spill identity with no checkpoint revision" 1 "SPILL_GIB=8" "$ident_stubs; snapshot_rev() { :; }; port_image_label() { :; }; port_ident_map img"
case_ "refuse: the spill identity with no Engram revision" 1 "SPILL_GIB=8" "$ident_stubs; engram_rev() { :; }; port_image_label() { :; }; port_ident_map img"
case_ "identity: an exported TF_DS_MODEL_REVISION stands in for an empty revision" 0 "SPILL_GIB=8 TF_DS_MODEL_REVISION=m@r" "$expect; $ident_stubs
  snapshot_rev() { :; }; port_image_label() { :; }; port_ident_map img; expect \"\${PORT_IDENT_ENV[2]}\" TF_DS_MODEL_REVISION=m@r"
case_ "identity: no revision is fine with the tier off" 0 "SPILL_GIB=0" "$ident_stubs; snapshot_rev() { :; }; engram_rev() { :; }; port_ident_map img"
case_ "POOL alone: left to start.sh (0008: TF_DS_POOL, TF_DS_WINDOW, rank 1's --pool), nothing of the port's" 0 "SPILL_GIB=0 KV_SHARD=0 POOL=2097152" "$expect
  port_validate; port_map
  expect \"\${#PORT_ENV[@]} \${#PORT_SERVE[@]} \$POOL\" '0 0 2097152'"
case_ "KV shard with the pool it needs" 0 "SPILL_GIB=0 KV_SHARD=1 POOL=2101248 KV_SHARD_DECODE=owner" "$expect
  port_validate; port_map
  expect \"\${PORT_ENV[*]}\" 'TF_DS_KV_SHARD=1 TF_DS_KV_SHARD_DECODE=owner TF_DS_KV_SHARD_CARRY=1 TF_DS_KV_SHARD_TWINS=1 TF_DS_KV_SHARD_CHECK=1024'
  expect \"\$POOL\" 2101248
  expect \"\$(port_shard_pool)\" 2101248
  expect \"\$(port_identity)\" 'pool=2101248,shard=1/owner/c1t1k1024,spill=0'"
case_ "KV shard: carry, twins and the check interval to both ranks" 0 "SPILL_GIB=0 KV_SHARD=1 POOL=2101248 KV_SHARD_CARRY=0 KV_SHARD_TWINS=0 KV_SHARD_CHECK=1" "$expect
  port_validate; port_map
  expect \"\${PORT_ENV[*]}\" 'TF_DS_KV_SHARD=1 TF_DS_KV_SHARD_CARRY=0 TF_DS_KV_SHARD_TWINS=0 TF_DS_KV_SHARD_CHECK=1'
  expect \"\$(port_identity)\" 'pool=2101248,shard=1/fetch/c0t0k1,spill=0'
  [[ 'TF_DS_KV_SHARD_TWINS=1' =~ ^(\$PORT_MAPPED_RE)= ]] || exit 1"
case_ "KV shard knobs exported alone are taken" 0 "export TF_DS_KV_SHARD=1 TF_DS_KV_SHARD_TWINS=0" "$expect
  port_validate; port_map; expect \"\$KV_SHARD_TWINS \$POOL\" '0 2101248'"
case_ "TF_DS_POOL / TF_DS_WINDOW exported by hand: ignored with a warning, POOL is the one setting" 0 "KV_SHARD=1; export TF_DS_POOL=999999 TF_DS_WINDOW=1" "$expect
  out=\$(port_validate); grep -q 'WARN: TF_DS_POOL=999999 is ignored' <<<\"\$out\" && grep -q 'WARN: TF_DS_WINDOW=1 is ignored' <<<\"\$out\" || { echo \"\$out\"; exit 1; }
  port_validate >/dev/null; port_map; expect \"\$POOL\" 2101248
  for v in TF_DS_POOL TF_DS_WINDOW; do [[ \"\$v=x\" =~ ^(\$PORT_MAPPED_RE)= ]] || { echo \"not mapped: \$v\"; exit 1; }; done"
case_ "KV shard knobs with KV_SHARD=0 warn and add nothing" 0 "SPILL_GIB=0 KV_SHARD=0 KV_SHARD_CARRY=0" "$expect
  port_validate | grep -q 'WARN: KV_SHARD_CARRY' || exit 1; port_map; expect \"\${#PORT_ENV[@]}\" 0"
case_ "refuse: KV_SHARD_TWINS=2" 1 "KV_SHARD=1 POOL=2101248 KV_SHARD_TWINS=2" "port_validate"
case_ "refuse: KV_SHARD_CHECK=-1" 1 "KV_SHARD=1 POOL=2101248 KV_SHARD_CHECK=-1" "port_validate"
case_ "engine variable exported alone is taken" 0 "export TF_DS_KV_SHARD=1" "$expect
  port_validate; port_map; expect \"\$KV_SHARD \$PORT_SHARD_EXPLICIT \$POOL\" '1 1 2101248'
  env | grep -E \"^(\$PORT_MAPPED_RE)=\" | sort | tr '\n' ' ' | grep -q 'TF_DS_KV_SHARD=1' || exit 1"
case_ "a smaller window at 262144: shard pool 528384" 0 "CONTEXT=262144 KV_SHARD=1 POOL=528384" "$expect; port_validate; expect \"\$(port_shard_pool)\" 528384"
case_ "a 65536 window: shard pool 135168, accepted" 0 "CONTEXT=65536 KV_SHARD=1 POOL=135168" "$expect; port_validate; expect \"\$(port_shard_pool)\" 135168"
case_ "refuse: a 65536 window, POOL 131072 cuts it" 1 "CONTEXT=65536 KV_SHARD=1 POOL=131072" "port_validate"
case_ "POOL over 4x CONTEXT warns" 0 "KV_SHARD=0 POOL=4198400" "port_validate | grep -q 'WARN: POOL (4198400) is more than 4x' || exit 1"
case_ "refuse: POOL over 16x CONTEXT" 1 "POOL=16781312" "port_validate"
case_ "POOL at 16x CONTEXT is accepted" 0 "POOL=16777216" "port_validate"
case_ "TF_DS_SPILL_GIB exported alone turns the tier on" 0 "export TF_DS_SPILL_GIB=8 TF_DS_SPILL_MIN_FREE_GIB=20" "$expect
  port_validate; port_map
  expect \"\${PORT_SERVE[*]}\" '--spill-dir /spill --spill-gib 8 --spill-min-free-gib 20 --spill-min-tokens 8192'"
case_ "TF_DS_SPILL_QUOTA exported alone is a share of the cap" 0 "export TF_DS_SPILL_GIB=64 TF_DS_SPILL_QUOTA=0.25" "$expect
  port_validate; port_map; expect \"\$SPILL_QUOTA_GIB\" 16
  expect \"\${PORT_SERVE[*]}\" '--spill-dir /spill --spill-gib 64 --spill-min-free-gib 50 --spill-min-tokens 8192 --spill-quota 0.25'"
case_ "refuse: TF_DS_SPILL_QUOTA above 1" 1 "export TF_DS_SPILL_GIB=64 TF_DS_SPILL_QUOTA=2" "port_validate"
case_ "TF_DS_SPILL_DIR exported alone warns" 0 "export TF_DS_SPILL_GIB=8 TF_DS_SPILL_DIR=/data/spill" "port_validate | grep -q 'WARN: TF_DS_SPILL_DIR=/data/spill is ignored' || exit 1"
case_ "STOP_TIMEOUT follows TF_SPILL_FLUSH_MS" 0 "unset STOP_TIMEOUT; SPILL_GIB=8 TF_SPILL_FLUSH_MS=1500" "$expect; expect \"\$STOP_TIMEOUT\" 32"
case_ "STOP_TIMEOUT for TF_SPILL_FLUSH_MS=0" 0 "unset STOP_TIMEOUT; SPILL_GIB=8 TF_SPILL_FLUSH_MS=0" "$expect; expect \"\$STOP_TIMEOUT\" 30"
case_ "refuse: TF_SPILL_FLUSH_MS not a number" 1 "SPILL_GIB=8 TF_SPILL_FLUSH_MS=60s" "port_validate"
case_ "the identity changes with the spill folders" 0 "SPILL_GIB=8 SPILL_DIR=/nvme/a" "$expect
  a=\$(port_identity); WORKER_SPILL_DIR=/nvme/b; b=\$(port_identity); [[ \"\$a\" != \"\$b\" ]] || exit 1"

# ---- refusals
case_ "KV shard on the 1M window without POOL: the pool it needs, derived" 0 "SPILL_GIB=0 KV_SHARD=1" "$expect
  port_validate; port_map; expect \"\$POOL \${PORT_ENV[0]}\" '2101248 TF_DS_KV_SHARD=1'"
case_ "refuse: KV shard, POOL 2097152 cuts the 1M window" 1 "KV_SHARD=1 POOL=2097152" "port_validate"
case_ "refuse: KV shard, POOL not whole 4096 parts" 1 "KV_SHARD=1 POOL=2103296" "port_validate"
case_ "KV shard and spill together (round 3)" 0 "KV_SHARD=1 POOL=2101248 SPILL_GIB=8" "port_validate"
case_ "refuse: POOL below CONTEXT" 1 "POOL=524288" "port_validate"
case_ "refuse: POOL not a number" 1 "POOL=1M" "port_validate"
case_ "refuse: KV_SHARD=2" 1 "KV_SHARD=2" "port_validate"
case_ "refuse: KV_SHARD_DECODE=both" 1 "KV_SHARD=1 POOL=2101248 KV_SHARD_DECODE=both" "port_validate"
case_ "refuse: SPILL_GIB=-1" 1 "SPILL_GIB=-1" "port_validate"
case_ "refuse: quota over the cap" 1 "SPILL_GIB=8 SPILL_QUOTA_GIB=9" "port_validate"
case_ "refuse: relative SPILL_DIR" 1 "SPILL_GIB=8 SPILL_DIR=spill" "port_validate"
case_ "refuse: SPILL_DIR with a space" 1 "SPILL_GIB=8 SPILL_DIR='/nvme/my spill'" "port_validate"
case_ "accept: WORKER_SPILL_DIR under \$HOME" 0 "SPILL_GIB=8 WORKER_SPILL_DIR='\$HOME/spill'" "port_validate"
case_ "a bad setting is not checked while spill is off" 0 "SPILL_GIB=0 SPILL_DIR=relative" "port_validate"

# ---- port_spill_judge on probe lines: <st> <uid> <gid> <me> <fs> <source> <avail> <exported> <overlap> <home/root> <path>
G=1073741824
ok_line="OK 1000 1000 1000 ext4 /dev/nvme0n1p2 $((500 * G)) 0 - 0 /home/u/.cache/dsv41-tensorfold-spill"
case_ "judge: local, yours, room" 0 "SPILL_GIB=64 SPILL_FLOOR_GIB=50" "$expect
  port_spill_judge 0 here '$ok_line'; expect \"\${PORT_SPILL_REAL[0]} \${PORT_SPILL_UID[0]}\" '/home/u/.cache/dsv41-tensorfold-spill 1000'"
j() { local name=$1 want=$2 line=$3; case_ "judge: $name" "$want" "SPILL_GIB=64 SPILL_FLOOR_GIB=50" "port_spill_judge 0 here '$line'"; }
j "refuse nfs" 1 "OK 1000 1000 1000 nfs4 srv:/x $((500 * G)) 0 - 0 /mnt/x/spill"
j "refuse tmpfs" 1 "OK 1000 1000 1000 tmpfs tmpfs $((500 * G)) 0 - 0 /run/spill"
j "refuse overlay" 1 "OK 1000 1000 1000 overlay overlay $((500 * G)) 0 - 0 /var/lib/docker/overlay2/x/merged/spill"
j "refuse squashfs" 1 "OK 1000 1000 1000 squashfs /dev/loop3 $((500 * G)) 0 - 0 /snap/x/spill"
j "refuse a zram device" 1 "OK 1000 1000 1000 ext4 /dev/zram1 $((500 * G)) 0 - 0 /mnt/zram/spill"
j "refuse sshfs" 1 "OK 1000 1000 1000 fuse.sshfs srv:/x $((500 * G)) 0 - 0 /mnt/x/spill"
j "refuse exported" 1 "OK 1000 1000 1000 ext4 /dev/nvme0n1p2 $((500 * G)) 1 - 0 /srv/export/spill"
j "refuse overlap with the cache" 1 "OK 1000 1000 1000 ext4 /dev/nvme0n1p2 $((500 * G)) 0 /home/u/.cache/deepseek-1017 0 /home/u/.cache/deepseek-1017/spill"
j "refuse home itself" 1 "OK 1000 1000 1000 ext4 /dev/nvme0n1p2 $((500 * G)) 0 - 1 /home/u"
j "refuse an archive folder" 1 "OK 1000 1000 1000 ext4 /dev/nvme0n1p2 $((500 * G)) 0 - 0 /mnt/models/Archive/spill"
j "refuse root-owned" 1 "OK 0 0 1000 ext4 /dev/nvme0n1p2 $((500 * G)) 0 - 0 /nvme/spill"
j "refuse under the floor" 1 "OK 1000 1000 1000 ext4 /dev/nvme0n1p2 $((49 * G)) 0 - 0 /nvme/spill"
j "warn under cap+floor, accept" 0 "OK 1000 1000 1000 ext4 /dev/nvme0n1p2 $((60 * G)) 0 - 0 /nvme/spill"
j "refuse a probe error" 1 "ERR cannot create /nvme/spill"
j "refuse a path with odd characters" 1 "OK 1000 1000 1000 ext4 /dev/nvme0n1p2 $((500 * G)) 0 - 0 /nvme/sp:ill"
j "accept a new folder under a root-owned parent (DRY_RUN)" 0 "NEW 0 0 1000 ext4 /dev/nvme0n1p2 $((500 * G)) 0 - 0 /nvme/spill"
j "accept another owner with a warning" 0 "OK 1001 1001 1000 xfs /dev/nvme0n1p2 $((500 * G)) 0 - 0 /nvme/spill"
# the default tier (SPILL_GIB not set): the same failures turn it off for the start instead of refusing it
case_ "judge, default spill: a failure returns 1 with the reason, never dies" 0 "" "$expect
  port_spill_judge 0 here 'OK 1000 1000 1000 nfs4 srv:/x $((500 * G)) 0 - 0 /mnt/x/spill' && exit 1
  [[ \"\$PORT_SPILL_WHY\" == *'network file system (nfs4)'* ]] || { echo \"[\$PORT_SPILL_WHY]\"; exit 1; }
  port_spill_judge 0 here 'OK 1000 1000 1000 ext4 /dev/nvme0n1p2 $((49 * G)) 0 - 0 /nvme/spill' && exit 1
  [[ \"\$PORT_SPILL_WHY\" == *'under SPILL_FLOOR_GIB'* ]] || exit 1
  port_spill_judge 0 here 'OK 0 0 1000 ext4 /dev/nvme0n1p2 $((500 * G)) 0 - 0 /nvme/spill' && exit 1
  port_spill_judge 0 here 'ERR cannot create /nvme/spill' && exit 1
  port_spill_judge 0 here '$ok_line'"
# port_spill_check with stubbed probes: <rank 0's line> <rank 1's line, or "unreachable">
pc_stubs() { printf "HEAD_ENGRAM_DIR=; DATA_DIR=/x; WORKER_ENGRAM_DIR=/y; WORKER_CACHE_DIR=/z; worker_host() { echo w; }
  spill_dir_probe() { echo '%s'; }; worker() { [[ '%s' != unreachable ]] || return 255; echo '%s'; }" "$1" "$2" "$2"; }
nfs_line="OK 1000 1000 1000 nfs4 srv:/x $((500 * G)) 0 - 0 /mnt/x/spill"
case_ "check, a fork's default spill: both Sparks usable -> on, both folders named" 0 "PT_FORK=1; unset STOP_TIMEOUT" "$expect; $(pc_stubs "$ok_line" "$ok_line")
  port_spill_check >/dev/null; expect \"\$SPILL_GIB \${PORT_SPILL_REAL[1]}\" '64 /home/u/.cache/dsv41-tensorfold-spill'; expect \"\$STOP_TIMEOUT\" 90"
case_ "check, a fork's default spill: the worker's folder on NFS -> off on both ranks, loudly" 0 "PT_FORK=1; unset STOP_TIMEOUT" "$expect; $(pc_stubs "$ok_line" "$nfs_line")
  out=\$(port_spill_check 2>&1); grep -q 'WARN: SPILL TIER OFF for this start: WORKER_SPILL_DIR /mnt/x/spill on w is on a network file system' <<<\"\$out\" || { echo \"\$out\"; exit 1; }
  port_spill_check >/dev/null 2>&1; port_map
  expect \"\$SPILL_GIB \${#PORT_SERVE[@]} \$(port_spill_args 0)\${STOP_TIMEOUT:-unset}\" '0 0 unset'
  expect \"\$(port_identity)\" 'pool=2101248,shard=1/fetch/c1t1k1024,spill=0'"
case_ "check, a fork's default spill: this Spark under the floor -> off" 0 "PT_FORK=1" "$expect; $(pc_stubs "OK 1000 1000 1000 ext4 /dev/nvme0n1p2 $((10 * G)) 0 - 0 /nvme/spill" "$ok_line")
  port_spill_check >/dev/null 2>&1; expect \"\$SPILL_GIB\" 0"
case_ "check, a fork's default spill: the worker's probe fails -> off" 0 "PT_FORK=1" "$expect; $(pc_stubs "$ok_line" unreachable)
  port_spill_check 2>&1 | grep -q 'SPILL TIER OFF for this start: could not check WORKER_SPILL_DIR' || exit 1"
case_ "refuse: check, SPILL_GIB=64 set: the worker's folder on NFS" 1 "SPILL_GIB=64" "$(pc_stubs "$ok_line" "$nfs_line"); port_spill_check"
case_ "refuse: check, SPILL_GIB=64 set: the worker's probe fails" 1 "SPILL_GIB=64" "$(pc_stubs "$ok_line" unreachable); port_spill_check"
case_ "spill args name the checked path" 0 "SPILL_GIB=8" "$expect
  port_spill_judge 1 there '$ok_line'
  expect \"\$(port_spill_args 1)\" '--label tf.recipe.spill=/home/u/.cache/dsv41-tensorfold-spill -v /home/u/.cache/dsv41-tensorfold-spill:/spill'"

# ---- port_engram_check: with the spill tier on, both ranks must list the same regular files of their Engram folder,
# because the engine's rank fingerprint compares exactly that (name and size; symlinks are skipped, so a HF snapshot
# lists nothing against the worker's copied files) — issue #61. Checked before anything is stopped.
et=$(mktemp -d "${TMPDIR:-/tmp}/port-engram.XXXXXX")
mkdir -p "$et/real" "$et/snap"
printf 'abcde' > "$et/real/config.json"
printf 'abcdefg' > "$et/real/model.safetensors.index.json"
ln -s "$et/real/config.json" "$et/snap/config.json"
ln -s "$et/real/model.safetensors.index.json" "$et/snap/model.safetensors.index.json"
et_list='config.json:5; model.safetensors.index.json:7; '
et_stubs="engram_rev() { echo 1bae0e1088993b9b82de08f59de5be5a34e42170; }; worker_host() { echo w; }; ENGRAM_ID=deepseek-ai/DeepSeek-V4.1-Flash-1017; ENGRAM_SHARDS='model-00047-of-00048.safetensors model-00048-of-00048.safetensors'"
case_ "engram listing: name:size of the regular files, sorted; symlinks skipped" 0 "" "$expect
  expect \"\$(engram_listing '$et/real' | tr '\n' ' ')\" '$et_list'
  expect \"[\$(engram_listing '$et/snap' | tr '\n' ' ')]\" '[]'"
case_ "engram dir: HEAD_ENGRAM_DIR when set" 0 "HEAD_ENGRAM_DIR=$et/real" "$expect
  expect \"\$(head_engram_dir)\" '$et/real'"
case_ "engram dir: no HEAD_ENGRAM_DIR -> the HF snapshot path on this Spark" 0 "" "$expect
  HF_CACHE=/hf/cache; engram_in_container() { echo /hf/hub/models--x--y/snapshots/abc; }
  expect \"\$(head_engram_dir)\" '/hf/cache/hub/models--x--y/snapshots/abc'"
case_ "engram check: the spill tier off compares nothing" 0 "SPILL_GIB=0 KV_SHARD=0" "$expect
  worker() { echo CALLED; }
  expect \"[\$(port_engram_check 2>&1)]\" '[]'"
case_ "engram check: KV sharding alone compares nothing (its agreement marks engram shard-only)" 0 "SPILL_GIB=0 KV_SHARD=1" "$expect
  worker() { echo CALLED; }
  expect \"[\$(port_engram_check 2>&1)]\" '[]'"
case_ "engram check: spill on, both folders list the same files -> ok" 0 "SPILL_GIB=64 HEAD_ENGRAM_DIR=$et/real WORKER_ENGRAM_DIR=/y" "$expect; $et_stubs
  worker() { printf 'config.json:5;\nmodel.safetensors.index.json:7;\n'; }
  expect \"[\$(port_engram_check 2>&1)]\" '[]'"
case_ "engram check: spill on, rank 0's folder is a HF snapshot (symlinks only) -> refused before anything stops" 0 "SPILL_GIB=64 HEAD_ENGRAM_DIR=$et/snap WORKER_ENGRAM_DIR=/y" "$expect; $et_stubs
  worker() { printf 'config.json:5;\nmodel.safetensors.index.json:7;\n'; }
  out=\$(port_engram_check 2>&1); rc=\$?
  expect \"\$rc\" 1
  grep -q 'list different files' <<<\"\$out\" || { echo \"\$out\"; exit 1; }
  grep -q 'HEAD_ENGRAM_DIR' <<<\"\$out\" || { echo \"\$out\"; exit 1; }
  grep -q 'issue #61' <<<\"\$out\" || { echo \"\$out\"; exit 1; }"
case_ "engram check: rank 0's folder not there yet (prepare has not run) -> silent" 0 "SPILL_GIB=64 HEAD_ENGRAM_DIR=$et/none WORKER_ENGRAM_DIR=/y" "$expect
  worker() { echo CALLED; }
  expect \"[\$(port_engram_check 2>&1)]\" '[]'"
case_ "engram check: the worker's folder not there yet -> silent" 0 "SPILL_GIB=64 HEAD_ENGRAM_DIR=$et/real WORKER_ENGRAM_DIR=/y" "$expect
  worker() { echo __NO_DIR__; }
  expect \"[\$(port_engram_check 2>&1)]\" '[]'"
case_ "engram check: the worker cannot be reached -> a warning, not a refusal" 0 "SPILL_GIB=64 HEAD_ENGRAM_DIR=$et/real WORKER_ENGRAM_DIR=/y" "$expect; $et_stubs
  worker() { return 255; }
  expect \"[\$(port_engram_check 2>&1)]\" \"[WARN: could not compare /y on w with rank 0's Engram folder ($et/real)]\""
rm -rf "$et"

# ---- spill_dir_probe itself (GNU coreutils and util-linux: Linux only)
if [[ "$(uname -s)" == Linux ]]; then
  tmp=$(mktemp -d)
  case_ "probe: creates, describes, finds an overlap" 0 "" "$expect
    l=\$(spill_dir_probe 1 '$tmp/a/spill' '$tmp/b'); set -- \$l; expect \"\$1 \$4 \${11}\" \"OK \$(id -u) \$(readlink -m '$tmp/a/spill')\"
    [[ -d '$tmp/a/spill' ]] || exit 1
    l=\$(spill_dir_probe 0 '$tmp/c/spill'); set -- \$l; expect \$1 NEW; [[ ! -e '$tmp/c' ]] || exit 1
    l=\$(spill_dir_probe 0 '$tmp/a/spill' '$tmp/a'); set -- \$l; expect \$9 \"\$(readlink -m '$tmp/a')\"
    l=\$(spill_dir_probe 0 '$tmp/a' '$tmp/a/spill/x'); set -- \$l; expect \$9 \"\$(readlink -m '$tmp/a/spill/x')\""
  rm -rf "$tmp"
else
  echo "skip spill_dir_probe (needs Linux: readlink -m, findmnt, df -B1, stat -c)"
fi

# ---- scripts/memavail.sh: MIN_AVAIL_GIB counts the pages parked on the per-CPU free lists, as the engine's 0119 does
# (native/cuda_memory.zig parkedBytes): its Zig tests' synthetic /proc/zoneinfo cases, translated
# zone <node> <name> <high_min or -> [counts...]: one zone's block as the kernel prints it (no counts: no pagesets)
zone() {
  local node=$1 name=$2 hm=$3 c n=0; shift 3
  printf 'Node %s, zone %8s\n  pages free     10\n        managed  1000\n      nr_free_cma  0\n' "$node" "$name"
  if (( $# )); then
    printf '  pagesets\n'
    for c in "$@"; do
      printf '    cpu: %s\n              count:    %s\n              high:     5000\n              batch:    63\n' "$n" "$c"
      [[ "$hm" == - ]] || printf '              high_min: %s\n' "$hm"
      printf '              high_max: 198283\n  vm stats threshold: 110\n'
      n=$((n + 1))
    done
  fi
  printf '  start_pfn:           4096\n'
}
mt=$(mktemp -d "${TMPDIR:-/tmp}/port-mem.XXXXXX")
printf 'MemTotal:       %s kB\nMemFree:        1000 kB\nMemAvailable:   %s kB\n' $((128 * 1048576)) $((100 * 1048576)) > "$mt/meminfo"
printf 'MemTotal:       %s kB\nMemAvailable:   %s kB\n' $((128 * 1048576)) $((125 * 1048576)) > "$mt/meminfo-high"
printf 'MemTotal:       1 kB\n' > "$mt/meminfo-none"
{ zone 0 DMA 2146 200000 3143 2332 17; zone 0 DMA32 2146; zone 0 Normal 2146 100000 100000 100000; zone 1 Normal 2146 400000; zone 1 Movable 2146; } > "$mt/z-many"
six=$(( (6 << 30) / 4096 / 4 + 2146 ))
zone 0 Normal 2146 $six $six $six $six > "$mt/z-six"
zone 0 Normal 2146 $(for i in $(seq 20); do printf '%s ' $((2146 + 1440)); done) > "$mt/z-warm"
zone 0 Normal 2146 $(for i in $(seq 20); do printf '100 '; done) > "$mt/z-below"
half=$(( ((1 << 30) + (1 << 29)) / 4096 / 2 ))
zone 0 Normal - $half $half > "$mt/z-old"
zone 0 Normal - 1000 1000 1000 1000 > "$mt/z-old-small"
{ printf '  count: 5\n      count: 6\n'; zone 0 Normal 2146 $((2146 + 10))
  printf '      count:    99999\n      high_min: 7\n'
  printf 'Node 0, zone Other\n      count:    77777\n  pagesets\n    cpu: 0\n              high:     1\n'; } > "$mt/z-only"
{ zone 0 Normal 2146 500000; printf '  pagesets\n    cpu: 0\n              count:    abc\n'; } > "$mt/z-bad"
printf '  pagesets\n    cpu: 0\n              count:    600000\r\n              high_min: 2146\r\n' > "$mt/z-crlf"
# mc <name> <meminfo> <zoneinfo> <want: avail GiB, parked GiB, parked bytes>
mc() { case_ "parked pages: $1" 0 "source '$here/memavail.sh'" "$expect; expect \"\$(mem_counting '$mt/$2' '$mt/$3' 4096)\" '$4'"; }
many=$(( (200000 + 3143 + 2332 + 17 + 300000 + 400000 - 8 * 2146) * 4096 ))
mc "every CPU, zone and node, less each list's resting level" meminfo z-many "$(awk -v b=$many 'BEGIN {printf "%.1f %.1f %.0f", 100 + b / 2^30, b / 2^30, b}')"
mc "an image copy's 6 GiB raise the budget by 6 GiB" meminfo z-six "106.0 6.0 6442450944"
mc "a warm start counts ~nothing (under 0.25 GiB)" meminfo z-warm "$(awk -v b=$((20 * 1440 * 4096)) 'BEGIN {printf "%.1f %.1f %.0f", 100 + b / 2^30, b / 2^30, b}')"
mc "below the resting level: nothing" meminfo z-below "100.0 0.0 0"
mc "no high_min (before 6.7) rests the first 512 MiB" meminfo z-old "101.0 1.0 1073741824"
mc "no high_min, under 512 MiB: nothing" meminfo z-old-small "100.0 0.0 0"
mc "only the count: and high_min: lines under pagesets" meminfo z-only "100.0 0.0 40960"
mc "an unparsable count counts nothing" meminfo z-bad "100.0 0.0 0"
mc "carriage returns are blanks" meminfo z-crlf "$(awk -v b=$(( (600000 - 2146) * 4096 )) 'BEGIN {printf "%.1f %.1f %.0f", 100 + b / 2^30, b / 2^30, b}')"
mc "capped at MemTotal" meminfo-high z-six "128.0 6.0 6442450944"
mc "no zoneinfo: MemAvailable alone" meminfo missing "100.0 0.0 0"
case_ "parked pages: no MemAvailable fails and prints nothing" 0 "source '$here/memavail.sh'" "
  out=\$(mem_counting '$mt/meminfo-none' '$mt/z-many' 4096) && exit 1; [[ -z \"\$out\" ]]"
case_ "parked pages: the page size from getconf by default" 0 "source '$here/memavail.sh'" "$expect
  expect \"\$(mem_counting '$mt/meminfo' '$mt/z-six')\" \"\$(mem_counting '$mt/meminfo' '$mt/z-six' \$(getconf PAGESIZE))\""
if [[ -r /proc/meminfo ]]; then
  case_ "parked pages: this host's /proc/meminfo and /proc/zoneinfo" 0 "source '$here/memavail.sh'" "
    read -r a p b <<<\"\$(mem_counting)\"; [[ \"\$a\" =~ ^[0-9]+\\.[0-9]\$ && \"\$b\" =~ ^[0-9]+\$ ]] || { echo \"[\$a \$p \$b]\"; exit 1; }"
fi

# ---- rank_cache_sha (scripts/port.sh): each rank's weight file's digest, cached beside it and recomputed when the
# file changes (GNU stat is stubbed from a fixed table so that this runs on a Mac too)
rc=$mt/rc
mkdir -p "$rc"
printf 'weights-one' > "$rc/rank0of2-abc.bin"
want_sha=$(printf 'weights-one' | (sha256sum 2>/dev/null || shasum -a 256) | cut -c1-64)
rstub="stat() { local f=\${@: -1}; [[ -e \"\$f\" ]] || return 1; echo \"\$(wc -c < \"\$f\" | tr -d ' ') \$(cat '$rc/.mtime' 2>/dev/null || echo 1) 7\"; }
  command -v sha256sum >/dev/null || sha256sum() { if [[ \$# -gt 0 && \"\$1\" != - ]]; then shasum -a 256 \"\$1\"; else shasum -a 256; fi; }"
case_ "weights digest: computed, then cached" 0 "$rstub" "$expect
  expect \"\$(rank_cache_sha 1 '$rc' 0)\" sha256:$want_sha
  [[ -f '$rc/.rank0of2-abc.bin.sha256' ]] || { echo 'no cache file'; exit 1; }
  expect \"\$(rank_cache_sha 0 '$rc' 0)\" sha256:$want_sha
  sha256sum() { echo 'should not hash again'; exit 1; }
  expect \"\$(rank_cache_sha 1 '$rc' 0)\" sha256:$want_sha"
case_ "weights digest: a changed file (mtime) is hashed again; DRY_RUN computes nothing" 0 "$rstub" "$expect
  echo 2 > '$rc/.mtime'
  expect \"\$(rank_cache_sha 0 '$rc' 0)\" PENDING
  expect \"\$(rank_cache_sha 1 '$rc' 0)\" sha256:$want_sha
  read -r a b c d < '$rc/.rank0of2-abc.bin.sha256'; expect \"\$b \$d\" \"2 $want_sha\""
case_ "weights digest: none or two files, or a link, is an error" 0 "$rstub" "$expect
  expect \"\$(rank_cache_sha 1 '$rc' 1)\" \"ERR 0 weight files rank1of2-*.bin in $rc\"
  printf y > '$rc/rank0of2-def.bin'
  expect \"\$(rank_cache_sha 1 '$rc' 0)\" \"ERR 2 weight files rank0of2-*.bin in $rc\"
  rm -f '$rc/rank0of2-def.bin'; ln -s '$rc/rank0of2-abc.bin' '$rc/rank1of2-abc.bin'
  expect \"\$(rank_cache_sha 1 '$rc' 1)\" \"ERR 0 weight files rank1of2-*.bin in $rc\""

# ---- scripts/port-patches.sh's README rewrite (newreadme): only marked TABLE rows are its own; prose that names the
# marker stays, and a second run over its own output changes nothing
# check_ <name> <script>: the script in a subshell; exit 0 passes
check_() {
  local name=$1 out got
  n=$((n + 1))
  out=$( ( eval "$2" ) 2>&1 ); got=$?
  if [[ "$got" == 0 ]]; then printf 'ok   %s\n' "$name"
  else fails=$((fails + 1)); printf 'FAIL %s (exit %s)\n%s\n' "$name" "$got" "$(sed 's/^/     /' <<<"$out")"; fi
}
# spill_fix_owner / spill_fix_target (maintainer review of #30: the root container that hands spill files back is narrowed
# to a marked, canonical, user-owned folder); docker is a stub that records its arguments
check_ "spill ownership: only a canonical, marked folder of your own is handed back, by a minimal container" '
  PT_ROOT=$(cd "$here/.." && pwd)
  die() { echo "DIE: $*"; exit 1; }; warn() { :; }; log() { :; }
  TP=2 CONTEXT=1048576 PARALLEL=8 ROUND_ROWS=16 HF_CACHE=/x/hf NFS_PATH=/x/hf POOL=
  source "$here/port.sh"
  t=$(mktemp -d "${TMPDIR:-/tmp}/port-test.XXXXXX"); t=$(cd "$t" && pwd -P); trap "rm -rf -- \"$t\"" EXIT
  docker() { printf "%s\n" "$*" >> "$t/docker"; }
  mkdir -p "$t/spill" "$t/real"; ln -s "$t/real" "$t/link"; ln -s "$t/spill" "$t/viaparent"; ln -s "$t" "$t/lp"
  # not marked yet: refused, nothing run
  spill_fix_owner "$t/spill" img 2>"$t/err" && { echo "an unmarked folder was handed back"; exit 1; }
  grep -q "has no .dsv41-tensorfold-spill" "$t/err" || { cat "$t/err"; exit 1; }
  [[ ! -e "$t/docker" ]] || { echo "docker ran for an unmarked folder"; exit 1; }
  # start.sh marks it (mark=1), then the container runs on exactly that folder
  spill_fix_owner "$t/spill" img 1 || { echo "a marked folder was refused"; exit 1; }
  [[ -f "$t/spill/.dsv41-tensorfold-spill" ]] || { echo "no marker"; exit 1; }
  c=$(cat "$t/docker")
  for w in "--network none" "--read-only" "--security-opt no-new-privileges" "--cap-drop ALL" "--cap-add CHOWN" \
           "--cap-add DAC_READ_SEARCH" "--user 0:0" "-v $t/spill:/spill" "/spill -xdev -mindepth 1 -uid 0 ( -type f -o -type d ) -exec chown --no-dereference $(id -u):$(id -g) {} +"; do
    [[ "$c" == *"$w"* ]] || { echo "missing: $w in [$c]"; exit 1; }
  done
  [[ "$(grep -o -- " -v " <<<" $c" | wc -l | tr -d " ")" == 1 ]] || { echo "more than one mount: $c"; exit 1; }
  : > "$t/docker"
  spill_fix_owner "$t/spill" img || { echo "a marked folder was refused at a stop"; exit 1; }
  # refused: a symlink, a path through a symlink, a relative path, a file, a symlinked marker, another owner
  : > "$t/real/.dsv41-tensorfold-spill"; : > "$t/docker"
  for bad in "$t/link" "$t/viaparent" "$t/lp/spill" "$t/spill/.dsv41-tensorfold-spill"; do
    spill_fix_owner "$bad" img 1 2>/dev/null && { echo "accepted: $bad"; exit 1; }
  done
  ( cd "$t" && spill_fix_owner spill img 1 2>/dev/null ) && { echo "accepted a relative path"; exit 1; }
  mkdir -p "$t/m"; ln -s "$t/spill/.dsv41-tensorfold-spill" "$t/m/.dsv41-tensorfold-spill"
  spill_fix_target "$t/m" >/dev/null && { echo "a symlinked marker was accepted"; exit 1; }
  ( id() { [[ "${1:-}" == -u ]] && echo 4242 || command id "$@"; }; spill_fix_target "$t/spill" ) | grep -q "belongs to uid" || { echo "another owner was accepted"; exit 1; }
  [[ ! -s "$t/docker" ]] || { echo "docker ran for a refused folder: $(cat "$t/docker")"; exit 1; }
  # nothing there yet: nothing to do, no container
  spill_fix_owner "$t/none" img || exit 1; [[ ! -s "$t/docker" ]] || exit 1
  # the worker gets every function it calls (stop.sh and start.sh send them over ssh)
  grep -q "declare -f spill_fix_target spill_fix_owner spill_fix_container" "$PT_ROOT/stop.sh" || { echo "stop.sh does not send spill_fix_target"; exit 1; }
  grep -q "declare -f spill_fix_target spill_fix_owner)" "$here/port.sh" || { echo "port_spill_fix_both does not send spill_fix_target"; exit 1; }'
# start.sh: the POOL port_map derives reaches both ranks through 0008's one mapping (TF_DS_POOL, TF_DS_WINDOW, --pool)
check_ "POOL: the derived sharding pool goes through start.sh's 0008 mapping, once" '
  src=$(cat "$here/../start.sh")
  eng=${src#*"ENGINE_ENV=("}; eng="ENGINE_ENV=(${eng%%"envs() {"*}"
  fol=$(grep "^FOLLOW=\"" "$here/../start.sh")
  die() { echo "DIE: $*"; exit 1; }; warn() { :; }; log() { :; }
  TP=2 CONTEXT=1048576 PARALLEL=4 ROUND_ROWS=16 HF_CACHE=/x/hf NFS_PATH=/x/hf POOL= KV_SHARD=1 SPILL_GIB=0
  REPLAY_FLOOR=0 HF_HUB_OFFLINE=1 PREFILL_INTERLEAVE_MS=0 RDMA_IDLE_S=2 RDMA_NAP_US=500 DRAFTS=1 GRAPHS=1 GRAPH_CAP=128 SIDE=0 PREFETCH=0 ENGRAM_AIO=0
  source "$here/port.sh"; port_validate >/dev/null; port_map
  eval "$eng"; eval "$fol"
  all=$(printf "%s\n" "${ENGINE_ENV[@]}")
  [[ "$(grep -c "^TF_DS_POOL=" <<<"$all")" == 1 && "$(grep -c "^TF_DS_WINDOW=" <<<"$all")" == 1 ]] || { echo "TF_DS_POOL / TF_DS_WINDOW not exactly once: $all"; exit 1; }
  grep -qx "TF_DS_POOL=2101248" <<<"$all" && grep -qx "TF_DS_WINDOW=1048576" <<<"$all" && grep -qx "TF_DS_KV_SHARD=1" <<<"$all" || { echo "$all"; exit 1; }
  [[ "$FOLLOW" == "--pool 2101248 --streams 4 "* ]] || { echo "rank 1: $FOLLOW"; exit 1; }'
# start.sh / stop.sh: a container left over, an exited one too (rank 0 after a broken pair's status 75), goes through
# stop.sh before a launch removes it, and stop.sh saves each rank's log before its docker rm: a restart keeps the evidence
check_ "start.sh: a container left over, running or exited, is stopped by stop.sh (its log saved) before the launch" '
  t=$(mktemp -d "${TMPDIR:-/tmp}/port-test.XXXXXX"); trap "rm -rf -- \"$t\"" EXIT
  eval "$(grep -E "^exists_(here|worker)\(\) " "$here/../start.sh")"
  CONTAINER_NAME=c
  docker() { [[ "$*" == "inspect c" && -e "$t/here" ]]; }
  worker() { [[ "$1" == 1 && "${*:2}" == "docker inspect c" && -e "$t/there" ]]; }
  ! exists_here && ! exists_worker 1 || { echo "a container found where there is none"; exit 1; }
  : > "$t/here"; exists_here || { echo "an exited container here is not found"; exit 1; }
  rm "$t/here"; : > "$t/there"; exists_worker 1 || { echo "a container on the worker is not found"; exit 1; }
  launch=$(sed -n "/^if (( ! DRY )); then/,/^  memguard_start/p" "$here/../start.sh")
  [[ "$launch" == *"if exists_here || exists_worker 1; then ./stop.sh; fi"*"docker rm -f"* ]] || { echo "start.sh removes a container stop.sh has not seen"; exit 1; }
  grep -q "^here=0; docker ps -a " "$here/../stop.sh" || { echo "stop.sh does not count an exited container here"; exit 1; }
  for f in stop_here stop_worker; do
    body=$(sed -n "/^$f() {/,/^}/p" "$here/../stop.sh")
    [[ "$body" == *save_log*"docker rm -f"* ]] || { echo "stop.sh $f: no log saved before docker rm"; exit 1; }
  done'
check_ "port-patches: newreadme keeps the prose and is idempotent" '
  PORT_PATCHES_LIB=1; source "$here/port-patches.sh"
  t=$(mktemp -d "${TMPDIR:-/tmp}/port-test.XXXXXX"); trap "rm -rf -- \"$t\"" EXIT
  readme="$here/../zig/patches/README.md"
  prose=$(grep -F "<!-- port-patches -->" "$readme" | grep -v "^|" | head -1)
  [[ -n "$prose" ]] || { echo "the README has no prose line naming the marker: the test needs one"; exit 1; }
  printf "| \`0007-glm-port.patch\` | x | off | y <!-- port-patches --> |\n" > "$t/rows"
  newreadme "$readme" "$t/rows" > "$t/a"; newreadme "$t/a" "$t/rows" > "$t/b"
  cmp -s "$t/a" "$t/b" || { echo "a second run changed the README:"; diff "$t/a" "$t/b"; exit 1; }
  grep -qxF "$prose" "$t/a" || { echo "the prose line was dropped: $prose"; exit 1; }
  [[ "$(grep -cE "$ROW_RE" "$t/a")" == 1 ]] || { echo "not exactly one generated row"; exit 1; }
  # the row lands right after the last hand-written row, and only hand-written lines are lost: none
  [[ "$(grep -vE "$ROW_RE" "$t/a")" == "$(grep -vE "$ROW_RE" "$readme")" ]] || { echo "hand-written lines changed"; exit 1; }
  last=$(grep -nE "^[|] \`[0-9]+-" "$t/a" | grep -v "port-patches" | tail -1 | cut -d: -f1)
  [[ "$(sed -n "$((last + 1))p" "$t/a")" == "$(cat "$t/rows")" ]] || { echo "the row is not after the last hand-written row"; exit 1; }'

# ---- start.sh DRY_RUN=1 against the recipe before the port (PORT_TEST_BASE, default 370daf0): both trees run with the
# same stub docker, ssh, flock and ss on PATH and the same fixed link (detect_link is replaced in both copies of
# scripts/nodes.sh: the routes and sysfs of a Spark pair are not what is compared), and the two ranks' docker commands
# are compared, against MiaAI-Lab's main 370daf0 (zig/patches/0008's POOL included). All settings off (SPILL_GIB=0
# KV_SHARD=0), and the defaults (off): they differ only by --label tf.recipe.port=off. KV_SHARD=1 + POOL against the
# base with the same POOL, and (on Linux) the spill tier: they differ only by the tokens the port adds, each of which
# must be there. A fork whose defaults are on (port.sh's PORT_SPILL_DEFAULT_GIB / PORT_SHARD_DEFAULT): its defaults are
# the same commands as those settings given explicitly, or, where the default spill folder fails its checks here, as
# SPILL_GIB=0 with them and a warning. A failing folder with SPILL_GIB set refuses. start.sh needs bash 4
# (BASH4=<path> picks one); without one, or without the base commit, this is skipped.
b4=""
for b in "${BASH4:-}" "$BASH" $(type -ap bash 2>/dev/null); do
  [[ -n "$b" && -x "$b" ]] || continue
  v=$("$b" -c 'echo "${BASH_VERSINFO[0]}"' 2>/dev/null) || continue
  [[ "$v" =~ ^[0-9]+$ ]] && (( v >= 4 )) && { b4=$b; break; }
done
gbase=${PORT_TEST_BASE:-370daf0}
groot=$(cd "$here/.." && pwd)
if [[ -z "$b4" ]]; then
  echo "skip start.sh DRY_RUN golden test (needs bash 4 or later: none found; BASH4=<path>)"
elif ! git -C "$groot" cat-file -e "$gbase^{commit}" 2>/dev/null; then
  echo "skip start.sh DRY_RUN golden test (no commit $gbase in $groot; PORT_TEST_BASE=<commit>)"
else
  G_T=$(mktemp -d "${TMPDIR:-/tmp}/port-golden.XXXXXX"); G_T=$(cd "$G_T" && pwd -P)
  mkdir -p "$G_T/bin" "$G_T/home" "$G_T/base" "$G_T/port"
  git -C "$groot" archive "$gbase" start.sh scripts | tar -x -C "$G_T/base"
  cp -R "$groot/start.sh" "$groot/scripts" "$G_T/port/"
  for d in base port; do
    rm -f "$G_T/$d/scripts/local.sh"
    cat >> "$G_T/$d/scripts/nodes.sh" <<'EOF'
# scripts/port_test.sh: a fixed link instead of the routes and sysfs of a Spark pair
detect_link() {
  HEAD_ADDR=10.9.0.1; HEAD_DEV=enp1s0f0np0; HEAD_HCA=rocep1s0f0; HEAD_GID=3
  WORKER_ADDR=10.9.0.2; WORKER_DEV=enp1s0f0np0; WORKER_HCA=rocep1s0f0; WORKER_GID=3
  HEAD_HCAS=rocep1s0f0,roceP2p1s0f0; WORKER_HCAS=rocep1s0f0,roceP2p1s0f0
  NODE_DEV=("$HEAD_DEV" "$WORKER_DEV"); NODE_HCAS=("$HEAD_HCAS" "$WORKER_HCAS"); NODE_GID=("$HEAD_GID" "$WORKER_GID")
  LINK_HEAD_ADDR[1]=$HEAD_ADDR; LINK_WORKER_ADDR[1]=$WORKER_ADDR
  MASTER_ADDR=${MASTER_ADDR:-$HEAD_ADDR}
}
EOF
  done
  # docker: no image here (so no capabilities check), nothing else expected in a dry run; ssh: `true` answers, any
  # other command runs here (the worker's spill-folder probe); flock and ss: nothing to lock or listen
  printf '#!/bin/sh\ncase "$1" in image|inspect|volume|ps) exit 1 ;; esac\necho "docker $*" >> "%s/unexpected"; exit 1\n' "$G_T" > "$G_T/bin/docker"
  printf '#!%s\nwhile [ "${1:-}" = -o ]; do shift 2; done\nshift\n[ "$*" = true ] && exit 0\nexec %q -c "$*"\n' "$b4" "$b4" > "$G_T/bin/ssh"
  printf '#!/bin/sh\nexit 0\n' > "$G_T/bin/flock"; cp "$G_T/bin/flock" "$G_T/bin/ss"
  chmod +x "$G_T/bin/"*
  # grun <tree> <out> [NAME=value...]: the tree's start.sh, DRY_RUN=1, the live serving settings; its two ranks'
  # docker commands (the lines after "[dry-run] rank ...") into <out>
  grun() {
    local tree=$1 out=$2; shift 2
    env -i PATH="$G_T/bin:$PATH" HOME="$G_T/home" WORKER=u@10.9.0.2 DRY_RUN=1 PREPARE=0 IMAGE=tensorfold-dsv41:golden \
      HF_CACHE="$G_T/home/hf" STATE_DIR="$G_T/state" LOG_DIR="$G_T/logs" CONTEXT=1048576 PARALLEL=8 ROUND_ROWS=16 "$@" \
      "$b4" "$G_T/$tree/start.sh" > "$G_T/$out.log" 2>&1 || { echo "start.sh ($tree) failed:"; tail -5 "$G_T/$out.log"; return 1; }
    awk 'p { print; p = 0 } /^\[dry-run\] rank [01] / { p = 1 }' "$G_T/$out.log" > "$G_T/$out"
    [[ "$(wc -l < "$G_T/$out" | tr -d ' ')" == 2 ]] || { echo "start.sh ($tree) printed no two docker commands:"; tail -5 "$G_T/$out.log"; return 1; }
  }
  # gcmp <out> <pool or -> <token>...: <out> with every token (each must be there), the tf.recipe.port label, the
  # container log's rotation (LOG_MAX_SIZE, issue #49; both ranks must carry it) and the graph cap (0033: rank 1's
  # --graph-cap, rank 0's TF_DS_GRAPH_CAP; both must carry it) removed,
  # and rank 1's --pool <pool> put back to --pool 1048576, equals the base's commands (GBASE: which base run, base.out)
  gcap_sed='s/ (--graph-cap |-e TF_DS_GRAPH_CAP=)[0-9]+//'   # 0033's two tokens, which the base has not
  gcmp() {
    local out=$1 pool=$2 tok x b=${GBASE:-base.out}; shift 2
    x=$(sed -E -e 's/ --label tf\.recipe\.port=[^ ]+//' -e 's/ --log-driver json-file --log-opt max-size=[^ ]+ --log-opt max-file=[^ ]+//' -e "$gcap_sed" "$G_T/$out")
    [[ "$(grep -c -- ' --label tf\.recipe\.port=' "$G_T/$out")" == 2 ]] || { echo "not both ranks carry tf.recipe.port"; return 1; }
    [[ "$(sed -n 1p "$G_T/$out")" == *" --aio 1 --graph-cap 128 "* && "$(sed -n 2p "$G_T/$out")" == *" -e TF_DS_GRAPHS=1 -e TF_DS_GRAPH_CAP=128 "* ]] || { echo "not both ranks carry the graph cap (0033: --graph-cap, TF_DS_GRAPH_CAP)"; return 1; }
    [[ "$(grep -c -- ' --log-driver json-file --log-opt max-size=100m --log-opt max-file=3 ' "$G_T/$out")" == 2 ]] || { echo "not both ranks carry the container log's rotation (LOG_MAX_SIZE, #49)"; return 1; }
    for tok in "$@"; do
      [[ "$x" == *" $tok"* ]] || { echo "missing: $tok"; return 1; }
      x=${x/" $tok"/}   # its first occurrence: a token both ranks get is listed twice
    done
    [[ "$pool" == - ]] || { [[ "$x" == *"--pool $pool "* ]] || { echo "rank 1 has no --pool $pool"; return 1; }; x=${x//"--pool $pool "/"--pool 1048576 "}; }
    diff <(cat "$G_T/$b") <(printf '%s\n' "$x") || { echo "(< $gbase, > the port with the tokens above removed)"; return 1; }
    [[ ! -s "$G_T/unexpected" ]] || { echo "unexpected docker calls:"; cat "$G_T/unexpected"; return 1; }
  }
  check_ "golden: base $gbase runs" 'grun base base.out'
  check_ "golden: all off = $gbase but for tf.recipe.port=off" '
    grun port off.out SPILL_GIB=0 KV_SHARD=0 || exit 1
    [[ "$(grep -c -- " --label tf\.recipe\.port=off " "$G_T/off.out")" == 2 ]] || { echo "the label is not off on both ranks"; exit 1; }
    gcmp off.out -'
  check_ "golden: LOG_MAX_SIZE= (empty) leaves the daemon's log settings: no rotation flags (#49)" '
    grun port nolog.out SPILL_GIB=0 KV_SHARD=0 LOG_MAX_SIZE= || exit 1
    ! grep -q -- "--log-opt" "$G_T/nolog.out" || { echo "a rotation flag with LOG_MAX_SIZE empty"; exit 1; }
    diff "$G_T/base.out" <(sed -E -e "s/ --label tf\.recipe\.port=[^ ]+//" -e "$gcap_sed" "$G_T/nolog.out")'
  check_ "golden: GRAPH_CAP=0 (keep every graph) reaches both ranks; a bad GRAPH_CAP refuses the start; a raw TF_DS_GRAPH_CAP export loses; GRAPHS=0 sends no --graph-cap (0033, #69)" '
    grun port cap0.out SPILL_GIB=0 KV_SHARD=0 GRAPH_CAP=0 || exit 1
    [[ "$(sed -n 1p "$G_T/cap0.out")" == *" --graph-cap 0 "* && "$(sed -n 2p "$G_T/cap0.out")" == *" -e TF_DS_GRAPH_CAP=0 "* ]] || { echo "GRAPH_CAP=0 is not on both ranks"; exit 1; }
    ! grun port badcap.out GRAPH_CAP=-1 2>/dev/null && grep -q "GRAPH_CAP is a count" "$G_T/badcap.out.log" || { echo "GRAPH_CAP=-1 did not refuse the start"; exit 1; }
    grun port rawcap.out SPILL_GIB=0 KV_SHARD=0 TF_DS_GRAPH_CAP=7 || exit 1
    [[ "$(grep -o -- "TF_DS_GRAPH_CAP=[0-9]*" "$G_T/rawcap.out" | tr "\n" " ")" == "TF_DS_GRAPH_CAP=128 " ]] || { echo "a raw TF_DS_GRAPH_CAP export reached a rank: $(grep -o -- "TF_DS_GRAPH_CAP=[0-9]*" "$G_T/rawcap.out" | tr "\n" " ")"; exit 1; }
    grun port nographs.out SPILL_GIB=0 KV_SHARD=0 GRAPHS=0 || exit 1
    ! grep -q -- "--graph-cap" "$G_T/nographs.out" || { echo "rank 1 got --graph-cap with GRAPHS=0 (an image from before 0033 refuses it)"; exit 1; }'
  check_ "golden: a LOG_MAX_SIZE that is not a size refuses the start (#49)" '
    ! grun port badlog.out LOG_MAX_SIZE="100m;true" 2>/dev/null && grep -q "LOG_MAX_SIZE is a size" "$G_T/badlog.out.log"'
  check_ "golden: KV_SHARD=1 POOL=2101248 adds only its variables to $gbase's POOL=2101248 (0008: TF_DS_POOL, TF_DS_WINDOW, --pool)" '
    grun base basepool.out POOL=2101248 || exit 1
    grep -q -- "-e TF_DS_POOL=2101248 -e TF_DS_WINDOW=1048576 " "$G_T/basepool.out" && grep -q -- "--pool 2101248 " "$G_T/basepool.out" || { echo "the base does not map POOL"; exit 1; }
    grun port kv.out SPILL_GIB=0 KV_SHARD=1 POOL=2101248 || exit 1
    grep -q -- "--label tf.recipe.port=pool=2101248,shard=1/fetch/c1t1k1024,spill=0 " "$G_T/kv.out" || { echo "label"; exit 1; }
    v="-e TF_DS_KV_SHARD=1 -e TF_DS_KV_SHARD_CARRY=1 -e TF_DS_KV_SHARD_TWINS=1 -e TF_DS_KV_SHARD_CHECK=1024"
    GBASE=basepool.out gcmp kv.out - "$v" "$v"'
  check_ "golden: KV_SHARD=1 alone derives POOL=2101248 (the same commands)" '
    grun port kvd.out SPILL_GIB=0 KV_SHARD=1 || exit 1
    diff "$G_T/kv.out" "$G_T/kvd.out"'
  gdef_gib=$(sed -n 's/^PORT_SPILL_DEFAULT_GIB=\([0-9.]*\).*/\1/p' "$groot/scripts/port.sh")
  gdef_shard=$(sed -n 's/^PORT_SHARD_DEFAULT=\([01]\).*/\1/p' "$groot/scripts/port.sh")
  if [[ "$gdef_gib" =~ ^0*(\.0*)?$ && "$gdef_shard" == 0 ]]; then
    check_ "golden: the defaults (everything off) = $gbase but for tf.recipe.port=off" '
      grun port def.out || exit 1
      diff "$G_T/off.out" "$G_T/def.out"'
  else
    check_ "golden: a fork's defaults = SPILL_GIB=$gdef_gib KV_SHARD=$gdef_shard given explicitly (SPILL_GIB=0 where the folder fails here)" '
      grun port def.out || exit 1
      if grep -q "SPILL TIER OFF" "$G_T/def.out.log"; then echo "(the default spill folder fails here: compared with SPILL_GIB=0)"; s=0; else s=$gdef_gib; fi
      grun port exp.out SPILL_GIB=$s KV_SHARD=$gdef_shard ${GDEF_EXTRA:-} || exit 1
      diff "$G_T/exp.out" "$G_T/def.out"'
    check_ "golden: a fork's default spill tier whose folder fails starts without it" '
      grun port deg.out SPILL_FLOOR_GIB=999999 || exit 1
      grep -q "WARN: SPILL TIER OFF for this start" "$G_T/deg.out.log" || { tail -5 "$G_T/deg.out.log"; exit 1; }
      ! grep -qE -- "--spill-dir|tf\.recipe\.spill|TF_DS_SPILL" "$G_T/deg.out" || { echo "spill left in the commands"; exit 1; }'
  fi
  check_ "golden: SPILL_GIB set with a folder that fails refuses" '
    if grun port ref.out SPILL_GIB=64 SPILL_FLOOR_GIB=999999 >/dev/null; then echo "SPILL_GIB=64 with a failing folder was not refused"; exit 1; fi
    grep -q "ERROR: SPILL_DIR" "$G_T/ref.out.log" || { tail -3 "$G_T/ref.out.log"; exit 1; }'
  gfs=$(findmnt -n -o FSTYPE -T "$G_T" 2>/dev/null | head -1)
  if [[ "$(uname -s)" != Linux ]]; then
    echo "skip golden spill-on variant (the spill-folder probe needs Linux)"
  elif [[ "$gfs" =~ ^(tmpfs|ramfs|devtmpfs|overlay|overlayfs|squashfs)$ ]]; then
    echo "skip golden spill-on variant ($G_T is on $gfs, which the spill check refuses: TMPDIR=<a folder on disk>)"
  else
    check_ "golden: SPILL_GIB=1 adds the mounts, labels, flags and variables" '
      grun port spill.out KV_SHARD=0 SPILL_GIB=1 SPILL_FLOOR_GIB=0 SPILL_DIR="$G_T/spill0" WORKER_SPILL_DIR="$G_T/spill1" || exit 1
      r0=$(readlink -m "$G_T/spill0"); r1=$(readlink -m "$G_T/spill1")
      [[ ! -e "$G_T/spill0" && ! -e "$G_T/spill1" ]] || { echo "DRY_RUN created a spill folder"; exit 1; }
      v="-e TF_DS_SPILL_DIR=/spill -e TF_DS_SPILL_GIB=1 -e TF_DS_SPILL_MIN_FREE_GIB=0 -e TF_DS_SPILL_MIN_TOKENS=8192"
      v+=" -e TF_DS_SPILL_QUOTA=0 -e TF_DS_SPILL_STAGING_MIB=128 -e TF_SPILL_TURNS=4 -e TF_SPILL_FLUSH_MS=60000"
      # the identity as the port tree derives it (no image here: the pins); no weight files, so no TF_DS_RANK_CACHE_SHA
      id=$(cd "$G_T/port" && env -i PATH="$PATH" HOME="$G_T/home" HF_CACHE="$G_T/home/hf" SPILL_GIB=1 "$b4" -c \
           "source scripts/config.sh >/dev/null 2>&1; port_image_label() { :; }; port_ident_map x; for e in \"\${PORT_IDENT_ENV[@]}\"; do printf -- \"-e %q \" \"\$e\"; done")
      [[ "$id" == *TF_DS_ENGINE_ID=tensorfold@* ]] || { echo "no identity derived: $id"; exit 1; }
      v+=" ${id% }"
      gcmp spill.out - "--label tf.recipe.spill=$r1 -v $r1:/spill" "--label tf.recipe.spill=$r0 -v $r0:/spill" "$v" "$v" \
        "--spill-dir /spill --spill-gib 1 --spill-min-free-gib 0 --spill-min-tokens 8192"'
  fi
  rm -rf -- "$G_T"
fi

rm -rf -- "$mt"
echo "$((n - fails)) of $n checks passed"
(( fails == 0 ))
