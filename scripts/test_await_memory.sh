#!/usr/bin/env bash
# Regression test for await_memory (scripts/nodes.sh, MIN_AVAIL_WAIT_S): takes the function (and worker_ids, the workers it
# reads) out of nodes.sh and drives it with stubs for mem_available_gib, worker and sleep. Run: bash scripts/test_await_memory.sh
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
eval "$(awk '/^await_memory\(\) \{/ {on=1} on {print} on && /^\}/ {exit}' "$here/nodes.sh")"
eval "$(grep -E '^worker_ids\(\) ' "$here/nodes.sh")"
declare -F await_memory >/dev/null || { echo "FAIL: await_memory is not defined in nodes.sh"; exit 1; }
declare -F worker_ids >/dev/null || { echo "FAIL: worker_ids is not defined in nodes.sh"; exit 1; }
TP=2
sleep() { :; }
# both stubs run in $( ) subshells, so the reading count lives in a file
fails=0; counter=$(mktemp); trap 'rm -f "$counter"' EXIT
pick() { local -n a=$1; local n; n=$(<"$counter"); echo "${a[$(( n < ${#a[@]} ? n : ${#a[@]} - 1 ))]}"; }
mem_available_gib() { pick HERE_SEQ; }
worker() { pick WORK_SEQ; echo $(( $(<"$counter") + 1 )) >"$counter"; }
reset() { echo 0 >"$counter"; }
check() { reads=$(<"$counter"); # name expected_rc expected_reads
  if [[ "$rc" == "$2" && "$reads" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: rc=$rc (want $2), reads=$reads (want $3)"; fails=$((fails + 1)); fi
}
MIN_AVAIL_GIB=112

MIN_AVAIL_WAIT_S=30; HERE_SEQ=(119.0); WORK_SEQ=(119.0); reset; await_memory; rc=$?
check "both enough: one reading" 0 1
MIN_AVAIL_WAIT_S=30; HERE_SEQ=(119.0); WORK_SEQ=(111.1 110.4 119.0); reset; await_memory; rc=$?
check "worker settles on the third reading" 0 3
MIN_AVAIL_WAIT_S=30; HERE_SEQ=(119.0); WORK_SEQ=(100.0); reset; await_memory; rc=$?
check "never enough: gives up after 30 s / 2 + 1 readings" 1 16
MIN_AVAIL_WAIT_S=0; HERE_SEQ=(119.0); WORK_SEQ=(100.0); reset; await_memory; rc=$?
check "MIN_AVAIL_WAIT_S=0 reads once" 1 1

# Two workers (worker_ids 1 2, as at TP=3): each its own readings; a reading is one pass over this Spark and both
# workers, counted here by this Spark's
TP=3
mem_available_gib() { local n; n=$(<"$counter"); echo $(( n + 1 )) >"$counter"; at HERE_SEQ "$n"; }
worker() { local n; n=$(( $(<"$counter") - 1 )); if [[ "$1" == 1 ]]; then at W1_SEQ "$n"; else at W2_SEQ "$n"; fi; }
at() { local -n a=$1; echo "${a[$(( $2 < ${#a[@]} ? $2 : ${#a[@]} - 1 ))]}"; }
MIN_AVAIL_WAIT_S=30; HERE_SEQ=(119.0); W1_SEQ=(119.0); W2_SEQ=(119.0); reset; await_memory; rc=$?
check "two workers, all enough: one reading" 0 1
MIN_AVAIL_WAIT_S=30; HERE_SEQ=(119.0); W1_SEQ=(119.0); W2_SEQ=(100.0 111.9 118.5); reset; await_memory; rc=$?
check "rank 2 settles on the third reading" 0 3
[[ "${WORKER_GB[1]} ${WORKER_GB[2]}" == "119.0 118.5" ]] || { echo "FAIL WORKER_GB: ${WORKER_GB[*]} (want 119.0 118.5)"; fails=$((fails + 1)); }
MIN_AVAIL_WAIT_S=30; HERE_SEQ=(119.0); W1_SEQ=(119.0); W2_SEQ=(100.0); reset; await_memory; rc=$?
check "rank 2 never enough: gives up after 30 s / 2 + 1 readings" 1 16
MIN_AVAIL_WAIT_S=30; HERE_SEQ=(119.0); W1_SEQ=(""); W2_SEQ=(119.0); reset; await_memory; rc=$?
check "an empty reading from rank 1 is not enough" 1 16
exit $(( fails > 0 ))
