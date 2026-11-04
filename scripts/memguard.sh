#!/usr/bin/env bash
# Memory watchdog for a DGX Spark running a rank: a Spark's GPU shares its 121 GiB with the host, so a rank that takes
# all of it does not get a clean CUDA out-of-memory error, the whole machine stops answering (and has to be power-cycled).
# This kills the rank's container (docker/podman kill, no grace period) before the machine gets there, on either of two levels:
#   - MemAvailable under the HARD floor (default 1 GiB) in two samples in a row (0.25 s apart): at once;
#   - MemAvailable under the (soft) floor for HOLD seconds without once coming back over it (default 4 s): a long prompt's workspace
#     takes a few GiB for a moment and gives them back, which is not a rank in trouble. A dip that recovers is logged, not killed.
# Self-contained: start.sh copies it to the worker.
#
# Usage: memguard.sh start <container> <floor GiB> <log file> <pid file> [hold seconds [hard floor GiB]]
#                                                       (replaces a guard already running for the pid file)
#        memguard.sh stop <pid file>
# The engine keeps its own reserve (TENSORFOLD_MEMORY_RESERVE_GIB, 12.2 GiB by default) free, so a healthy rank stays well
# above a floor of a few GiB; only something else eating into the reserve (or a reserve set too low) trips this.
set -u

# The container runtime (scripts/runtime.sh travels beside this script: start.sh copies both to the worker); ctr
# resolves to docker, podman or "sudo -n podman".
_here=$(cd "$(dirname "$0")" && pwd)
[[ -r "$_here/runtime.sh" ]] && . "$_here/runtime.sh"
if ! declare -F ctr >/dev/null; then
  ctr() { if [[ "${CONTAINER_RT:-docker}" == podman ]]; then if [[ "${CONTAINER_RT_SUDO:-0}" == 1 ]]; then sudo -n podman "$@"; else podman "$@"; fi; else docker "$@"; fi; }
fi

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
avail_kb() { local k v; while read -r k v _; do [[ $k == MemAvailable: ]] && { echo "$v"; return; }; done <"${MEMGUARD_MEMINFO:-/proc/meminfo}"; }

stop_guard() {
  local pidfile=$1 pid
  [[ -f "$pidfile" ]] || return 0
  pid=$(cat "$pidfile" 2>/dev/null)
  if [[ "$pid" =~ ^[0-9]+$ ]] && grep -q memguard "/proc/$pid/cmdline" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 50); do [[ -d /proc/$pid ]] || break; sleep 0.1; done   # its exit handler runs first
  fi
  rm -f "$pidfile"
}
# a sleep a signal ends at once (a plain sleep holds the guard's exit handler back until it is over)
nap() { sleep "$1" & wait $!; }

run_guard() {
  local c=$1 floor_kb hard_kb log=$3 pidfile=$4 hold_us a snap now since=0 low_min=0 hard_low=0 floor_gib=$2 hold=${5:-4} hard_gib=${6:-1}
  floor_kb=$(awk -v g="$floor_gib" 'BEGIN {printf "%d", g * 1048576}')
  hard_kb=$(awk -v g="$hard_gib" 'BEGIN {printf "%d", g * 1048576}')
  hold_us=$(awk -v h="$hold" 'BEGIN {printf "%d", h * 1000000}')
  snap=$(mktemp "${TMPDIR:-/tmp}/memguard-snap.XXXXXX") || snap=/dev/null
  mkdir -p "$(dirname "$log")" "$(dirname "$pidfile")"
  echo $$ > "$pidfile"
  trap '[[ "$(cat "$pidfile" 2>/dev/null)" == "$$" ]] && rm -f "$pidfile"; rm -f "$snap"; exit 0' TERM INT
  echo "$(stamp) memguard: watching $c, floor $floor_gib GiB for $hold s, hard floor $hard_gib GiB (MemAvailable $(( $(avail_kb) / 1048576 )) GiB now)" >> "$log"
  while nap 0.25; do
    a=$(avail_kb)
    now=${EPOCHREALTIME/./}
    if (( a >= floor_kb )); then
      if (( since )); then
        echo "$(stamp) memguard: MemAvailable was under the $floor_gib GiB floor for $(( (now - since) / 1000 )) ms (lowest $(( low_min / 1024 )) MiB) and recovered: no kill" >> "$log"
      fi
      since=0; hard_low=0; continue
    fi
    if (( ! since )); then since=$now; low_min=$a; fi
    (( a < low_min )) && low_min=$a
    if (( a < hard_kb )); then (( ++hard_low )); else hard_low=0; fi
    # the hard floor needs two samples (a single read can be a glitch), the soft floor needs HOLD seconds under it throughout
    if (( hard_low < 2 && now - since < hold_us )); then continue; fi
    since=0; hard_low=0
    # Kill first, log afterwards: on a machine that is already thrashing every command ahead of the kill costs time. The
    # process snapshot runs beside it (it shows who else holds the memory); docker kill fails when the container is not running.
    ps -eo rss,pid,comm --sort=-rss 2>/dev/null | head -6 > "$snap" &
    if timeout 20 ctr kill "$c" >/dev/null 2>&1; then
      wait
      {
        echo "$(stamp) memguard: MemAvailable $(( a / 1024 )) MiB (lowest $(( low_min / 1024 )) MiB) was under the $floor_gib GiB floor for $hold s or under the $hard_gib GiB hard floor: killed container $c"
        sed 's/^/    /' "$snap"
      } >> "$log"
      nap 5
    else
      wait
    fi
  done
}

case "${1:-}" in
  start)
    shift
    (( $# >= 4 && $# <= 6 )) || { echo "usage: memguard.sh start <container> <floor GiB> <log file> <pid file> [hold seconds [hard floor GiB]]" >&2; exit 2; }
    [[ "$2" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "memguard: floor is a number of GiB (2 or 2.5), not $2" >&2; exit 2; }
    [[ -z "${5:-}" || "$5" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "memguard: hold is a number of seconds, not $5" >&2; exit 2; }
    [[ -z "${6:-}" || "$6" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "memguard: the hard floor is a number of GiB, not $6" >&2; exit 2; }
    stop_guard "$4"
    # detach: its own session, no terminal, so it outlives the ssh or shell that started it. Close fd 8, start.sh's lock
    # (flock on STATE_DIR/start.lock): a guard that kept it would hold the lock for as long as it runs.
    setsid nohup "$0" run "$@" >/dev/null 2>&1 < /dev/null 8>&- 9>&- &
    for _ in 1 2 3 4 5 6 7 8 9 10; do [[ -f "$4" ]] && exit 0; sleep 0.2; done
    echo "memguard: did not start" >&2; exit 1 ;;
  run) shift; run_guard "$@" ;;
  stop) stop_guard "${2:?usage: memguard.sh stop <pid file>}" ;;
  *) echo "usage: memguard.sh start|stop ..." >&2; exit 2 ;;
esac
