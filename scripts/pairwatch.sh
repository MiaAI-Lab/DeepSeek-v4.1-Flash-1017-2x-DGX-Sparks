#!/usr/bin/env bash
# The pair-restart guard for rank 0 (the port's): with the spill tier or KV sharding on, the engine's broken-pair policy
# (TF_DS_PAIR_BROKEN, default `exit` then) ends rank 0's process with status 75 when the pair breaks (rank 1 gone or
# hung, a spill-tier or KV-transfer break). start.sh runs detached containers, so nothing would restart them until
# the deployment's health watchdog gave up on /health (minutes). This guard polls rank 0's container every
# PAIRWATCH_POLL_S seconds (2) and, when it finds it `exited 75`, runs `./start.sh restart` from the recipe folder: both
# ranks stop and start again, the load included. docs/PORT-RECIPE.md "When the rank pair breaks".
#
# It acts only on 75 (memguard's kill is 137, a crash another code, a clean stop 0: those stay the watchdog's), only
# while the arm file exists (start.sh creates it, stop.sh removes it before it stops anything, so a stop's own exit is
# never restarted), never while the hold file (if one is given) exists, and at most MAX times an hour (then it logs and
# ends, leaving the stack to a person or the watchdog). It replaces itself with the restart (exec): the restarted
# start.sh starts a new guard. Self-contained, like scripts/memguard.sh.
#
# Usage: pairwatch.sh start <container> <recipe dir> <log file> <pid file> <arm file> [max an hour [hold file]]
#        pairwatch.sh stop <pid file>
set -u

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
nap() { sleep "$1" & wait $!; }

stop_watch() {
  local pidfile=$1 pid
  [[ -f "$pidfile" ]] || return 0
  pid=$(cat "$pidfile" 2>/dev/null)
  if [[ "$pid" =~ ^[0-9]+$ ]] && grep -q pairwatch "/proc/$pid/cmdline" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 50); do [[ -d /proc/$pid ]] || break; sleep 0.1; done
  fi
  rm -f "$pidfile"
}

run_watch() {
  local c=$1 dir=$2 log=$3 pidfile=$4 arm=$5 max=${6:-6} hold=${7:-} hist st now n held=0 why
  hist="$(dirname "$pidfile")/pairwatch.restarts"
  mkdir -p "$(dirname "$log")" "$(dirname "$pidfile")"
  echo $$ > "$pidfile"
  trap '[[ "$(cat "$pidfile" 2>/dev/null)" == "$$" ]] && rm -f "$pidfile"; exit 0' TERM INT
  echo "$(stamp) pairwatch: watching $c for exit status 75 (at most $max restarts an hour${hold:+, never while $hold exists})" >> "$log"
  while nap "${PAIRWATCH_POLL_S:-2}"; do
    st=$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$c" 2>/dev/null) || continue
    [[ "$st" == "exited 75" ]] || { held=0; continue; }
    [[ -e "$arm" ]] || continue                                   # stop.sh is stopping the stack: not a break
    if [[ -n "$hold" && -e "$hold" ]]; then
      (( held )) || echo "$(stamp) pairwatch: $c exited with status 75, but $hold exists: not restarting" >> "$log"
      held=1; continue
    fi
    now=$(date +%s)
    n=$(( $(awk -v t=$((now - 3600)) '$1 > t' "$hist" 2>/dev/null | wc -l) ))
    if (( n >= max )); then
      echo "$(stamp) pairwatch: $c exited with status 75, but the pair was restarted $n times in the last hour: not again (docker logs $c)" >> "$log"
      rm -f "$pidfile"; exit 0
    fi
    echo "$now" >> "$hist"
    why=$(docker logs --tail 300 "$c" 2>&1 | grep -m1 -E 'rank pair broken|the rank pair broke' || true)
    echo "$(stamp) pairwatch: $c exited with status 75 (${why:-no rank pair line in its log}): restarting both ranks ($dir/start.sh restart)" >> "$log"
    rm -f "$pidfile"; trap - TERM INT
    cd "$dir" || { echo "$(stamp) pairwatch: no recipe folder $dir" >> "$log"; exit 1; }
    exec ./start.sh restart >> "$log" 2>&1
  done
}

case "${1:-}" in
  start)
    shift
    (( $# >= 5 && $# <= 7 )) || { echo "usage: pairwatch.sh start <container> <recipe dir> <log file> <pid file> <arm file> [max an hour [hold file]]" >&2; exit 2; }
    [[ -z "${6:-}" || "$6" =~ ^[1-9][0-9]*$ ]] || { echo "pairwatch: the hourly limit is a positive count, not $6" >&2; exit 2; }
    [[ -x "$2/start.sh" ]] || { echo "pairwatch: no $2/start.sh" >&2; exit 2; }
    stop_watch "$4"
    # detached as memguard is (its own session, no terminal), without start.sh's lock (fd 8) or other fds
    setsid nohup "$0" run "$@" >/dev/null 2>&1 < /dev/null 8>&- 9>&- &
    for _ in 1 2 3 4 5 6 7 8 9 10; do [[ -f "$4" ]] && exit 0; sleep 0.2; done
    echo "pairwatch: did not start" >&2; exit 1 ;;
  run) shift; run_watch "$@" ;;
  stop) stop_watch "${2:?usage: pairwatch.sh stop <pid file>}" ;;
  *) echo "usage: pairwatch.sh start|stop ..." >&2; exit 2 ;;
esac
