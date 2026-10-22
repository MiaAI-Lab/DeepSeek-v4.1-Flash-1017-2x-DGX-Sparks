# The memory a Spark can give a starting rank, as the engine's CUDA startup budget counts it (wojo's GLM 0119, PR #149;
# the engine's native/cuda_memory.zig parkedBytes and counting, ported to awk). Sourced by scripts/nodes.sh; start.sh
# sends mem_counting to the worker over ssh (self-contained: awk and getconf only). bash 3.2 and POSIX awk (mawk, gawk,
# BSD awk), so that scripts/port_test.sh runs it on a Mac against synthetic /proc files.
#
# A bulk free (a docker save | load of the image, a compiler's exit, the first start of a new image) leaves 2-9 GiB on
# the kernel's per-CPU free lists for a minute or two. MemFree and MemAvailable do not count them, so a check right
# after an image build read 110.3 GiB where the engine (which counts them) had room, and MemAvailable came back to
# 116 GiB minutes later. The engine adds them back since 0119; start.sh's MIN_AVAIL_GIB check does the same here.
#
# The rule, exactly the engine's: in /proc/zoneinfo, under each zone's `pagesets` line (the indented lines that follow
# it: four spaces, or the "  vm stats threshold" line), sum every `count:` (pages on a CPU's list) and every
# `high_min:` (that list's resting level); parked = count - high_min pages, or count pages less 512 MiB when the kernel
# prints no high_min (before 6.7); never below 0. Any unparsable count or high_min counts nothing (0), as the engine
# does. Then available = MemAvailable + parked, at most MemTotal. /proc/meminfo is read first, /proc/zoneinfo right
# after (pages that drain in between are counted by neither, never twice).

# mem_counting [meminfo] [zoneinfo] [page bytes]: "<available GiB> <parked GiB> <parked bytes>" (GiB with one
# decimal), from /proc/meminfo, /proc/zoneinfo and getconf PAGESIZE by default. A missing zoneinfo counts no parked
# pages. Prints nothing (and fails) when the meminfo has no MemAvailable.
mem_counting() {
  local mi=${1:-/proc/meminfo} zi=${2:-/proc/zoneinfo} pg=${3:-}
  [ -n "$pg" ] || pg=$(getconf PAGESIZE 2>/dev/null)
  case "$pg" in ''|*[!0-9]*) pg=4096 ;; esac
  [ -r "$zi" ] || zi=/dev/null
  awk -v page="$pg" -v rest_default=536870912 -v zfile="$zi" '
    FILENAME != zfile {
      if ($1 == "MemTotal:") total = $2 * 1024
      else if ($1 == "MemAvailable:") { avail = $2 * 1024; seen = 1 }
      next
    }
    {
      row = $0
      gsub(/\r/, " ", row)
      t = row; sub(/^[ \t]+/, "", t); sub(/[ \t]+$/, "", t)
      if (t == "pagesets") { listed = 1; next }
      if (listed && substr($0, 1, 4) != "    " && substr($0, 1, 10) != "  vm stats") { listed = 0; next }
      if (!listed) next
      n = split(t, w, /[ \t]+/)
      if (w[1] != "count:" && w[1] != "high_min:") next
      if (n < 2 || w[2] !~ /^[0-9]+$/) { bad = 1; next }
      if (w[1] == "count:") count += w[2]; else rest += w[2]
    }
    END {
      if (!seen) exit 1
      parked = 0
      if (!bad && count > 0) {
        resting = rest > 0 ? rest * page : rest_default
        parked = count * page - resting
        if (parked < 0) parked = 0
      }
      a = avail + parked
      if (total > 0 && a > total) a = total
      printf "%.1f %.1f %.0f\n", a / 1073741824, parked / 1073741824, parked
    }' "$mi" "$zi"
}
