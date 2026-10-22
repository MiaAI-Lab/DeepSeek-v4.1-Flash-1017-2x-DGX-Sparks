#!/usr/bin/env bash
# Export an engine branch (wojo's GLM spill v2 / continuation / KV sharding / parked-page port into the
# native dsv41 engine) as zig/patches/NNNN-... files (numbered after the last hand-written patch) that zig/Dockerfile applies with `git apply` after the hand-written
# patches, in filename order. Nothing is written unless every patch applies:
#   1. the diff of BASE..BRANCH in the engine repository (squash: one patch; commits: one patch per first-parent commit,
#      a merge as its whole diff against its first parent), without the port's own notes (PORT.md, docs/port/)
#   2. a clean checkout of TF_REF (the engine pin, scripts/config.sh) gets the hand-written zig/patches in filename order,
#      exactly as the Dockerfile applies them, and must then equal BASE (so BASE is TF_REF plus those patches)
#   3. each new patch must pass `git apply --check` there and apply, and the result must equal BRANCH
#   4. only then: the previously exported patches (first line "# port-patches.sh") are replaced in zig/patches and
#      zig/patches/README.md's generated table rows (marked <!-- port-patches -->) rewritten
# The image tag (scripts/config.sh, image_hash) changes with the patches, so the next ./start.sh rebuilds the image.
#
# Usage: scripts/port-patches.sh [--dry-run] [--engine DIR] [--branch REF] [--base REF] [--mode squash|commits]
#                                [--name SLUG] [--title TEXT] [--exclude PATH]... [--no-default-excludes] [--keep]
#   --dry-run         build and check the patches in a temporary folder, print what would change, write nothing
#   --engine DIR      the engine repository (default ~/GitHub/tensorfold-dsv41)
#   --branch REF      what to export (required): engine work on top of BASE
#   --base REF        what it is relative to: TF_REF plus the hand-written patches (default port/series-pair-370daf0,
#                     the tip of the series zig/patches carries)
#   --mode            commits (default: one patch per first-parent commit, as the series) or squash (one patch,
#                     NNNN-<name>.patch)
#   --name SLUG       the squashed patch's name (default spill-kvshard)
#   --title TEXT      its README description (default: Robert Wojciechowski's (X @rwojo) GLM 0088/0110/0111/0118/0119, ported to the native engine)
#   --exclude PATH    leave PATH out of the patches as well (repeatable); --no-default-excludes keeps PORT.md, docs/port
#   --keep            keep the temporary folder (printed) for inspection
# Env: TF_REF, TF_REPO (default: scripts/config.sh's pins). TF_REF is taken from the engine repository when it has the
# commit, else fetched from TF_REPO (network). Runs on Linux and macOS (bash 3.2 or later, git).
[[ "${PORT_PATCHES_LIB:-0}" == 1 ]] || set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
die()  { printf '[port-patches] ERROR: %s\n' "$*" >&2; exit 1; }
log()  { printf '[port-patches] %s\n' "$*"; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
# A generated row: a table row (it starts with "| ") that carries the marker. Prose that names the marker is kept.
ROW_RE='^[|] .*<!-- port-patches -->'   # [|], not \|: awk -v would read \| as an escape
newreadme() {  # <readme> <rows file>: the README with the generated rows replaced, after the table's last hand-written row
  awk -v rows="$2" -v re="$ROW_RE" '
    $0 ~ re { next }
    { lines[++n] = $0 }
    END {
      last = 0
      for (i = 1; i <= n; i++) if (lines[i] ~ /^\| `[0-9]+-/) last = i
      for (i = 1; i <= n; i++) {
        print lines[i]
        if (i == last) while ((getline r < rows) > 0) print r
      }
    }' "$1"
}
# Sourced with PORT_PATCHES_LIB=1 (scripts/port_test.sh): define newreadme and stop here
[[ "${PORT_PATCHES_LIB:-0}" != 1 ]] || return 0

engine="$HOME/GitHub/tensorfold-dsv41"; branch=""; base=port/series-pair-370daf0; mode=commits; name=spill-kvshard
title="Robert Wojciechowski's (GitHub wojo, X @rwojo) GLM-5.3-Flash patches 0088/0110/0111/0118/0119 (spill v2, continuation, KV sharding with prompt borrowing and the ring exchange on 0008's pool, parked pages; GLM PRs #78 #124 #128 #152 #149), ported by him to the native dsv41 engine"
dry=0; keep=0; defaults=1; excludes=()
while (( $# )); do
  case "$1" in
    --dry-run) dry=1 ;;
    --keep) keep=1 ;;
    --engine) engine=${2:?--engine needs a directory}; shift ;;
    --branch) branch=${2:?--branch needs a ref}; shift ;;
    --base) base=${2:?--base needs a ref}; shift ;;
    --mode) mode=${2:?--mode needs squash or commits}; shift ;;
    --name) name=${2:?--name needs a slug}; shift ;;
    --title) title=${2:?--title needs text}; shift ;;
    --exclude) excludes+=("${2:?--exclude needs a path}"); shift ;;
    --no-default-excludes) defaults=0 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (--help)" ;;
  esac
  shift
done
[[ -n "$branch" ]] || die "--branch names the engine work to export, on top of $base (--base)"
[[ "$mode" == squash || "$mode" == commits ]] || die "--mode is squash or commits, not $mode"
[[ "$name" =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "--name is a lowercase slug (a-z 0-9 -), not $name"
(( defaults )) && excludes=(PORT.md docs/port ${excludes[@]+"${excludes[@]}"})

# the pins, as scripts/config.sh sets them (read, not sourced: config.sh needs bash 4 and a Spark's tools)
pin() { sed -n "s/^$1=\"\\\${$1:-\\([^}]*\\)}\".*/\\1/p" "$root/scripts/config.sh" | head -1; }
TF_REF=${TF_REF:-$(pin TF_REF)}; TF_REPO=${TF_REPO:-$(pin TF_REPO)}
[[ "$TF_REF" =~ ^[0-9a-f]{40}$ ]] || die "TF_REF is not a full commit hash: '$TF_REF'"

git -C "$engine" rev-parse --git-dir >/dev/null 2>&1 || die "$engine is not a git repository (--engine)"
branch_sha=$(git -C "$engine" rev-parse --verify -q "$branch^{commit}") || die "no $branch in $engine (--branch)"
base_sha=$(git -C "$engine" rev-parse --verify -q "$base^{commit}") || die "no $base in $engine (--base)"
git -C "$engine" merge-base --is-ancestor "$base_sha" "$branch_sha" || die "$base is not an ancestor of $branch: rebase or merge it first"
[[ "$base_sha" != "$branch_sha" ]] || die "$branch is $base itself: nothing to export"

spec=(.)
for x in ${excludes[@]+"${excludes[@]}"}; do spec+=(":(exclude)$x"); done
diffopts=(--binary --full-index --no-renames --no-color --no-ext-diff --src-prefix=a/ --dst-prefix=b/)

# the patches already there: hand-written ones stay and set the next number; exported ones (first line) are replaced
generated() { head -n 1 "$1" 2>/dev/null | grep -q '^# port-patches.sh:'; }
hand=(); old=(); last=0
for p in $(cd "$root/zig/patches" && LC_ALL=C ls -1 -- *.patch 2>/dev/null); do
  if generated "$root/zig/patches/$p"; then old+=("$p"); continue; fi
  hand+=("$p")
  [[ "$p" =~ ^([0-9]+)- ]] && (( 10#${BASH_REMATCH[1]} > last )) && last=$((10#${BASH_REMATCH[1]}))
done
(( ${#hand[@]} )) || die "no hand-written patches in zig/patches: refusing to guess the base"

tmp=$(mktemp -d "${TMPDIR:-/tmp}/port-patches.XXXXXX")
if (( keep )); then trap 'log "kept $tmp"' EXIT; else trap 'rm -rf -- "$tmp"' EXIT; fi
mkdir -p "$tmp/out"

# ---------------------------------------------------------------- 1. the patches
slug() { tr '[:upper:]' '[:lower:]' <<<"$1" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//' | cut -c1-48 | sed -E 's/-+$//'; }
num=$last; new=(); subjects=()
emit() {  # <from> <to> <slug> <subject>  (the diff goes to the file as git prints it: a binary hunk ends in a blank line)
  local f
  num=$((num + 1)); f=$(printf '%04d-%s.patch' "$num" "$3")
  {
    printf '# port-patches.sh: %s @ %s over %s @ %s (%s %s..%s)\n' "$branch" "${branch_sha:0:12}" "$base" "${base_sha:0:12}" "$mode" \
      "$(git -C "$engine" rev-parse --short=12 "$1")" "$(git -C "$engine" rev-parse --short=12 "$2")"
    printf '# %s\n' "$4"
    printf '# Applied by zig/Dockerfile with git apply after the hand-written patches; regenerate with scripts/port-patches.sh, do not edit.\n'
    git -C "$engine" diff "${diffopts[@]}" "$1" "$2" -- "${spec[@]}"
  } > "$tmp/out/$f"
  if ! grep -q '^diff --git ' "$tmp/out/$f"; then
    rm -f -- "$tmp/out/$f"; num=$((num - 1)); log "skip $(git -C "$engine" rev-parse --short=12 "$2") ($4): nothing outside the excluded paths"; return 0
  fi
  new+=("$f"); subjects+=("$4")
}
if [[ "$mode" == squash ]]; then
  emit "$base_sha" "$branch_sha" "$name" "$title"
else
  for c in $(git -C "$engine" rev-list --first-parent --reverse "$base_sha..$branch_sha"); do
    s=$(git -C "$engine" log -1 --format=%s "$c")
    emit "$c^1" "$c" "$(slug "$s")" "$s"
  done
fi
(( ${#new[@]} )) || die "$base..$branch changes nothing outside ${excludes[*]:-the excludes}: nothing to export"

# ---------------------------------------------------------------- 2. a clean TF_REF with the hand-written patches = BASE
log "Checking on a clean checkout of TF_REF ${TF_REF:0:12}"
if git -C "$engine" cat-file -e "$TF_REF^{commit}" 2>/dev/null; then
  git clone -q --no-checkout --shared "$engine" "$tmp/tf"
else
  log "TF_REF is not in $engine: fetching it from $TF_REPO"
  git init -q "$tmp/tf"; git -C "$tmp/tf" remote add origin "$TF_REPO"
  git -C "$tmp/tf" fetch -q --depth 1 origin "$TF_REF" || die "cannot fetch TF_REF from $TF_REPO"
  # BASE and BRANCH from the engine repository, for the comparisons
  git -C "$tmp/tf" fetch -q "$engine" "$base_sha" "$branch_sha" || die "cannot read $base and $branch from $engine"
fi
git -C "$tmp/tf" -c advice.detachedHead=false checkout -q --detach "$TF_REF"
[[ "$(git -C "$tmp/tf" rev-parse HEAD)" == "$TF_REF" ]] || die "the checkout is not at TF_REF"
apply() {  # <patch file> <label>: as zig/Dockerfile does (git apply --check, then git apply, in the checkout's root)
  ( cd "$tmp/tf" && git apply --check "$1" && git apply "$1" ) 2>"$tmp/apply.err" ||
    { sed 's/^/    /' "$tmp/apply.err" >&2; die "$2 does not apply on a clean TF_REF (git apply --check): nothing was written"; }
}
same_as() {  # <commit>: the checkout (untracked and ignored files too: -f) equals <commit> outside the excludes
  git -C "$tmp/tf" add -A -f >/dev/null
  git -C "$tmp/tf" diff --cached --quiet "$1" -- "${spec[@]}"
}
for p in "${hand[@]}"; do apply "$root/zig/patches/$p" "zig/patches/$p"; done
if ! same_as "$base_sha"; then
  git -C "$tmp/tf" diff --cached --stat "$base_sha" -- "${spec[@]}" | tail -15 >&2
  die "$base is not TF_REF plus zig/patches ${hand[*]} (the differences above): export relative to the base the image is built from"
fi
log "TF_REF + ${#hand[@]} hand-written patches = $base (${base_sha:0:12})"

# ---------------------------------------------------------------- 3. the new patches apply, and reproduce BRANCH
for p in "${new[@]}"; do apply "$tmp/out/$p" "$p"; log "  $p applies ($(grep -c '^diff --git ' "$tmp/out/$p") files, $(wc -c <"$tmp/out/$p" | tr -d ' ') bytes)"; done
if ! same_as "$branch_sha"; then
  git -C "$tmp/tf" diff --cached --stat "$branch_sha" -- "${spec[@]}" | tail -15 >&2
  die "the exported patches applied but do not reproduce $branch (above): nothing was written"
fi
log "TF_REF + all patches = $branch (${branch_sha:0:12}) outside ${excludes[*]:-nothing}"

# ---------------------------------------------------------------- 4. write (not with --dry-run)
readme="$root/zig/patches/README.md"
# the Default column follows the recipe's own defaults (scripts/port.sh): off unless a fork turns them on there
recipe_default() {
  local gib shard
  gib=$(sed -n 's/^PORT_SPILL_DEFAULT_GIB=\([0-9.]*\).*/\1/p' "$root/scripts/port.sh" | head -1)
  shard=$(sed -n 's/^PORT_SHARD_DEFAULT=\([01]\).*/\1/p' "$root/scripts/port.sh" | head -1)
  if [[ "${gib:-0}" =~ ^0*(\.0*)?$ && "${shard:-0}" == 0 ]]; then
    printf 'off (engine and recipe: `SPILL_GIB=0`, `KV_SHARD=0`, `POOL` empty; set them in `scripts/local.sh`)'
  else
    printf 'engine: off; **recipe: on** (`SPILL_GIB=%s`, `KV_SHARD=%s`, `POOL` derived; `0` turns each off)' "${gib:-0}" "${shard:-0}"
  fi
}
rows() {
  local i d; d=$(recipe_default)
  for i in "${!new[@]}"; do
    printf '| `%s` | the ported settings: `SPILL_GIB`, `KV_SHARD`, `POOL` ([docs/PORT-RECIPE.md](../../docs/PORT-RECIPE.md)) | %s | %s (engine `%s` @ `%s`, exported by `scripts/port-patches.sh`) <!-- port-patches --> |\n' \
      "${new[$i]}" "$d" "${subjects[$i]//|/\\|}" "$branch" "${branch_sha:0:12}"
  done
}
rows > "$tmp/rows"
newreadme "$readme" "$tmp/rows" > "$tmp/README.md"
grep -qE "$ROW_RE" "$tmp/README.md" || die "found no patch table in $readme to add the rows to"
if (( dry )); then
  log "DRY RUN: would remove ${old[*]:-nothing} and write ${new[*]} into zig/patches, and update its README:"
  diff -u "$readme" "$tmp/README.md" | sed 's/^/    /' || true
  exit 0
fi
for p in ${old[@]+"${old[@]}"}; do rm -f -- "$root/zig/patches/$p"; done
for p in "${new[@]}"; do cp -- "$tmp/out/$p" "$root/zig/patches/$p"; done
cp -- "$tmp/README.md" "$readme"
log "Wrote ${new[*]} (removed ${old[*]:-none}); zig/patches/README.md updated. The image tag changes: ./start.sh restart rebuilds it"
