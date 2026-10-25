#!/usr/bin/env bash
# Regression test for issue #42 (scripts/prepare.sh, ABLIT=1): takes the overlay's staging lines out of prepare.sh (from
# stock.new's mkdir up to the overlay's call) and runs them on a snapshot whose blobs are read-only, as huggingface_hub
# stores them (0444); then writes the index through a hard link of it, as the overlay's graft does when it rewrites the
# index in its destination. Run: bash scripts/test_ablit_index_writable.sh
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
(( EUID != 0 )) || { echo "skip: as root a read-only file opens for writing anyway"; exit 0; }
stage=$(awk '/mkdir -p "\$DATA_DIR\/stock\.new"/ {on=1} on && /apply_mia29_wo_b_graft\.py/ {exit} on {print}' "$here/prepare.sh")
[[ -n "$stage" ]] || { echo "FAIL: no stock.new staging in prepare.sh"; exit 1; }
fails=0; t=$(mktemp -d); trap 'chmod -R u+w "$t"; rm -rf -- "$t"' EXIT
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
mkdir -p "$t/blobs" "$t/snap" "$t/data/checkpoint.new"
printf '{"weight_map": {"a": "model-00001-of-00001.safetensors"}}' > "$t/blobs/index"
printf 'shard bytes' > "$t/blobs/shard"
chmod 0444 "$t/blobs/index" "$t/blobs/shard"
ln -s ../blobs/index "$t/snap/model.safetensors.index.json"
ln -s ../blobs/shard "$t/snap/model-00001-of-00001.safetensors"
SNAP=$t/snap; DATA_DIR=$t/data
die() { echo "die: $*"; return 1; }
eval "$stage"; rc=$?
check "the staging lines run" '[[ $rc == 0 ]]'
ln "$t/data/stock.new/model.safetensors.index.json" "$t/data/checkpoint.new/model.safetensors.index.json" 2>/dev/null
check "the overlay can rewrite the index through its hard link" \
  '{ printf "{\"weight_map\": {}}" > "$t/data/checkpoint.new/model.safetensors.index.json"; } 2>/dev/null'
check "the snapshot's index blob stays 0444, its bytes unchanged" \
  '[[ $(stat -c %a "$t/blobs/index") == 444 && $(<"$t/blobs/index") == *model-00001-of-00001* ]]'
check "the shards stay hard links of their blobs" \
  '[[ $(stat -c %i "$t/data/stock.new/model-00001-of-00001.safetensors") == $(stat -c %i "$t/blobs/shard") ]]'
exit $(( fails > 0 ))
