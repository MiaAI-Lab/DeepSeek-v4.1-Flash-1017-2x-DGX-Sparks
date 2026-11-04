# scripts/prepare.sh's steps 4 to 7 at TP=3 (three Sparks, experimental; docs/tp3.md), sourced by it after step 3 (the
# image from zig/Dockerfile.tp3, the checkpoint, Engram's tables and the ABLIT=1 overlay are ready). Each rank's files
# in LakyFx's layout, from the checkpoint alone, no GPU, every tool run in the image (its /opt/tp3: LakyFx's runtime
# tools, kit and RoPE reference):
#   4. the checkpoint's own files (hard links), their source manifest and LakyFx's receipt (source_verify.py re-hashes
#      every shard), and the plan (scripts/tp3_plan.py, checked against LakyFx's TF29 plan)
#   5. the kit with RoPE tables cut to POOL (LakyFx's make_rope.py at its 10M reference pool, then
#      scripts/tp3_inputs.py rope-cut) and the model folder (config for three ranks, tokenizer, the recipe's chat
#      template, the vision tower)
#   6. for each rank: its weight file (LakyFx's rank_cache.py, re-read in full by rank_verify.py) and its Engram rows
#      (LakyFx's prepare_engram.py: a sparse copy of its 8 of the 24 hash columns)
#   7. each worker's folder (WORKER_TP3_DIR): the kit, the model folder, its Engram rows and its weight file, copied
#      over the NFS volume and checked by sha256 there; rank 1's and 2's files then leave this Spark
# Rank 0's folder is $DATA_DIR/tp3 (TP3_SUB). Safe to re-run: every step skips what is done.
# Adapted from easthambuilds' scripts/prepare.sh (easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold,
# Apache-2.0), on this recipe's steps 1 to 3, NFS volume and copies.

T=$HF_CACHE/$TP3_SUB              # rank 0's folder and the work files (kit, model, rank-cache, engram, ...)
T_C=/hf/$TP3_SUB                  # the same in a container
LX=/opt/tp3                       # the image's LakyFx runtime (kit, reference, tools, lakyfx)
STOCK="$(model_cache_dir)/snapshots/$(snapshot_rev)"
STOCK_C="/hf/hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"
ESNAP_C=$(engram_in_container)
# the recipe's Python tools (scripts/) at /recipe, LakyFx's with -E -s (they import their neighbours)
tp3_py() { in_hf -v "$PWD/scripts:/recipe:ro" -v "$PWD/model:/recipe-model:ro" --entrypoint python3 "$IMAGE" "$@"; }
mkdir -p "$T"

# ---------------------------------------------------------------- 4. (TP=3) the checkpoint's receipt and plan
# LakyFx's tools refuse a file that resolves outside its folder, so the Hub snapshot's links into ../../blobs will not
# do: the stock checkpoint as a folder of hard links (no copy), the ABLIT=1 one as prepare.sh grafted it.
if [[ "$ABLIT" == 1 ]]; then
  CK=$DATA_DIR/checkpoint; CK_C=/hf/$DATA_SUB/checkpoint
  SRC_ID="$MODEL_ID+$ABLIT_ID"; SRC_REV="$(snapshot_rev)+$ABLIT_REVISION"   # a local checkpoint: on no Hub
else
  CK=$T/checkpoint; CK_C=$T_C/checkpoint
  SRC_ID=$MODEL_ID; SRC_REV=$(snapshot_rev)
  if [[ ! -f "$CK/.tp3-ready" ]]; then
    log "The checkpoint's files as hard links in $CK (no copy; LakyFx's tools refuse the snapshot's links)"
    rm -rf "$CK.new" && mkdir -p "$CK.new" && cp -alL "$STOCK/." "$CK.new/" && touch "$CK.new/.tp3-ready" ||
      die "could not link the checkpoint's files into $CK.new (HF_CACHE must be one file system)"
    rm -rf "$CK" && mv "$CK.new" "$CK"
  fi
fi
links=$(find "$CK" -maxdepth 1 -type l -name '*.safetensors' | head -3)
[[ -z "$links" ]] || die "$CK still holds symbolic links ($links): LakyFx's tools need the files themselves"
# the receipt: source_verify.py hashes every shard against the manifest and records each file's identity, which
# rank_cache.py checks again before and after it reads; anything that touches a shard since (a hard link made or
# removed changes its ctime) makes it stale, and it is made again
receipt_ok() {
  [[ -s "$T/receipt.json" ]] &&
    tp3_py -I /recipe/tp3_inputs.py receipt-ok --tools $LX/tools "$T_C/receipt.json" "$CK_C" "$SRC_ID" "$SRC_REV" >/dev/null
}
ensure_receipt() {
  receipt_ok && return 0
  rm -f "$T/manifest.json" "$T/receipt.json" "$T/receipt.json.tmp"
  log "Source manifest of $CK's shards (sizes and sha256$([[ "$ABLIT" == 1 ]] && echo "; the grafted ones are hashed here, never taken from the Hub"))"
  tp3_py -I /recipe/tp3_inputs.py manifest "$CK_C" "$SRC_ID" "$SRC_REV" "$T_C/manifest.json" --snapshot "$STOCK_C" ||
    die "could not write the source manifest of $CK"
  log "LakyFx's receipt: every shard re-hashed against it (source_verify.py; ~196 GiB read)"
  tp3_py -E -s $LX/tools/source_verify.py --manifest "$T_C/manifest.json" --root "$CK_C" --out "$T_C/receipt.json.tmp" \
    --weights-only >/dev/null || die "a shard of $CK does not match its manifest (source_verify.py)"
  mv "$T/receipt.json.tmp" "$T/receipt.json"
}
if [[ ! -s "$T/plan.json" ]]; then
  log "LakyFx's TP3 plan of the checkpoint (scripts/tp3_plan.py: every tensor the ranks read, checked against LakyFx's TF29 plan)"
  rm -f "$T/plan.json.tmp"
  tp3_py -I /recipe/tp3_plan.py "$CK_C" "$TP3_ARM" "$SRC_ID" "$SRC_REV" $LX/lakyfx/tensorfold/preparation/TF29.json \
    $LX/lakyfx/tensorfold/preparation "$T_C/plan.json.tmp" > "$T/plan.log" 2>&1 ||
    die "the checkpoint does not fit LakyFx's TP3 layout: $(grep -E 'NOT WRITTEN|rror|differ' "$T/plan.log" | head -3) ($T/plan.log)"
  mv "$T/plan.json.tmp" "$T/plan.json"
fi
psha=$(sha256sum "$T/plan.json" | cut -c1-64)
if [[ "$ABLIT" != 1 && "$MODEL_ID@$(snapshot_rev)" == "$TP3_PLAN_FOR" && -n "$TP3_PLAN_SHA256" ]]; then
  [[ "$psha" == "$TP3_PLAN_SHA256" ]] || die "the plan's sha256 is $psha, not the pinned $TP3_PLAN_SHA256 (scripts/config.sh): remove $T/plan.json and run this again; if it repeats, the checkpoint's headers differ from the published ones"
fi
p20=${psha:0:20}
log "Plan ${psha:0:16} of the checkpoint (scripts/tp3_plan.py's summary: $T/plan.log)"

# ---------------------------------------------------------------- 5. (TP=3) the kit with its RoPE tables, the model folder
rows=$(tp3_rope_rows)
if [[ "$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "$T/kit/rope.json" 2>/dev/null)" != "$rows" ]]; then
  if [[ ! -s "$T/rope-ref/rope.json" ]]; then
    log "LakyFx's RoPE tables at its 10,000,000-position reference pool (make_rope.py on CPU, checked against its qualified hashes; ~5 GB)"
    rm -rf "$T/rope-ref"; mkdir -p "$T/rope-ref"
    in_hf --entrypoint bash "$IMAGE" -c "mkdir -p $T_C/rope-ref/models/TF351 && cp $LX/kit/models/TF351/config.json $T_C/rope-ref/models/TF351/ &&
      python3 $LX/lakyfx/tensorfold/make_rope.py --reference $LX/reference --kit $T_C/rope-ref --pool 10000000" > "$T/rope.log" 2>&1 &&
      grep -q CPU_BYTES_MATCH_QUALIFIED_ROPE "$T/rope.log" || { rm -rf "$T/rope-ref"; die "make_rope.py did not reproduce LakyFx's qualified tables: $(tail -3 "$T/rope.log")"; }
  fi
  log "The kit, its RoPE tables cut to $rows rows (POOL $POOL)"
  rm -rf "$T/kit.new"
  in_hf -v "$PWD/scripts:/recipe:ro" --entrypoint bash "$IMAGE" -c "cp -r $LX/kit $T_C/kit.new && rm -f $T_C/kit.new/rope*.f32 $T_C/kit.new/rope.json &&
    python3 -I /recipe/tp3_inputs.py rope-cut $T_C/rope-ref $T_C/kit.new $POOL $LX/lakyfx/tensorfold/rope-hashes.json" ||
    die "the RoPE cut failed"
  rm -rf "$T/kit" && mv "$T/kit.new" "$T/kit"
fi
if [[ "$(cat "$T/model/.plan" 2>/dev/null)" != "$psha $(sha256sum model/chat_template.jinja | cut -c1-64)" ]]; then
  log "The model folder: config for three ranks, tokenizer, the recipe's chat template, the vision tower (every byte compared)"
  ensure_receipt
  rm -rf "$T/model.new"
  tp3_py -I /recipe/tp3_inputs.py model --tools $LX/tools "$CK_C" "$T_C/plan.json" "$T_C/receipt.json" /recipe-model/chat_template.jinja \
    "$(engram_rev)" "$T_C/model.new" "$TP3_ARM" > "$T/model.log" 2>&1 || die "the model folder: $(tail -3 "$T/model.log")"
  echo "$psha $(sha256sum model/chat_template.jinja | cut -c1-64)" > "$T/model.new/.plan"
  rm -rf "$T/model" && mv "$T/model.new" "$T/model"
fi
log "Kit and model folder ready ($(du -sh "$T/kit" | cut -f1), $(du -sh "$T/model" | cut -f1))"

# ---------------------------------------------------------------- 6 and 7. (TP=3) each rank's weight file and Engram rows
# rank_file <r>: the name of rank r's weight file for this plan (the engine finds the one rank<r>of3-*.bin)
rank_file() { echo "rank${1}of3-$TP3_ARM-$p20.bin"; }
# rank_export <r>: rank r's weight file into $T/out (rank_cache.py), re-read in full against its receipt (rank_verify.py)
rank_export() {
  local r=$1 name out_c status
  name=$(rank_file "$r"); out_c=$T_C/out/$name
  mkdir -p "$T/out"
  rm -f "$T/out/rank${r}of3-"*
  local free; free=$(df -B1 --output=avail "$T" | tail -1)
  (( free > 80 * 1024 ** 3 )) || die "$(( free / 1024 ** 3 )) GiB free in $T: rank $r's weight file needs ~67 GiB"
  ensure_receipt
  log "Rank $r's weight file (~67 GiB, LakyFx's rank_cache.py), then re-read in full (rank_verify.py)"
  local common=(--plan "$T_C/plan.json" --keys $LX/lakyfx/tensorfold/preparation/rank$r.keys --pollard "$CK_C" --mia "$CK_C")
  tp3_py -E -s $LX/tools/rank_cache.py "${common[@]}" --out "$out_c" --source-receipt "mia=$T_C/receipt.json" --write \
    > "$T/rank$r-export.log" 2>&1 || die "rank $r's weight file: $(tail -3 "$T/rank$r-export.log")"
  tp3_py -E -s $LX/tools/rank_verify.py "${common[@]}" --artifact "$out_c" --out "${out_c%.bin}.verified.json" --full-hash \
    --source-receipt "mia=$T_C/receipt.json" > "$T/rank$r-verify.log" 2>&1 || die "rank $r's weight file does not verify: $(tail -3 "$T/rank$r-verify.log")"
  status=$(python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["status"], d["full_sha256_checked"])' "$T/out/${name%.bin}.verified.json")
  [[ "$status" == "CPU_CHECKED_NOT_GPU_QUALIFIED True" ]] || die "rank $r's weight file: rank_verify.py says $status"
  rm -f "$T/out/${name%.bin}.partial"   # rank_cache.py's hard link of the file it wrote
}
rank_sha() { python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact_sha256"])' "$1"; }
# engram_cut <r> <folder under T>: rank r's Engram rows (LakyFx's prepare_engram.py) with their SOURCE.json and sha256 list
engram_cut() {
  local r=$1 sub=$2 pins=() f
  for f in "${shards[@]}"; do pins+=("$f=${want_sha[$f]:-}"); done
  rm -rf "$T/$sub" "$T/$sub.sha256"
  log "Rank $r's Engram rows: its 8 of the 24 hash columns of layers 1 and 14, a sparse copy (LakyFx's prepare_engram.py)"
  in_hf -v "$PWD/scripts:/recipe:ro" --entrypoint bash "$IMAGE" -c "python3 $LX/lakyfx/tensorfold/prepare_engram.py --source $ESNAP_C --kit $T_C/kit --rank $r --out $T_C/$sub --write &&
    python3 -I /recipe/tp3_inputs.py engram-source $T_C/$sub $r $ENGRAM_ID $(engram_rev) ${pins[*]}" > "$T/engram-r$r.log" 2>&1 ||
    { rm -rf "$T/$sub"; die "rank $r's Engram rows: $(tail -3 "$T/engram-r$r.log")"; }
  (cd "$T/$sub" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$T/$sub.sha256" || die "could not hash rank $r's Engram rows"
}

# rank 0: here, in $T/rank-cache and $T/engram
name=$(rank_file 0)
if [[ -f "$T/rank-cache/$name" && -f "$T/rank-cache/${name%.bin}.verified.json" ]]; then log "Weight file of rank 0: ready"
else
  rank_export 0
  mkdir -p "$T/rank-cache"
  find "$T/rank-cache" -maxdepth 1 -name 'rank0of3-*' -delete   # the engine takes exactly one rank0of3-*.bin
  mv "$T/out/$name" "$T/out/${name%.bin}.receipt.json" "$T/out/${name%.bin}.verified.json" "$T/rank-cache/"
fi
if [[ -s "$T/engram/SOURCE.json" ]]; then log "Engram rows of rank 0: ready"
else engram_cut 0 engram; fi

# each worker: its folder WORKER_TP3_DIR, filled from this Spark's export over the NFS volume (as at TP=2 its Engram
# tables), every file checked by sha256 on the worker
# wcopy <i> <path under T> <folder under WORKER_TP3_DIR>: a folder (sparse files kept sparse) or a file, replaced whole
wcopy() {
  local i=$1 src=$2 dst=$3
  worker "$i" "mkdir -p $WORKER_TP3_DIR && docker run --rm --network none --user \$(id -u):\$(id -g) -v '$NFS_VOLUME:/hf:ro' -v $WORKER_TP3_DIR:/out --entrypoint bash '$IMAGE' -c 'rm -rf /out/$dst.part && mkdir -p \$(dirname /out/$dst) && cp -r --sparse=always /hf/$TP3_SUB/$src /out/$dst.part && rm -rf /out/$dst && mv /out/$dst.part /out/$dst'" ||
    die "copying $src to $(worker_host "$i") failed"
}
# wcheck <i> <folder under WORKER_TP3_DIR>: the worker's copy against the sha256 list on stdin
wcheck() { worker "$1" "cd $WORKER_TP3_DIR/$2 && sha256sum -c --quiet --strict" >/dev/null; }
tree_sums() { (cd "$1" && find . -type f ! -name .plan -print0 | LC_ALL=C sort -z | xargs -0 sha256sum); }
kit_sums=$(tree_sums "$T/kit"); model_sums=$(tree_sums "$T/model")
for i in $(worker_ids); do
  h=$(worker_host "$i")
  ensure_nfs_volume "$i"
  for part in kit model; do
    sums=$kit_sums; [[ "$part" == kit ]] || sums=$model_sums
    if wcheck "$i" "$part" <<<"$sums" 2>/dev/null; then log "Worker $h: its $part: ready"; continue; fi
    log "Worker $h: copying the $part over NFS"
    wcopy "$i" "$part" "$part"
    wcheck "$i" "$part" <<<"$sums" || die "$h's copy of the $part differs from this Spark's (sha256)"
  done
  if worker "$i" "[ -s $WORKER_TP3_DIR/engram/SOURCE.json ]"; then log "Worker $h: Engram rows of rank $i: ready"
  else
    [[ -s "$T/out/engram-r$i/SOURCE.json" && -s "$T/out/engram-r$i.sha256" ]] || engram_cut "$i" "out/engram-r$i"
    free=$(worker "$i" "mkdir -p $WORKER_TP3_DIR && df -B1 --output=avail $WORKER_TP3_DIR | tail -1")
    used=$(du -sB1 "$T/out/engram-r$i" | cut -f1)
    (( free > used + 90 * 1024 ** 3 )) || die "$h has $(( free / 1024 ** 3 )) GiB free in WORKER_TP3_DIR: rank $i's Engram rows need $(( used / 1024 ** 3 )) GiB and its weight file ~67 GiB"
    log "Worker $h: copying rank $i's Engram rows over NFS ($(( used / 1024 ** 3 )) GiB on disk, sparse)"
    wcopy "$i" "out/engram-r$i" engram
    wcheck "$i" engram < "$T/out/engram-r$i.sha256" || die "$h's copy of rank $i's Engram rows differs from this Spark's (sha256)"
    rm -rf "$T/out/engram-r$i" "$T/out/engram-r$i.sha256"
  fi
  name=$(rank_file "$i")
  if worker "$i" "[ -f $WORKER_TP3_DIR/rank-cache/$name ] && [ -f $WORKER_TP3_DIR/rank-cache/${name%.bin}.verified.json ]"; then
    log "Worker $h: weight file of rank $i: ready"
  else
    [[ -f "$T/out/$name" && -f "$T/out/${name%.bin}.verified.json" ]] || rank_export "$i"
    free=$(worker "$i" "mkdir -p $WORKER_TP3_DIR && df -B1 --output=avail $WORKER_TP3_DIR | tail -1")
    (( free > 75 * 1024 ** 3 )) || die "$h has $(( free / 1024 ** 3 )) GiB free in WORKER_TP3_DIR: rank $i's weight file needs ~67 GiB"
    log "Worker $h: copying rank $i's weight file over NFS (~67 GiB)"
    worker "$i" "mkdir -p $WORKER_TP3_DIR/rank-cache && find $WORKER_TP3_DIR/rank-cache -maxdepth 1 -name 'rank${i}of3-*' -delete"
    mkdir -p "$T/out/rank$i" && mv "$T/out/$name" "$T/out/${name%.bin}.receipt.json" "$T/out/${name%.bin}.verified.json" "$T/out/rank$i/"
    wcopy "$i" "out/rank$i" rank-cache
    wcheck "$i" rank-cache <<<"$(rank_sha "$T/out/rank$i/${name%.bin}.receipt.json")  $name" ||
      die "$h's copy of rank $i's weight file differs from its receipt (sha256)"
    rm -rf "$T/out/rank$i"
  fi
done
rm -rf "$T/out"

write_prepared_marker || die "cannot record the prepared state (the previous marker is left as it was): see the error above"
log "Done: all three Sparks are ready for TP=3 (experimental; docs/tp3.md). Start the server with TP=3 ./start.sh (port $PORT)."
