#!/usr/bin/env bash
# Prepare two Sparks to serve Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw with TensorFold's native engine. The
# checkpoint, the kit and the served folder land on this Spark; the worker reads them over NFS:
#   1. preflight: docker and the GPU on both Sparks, key-based ssh, the link between them
#   2. the engine image on this Spark (zig/Dockerfile: BertholomusAI's TensorFold fork at TF_REF plus zig/patches, and
#      the inputs' tools of the fork's four-node branch at INPUTS_REF, on BASE_IMAGE), then on the worker
#      (docker save | docker load: only what it lacks is stored)
#   3. the checkpoint into HF_CACHE (~196 GiB, resumable), at its pinned revision, and Engram's two tables, the base
#      model's shards 47 and 48 (~203 GB) with its index and config, at theirs
#   4. the kernel kit: BertholomusAI's kit/ at KIT_REF, its RoPE tables and image routing bias made from the checkpoint,
#      every file checked against its MANIFEST (the image bias is 1017's own: the one file allowed to differ)
#   5. Engram's token map, rank 0's weight file (~106 GB, here) and the folder rank 0 serves (the checkpoint's files
#      linked, beside model/chat_template.jinja), from the checkpoint, no GPU
#   6. a read-only NFS volume of this Spark's HF_CACHE on the worker, checked: every file a rank reads, by size
#   7. on the worker's own disk: Engram's tables, and rank 1's weight file (~106 GB, written there from the NFS volume)
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch; PULL=0 never pulls a published image. VERIFY_SHA=1 also checks Engram's shards against the sha256 the
# checkpoint's model card lists (a few minutes of reading 203 GB).
# ABLIT=1 applies Keys' matching Mia EAQ 2.9 bpw overlay in a separate folder. Requires your Hugging Face read token
# (HF_TOKEN or hf auth login) AND access to https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
source ./scripts/nodes.sh

REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done
ablit_notice

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$STATE_DIR" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it or stop it: pgrep -af prepare.sh"
# Check gated access before building images or downloading hundreds of GB. hf uses HF_TOKEN or its saved login;
# never pass the token as an argument or forward it to the worker or serving containers.
if [[ "$ABLIT" == 1 ]]; then
  command -v hf >/dev/null || die "install the Hugging Face CLI: pip install -U huggingface_hub"
  log "Downloading the matching Mia 2.9 bpw overlay from $ABLIT_ID @ ${ABLIT_REVISION:0:8}"
  hf download "$ABLIT_ID" --revision "$ABLIT_REVISION" --cache-dir "$HF_CACHE/hub" \
    --include apply_mia29_wo_b_graft.py --include "$ABLIT_SIDECAR" --include EXL3_MIA29_ABLIT_META.json --include README.md >/dev/null || \
    die "cannot download the gated overlay: use your HF read token (HF_TOKEN or hf auth login) and obtain access at https://huggingface.co/$ABLIT_ID"
fi
log "Preflight checks on both Sparks"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
nvidia-smi -L >/dev/null 2>&1 || warn "nvidia-smi failed on this node: is the NVIDIA driver working?"
check_workers
need_worker 1
worker 1 'docker info >/dev/null 2>&1' || die "$(worker_host 1) cannot talk to its docker daemon (docker group?)"
worker 1 'nvidia-smi -L >/dev/null 2>&1' || warn "nvidia-smi failed on $(worker_host 1)"
detect_links
log "Link: this node $HEAD_ADDR ($HEAD_DEV, RoCE $HEAD_HCAS) <-> $(worker_host 1) $WORKER_ADDR ($WORKER_DEV, RoCE $WORKER_HCAS); NFS from $(nfs_server 1)"

# ---------------------------------------------------------------- 2. the image
hash=$(image_hash)
built=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
prebuilt=$(prebuilt_image)            # the pinned digest (config.sh's IMAGE_TAG / IMAGE_DIGEST), else the hash's tag
if [[ $REBUILD -eq 0 && "${PULL:-1}" == 1 && "$built" != "$hash" ]]; then
  log "Pulling the prebuilt image $prebuilt if there is one (PULL=0 builds instead)"
  if docker pull "$prebuilt" >/dev/null 2>&1 &&
     [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$prebuilt")" == "$hash" ]]; then
    docker tag "$prebuilt" "$IMAGE"; built=$hash
    log "Using $prebuilt as $IMAGE"
  else
    log "No prebuilt image for these pins ($prebuilt): building it locally"
  fi
fi
if [[ $REBUILD -eq 1 || "$built" != "$hash" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || { log "Pulling $BASE_IMAGE"; docker pull "$BASE_IMAGE" >/dev/null; }
  log "Building $IMAGE (TensorFold ${TF_REF:0:12} + inputs' tools ${INPUTS_REF:0:12}, $(compgen -G 'zig/patches/*.patch' | wc -l) patches, Zig $ZIG_VERSION; two Zig builds, ~6 min on a Spark)"
  ctx=$(mktemp -d "$STATE_DIR/build.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  mkdir -p "$ctx/zig"; cp -r zig/patches "$ctx/zig/patches"; cp zig/Dockerfile "$ctx/Dockerfile"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  docker build "${nocache[@]}" -t "$IMAGE" -f "$ctx/Dockerfile" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" --build-arg INPUTS_REF="$INPUTS_REF" \
    --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" --build-arg PATCHES_HASH="$hash" "$ctx"
  rm -rf -- "$ctx"; trap - EXIT
else
  log "Image $IMAGE already built (hash $hash; --rebuild forces it)"
fi
ident=$(image_ident "$IMAGE")
if [[ "$(worker_image_ident 1 "$IMAGE")" != "$ident" ]]; then
  if [[ "${PULL:-1}" == 1 ]] && docker image inspect "$prebuilt" >/dev/null 2>&1 && worker 1 docker pull "$prebuilt" >/dev/null 2>&1 &&
     [[ "$(worker_image_ident 1 "$prebuilt")" == "$ident" ]]; then
    worker 1 docker tag "$prebuilt" "$IMAGE"
    log "Using $prebuilt as $IMAGE on $(worker_host 1)"
  else
    log "Copying $IMAGE to $(worker_host 1) (docker save | docker load; only missing layers are stored)"
    docker save "$IMAGE" | worker 1 docker load >/dev/null
  fi
  [[ "$(worker_image_ident 1 "$IMAGE")" == "$ident" ]] || die "$(worker_host 1)'s $IMAGE differs from this Spark's"
fi
log "Image $IMAGE identical on both Sparks"

# ---------------------------------------------------------------- 3. the checkpoint and Engram's tables
command -v hf >/dev/null || die "the Hugging Face CLI ('hf') is not installed on this Spark: pip install -U huggingface_hub"
avail=$(df -B1 --output=avail "$HF_CACHE" | tail -1)
# A checkpoint that is not in the cache under MODEL_ID's name and revision is downloaded whole (196 GiB). The cache is keyed by the
# repo's name, so a copy downloaded under another name (the repo was renamed from ...-1017-EAQ-v1-2.9-bpw to ...-1017-EXL3-EAQ-v1-2.9-bpw) or
# another revision does not count: refuse here when the disk cannot hold it, instead of filling the disk half way (README, Status)
if [[ ! -f "$(model_cache_dir)/snapshots/$(snapshot_rev)/model.safetensors.index.json" ]]; then
  (( avail > 215 * 1024 ** 3 )) || die "$MODEL_ID @ $(snapshot_rev | cut -c1-8) is not in $HF_CACHE/hub yet and only $(( avail / 1024 ** 3 )) GiB are free there; it is 196 GiB. If you already have this checkpoint under another repo name or revision, set MODEL_ID and MODEL_REVISION in scripts/local.sh to what $HF_CACHE/hub holds (scripts/local.sh.example); otherwise free the disk"
  (( avail > 420 * 1024 ** 3 )) ||
    warn "$(( avail / 1024 ** 3 )) GiB free in $HF_CACHE: the checkpoint (196 GiB), Engram's tables (203 GB) and rank 0's weight file (~106 GB) need ~505 GB"
fi
log "Downloading $MODEL_ID${MODEL_REVISION:+ @ ${MODEL_REVISION:0:8}} into $HF_CACHE/hub (196 GiB the first time; resumes)"
if ! hf download "$MODEL_ID" ${MODEL_REVISION:+--revision "$MODEL_REVISION"} --cache-dir "$HF_CACHE/hub" >/dev/null; then
  [[ -n "$MODEL_REVISION" && -f "$(model_cache_dir)/snapshots/$MODEL_REVISION/model.safetensors.index.json" ]] ||
    die "$MODEL_ID: the download failed"
  warn "$MODEL_ID: could not reach Hugging Face; using the snapshot already here"
fi
SNAP="$(model_cache_dir)/snapshots/$(snapshot_rev)"
[[ -f "$SNAP/model.safetensors.index.json" ]] || die "$MODEL_ID: no complete snapshot at $SNAP"
log "Checkpoint: $SNAP ($(du -shL "$SNAP" | cut -f1))"
shards=($ENGRAM_SHARDS)
inc=(--include model.safetensors.index.json --include config.json); for f in "${shards[@]}"; do inc+=(--include "$f"); done
log "Downloading Engram's tables: ${ENGRAM_SHARDS// /, } of $ENGRAM_ID${ENGRAM_REVISION:+ @ ${ENGRAM_REVISION:0:8}} (~203 GB the first time; resumes)"
if ! hf download "$ENGRAM_ID" ${ENGRAM_REVISION:+--revision "$ENGRAM_REVISION"} --cache-dir "$HF_CACHE/hub" "${inc[@]}" >/dev/null; then
  [[ -n "$ENGRAM_REVISION" ]] || die "$ENGRAM_ID: the download failed"
  warn "$ENGRAM_ID: could not reach Hugging Face; using the files already here"
fi
ESNAP="$(engram_cache_dir)/snapshots/$(engram_rev)"
# the 1017 tables are new: the shards of the older release must not be used (the card lists these sizes and sha256)
declare -A want_size=([model-00047-of-00048.safetensors]=101535150936 [model-00048-of-00048.safetensors]=101537926640)
declare -A want_sha=([model-00047-of-00048.safetensors]=76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5
                     [model-00048-of-00048.safetensors]=9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd)
for f in model.safetensors.index.json config.json "${shards[@]}"; do [[ -s "$ESNAP/$f" ]] || die "$ENGRAM_ID: $ESNAP/$f is missing"; done
if [[ "$ENGRAM_ID" == deepseek-ai/DeepSeek-V4.1-Flash-1017 ]]; then
  for f in "${shards[@]}"; do
    [[ -n "${want_size[$f]:-}" ]] || continue
    [[ "$(stat -L -c %s "$ESNAP/$f")" == "${want_size[$f]}" ]] ||
      die "$f is $(stat -L -c %s "$ESNAP/$f") bytes, not the 1017 release's ${want_size[$f]}: remove $ESNAP and download again"
    if [[ "${VERIFY_SHA:-0}" == 1 ]]; then
      log "sha256 of $f"
      [[ "$(sha256sum "$ESNAP/$f" | cut -d' ' -f1)" == "${want_sha[$f]}" ]] || die "$f does not match the model card's sha256"
    fi
  done
fi
log "Engram: $ESNAP ($(du -shL "$ESNAP" | cut -f1))"
# a container's view of HF_CACHE (the snapshot's files are links into ../../blobs)
in_hf() { docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HF_CACHE:/hf" "$@"; }

# Apply before either rank's cache is built: the index hash alone cannot distinguish changed tensor contents.
if [[ "$ABLIT" == 1 ]]; then
  overlay_sub="hub/models--${ABLIT_ID//\//--}/snapshots/$ABLIT_REVISION"
  for f in apply_mia29_wo_b_graft.py "$ABLIT_SIDECAR" EXL3_MIA29_ABLIT_META.json; do
    [[ -s "$HF_CACHE/$overlay_sub/$f" ]] || die "matching Mia overlay file is missing: $f"
  done
  # The overlay script casts four MTP tensors to F32 for another engine; this one reads the drafter's confidence head
  # as F16 whatever the file holds, so an F32 copy makes it draft almost nothing. Give them their stock dtypes back
  # (scripts/ablit_stock_dtypes.py; headers only when they already match).
  stock_dtypes() {
    in_hf -v "$PWD/scripts/ablit_stock_dtypes.py:/ablit_stock_dtypes.py:ro" --entrypoint python3 "$IMAGE" \
      /ablit_stock_dtypes.py "/hf/hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)" "$1"
  }
  mkdir -p "$DATA_DIR"
  if [[ ! -f "$DATA_DIR/checkpoint/.ablit-ready" ]]; then
    log "Applying the Mia EAQ 2.9 bpw overlay in $DATA_DIR/checkpoint (stock stays intact; changed shards need extra disk)"
    rm -rf "$DATA_DIR/checkpoint.new" "$DATA_DIR/stock.new"
    mkdir -p "$DATA_DIR/stock.new"
    # Dereference HF's relative blob symlinks before upstream hardlinks unchanged shards into the destination.
    cp -alL "$SNAP/." "$DATA_DIR/stock.new/"
    in_hf --entrypoint python3 "$IMAGE" "/hf/$overlay_sub/apply_mia29_wo_b_graft.py" \
      --src "/hf/$DATA_SUB/stock.new" --dst "/hf/$DATA_SUB/checkpoint.new" \
      --wo-b "/hf/$overlay_sub/$ABLIT_SIDECAR" || die "applying the matching Mia overlay failed; rerun prepare.sh to retry"
    python3 -I -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["n_edited"] == 104, "expected 104 Mia overlay tensors"' \
      "$DATA_DIR/checkpoint.new/ABLIT_META.json" || die "Mia overlay did not replace all 104 tensors"
    stock_dtypes "/hf/$DATA_SUB/checkpoint.new" || die "could not give the overlay checkpoint its stock dtypes back"
    touch "$DATA_DIR/checkpoint.new/.ablit-ready"
    rm -rf "$DATA_DIR/checkpoint"
    mv "$DATA_DIR/checkpoint.new" "$DATA_DIR/checkpoint"
    rm -rf "$DATA_DIR/stock.new"
  else
    log "Matching Mia overlay checkpoint: ready"
    # prepared before the dtype step: a tensor given back changes the weight files' index, so step 5 and 7 rebuild them
    stock_dtypes "/hf/$DATA_SUB/checkpoint" || die "could not give the overlay checkpoint its stock dtypes back"
  fi
  SNAP="$DATA_DIR/checkpoint"
fi

# ---------------------------------------------------------------- 4. the kit
KIT_DIR="$HF_CACHE/$KIT_SUB"
if [[ ! -d "$KIT_DIR/.git" ]]; then
  log "Fetching the kernel kit: $KIT_REPO @ ${KIT_REF:0:8}"
  rm -rf "$KIT_DIR"; git init -q "$KIT_DIR"
  git -C "$KIT_DIR" remote add origin "$KIT_REPO"
  git -C "$KIT_DIR" fetch -q --depth 1 origin "$KIT_REF"
  git -C "$KIT_DIR" checkout -q --detach FETCH_HEAD
fi
[[ "$(git -C "$KIT_DIR" rev-parse HEAD)" == "$KIT_REF" ]] || die "$KIT_DIR is not at KIT_REF ${KIT_REF:0:8}: remove it and run this again"
KIT_IN=/hf/$KIT_SUB/kit
SNAP_C=$(model_in_container)
TOOLS=/opt/tensorfold/tools/inputs
if [[ ! -s "$KIT_DIR/kit/rope-plain-cos.f32" || ! -s "$KIT_DIR/kit/rope-compressed-sin.f32" ]]; then
  log "Making the kit's RoPE tables from the checkpoint's config (CPU, PyTorch of the image)"
  in_hf --entrypoint python3 "$IMAGE" $TOOLS/make_rope.py "$SNAP_C/config.json" "$KIT_IN" >/dev/null ||
    die "the RoPE tables do not match the kit's rope.json: the checkpoint's config differs from the one the kit was made for"
fi
if [[ ! -s "$KIT_DIR/kit/vision/gate_bias_vl.safetensors" ]]; then
  log "Making the image routing bias (vision/gate_bias_vl.safetensors) from the checkpoint's gate.bias_vl tensors"
  in_hf --entrypoint python3 "$IMAGE" $TOOLS/make_bias_vl.py "$SNAP_C" "$KIT_IN/vision/gate_bias_vl.safetensors" >/dev/null ||
    die "could not make the image routing bias: does $MODEL_ID hold the gates' bias_vl tensors?"
fi
# every file MANIFEST lists, by sha256; the image bias is 1017's own, so it alone may differ from the release the kit was made for
bad=(); n=0
while read -r sum path; do
  n=$((n + 1))
  if [[ ! -f "$KIT_DIR/kit/$path" ]]; then bad+=("missing: $path"); continue; fi
  [[ "$(sha256sum "$KIT_DIR/kit/$path" | cut -d' ' -f1)" == "$sum" ]] || bad+=("differs: $path")
done < "$KIT_DIR/kit/MANIFEST"
for b in "${bad[@]}"; do [[ "$b" == "differs: vision/gate_bias_vl.safetensors" ]] || die "the kit does not check out ($b): $KIT_DIR/kit/MANIFEST"; done
log "Kit: $((n - ${#bad[@]})) of $n files match MANIFEST$( (( ${#bad[@]} )) && echo "; the image bias is 1017's own")"

# ---------------------------------------------------------------- 5. token map, rank 0's weight file, the served folder
mkdir -p "$DATA_DIR/rank-cache"
if [[ ! -s "$DATA_DIR/dsv41_token_map.json" ]]; then
  log "Engram's token map from the checkpoint's tokenizer"
  in_hf --entrypoint python3 "$IMAGE" $TOOLS/make_token_map.py "$SNAP_C" "/hf/$DATA_SUB/dsv41_token_map.json" >/dev/null
fi
tm=$(sha256sum "$DATA_DIR/dsv41_token_map.json" | cut -d' ' -f1)
if [[ "$tm" == "$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["token_map_sha256"])' "$KIT_DIR/kit/engram.json")" ]]; then
  log "Token map: the same bytes the kit's Engram fixtures were recorded with"
else
  warn "the token map ($tm) differs from the kit's recorded one: the 1017 tokenizer is not the one the kit was recorded with (the map has the config's compressed-vocabulary size, which make_token_map.py checks)"
fi
# the weight file the engine looks for: its name carries the index's hash (tf-dsv41-rank-cache --index-only)
want=$(in_hf --entrypoint bash "$IMAGE" -c "tf-dsv41-rank-cache $SNAP_C /tmp/x 0 2 --index-only | sha256sum | cut -c1-20")
if [[ -f "$DATA_DIR/rank-cache/rank0of2-$want.bin" ]]; then log "Weight file of rank 0: ready"
else
  rm -f "$DATA_DIR/rank-cache/rank0of2-"*.bin "$DATA_DIR/rank-cache/.rank0of2.partial"
  log "Weight file of rank 0 from the checkpoint (~106 GB, a few minutes)"
  in_hf --entrypoint tf-dsv41-rank-cache "$IMAGE" "$SNAP_C" "/hf/$DATA_SUB/rank-cache" 0 2 >/dev/null
fi
# the served folder: every file of the snapshot as a relative link (it resolves under /hf in a container too), and the
# recipe's chat template
VIEW="$DATA_DIR/model"
rm -rf "$VIEW.new" && mkdir -p "$VIEW.new"
for f in "$SNAP"/*; do ln -s "$(realpath -s --relative-to="$VIEW.new" "$f")" "$VIEW.new/"; done
# Never copy through a snapshot's chat-template symlink into stock or grafted weights.
rm -f "$VIEW.new/chat_template.jinja"
cp model/chat_template.jinja "$VIEW.new/chat_template.jinja"
rm -rf "$VIEW" && mv "$VIEW.new" "$VIEW"
[[ -s "$VIEW/config.json" ]] || die "the served folder $VIEW does not reach the checkpoint"
log "Served folder: $VIEW (the checkpoint linked, the recipe's chat template)"

# ---------------------------------------------------------------- 6. the worker's NFS volume
manifest() {  # the files a rank reads, with their sizes, as paths under HF_CACHE
  (cd "$HF_CACHE" && find -L "hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)" \
     "hub/models--${ENGRAM_ID//\//--}/snapshots/$(engram_rev)" "$DATA_SUB" "$KIT_SUB/kit" \
     -path '*/.git' -prune -o -type f -printf '%p %s\n' | LC_ALL=C sort)
}
want=$(manifest)
ensure_nfs_volume 1
if ! out=$(worker_nfs 1 true 2>&1); then
  die "$(worker_host 1) cannot mount this Spark's :$NFS_PATH from $(nfs_server 1) over NFS ($NFS_VOLUME): $(tail -1 <<<"$out")
    This Spark must export HF_CACHE (as $NFS_PATH) read-only to that worker's address on its link (README: NFS)"
fi
have=$(worker 1 "docker run --rm --network none -v '$NFS_VOLUME:/hf:ro' -w /hf --entrypoint find '$BASE_IMAGE' -L 'hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)' 'hub/models--${ENGRAM_ID//\//--}/snapshots/$(engram_rev)' '$DATA_SUB' '$KIT_SUB/kit' -path '*/.git' -prune -o -type f -printf '%p %s\n'" 2>/dev/null | LC_ALL=C sort || true)
[[ "$have" == "$want" ]] || { diff <(printf '%s\n' "$want") <(printf '%s\n' "$have") | head -10 >&2
  die "$(worker_host 1) does not see every file over NFS ($NFS_VOLUME)"; }
log "Worker ($(worker_host 1)) reads the checkpoint, the kit and rank 0's files over NFS"

# ---------------------------------------------------------------- 7. the worker's own disk: Engram's tables, rank 1's weight file
SNAP_REL="hub/models--${ENGRAM_ID//\//--}/snapshots/$(engram_rev)"
need=0; for f in "${shards[@]}"; do need=$(( need + $(stat -L -c %s "$ESNAP/$f") )); done
have=$(worker 1 "d=$WORKER_ENGRAM_DIR; s=0; for f in ${shards[*]}; do [[ -f \$d/\$f ]] && s=\$(( s + \$(stat -L -c %s \$d/\$f) )); done; echo \$s")
if [[ "$have" == "$need" ]]; then log "Worker: Engram tables on its disk"
else
  free=$(worker 1 "mkdir -p $WORKER_ENGRAM_DIR && df -B1 --output=avail $WORKER_ENGRAM_DIR | tail -1")
  # the ~140 GiB for rank 1's weight file is only needed while that file is not on its disk yet
  spare=150; worker 1 "ls $WORKER_CACHE_DIR/rank1of2-*.bin >/dev/null 2>&1" && spare=10
  (( free + have > need + spare * 1024 ** 3 )) || die "$(worker_host 1) has $(( free / 1024 ** 3 )) GiB free: Engram's tables need $(( need / 1024 ** 3 )) GiB on its disk (WORKER_ENGRAM_DIR), its weight file ~106 GB (unless it is already there), and $spare GiB to spare"
  log "Worker: copying Engram's tables to its disk ($(( need / 1024 ** 3 )) GiB over NFS, once)"
  worker 1 "docker run --rm --network none --user \$(id -u):\$(id -g) -v '$NFS_VOLUME:/hf:ro' -v $WORKER_ENGRAM_DIR:/out --entrypoint bash '$BASE_IMAGE' -c 'for f in model.safetensors.index.json config.json ${shards[*]}; do cp /hf/$SNAP_REL/\$f /out/\$f.part && mv /out/\$f.part /out/\$f || exit 1; done'" ||
    die "copying Engram's tables to $(worker_host 1) failed"
fi
wwant=$(worker 1 "docker run --rm --network none --user \$(id -u):\$(id -g) -e HOME=/tmp -v '$NFS_VOLUME:/hf:ro' --entrypoint bash '$IMAGE' -c 'tf-dsv41-rank-cache $SNAP_C /tmp/x 1 2 --index-only | sha256sum | cut -c1-20'")
if worker 1 "[[ -f $WORKER_CACHE_DIR/rank1of2-$wwant.bin ]]"; then log "Worker: weight file of rank 1 ready"
else
  worker 1 "mkdir -p $WORKER_CACHE_DIR && rm -f $WORKER_CACHE_DIR/rank1of2-*.bin $WORKER_CACHE_DIR/.rank1of2.partial"
  free=$(worker 1 "df -B1 --output=avail $WORKER_CACHE_DIR | tail -1")
  (( free > 130 * 1024 ** 3 )) || die "$(worker_host 1) has $(( free / 1024 ** 3 )) GiB free for rank 1's weight file (WORKER_CACHE_DIR): it needs ~106 GB"
  log "Worker: weight file of rank 1 from the checkpoint over NFS (~106 GB, a few minutes)"
  worker 1 "docker run --rm --network none --user \$(id -u):\$(id -g) -e HOME=/tmp -v '$NFS_VOLUME:/hf:ro' -v $WORKER_CACHE_DIR:/out --entrypoint tf-dsv41-rank-cache '$IMAGE' '$SNAP_C' /out 1 2" >/dev/null ||
    die "writing rank 1's weight file on $(worker_host 1) failed"
fi

prepared_state > "$PREPARED_MARKER"
log "Done: both Sparks are ready. Start the server with ./start.sh (port $PORT)."
