# The two Sparks: this machine (rank 0) and WORKER (rank 1, over ssh). The link between them is found from the routes and
# subnets between them (detect_link): one direct CX7 cable, its PCIe twin as a second rail when it reaches the peer.
# TP=3 (experimental, docs/tp3.md): WORKER2 is rank 2, and the three are a triangle of direct cables, paired by subnet
# (detect_triangle: the GLM-5.3 3x recipe's node_inventory and pair_links, with LakyFx's four ordered rails a rank).
# Sourced by start.sh, stop.sh and prepare.sh after config.sh.
# From the node helpers of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold (Mia's AI Lab), cut to two Sparks.

# wvar <NAME> <i>: the name of worker i's own setting: NAME for worker 1 (WORKER, FABRIC_PEER, NFS_SERVER), NAME<i>
# for the others (WORKER2, FABRIC_PEER2, NFS_SERVER2). The helpers keep the GLM scripts' per-worker form: every step
# that reaches a worker goes through worker_ids, which is the one worker at TP=2.
wvar() { if (( $2 == 1 )); then echo "$1"; else echo "$1$2"; fi; }
wval() { local v; v=$(wvar "$1" "$2"); echo "${!v:-}"; }
worker_host() { wval WORKER "$1"; }
# The workers this start uses (1 .. TP-1), and every worker configured at all (stop.sh stops them all).
worker_ids() { seq 1 $((TP - 1)); }
configured_workers() { local i; for i in 1 2 3; do [[ -z "$(worker_host "$i")" ]] || echo "$i"; done; }

# check_workers: this recipe serves two Sparks (the dsv41 family's TP=2) or, experimental, three (TP=3, LakyFx's
# engine), and WORKER .. WORKER<TP-1> are set and distinct (a WORKER2 set at TP=2 is left alone).
check_workers() {
  local i j h what
  [[ "$TP" == 2 || "$TP" == 3 ]] || die "TP is 2 (two Sparks) or 3 (three Sparks, experimental: docs/tp3.md), not $TP"
  for i in $(worker_ids); do
    h=$(worker_host "$i"); what="the second Spark"; (( i == 1 )) || what="rank $i's Spark"
    [[ -n "$h" ]] || die "$(wvar WORKER "$i") is not set: put $(wvar WORKER "$i")=user@<address of $what> in scripts/local.sh (see scripts/local.sh.example)"
    for (( j = 1; j < i; j++ )); do
      [[ "$h" != "$(worker_host "$j")" ]] || die "$(wvar WORKER "$i") and $(wvar WORKER "$j") are the same node ($h)"
    done
  done
}

# DRY_RUN=1 goes on without workers it cannot reach (a fictitious WORKER): its values print as <WORKER:...>.
declare -A WORKER_DOWN=()
# worker <i> <command...>: run a command on worker i (rank i). Key-based ssh only, no prompts.
worker() {
  local i=$1; shift
  [[ -z "${WORKER_DOWN[$i]:-}" ]] || return 255
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$(worker_host "$i")" "$@"
}

# save_log <dir> <rank> <container> <keep>: the container's log (stdout and stderr, with timestamps) gzipped into
# <dir>/<date>-<time>-rank<N>.log.gz before the container is removed (docker rm deletes it), keeping the newest <keep>
# of that rank; prints the file. Runs on the worker too (its definition is sent over ssh): no config.sh there.
save_log() {
  local dir=$1 rank=$2 c=$3 keep=$4 stamp f i=0
  [[ "$keep" =~ ^[0-9]+$ ]] && (( keep > 0 )) || return 0
  docker inspect "$c" >/dev/null 2>&1 || return 0
  mkdir -p "$dir" || return 1
  stamp=$(date +%Y%m%d-%H%M%S); f="$dir/$stamp-rank$rank.log.gz"
  while [[ -e "$f" ]]; do f="$dir/$stamp.$((++i))-rank$rank.log.gz"; done   # same second: still sorts as newer
  if ! docker logs --timestamps "$c" 2>&1 | gzip > "$f"; then rm -f "$f"; return 1; fi
  ls -1 "$dir"/*-rank"$rank".log.gz 2>/dev/null | { grep -vxF "$f" || true; } | LC_ALL=C sort -r | tail -n +"$keep" | xargs -r rm -f --
  echo "$f"
}
# worker_save_log <i>: the same, for rank i on worker i (into ~/.cache/dsv41-tensorfold/logs there)
worker_save_log() {
  worker "$1" "$(declare -f save_log); save_log \"\$HOME/.cache/dsv41-tensorfold/logs\" $1 '$CONTAINER_NAME' '$LOG_KEEP'"
}

need_worker() {
  local i=$1 v h
  v=$(wvar WORKER "$i"); h=$(worker_host "$i")
  [[ -n "$h" ]] || die "$v is not set: put $v=user@<worker address> in scripts/local.sh (see scripts/local.sh.example)"
  worker "$i" true 2>/dev/null && return 0
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    warn "DRY_RUN: cannot ssh to $h ($v): its values print as <$v:...>"
    WORKER_DOWN[$i]=1
    return 0
  fi
  die "cannot ssh to $h without a password: set up key-based ssh (ssh-copy-id $h)"
}
need_workers() { local i; for i in $(worker_ids); do need_worker "$i"; done; }

# Worker i's Hugging Face cache: its own HF_HOME, else ~/.cache/huggingface there. prepare.sh copies the checkpoint
# into it and start.sh mounts it into rank i. WORKER_HF_CACHE (worker 1) / WORKER_HF_CACHE<i> wins: a worker whose
# cache is not its HF_HOME (e.g. a shared models disk).
worker_hf_cache() {
  local c; c=$(wval WORKER_HF_CACHE "$1")
  if [[ -n "$c" ]]; then echo "$c"; return; fi
  [[ -z "${WORKER_DOWN[$1]:-}" ]] || { echo "<$(wvar WORKER "$1"):hf-cache>"; return 0; }
  worker "$1" 'echo "${HF_HOME:-$HOME/.cache/huggingface}"'
}

# Weights over NFS: rank i mounts the head's HF_CACHE read-only through the docker volume NFS_VOLUME on worker i, from
# nfs_server i: NFS_SERVER (worker 1) / NFS_SERVER<i>, else the head's address on that worker's link (detect_links).
# ensure_nfs_volume i creates it (or checks the one there names the same export); worker_nfs i <cmd...> runs a command
# in a throwaway container on worker i with the volume at /hf.
nfs_server() { local s; s=$(wval NFS_SERVER "$1"); echo "${s:-${LINK_HEAD_ADDR[$1]:-}}"; }
ensure_nfs_volume() {
  local i=$1 server have h
  server=$(nfs_server "$i"); h=$(worker_host "$i")
  [[ -n "$server" ]] || die "no NFS server address for $h: set $(wvar NFS_SERVER "$i") to the head's address on that worker's link"
  have=$(worker "$i" "docker volume inspect -f '{{index .Options \"device\"}} {{index .Options \"o\"}}' '$NFS_VOLUME'" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    worker "$i" "docker volume create --driver local --opt type=nfs --opt device=':$NFS_PATH' \
      --opt o='addr=$server,nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600' '$NFS_VOLUME'" >/dev/null ||
      die "could not create the docker volume $NFS_VOLUME on $h"
  elif [[ "${have%% *}" != ":$NFS_PATH" || "$have" != *"addr=$server,"* ]]; then
    die "$h's docker volume $NFS_VOLUME is ${have%% *} from ${have#* }, not :$NFS_PATH from $server: remove it there (docker volume rm $NFS_VOLUME) or set NFS_VOLUME / NFS_PATH / $(wvar NFS_SERVER "$i")"
  fi
}
worker_nfs() { local i=$1; shift; worker "$i" "docker run --rm --entrypoint '$1' -v '$NFS_VOLUME:/hf:ro' '$IMAGE' $(printf '%q ' "${@:2}")"; }
# Check the worker's actual file view, not just a successful mount or a prepared marker.
# Also works with the documented PREPARE=0 / host-path NFS_VOLUME setup: do not create a volume here.
worker_files_check() {
  local i=$1 out
  (( ${TP:-2} == 2 )) || { tp3_files_check "$i"; return; }   # TP=3: no rank reads over NFS; its own folder instead
  if out=$(worker_nfs "$i" sh -ec '
    for f in "$@"; do
      if [ ! -f "$f" ] || [ ! -r "$f" ] || [ ! -s "$f" ]; then
        echo "Missing, empty or unreadable on worker: $f" >&2
        exit 1
      fi
    done
  ' sh "$(model_in_container)/config.json" "/hf/$KIT_SUB/kit/MANIFEST" "/hf/$DATA_SUB/dsv41_token_map.json" 2>&1); then
    return 0
  fi
  printf '%s\n' "$out" >&2
  die "$(worker_host "$i") cannot read the model/kit through $NFS_VOLUME (NFS :$NFS_PATH from $(nfs_server "$i")). Check the actual NFSv4 root, stale file handles and competing host/container NFS servers (README: NFS); for a host-path mount check that worker's local files. No ranks were launched by this check"
}
# tp3_files_check <r>: at TP=3, rank r's own folder (rank 0's under HF_CACHE, worker r's WORKER_TP3_DIR) holds what the
# rank reads: the kit's RoPE tables, kernels and token map, the model folder with its vision tower, its Engram rows and
# exactly one weight file of rank r (the engine refuses two: SeveralRankCaches). prepare.sh put them there, by sha256.
# find, not a glob: a worker's login shell may be zsh, which fails a command whose glob matches nothing.
TP3_FILES_CHECK='cd "$1" || exit 1
for f in kit/rope.json kit/aot/aot.json kit/engram-token-map.json model/config.json model/vision.safetensors engram/SOURCE.json; do
  [ -s "$f" ] || { echo "$1/$f is missing or empty"; exit 1; }
done
n=$(find rank-cache -maxdepth 1 -type f -name "rank$2of3-*.bin" 2>/dev/null | wc -l)
[ "$n" -eq 1 ] || { echo "$1/rank-cache holds $n weight files of rank $2 (rank$2of3-*.bin), not one"; exit 1; }'
tp3_files_check() {
  local r=$1 out where
  if (( r == 0 )); then
    out=$(sh -c "$TP3_FILES_CHECK" sh "$HF_CACHE/$TP3_SUB" 0 2>&1) && return 0
    where="this Spark"
  else
    out=$(worker "$r" "sh -c $(printf '%q' "$TP3_FILES_CHECK") sh $WORKER_TP3_DIR $r" 2>&1) && return 0
    where=$(worker_host "$r")
  fi
  die "$where is not ready for rank $r at TP=3: ${out:-it cannot be reached}. Run scripts/prepare.sh with TP=3. No ranks were launched by this check"
}

# link_info <peer address>: this node's side of the link to <peer>: "<address> <netdev> <rdma device> <gid index>".
# The netdev and source address come from the route to the peer; the RDMA device from sysfs; the GID index is the
# RoCE v2 entry that holds this node's IPv4 address (an all-zero or v1 entry makes NCCL fail about a minute in).
link_info() {
  local peer=$1 route dev src hca idx type gid want
  route=$(ip -o -4 route get "$peer" 2>/dev/null) || return 1
  dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  src=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  hca=$(ls /sys/class/net/"$dev"/device/infiniband 2>/dev/null | head -1)
  idx=""
  if [[ -n "$hca" ]]; then
    want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"$src"))
    for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      gid=$(cat "$f" 2>/dev/null) || continue
      [[ "$gid" == "$want" ]] || continue
      type=$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)
      [[ "$type" == *"v2"* ]] && { idx=${f##*/}; break; }
    done
  fi
  echo "$src $dev ${hca:--} ${idx:--}"
}

# rails <netdev> <gid index>: the RoCE devices this node can run the link on - the link's own device first, then every
# other RoCE device whose RoCE v2 IPv4 GID sits at the same index: one on the link's subnet (a second CX7 port, when it
# is up and addressed there), and the second PCIe link of the same QSFP port (a DGX Spark's "twin", which lives in its
# own subnet, so the subnet scan below cannot see it).
rails() {
  local dev=$1 gid=$2 net cidr hcas other ip idx
  cidr=$(ip -o -4 addr show dev "$dev" | awk '{print $4}' | head -1)
  net=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "$cidr")
  hcas=$(ls /sys/class/net/"$dev"/device/infiniband | head -1)
  for n in /sys/class/net/*; do
    other=${n##*/}
    [[ "$other" == "$dev" || ! -d $n/device/infiniband || "$(cat $n/operstate 2>/dev/null)" != up ]] && continue
    ip=$(ip -o -4 addr show dev "$other" | awk '{print $4}' | head -1)
    [[ -n "$ip" ]] || continue
    python3 -c "import ipaddress,sys; sys.exit(ipaddress.ip_interface(sys.argv[1]) not in ipaddress.ip_network(sys.argv[2]) and ipaddress.ip_interface(sys.argv[1]).ip not in ipaddress.ip_network(sys.argv[2]))" "$ip" "$net" || continue
    idx=$(ls $n/device/infiniband | head -1)
    [[ "$(cat /sys/class/infiniband/$idx/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$idx"
  done
  # A DGX Spark's QSFP port reaches the GB10 over two independent PCIe Gen5 x4 links, so one cabled port shows up as
  # two netdevs and two RoCE devices (rocep1s0f0 and roceP2p1s0f0), and NVIDIA's own two-Spark playbook gives the two
  # twins different subnets - so the scan above never sees the twin and NCCL gets one x4: ~112 Gb/s of the port's 200.
  # Pair the twin by name instead of by subnet: same fN/npM tail, different PCIe prefix. Same conditions as above
  # (link up, a RoCE v2 GID at $gid), and never add a device twice.
  # (only for Spark-style names, <prefix>f<N>np<M>; the check below compares whole comma-separated names, so a twin the
  # subnet scan already added, as when both twins share the link's subnet, is not listed twice)
  local tail=${dev##*f} other2 ib2
  [[ "$dev" =~ f[0-9]+np[0-9]+$ ]] || { echo "$hcas"; return 0; }
  for n in /sys/class/net/*; do
    other2=${n##*/}
    [[ "$other2" == "$dev" || "$other2" != *f"$tail" ]] && continue
    [[ -d $n/device/infiniband && "$(cat $n/operstate 2>/dev/null)" == up ]] || continue
    ib2=$(ls $n/device/infiniband | head -1)
    [[ -n "$ib2" && ",$hcas," != *",$ib2,"* ]] || continue
    [[ "$(cat /sys/class/infiniband/$ib2/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$ib2"
  done
  echo "$hcas"
}

# rail_info <rdma device>: "<netdev> <IPv4 address> <MTU>" of the port a RoCE device sits on ("- - -" when it has none).
# Self-contained: it runs on the workers too (its definition is sent over ssh).
rail_info() {
  local n
  for n in /sys/class/infiniband/"$1"/device/net/*; do
    [[ -d $n ]] || continue
    echo "${n##*/} $(ip -o -4 addr show dev "${n##*/}" | awk '{print $4}' | head -1 | cut -d/ -f1) $(cat "$n/mtu" 2>/dev/null || echo 1500)"
    return 0
  done
  echo "- - -"
}

# rail_ping <local netdev> <peer address> <MTU>: one rail's traffic reaches the peer's twin - a full-size, unfragmentable
# ping bound to that rail's own port (-I <netdev>: an address alone would be routed out of whichever port of the same
# subnet the kernel picks, usually the cabled one), so an unaddressed, differently-subnetted or smaller-MTU twin fails
# here instead of as vendor_err 0x81 (transport retry exceeded) in the first all-gather (issue #66). Self-contained.
rail_ping() { ping -c2 -W2 -I "$1" -M do -s $(( $3 - 28 )) "$2" >/dev/null 2>&1; }

# The same function on worker i (its definition is sent over ssh).
worker_link_info() { worker "$1" "$(declare -f link_info); link_info $2"; }
# An image's identity by content (its layers' diffIDs and runtime config), the same under Docker's overlay2 and
# containerd image stores: .Id is the config digest under one and the manifest digest under the other, so it never
# matches across a mixed pair (issue #8). The template holds no spaces: worker() passes it through ssh, which
# re-splits arguments. A missing image is "missing". worker_image_ident <i> <image>: on worker i.
IMAGE_IDENT='{{.RootFS.Layers}}{{.Config.Env}}{{.Config.Entrypoint}}{{.Config.Cmd}}{{.Config.WorkingDir}}'
image_ident() { local s; s=$(docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
worker_image_ident() {
  local s; s=$(worker "$1" docker image inspect -f "$IMAGE_IDENT" "$2" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing
}
# cx7_peer <i>: worker i's address on a CX7 port that one of this node's CX7 ports reaches directly (same subnet, no
# gateway), for a worker given by its LAN address (issue #9). Empty when there is none.
cx7_peer() {
  local a route dev
  for a in $(worker "$1" 'for d in /sys/class/net/*; do [ -d "$d/device/infiniband" ] && ip -o -4 addr show dev "${d##*/}"; done' 2>/dev/null |
             awk '{print $4}' | cut -d/ -f1); do
    route=$(ip -o -4 route get "$a" 2>/dev/null) || continue
    [[ "$route" == *" via "* ]] && continue
    dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
    [[ -d /sys/class/net/$dev/device/infiniband ]] && { echo "$a"; return 0; }
  done
  return 1
}

# probe_rails: before a second rail (the PCIe twin of the cabled port, or a second cabled port) is used, check that it
# reaches the peer. The RDMA ring pairs the two nodes' devices by list position, so a twin that is up and has a GID
# but no path to the peer's twin (no address, another subnet, another MTU, a cable on a different port) ends in vendor_err
# 0x81. Rail k of the head is probed against rail k of the worker, in both directions; a rail that fails is dropped from both
# lists with a warning (the first device, the link itself, is never probed: the launcher reached the worker over it).
probe_rails() {
  local -a h w keep_h=() keep_w=()
  local k hdev hip hmtu wdev wip wmtu mtu=0 ok
  IFS=, read -ra h <<<"$HEAD_HCAS"; IFS=, read -ra w <<<"$WORKER_HCAS"
  (( ${#h[@]} > 1 && ${#h[@]} == ${#w[@]} )) || return 0
  [[ -z "${WORKER_DOWN[1]:-}" ]] || { warn "DRY_RUN: the second rail ($HEAD_HCAS / $WORKER_HCAS) is not probed: the worker cannot be reached"; return 0; }
  keep_h=("${h[0]}"); keep_w=("${w[0]}")
  for (( k = 1; k < ${#h[@]}; k++ )); do
    read -r hdev hip hmtu <<<"$(rail_info "${h[k]}")"
    read -r wdev wip wmtu <<<"$(worker 1 "$(declare -f rail_info); rail_info ${w[k]}")"
    ok=0
    if [[ "$hip" =~ ^[0-9.]+$ && "$wip" =~ ^[0-9.]+$ && "$hmtu" =~ ^[0-9]+$ && "$wmtu" =~ ^[0-9]+$ ]]; then
      mtu=$(( hmtu < wmtu ? hmtu : wmtu ))
      if rail_ping "$hdev" "$wip" "$mtu" && worker 1 "$(declare -f rail_ping); rail_ping $wdev $hip $mtu"; then ok=1; fi
    fi
    if (( ok )); then
      keep_h+=("${h[k]}"); keep_w+=("${w[k]}")
      log "Rail ${h[k]} <-> ${w[k]}: $hip <-> $wip reaches the peer (MTU $mtu): used"
    else
      warn "second rail ${h[k]} (${hip:--}, MTU ${hmtu:--}) <-> ${w[k]} (${wip:--}, MTU ${wmtu:--}) does not reach the peer: NOT used, running on fewer rails. Give both twins an IPv4 address on one subnet per rail with the same MTU on both Sparks (README), or set NCCL_RAILS=1 to silence this"
    fi
  done
  HEAD_HCAS=$(IFS=,; echo "${keep_h[*]}"); WORKER_HCAS=$(IFS=,; echo "${keep_w[*]}")
}

# TP=2: HEAD_ADDR / HEAD_DEV / HEAD_HCA / HEAD_GID and WORKER_ADDR / WORKER_DEV / WORKER_HCA / WORKER_GID: the link the
# two ranks talk over (NCCL and the rendezvous). FABRIC_PEER overrides the worker's link address when WORKER is reached
# over another network.
detect_link() {
  local peer=${FABRIC_PEER:-${WORKER#*@}} cx7
  # a host name (an /etc/hosts alias of the CX7 address, say) is resolved first: ``ip route get`` takes addresses only
  if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    peer=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
    [[ -n "$peer" ]] || die "cannot resolve ${FABRIC_PEER:-${WORKER#*@}} to an IPv4 address"
  fi
  read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  [[ -n "${HEAD_ADDR:-}" ]] || die "no route from this node to $peer"
  # WORKER given by a LAN address (the route goes out a port without RoCE): use the worker's CX7 address instead
  if [[ -z "${FABRIC_PEER:-}" && "${HEAD_HCA:--}" == "-" ]] && cx7=$(cx7_peer 1); then
    log "$peer is reached over $HEAD_DEV, not a CX7 port: using the worker's CX7 address $cx7 for the link (FABRIC_PEER)"
    peer=$cx7
    read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  fi
  read -r WORKER_ADDR WORKER_DEV WORKER_HCA WORKER_GID <<<"$(worker_link_info 1 "$HEAD_ADDR")" || true
  [[ -n "${WORKER_ADDR:-}" ]] || die "the worker has no route back to $HEAD_ADDR"
  # every rail (both PCIe twins of the cabled port, a second cabled port): a chunk's all-gather is ~1.8x faster on two
  HEAD_HCAS=$(rails "$HEAD_DEV" "$HEAD_GID")
  WORKER_HCAS=$(worker 1 "$(declare -f rails); rails $WORKER_DEV $WORKER_GID")
  [[ "${NCCL_RAILS:-2}" == 1 ]] && { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  [[ "${NCCL_RAILS:-2}" == 1 ]] || probe_rails
  [[ "$(tr ',' '\n' <<<"$HEAD_HCAS" | wc -l)" == "$(tr ',' '\n' <<<"$WORKER_HCAS" | wc -l)" ]] ||
    { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  for v in HEAD_HCA HEAD_GID WORKER_HCA WORKER_GID; do
    [[ "${!v}" != "-" ]] || die "no RoCE device or RoCE v2 GID for the link ($v): is $HEAD_DEV / $WORKER_DEV the CX7 port between the Sparks? Set FABRIC_PEER to the worker's CX7 address"
  done
  NODE_DEV=("$HEAD_DEV" "$WORKER_DEV"); NODE_HCAS=("$HEAD_HCAS" "$WORKER_HCAS"); NODE_GID=("$HEAD_GID" "$WORKER_GID")
  LINK_HEAD_ADDR[1]=$HEAD_ADDR; LINK_WORKER_ADDR[1]=$WORKER_ADDR
  MASTER_ADDR=${MASTER_ADDR:-$HEAD_ADDR}
}

# ---------------------------------------------------------------- TP=3: the triangle
# node_inventory [master]: this node's default-route netdev ("default <netdev>"), whether it routes to <master>
# ("master ok|none"), and one line per RoCE netdev that is up with an IPv4 address: "<netdev> <rdma device>
# <address/prefix> <RoCE v2 GID index of that address, or ->". Read-only (ip and sysfs).
# (The GLM-5.3 3x recipe's, MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold, unchanged.)
node_inventory() {
  local n dev hca cidr want idx f
  echo "default $(ip -o -4 route show default 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)"
  [[ -z "${1:-}" ]] || { ip -o -4 route get "$1" >/dev/null 2>&1 && echo "master ok" || echo "master none"; }
  for n in /sys/class/net/*; do
    dev=${n##*/}
    [[ -d $n/device/infiniband && "$(cat "$n/operstate" 2>/dev/null)" == up ]] || continue
    hca=$(ls "$n/device/infiniband" | head -1)
    for cidr in $(ip -o -4 addr show dev "$dev" | awk '{print $4}'); do
      want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"${cidr%/*}"))
      idx=-
      for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
        [[ "$(cat "$f" 2>/dev/null)" == "$want" ]] || continue
        [[ "$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)" == *v2* ]] && { idx=${f##*/}; break; }
      done
      echo "$dev $hca $cidr $idx"
    done
  done
}

# pair_triangle (stdin: node_inventory lines, each prefixed with its rank 0, 1 or 2): the three nodes' rails as LakyFx's
# runtime takes them (its tensorfold/launch.py and cluster.example.json; easthambuilds' tools/topology.py works them
# out the same way). A rank's rails are its four RoCE devices that are up with an IPv4 address, in port order, each
# port's first PCIe function before its twin (rocep1s0f0, roceP2p1s0f0, rocep1s0f1, roceP2p1s0f1): two direct cables,
# each seen as two devices, each device in a subnet shared only with its twin at the cable's other end. Two ranks are
# linked by the rails whose subnets they share. Prints, for each rank,
#   "rank <r> <rails> <peer masks> <NCCL devices> <GID index> <socket netdev> <socket address>": the peer masks
#     (TF_RDMA_PEER_MASKS: for ranks 0, 1 and 2 the bits of the rails that reach it, 0 for itself), the NCCL devices
#     (the first device of each cable: switchless NCCL listens on two GIDs only), the one RoCE v2 GID index of its
#     four rails, its NCCL socket (its first rail to rank 0; rank 0's: its first rail to rank 1),
#   "master <address>": rank 0's address on its first rail to rank 1 (where the ranks meet: TF_TP_MASTER),
#   "peer <i> <worker i's address> <rank 0's address>": on their first shared rail (worker i's NFS server),
#   "range <cidr>": the smallest network holding every rail's subnet (NCCL_IB_ADDR_RANGE),
#   "route <r> <address> <via or -> <netdev>": each address rank r must reach out of that netdev: the far end of each of
#     its rails, then the master and the others' sockets, which are on cables it is not on: through the cable it
#     shares with their owner;
# or only "error <message>" lines, one for each thing that does not fit.
pair_triangle() {
  python3 -c '
import ipaddress, re, sys
nodes = {0: [], 1: [], 2: []}
for line in sys.stdin:
    f = line.split()
    if len(f) == 5 and f[0] in ("0", "1", "2"):
        rails = nodes[int(f[0])]
        if all(e["hca"] != f[2] for e in rails):
            rails.append({"dev": f[1], "hca": f[2], "ip": ipaddress.ip_interface(f[3]), "gid": f[4]})
def order(e):
    port = re.search(r"f(\d+)np\d+$", e["dev"])
    return (int(port.group(1)) if port else 9, "P" in e["dev"][:5], e["hca"])
def fail(errors):
    for e in errors:
        print("error", e)
    sys.exit(0)
errors = []
for r, rails in nodes.items():
    rails.sort(key=order)
    if len(rails) != 4:
        errors.append("rank %d has %d RoCE devices up with an IPv4 address (%s), not 4: both ports cabled, and both PCIe "
                      "functions of each port addressed" % (r, len(rails), ",".join(e["hca"] for e in rails) or "none"))
if errors:
    fail(errors)
def shared(i, j):
    nets = {e["ip"].network for e in nodes[j]}
    return [k for k, e in enumerate(nodes[i]) if e["ip"].network in nets]
for i in range(3):
    for j in range(i + 1, 3):
        a, b = shared(i, j), shared(j, i)
        if not a:
            errors.append("ranks %d and %d share no subnet: three Sparks need a cable between every pair (a triangle)" % (i, j))
        elif [nodes[i][k]["ip"].network for k in a] != [nodes[j][k]["ip"].network for k in b]:
            errors.append("ranks %d and %d do not list their rails to each other in one order (%s against %s): the runtime "
                          "pairs them by position, so each cable needs one subnet a PCIe function, alike at both ends"
                          % (i, j, ",".join(nodes[i][k]["hca"] for k in a), ",".join(nodes[j][k]["hca"] for k in b)))
    others = [shared(i, j) for j in range(3) if j != i]
    if set(others[0]) & set(others[1]):
        errors.append("rank %d has a rail in a subnet of both other ranks (a switch, or one subnet for two cables): "
                      "give each cable subnets of its own" % i)
    gids = {e["gid"] for e in nodes[i]}
    if len(gids) != 1 or "-" in gids:
        errors.append("rank %d: its rails have RoCE v2 IPv4 GID indexes %s; the runtime takes one for all four" % (i, ",".join(sorted(gids))))
if errors:
    fail(errors)
first = lambda i, j: shared(i, j)[0]
addr = lambda i, k: str(nodes[i][k]["ip"].ip)
sock = {0: first(0, 1), 1: first(1, 0), 2: first(2, 0)}
for i in range(3):
    masks = [sum(1 << k for k in shared(i, j)) if j != i else 0 for j in range(3)]
    nccl = [nodes[i][k]["hca"] for k in sorted(first(i, j) for j in range(3) if j != i)]
    print("rank", i, ",".join(e["hca"] for e in nodes[i]), ",".join(map(str, masks)), ",".join(nccl),
          nodes[i][0]["gid"], nodes[i][sock[i]]["dev"], addr(i, sock[i]))
master = addr(0, first(0, 1))
print("master", master)
for i in (1, 2):
    print("peer", i, addr(i, first(i, 0)), addr(0, first(0, i)))
nets = [e["ip"].network for rails in nodes.values() for e in rails]
span = nets[0]
while not all(n.subnet_of(span) for n in nets):
    span = span.supernet()
print("range", span)
for i in range(3):
    seen = set()
    for k, e in enumerate(nodes[i]):
        for j in range(3):
            if j != i:
                for o in nodes[j]:
                    if o["ip"].network == e["ip"].network:
                        print("route", i, o["ip"].ip, "-", e["dev"]); seen.add(str(o["ip"].ip))
    want = [(master, 0)] if i else []
    want += [(addr(j, sock[j]), j) for j in range(3) if j != i]
    for a, j in want:
        if a in seen or j == i:
            continue
        seen.add(a)
        k = first(i, j)
        via = next(o for o in nodes[j] if o["ip"].network == nodes[i][k]["ip"].network)
        print("route", i, a, via["ip"].ip, nodes[i][k]["dev"])
'
}

# tp3_route_missing <address,via,netdev>...: those of the given routes that `ip route get` does not send out that
# netdev (through that gateway, when one is named), one "<address> <via> <netdev>" a line. Self-contained: it runs on
# the workers too (its definition is sent over ssh).
tp3_route_missing() {
  local x a v d got
  for x in "$@"; do
    IFS=, read -r a v d <<<"$x"
    got="$(ip -o -4 route get "$a" 2>/dev/null) "
    [[ "$got" == *" dev $d "* && ( "$v" == - || "$got" == *" via $v "* ) ]] || echo "$a $v $d"
  done
}

# tp3_routes_check: every route pair_triangle asked for, on every rank. With one subnet a cable, a rank reaches the master
# and the others' NCCL sockets only through a /32 route over the cable it shares with their owner; and a VPN's routing
# table (Tailscale's, say) can capture a cable's own subnet, so the far end of each rail is checked too. A missing route
# is printed as the command that adds it (until the next reboot: make it permanent in netplan or NetworkManager); it
# refuses, or warns with DRY_RUN=1.
tp3_routes_check() {
  local r x out a v d fixes="" who
  local -a want
  for r in 0 $(worker_ids); do
    [[ -z "${WORKER_DOWN[$r]:-}" ]] || continue
    want=()
    for x in "${TP3_ROUTES[@]}"; do
      if [[ "${x%% *}" == "$r" ]]; then read -r _ a v d <<<"$x"; want+=("$a,$v,$d"); fi
    done
    if (( r == 0 )); then out=$(tp3_route_missing "${want[@]}"); who="this node"
    else
      who=$(worker_host "$r")
      out=$(worker "$r" "$(declare -f tp3_route_missing); tp3_route_missing ${want[*]}") || die "cannot check the routes on $who"
    fi
    while read -r a v d; do
      [[ -n "$a" ]] || continue
      fixes+=$'\n'"  rank $r ($who): sudo ip route replace $a/32 $( [[ "$v" == - ]] || echo "via $v ")dev $d"
    done <<<"$out"
  done
  [[ -n "$fixes" ]] || return 0
  local msg="routes over the cables are missing: each rank must reach the far end of each of its rails, the master address and the other ranks' NCCL sockets over the cables (docs/tp3.md, The triangle). Add them, and make them permanent in netplan or NetworkManager:$fixes"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then warn "DRY_RUN: $msg"; return 0; fi
  die "$msg"
}

# detect_triangle: TP=3's links (detect_links). NODE_HCAS: each rank's four rails (TF_RDMA_DEVICES, --rdma), NODE_MASKS
# their peer masks, NODE_NCCL_HCAS its NCCL devices, NODE_GID, NODE_DEV its NCCL socket's netdev (SOCKET_IFNAME
# overrides it); LINK_HEAD_ADDR / LINK_WORKER_ADDR each worker's first cable to the head; MASTER_ADDR (unless set) the
# master pair_triangle names; NCCL_ADDR_RANGE. FABRIC_PEER2 is not read: the cables are found from the subnets alone.
declare -a NODE_MASKS=() NODE_NCCL_HCAS=() TP3_ROUTES=()
NCCL_ADDR_RANGE=""
detect_triangle() {
  local r i kind a b c d e f g pairs errors down="" n
  local -a inv=()
  inv[0]=$(node_inventory)
  for i in $(worker_ids); do
    if [[ -n "${WORKER_DOWN[$i]:-}" ]]; then down+=" $(wvar WORKER "$i")"; continue; fi
    inv[i]=$(worker "$i" "$(declare -f node_inventory); node_inventory") || die "cannot read $(worker_host "$i")'s RoCE devices"
  done
  if [[ -n "$down" ]]; then
    warn "DRY_RUN:$down cannot be reached: the triangle is not paired, its values print as <...>"
    for r in 0 $(worker_ids); do
      n=head; (( r == 0 )) || n=$(wvar WORKER "$r")
      NODE_HCAS[r]="<$n:rails>"; NODE_MASKS[r]="<$n:masks>"; NODE_NCCL_HCAS[r]="<$n:nccl-hcas>"; NODE_GID[r]="<$n:gid>"
      NODE_DEV[r]=${SOCKET_IFNAME:-<$n:netdev>}
      (( r == 0 )) || { LINK_HEAD_ADDR[r]="<head-toward-$n>"; LINK_WORKER_ADDR[r]="<$n:address>"; }
    done
    MASTER_ADDR=${MASTER_ADDR:-<master>}; NCCL_ADDR_RANGE="<fabric-range>"
    return 0
  fi
  pairs=$(for r in "${!inv[@]}"; do sed "s/^/$r /" <<<"${inv[r]}"; done | pair_triangle) || die "could not pair the three nodes' rails"
  errors=$(sed -n 's/^error /  /p' <<<"$pairs")
  [[ -z "$errors" ]] || die "the three Sparks' cables do not fit TP=3 (docs/tp3.md, The triangle):"$'\n'"$errors"
  TP3_ROUTES=()
  while read -r kind a b c d e f g; do
    case "$kind" in
      rank) NODE_HCAS[a]=$b; NODE_MASKS[a]=$c; NODE_NCCL_HCAS[a]=$d; NODE_GID[a]=$e; NODE_DEV[a]=${SOCKET_IFNAME:-$f} ;;
      master) MASTER_ADDR=${MASTER_ADDR:-$a} ;;
      peer) LINK_WORKER_ADDR[a]=$b; LINK_HEAD_ADDR[a]=$c ;;
      range) NCCL_ADDR_RANGE=$a ;;
      route) TP3_ROUTES+=("$a $b $c $d") ;;
    esac
  done <<<"$pairs"
  tp3_routes_check
}

# rendezvous_addr <i>: where rank i meets rank 0 (the followers' MASTER argument, rank 0's TF_TP_MASTER): the head's
# address on worker i's link at TP=2; at TP=3 rank 0 listens on one address for both, MASTER_ADDR.
rendezvous_addr() { if (( ${TP:-2} == 3 )); then echo "$MASTER_ADDR"; else echo "${LINK_HEAD_ADDR[$1]:-$MASTER_ADDR}"; fi; }

# The NCCL settings of rank r's container (docker -e arguments): the RoCE link's netdev, RoCE devices (both rails) and GID
# index, as the dsv41 family's two-node README runs them (NCCL_SOCKET_IFNAME, NCCL_IB_HCA, NCCL_IB_GID_INDEX).
# Every other exported NCCL_* variable goes to both ranks unchanged (an A/B of NCCL settings: NCCL_NET_PLUGIN=none,
# NCCL_MAX_NCHANNELS=4, NCCL_IB_QPS_PER_CONNECTION=2, NCCL_DEBUG_SUBSYS=INIT,NET ...); none set: no change. The names and
# values are checked here, in the main shell, before anything is stopped: a value is [A-Za-z0-9_.,:=/-] only (no spaces,
# quotes, $, ;, newlines), because it ends up in a command that the worker's shell runs. NCCL_SOCKET_IFNAME, NCCL_IB_HCA
# and NCCL_IB_GID_INDEX are this script's own (from the link it finds); NCCL_RAILS is its setting, not NCCL's.
NCCL_EXTRA_ENV=()
NCCL_ARGS=()
nccl_extra_collect() {
  local name val
  NCCL_EXTRA_ENV=()
  while IFS= read -r name; do
    [[ "$name" =~ ^NCCL_[A-Za-z0-9_]+$ ]] || continue
    val=${!name}
    case "$name" in
      NCCL_RAILS) continue ;;
      NCCL_SOCKET_IFNAME|NCCL_IB_HCA|NCCL_IB_GID_INDEX) warn "$name is set by this script (from the link it finds): your value is ignored"; continue ;;
    esac
    if [[ -z "$val" ]]; then warn "$name is empty: not passed on"; continue; fi
    [[ "$val" =~ ^[A-Za-z0-9_.,:=/-]+$ ]] || die "$name has a value with characters outside A-Z a-z 0-9 _ . , : = / - (it goes into a shell command on the worker): not passed on, nothing started"
    [[ "$name" == NCCL_DEBUG ]] || NCCL_EXTRA_ENV+=("$name=$val")
  done < <(compgen -e | LC_ALL=C sort | grep -E '^NCCL_' || true)
}

nccl_env() {
  local dev=$1 hcas=$2 gid=$3
  NCCL_ARGS=(-e "NCCL_SOCKET_IFNAME=$dev" -e "NCCL_IB_HCA=$hcas" -e "NCCL_IB_GID_INDEX=$gid")
  nccl_line
}
# nccl_line: NCCL_DEBUG and the exported NCCL_* added to NCCL_ARGS, and the result printed as one line for a remote
# shell: any word with a character outside [A-Za-z0-9_.,:=/@%+-] is shell-quoted (printf %q), so a value can never be
# taken for shell syntax there (the values nccl_extra_collect lets through need no quoting, which keeps the line as it
# always was)
nccl_line() {
  [[ -z "${NCCL_DEBUG:-}" ]] || NCCL_ARGS+=(-e "NCCL_DEBUG=$NCCL_DEBUG")
  local x; for x in "${NCCL_EXTRA_ENV[@]}"; do NCCL_ARGS+=(-e "$x"); done
  local out="" w
  for w in "${NCCL_ARGS[@]}"; do
    if [[ "$w" =~ ^[A-Za-z0-9_.,:=/@%+-]+$ ]]; then out+="$w "; else out+="$(printf '%q' "$w") "; fi
  done
  echo "${out% }"
}
# TP=3: LakyFx's NCCL settings (its tensorfold/launch.py) with each rank's own socket netdev ("=": that name exactly)
# and devices, and what three Sparks on direct cables needed beyond them with the switchless NCCL build: all four of
# NCCL_SWITCHLESS_RING_ONLY=1, NCCL_ALGO=Ring, NCCL_IB_SUBNET_AWARE_ROUTING=1 and NCCL_IB_MERGE_NICS=0 (launch.py sets
# only the last two), and NCCL_IB_HCA the first device of each cable only: the switchless build takes two listener GIDs,
# while TensorFold's own RDMA ring (TF_RDMA_DEVICES) runs on all four. No network is shared by all three: the sockets
# are on the cables, and NCCL_IB_ADDR_RANGE holds every cable's subnet. The exported NCCL_* come last, as at TP=2.
nccl_env_tp3() {
  local dev=$1 hcas=$2
  NCCL_ARGS=(-e "NCCL_SOCKET_IFNAME==$dev" -e "NCCL_IB_HCA=$hcas" -e "NCCL_IB_ADDR_RANGE=$NCCL_ADDR_RANGE"
             -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_ALGO=Ring
             -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_IB_DISABLE=0
             -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1)
  nccl_line
}
rank_nccl_env() {
  if (( ${TP:-2} == 3 )); then nccl_env_tp3 "${NODE_DEV[$1]}" "${NODE_NCCL_HCAS[$1]}"
  else nccl_env "${NODE_DEV[$1]}" "${NODE_HCAS[$1]}" "${NODE_GID[$1]}"; fi
}
# rank_nccl_args <r>: the same as the array NCCL_ARGS (docker arguments), for a command run here without a shell between
rank_nccl_args() { rank_nccl_env "$1" >/dev/null; }

# detect_links: the one link of two Sparks (start.sh and prepare.sh call it; the GLM scripts' name for it at any TP).
# detect_link and probe_rails are the two-node path in the GLM-5.3 3x and the 4x recipes too; at TP=3, the triangle
# (detect_triangle above).
detect_links() { if (( ${TP:-2} == 3 )); then detect_triangle; else detect_link; fi; }

# Rank 0's checkpoint volume (HEAD_NFS_SERVER): like ensure_nfs_volume, for this Spark. Prints what rank 0 mounts at /hf.
head_hf_mount() {
  [[ -n "$HEAD_NFS_SERVER" ]] || { echo "$HF_CACHE"; return 0; }
  local have
  have=$(docker volume inspect -f '{{index .Options "device"}} {{index .Options "o"}}' "$HEAD_NFS_VOLUME" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    docker volume create --driver local --opt type=nfs --opt device=":$NFS_PATH" \
      --opt o="addr=$HEAD_NFS_SERVER,nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600" "$HEAD_NFS_VOLUME" >/dev/null ||
      die "could not create the docker volume $HEAD_NFS_VOLUME here"
  elif [[ "${have%% *}" != ":$NFS_PATH" || "$have" != *"addr=$HEAD_NFS_SERVER,"* ]]; then
    die "the docker volume $HEAD_NFS_VOLUME here is ${have%% *} from ${have#* }, not :$NFS_PATH from $HEAD_NFS_SERVER: remove it (docker volume rm $HEAD_NFS_VOLUME) or set HEAD_NFS_VOLUME / HEAD_NFS_SERVER"
  fi
  echo "$HEAD_NFS_VOLUME"
}

# mem_available_gib: MemAvailable here, in GiB (one decimal)
# mem_available_gib: what start.sh's MIN_AVAIL_GIB check counts here, MemAvailable plus the pages parked on the per-CPU
# free lists (mem_counting below; plain MemAvailable if /proc/zoneinfo cannot be read)
mem_available_gib() { local m; m=$(mem_counting 2>/dev/null) && echo "${m%% *}" || awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo; }
# mem_counting: MemAvailable plus the pages parked on the per-CPU free lists, as the engine's startup budget counts it
# (scripts/memavail.sh; start.sh's MIN_AVAIL_GIB check)
source "${_cfg_root:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/memavail.sh"

# await_memory: sets HERE_GB and WORKER_GB[i] (GiB) to the memory available on this Spark and on each worker i, once all
# reach MIN_AVAIL_GIB or after MIN_AVAIL_WAIT_S seconds at the latest (returns 1 then; the caller words the refusal). A build or a large
# copy that has just ended leaves page cache that MemAvailable counts only as it is written back (111.6 GiB was read a few
# seconds after a build, 119 GiB once it settled), so one reading right after it refuses a start the Spark can take.
declare -a WORKER_GB=()
await_memory() {
  local waited=0 i ok
  while :; do
    HERE_GB=$(mem_available_gib)
    ok=1; awk -v a="$HERE_GB" -v m="$MIN_AVAIL_GIB" 'BEGIN {exit !(a >= m)}' || ok=0
    for i in $(worker_ids); do
      WORKER_GB[i]=$(worker "$i" "$(declare -f mem_counting mem_available_gib); mem_available_gib" 2>/dev/null || echo 0)
      awk -v a="${WORKER_GB[i]}" -v m="$MIN_AVAIL_GIB" 'BEGIN {exit !(a >= m)}' || ok=0
    done
    (( ok )) && return 0
    (( waited < MIN_AVAIL_WAIT_S )) || return 1
    sleep 2
    waited=$((waited + 2))
  done
}

# The memory guard (scripts/memguard.sh) on this Spark and on the worker, which gets a copy of the script.
MEMGUARD_REMOTE_DIR='$HOME/.cache/dsv41-tensorfold'
memguard_start() {
  awk -v g="$MEMGUARD_GIB" 'BEGIN {exit !(g > 0)}' || return 0
  "$_cfg_root/scripts/memguard.sh" start "$CONTAINER_NAME" "$MEMGUARD_GIB" "$LOG_DIR/memguard.log" "$STATE_DIR/memguard.pid" "$MEMGUARD_HOLD_S" "$MEMGUARD_HARD_GIB" ||
    die "the memory guard did not start here"
  local i
  for i in $(worker_ids); do
    worker "$i" "mkdir -p $MEMGUARD_REMOTE_DIR/logs && cat > $MEMGUARD_REMOTE_DIR/memguard.sh && chmod +x $MEMGUARD_REMOTE_DIR/memguard.sh" \
      < "$_cfg_root/scripts/memguard.sh" || die "could not copy the memory guard to $(worker_host "$i")"
    worker "$i" "$MEMGUARD_REMOTE_DIR/memguard.sh start '$CONTAINER_NAME' $MEMGUARD_GIB $MEMGUARD_REMOTE_DIR/logs/memguard.log $MEMGUARD_REMOTE_DIR/memguard.pid $MEMGUARD_HOLD_S $MEMGUARD_HARD_GIB" ||
      die "the memory guard did not start on $(worker_host "$i")"
  done
  log "Memory guard on: each rank's container is killed when the memory available stays under $MEMGUARD_GIB GiB for $MEMGUARD_HOLD_S s, or falls under $MEMGUARD_HARD_GIB GiB (log: memguard.log)"
}
memguard_stop() {
  local i
  "$_cfg_root/scripts/memguard.sh" stop "$STATE_DIR/memguard.pid" 2>/dev/null || true
  for i in $(configured_workers); do
    worker "$i" "[ -x $MEMGUARD_REMOTE_DIR/memguard.sh ] && $MEMGUARD_REMOTE_DIR/memguard.sh stop $MEMGUARD_REMOTE_DIR/memguard.pid; true" 2>/dev/null || true
  done
}
