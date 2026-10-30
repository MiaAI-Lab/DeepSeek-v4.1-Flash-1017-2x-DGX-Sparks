# Requirements

The hardware, link, ssh, Docker, disk and NFS setup the recipe needs. The README has the short version.

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory), with nothing else large on them: the weights take
  ~98 GiB a Spark and rank 0 holds ~104 GiB on the GPU, so a Spark's memory is the limit, not its disk or its link.
  `start.sh` refuses to start unless `MIN_AVAIL_GIB` (112) is available on each Spark. A head that also runs a desktop
  and development tools (~6 GiB here) was left with ~3.1 GiB at the worst point of a 100K-token prompt at the full window
  and ~3.7 GiB at 524,288 tokens: give the head less to do, or lower `CONTEXT`.
- **The same GPU clock ceiling on both Sparks** (`nvidia-smi -lgc 0,2200`; check with `nvidia-smi -q -d CLOCK`). With TP=2 the
  ranks meet twice a layer, so the slower GPU sets the pace: one Spark found locked at 1,800 MHz cost about 3-6% of prefill
  and 3% of decode against 2,200 MHz on both (rigmark, same prompts; the GPUs peaked at 85 / 81 C and nothing throttled).
- **A direct ConnectX-7 link:** a QSFP cable between the CX7 ports and an IPv4 address on each end in one private subnet
  (e.g. `192.0.2.1/24` and `192.0.2.2/24`; `ping` must work), with a RoCE v2 GID (`start.sh` checks). One QSFP port of a
  Spark reaches the GB10 over two PCIe Gen5 x4 links, so it appears as two netdevs and two RoCE devices (`enp1s0f0np0` /
  `enP2p1s0f0np0`, `rocep1s0f0` / `roceP2p1s0f0`). Address both twins on each Spark (one subnet each is fine, the same
  MTU on both Sparks): both are then used as two rails; one rail is one x4 (~112 Gb/s of the port's 200).
- **Key-based ssh** from the first Spark (the head, which runs `./start.sh` and the API) to the second (the worker):
  `ssh-copy-id user@<worker>`; check with `ssh -o BatchMode=yes user@<worker> true`.
- Docker with the NVIDIA container runtime and your user in the `docker` group, on both Sparks; `git`, `python3` and the
  `hf` CLI (`pip install -U huggingface_hub`) on the head.
- **Disk.** Head: ~505 GB (the checkpoint 196 GiB, Engram's tables ~203 GB, rank 0's weight file ~106 GB) plus ~25 GB for
  the image under Docker's root. Worker: ~310 GB (Engram's tables and rank 1's weight file) plus the image; the worker
  reads the checkpoint from the head over NFS.
- **NFS.** The worker mounts the head's Hugging Face cache read-only. If a host or container NFS server already exports
  that cache, use it (see below); do not start a competing server. Otherwise, on the head, once (needs root; the worker needs
  nothing installed):

  ```bash
  sudo apt install nfs-kernel-server
  echo "$HOME/.cache/huggingface <worker CX7 address>(ro,no_subtree_check)" | sudo tee -a /etc/exports
  sudo exportfs -ra
  ```

  If `prepare.sh` or `start.sh` stops with "cannot read the model/kit", read the output above it: Docker mount errors are
  preserved, or the check names the missing, empty or unreadable file. Check `sudo exportfs -v` on the head: the export must
  list the *worker's* address on the link (the user@address in the message), the exported path must be `NFS_PATH`, and the worker must
  reach the head's port 2049; the worker's docker mounts the volume itself, so it needs the host's NFS mount helper (`nfs-common` on
  Ubuntu) if it reports a missing `mount.nfs` helper.
  `prepare.sh` then creates a read-only docker NFS volume on the worker (`NFS_VOLUME`, no sudo) and checks that the worker
  sees every file the ranks read. `NFS_PATH` is the exported path as the worker mounts it (default: the head's `HF_CACHE`;
  `/` for an NFSv4 export with `fsid=0`), `NFS_SERVER` the head's address (default: its address on the link).

  **An existing NFS server (including a container).** Use the server already exporting the cache rather than starting
  a second one. Host and privileged container NFS servers can share kernel nfsd but run different `rpc.mountd` instances:
  `showmount -e` or the host's `/etc/exports` can then disagree with the namespace the worker actually mounts. Inspect
  `/proc/fs/nfs/exports` on the head (and inside the exporter container, if any), and check the worker's actual `/hf` view.
  If the cache itself is exported as `fsid=0`, its client path is `/`, not its absolute host path. Set `NFS_PATH` accordingly
  in `scripts/local.sh`; changing it does not update an existing Docker volume. Stop the model ranks before removing the
  unused worker volume with `docker volume rm <NFS_VOLUME>` and rerun preparation to recreate it. Do not delete model files.
  If you see stale file handles, resolve the conflicting exporters first, then remount the affected client volumes while
  ranks are stopped. Choose one NFS owner deliberately; the recipe does not disable host services or containers for you.

  Every fresh launch/restart checks the worker's model config, kit manifest and token map, even when preparation was
  previously successful. A wrong root that still mounts successfully is rejected before GPU weights are loaded. This is
  a small-file preflight, not a replacement for preparation's full file-size check; it also checks the documented local
  host-path setup below. `DRY_RUN=1` skips it, and an already-running server is left alone as before.

  **Or no NFS at all.** Each Spark can run from a local copy: set `HF_CACHE` to a local tree holding the checkpoint's small files
  and its vision shard (`model-00039-of-00039`, the only shard read at run time), the served `model/` folder, the kit, the token
  map and, on the head, rank 0's weight file; set `NFS_VOLUME` to the same path as a *host path* (the worker bind-mounts it) and
  leave `HEAD_NFS_SERVER` empty. Set `PREPARE=0` too: `scripts/prepare.sh` assumes a complete cache and would start downloading
  the 196 GiB checkpoint into a partial one.

The former default checkpoint revision `6d54be4f` no longer resolves on the Hub. The default now pins `45ee5d0f` (EAQ v1); `EAQ_V1_1=1` pins `aee640e2` (EAQ v1.1). Changing revisions changes the Hugging Face snapshot path and may require another 196 GiB download. Keep an existing local snapshot only by explicitly setting `MODEL_REVISION` and using the documented cached/offline preparation flow.

## What gets downloaded

The checkpoint: `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` (39 shards, routed experts at 3 bits a
  weight, 2 bits in layers 18-22, attention at 5, shared experts at 4-5, head at 6, DSpark blocks at 4; the vision tower
  and the DSpark blocks are in it; the revision is pinned in `scripts/config.sh`), plus Engram's two tables, unquantized, from
  `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` (shards 47 and 48, ~203 GB; the quantization leaves them out).
  `EAQ_V1_1=1` switches to `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw` @ `aee640e2` (39 shards, ~196 GiB;
  27 indexer/compressor linears are BF16 and load as plain BF16 linears through patch `0024`'s mixed-pack loader)
