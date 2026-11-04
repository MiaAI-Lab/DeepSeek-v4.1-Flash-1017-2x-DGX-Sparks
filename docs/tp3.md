# Three Sparks (TP=3, experimental)

`TP=3` serves the same checkpoint on **three** DGX Sparks cabled as a triangle (each Spark's two ConnectX-7 ports to
the other two, three cables, no switch). It is opt-in and experimental: **it has not been run on three Sparks in this
form yet** (below: what was checked, and what is left). `TP=2`, the default, is unchanged.

BertholomusAI's engine at `TF_REF` runs the dsv41 family over two or four ranks, not three. At `TP=3` the image is
**LakyFx's TensorFold TP3 port** instead: [`LakyFx/deepseek-v41-flash-exl3-tp3-tp4-spark`](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-tp4-spark)
(named `...-tp3-spark` when its tag `v1.2.2TF` was made) at `14a6d49`, which ships the TensorFold runtime as a frozen
snapshot (`61357deb`) with its kernel kit and preparation tools. The way this runtime serves a 1017 EXL3 checkpoint on
three Sparks follows **easthambuilds' recipe**
([`easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold`](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold),
`7d72d1e`, Apache-2.0), which ran it end to end on three Sparks with its own 3.5 bpw checkpoint: its checkpoint plan,
topology, RoPE cut and Engram steps are adapted here. What stays this recipe's: the checkpoint
(`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` @ `45ee5d0f`), Engram's tables from the base model's shards 47
and 48, `ABLIT=1`, and the scripts (`start.sh`, `stop.sh`, `scripts/memguard.sh`, the prepared marker, the docker
containers).

**The chat template at `TP=3` is LakyFx's**, not `model/chat_template.jinja`: its qualified 1017 template (its port of
DeepSeek's `encoding.py`, sha256 `7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e`, the one
easthambuilds pins as `CHAT_TEMPLATE_SHA256`). The runtime's model-file step checks for exactly that template; the
checkpoint's own (34,236 bytes, `3e0c2a31...`) and this recipe's (`2ad46996...`) both stopped a TP=3 setup there.
`scripts/prepare-tp3.sh` downloads it from `LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold` (public, MIT) at
`8621801c`, checks its sha256 and puts it in the model folder (`TP3_TEMPLATE_*` in `scripts/config.sh`). `TP=2` keeps
serving `model/chat_template.jinja`, unchanged.

## Setup

In `scripts/local.sh`:

```bash
TP=3
WORKER=user@192.0.2.2         # rank 1
WORKER2=user@192.0.2.3        # rank 2
```

Then `DRY_RUN=1 ./start.sh` (the cables it finds, the routes it misses, the three rank commands) and `./start.sh`.
The first run builds the TP=3 image on the head (`zig/Dockerfile.tp3`) and copies it to both workers, downloads what
`TP=2` downloads, and makes every rank's files (`scripts/prepare-tp3.sh`, below). Setting `TP=3` on a pair that served
`TP=2` re-runs the preparation; the `TP=2` files stay where they are.

## Requirements

- **Three Sparks cabled as a triangle.** Each cable shows up as two RoCE devices at each end (one per PCIe function of
  the port, e.g. `rocep1s0f0` and `roceP2p1s0f0`). Give each of them an IPv4 address in a subnet of its own, shared only
  with the device at the cable's other end: six subnets in all, the same RoCE v2 GID index for all four devices of a
  Spark, the same MTU at both ends. `scripts/nodes.sh` (`detect_triangle`) pairs the Sparks by these subnets and refuses
  what does not fit, saying why.
- **Host routes over the cables.** With one subnet a cable, rank 2 has no direct path to the master address (rank 0's
  address on its cable to rank 1), and ranks 1 and 2 none to each other's NCCL socket address. `start.sh` and
  `scripts/prepare.sh` check every route a rank needs and print the missing ones as the commands that add them, e.g.
  `sudo ip route replace 192.0.2.1/32 via 192.0.2.34 dev enp1s0f0np0` (they do not run sudo). The same check covers
  the far end of each rail: a VPN's policy routing table (Tailscale's, say) can capture a cable's own subnets, and
  then a `/32` route over the cable is needed for those too. Make the routes permanent in netplan or NetworkManager.
- **NFS:** the head exports `NFS_PATH` (default `HF_CACHE`) to both workers' cable addresses. At `TP=3` no rank reads
  the checkpoint over NFS: the workers copy their files from it once (`NFS_SERVER2` overrides rank 2's server address).
- **Memory:** `MIN_AVAIL_GIB` defaults to 80 at `TP=3`: a rank's weight file is a third of the checkpoint in LakyFx's
  layout, 71,736,875,432 bytes of tensors (66.8 GiB, computed from the checkpoint's headers), plus the engine's 12.2 GiB
  reserve.
- **Disk:** on the head, the checkpoint (196 GiB) and Engram's tables (~203 GB) as at `TP=2`, then at the peak ~150 GB
  for two weight files and ~140 GB for two ranks' Engram rows (sparse; rank 0's stay, each worker's leave once
  copied), 5 GB of reference RoPE tables and ~2 GB for the kit and the model folder: ~700 GB in all. `ABLIT=1` at
  `TP=3` needs **~127 GB more on the head**, on top of the rank files (~215 GB for all three, of which this recipe
  holds two at a time before the workers' move to them): the overlay rewrites `wo_b` of layers 10 to 35, which lie in
  24 of the checkpoint's 39 shards, as new files. On each
  worker, ~145 GB in `WORKER_TP3_DIR` (its weight file, its Engram rows, the kit and the model folder) and the image.

## Settings

| Setting | Default at TP=3 | |
| --- | --- | --- |
| `TP` | 2 | `3` for three Sparks |
| `WORKER2`, `NFS_SERVER2` | | rank 2's ssh target; the head's address it copies over NFS from (default: the head's address on their first cable) |
| `WORKER_TP3_DIR` | `$HOME/.cache/dsv41-tensorfold/tp3-<rev>` | each worker's own folder (`-ablit-<rev>` added with `ABLIT=1`) |
| `POOL` | 3300000 | positions all requests share (`TF_DS_POOL_TOKENS`); at least `CONTEXT + 50` in whole 2,048-position blocks, at most 10,000,000. 3,300,000 is LakyFx's qualified TF29 pool |
| `CONTEXT`, `PARALLEL` | 401408, 8 | a request's positions, requests at once (LakyFx's TF29 profile) |
| `ROUND_ROWS` | 16 | the only value taken at `TP=3` (the profile's) |
| `PROMPT_CACHE_GIB` | 1 | rank 0's `--prompt-cache-gib` |
| `MIN_AVAIL_GIB` | 80 | above |
| `SOCKET_IFNAME`, `MASTER_ADDR` | the cables' | NCCL's socket netdev on every rank, and where the ranks meet, when not over the cables |

The serving settings (`THINKING`, `REASONING_EFFORT`, `THINKING_BUDGET`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `DRAFTS`,
`MAX_TOKENS`, `API_KEY`, `HOST`, `PORT`, `SERVED_NAME`, `GRAPHS`, `SIDE`, `PREFETCH`, `ENGRAM_AIO`, `REPLAY_FLOOR`,
`RDMA`, `MEMGUARD_*`, `HEAD_NFS_SERVER`) work as at `TP=2`. **Refused at `TP=3`**, before anything is stopped, because
only the two-Spark engine has them (its `zig/patches` and `scripts/port.sh`): `SPILL_GIB` above 0, `KV_SHARD=1`,
`EAQ_V1_1=1` (the mixed-pack loader), `GRAPH_CAP` (0033), `RDMA_IDLE_S` and `RDMA_NAP_US` (0010),
`PREFILL_INTERLEAVE_MS` (0005), `TF_DS_CONNECT_S` (0037) and `HEAD_ENGRAM_DIR` (rank 0 reads its own Engram rows).
The patches that change server behavior without a setting (the Anthropic API's leniencies, the stream keepalive
pings, the `GET`/`DELETE` body discard of 0046, the cached-prompt metrics, the Engram failure paths of 0035) are not in
LakyFx's engine either.

## What prepare.sh does at TP=3

Steps 1 to 3 as at `TP=2`, for both workers: the preflight with the triangle and its routes, the image (built from
`zig/Dockerfile.tp3`: LakyFx's repository at the pinned commit, its `extract.py` checking every shipped input and the
runtime snapshot, Zig 0.17.0 building `tensorfold-native` and `tf-dsv41-lanes` with the kit's fatbins, NCCL 2.30.7
switchless from [`alexellis/switchless-nccl`](https://github.com/alexellis/switchless-nccl) `v0.0.1` checked by sha256),
the downloads, the `ABLIT=1` overlay. Then `scripts/prepare-tp3.sh`:

4. **Receipt and plan.** LakyFx's tools refuse a file that resolves outside its folder, so the checkpoint becomes a
   folder of hard links (`ABLIT=1`: the grafted folder; a shard the overlay left as a symbolic link becomes the stock
   shard's file itself). `scripts/tp3_inputs.py manifest` writes a source manifest from the files (size and sha256 of
   each shard: a stock shard's sha256 is the one its Hugging Face blob is named by; a grafted one is hashed, never
   fetched or "repaired" from the Hub), and LakyFx's `source_verify.py` re-hashes every shard against it into the receipt
   its exporter checks. `scripts/tp3_plan.py` (from easthambuilds' `make_plan.py`) writes the plan: every tensor of the
   checkpoint with its shard and byte range, checked against LakyFx's TF29 plan (Mia's earlier 2.9 bpw export) for every
   one of the 144,286 tensors the ranks read. The default checkpoint's plan is pinned (`TP3_PLAN_SHA256`).
5. **Kit and model folder.** LakyFx's `make_rope.py` makes the RoPE tables at its 10,000,000-position reference pool on
   CPU and checks them against its qualified hashes; `scripts/tp3_inputs.py rope-cut` keeps the rows `POOL` needs (for
   3,300,000, 3,301,376 rows, which must equal LakyFx's qualified 3.3M tables). The model folder holds the checkpoint's
   config for three ranks (LakyFx's `tp3_layout`: 72 heads in 9 groups, a padded vocabulary), the tokenizer, LakyFx's
   qualified chat template (checked by sha256, above) and the vision tower (its 266 BF16 tensors, every byte compared
   with the checkpoint).
6. **Each rank's files.** LakyFx's `rank_cache.py` writes the rank's weight file (`rank<r>of3-EAQ29-<plan>.bin`) and
   `rank_verify.py` re-reads it in full; `prepare_engram.py` cuts the rank's 8 of Engram's 24 hash columns as a sparse
   copy, with its `SOURCE.json`.
7. **The workers.** Each worker's folder gets the kit, the model folder, its Engram rows and its weight file, copied over
   the NFS volume and checked by sha256 there; the head then deletes its copy of the worker's files.

## The launch

`start.sh` starts ranks 1 and 2 on the workers, then rank 0, each with LakyFx's `tensorfold/launch.py` environment:
`TF_DS_EXPERIMENTAL_TP3=1` (the engine refuses three ranks without it), `TF_DS_START_TIMEOUT=1800` (how long the
followers retry), `TF_DS_POOL_TOKENS`, `TF_DS_CACHE_SHORT_CONTINUATION=1` and `TF_DS_TOOL_SEPARATOR_ROUNDTRIP=1` (its
tested profile's cache and tool-history repairs), TensorFold's RDMA ring on all four rails (`TF_RDMA_DEVICES` or
`--rdma`) with each rank's `TF_RDMA_PEER_MASKS` (which rails reach which rank), and NCCL as LakyFx's launcher sets it,
with these settings, which the three Sparks needed with the switchless build:

- `NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_ALGO=Ring`, `NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_IB_MERGE_NICS=0`, all
  four (LakyFx's launcher sets only the last two);
- `NCCL_IB_HCA` the **first device of each cable** only: the switchless build takes two listener GIDs, while
  TensorFold's own RDMA ring uses all four devices;
- `NCCL_SOCKET_IFNAME` each rank's own cable to rank 0 (no network is shared by all three), and `NCCL_IB_ADDR_RANGE`
  the network holding every cable's subnet.

The rank commands are printed by `DRY_RUN=1 ./start.sh`. The smoke test sends one request through all three ranks.
`./stop.sh`, the memory guard on every Spark, `FOREGROUND=1` and the logs work as at `TP=2`.

## What was checked, and what was not

Checked without the Sparks:

- the default checkpoint against LakyFx's layout, from its 39 shards' published headers: all 144,286 tensors the three
  ranks read exist, with TF29's dtypes and shapes and the same EXL3 widths; the 40 `ffn.gate.bias` are F32 where TF29's
  are F16 (the exporter reads them as float32 either way); Engram's projections are F32 as in TF29. The plan written
  from those headers is the pinned one, and LakyFx's `rank_cache.py` (dry run) lays out 143,641 entries a rank,
  71,736,875,432 bytes each;
- the scripts: `scripts/test_tp3.py` (the settings and refusals, the triangle's pairing and routes, the three rank
  commands, starts, restarts and failures with fakes, and `TP=2` compared with the tree before `TP=3`: the same output,
  ssh commands and docker calls), `scripts/test_tp3_inputs.py` (the plan, the manifest, the RoPE cut, `SOURCE.json`, the
  receipt check against LakyFx's own `source_verify.py`), and the existing suites.

Not run yet on three Sparks: the image build, `scripts/prepare-tp3.sh` end to end (the receipt, the RoPE tables, the
rank files, the Engram rows, the copies), the launch, the smoke test, `ABLIT=1` at `TP=3`, the memory left on each
Spark, and any speed or quality measurement. Until then `TP=3` is a recipe for testing, not for serving.

## Credits

LakyFx (the TP3 runtime, its tools, kit and profiles, MIT), easthambuilds (the three-Spark recipe this follows, Apache-2.0),
BertholomusAI (the DeepSeek-V4.1 Zig engine the TP3 port is built on), Alex Ellis / OpenFaaS (the switchless NCCL build),
Ash Hart and the TensorFold contributors. Details in [CREDITS](../CREDITS.md).
