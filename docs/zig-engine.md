# The engine: what is in the image, and what is not verified

`./start.sh` serves the EXL3 checkpoint with TensorFold's native Zig engine, `tensorfold-native`, carrying BertholomusAI's
`dsv41` family (the DeepSeek-V4.1-Flash model on TensorFold's CUDA lane core). Two ranks, one on each Spark; rank 0
serves the API and drives the rounds, rank 1 (`tf-dsv41-lanes`) follows. Credits: [`../CREDITS.md`](../CREDITS.md).

## What is in the image

`zig/Dockerfile`, built by `scripts/prepare.sh` (no GPU needed), on NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`:

| Part | Source |
| --- | --- |
| `tensorfold-native`, `tf-dsv41-lanes` | BertholomusAI's `bertholomus/TensorFold`, branch `deepseek-v41-zig`, commit `629c524` (`TF_REF`), plus `zig/patches/*.patch` (three), built with Zig 0.17.0 and nvcc (sm_121 fatbins are embedded in the binary) |
| `tf-dsv41-rank-cache`, `make_token_map.py`, `make_rope.py`, `make_bias_vl.py` | the same fork's branch `deepseek-v41-zig-tp4`, commit `5940208` (`INPUTS_REF`); only these four files are kept |
| the kernel kit (`TF_DS_KIT`, `TENSORFOLD_CUDA_KERNELS=<kit>/aot`) | not in the image: `kit/` of BertholomusAI's `deepseek-v4.1-tensorfold-tp2-2xgb10` at `abe07c7` (`KIT_REF`), fetched by `scripts/prepare.sh` into the Hugging Face cache and read by both ranks over NFS |

The image tag is `tensorfold-dsv41:exl3-tp2-<hash>` of the patches, the Dockerfile and the pins (`IMAGE` overrides it).

## Why two engine checkouts

The `deepseek-v41-zig` branch is the two-node one: it serves TP=2, images and the RDMA ring, and has the later fixes
(RDMA GID and MTU, the prompt indexer's alignment, the loader's expert runs, the image placeholder). But it writes a rank's
weight file (`TF_DS_RANK_CACHE`) and Engram's token map only from the Python engine's first start (the
`deepseek-v41-tp2` engine on a GPU, on both nodes). The `deepseek-v41-zig-tp4` branch grew `tf-dsv41-rank-cache`, which
writes the same file from the checkpoint alone (no GPU, no Python; WORLD 1, 2 or 4), and the standalone scripts. Its
file format is the Python loader's ("both engines load from the one file", `rank_cache.zig`); for `--parity` off at
WORLD 2 its split plan (`plan.zig`) is the two-node branch's, the diff between the branches being the parity split of the
four-node lane. So the image builds the first branch to serve and takes those four files from the second.

## Inputs

| Input | Made by | Where | Size |
| --- | --- | --- | --- |
| rank 0's weight file `rank0of2-<hash>.bin` | `tf-dsv41-rank-cache $MODEL <dir> 0 2` | head, `<HF_CACHE>/tensorfold-dsv41/<rev>/rank-cache` | 98.9 GiB |
| rank 1's weight file `rank1of2-<hash>.bin` | the same, `1 2`, from the NFS volume | worker, `WORKER_CACHE_DIR` | 98.9 GiB |
| Engram's token map | `make_token_map.py` | head, `<...>/dsv41_token_map.json` (the worker reads it over NFS) | 0.9 MB |
| Engram's tables | `hf download` of shards 47 and 48 of the base model | head (its cache); a copy in `WORKER_ENGRAM_DIR` | ~203 GB |
| the four RoPE tables, the image routing bias | `make_rope.py`, `make_bias_vl.py` | in the kit folder | 4 x 136 MiB, 43 small tensors |

## Settings

How `scripts/config.sh` reaches the two commands (the flags and variables are those of BertholomusAI's two-node README):

| Setting | Rank 0 (`tensorfold-native serve`) | Rank 1 (`tf-dsv41-lanes`) |
| --- | --- | --- |
| `CONTEXT` | `--context` | `--pool` |
| `PARALLEL` | `--parallel`, `TF_DS_STREAMS` | `--streams`, `TF_DS_STREAMS` |
| `ROUND_ROWS` | `TF_DS_ROUND_ROWS` | `--round-rows` |
| `DRAFTS` | `--no-drafts` when 0 | `--drafts` |
| `GRAPHS`, `SIDE`, `PREFETCH`, `ENGRAM_AIO` | `TF_DS_GRAPHS`, `TF_DS_HC_SIDE`, `TF_DS_L2_PREFETCH`, `TF_DS_ENGRAM_AIO` | `--graphs`, `--side`, `--prefetch`, `--aio` |
| `RDMA` | `TF_RDMA_DEVICES`, `TF_RDMA_GID_INDEX` | `--rdma`, `TF_RDMA_GID_INDEX` |
| `RDMA_IDLE_S`, `RDMA_NAP_US` | `TF_RDMA_IDLE_S`, `TF_RDMA_NAP_US` | the same, through `ENGINE_ENV` (`zig/patches/0010`; recipe default 2 s / 500 µs) |
| `REPLAY_FLOOR` | `TF_DS_REPLAY_FLOOR` | `TF_DS_REPLAY_FLOOR` |
| `MAX_TOKENS`, `THINKING`, `REASONING_EFFORT`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `SERVED_NAME`, `HOST`, `PORT` | the same-named serve flags | |
| `MASTER_ADDR`, `MASTER_PORT` | `TF_TP_MASTER`, `TF_TP_PORT`, `TF_TP_WORLD=2` | the positional arguments |
| `HEAD_ENGRAM_DIR` | `TF_DS_ENGRAM=/engram` (the folder mounted there) | `--engram /engram` from `WORKER_ENGRAM_DIR` |
| `API_KEY` | `--api-key-file` (the key file mounted read-only) | |
| `MIN_AVAIL_GIB`, `MEMGUARD_GIB` | none: `start.sh` and `scripts/memguard.sh` read `MemAvailable` on each Spark | |

Rank 1 takes the round's settings as flags only (it reads no `TF_DS_GRAPHS` and the like), and its pool and streams must
equal rank 0's context and streams, or the two ranks wait on each other.

## Not verified

Served and measured on one pair of Sparks (README, Status). Still open:

- other cabling and hardware: the two rails (the cabled port and its PCIe twin) carry traffic on the pair this was run on,
  with zero RDMA error counters, but `TF_RDMA_GID_INDEX` comes from the GID `scripts/nodes.sh` found on that pair;
- memory on a head with less to spare: with ~6 GiB of other software the head had ~3.1 GiB left at the worst point of a
  100K-token prompt at the full window (`MIN_AVAIL_GIB` is a fixed 112, not a calculation from `CONTEXT`);
- image input and `tools/validate.py` on this pair;
- the speed figures are one pair's, medians of one or two repetitions;
- the engine's prefill phases (compute, exchange, Engram) were not profiled: most of the prefill math runs recorded Triton
  and EXL3 kernels from the kit, so a prefill change beyond configuration needs the kit's kernels recaptured.

Before the first serve, `DRY_RUN=1 ./start.sh` shows both commands. If rank 1 exits at start, `docker logs` there says
which input it could not read.
