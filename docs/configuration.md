# Configuration

Settings come from the environment, `scripts/local.sh` or `.env`; the defaults and their notes are in
[`scripts/config.sh`](../scripts/config.sh). The ones you are likely to touch:

| Setting | Default | |
| --- | --- | --- |
| `EAQ_V1_1` | 0 | `0`: EAQ v1 (the recommended pack); `1`: the experimental EAQ v1.1 mixed pack (27 BF16 indexer/compressor linears; the image's mixed-pack loader handles them) |
| `MODEL_ID`, `MODEL_REVISION` | `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` @ `45ee5d0f` | the checkpoint. `EAQ_V1_1=1` selects `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw` @ `aee640e2`; an explicit `MODEL_ID`/`MODEL_REVISION` wins. The Hugging Face cache is keyed by the repo's name and revision: a copy under another name or revision is not found under the default, and `scripts/prepare.sh` would download 196 GiB. It stops when the disk cannot hold that. To keep serving a copy you have, set both to what `~/.cache/huggingface/hub` holds. v1.1's 27 BF16 linears load through the image's mixed-pack loader (`zig/patches/0024` + `zig/patches-inputs/0001`); the load prints `mixed pack: 27 float linears`. An EXL3-only pack (EAQ v1) has no plain entries and takes the old path |
| `ABLIT` | 0 | `0`: stock Mia EAQ 2.9 bpw; `1`: apply the matching gated Keys overlay; requires your HF token and repository access (see [Abliterated mode](abliterated.md)) |
| `ABLIT_REVISION` | `0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0` | pinned Keys overlay and application script; full commit hash |
| `HF_TOKEN` | unset | Hugging Face read token for downloads; alternatively authenticate with `hf auth login` on the head |
| `CONTEXT` | 1048576 | a request's positions at most; rank 1's `--pool` follows it, or `POOL` |
| `POOL` | empty | positions the streams' extents and kept prompts share, when more than `CONTEXT` (empty: `CONTEXT`; not below it; `zig/patches/0008`): more long prompts stay resident and more long requests run at once. A request still stops at `CONTEXT`. Its KV caches grow with it, about 0.9 GiB a rank per 1,048,576 positions, in host memory: at `POOL=2097152` the head's idle `MemAvailable` went from 6.76 to 5.71 GiB and its lowest during a cold 96K-token prompt from 5.89 to 4.49 GiB (worker: 8.63 to 7.43 and 8.60 to 7.33), so measure the head's headroom before raising it. With `KV_SHARD=1` the same `POOL` is cut into one part a Spark (each must hold a whole `CONTEXT` window; empty: the pool sharding needs, 2101248 at the 1M window), and each Spark keeps only its own part's compressed KV: about half the KV memory a Spark for the same `POOL` (measured at 2,101,248: 1,062 MiB a rank against 1,783 MiB replicated) |
| `PARALLEL` | 4 | requests at once (the engine's streams, 1 to 16; the family was measured at 4) |
| `THINKING`, `REASONING_EFFORT` | 1, empty | think first unless a request turns it off; `low` / `medium` / `high` / `xhigh` as the default effort |
| `THINKING_BUDGET` | empty | reasoning tokens a reply may spend before the engine closes the think block itself and lets the answer continue (`--thinking-budget`; a request's `thinking_budget` wins); empty: no cap. A hard task can otherwise spend its whole `MAX_TOKENS` thinking and end with no text and no tool call. A model that keeps rambling after the cut is not stopped by it. For long agentic or tool-calling jobs, leave it empty or set it large: a small budget cuts the reasoning mid-plan, which can make the model repeat a tool call it already made |
| `TEMPERATURE`, `TOP_P`, `TOP_K` | 1.0, 0.95, 20 | the server's sampling defaults; a request's own values win. This engine refuses a `top_k` above 56 with `TooManyCandidates` (the finite candidate list its sampler keeps), so `TOP_K` is at most 56; `top_k` 0 or -1 is refused too, with a different, misleading message |
| `DRAFTS` | 1 | the checkpoint's DSpark draft blocks (drafted replies equal plain ones) |
| `RDMA`, `GRAPHS`, `SIDE`, `PREFETCH`, `ENGRAM_AIO` | 1 | the engine's served settings |
| `GRAPH_CAP` | 128 | the CUDA graphs each rank keeps with `GRAPHS=1`; past it the least recently replayed one is destroyed and captured again when needed (`zig/patches/0033`, issue #69). `0` keeps every graph: the cache then grows for days with growing conversations and is never freed. A count that keeps rising in the log ("destroyed for newer ones") means the cap is below the working set; that costs a capture per stretch (about 5% of single-stream decode at worst on the pair), not memory |
| `RDMA_IDLE_S`, `RDMA_NAP_US` | 2, 500 | the RDMA proxy naps after 0–3600 quiet seconds, sleeping `RDMA_NAP_US` (1–1000000 microseconds) between looks (`zig/patches/0010`); `RDMA_IDLE_S=0` keeps the stock always-spinning proxy. The patch's engine fallback is 100 µs if `TF_RDMA_NAP_US` is unset; the recipe sets 500 µs because the Zig ring's two CQ polls per wake cost more than the Python line's. 2 s is enough that active decode never naps |
| `MAX_TOKENS` | 32768 | a reply's budget when a request sets none |
| `MIN_AVAIL_GIB`, `MIN_AVAIL_WAIT_S` | 112, 30 | `start.sh` refuses to start with less memory available than this on either Spark (`0` skips the check); it re-reads the memory for up to `MIN_AVAIL_WAIT_S` seconds first, because the page cache a build or a large copy just left counts as available only once it is written back (`0` reads once) |
| `MEMGUARD_GIB`, `MEMGUARD_HOLD_S`, `MEMGUARD_HARD_GIB` | 2.5, 4, 1 | `scripts/memguard.sh` kills a rank's container when the memory available stays under the floor for `MEMGUARD_HOLD_S` seconds without coming back over it, or at once when it falls under the hard floor. A dip that recovers is logged and not killed (decimals fine; `MEMGUARD_GIB=0` turns the guard off) |
| `HEAD_ENGRAM_DIR` | empty | a local folder with Engram's two shards for rank 0 (as `WORKER_ENGRAM_DIR` is on the worker); empty reads them from `HF_CACHE` |
| `HEAD_NFS_SERVER`, `HEAD_NFS_VOLUME` | empty | rank 0 reads the checkpoint over NFS through a docker volume the docker daemon mounts (no root, survives a reboot); empty uses `HF_CACHE` as a host path. sshfs does not work: it lists files as type "unknown", which the engine skips |
| `TF_DS_ENGRAM_LEAD`, `TF_DS_IDX_SPLIT` | unset (the indexer split from token 131,072) | the engine's and its patches' switches (`zig/patches/`, see [Status and measurements](measurements.md)); a `TF_DS_*` variable you export goes to both ranks |
| `TF_FILL_CHUNKS`, `TF_FILL_ROUNDS`, `TF_FILL_FIRST`, `TF_FILL_CUT_IN` | unset (each prompt pass runs whole) | the engine's prompt pass in pieces between the other streams' rounds (`a27d2f0`, `45c25b1`) and a short prompt cutting in between them (`16701b4`); read by rank 0's server, passed by `start.sh` like a `TF_DS_*` variable. With `TF_FILL_CHUNKS` set, `0005`'s interleave stays off |
| `RANK1_PROFILE` | empty | `1` or `2` starts rank 1 with `--profile`, which prints its prefill phase timings when rank 0 stops first (adds a sync per chunk: for attribution, not for timing) |
| `API_KEY` | 0 | `1` makes rank 0 require a key (`Authorization: Bearer`; generated in `$STATE_DIR/api-key`, mode 600). The API is open on the network by default |
| `HOST`, `PORT`, `SERVED_NAME` | 0.0.0.0, 8888, `DeepSeek-V4.1-Flash-1017` | |
| `WORKER`, `FABRIC_PEER`, `NFS_*` | | see [Requirements](requirements.md) |
| `TF_REF`, `INPUTS_REF`, `KIT_REF` | pinned | the engine, the inputs' tools and the kit (changing one rebuilds or re-checks) |
| `SPILL_GIB`, `SPILL_DIR`, `WORKER_SPILL_DIR`, `SPILL_FLOOR_GIB`, `SPILL_QUOTA_GIB` | 0 (off), beside `HF_CACHE`, `SPILL_DIR`, 50, 0 | spill v2: kept prompt states on each Spark's own NVMe ([Spill v2 and KV sharding](spill-and-kv-sharding.md)) |
| `KV_SHARD`, `KV_SHARD_DECODE` | 0 (off), empty | KV sharding: the same `POOL` cut into one part a Spark, each Spark keeping only its part's compressed KV (same section) |
| `ZIG_MIRRORS` | `https://zig.linus.dev/zig` | where the image build downloads Zig first (ziglang.org last), with bounded retries; the bytes are pinned by `ZIG_SHA256` |
| `PULL` | 1 | `0` never pulls a published image (`GHCR_IMAGE`, `IMAGE_TAG`, `IMAGE_DIGEST`; the image of these pins is `ghcr.io/miaai-lab/deepseek-v4.1-flash-1017-2x-dgx-sparks-tensorfold:exl3-tp2-7c9bf5d36350`) |

Extra arguments to `./start.sh` go to rank 0's `tensorfold-native serve` after the defaults; every `TENSORFOLD_*`,
`TF_DS_*` and `TF_SPILL_*` variable you set goes to both ranks. Images are always on (rank 0 runs the vision tower).

## Logs

`stop.sh` saves each rank's log (gzipped, newest 10 kept, `LOG_KEEP`) to `~/.cache/dsv41-tensorfold/logs` before it removes
the container. While a rank runs, docker rotates its log at `LOG_MAX_SIZE` (`100m`), keeping `LOG_MAX_FILE` (3) files: the
engine writes a line for every response, so a monitoring poll (`/health`, `/metrics`) no longer grows it without end (#49).
`LOG_MAX_SIZE=` (empty) leaves the docker daemon's own log settings.
