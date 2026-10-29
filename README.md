<h1 align="center">DeepSeek-V4.1-Flash-1017 EXL3 on 2x DGX Sparks with TensorFold</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve **DeepSeek-V4.1-Flash-1017** from two NVIDIA DGX Sparks (GB10, 128 GB each, linked by their ConnectX-7 ports)
through an OpenAI-compatible API, with the full **1,048,576-token window**, **image input**, tool calling, thinking on
or off, and **DSpark** speculative decoding from the checkpoint's own draft blocks, using
[TensorFold](https://github.com/ashhart/TensorFold)'s native Zig engine and the 2.9 bit EXL3 quantization
[`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
(196 GiB; its model card has no perplexity or KL measurement against the unquantized model). EAQ v1 is the recommended
pack; the experimental EAQ v1.1 pack is opt-in with `EAQ_V1_1=1`.

- **API:** OpenAI-compatible, model id `DeepSeek-V4.1-Flash-1017`, port 8888.
- **Engine:** TensorFold native (Zig 0.17.0), BertholomusAI's `dsv41` family at TP=2, one rank on each Spark.
- **One command** on the first Spark: `./start.sh` sets up both Sparks and starts both ranks; `./stop.sh` stops them.
- **Optional:** a [systemd user service](docs/systemd.md) to supervise both ranks (you copy and enable it; the recipe installs nothing),
  and `ABLIT=1`, the matching [Keys overlay by drowzeys](docs/abliterated.md) (needs your Hugging Face token and gated access).

## Performance

Measured on one pair of Sparks (the head also runs a desktop and ~18 small containers; the worker is otherwise idle) with both GPUs
at 2,200 MHz, the server at `CONTEXT=524288`, up to 4 streams, drafts on, real prose and code prompts, as [sparkDash](https://github.com/MiaAI-Lab/sparkDash) reported them:

| Decode, tok/s (total, per stream, TTFT) | 1 stream | 2 streams | 3 streams | 4 streams |
| --- | --- | --- | --- | --- |
| Prose, temperature 0, thinking off | 80.7 (TTFT 133 ms) | 110.6 total, 56.0 each (TTFT 239 ms) | 133.6 total, 45.4 each (TTFT 339 ms) | 154.0 total, 39.9 each (TTFT 401 ms) |
| Prose, temperature 0.7, thinking on | 63.9 (TTFT 214 ms) | 93.3 total, 48.6 each (TTFT 389 ms) | 107.8 total, 38.0 each (TTFT 515 ms) | 120.6 total, 32.0 each (TTFT 646 ms) |
| Code, temperature 0, thinking off | 118.1 (TTFT 213 ms) | 149.4 total, 76.4 each (TTFT 290 ms) | 171.0 total, 58.8 each (TTFT 425 ms) | 166.1 total, 45.0 each (TTFT 516 ms) |

| Prefill, prompt tokens | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tok/s | 2,105.9 | 2,152.0 | 2,154.8 | 2,153.6 | 2,136.2 | 2,077.1 | 2,065.0 |
| TTFT | 2.13 s | 3.99 s | 7.78 s | 15.40 s | 30.86 s | 63.28 s | 127.13 s |
| Prompt tokens counted | 4,114 | 8,215 | 16,403 | 32,788 | 65,554 | 131,092 | 262,163 |

These need **Engram's tables on local disk on both Sparks** (read over NFS, prefill falls to ~690-890 tok/s). The earlier
GPU-clock run, the `tools/bench.py` numbers on random-word prompts, the comparison with BertholomusAI's figures, the 479,012-token
needle test and what was and was not checked are in [docs/measurements.md](docs/measurements.md).

## Requirements

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory) with nothing else large on them: the weights take ~98 GiB a
  Spark, and `start.sh` refuses to start unless `MIN_AVAIL_GIB` (112) is available on each.
- **The same GPU clock ceiling on both** (`nvidia-smi -lgc 0,2200`): the ranks meet twice a layer, so the slower GPU sets the pace.
- **A direct ConnectX-7 link** (a QSFP cable, an IPv4 address on each end in one private subnet, a RoCE v2 GID); address both PCIe
  "twin" netdevs of the port on each Spark and both are used as rails.
- **Key-based ssh** from the head (which runs `./start.sh` and the API) to the worker; Docker with the NVIDIA runtime on both; `git`,
  `python3` and the `hf` CLI on the head.
- **Disk:** ~505 GB on the head plus ~25 GB for the image; ~310 GB on the worker plus the image.
- **NFS:** the worker mounts the head's Hugging Face cache read-only (or both Sparks run from a local copy).

The details, including the NFS setup and its troubleshooting, are in [docs/requirements.md](docs/requirements.md).

## Quick start

On the head:

```bash
git clone https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks.git
cd DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
cp scripts/local.sh.example scripts/local.sh     # set WORKER=user@192.0.2.2 in it (and FABRIC_PEER, see the paragraph below)
DRY_RUN=1 ./start.sh                              # prints what it would do; changes and starts nothing
./start.sh
```

`WORKER` is the worker's ssh target. The ranks talk over the route to it: if it is on another network than the link, set
`FABRIC_PEER` to the worker's CX7 address. The worker needs no copy of this repository.

The first run sets up both Sparks ([what it does](docs/how-it-works.md)): the image on each (two Zig builds, ~6 minutes on a Spark), the checkpoint (196 GiB)
and Engram's tables (~203 GB) downloaded on the head, the kit checked, both ranks' weight files written, Engram's tables
and rank 1's weight file on the worker's disk. Later starts load ~98 GiB of weights on each Spark. `start.sh` shows each
step and the server's log, runs a smoke test through both ranks, and prints `DeepSeek-V4.1-Flash-1017 is now LIVE!
on port 8888` with the endpoint.

Any OpenAI client works with `base_url = "http://<head-address>:8888/v1"` and the model `DeepSeek-V4.1-Flash-1017`.
The model thinks before it answers unless a request turns thinking off, so give replies enough `max_tokens`.

```bash
./start.sh restart                                       # restart both ranks, e.g. after changing a setting
./stop.sh                                                # stop both ranks and free their GPU memory
docker logs -f dsv41-1017-exl3-tf                        # rank 0's log (here)
ssh <worker> docker logs -f dsv41-1017-exl3-tf           # rank 1's log (why it exited, if it did)
python3 tools/validate.py http://<head-address>:8888 --model DeepSeek-V4.1-Flash-1017   # text, thinking, tools, image, long prompt
API_URL=http://<head-address>:8888 python3 tools/toolcheck.py    # also needle.py (a passphrase in ~195k tokens), end_of_turn.py
```

## Main settings

Settings come from the environment, `scripts/local.sh` or `.env`; every setting and its default is in
[`scripts/config.sh`](scripts/config.sh), and [docs/configuration.md](docs/configuration.md) explains them. The ones you are likely to touch:

| Setting | Default | |
| --- | --- | --- |
| `WORKER`, `FABRIC_PEER` | | the worker's ssh target, and its CX7 address if it is on another network than the link |
| `CONTEXT` | 1048576 | the most positions one request may use; on a head that carries other software, 524288 is the practical limit ([memory](docs/memory-and-safety.md)) |
| `PARALLEL` | 4 | requests at once (the family was measured at 4) |
| `THINKING`, `REASONING_EFFORT` | 1, empty | think first unless a request turns it off; the default effort |
| `DRAFTS` | 1 | the checkpoint's DSpark draft blocks (drafted replies equal plain ones) |
| `EAQ_V1_1` | 0 | `1` switches to the experimental EAQ v1.1 pack |
| `ABLIT` | 0 | `1` applies the Keys overlay ([abliterated mode](docs/abliterated.md)) |
| `API_KEY` | 0 | `1` makes rank 0 require a key; the API is open on the network by default |
| `HOST`, `PORT`, `SERVED_NAME` | 0.0.0.0, 8888, `DeepSeek-V4.1-Flash-1017` | |
| `MIN_AVAIL_GIB`, `MEMGUARD_GIB` | 112, 2.5 | the memory `start.sh` demands, and the floor under which `scripts/memguard.sh` kills a rank |
| `SPILL_GIB`, `KV_SHARD`, `POOL` | 0 (off), 0 (off), empty | [spill v2 and KV sharding](docs/spill-and-kv-sharding.md) |

Extra arguments to `./start.sh` go to rank 0's `tensorfold-native serve` after the defaults; every `TENSORFOLD_*`, `TF_DS_*`,
`TF_FILL_*` and `TF_SPILL_*` variable you set goes to both ranks.

## Documentation

- [Requirements](docs/requirements.md): hardware, link, ssh, disk, NFS and what gets downloaded
- [How `start.sh` and `prepare.sh` work](docs/how-it-works.md)
- [Configuration](docs/configuration.md): every setting, and the log settings
- [Memory and safety](docs/memory-and-safety.md): `CONTEXT` and `POOL`, admission, `memguard`, the head's headroom
- [Spill v2 and KV sharding](docs/spill-and-kv-sharding.md) (ported, off by default) and [the port's settings](docs/PORT-RECIPE.md)
- [Abliterated mode](docs/abliterated.md)
- [Status and measurements](docs/measurements.md): what was checked, benchmarks, the engine patches' results
- [The engine](docs/zig-engine.md), [systemd](docs/systemd.md), [Hermes Agent](docs/hermes-agent.md), [fork deploys](docs/FORK-DEPLOY.md)
- [Engine patches](zig/patches/README.md), [CHANGELOG](CHANGELOG.md), [CREDITS](CREDITS.md)

## Repository layout

```
start.sh, stop.sh            start and stop both ranks
scripts/config.sh            every setting and its default (layered: environment, scripts/local.sh, .env)
scripts/nodes.sh             the worker over ssh, the CX7 link and its RoCE rails, NFS volume, NCCL settings
scripts/prepare.sh           image, downloads, kit, token map, weight files, the worker's NFS volume and disk
scripts/memguard.sh          the memory watchdog start.sh runs on both Sparks (kills a rank's container below MEMGUARD_GIB)
scripts/zig_flags.py         start.sh's check of rank 0's serve flags against the image's `capabilities --json`
scripts/port.sh              the ported features' settings (spill v2, KV sharding, POOL), checks and engine mapping
scripts/port_test.sh         checks of scripts/port.sh (no Sparks needed)
scripts/port-patches.sh      exports engine work after the port's series as zig/patches/0024-..., checked on a clean TF_REF
scripts/publish-image.sh     pushes the built image to ghcr.io (only on purpose)
zig/Dockerfile, zig/patches  the engine image; patches to the pinned engine (see `zig/patches/README.md`)
model/chat_template.jinja    the chat template rank 0 serves
tools/validate.py            an end-to-end check of a running server
tools/bench.py               prefill, decode, multi-stream and needle measurements as JSON, with `--compare A.json B.json`
tools/exact.py               records a server's replies (token hashes) on a fixed prompt set and compares two records: proves a new image equals the old one
tools/smoke.py               a quick correctness check: health, stop strings, streaming, a tool call and its result
tools/prompt_reuse.py        a conversation resumes from its kept prompt state (from the GLM-5.3-Flash recipe)
tools/toolcheck.py, needle.py, end_of_turn.py, anthropiccheck.py   single checks, from the GLM-5.3-Flash recipe
docs/zig-engine.md           how the pieces fit, and what is not verified
docs/measurements.md       what was checked and measured on a pair of Sparks
docs/requirements.md, how-it-works.md, configuration.md, memory-and-safety.md, abliterated.md, spill-and-kv-sharding.md
                             the recipe's setup, settings and safety notes, one page each
docs/PORT-RECIPE.md          the GLM port's settings, their engine mapping, deploy and rollback
docs/FORK-DEPLOY.md          updating and rolling back the port fork that serves on gx10-a
docs/hermes-agent.md         Hermes Agent as a client (by Steve Darlow)
```

## License

This recipe's own code, scripts and documentation: Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). The weights keep
their own MIT license (DeepSeek's; Mia's AI Lab's quantization). TensorFold, BertholomusAI's fork and kit, and the other
third-party parts keep theirs ([`NOTICE`](NOTICE)).

## Credits

- **[DeepSeek](https://huggingface.co/deepseek-ai)**: DeepSeek-V4.1-Flash-1017, its weights, its DSpark draft blocks, its
  Engram tables and its reference inference code (MIT).
- **Mia's AI Lab** ([Mia-AiLab](https://huggingface.co/Mia-AiLab)): the EAQ v1 EXL3 2.9 bpw quantization served by default,
  [`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
  (MIT), the experimental EAQ v1.1 pack
  [`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw)
  (opt-in with `EAQ_V1_1=1`), the earlier release
  [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) whose
  per-tensor widths they equal, and the chat template (`model/chat_template.jinja`, written for that earlier release).
- **[turboderp](https://github.com/turboderp-org)**: [ExLlamaV3](https://github.com/turboderp-org/exllamav3), the EXL3 format
  and its quantizer (MIT).
- **[drowzeys](https://huggingface.co/drowzeys)**: authored the optional [Keys abliteration overlay and Mia EAQ 2.9 bpw
  application script](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated/tree/0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0)
  used by `ABLIT=1` (MIT).
- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors (Apache-2.0): the native Zig engine and its OpenAI-compatible server.
- **[BertholomusAI (Albert Lee, @bertholomus)](https://github.com/bertholomus)** (Apache-2.0): the DeepSeek-V4.1-Flash
  family on TensorFold's native engine (the two-node tensor-parallel split, the EXL3 kernels' use, Engram, DSpark drafting,
  the RDMA ring, the vision tower), in their fork
  [bertholomus/TensorFold, branch `deepseek-v41-zig`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig) at
  [`fbbc971`](https://github.com/bertholomus/TensorFold/commit/fbbc97132cd004237d9887b8d3185896efaedc5c) (the pin of pull request #77), and its
  weight-file and token-map tools, in their branch
  [`deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4) at
  [`5940208`](https://github.com/bertholomus/TensorFold/commit/59402086381a42c74d53053443224eec2790b4c5); the kernel kit,
  the served settings and the two-node launch commands, in their recipe
  [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) at
  [`abe07c7`](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/commit/abe07c7d509a0199cc7fec15263be2aa52ac652f).
  Commits in that branch by [sayyidfareed](https://github.com/sayyidfareed) (RDMA GID and MTU),
  [Lucas Peters](https://github.com/lucas-peters) (indexer alignment; the kit's `_topk_sel` variant), lepetitprince716-prog
  (expert-run loading), LeetJoe (image placeholder escaping) and Manrique ([hashd1ve](https://github.com/hashd1ve)) (kept-state fixes,
  the prompt pass in pieces, image prompt states, DSML tag reading).
- **[Coinupbtc](https://github.com/Coinupbtc)** (@Coinupbtc) authored `zig/patches/0005-prefill-interleave.patch`, in
  [pull request #5](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5) (commit [`27e02c6`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5/commits/27e02c605eae0997ed9161baaf583dd3ad431bb3)): decode rounds and cancels for the other live streams between the chunks of a long prompt pass.
- **Hector Isaza** ([hecisaza](https://github.com/hecisaza)) authored `zig/patches/0035-engram-lead-failure-paths.patch`, in
  [pull request #79](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/79) (commit
  [`549f473`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/79/commits/549f473d4a5fd2ba39dc6eb666255a18bece098e)):
  an Engram prompt read that fails on one rank ends the pair instead of hanging it, a table shorter than the hasher's row range is
  refused at start, and the Engram readers and the AIO, `Model.open`, `Pool.init` and `Tables.open` failure paths release what they made.
- **eleata** ([eleata](https://github.com/eleata)) authored the engine pin `fbbc971` and the rebase of the patch series onto it, in [#77](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/77).
- **Maciej Kowalik** ([koniaik](https://github.com/koniaik)) authored `zig/patches/0029` to `0032` (the Anthropic frontend's thinking
  budget, `clear_thinking`, mid-conversation `system` and the stream keep-alive ping), in [#53](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/53), [#71](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/71), [#72](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/72) and [#76](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/76).
- **E-Zou Shen** ([ezoushen](https://github.com/ezoushen)) authored `zig/patches/0033-graph-cap.patch` (`GRAPH_CAP`), in [#70](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/70).
- **Coinupbtc** also authored `zig/patches/0027` (the text before a reply's tool calls) and `0028` (a request's own text never becomes a
  control token), in [#44](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/44) and [#43](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/43).
- **Qingchen Yu** ([markyu0401](https://github.com/markyu0401)) authored the check that the ranks' Engram folders match before a stop, in [#64](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/64).
- **Mykhailo Tamarin** ([MykhailoTamarin](https://github.com/MykhailoTamarin)) authored the EAQ v1.1 support (`EAQ_V1_1=1`), in [#57](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/57).
- **Robert Wojciechowski** ([wojo](https://github.com/wojo)) authored the GLM-5.3-Flash recipe's spill tier and KV sharding that
  `zig/patches/0011` to `0022` port, the port's scripts, and `zig/patches/0023` (also his [#40](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/40)); **[BadAd84](https://github.com/BadAd84)**
  authored the spill block tier's design and checks there.
- **aaa4xu** ([aaa4xu](https://github.com/aaa4xu)) authored the indexer split's default from token 131,072, in [#35](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/35);
  **aoeu10** ([aoeu10](https://github.com/aoeu10)) the worker file-view check, in [#25](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/25); **Matthew Vandergrift**
  ([mvandergrift](https://github.com/mvandergrift)) the worker cache cleanup without bare globs, in [#19](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/19);
  **yume-arasaki** ([yume-arasaki](https://github.com/yume-arasaki)) `tools/bench.py`'s port default, in [#39](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/39);
  **sunlinhua-dotcom** ([sunlinhua-dotcom](https://github.com/sunlinhua-dotcom)) the `ABLIT=1` stock-dtype fix, in [#12](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/12).
- **[SvangenStudios](https://github.com/SvangenStudios)** authored `stop.sh`'s exit status, the prepared marker and `scripts/test_scripts.py`, and
  `tools/validate.py`'s checks and tests, in pull requests [#4](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/4) and [#5](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/5) of the 4-Spark recipe;
  **[manuel1618](https://github.com/manuel1618)** the cached prompt-token counter in `/metrics`, in [#8](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/8) there, a backport of
  the TensorFold commit `ce0d9053` by **[juliankang4](https://github.com/juliankang4)**; **[shanedertrain](https://github.com/shanedertrain)**
  diagnosed the early-rendezvous problem in [that recipe's issue #3](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/3).
- **Mia's AI Lab**: this recipe's scripts, adapted from
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (branch `zig-engine`), with code by webzone (both PCIe links of a port as twin rails), [Alexbob0](https://github.com/Alexbob0)
  (workers by host name; `tools/end_of_turn.py`) and john (each rank's rendezvous address); `tools/anthropiccheck.py` by
  Eduardo Florencio ([eduffd](https://github.com/eduffd)), and the Hermes Agent page by Steve Darlow
  ([@kerpopule](https://github.com/kerpopule)).
- **hashd1ve** ([hashd1ve](https://github.com/hashd1ve)): the change to `model/chat_template.jinja` that writes each piece of the
  prompt out as it is made instead of adding it to one growing string, which cut rank 0's memory growth with long
  conversations ([issue #1](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/1)).
- **Mia's AI Lab**: this recipe's own engine patches, `zig/patches/0001` (a change to BertholomusAI's engine, written
  for this recipe, Apache-2.0: the prefill indexer split between the two ranks); `0002` (a second AIO ring for Engram reads) and
  `0006` (image prompt states) were retired when the pin moved to `fbbc971`, and a prompt-cache placement patch before them when the
  pin moved to `629c524`; `0037` and `0038` (the early rendezvous port; the 16-byte aligned index views); also `tools/exact.py` (token-hash replay check) and `scripts/memguard.sh`.
  The idea of splitting the indexer is drowzeys's, from the GLM-5.3-Flash recipe's design (no code of it was used).
  `0004` (`tool_choice` enforcement) is Mia's AI Lab's Zig call gate from its Flash Next CUDA engine, carried into the
  DeepSeek-V4.1 lanes; that gate is a port of the call gate in TensorFold's Python engine (`call_gate.py`), by the
  TensorFold contributors (Apache-2.0).
- **NVIDIA**: the DGX Spark and GB10 hardware, and the PyTorch container (`nvcr.io/nvidia/pytorch:26.07-py3`) the image is
  built on (CUDA, NCCL, cuBLAS; under NVIDIA's container license).

Full list, with commits: [`CREDITS.md`](CREDITS.md).
