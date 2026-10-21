# Credits

This repository is a recipe: scripts, a Dockerfile and a checked list of pins that put other people's work together on
two DGX Sparks. Almost everything that makes it work was built by others. Its own work is licensed under the Apache
License 2.0 ([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE) carries the notices that go with it.

## Model and weights

- **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** by
  [DeepSeek](https://huggingface.co/deepseek-ai): the model, its weights (MIT, Copyright (c) 2023 DeepSeek), its DSpark
  draft blocks and its reference inference code. Engram's two n-gram tables are read unquantized from DeepSeek's own
  shards 47 and 48 of this release; `scripts/prepare.sh` downloads them.
- **[`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)**
  by Mia's AI Lab ([Mia-AiLab](https://huggingface.co/Mia-AiLab)) (MIT): the EAQ v1 EXL3 quantization this recipe serves
  (experimental; its model card measures no quality against the source model). The weights are not part of this repository.
- **[`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)** by
  Mia's AI Lab: the earlier EXL3 release whose per-tensor widths the checkpoint above equals (46,702 comparable tensors, by
  its model card), and the checkpoint the dsv41 family's two-node recipe was written against.
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by turboderp ([turboderp-org](https://github.com/turboderp-org)) (MIT): the
  EXL3 format and its quantizer.

## Engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors (Apache-2.0): the native Zig engine, its OpenAI-compatible server, its lane core, CUDA runtime,
  Triton AOT loader and EXL3 module.
- Whatever else in this recipe is TensorFold's own code, behavior or format (the server and its flags, the API key file's
  format and its permission rule, the lane core, the capabilities report the scripts check against) is credited to the
  TensorFold contributors named above; this recipe only runs it.
- **The DeepSeek-V4.1-Flash family on TensorFold's native engine** was written by
  [BertholomusAI (Albert Lee, @bertholomus)](https://github.com/bertholomus): the `dsv41` family (prompt and decode
  paths, DSpark drafting, Engram, the RDMA ring between two nodes, the vision tower and aligner), in their fork
  [bertholomus/TensorFold, branch `deepseek-v41-zig`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig)
  at commit [`629c524`](https://github.com/bertholomus/TensorFold/commit/629c524a0ea57512a5342abb498d1b7c25353343)
  (Apache-2.0). This recipe builds that commit and serves both ranks with it. Commits by other authors in that branch:
  - **[sayyidfareed](https://github.com/sayyidfareed)**: the RDMA GID index and endpoint MTU negotiation (#12) and a
    buffered descriptor for the direct-IO test (#9);
  - **[Lucas Peters (@lucas-peters)](https://github.com/lucas-peters)**: the prompt indexer's 16-byte-aligned row-block
    pointers (#4);
  - **lepetitprince716-prog**: the loader reading each layer's expert run from the rank cache whole (#18);
  - **LeetJoe**: escaping the image placeholder in a message's own text (#17);
  - **Manrique ([hashd1ve](https://github.com/hashd1ve))**: a prefix two conversations share no longer stales the first one's
    kept states, and the room each kept extent keeps for its conversation's next turn (#26, `629c524`);
  - **[Jay Leaton (jayleaton)](https://github.com/jayleaton)**: the JSON writer escapes a UTF-8 sequence cut off at a string's
    end as U+FFFD instead of slicing past it (`0937cc1`);
  - **[Nathan McBride (nmcbride)](https://github.com/nmcbride)**: the chat-template appends in place when a namespace string
    accumulates, and sizes a new append buffer exactly (`6ddebc6`, `5699d86`);
  - **[sayyidfareed](https://github.com/sayyidfareed)** reported the overwrite of a retained prefix by a growing history
    (TensorFold PR #10) that `9f82099` fixes, by BertholomusAI.
- **The inputs' tools** (`tf-dsv41-rank-cache`, which writes a rank's weight file from the checkpoint alone with no GPU
  and no Python, and the standalone `make_token_map.py`, `make_rope.py` and `make_bias_vl.py`) were written by
  BertholomusAI in their fork's branch
  [`deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4) at commit
  [`5940208`](https://github.com/bertholomus/TensorFold/commit/59402086381a42c74d53053443224eec2790b4c5) (Apache-2.0).
  `zig/Dockerfile` builds that commit as well and keeps only those files; it serves nothing.
- **The kernel kit** (the family's Triton kernels as captured binaries, the EXL3 kernels, Engram's constants, the vision
  tower's attention kernel) is the `kit/` of BertholomusAI's two-node recipe
  [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10)
  at [`abe07c7`](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/commit/abe07c7d509a0199cc7fec15263be2aa52ac652f),
  authored by BertholomusAI (Apache-2.0), with the `_topk_sel` variant for prompts of exactly 512 or 513 tokens by
  [Lucas Peters (@lucas-peters)](https://github.com/lucas-peters) (#13); `scripts/prepare.sh` fetches it at that commit and
  checks it against its MANIFEST. The kit's own licenses (`kit/KIT-LICENSES.md`) cover the parts it carries: Triton (MIT),
  PyTorch's attention kernel (BSD-3-Clause, built on NVIDIA CUTLASS, BSD-3-Clause) and the EXL3 format of ExLlamaV3.
- The served settings `scripts/config.sh` starts with (`TF_DS_REPLAY_FLOOR=1`, CUDA graphs, the mixes' side stream, the
  paced L2 prefetch, Engram by AIO, 1,048,576-token window, 4 streams, temperature 1.0 / top-p 0.95 / top-k 20) and the
  two ranks' launch commands in `start.sh` are BertholomusAI's, from their recipe's README ("Run it (v0.6.0, the Zig
  engine)").

## Scripts

- The scripts' structure, the layered settings (`scripts/config.sh`), the node discovery and NFS volumes
  (`scripts/nodes.sh`), `stop.sh`, the start banner and the image's build pattern come from Mia's AI Lab's
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (branch `zig-engine`) and from Mia's AI Lab's four-Spark DeepSeek-V4.1-Flash-1017 recipe, with code those contributors
  authored in the GLM recipe:
  - **webzone**: both PCIe links of a cabled QSFP port as rails (the DGX Spark "twins") in `scripts/nodes.sh`
    ([`d5afeb0`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/d5afeb0));
  - **[Alexbob0](https://github.com/Alexbob0)**: workers given by host name and `FABRIC_PEER` resolution in
    `scripts/nodes.sh` ([`5e1bf27`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/5e1bf27));
  - **john**: each rank given its link's address of the head as its rendezvous (`LINK_HEAD_ADDR`), kept in
    `start.sh` ([`58eaed0`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/58eaed0)).
- `start.sh`'s `FOREGROUND=1` watch loop (`foreground()`) is adapted from the same recipe's `start.sh` (Mia's AI Lab), cut down to
  the one worker. The `API_KEY=1` option (a mode-600 key file, mounted read-only into rank 0 for `tensorfold-native serve
  --api-key-file`, and the `Authorization` header in `start.sh` and `tools/client.py`) was written for this recipe; the key
  file's format and permission rule are the engine's (`zig/src/server/auth.zig` in TensorFold, Apache-2.0).
- **`model/chat_template.jinja`** was written by Mia's AI Lab ([Mia-AiLab](https://huggingface.co/Mia-AiLab)) for
  [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw): a Jinja
  port of the checkpoint's own `encoding/encoding.py` (DeepSeek, MIT). The file here is that file with one change, by
  **hashd1ve** ([hashd1ve](https://github.com/hashd1ve), issue
  [#1](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/1)): each piece of the prompt is written out
  as it is made (`{{- "" + X -}}`) instead of being added to one growing string (`{% set ns.out = ns.out + X %}`), which the
  `dsv41` engine copied on every append. hashd1ve wrote the change and checked that the rendered prompt is unchanged.
- **`tools/validate.py`** was written by Mia's AI Lab for the four-Spark DeepSeek-V4.1-Flash-1017 recipe.
- Taken from the GLM-5.3-Flash recipe, with the model name and the native server's reply fields changed:
  `tools/client.py`, `tools/toolcheck.py` and `tools/needle.py` by Mia's AI Lab; `tools/end_of_turn.py` by
  **[Alexbob0](https://github.com/Alexbob0)**; `tools/anthropiccheck.py` by **Eduardo Florencio**
  ([eduffd](https://github.com/eduffd)); `docs/hermes-agent.md`, the Hermes Agent setup, by **Steve Darlow**
  ([@kerpopule](https://github.com/kerpopule)); `scripts/zig_flags.py` (the serve flags checked against
  `tensorfold-native capabilities --json`), `scripts/publish-image.sh` and the registry pull in `scripts/prepare.sh`, and the
  issue and pull request templates and `FUNDING.yml` in `.github`, by Mia's AI Lab.
- **`zig/patches/0001-0002`** are changes to BertholomusAI's engine written by Mia's AI Lab for this recipe (Apache-2.0), a second AIO
  ring for Engram reads (0002, off by default) and a row split of the prefill indexer between the two ranks (0001, by default only in
  prompt chunks that end at token 131,072 or later). A prompt-cache placement
  patch (0003) was dropped when the pin moved to `629c524`, which places kept states itself. The idea of splitting the indexer is credited to drowzeys, from the GLM-5.3-Flash recipe's design; no code of
  that recipe was used. They modify code by TensorFold's contributors and BertholomusAI, which keeps their licences.
- **`zig/patches/0005-prefill-interleave.patch`** was authored by **[Coinupbtc](https://github.com/Coinupbtc)** (@Coinupbtc), in [pull request #5](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5)
  (commit [`27e02c6`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5/commits/27e02c605eae0997ed9161baaf583dd3ad431bb3)) of this repository, for issue #3: between two chunks of a long prompt pass, the other live streams get decode rounds and
  their cancels are taken (`TF_DS_PREFILL_INTERLEAVE_MS`), with its host test and the bit-identity runs in the pull request. It is carried here as it
  was sent; **`zig/patches/0006-image-prompt-cache.patch`** (issue #2) was written for this recipe on top of it.
- **`zig/patches/0004-call-gate.patch`** (`tool_choice` enforcement) was written by Mia's AI Lab (Apache-2.0). Its new file,
  `zig/src/core/lanes/call_gate.zig`, is Mia's AI Lab's Zig call gate for the Flash Next CUDA engine (commits `49350ed6` and
  `c948979c` of 2026-10-07, with fixes `06cbbf72` and `1b082936`), which is itself a port of the call gate in TensorFold's
  Python engine (`call_gate.py`, by the TensorFold contributors, Apache-2.0). The rest of the patch wires it into the
  DeepSeek-V4.1 lanes.
- **Written for this recipe** (Apache-2.0, see [`NOTICE`](NOTICE)): `tools/exact.py`, `scripts/memguard.sh` (the memory watchdog), the memory
  preflight (`MIN_AVAIL_GIB`), `HEAD_ENGRAM_DIR` and `HEAD_NFS_SERVER` / `HEAD_NFS_VOLUME` (rank 0's local Engram folder
  and docker NFS volume), and `tools/bench.py` and `tools/smoke.py`. They use the engine's flags, `/health` and
  environment as TensorFold documents or exposes them.
- `tools/prompt_reuse.py` is taken from the GLM-5.3-Flash recipe (`tools/prompt_reuse.py`, Mia's AI Lab, Apache-2.0), with the
  model name, thinking switch and cached-token lookup changed and `--help` added. `tools/bench.py` and `tools/smoke.py` were
  written for this recipe; their measurements follow that recipe's `tools/needle.py`, `tools/toolcheck.py`,
  `tools/hermes_benchmark.py` (Mia's AI Lab; `hermes_benchmark.py` with Steve Darlow, @kerpopule) and the shared prose
  generator of `tools/client.py`.

## Tools and platform

- [Zig](https://ziglang.org) 0.17.0 (MIT), the compiler of the engine.
- NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`, the image's base: its CUDA toolchain, NCCL, cuBLAS,
  libibverbs, PyTorch (which makes the kit's RoPE tables) and the Python and Pillow the engine calls for picture formats
  other than PNG (NVIDIA's container license and the components' own).
- [huggingface_hub](https://github.com/huggingface/huggingface_hub) (Apache-2.0), the `hf` download.

"DeepSeek" belongs to DeepSeek; "DGX Spark" and "GB10" belong to NVIDIA. This recipe is not affiliated with or endorsed
by DeepSeek, NVIDIA, the TensorFold authors or BertholomusAI.
