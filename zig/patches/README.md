# zig/patches

Patches to BertholomusAI's TensorFold fork at `TF_REF` (`scripts/config.sh`), applied in filename order by `zig/Dockerfile`
(`git apply` in the checkout's root). The image tag changes with them.

| Patch | Switch (both ranks) | Default | What it does |
| --- | --- | --- | --- |
| `0001-c7-indexer-split.patch` | `TF_DS_IDX_SPLIT=1` | off | splits the prefill indexer's row blocks between the two ranks (idea: drowzeys, GLM-5.3-Flash recipe) |
| `0002-c3-engram-lanes.patch` | `TF_DS_ENGRAM_LANES=2` | off | a second AIO ring for a prompt chunk's Engram reads |
| `0005-prefill-interleave.patch` | `TF_DS_PREFILL_INTERLEAVE_MS=N` (rank 0 only; recipe setting `PREFILL_INTERLEAVE_MS`) | off | between two chunks of a long prompt pass, rank 0 runs the other live streams' decode rounds for up to N ms and takes their cancels, so a cold long prompt slows a chat instead of freezing it (issue #3). **By Coinupbtc**, [pull request #5](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5), tested there for bit-identity (long prompt and chat replies equal the unpatched solo ones, with and without drafts) |
| `0006-image-prompt-cache.patch` | `TF_DS_IMAGE_CACHE=0` turns it off (server; rank 0) | **on** | a prompt with pictures resumes and keeps prompt states, keyed by the pictures' content; a request with pictures and no key never touches the cache (issue #2) |
| `0008-pool-window.patch` | `TF_DS_POOL=N`, `TF_DS_WINDOW=W` (both ranks; recipe setting `POOL`, which sets both and `--pool` on rank 1) | off | the shared pool may be larger than the window (`--context`): the streams' extents and kept prompts share N positions while a request still stops at W; the RoPE tables are sized by W (a stream's positions run from 0 inside its extent), not by the pool. Applies on its own and in either order with a patch that adds `display.zig` |
| `0004-call-gate.patch` | none (always on) | **on** | enforces `tool_choice` `"required"` and a named function: the answer opens with the first token of the template's tool-call markup (for this model, the `<` before `｜DSML｜`) and finishes an offered tool name; requests without `tool_choice` are unchanged (taken from the spark1/spark2 deployment's copy of this recipe; tested: required and named choices return tool calls, auto/none/absent unchanged) |

With every switch unset, and for requests without a `tool_choice`, the engine behaves like the pinned one (`tools/exact.py` compares token hashes; bit-identity verified for 0001 and 0002 on the earlier pin; 0004 is checked by the same replay; the image of the current pin was not run on a pair). Remove a patch from
this folder to leave it out of the image; never run `scripts/prepare.sh` to rebuild with a partial cache.

`0003-c1-cache-placement.patch` (`TF_DS_CACHE_PLACEMENT=1`: a resumed prompt moves its kept rows to a free extent instead of over another
conversation's) is gone: the pin moved to `629c524`, whose `lanes.zig` does this itself (`placeSoft` leaves each kept extent room to grow, a
conversation that shares a prefix copies the shared rows to a free extent, `9f82099`/`629c524`), and the patch no longer applies to it. A
`TF_DS_CACHE_PLACEMENT` in your `scripts/local.sh` is ignored now.
