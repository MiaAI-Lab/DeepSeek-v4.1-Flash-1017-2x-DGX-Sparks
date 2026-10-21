# zig/patches

Patches to BertholomusAI's TensorFold fork at `TF_REF` (`scripts/config.sh`), applied in filename order by `zig/Dockerfile`
(`git apply` in the checkout's root). The image tag changes with them.

| Patch | Switch (both ranks) | Default | What it does |
| --- | --- | --- | --- |
| `0001-c7-indexer-split.patch` | `TF_DS_IDX_SPLIT=1` | off | splits the prefill indexer's row blocks between the two ranks (idea: drowzeys, GLM-5.3-Flash recipe) |
| `0002-c3-engram-lanes.patch` | `TF_DS_ENGRAM_LANES=2` | off | a second AIO ring for a prompt chunk's Engram reads |
| `0005-prefill-interleave.patch` | `TF_DS_PREFILL_INTERLEAVE_MS=N` (rank 0 only; recipe setting `PREFILL_INTERLEAVE_MS`) | off | between two chunks of a long prompt pass, rank 0 runs the other live streams' decode rounds for up to N ms and takes their cancels, so a cold long prompt slows a chat instead of freezing it (issue #3). **By Coinupbtc**, [pull request #5](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5), tested there for bit-identity (long prompt and chat replies equal the unpatched solo ones, with and without drafts) |
| `0006-image-prompt-cache.patch` | `TF_DS_IMAGE_CACHE=0` turns it off (server; rank 0) | **on** | a prompt with pictures resumes and keeps prompt states, keyed by the pictures' content; a request with pictures and no key never touches the cache (issue #2) |
| `0004-call-gate.patch` | none (always on) | **on** | enforces `tool_choice` `"required"` and a named function: the answer opens with the first token of the template's tool-call markup (for this model, the `<` before `｜DSML｜`) and finishes an offered tool name; requests without `tool_choice` are unchanged (taken from the spark1/spark2 deployment's copy of this recipe; tested: required and named choices return tool calls, auto/none/absent unchanged) |
| `0007-spill-kvshard.patch` | the ported settings: `SPILL_GIB`, `KV_SHARD`, `POOL` ([docs/PORT-RECIPE.md](../../docs/PORT-RECIPE.md)) | engine: off; **recipe: on** (`SPILL_GIB=64`, `KV_SHARD=1`, `POOL` derived; `0` turns each off) | Robert Wojciechowski's (wojo) GLM-5.3-Flash patches 0088/0110/0111/0118/0119 (spill v2, continuation, KV sharding, parked pages; GLM PRs #78 #124 #128 #152 #149), ported by him to the native dsv41 engine (engine `port/integration` @ `569bdd12f9e4`, exported by `scripts/port-patches.sh`) <!-- port-patches --> |

With every switch unset (for `0007`: `SPILL_GIB=0 KV_SHARD=0`, since the recipe sets them by default), and for requests without a `tool_choice`, the engine behaves like the pinned one (`tools/exact.py` compares token hashes; bit-identity verified for 0001 and 0002 on the earlier pin; 0004 is checked by the same replay; the image of the current pin was not run on a pair). Remove a patch from
this folder to leave it out of the image; never run `scripts/prepare.sh` to rebuild with a partial cache.

**The port of Robert Wojciechowski's ([wojo](https://github.com/wojo)) GLM patches** (spill v2, continuation, KV sharding, parked pages; engine branch
`port/integration` of the port) arrives here as `0007-...` and later, written by `scripts/port-patches.sh`, never by hand:
it exports the engine branch relative to `baseline-e65785c` (TF_REF plus the patches above), refuses unless every patch
passes `git apply --check` on a clean `TF_REF` checkout after the hand-written ones and the result equals the branch, and
rewrites the table rows it owns (marked `<!-- port-patches -->`). Every switch of these patches is off by default in
the engine; the recipe turns spill v2 and KV sharding on by default (`SPILL_GIB=64`, `KV_SHARD=1`, the pool derived from
`CONTEXT`; `SPILL_GIB=0` / `KV_SHARD=0` turn them off). The recipe's settings for them (`SPILL_GIB`, `KV_SHARD`, `POOL`, ...) are in `scripts/port.sh`, and
[docs/PORT-RECIPE.md](../../docs/PORT-RECIPE.md) has their mapping, deployment and rollback. `scripts/port-patches.sh --dry-run`
checks without writing.

`0003-c1-cache-placement.patch` (`TF_DS_CACHE_PLACEMENT=1`: a resumed prompt moves its kept rows to a free extent instead of over another
conversation's) is gone: the pin moved to `629c524`, whose `lanes.zig` does this itself (`placeSoft` leaves each kept extent room to grow, a
conversation that shares a prefix copies the shared rows to a free extent, `9f82099`/`629c524`), and the patch no longer applies to it. A
`TF_DS_CACHE_PLACEMENT` in your `scripts/local.sh` is ignored now.
