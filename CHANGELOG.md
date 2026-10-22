# Changelog

## Unreleased

- `zig/patches/0038-step-ahead.patch` (`STEP_AHEAD=1`; off by default): when a reply ends and nothing else runs or waits,
  rank 0 reads it ahead while the client runs its tools. An agent's next request is its last prompt, the reply and the tool
  results, but the prompt cache keeps states only at a prompt pass's chunk ends, so that request resumed at its previous
  prompt's last kept chunk end and read its own previous reply again (on one Hermes Agent deployment: 2,178 of the 3,277
  tokens a step re-read were its own previous reply). The background pass reads prompt + reply, cut at the last chunk
  end that leaves the window, and keeps a state there; it is the next request's own cold pass up to that point, so replies
  keep their bits. A queued pass yields to any arrival; with prompt steps (`0037`) a request that continues a pass that is
  mid-way waits for its state, and one that does not cancels it. The author's agent-loop measurement: time to first token
  of a step 1.99 s to 1.34 s (8 steps, 62-72K tokens). By **Coinupbtc ([Coinupbtc](https://github.com/Coinupbtc))**
  ([pull request #41](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/41)), rebased by **Braincramps ([Braincramps](https://github.com/Braincramps))**
  ([pull request #66](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/66)) and carried onto the prompt steps here; host tests, also run with prompts read a chunk a turn.
  Its two open questions from the review of #41 (a run with `SPILL_GIB` and `KV_SHARD=1`, and whether a pass may park on a
  disk restore) are not answered by a run: it is off by default, and **not run on a pair here**.
- `zig/patches/0037-prompt-steps.patch` (`PREFILL_YIELD=1`; off by default): a prompt can be read a chunk at a time. Until
  now a prompt's whole pass ran inside its admission, so a request that arrived while a long prompt was being read waited
  for all of it (a 250K-token cold prompt: two minutes; the live streams already got decode rounds between its chunks,
  `PREFILL_INTERLEAVE_MS`, but admission waited). With the switch on, rank 0 reads its admitted prompts a chunk a turn, in
  turn, with the live streams' rounds between two chunks: a short request that arrives meanwhile is read after one chunk
  and decodes beside the long one, which finishes about as soon as it would have. Replies are the same bits (each chunk is
  the unchanged 2,048-row chunk, from the same positions; the slot's rows and rings live in the slot between chunks);
  rank 0 sends rank 1 a new frame to return to a prompt another prompt's chunk interrupted, so both ranks need the image.
  Prompts with pictures are read whole either way. Off, the code that runs is the code that ran before. Host tests with
  the fake backend (a late short request completes first, with the same tokens as solo; cancels, a refused request, a full
  pool), and the existing host suites also run with prompts read a chunk a turn (`TF_TEST_PREFILL_STEPS=1`: parked and
  deferred restores, pair break, stall monitor, prompt-cache resume and eviction). **The dsv41 step functions need a GPU
  and have not been run on a pair**: compare replies with `tools/exact.py` and the long-prompt/short-request timing
  before turning it on. Design after the 4x recipe's `0031` by **Mauvis Ledford ([krunkosaurus](https://github.com/krunkosaurus))**
  ([pull request #15](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/15)); re-written for this recipe's spill tier, parked requests, KV parts and pair handling.
- `zig/patches/0036-metrics-cached-prompt-tokens.patch` (always on, issue #58): `/metrics` publishes
  `prompt_tokens_cached_total`, the prompt tokens of finished requests restored from kept prompt states (they are in
  `prompt_tokens_total` too), repeated under vLLM's `prefix_cache_hits_total` / `prefix_cache_queries_total`; `/health`
  carries `prompt_tokens_total`, `cached_tokens_total` (the name sparkDash reads) and `generation_tokens_total`, read
  together with the `/metrics` counters so a scraper never subtracts a cached count from another moment's prompt count;
  `/v1/models` names `max_model_len` (the context window; sparkDash's Context tile reads it). Computed prompt tokens
  are `prompt_tokens_total - cached_tokens_total`. Counters are left out for an engine that keeps no prompt states. The
  counter and its vLLM names are by **manuel1618 ([manuel1618](https://github.com/manuel1618))**
  ([pull request #8](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/8) of the 4x recipe, a backport of BertholomusAI's `ce0d9053`), adapted here with
  `/health` and `/v1/models` added; host tests in `zig/src/server/status_totals_test.zig` and `metrics.zig`.
- `zig/patches/0035-aligned-index-views.patch` (always on): the prompt pass's indexer scores rows in blocks, and a block
  size that was not a multiple of 16 (137,612 compressed keys gave 121-row blocks) put a later block's views off the
  16-byte boundary the kit's kernels were captured for: no matching variant, `MissingTritonVariant`. Blocks are now a
  multiple of 16 rows (a row's scores are its own: no bits change), and a replay cut with an odd first row copies the
  positions to an aligned buffer. From the 4x recipe's `0005` and `0007` (Mia's AI Lab); a host test for the block
  size. Not run on a pair: it may be related to #65 and #36 but that is not shown, so compare replies with
  `tools/exact.py` before relying on it.
- `zig/patches/0034-rendezvous-ports-early.patch` (on): rank 0 opens its follower's port before its weights load and takes
  the connection afterwards. It used to listen only after its own weights had loaded, while the worker retried its
  connect for a fixed 300 s, so a worker that finished loading more than 300 s before the head (the head reading its
  weights cold, say) exited with `ConnectFailed`; now it waits in the port's backlog. `TF_DS_CONNECT_S` (default 300)
  sets that retry window on both ranks; `start.sh` refuses a value that is not 1 to 86400. Host tests: a peer that
  connects before the accept (also through `Fan`) is linked once rank 0 takes its port, and the setting's parsing; a
  `scripts/port_test.sh` check that it reaches both ranks. After the 4x recipe's `0035` (Mia's AI Lab), which found it
  there (its issue #3, diagnosed by shanedertrain).
- `./stop.sh` exits 1 when a rank could not be stopped or a Spark's containers could not be listed (ssh or `docker ps`
  failed), and says which Spark it left alone. Before, such a Spark counted as having no container, stop.sh exited 0,
  and `./start.sh restart` went on to launch a rank while the old one might still hold the GPU's memory. The prepared
  marker (`scripts/config.sh`, `prepared_state`) now covers the chat template's sha256, `HF_CACHE`, `NFS_PATH`,
  `NFS_VOLUME` and each `NFS_SERVER`, quotes every field, is swapped in whole, and a state that cannot be computed
  (the template cannot be hashed) stops `start.sh` instead of comparing equal to a missing marker, which read as "ready".
  The marker's text changes: the next `./start.sh` on a running pair asks for one `./start.sh restart`. By **[SvangenStudios](https://github.com/SvangenStudios)**
  ([pull request #4](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/4) of the 4x recipe), adapted to two ranks; offline tests `scripts/test_scripts.py`
  (fake `ssh` and `docker`, no Spark): 22 of 26 fail on the previous scripts.
- `tools/validate.py` passes a check only on a real completion: an HTTP error, an `error` object, a reply that is not
  the assistant's, a tool call without an id, a function name or JSON-object arguments, and a short, empty or non-list
  `--compare` reference now fail with a message instead of passing or crashing, and an option without a value stops with
  one. By **[SvangenStudios](https://github.com/SvangenStudios)** ([pull request #5](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/5) of the 4x recipe, without its `TF_DEBUG_PROMPT_FILE` option);
  offline tests `tools/test_validate.py` (a mock server, 28 tests; 31 checks fail on the previous script).
- `zig/patches/0030-anthropic-context-management-lenient.patch` (always on): the Anthropic frontend accepts any
  `clear_thinking_<date>` edit in `context_management` with any `keep`. It accepted only `clear_thinking_20251015` with
  `keep: "all"`, so a client that clears earlier thinking when the user sends a message mid-conversation (the original
  `clear_thinking_20250929` with `keep: "none"`, which the cloud API takes) got a non-retryable 400. Other edit types are
  still refused. By **Maciej Kowalik ([koniaik](https://github.com/koniaik))**, [pull request #71](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/71); a host test in `late_system_test.zig`.
- `zig/patches/0031-anthropic-system-anywhere.patch` (always on): a `system` turn inside `messages` is accepted anywhere,
  as on the cloud API. It had to follow a user or system turn and precede an assistant turn or the end, so a client that
  injects a reminder before the newest user turn got a 400; turn-scoped `clear_at` and per-message `output_config` are
  still refused. By **Maciej Kowalik ([koniaik](https://github.com/koniaik))**, [pull request #72](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/72); the placement assertion in `late_system_test.zig` now
  expects the new behaviour.
- `zig/patches/0032-stream-keepalive-ping.patch` (always on): while a streaming reply waits for the engine and nothing has
  been sent for 25 s, it sends a keep-alive (`event: ping` on Messages, a `: ping` comment on the OpenAI frontends), so a
  proxy that closes an idle stream at about 126 s no longer cuts a long cold prefill. Replies are unchanged; a warm
  prompt sends no ping. By **Maciej Kowalik ([koniaik](https://github.com/koniaik))**, [pull request #76](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/76), measured on a pair behind such a proxy.
- `zig/patches/0033-graph-cap.patch` (on, `GRAPH_CAP=128`): the CUDA graph cache keeps at most `GRAPH_CAP` captured
  stretches a rank and destroys the least recently replayed one past it (issue #69). Before, every graph stayed until
  the process ended, so both ranks' device and host memory grew for days with growing conversations until memguard
  killed rank 0. Replies are unchanged (`tools/exact.py` 31/31 with an eviction almost every stretch) and decode at the
  default equals uncapped graphs; `GRAPH_CAP=0` keeps the old behavior. `scripts/port_test.sh` checks that both ranks
  get the setting. By **E-Zou Shen ([ezoushen](https://github.com/ezoushen))**, [pull request #70](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/70);
  renumbered 0031 to 0033.
- `start.sh` compares the two ranks' Engram folders before anything is stopped when the spill tier is on
  (`port_engram_check`, `scripts/port.sh`). The engine's rank fingerprint covers the folder's regular files by name and
  size (`zig/patches/0016`, `fingerprint.zig`'s `listing()`), and a Hugging Face snapshot holds symlinks into `blobs/`:
  with the default empty `HEAD_ENGRAM_DIR` rank 0 listed nothing against the worker's four files, so the pair refused
  to load with `FingerprintMismatch` in `engram` — after the running server had already been stopped (issue #61). The
  check refuses before the stop, naming both listings and the fix (`HEAD_ENGRAM_DIR` on a local folder holding the same
  files, hardlinked: no extra disk, and rank 0 then reads Engram locally); it stays silent while a folder is not there
  yet, warns when the worker cannot be reached, and does nothing with the tier off (KV sharding alone marks `engram`
  `shard-only`, `0018`). `scripts/port_test.sh` covers the listing, the refusal and its message, both silent paths and
  the warning. By **Qingchen Yu ([markyu0401](https://github.com/markyu0401))**,
  [pull request #64](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/64).
- `ABLIT=1` on a read-only Hugging Face blob cache: `prepare.sh` gives the overlay's work folder its own writable copy of
  the checkpoint index (the shards stay hard links), so the graft no longer fails with `PermissionError` when it rewrites
  the index (issue #42, reported by unitea1992). By **Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub,
  [@rwojo](https://x.com/rwojo) on X)**, [pull request #45](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/45);
  regression test `scripts/test_ablit_index_writable.sh`.
- Each rank's container log rotates (`LOG_MAX_SIZE`, default `100m`, and `LOG_MAX_FILE`, 3; `LOG_MAX_SIZE=` leaves the
  docker daemon's settings): the engine writes an access line for every response, so a 1 s monitoring poll grew rank 0's
  log by ~265k lines a day for as long as it ran (issue #49, reported by eleata). Checked by the `start.sh` dry-run
  golden test (`scripts/port_test.sh`).
- README: what one stream reserves out of the pool at admission (`prompt + max_tokens + ROUND_ROWS + 2`, aligned to
  2,048), with a measurement of four concurrent 500,601-token prompts in a 2,097,152-position pool. By **Maciej Kowalik
  ([koniaik](https://github.com/koniaik))**, [pull request #56](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/56).
- `zig/patches/0029-anthropic-budget-default.patch` (always on): the Anthropic frontend accepts `thinking: {"type":
  "enabled"}` without `budget_tokens`; such a request gets the server's `--thinking-budget` default, as an OpenAI-format
  request without `thinking_budget` already does (stock refused it: `thinking.budget_tokens must be positive and less than
  max_tokens`). A `budget_tokens` that is sent is checked and passed on as before (zero, negative, not an integer or not
  below `max_tokens` is still refused). By **Maciej Kowalik ([koniaik](https://github.com/koniaik))**, [pull request #53](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/53) (sent as `0014`; the refusal of a `budget_tokens` that is sent but invalid kept when ported).
- `zig/patches/0028-plain-request-text.patch` (always on): a request's own text never becomes a control token (issue #21).
  The server renders the chat template to text and encodes it with the added vocabulary, so a control token's spelling
  in request text came back as the token: a tool result holding `<｜User｜>`, a DSML call or `</think>` gave the model a
  role marker, a call or the end of its thinking. Every string of the request's messages and tools now gets the
  tokenizer's plain mark inside each such spelling before rendering (`Tokenizer.maskAdded`, `messages.maskControls`), and
  the encoder drops the mark before the pre-tokenizer. Measured through `/tokenize` on a pair: four prompts whose tool
  result spells role markers, think tags, a DSML call or the sequence tokens held 30-34 control ids before, 28 after
  (the template's own); three clean conversations (plain, Unicode, JSON) give identical ids; reply ids unchanged
  (greedy, seeded temperature 1, thinking, a tool call, a 10K-token prompt); `tools/validate.py` passes (video aside).
  The output side (`reply_text.splitThinking` reads `</think>` from text) is not changed. The Anthropic `count_tokens` route masks the same way (added when ported). By **[Coinupbtc](https://github.com/Coinupbtc)**, [pull request #43](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/43) (sent as `0013`).
- `zig/patches/0027-tool-call-prose.patch` (always on): the text a reply writes before its tool calls is the reply's
  content, without the two newlines that start the calls, as the checkpoint's reference parser reads it
  (`encoding/encoding.py`, `parse_message_from_completion_text`: the calls start at `\n\n<｜DSML｜ calls`; `render_message`
  writes `content + "\n\n" + calls` back). A reply without streaming used to return `"content": null` whenever it made
  calls, so the client never saw the text (in one case a 3,671-character plan) and sent the turn back without it. A
  streamed reply sent the two newlines as content, so a client that keeps the content as it arrived sent back
  `text\n\n\n\n<｜DSML｜ calls>`. Measured on a pair: three long tool-calling replies (thinking high, 780-4,757 characters of
  reasoning, 2,369-3,645 of text) sent back as they streamed now give the next prompt exactly the generated tokens (3 of 3;
  before: 0 of 3, each diverging right before the calls), and the non-streamed reply's turn sent back as returned does
  too. Reply ids are unchanged (greedy, seeded temperature 1, thinking, a tool call, a 10K-token prompt). A streamed
  paragraph break now waits for the next token before it is sent. By **[Coinupbtc](https://github.com/Coinupbtc)**, [pull request #44](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/44) (sent as `0012`).
- **Rank 0's memory with long prompts:** `zig/patches/0023-server-prepare-memory.patch`, a fix to the base engine after
  the port's series (also offered alone as its own pull request). A prompt's render and tokenization kept their scratch
  in the request's memory until the reply ended: a resumed ~948K-token conversation held ~1.3-1.5 GB on rank 0, and
  eight at once took its MemAvailable from 6 GiB to 0.16 GiB until the memory guard killed it. The tokenizer and the
  template give their scratch back, a prepare waits while others are in flight and MemAvailable would fall under
  `TENSORFOLD_PREPARE_FLOOR_GIB` (default 3; `0` off), and a kept-alive connection keeps at most 1 MiB between its
  requests. Replies are unchanged (`zig build test-golden`). By **Robert Wojciechowski ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X)**.
- **Port of Robert Wojciechowski's (**[wojo](https://github.com/wojo)** on GitHub, **[@rwojo](https://x.com/rwojo)** on X) GLM-5.3-Flash patches, off by default**
  ([docs/PORT-RECIPE.md](docs/PORT-RECIPE.md), README "Spill v2 and KV sharding"): the spill tier and its v2 block store
  with quotas and crash recovery (GLM `0088`/`0110`, PRs #78 and #124, on **[BadAd84](https://github.com/BadAd84)**'s
  block-tier design, #118), chat continuation (`0111`, #128), KV sharding with prompt borrowing and the ring exchange
  (`0118`, #152) and the parked-page startup budget (`0119`, #149), ported to the native engine as twelve
  patches, `zig/patches/0011` to `0022`, one per feature with a header in the GLM recipe's style (each checked with
  `git apply --check` on a clean `TF_REF` after `0001`-`0010` and the ones before it; the new `scripts/port-patches.sh`
  exports later engine work on top). **KV sharding builds on `0008`'s pool:** one `POOL` setting; with
  `KV_SHARD=1` the same pool is cut into one part a Spark, each holding a whole window (empty `POOL`: the pool that
  needs, 2,101,248 at the 1M window), and each Spark keeps only its part's compressed KV, about half the KV memory a
  Spark for the same `POOL` (measured at 2,101,248: 1,062 MiB a rank against 1,783 MiB replicated; decode 1-4%
  slower). Settings `SPILL_GIB`, `SPILL_DIR`, `WORKER_SPILL_DIR`, `SPILL_FLOOR_GIB`, `SPILL_QUOTA_GIB`, `KV_SHARD`,
  `KV_SHARD_DECODE` and friends, mapped in one place (`scripts/port.sh`) and passed to both ranks alike; each Spark's
  own spill folder checked on both (local disk, outside the NFS export, the model folders and archives, not
  root-owned, above the floor) and mounted at `/spill`; a set setting that cannot run refuses the start. With
  everything off the only change in the docker commands is a `tf.recipe.port=off` label. Also: a broken rank pair
  ends rank 0 with status 75 instead of hanging it (`0020`; on with two ranks whatever the settings, since it acts only
  after a failure; `TF_DS_PAIR_BROKEN=degrade` opts out), for a supervisor such as the systemd service below to restart,
  and `start.sh` hands an exited container to `stop.sh`, which saves its log before it is removed; spill files are
  handed back to the folder's owner by a minimal container (only the marked folder, root-owned files and directories
  only, no symlinks, no network, `CHOWN` + `DAC_READ_SEARCH` only); `MIN_AVAIL_GIB` counts parked per-CPU pages
  (`scripts/memavail.sh`); the end-to-end tools in `tools/` read an API key only from `API_KEY` or `--api-key-file`
  and never write it to receipts; `ZIG_MIRRORS` for the compiler download. Validated on the pair:
  replies byte-identical to the stock engine in every configuration; resuming a 237k-token conversation from disk
  after a restart takes 1.7 s against 118 s recomputed. By **Robert Wojciechowski** ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X), pull request
  [#30](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/30).
- The default checkpoint stays EAQ v1 (`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` @ `45ee5d0f`, 39 shards,
  ~196 GiB); the experimental EAQ v1.1 pack is opt-in with `EAQ_V1_1=1` (`...-EXL3-EAQ-v1.1-2.9-bpw` @ `aee640e2`). v1.1
  keeps EAQ v1's widths but stores 27 indexer/compressor linears as BF16, so the engine must have the mixed-pack loader.
  `ABLIT=1` works with the v1 default and is refused when v1.1 is selected. `scripts/test_checkpoint_pin.py` checks both
  pins. The recorded `tools/exact.py` token hashes are EAQ v1's; v1.1's weights differ, so record a new baseline before
  comparing replies. EAQ v1.1 support by **Mykhailo Tamarin** ([MykhailoTamarin](https://github.com/MykhailoTamarin)), pull request
  [#57](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/57) (this entry and the three below; its engine
  patches rebased after the port's series as `0024`-`0026`).
- Mixed-pack loader: `zig/patches/0024-mixed-pack.patch` and `zig/patches-inputs/0001-mixed-pack-rank-cache.patch`.
  The rank-cache planner reads the checkpoint's headers and, for a whole replicated linear stored as a BF16 `.weight`
  instead of trellis/suh/svh, plans one plain entry. The serving engine loads that entry as a plain BF16 linear and runs
  it through cuBLAS (`cublas.Blas.linearBf16`, torch's bf16 `nn.Linear`) in decode and prefill, with the exact
  bf16 -> fp32 widen where the caller wants fp32. EAQ v1.1's 27 linears are 8 `attn.wq_a`, 8 `attn.indexer.wq_b`,
  4 `attn.compressor.wkv`, 3 `attn.compressor.wgate` and 4 `attn.indexer.wk`; the load prints
  `mixed pack: 27 float linears`. An EXL3-only pack (EAQ v1) plans no plain entries and runs exactly as before, so the
  same image serves both. The patch is checked against `TF_REF`/`INPUTS_REF` by `scripts/test_mixed_pack_patch.py`. (#57)
- BF16 sources widen to the f32 entries the loaders plan: `zig/patches-inputs/0002-widen-bf16.patch` in the rank-cache
  writer, and `zig/patches/0025-widen-bf16.patch` in the engine's checkpoint reader. EAQ v1.1 (as v1) stores the eight
  `attn.indexer.weights_proj.weight` tensors as BF16 while both plans ask for f32, so the first rank-cache write stopped
  with `error: UnsupportedConversion`; the widen is exact (BF16's bits in the high half, zeros below), the loader's
  `.to(torch.float32)`. `scripts/test_mixed_pack_patch.py` checks both patches. (#57)
- A mixed pack's plain (cuBLAS) linear runs over a fixed number of rows: `zig/patches/0026-plain-rows.patch` and
  `TF_DS_PLAIN_ROWS` (default 16, never beyond a round's rows; `0` turns the padding off). cuBLAS picks its kernel and
  split-K from the GEMM's shape and its rounding follows the pick, so a decode round whose row count changes with the
  streams or the draft block could change a token's bits. On the pair, without it, drafts-on and drafts-off replies
  matched 4 of 8 `tools/validate.py` prompts, and its 8-concurrent check matched 4 of 8 with drafts off (6 of 8 with
  drafts on); with it, the saved replies matched 8 of 8, the 8-at-once check matched 8 of 8 with drafts on and off, and
  all 31 `tools/exact.py` cases were identical with drafts on vs off. The EXL3 linears are untouched: their kernels
  compute each row in a fixed order, which is why EAQ v1 kept the same bits across round shapes. (#57)

- Optional `systemd/dsv41-tensorfold.service` and `docs/systemd.md`: a manually configured user service for the existing
  foreground launcher, with `PREPARE=0`, a 60-second failure restart delay and at most three starts in 30 minutes.
  Nothing installs or enables it automatically. Unit validation and opt-in transient stub tests cover failure exits,
  the start limit, successful exit and an explicit stop. Requested by **m-naoki-m**, issue
  [#32](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/32).

- Document shared-pool residency (#20, **meleesciony**), cached versus computed prefill, and effort/budget limits for recurring reasoning-loop questions (#33, **Lcys**; #37, **koniaik**).

- Bound RDMA idle/nap settings in both the launcher and engine; reject invalid or excessively long sleeps. Added regressions fail on the original #26 parser. The idle back-off is by **MykhailoTamarin**, pull request [#26](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/26).
- Keep both ranks' pool and RoPE window consistent: raw `TF_DS_POOL` / `TF_DS_WINDOW` exports cannot override the launcher's `POOL` / `CONTEXT` mapping. A regression exercises conflicting exports against both rank settings.

- Repair the default checkpoint pin independently of #38's credential-handling changes: the removed `6d54be4f` revision returns HTTP 404, while `45ee5d0f` resolves with all 39 checkpoint shards. Reported by **ajbmachon** (#36) and **yume-arasaki** (#38). `RUN_HF_PIN_CHECK=1 python3 scripts/test_checkpoint_pin.py` checks the public Hub revision and required files; it fails on the removed pin.

- Reject `ROUND_ROWS` above 16 before starting: the pinned kit has no `_rowmm_parts` specialization above 16, so larger configured rounds could fail with `CudaFailed` (issue [#23](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/23), reported by **Nollio**). CPU regression tests cover the unsupported settings; wider kernels remain under review in #31.

- Check the worker's model/kit/token-map file view before stopping or launching ranks, including cached preparation; reject directories as required files. By **aoeu10**, pull request [#25](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/25), with CPU regression coverage.

- Worker cache cleanup handles login shells that reject unmatched globs; regression tests cover an empty cache and stale-file cleanup. By **mvandergrift**, pull request [#19](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/19).

- `tools/bench.py` uses the port the server actually listens on when neither `--url` nor `API_URL` is given: `PORT` from the
  environment, then from `scripts/local.sh` (the file the README says to put settings in), then the recipe's 8888. A pair that
  moved `PORT` no longer gets a page of `Connection refused` from the recipe's own bench. `--url`, `API_URL` and `PORT` keep
  their precedence. By **yume-arasaki**, pull request [#39](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/39); regression coverage also checks repeated assignments and avoids evaluating shell text.
- `zig/patches/0008-pool-window.patch` and `POOL` (empty: `CONTEXT`, as before): the shared pool may be larger than the window. A kept prompt holds its extent, so at the default 1,048,576
  positions three 386K-token conversations cannot all stay resident (issue [#16](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/16) is the picture case of the same
  placement). The RoPE tables are sized by the window now, not the pool (the kit's tables already hold 1,114,112 rows). The extra KV is host memory: about 0.9 GiB a rank per 1,048,576 positions
  (`POOL=2097152`: head idle `MemAvailable` 6.76 to 5.71 GiB, lowest during a cold 96K-token prompt 5.89 to 4.49 GiB; worker 8.63 to 7.43 and 8.60 to 7.33). The patch applies on its own. By **ezoushen**, pull request [#10](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/10).
- `zig/patches/0001` (the prefill indexer split between the two ranks) runs by default in prompt chunks that end at 65,536
  compressed keys or more (token 131,072), where the indexer's work outgrows the all-gather of its selections; below that the
  replicated loop runs as before. Cold prefill against the image without it (three alternating pairs, both images built on the
  head): +4.25% at 256K and +7.46% at 400K; 8K/32K prefill, decode and four streams within 0.9%. Replies keep their bits.
  `TF_DS_IDX_SPLIT=1` splits every eligible chunk as before, any other value turns it off. By **aaa4xu**, pull request [#35](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/35).

- `zig/patches/0010-rdma-idle-nap.patch` and `RDMA_IDLE_S` / `RDMA_NAP_US` (on by default; recipe 2 s quiet, 500 µs
  nap; the patch's engine fallback is 100 µs): the RDMA gather's proxy naps once the engine is idle instead of spinning a
  core on each rank (the logic of [TensorFold PR #27](https://github.com/bertholomus/TensorFold/pull/27), ported to the
  Zig ring). A decode round's gathers come microseconds apart, so they never see a nap; the first gather after a quiet
  spell waits at most one nap (0.5 ms at 500 µs). `RDMA_IDLE_S=0` keeps the stock always-spinning proxy for an A/B. The
  patch applies after 0001, 0002 and 0004-0006; `zig fmt --check` and `zig ast-check` pass. Measured on this pair with
  `RDMA_NAP_US=100`: the proxy thread used 11.2% (rank 0) and 8.8% (rank 1) over a 20 s window with the lane host idle,
  against ~100% a rank before. The Zig ring's two CQ polls per wake cost more than PR #27's Python line (3.9-5.2% at
  100 µs), so the recipe sets 500 µs; the wake rate falls 5x and the proxy should sit near 2% a rank. 2 s is enough that
  active decode never naps.
- `prepare.sh`'s "cannot mount ... over NFS" error shows the last four lines of docker's output, not only the last (which was always
  docker's "Run 'docker run --help'" line, never the reason); the README's NFS section says what to check (issue
  [#15](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/15), reported by **aoeu10**). The README's memory note now
  records the 464K-token head kill at `CONTEXT=655360` and keeps the 512K guidance (issue
  [#17](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/17), reported by **tanchunsiong**).
- `ABLIT=1`: the drafter drafts again (**sunlinhua-dotcom**, issue [#11](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/11) and
  [pull request #12](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/12); the same finding, with a second measurement
  on stock-vs-grafted decode, was reported by **Ye1806431561** in issue [#13](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/13)
  and pull request [#14](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/14), which relinks the unedited shards to stock instead;
  #12 is the one taken). The overlay's `apply_mia29_wo_b_graft.py` casts four MTP tensors from F16 to F32 for another
  engine; this one uploads `mtp.2.confidence_head.proj.weight` as stored and reads it as F16 (`draft.zig`), so from the F32 copy the
  confidence head computed garbage and the engine drafted 0.2-0.8 tokens a round while acceptance stayed high. `scripts/ablit_stock_dtypes.py`
  (prepare.sh, after the overlay, and once on a checkpoint prepared before) gives the four tensors stock's F16 bytes back after checking that
  each is a pure cast (`scripts/test_ablit_stock_dtypes.py`, run in the image, covers the give-back, the untouched hardlinked stock shard, the second run and the refusal); every other tensor, the 104 overlay tensors included, is left as it is, and the checkpoint's headers then match stock's.
  A checkpoint prepared before this rebuilds both ~106 GB weight files on the next start. **Measured on the pair** (the same F16 bytes written
  into both ranks' weight files in place; after and stock back to back, same script): ABLIT decode is back at stock's speed, default sampling
  50.8 -> 68.6 tok/s (stock 69.1, 99%), code 60.8 -> 114.8 (stock 110.8), prose 52.5 -> 70.7 (stock 75.1), structured 48.7 -> 142.1 (stock 137.8),
  single stream; 4 streams prose 104 -> 134.9 (131.0), code 112 -> 157.4 (155.6); acceptance equal to stock's.
  The script was run on copies of the real shards 38 and 39: four tensors given back, equal to stock's bytes, the other 9,788 equal to the
  overlay's, the originals untouched, a second run writes nothing; a tensor that is not a pure cast is refused (exit 1).
- `start.sh` re-reads the memory available on both Sparks for up to `MIN_AVAIL_WAIT_S` seconds (default 30; 0 reads once) before it
  refuses a start under `MIN_AVAIL_GIB`. Right after an image build or a large copy the page cache is counted as available only as it
  is written back: 111.1 to 111.6 GiB was read on a worker a few seconds after a build, 119 GiB once it settled, and the 112 GiB check
  refused a start the Spark could take. The README's `MIN_AVAIL_GIB` default reads 112, as `scripts/config.sh` has it (it said 109).
  By **ezoushen**, [pull request #7](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/7).
- `THINKING_BUDGET` (empty by default): passes `--thinking-budget` to rank 0's `tensorfold-native serve`, so a turn that would spend its whole `MAX_TOKENS` thinking
  has its think block closed after that many reasoning tokens and the answer continues (a request's own `thinking_budget` wins). Checked on the pair: a budget of 1,500
  gave about 1.5K reasoning tokens and a finished answer. README: the engine refuses a `top_k` above 56 (`TooManyCandidates`) and `top_k` 0 or -1.
  By **ezoushen**, [pull request #8](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/8).

- Issues #3 and #2 in the engine (patches, both **off or on as noted, not yet merged**): `zig/patches/0005-prefill-interleave.patch` (**Coinupbtc**,
  [pull request #5](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5)) gives the streams already live decode rounds and their cancels between the
  chunks of a long prompt pass, `PREFILL_INTERLEAVE_MS` (default 0: off); `0006-image-prompt-cache.patch` lets a prompt with pictures resume and keep prompt states,
  keyed by the pictures' content (on; `TF_DS_IMAGE_CACHE=0` turns it off). The patches apply in order on the pin, and the engine's host tests pass with all five (21/21 steps).
  **Measured on the pair** (2026-10-19, image `exl3-tp2-8eb5919ff0a4`, interleave 250 ms): a 120K-token cold prompt no longer freezes a decoding stream (longest gap 1.3 s, was
  90 s; the prompt takes 99 s, was 91 s) and a cancel is answered 0.74 s after the client leaves, inside the other prompt's pass; a conversation with a picture resumes
  26,624 of 27,142 tokens on turn 2 (0.8 s, cold 14 s) and another picture with the same text stays cold; `tools/exact.py`: 31 of 31 cases IDENTICAL to the previous
  image (4 concurrent equal to the 4 solo, resumes, 100K prompt). `PREFILL_INTERLEAVE_MS` defaults to 250 (0: off).
  An earlier version of 0005 written here was dropped for the pull request's, which joins the Engram read-ahead only when another stream wants a round.
- memguard is looser (issue #4): the default floor goes from 4 to 2.5 GiB and a kill needs the memory to stay under it for `MEMGUARD_HOLD_S` (4)
  seconds without once coming back over it; under `MEMGUARD_HARD_GIB` (1) in two samples 0.25 s apart it still kills at once. A dip that
  recovers is logged with its length and lowest value, not killed. Measured here: a 120K-token cold prompt beside a decoding stream took
  the head to 3.3 GiB, over the old 4 GiB floor. The decision logic was tested with a fake `/proc/meminfo` and a stub `docker` (a dip that
  recovers, a sustained dip, a crash under the hard floor, flapping); no rank was killed or spared by it on the pair yet.
- Issues [#2](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/2),
  [#3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/3),
  [#4](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/4) (reported by lukemdanastasi): `TF_REF` moves from
  `3ff3b2ac` to `629c524` (the branch tip of 2026-10-19). It brings `9f82099` (a long prompt pass lets streams near their
  `max_tokens` finish before the next pass; kept prompts hold their extent), `629c524` (a shared prefix no longer stales the first
  conversation's kept states) and the template and JSON-writer fixes. `zig/patches/0003` (cache placement) is removed: the engine does this
  itself now and the patch no longer applies. Image `tensorfold-dsv41:exl3-tp2-bd95cdfc712d`; the engine's host tests pass with patches 0001, 0002 and 0004
  applied (`zig build test`: 21/21 steps, 221/221 tests). **Served on the pair** (2026-10-19, `tools/smoke.py` passes): cold prompts of 131K and 262K
  tokens prefill at 1,608 and 1,868 tok/s; `tools/prompt_reuse.py` resumes 99-100% of a 159K and a 261K-token history after another
  conversation's request (1.5-1.9 s a turn). `MemAvailable` on the head (6.5 GiB after the load) fell to 4.9 GiB at 159K, 4.2 GiB at 261K and
  **3.3 GiB** while a 120K-token cold prompt prefilled beside a decoding stream: under the default 4 GiB `MEMGUARD_GIB` (this pair runs 2.5).
  Issue #3 is **not** helped by the pin: that decoding stream got no token for 90.1 s while the other prompt prefilled (16 tokens in 90 s). Not fixed, because they are engine behaviour: prompts with images run whole and keep nothing (#2); a prompt pass runs
  to completion before other streams decode and a cancel waits for it (#3, only the tip's narrow mitigation applies).
  For #4 `start.sh` now reports `MemAvailable` on both Sparks after the load and warns when it is within 3 GiB of `MEMGUARD_GIB`, and the
  README says what holds the memory above the floor (nothing at run time) and what to do on a shared head.
- First version: DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw on two DGX
  Sparks, TensorFold's native Zig engine with BertholomusAI's `dsv41` family (`deepseek-v41-zig` @ `3ff3b2ac`), their kit
  (`abe07c7`) and their weight-file tool (`deepseek-v41-zig-tp4` @ `5940208`). Scripts adapted from the GLM-5.3-Flash
  recipe. Not served yet: see the README's Status.
- From the GLM-5.3-Flash recipe: the node and NFS helpers, `stop.sh`, the start banner, the layered settings,
  `scripts/zig_flags.py` (run by `start.sh` against the image's `capabilities --json` before anything is stopped),
  `scripts/publish-image.sh` and the registry pull, the API checks in `tools/` (`toolcheck.py`, `needle.py`,
  `end_of_turn.py`, `anthropiccheck.py`), the Hermes Agent page and the `.github` templates. Not ported, because they read
  the GLM Python engine's `/health` counters or its prompt cache, patches or settings: `tools/hermes_benchmark.py`,
  the pool, kept-state, display-KV and CPU/GPU tests, the spill tier, `SHARED_PREFIX`, the three and four Spark scripts,
  and the Ablit weights. (`tools/prompt_reuse.py` is ported since, with its cached-token lookup adapted.)
- `model/chat_template.jinja` writes each piece of the prompt out as it is made instead of adding it to one growing
  string (fix by hashd1ve, issue #1): rank 0's host memory no longer grows with messages x prompt length (40K tokens in
  2,000 messages: +1,555 MiB before, +244 MiB after, as the issue measured). The rendered prompt is identical (3,000 random
  conversations compared on jinja2).
- `stop.sh` counts running requests from the native server's `/health` (`live.connections`).
- Served and measured on one pair of Sparks (README, Status). With sparkDash's real prompts:
  prefill ~2,140-2,180 tok/s from 8K to 64K prompts (2,000 at 256K), decode 74.9 tok/s (prose) and 109.9 tok/s
  (code) for one stream, 130.0 and 155.5 tok/s for four. `tools/bench.py`'s random-word prompts give less decode (55-61 tok/s, 113-117 for four): the drafter
  cannot predict them.
- Both GPUs must share one clock ceiling: a Spark locked at 1,800 MHz against 2,200 MHz on the other cost ~3-6% of prefill
  and ~3% of decode (TP=2 meets twice a layer, so the slower GPU sets the pace).
- Rank 0 reads Engram's tables from local disk (`HEAD_ENGRAM_DIR`): over NFS a random read took ~940 us against ~90 us
  locally, and prefill was ~690-890 tok/s instead of ~1,500-2,000.
- Memory safety: `scripts/memguard.sh` (kills a rank's container under `MEMGUARD_GIB`), the `MIN_AVAIL_GIB` preflight, setup
  stops a running server before it builds or downloads, and `HEAD_ENGRAM_DIR` is checked (both shards, exact sizes) before
  anything starts.
- `HEAD_NFS_SERVER` / `HEAD_NFS_VOLUME`: rank 0 reads the checkpoint over NFS through a docker volume (sshfs does not work:
  it lists files as type "unknown" and the engine skips them).
- `API_KEY=1`: an opt-in API key (off by default; the API is open on the network otherwise).
- `tools/bench.py`, `tools/smoke.py` and `tools/prompt_reuse.py`.
- Engine patches in `zig/patches/` (off by default; bit-identical to the unpatched image on 31 of 31 prompts): prompt-cache placement
  (`TF_DS_CACHE_PLACEMENT=1`, **on**: a resumed turn no longer misses the cache because another conversation's kept rows were in
  the way), a second Engram AIO ring (`TF_DS_ENGRAM_LANES=2`, off: 1.5-2.6% slower) and a prefill indexer row split
  (`TF_DS_IDX_SPLIT=1`, off: no measurable gain). `tools/exact.py` records and compares replies.
- `zig/patches/0004-call-gate.patch`: the engine enforces `tool_choice` `"required"` and a named function (taken from the spark1/spark2
  deployment's copy; built into the running image `exl3-tp2-7c9bf5d36350`: required, named, streaming and thinking-on requests
  return valid tool calls, auto/none/absent are unchanged).
- `RANK1_PROFILE` (rank 1 `--profile`); both GPUs must share one clock ceiling; each Spark can run from a local copy of the
  cache (no NFS), with `PREPARE=0`.
