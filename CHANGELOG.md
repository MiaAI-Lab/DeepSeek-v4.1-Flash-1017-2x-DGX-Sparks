# Changelog

## Unreleased

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

- **Spill v2 and KV sharding ON by default** (2026-10-20, the daily configuration): `SPILL_GIB=64` a Spark, `KV_SHARD=1`
  and `POOL` derived from `CONTEXT` by the engine's rule (each of the two owner parts holds a whole window
  plus a round's rows and two, in 4,096-position parts: 2,101,248 at `CONTEXT=1048576` and 16-row rounds; another `CONTEXT` or
  `ROUND_ROWS` gets its own). The engine's own defaults are unchanged (patch `0007` is the same); only the recipe's settings
  changed. Each turns off explicitly: `SPILL_GIB=0`, `KV_SHARD=0` (with `POOL` empty the pool is `CONTEXT` again), both for the
  recipe before the port. **A default that cannot run degrades, an explicit one refuses:** when the default spill folder fails
  its checks on either Spark (missing and not creatable, not yours, on NFS or a memory file system, inside the export or a model
  folder, below `SPILL_FLOOR_GIB`, an unusable path, the worker's probe not answering) `start.sh` warns
  `SPILL TIER OFF for this start` and starts both ranks without the tier; `SPILL_GIB` (or `TF_DS_SPILL_GIB`) set explicitly
  refuses as before. The default `KV_SHARD=1` is likewise turned off with a warning when `TP` is not 2 or an explicit `POOL`
  cannot hold two windows (`POOL=1048576`); explicit `KV_SHARD=1` refuses. The spill folders are now checked before the flags
  and the `tf.recipe.port` label are made (and `need_workers` runs first), so a degraded start is labelled `spill=0` and
  `./start.sh` adopts a running pair against the same decision. Disk: ~890 bytes a token a Spark replicated; sharded, the owner
  ~890 and the other ~170; at most 64 GiB a Spark, with 50 GiB always left free. Rolling back to an image without the port now
  needs the switches off explicitly: `SPILL_GIB=0 KV_SHARD=0 IMAGE=<tag> PREPARE=0 ./start.sh restart`. Containers started with
  every switch off (`tf.recipe.port=off`) are not adopted by a default `./start.sh`: restart, or set both to 0.
  `scripts/port_test.sh` tests the defaults, each switch off, the degrade and refuse paths and (bash 4) the golden DRY_RUN cases
  for the defaults; `tools/gpu_validate.sh`'s configurations name the switch they do not test as off.
- **Renamed** (2026-10-20): the branch `port/recipe` is now `spill-kvshard` (on GitHub `wojo/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`,
  the old name redirects) and the engine export `zig/patches/0007-glm-port.patch` is now `0007-spill-kvshard.patch`
  (`scripts/port-patches.sh`'s default `--name`). The image tag changes with the file: `./start.sh restart` rebuilds. Docs no
  longer say spill and KV sharding refuse each other (they run together since `PORT_SHARD_SPILL_READY=1`). `NOTICE` lists
  `0007` and its author (Robert Wojciechowski, wojo) beside the other patches, and the patch header names him in full.
- **Port, recipe: a broken pair restarts itself** ([docs/PORT-RECIPE.md](docs/PORT-RECIPE.md) "When the rank pair breaks"):
  with spill or KV sharding on, the engine ends rank 0 with status 75 when the rank pair breaks
  (`TF_DS_PAIR_BROKEN=exit`). The new `scripts/pairwatch.sh` runs on rank 0 beside the memory guard and answers an
  `exited 75` with `./start.sh restart` within ~2 s (`PAIR_RESTART`, `PAIR_RESTART_MAX`, `PAIR_RESTART_HOLD`).
  `stop.sh` disarms it first, and with both features off nothing changes. `tools/gpu_validate.sh` gains `spill-shard`,
  `spill-shard-pressure` (opt-in), `spill-pictures`, `spill-pictures-mismatch`, `pair-kill` and `pair-stop`.
  `tools/pair_check.py` is new, and `spill_check.py` has `--checks pictures` and `--pictures`.
- **Port, recipe:** the fake server's `parked` gauge counts requests waiting on a restore (the reader and each coalesced one),
  so `spill_check.py`'s settled check is exercised against a non-zero value (`tools/selftest.py`); `tools/prefix_fanout.py`
  runs in the `off` configuration with `--expect-one-fill` (the native engine computes a shared prefix once at every level).
  A running pair started by an earlier tree carries another `tf.recipe.port` label and is not adopted by `./start.sh`: run
  `./start.sh restart`, as a new image needs anyway.
- **Port, round 2 (recipe):** rank 1 gets the engine's own spill names (`TF_DS_SPILL_DIR`, `_GIB`, `_MIN_FREE_GIB`,
  `_MIN_TOKENS`, `_QUOTA`, `_STAGING_MIB`, `TF_SPILL_TURNS`, `TF_SPILL_FLUSH_MS`; the round-1 `TF_SPILL_*` placeholders are gone) and
  rank 0 `--spill-min-tokens` from the same `SPILL_MIN_TOKENS`; the identity the engine requires with spill (`TF_DS_ENGINE_ID`,
  `TF_DS_RECIPE_PATCHES` from the image's labels, `TF_DS_MODEL_REVISION`, `TF_DS_ENGRAM_REVISION`, each rank's `TF_DS_RANK_CACHE_SHA`
  cached beside its weight file); `KV_SHARD_CARRY` / `_TWINS` / `_CHECK`; `MIN_AVAIL_GIB` counts pages parked on the per-CPU free lists
  as the engine's 0119 does and waits up to `MIN_AVAIL_WAIT_S` (`scripts/memavail.sh`); the tools read the engine's `kv_shard`
  counters; `tools/gpu_validate.sh` covers the engine's GPU validation lists; [docs/FORK-DEPLOY.md](docs/FORK-DEPLOY.md).
- **Port of Robert Wojciechowski's (wojo) GLM patches, recipe side** (all off by default; [docs/PORT-RECIPE.md](docs/PORT-RECIPE.md), README "Spill v2 and KV
  sharding"): the settings `SPILL_GIB`, `SPILL_DIR`, `WORKER_SPILL_DIR`, `SPILL_FLOOR_GIB`, `SPILL_QUOTA_GIB`, `KV_SHARD`,
  `KV_SHARD_DECODE` and `POOL` (total pool positions, apart from `CONTEXT`, the window), mapped to the engine's flags and variables in one
  place (`scripts/port.sh`) and passed to both ranks alike; each rank's own local spill folder mounted at `/spill` after checks on both
  Sparks (local disk, outside the NFS export, the model folders and archives, not root-owned, above the floor), its files given back to
  the folder's owner; both containers labelled with the settings, and running ranks that disagree are not adopted. The engine side:
  Robert Wojciechowski's (**[wojo](https://github.com/wojo)**) spill tier and spill v2 blocks (GLM PRs #78 and #124, on
  **[BadAd84](https://github.com/BadAd84)**'s block-tier design), chat continuation (#128), KV sharding (#152) and the parked-page
  startup budget (#149), ported to the native engine and exported as `zig/patches/0007-...` by the new `scripts/port-patches.sh`
  (checked with `git apply --check` on a clean `TF_REF`). Validated on the pair (README, "Spill v2 and KV sharding"). With every setting off the only change in the docker
  commands is a `tf.recipe.port=off` label (`scripts/port_test.sh` compares `DRY_RUN=1 ./start.sh` against the recipe before the port,
  with bash 4). Spill folders on tmpfs, ramfs, overlay, squashfs or zram are refused, as are ones inside an export listed in
  `/var/lib/nfs/etab` or `/etc/exports(.d)`; `STOP_TIMEOUT` follows `TF_SPILL_FLUSH_MS`; `POOL` above 4x `CONTEXT` warns and above 16x is
  refused. Rollback to the live image: `IMAGE=<noted tag> PREPARE=0 ./start.sh restart`, or the exact tree (commit and working-tree diff).
- The two fixes of the live two-Spark deployment: `zig/Dockerfile` downloads Zig from `ZIG_MIRRORS` (default `zig.linus.dev`) and then
  ziglang.org, each with bounded retries (20 s connect, 300 s a try, 3 retries, 600 s of retrying at most) and the pinned sha256
  (`ZIG_MIRRORS=` empty: ziglang.org only); `scripts/prepare.sh`'s worker
  checks (the NFS file list, Engram's copy) run in the serving image the worker already holds instead of pulling the base image again.
  The Dockerfile change gives the image a new tag (one rebuild).
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
