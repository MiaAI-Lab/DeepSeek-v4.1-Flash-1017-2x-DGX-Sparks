# Memory and safety

`CONTEXT` limits one request and, when `POOL` is empty, also sizes the shared pool of KV positions. Lowering it can evict conversations even when each conversation fits individually: several histories compete for the same positions. `POOL` can now keep more positions than the request window, but uses additional memory on both ranks; set `POOL`, not raw `TF_DS_POOL` / `TF_DS_WINDOW`. The separate prompt-state budget (512 MiB by default) does not add KV positions, and increasing it alone cannot fix an undersized pool. Placement still retires the oldest kept states, so a larger pool does not guarantee that an idle coordinator survives a succession of other conversations (#20, #27).

Admission reserves a stream's prompt *and* its reply together: `prompt + max_tokens + ROUND_ROWS + 2` positions (18 more at the default 16-row rounds), aligned to 2,048 (`zig/src/families/dsv41/lanes.zig:768` with `zig/patches` applied; a request that names no budget gets `MAX_TOKENS`), and a request whose prompt and budget together exceed the pool is refused (`PromptTooLong`), not queued. So the equal split of a pool is not `POOL / PARALLEL` of context: at `POOL=2097152` with a 131,072-token reply budget, four equal streams fit about 393,000 tokens of context each (`2097152 / 4 - 131072 - 18`). Keeping `MAX_TOKENS` at what the harnesses send explicitly keeps requests that name no budget from reserving more than that.

Measured on a 2x DGX Spark serving (`POOL=2097152`, drafts on, the pool of `0008`, #10): four concurrent cold prompts of 500,601 tokens each — 2,002,404 positions live at once, 96% of the pool — were all admitted and answered (a 1,273 s round). The head's `MemAvailable` sat at 6.2-6.3 GiB idle and bottomed around 3.3 GiB during the four parallel cold prefills (memguard's floor: 2.5 GiB): four parallel cold prefills of long prompts are the worst case for available memory, and staggering session starts a minute or two apart keeps the dip well above the floor. These numbers are with the head running headless — a desktop session on the head takes a GiB or two of the same headroom, and with one running the worst-case dip lands under memguard's floor (an earlier setup here, desktop on, could not hold even a 1,048,576-position pool and ran a halved window instead; the head went headless before the pool was raised at all).

“Computed prefill” counts prompt tokens processed again; “cached prefill” counts reused prompt tokens. Fresh unrelated prompts, including the default randomized `tools/bench.py` workload, are expected to have no cache hits. For a continuing long conversation, check the server's cached-token counters and use `python3 tools/prompt_reuse.py --require-counter`; a dashboard showing zero may also mean it does not read the server's counter. Short prompts may not retain a snapshot.

`/health`'s `prefill_tokens_per_second` is set when a prompt pass ends and shown for 2 s after it (the engine's
`zig/src/core/lane_host.zig`), so it reads 0 for the whole of a long pass: a dashboard showing high power with no decode
and prefill 0 is usually one long cold prefill, not a hang (#18, #34).

A Spark's GPU shares its memory with the host. A rank that takes all of it does not get a clean error: the machine stops
answering and has to be power-cycled (this happened once, with the engine's reserve lowered to 10 GiB). So:

- the engine's own reserve (`TENSORFOLD_MEMORY_RESERVE_GIB`, 12.2 GiB) is left alone: do not lower it to make the weights fit;
- `start.sh` checks `MIN_AVAIL_GIB` on both Sparks before it stops or starts anything, and stops a running server before
  it builds or downloads;
- `scripts/memguard.sh` runs on both Sparks while the server is up and kills the rank's container (kill first, log after)
  on either of two levels, sampling every 0.25 s: `MemAvailable` under `MEMGUARD_HARD_GIB` (1 GiB) in two samples in a row, or under
  `MEMGUARD_GIB` (2.5 GiB) for `MEMGUARD_HOLD_S` (4) seconds without once coming back over it. A long prompt's workspace takes a few GiB
  for a moment and gives them back; such a dip is logged with its length and lowest value (`... and recovered: no kill`) and the rank
  lives. It is a brake, not a guarantee: a very fast allocation can outrun it. Its log is
  `~/.cache/dsv41-tensorfold/logs/memguard.log`;
- **what holds `MemAvailable` above the floor while a rank runs: nothing**, with `ENGINE=zig`. The 12.2 GiB reserve only
  shapes what the engine allocates when it loads (its boot line reads `4 streams at once, 0.00 GiB each ... of 0.0 GiB left after a
  12.2 GiB reserve` on a head with a 524,288-token window: nothing is left over for the streams); a prompt's prefill workspace, the
  prompt cache (512 MiB), the graphs and everything else on the Spark then draw from what `MemAvailable` still shows, and
  `memguard` kills the rank when that reaches `MEMGUARD_GIB` ([issue #4](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/4):
  a head with a desktop, sparkDash and hermes read 7 GiB right after the load and was killed at 4.085 GiB during a cache-hit turn at 265K tokens).
  `start.sh` now prints `MemAvailable` on both Sparks after the load and warns when either is within 3 GiB of the floor. On a head
  that is that close: stop what does not need to run there (the desktop session: `sudo systemctl set-default multi-user.target`
  and reboot; other containers; sparkDash and similar on the worker or another machine), keep `MIN_AVAIL_GIB` as it is, and do not
  raise `MEMGUARD_GIB` (a higher floor kills sooner). The defaults are the loose end of what has been run here (2.5 GiB floor, 4 s hold, 1 GiB
  hard floor; the long-prompt runs in [Status and measurements](measurements.md) used 2.5): a lower floor or a longer hold trades the kill for the risk of freezing the Spark. `ENGINE=jl` does check at run time: it refuses admissions under a 4 GiB hard floor, falls
  back to 2K prompt segments and runs a missing graph eagerly instead of capturing it ([docs/jl-engine.md](jl-engine.md)); its boot check also sizes the pool to what the
  head can spare.
- the KV cache is small (compressed FP4 with block scales; about 0.5 GiB at 262,144 tokens and ~1.8 GiB at the full window,
  by two memory readings); it is sized from `CONTEXT` when the model loads and shared by the streams. Most of the
  memory is the weights (~99 GiB, of which ~3.5 GiB are the drafter's), so the head's other software decides what is left.
