# Status and measurements

What was checked on a pair of Sparks, and how it was measured. The headline decode and prefill numbers are in the [README](../README.md#performance).

## What is checked

- `DRY_RUN=1 ./start.sh` prints the link it found (on a pair of Sparks) and both ranks' docker commands; `./stop.sh`
  and `scripts/*.sh` pass `bash -n`.
- The kit's RoPE tables, made from this checkpoint's `config.json`, match the kit's recorded sha256 (4 of 4), and the
  token map made from its `tokenizer.json` is byte for byte the one the kit's Engram fixtures were recorded with.
  `scripts/prepare.sh` checks both on every run. The checkpoint holds all 43 `gate.bias_vl` tensors (F32) the image
  routing bias is made from.
- `zig/Dockerfile` builds (the image of 29 October 2026, `tensorfold-dsv41:exl3-tp2-7d71dcdf7f2f`, 24.9 GB, about 6 minutes of Zig compiling on a
  Spark): `tensorfold-native`, `tf-dsv41-lanes`, `tf-dsv41-rank-cache` and the three inputs' scripts are in it.
- On the pair, with that image (29 October 2026, engine `fbbc971` and the 33 patches, `TF_FILL_CHUNKS=4 TF_FILL_FIRST=4
  TF_FILL_CUT_IN=8192`): the replies of the 39 prompts of `tools/exact.py` (through 100,000 tokens, cached resumes, 8 concurrent
  against the same 8 alone) are identical to the previous image's; `tools/validate.py` passes (text, thinking, tool call, tool
  result, image, a 21,875-token prompt, 8 of 8 concurrent replies equal to their solo runs); two 30K-token cold prompts at once
  with short requests arriving during both (1.2-3.5 s each) and a client leaving mid-prompt leave the pair healthy; a short
  request sent 8 s into a 70K-token cold prompt takes 3.84 s (26 s without the `TF_FILL_*` switches).
- `tf-dsv41-rank-cache` plans the TP=2 split: run on a copy of the real headers (tensor bytes left out) of an earlier 43-shard
  2.9 bpw EXL3 checkpoint with the same tensor widths, it gives each rank 143,770 entries and 98.9 GiB, the ~98 GiB a
  Spark the weights need. (Not re-run on the 39-shard checkpoint.)

## Measurements

Served and measured on one pair of Sparks (the head also runs a desktop and ~18 small containers; the worker is otherwise
idle). Two sets of numbers, from two tools: sparkDash (the README's tables) and `tools/bench.py`.

An earlier sparkDash run, with one Spark's GPU locked at 1,800 MHz (the other at 2,200), read prose 74.6 and code 108.1
tok/s for one stream, 140.6 and 152.7 for four, and prefill 2,008-2,062 tok/s from 4K to 64K prompts (1,877 at 256K).

**`tools/bench.py`** (thinking off, temperature 0, random-word prompts so no prefix cache is hit and the drafter has little
to predict; medians of one or two repetitions, so differences of a few percent are noise):

| | Result |
| --- | --- |
| Prefill, 2K / 8K / 30K / 100K-token prompts | ~1,500-1,600 / ~1,960 / ~2,000 / ~1,950 tok/s (TTFT 1.2 s / 3.9 s / 14.4 s / 49 s) |
| Decode, 1 stream | 55-61 tok/s |
| Decode, 4 streams together | 113-117 tok/s |
| Drafting (DSpark) | ~2.2-2.4 tokens a round on random-word prompts (about 50 of 88 and 58 of 93 drafted tokens accepted) |

The lower `bench.py` decode numbers are the prompt, not the engine: real text drafts much better than random words.
`bench.py` talks to `--url`, else `API_URL`, else `PORT` from the environment, else a literal `PORT` assignment in `scripts/local.sh` (8888 otherwise), so it
follows a pair that moved the server off the recipe's default port.

These need **Engram's tables on local disk on both Sparks**: with the head reading them over NFS (`HEAD_ENGRAM_DIR` unset
and a remote `HF_CACHE`) a random 4 KiB read took ~940 us against ~90 us from local disk, and prefill was ~690-890 tok/s.
BertholomusAI's figures for the earlier checkpoint on two Sparks (decode 89.8 / 59.0 / 106.6 tok/s for code / prose /
structured, 4 streams 118.9 tok/s together, prompts at ~2,400 tok/s up to 128K) are the reference these are compared with:
with real prompts decode is above them (one stream: prose 80.7 against 59.0, code 118.1 against 89.8; four streams:
154.0 for prose and 166.1 for code against 118.9), and prefill is ~10-11% under them from 8K to 64K prompts (~2,136-2,155 against ~2,400 tok/s).

A needle-in-a-haystack test at `CONTEXT=524288` (`tools/bench.py --suite needle`, a passphrase hidden at 10%, 50% and 90%
depth of a 479,012-token prompt) found it 3 of 3 times; each prefill took ~280 s (~1,710 tok/s). The head's lowest free
memory during it was 2.67 GiB, 0.17 GiB above the 2.5 GiB floor `memguard` was set to, so a 512K window is the practical
limit for a head that carries ~6 GiB of other software. A reported run at `CONTEXT=655360` (issue #17; 4.9 GiB free after the load on
a head with its other services stopped) fell under the 2 GiB floor for the 4 s `MEMGUARD_HOLD_S` and had rank 0 killed on a turn
that took a cache-hit history from 408K to 464K tokens: the head's free memory falls as a history grows (see the CHANGELOG: 6.5 GiB after
the load, 4.9 at 159K tokens, 4.2 at 261K), and a larger `CONTEXT` only allows longer histories than the head can carry. Keep `CONTEXT` at 524288 or less on
such a head, or free more of its memory first. A 100K-token prompt also ran without errors at the full
1,048,576-token window (3.14 GiB free at the lowest) and at 524,288 tokens.

**Engine patches** (`zig/patches/`; `zig/patches/README.md` lists them). The indexer split runs by default in chunks ending at token 131,072 or later; idle RDMA back-off is also enabled. EAQ v1.1's mixed-pack loader (`0024` + `zig/patches-inputs/0001`), its BF16 -> f32 widen (`0025` + `zig/patches-inputs/0002`) and its plain-linears row padding (`0026`, `TF_DS_PLAIN_ROWS`) are automatic. The recorded token-hash checks cover 31 prompts through 100,000 tokens, cached resumes and four concurrent streams. Each switch is a `TF_DS_*` or `TF_RDMA_*` variable, which
`start.sh` gives to both ranks, and each was tested on its own:

| Switch | Result |
| --- | --- |
| `TF_DS_CACHE_PLACEMENT=1` (gone) | Measured on the earlier pin: in the three-conversation scenario conversation B's second turn used to miss the prompt cache and re-read its whole 6K-token prompt; with the patch it reused 4,096, with an identical reply. The pin is now `629c524`, whose engine places kept states itself (a conversation sharing a prefix copies the shared rows to a free extent); the patch was removed and the scenario was not re-measured on it. |
| `TF_DS_ENGRAM_LANES=2` (gone) | A second I/O ring for the prompt's Engram reads; it was bit-identical but prefill 1.5-2.6% *slower* from 8K to 131K. Since the pin moved to `fbbc971` the engine's own `TF_DS_ENGRAM_LEAD=1` (`9673a5e`) reads each Engram layer's prompt rows a chunk ahead on a reader of its own, on the same second ring, and `0002` is gone. |
| `TF_DS_IDX_SPLIT` (unset: from 131,072 tokens) | Splits the prefill indexer's rows between the two ranks and all-gathers their selections. Bit-identical. The indexer's work grows with the history and the gather's does not, so it pays only in long prompts: with the switch unset it runs in chunks that end at 65,536 compressed keys or more (token 131,072 of a ratio-2 layer) and below that the replicated loop runs as before. Cold prefill against the image without it, three alternating pairs each: **+4.25% at 256K** (1,976 to 2,060 tok/s) and **+7.46% at 400K** (1,874 to 2,014 tok/s, TTFT 213 to 198 s); 8K/32K prefill, decode and four streams within 0.9%. `1` splits every eligible chunk (earlier: no measurable gain up to 128K), any other value turns it off. Both ranks must agree or the pair hangs: `start.sh` gives the variable to both. |
| `TF_RDMA_IDLE_S` / `TF_RDMA_NAP_US` (on by default) | The RDMA gather's proxy naps when the engine is idle instead of spinning a core on each rank (the logic of [TensorFold PR #27](https://github.com/bertholomus/TensorFold/pull/27), ported to the Zig ring). Recipe defaults: 2 s quiet, then 500 µs between looks; the patch's engine fallback is 100 µs if `TF_RDMA_NAP_US` is unset. `TF_RDMA_IDLE_S=0` restores the stock always-spinning proxy. A decode round's gathers never see a nap. Measured on this pair with 100 µs: 11.2% (rank 0) and 8.8% (rank 1) idle, against ~100% before; at 500 µs the wake rate falls 5x, putting the proxy at about 2% a rank, with at most 0.5 ms on the first gather after a quiet spell. |

Two limits of BertholomusAI's engine, reported in issues [#2](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/2) and
[#3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/3), are patched here (`zig/patches/0005`, and `0006` until the pin moved to `fbbc971`, whose `3de79d8` does it), and measured on the pair
(engine `629c524` + patches; the replies of 31 `tools/exact.py` prompts, 4 concurrent among them, are identical to the previous image's):
- **A prompt pass no longer freezes the other streams** (`0005`, by Coinupbtc, pull request #5; `PREFILL_INTERLEAVE_MS`, 250 by default, 0 turns it off): between two
  chunks of a long prompt pass, the streams already live get decode rounds and their cancels are taken. A 120K-token cold prompt used to leave a decoding
  stream without a token for 90 s; now its longest gap is 1.3 s, and a cancel is answered 0.74 s after the client leaves, in the middle of the other
  prompt's pass. The cold prompt itself takes about 10 % longer (99 s against 91 s).
- **A prompt with pictures resumes and keeps prompt states** (`0006`, the engine's own `3de79d8` since `fbbc971`; `TF_DS_IMAGE_CACHE=0` turns it off): the cache keys the pictures' positions by the
  picture's content. With a picture first and 27K tokens after it, turn 2 resumed 26,624 tokens in 0.8 s (cold: 14 s); another picture with identical text was
  cold, and the first picture again hit with a byte-identical reply.
`ENGINE=jl` interleaves prompt pieces with other slots' decode (`TF_DSV41_PREFILL_PIECES`); its image prompt reuse was not tested here.

What is **not** checked: video input (`tools/validate.py` skips it where PyAV is missing); other cabling, hardware or a head with
less memory to spare; days of uptime; the spill tier and KV sharding on the current image (both are off in the deployment it was
checked in); and the engine's behaviour at the memory floor under heavy concurrent long prompts (the watchdog exists because
that can freeze a Spark; see [Memory and safety](memory-and-safety.md)).
