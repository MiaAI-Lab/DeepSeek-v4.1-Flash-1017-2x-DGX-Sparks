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
(196 GiB; experimental: its model card has no perplexity or KL measurement against the unquantized model).

- Checkpoint: `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` (39 shards, routed experts at 3 bits a
  weight, 2 bits in layers 18-22, attention at 5, shared experts at 4-5, head at 6, DSpark blocks at 4; the vision tower
  and the DSpark blocks are in it; the revision is pinned in `scripts/config.sh`), plus Engram's two tables, unquantized,
  from `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` (shards 47 and 48, ~203 GB; the quantization leaves them out)
- API model id: `DeepSeek-V4.1-Flash-1017`, port 8888
- Engine: TensorFold native (Zig 0.17.0), BertholomusAI's `dsv41` family at TP=2, one rank on each Spark
- One command on the first Spark: `./start.sh` sets up both Sparks and starts both ranks; `./stop.sh` stops them
- Optional `ABLIT=1`: keeps Mia EAQ 2.9 bpw and applies the matching [Keys overlay by drowzeys](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated).
  Requires your Hugging Face token and access to that gated repository; see [Abliterated mode](#abliterated-mode).

## Status

What is checked, and how:

- `DRY_RUN=1 ./start.sh` prints the link it found (on a pair of Sparks) and both ranks' docker commands; `./stop.sh`
  and `scripts/*.sh` pass `bash -n`.
- The kit's RoPE tables, made from this checkpoint's `config.json`, match the kit's recorded sha256 (4 of 4), and the
  token map made from its `tokenizer.json` is byte for byte the one the kit's Engram fixtures were recorded with.
  `scripts/prepare.sh` checks both on every run. The checkpoint holds all 43 `gate.bias_vl` tensors (F32) the image
  routing bias is made from.
- `zig/Dockerfile` builds (`tensorfold-dsv41:exl3-tp2-136c93b359b4`, 24.9 GB, about 6 minutes of Zig compiling on a
  Spark): `tensorfold-native 1.0.2`, `tf-dsv41-lanes`, `tf-dsv41-rank-cache` and the three inputs' scripts are in it.
- `tf-dsv41-rank-cache` plans the TP=2 split: run on a copy of the real headers (tensor bytes left out) of an earlier 43-shard
  2.9 bpw EXL3 checkpoint with the same tensor widths, it gives each rank 143,770 entries and 98.9 GiB, the ~98 GiB a
  Spark the weights need. (Not re-run on the 39-shard checkpoint.)

Served and measured on one pair of Sparks (the head also runs a desktop and ~18 small containers; the worker is otherwise
idle). Two sets of numbers, from two tools:

**[sparkDash](https://github.com/MiaAI-Lab/sparkDash)** (real prose and code prompts, as sparkDash reported them, with the
server at `CONTEXT=524288`, up to 4 streams, drafts on):

| Decode, tok/s | 1 stream | 2 streams | 4 streams |
| --- | --- | --- | --- |
| Prose | 74.9 (TTFT 131 ms) | 105.5 total, 55.0 each (TTFT 243 ms) | 130.0 total, 34.9 each (TTFT 411 ms) |
| Code | 109.9 (TTFT 200 ms) | 139.8 total, 74.8 each (TTFT 291 ms) | 155.5 total, 44.4 each (TTFT 511 ms) |

| Prefill, prompt tokens | 8K | 16K | 32K | 64K | 128K | 256K |
| --- | --- | --- | --- | --- | --- | --- |
| tok/s | 2,166 | 2,183 | 2,175 | 2,136 | 2,079 | 2,000 |
| TTFT | 4.0 s | 7.7 s | 15.3 s | 30.9 s | 63.2 s | 131.3 s |

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

These need **Engram's tables on local disk on both Sparks**: with the head reading them over NFS (`HEAD_ENGRAM_DIR` unset
and a remote `HF_CACHE`) a random 4 KiB read took ~940 us against ~90 us from local disk, and prefill was ~690-890 tok/s.
BertholomusAI's figures for the earlier checkpoint on two Sparks (decode 89.8 / 59.0 / 106.6 tok/s for code / prose /
structured, 4 streams 118.9 tok/s together, prompts at ~2,400 tok/s up to 128K) are the reference these are compared with:
with real prompts decode is above them (one stream: prose 74.9 against 59.0, code 109.9 against 89.8; four streams:
130.0 and 155.5 against 118.9), and prefill is ~9-11% under them from 8K to 64K prompts (~2,140-2,180 against ~2,400 tok/s).

A needle-in-a-haystack test at `CONTEXT=524288` (`tools/bench.py --suite needle`, a passphrase hidden at 10%, 50% and 90%
depth of a 479,012-token prompt) found it 3 of 3 times; each prefill took ~280 s (~1,710 tok/s). The head's lowest free
memory during it was 2.67 GiB, 0.17 GiB above the 2.5 GiB floor `memguard` was set to, so a 512K window is the practical
limit for a head that carries ~6 GiB of other software. A reported run at `CONTEXT=655360` (issue #17; 4.9 GiB free after the load on
a head with its other services stopped) fell under the 2 GiB floor for the 4 s `MEMGUARD_HOLD_S` and had rank 0 killed on a turn
that took a cache-hit history from 408K to 464K tokens: the head's free memory falls as a history grows (see the CHANGELOG: 6.5 GiB after
the load, 4.9 at 159K tokens, 4.2 at 261K), and a larger `CONTEXT` only allows longer histories than the head can carry. Keep `CONTEXT` at 524288 or less on
such a head, or free more of its memory first. A 100K-token prompt also ran without errors at the full
1,048,576-token window (3.14 GiB free at the lowest) and at 524,288 tokens.

**Engine patches** (`zig/patches/`, three, written for this recipe). Each is **off by default**, so the image behaves like
the pinned engine: its replies' token hashes equal the unpatched image's on 31 of 31 prompts (1 to 100,000 tokens, cached-prefix
resumes, four at once, a three-conversation scenario; `tools/exact.py`). Each is switched on with a `TF_DS_*` variable, which
`start.sh` gives to both ranks, and each was tested on its own:

| Switch | Result |
| --- | --- |
| `TF_DS_CACHE_PLACEMENT=1` (gone) | Measured on the earlier pin: in the three-conversation scenario conversation B's second turn used to miss the prompt cache and re-read its whole 6K-token prompt; with the patch it reused 4,096, with an identical reply. The pin is now `629c524`, whose engine places kept states itself (a conversation sharing a prefix copies the shared rows to a free extent); the patch was removed and the scenario was not re-measured on it. |
| `TF_DS_ENGRAM_LANES=2` (off) | A second I/O ring for the prompt's Engram reads. Bit-identical, but prefill 1.5-2.6% *slower* from 8K to 131K: the drive is the limit. |
| `TF_DS_IDX_SPLIT=1` (off) | Splits the prefill indexer's rows between the two ranks. Bit-identical, with no measurable gain on this pair (modeled at +1% at 64K and +2% at 128K; the undisturbed 32K repetitions tie the baseline). If the two ranks ever disagree on the setting the pair hangs, so it stays off. |

Two limits of BertholomusAI's engine, reported in issues [#2](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/2) and
[#3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/3), are patched here (`zig/patches/0005`, `0006`), and measured on the pair
(engine `629c524` + patches; the replies of 31 `tools/exact.py` prompts, 4 concurrent among them, are identical to the previous image's):
- **A prompt pass no longer freezes the other streams** (`0005`, by Coinupbtc, pull request #5; `PREFILL_INTERLEAVE_MS`, 250 by default, 0 turns it off): between two
  chunks of a long prompt pass, the streams already live get decode rounds and their cancels are taken. A 120K-token cold prompt used to leave a decoding
  stream without a token for 90 s; now its longest gap is 1.3 s, and a cancel is answered 0.74 s after the client leaves, in the middle of the other
  prompt's pass. The cold prompt itself takes about 10 % longer (99 s against 91 s).
- **A prompt with pictures resumes and keeps prompt states** (`0006`; `TF_DS_IMAGE_CACHE=0` turns it off): the cache keys the pictures' positions by the
  picture's content. With a picture first and 27K tokens after it, turn 2 resumed 26,624 tokens in 0.8 s (cold: 14 s); another picture with identical text was
  cold, and the first picture again hit with a byte-identical reply.
`ENGINE=jl` interleaves prompt pieces with other slots' decode (`TF_DSV41_PREFILL_PIECES`); its image prompt reuse was not tested here.

What is **not** checked: image input and `tools/validate.py` on this pair; other cabling, hardware or a head with less
memory to spare; days of uptime; and the engine's behaviour at the memory floor under heavy concurrent long prompts (the
watchdog exists because that can freeze a Spark; see [Memory and safety](#memory-and-safety)).

## Requirements

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory), with nothing else large on them: the weights take
  ~98 GiB a Spark and rank 0 holds ~104 GiB on the GPU, so a Spark's memory is the limit, not its disk or its link.
  `start.sh` refuses to start unless `MIN_AVAIL_GIB` (112) is available on each Spark. A head that also runs a desktop
  and development tools (~6 GiB here) was left with ~3.1 GiB at the worst point of a 100K-token prompt at the full window
  and ~3.7 GiB at 524,288 tokens: give the head less to do, or lower `CONTEXT`.
- **The same GPU clock ceiling on both Sparks** (`nvidia-smi -lgc 0,2200`; check with `nvidia-smi -q -d CLOCK`). With TP=2 the
  ranks meet twice a layer, so the slower GPU sets the pace: one Spark found locked at 1,800 MHz cost about 3-6% of prefill
  and 3% of decode against 2,200 MHz on both (rigmark, same prompts; the GPUs peaked at 85 / 81 C and nothing throttled).
- **A direct ConnectX-7 link:** a QSFP cable between the CX7 ports and an IPv4 address on each end in one private subnet
  (e.g. `192.0.2.1/24` and `192.0.2.2/24`; `ping` must work), with a RoCE v2 GID (`start.sh` checks). One QSFP port of a
  Spark reaches the GB10 over two PCIe Gen5 x4 links, so it appears as two netdevs and two RoCE devices (`enp1s0f0np0` /
  `enP2p1s0f0np0`, `rocep1s0f0` / `roceP2p1s0f0`). Address both twins on each Spark (one subnet each is fine, the same
  MTU on both Sparks): both are then used as two rails; one rail is one x4 (~112 Gb/s of the port's 200).
- **Key-based ssh** from the first Spark (the head, which runs `./start.sh` and the API) to the second (the worker):
  `ssh-copy-id user@<worker>`; check with `ssh -o BatchMode=yes user@<worker> true`.
- Docker with the NVIDIA container runtime and your user in the `docker` group, on both Sparks; `git`, `python3` and the
  `hf` CLI (`pip install -U huggingface_hub`) on the head.
- **Disk.** Head: ~505 GB (the checkpoint 196 GiB, Engram's tables ~203 GB, rank 0's weight file ~106 GB) plus ~25 GB for
  the image under Docker's root. Worker: ~310 GB (Engram's tables and rank 1's weight file) plus the image; the worker
  reads the checkpoint from the head over NFS.
- **NFS.** The worker mounts the head's Hugging Face cache read-only. On the head, once (needs root; the worker needs
  nothing installed):

  ```bash
  sudo apt install nfs-kernel-server
  echo "$HOME/.cache/huggingface <worker CX7 address>(ro,no_subtree_check)" | sudo tee -a /etc/exports
  sudo exportfs -ra
  ```

  If `prepare.sh` then stops with "cannot mount this Spark's ... over NFS", read the lines under it: they are the end of docker's own
  error (the reason is above docker's closing "Run 'docker run --help'" line). Check `sudo exportfs -v` on the head: the export must
  list the *worker's* address on the link (the user@address in the message), the exported path must be `NFS_PATH`, and the worker must
  reach the head's port 2049; the worker's docker mounts the volume itself, so it needs the host's NFS mount helper (`nfs-common` on
  Ubuntu) if it reports a missing `mount.nfs` helper.
  `prepare.sh` then creates a read-only docker NFS volume on the worker (`NFS_VOLUME`, no sudo) and checks that the worker
  sees every file the ranks read. `NFS_PATH` is the exported path as the worker mounts it (default: the head's `HF_CACHE`;
  `/` for an NFSv4 export with `fsid=0`), `NFS_SERVER` the head's address (default: its address on the link).

  **Or no NFS at all.** Each Spark can run from a local copy: set `HF_CACHE` to a local tree holding the checkpoint's small files
  and its vision shard (`model-00039-of-00039`, the only shard read at run time), the served `model/` folder, the kit, the token
  map and, on the head, rank 0's weight file; set `NFS_VOLUME` to the same path as a *host path* (the worker bind-mounts it) and
  leave `HEAD_NFS_SERVER` empty. Set `PREPARE=0` too: `scripts/prepare.sh` assumes a complete cache and would start downloading
  the 196 GiB checkpoint into a partial one.

## Quick start

On the head:

```bash
git clone https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks.git
cd DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
cp scripts/local.sh.example scripts/local.sh     # set WORKER=user@192.0.2.2 in it (and FABRIC_PEER, see below)
DRY_RUN=1 ./start.sh                              # prints what it would do; changes and starts nothing
./start.sh
```

`WORKER` is the worker's ssh target. The ranks talk over the route to it: if it is on another network than the link, set
`FABRIC_PEER` to the worker's CX7 address. The worker needs no copy of this repository.

The first run sets up both Sparks (see below): the image on each (two Zig builds, ~6 minutes on a Spark), the checkpoint (196 GiB)
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

`stop.sh` saves each rank's log (gzipped, newest 10 kept, `LOG_KEEP`) to `~/.cache/dsv41-tensorfold/logs` before it removes
the container.

## Abliterated mode

`ABLIT` accepts **`0` or `1`** (default `0`). Set it in the environment, `scripts/local.sh` or `.env`.
`ABLIT=0` serves the stock Mia EAQ v1 2.9 bpw checkpoint. `ABLIT=1` keeps that checkpoint and applies
`exl3_wo_b_l10_35_mia29.safetensors` from
[`drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated`](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated).
This repository supplies overlays, not a full checkpoint. The recipe uses its **matching Mia 2.9 bpw** overlay and
`apply_mia29_wo_b_graft.py`; the FP8, LibertAI 3.5 bpw and Jayleaton 2.8 bpw sidecars are for other packs.

Before preparing or starting with `ABLIT=1`:

1. Log in to Hugging Face and obtain access by accepting the terms on the
   [gated repository page](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated).
2. Use a Hugging Face read token from **the same account**, with permission to read that gated repository.
   Authenticate on the head with `hf auth login`, or export `HF_TOKEN` as below. A token alone does not grant repository access.

```bash
hf auth login                 # enter your HF read token, after obtaining repository access
ABLIT=1 ./scripts/prepare.sh   # optional: prepare separately; start.sh also prepares automatically
ABLIT=1 ./start.sh restart    # restart both ranks with the overlay
ABLIT=0 ./start.sh restart    # switch back to stock
```

Alternatively, supply your token through the environment without writing it into a command's history:

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN; echo
export HF_TOKEN
ABLIT=1 ./start.sh restart
unset HF_TOKEN
```

Both `scripts/prepare.sh` and `start.sh` print the token and gated-access requirement when `ABLIT=1`, including a link
to the repository. Preparation downloads the matching overlay before the image build and large checkpoint downloads,
and reports an access error if the download fails. `hf` uses the token on the head; the recipe does not pass `HF_TOKEN`
in command arguments or forward it as an environment variable to the worker or serving containers. A saved login token
inside `HF_CACHE` is part of the shared cache; use `HF_TOKEN` or a token path outside that cache to keep it out of the mounts.
`DRY_RUN=1 ABLIT=1 ./start.sh` prints the notice without downloading or checking access.

The overlay is pinned at `0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0` (`ABLIT_REVISION`, a full commit hash).
It replaces 104 attention tensors in layers 10–35. Its application script also converts four MTP tensors from F16 to F32
for another Zig engine, as described in the upstream `TENSORFOLD-ZIG-MIA.md`; preparation then gives them their stock F16
back (`scripts/ablit_stock_dtypes.py`). This recipe's engine reads the drafter's `mtp.2.confidence_head.proj.weight` as F16
whatever the file holds: from the F32 copy it drafts 0.2–0.8 tokens a round and decode falls 25–45% below stock. A
checkpoint prepared before that step gets it on the next start, which also rebuilds both ~106 GB weight files. The stock Hugging Face snapshot stays intact:
preparation hardlinks unchanged shards and writes changed shards under a separate
`$HF_CACHE/tensorfold-dsv41/<model-revision>-ablit-<overlay-revision>/checkpoint` directory. **Allow extra disk space for
rewritten shards (up to another checkpoint's size), the ~651 MiB sidecar, and a separate ~106 GB rank cache on each Spark.**
Applying the overlay runs on CPU using the image's Python, PyTorch and safetensors.

Both ranks build their caches from the modified checkpoint. Abliterated data and worker cache paths include the overlay
revision, including when `WORKER_CACHE_DIR` is customized, so stock caches cannot be reused accidentally. Keep the same
`ABLIT` setting for preparation and startup; put `ABLIT=1` in your local configuration to make it persistent. Use
`restart` when switching modes on a running server. Startup checks both running containers' mode and overlay revision;
a separately prepared checkpoint cannot make a stock server count as abliterated. The API model name remains `SERVED_NAME`.
The stock performance and correctness results above do not validate the abliterated mode on this recipe's engine.

## What `start.sh` and `scripts/prepare.sh` do

`scripts/prepare.sh` (run by `start.sh` when something is missing; safe to re-run) on the head:

1. checks docker, the GPU, ssh and the link;
2. builds the image from `zig/Dockerfile` (the pinned engine plus the inputs' tools) and copies it to the worker;
3. downloads the checkpoint and Engram's two shards (with their index and config) at their pinned revisions, and checks the
   shards' sizes (`VERIFY_SHA=1` also their sha256, as the model card lists them: the 1017 tables are new, the older
   release's must not be used);
4. fetches the kit at `KIT_REF`, makes its four RoPE tables and the image routing bias from the checkpoint, and checks every
   file against the kit's MANIFEST (the image bias may differ: it is 1017's own);
5. writes Engram's token map and rank 0's weight file from the checkpoint (no GPU), and the folder rank 0 serves (the
   checkpoint's files linked, with `model/chat_template.jinja`: the release ships its chat format as `encoding/encoding.py`,
   which the server cannot run);
6. creates the worker's NFS volume and checks it;
7. copies Engram's tables to the worker's disk (a decode round reads ~70 random 4 KiB rows from them; over NFS that took most
   of a round) and writes rank 1's weight file there from the checkpoint over NFS.

`start.sh` then starts rank 1 on the worker (`tf-dsv41-lanes`, following) and rank 0 here (`tensorfold-native serve`), waits
for the API, and runs a smoke test.

## Configuration

Settings come from the environment, `scripts/local.sh` or `.env`; the defaults and their notes are in
[`scripts/config.sh`](scripts/config.sh). The ones you are likely to touch:

| Setting | Default | |
| --- | --- | --- |
| `MODEL_ID`, `MODEL_REVISION` | `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` @ `6d54be4f` | the checkpoint. The Hub repo was renamed from `...-1017-EAQ-v1-2.9-bpw` (the old name redirects), and the Hugging Face cache is keyed by the repo's name and revision: a copy under the old name or an older revision (e.g. `e8ac7684`, which the Hub no longer has) is not found under the default, and `scripts/prepare.sh` would download 196 GiB. It now stops when the disk cannot hold that. To keep serving a copy you have, set both to what `~/.cache/huggingface/hub` holds |
| `ABLIT` | 0 | `0`: stock Mia EAQ 2.9 bpw; `1`: apply the matching gated Keys overlay; requires your HF token and repository access (see Abliterated mode) |
| `ABLIT_REVISION` | `0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0` | pinned Keys overlay and application script; full commit hash |
| `HF_TOKEN` | unset | Hugging Face read token for downloads; alternatively authenticate with `hf auth login` on the head |
| `CONTEXT` | 1048576 | a request's positions at most; the pool follows it (two windows with the default `KV_SHARD=1`, one with `KV_SHARD=0`) unless `POOL` is set |
| `PARALLEL` | 4 | requests at once (the engine's streams, 1 to 16; the family was measured at 4) |
| `THINKING`, `REASONING_EFFORT` | 1, empty | think first unless a request turns it off; `low` / `medium` / `high` / `xhigh` as the default effort |
| `THINKING_BUDGET` | empty | reasoning tokens a reply may spend before the engine closes the think block itself and lets the answer continue (`--thinking-budget`; a request's `thinking_budget` wins); empty: no cap. A hard task can otherwise spend its whole `MAX_TOKENS` thinking and end with no text and no tool call. A model that keeps rambling after the cut is not stopped by it. For long agentic or tool-calling jobs, leave it empty or set it large: a small budget cuts the reasoning mid-plan, which can make the model repeat a tool call it already made |
| `TEMPERATURE`, `TOP_P`, `TOP_K` | 1.0, 0.95, 20 | the server's sampling defaults; a request's own values win. This engine refuses a `top_k` above 56 with `TooManyCandidates` (the finite candidate list its sampler keeps), so `TOP_K` is at most 56; `top_k` 0 or -1 is refused too, with a different, misleading message |
| `DRAFTS` | 1 | the checkpoint's DSpark draft blocks (drafted replies equal plain ones) |
| `RDMA`, `GRAPHS`, `SIDE`, `PREFETCH`, `ENGRAM_AIO` | 1 | the engine's served settings |
| `MAX_TOKENS` | 32768 | a reply's budget when a request sets none |
| `MIN_AVAIL_GIB`, `MIN_AVAIL_WAIT_S` | 112, 30 | `start.sh` refuses to start with less memory available than this on either Spark (`0` skips the check); it re-reads the memory for up to `MIN_AVAIL_WAIT_S` seconds first, because the page cache a build or a large copy just left counts as available only once it is written back (`0` reads once) |
| `MEMGUARD_GIB`, `MEMGUARD_HOLD_S`, `MEMGUARD_HARD_GIB` | 2.5, 4, 1 | `scripts/memguard.sh` kills a rank's container when the memory available stays under the floor for `MEMGUARD_HOLD_S` seconds without coming back over it, or at once when it falls under the hard floor. A dip that recovers is logged and not killed (decimals fine; `MEMGUARD_GIB=0` turns the guard off) |
| `HEAD_ENGRAM_DIR` | empty | a local folder with Engram's two shards for rank 0 (as `WORKER_ENGRAM_DIR` is on the worker); empty reads them from `HF_CACHE` |
| `HEAD_NFS_SERVER`, `HEAD_NFS_VOLUME` | empty | rank 0 reads the checkpoint over NFS through a docker volume the docker daemon mounts (no root, survives a reboot); empty uses `HF_CACHE` as a host path. sshfs does not work: it lists files as type "unknown", which the engine skips |
| `TF_DS_ENGRAM_LANES`, `TF_DS_IDX_SPLIT` | unset | the engine patches' switches (`zig/patches/`, see Status); a `TF_DS_*` variable you export goes to both ranks |
| `RANK1_PROFILE` | empty | `1` or `2` starts rank 1 with `--profile`, which prints its prefill phase timings when rank 0 stops first (adds a sync per chunk: for attribution, not for timing) |
| `API_KEY` | 0 | `1` makes rank 0 require a key (`Authorization: Bearer`; generated in `$STATE_DIR/api-key`, mode 600). The API is open on the network by default |
| `HOST`, `PORT`, `SERVED_NAME` | 0.0.0.0, 8888, `DeepSeek-V4.1-Flash-1017` | |
| `WORKER`, `FABRIC_PEER`, `NFS_*` | | see Requirements |
| `TF_REF`, `INPUTS_REF`, `KIT_REF` | pinned | the engine, the inputs' tools and the kit (changing one rebuilds or re-checks) |
| `SPILL_GIB`, `SPILL_DIR`, `WORKER_SPILL_DIR`, `SPILL_FLOOR_GIB`, `SPILL_QUOTA_GIB` | 64 (`0`: off), beside `HF_CACHE`, `SPILL_DIR`, 50, 0 | spill v2: kept prompt states on each Spark's own NVMe ([Spill v2 and KV sharding](#spill-v2-and-kv-sharding-ported-on-by-default)) |
| `KV_SHARD`, `KV_SHARD_DECODE`, `POOL` | 1 (`0`: off), empty, empty (= the pool `KV_SHARD=1` needs, 2101248 at the 1M window; `CONTEXT` with `KV_SHARD=0`) | KV sharding and the pool apart from the window (same section) |
| `ZIG_MIRRORS` | `https://zig.linus.dev/zig` | where the image build downloads Zig first (ziglang.org last), with bounded retries; the bytes are pinned by `ZIG_SHA256` |
| `PULL` | 1 | `0` never pulls a published image (`GHCR_IMAGE`, `IMAGE_TAG`, `IMAGE_DIGEST`; the image of these pins is `ghcr.io/miaai-lab/deepseek-v4.1-flash-1017-2x-dgx-sparks-tensorfold:exl3-tp2-7c9bf5d36350`) |

Extra arguments to `./start.sh` go to rank 0's `tensorfold-native serve` after the defaults; every `TENSORFOLD_*`,
`TF_DS_*` and `TF_SPILL_*` variable you set goes to both ranks. Images are always on (rank 0 runs the vision tower).

## Spill v2 and KV sharding (ported, on by default)

The port of Robert Wojciechowski's ([wojo](https://github.com/wojo)) GLM-5.3-Flash patches into this engine: the spill
tier and its v2 block store (GLM `0088` and `0110`, PRs #78 and #124; the block layout, lineage, write-at-keep ring and
fsync order are [BadAd84](https://github.com/BadAd84)'s block-tier design), chat continuation (`0111`, PR #128), KV sharding (`0118`, PR #152) and the parked-page startup budget (`0119`, PR #149). The
engine changes come in as `zig/patches/0007-...` (exported from the port's engine branch by `scripts/port-patches.sh`);
**the recipe turns them on by default**, as the daily server runs them: spill v2 at `SPILL_GIB=64` a Spark, KV
sharding (`KV_SHARD=1`) and the 2,101,248-position pool it needs at the 1,048,576-token window
(derived from `CONTEXT`, so another window gets its own). The engine's own switches stay off by default; only the
recipe's settings changed. Status: validated on the pair (replies byte-identical to the stock engine in every
configuration: all off, `POOL`, KV sharding at 1 and 8 streams, spill, spill with KV sharding; restore after
eviction, a restart and damaged files); the fork serving on gx10-a runs this configuration, updated as
[docs/FORK-DEPLOY.md](docs/FORK-DEPLOY.md) says. Full detail: [docs/PORT-RECIPE.md](docs/PORT-RECIPE.md).

**Turning them off:** `SPILL_GIB=0` (spill), `KV_SHARD=0` (sharding, and with `POOL` empty the pool is `CONTEXT`
again), both for the recipe as before the port (no flag, variable or mount the engine did not get before). **A spill
folder that cannot be used** (missing and not creatable, not yours, on NFS or tmpfs, inside the export or a model
folder, below `SPILL_FLOOR_GIB` free) on either Spark makes a default start go on **without** the tier, with a loud
`SPILL TIER OFF for this start` warning; set `SPILL_GIB` explicitly (`SPILL_GIB=64` in `scripts/local.sh`) and the
same problem refuses the start instead. The same holds for the default `KV_SHARD=1` where it cannot run (an explicit
`POOL` it cannot use, such as `POOL=1048576`): off with a warning, explicit `KV_SHARD=1` refuses.

- **Spill v2** (`SPILL_GIB`): each kept prompt state is written, when it is kept, to each Spark's own local disk in
  2,048-token blocks that later turns of the conversation share, and read back instead of a new prefill: after
  eviction, after a restart, after a crash. Room is made by conversation (old turns past the newest few, abandoned
  forks, whole conversations least recently used, shared prefixes last), with an optional per-conversation quota.
- **KV sharding** (`KV_SHARD=1`): each conversation's compressed latents are stored once, on the Spark whose half of the
  pool holds it, instead of on both. It needs `POOL`, the total pool positions, apart from `CONTEXT`, the window one
  request may use: each Spark's half must hold a whole window, so at the 1,048,576-token window `POOL=2101248` (start.sh
  refuses a pool that would cut a conversation's window and names the value; left empty, it is derived). Calculated:
  530 instead of 890 bytes a token a Spark for these planes, about 1,060 MiB a Spark for the 2,101,248-position pool
  against 890 MiB for the replicated 1M pool. Runs together with spill (`port/spill-shard`).

| Switch | Default | |
| --- | --- | --- |
| `SPILL_GIB` | `64` (`0`: off) | the spill tier's cap a Spark, GiB. Left at the default, a Spark whose folder fails the checks starts the pair without the tier (a warning); set explicitly, that refuses |
| `SPILL_DIR` / `WORKER_SPILL_DIR` | `$(dirname HF_CACHE)/dsv41-tensorfold-spill` / `SPILL_DIR` | each rank's own folder on its local NVMe, mounted at `/spill`. Never inside `HF_CACHE` (it is the NFS export the worker reads), never on a network file system, never under an archive folder: `start.sh` checks this on both Sparks, creates the folder as you, and refuses a root-owned one |
| `SPILL_FLOOR_GIB` | `50` | free disk the tier always leaves on that file system |
| `SPILL_QUOTA_GIB` | `0` (none) | one conversation's share of `SPILL_GIB` before its own old turns and forks go first |
| `SPILL_MIN_TOKENS`, `SPILL_STAGING_MIB`, `SPILL_TURNS`, `SPILL_FLUSH_MS` | `8192`, `128`, `4`, `60000` | the tier's shortest written state, pinned staging, turns kept, stop flush bound: the same on both ranks |
| `KV_SHARD`, `KV_SHARD_DECODE` | `1` (`0`: off), empty (`fetch`) | KV sharding; `owner` for owner-computed decode |
| `KV_SHARD_CARRY`, `KV_SHARD_TWINS`, `KV_SHARD_CHECK` | `1`, `1`, `1024` | carries and live moves, borrowed twins, rounds between the ranks' planner checks |
| `POOL` | empty (with `KV_SHARD=1`: two owner parts of a whole `CONTEXT` window each, 2101248 at 1048576; with `KV_SHARD=0`: `CONTEXT`) | total pool positions; rank 1's `--pool` |

Both ranks get the same settings (the engine's flags and variables are mapped in one place, `scripts/port.sh`); both
containers are labelled with them, and `./start.sh` will not adopt two running ranks that disagree. With the spill tier
on, `start.sh` also gives both ranks the identity the engine requires (the image's patch hash and ref, the checkpoint's
and Engram's revisions, each rank's weight-file digest, computed once and cached beside the file). **Disk use**
(calculated): ~890 bytes a token a Spark replicated, 1,822,720 bytes per 2,048-token block (~85 MiB for a 100K-token
conversation, ~890 MiB for the 1M window) plus 6.4 MiB a kept point a Spark; with KV sharding (the default) the owner
Spark writes ~890 and the other ~170 bytes a token. Blocks a later turn shares are stored once. The tier holds at most
`SPILL_GIB` (64 GiB) of each Spark's own disk and always leaves `SPILL_FLOOR_GIB` (50 GiB) free there. The engine runs as root; spill
files are given back to the folder's owner before each start and at every stop. With the tier on, `./stop.sh` gives
the ranks 90 s (`SPILL_FLUSH_MS` plus 30 s) to write what still waits.

**Rollback:** `SPILL_GIB=0` / `KV_SHARD=0` (and `POOL` empty) with `./start.sh restart` turns a feature
off on the same image. The previous image comes back with `SPILL_GIB=0 KV_SHARD=0 IMAGE=<its tag> PREPARE=0 ./start.sh
restart` (every port setting off, explicitly: the defaults are on), or by restoring the exact tree it was built from, commit **and** working-tree diff (the image tag is a
hash of `zig/patches` and `zig/Dockerfile` as they are on disk), then `./start.sh restart`: record both at deploy, as
[docs/PORT-RECIPE.md](docs/PORT-RECIPE.md) describes. Spilled files stay in each Spark's folder for inspection
(`rm -rf <SPILL_DIR>/native-spill-*` reclaims them); no weights, Engram or NFS change in either direction, and the
memory guards are unchanged.

## Memory and safety

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
  hard floor; the long-prompt runs in Status used 2.5): a lower floor or a longer hold trades the kill for the risk of freezing the Spark. `ENGINE=jl` does check at run time: it refuses admissions under a 4 GiB hard floor, falls
  back to 2K prompt segments and runs a missing graph eagerly instead of capturing it ([docs/jl-engine.md](docs/jl-engine.md)); its boot check also sizes the pool to what the
  head can spare.
- the KV cache is small (compressed FP4 with block scales; about 0.5 GiB at 262,144 tokens and ~1.8 GiB at the full window,
  by two memory readings); it is sized from `CONTEXT` when the model loads and shared by the streams. Most of the
  memory is the weights (~99 GiB, of which ~3.5 GiB are the drafter's), so the head's other software decides what is left.

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
scripts/port-patches.sh      exports the port's engine branch as zig/patches/0007-..., checked on a clean TF_REF
scripts/publish-image.sh     pushes the built image to ghcr.io (only on purpose)
zig/Dockerfile, zig/patches  the engine image; patches to the pinned engine (five by hand, then the port's 0007-... by scripts/port-patches.sh)
model/chat_template.jinja    the chat template rank 0 serves
tools/validate.py            an end-to-end check of a running server
tools/bench.py               prefill, decode, multi-stream and needle measurements as JSON, with `--compare A.json B.json`
tools/exact.py               records a server's replies (token hashes) on a fixed prompt set and compares two records: proves a new image equals the old one
tools/smoke.py               a quick correctness check: health, stop strings, streaming, a tool call and its result
tools/prompt_reuse.py        a conversation resumes from its kept prompt state (from the GLM-5.3-Flash recipe)
tools/toolcheck.py, needle.py, end_of_turn.py, anthropiccheck.py   single checks, from the GLM-5.3-Flash recipe
docs/zig-engine.md           how the pieces fit, and what is not verified
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
- **Mia's AI Lab** ([Mia-AiLab](https://huggingface.co/Mia-AiLab)): the EAQ v1 EXL3 2.9 bpw quantization served here,
  [`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
  (MIT), the earlier release
  [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) whose
  per-tensor widths it equals, and the chat template (`model/chat_template.jinja`, written for that earlier release).
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
  [`3ff3b2a`](https://github.com/bertholomus/TensorFold/commit/3ff3b2ac1d71a789cb930068c46d9abed025eefa), and its
  weight-file and token-map tools, in their branch
  [`deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4) at
  [`5940208`](https://github.com/bertholomus/TensorFold/commit/59402086381a42c74d53053443224eec2790b4c5); the kernel kit,
  the served settings and the two-node launch commands, in their recipe
  [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) at
  [`abe07c7`](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/commit/abe07c7d509a0199cc7fec15263be2aa52ac652f).
  Commits in that branch by [sayyidfareed](https://github.com/sayyidfareed) (RDMA GID and MTU),
  [Lucas Peters](https://github.com/lucas-peters) (indexer alignment; the kit's `_topk_sel` variant), lepetitprince716-prog
  (expert-run loading) and LeetJoe (image placeholder escaping).
- **[Coinupbtc](https://github.com/Coinupbtc)** (@Coinupbtc) authored `zig/patches/0005-prefill-interleave.patch`, in
  [pull request #5](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5) (commit [`27e02c6`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5/commits/27e02c605eae0997ed9161baaf583dd3ad431bb3)): decode rounds and cancels for the other live streams between the chunks of a long prompt pass.
- **Mia's AI Lab**: this recipe's scripts, adapted from
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (branch `zig-engine`), with code by webzone (both PCIe links of a port as twin rails), [Alexbob0](https://github.com/Alexbob0)
  (workers by host name; `tools/end_of_turn.py`) and john (each rank's rendezvous address); `tools/anthropiccheck.py` by
  Eduardo Florencio ([eduffd](https://github.com/eduffd)), and the Hermes Agent page by Steve Darlow
  ([@kerpopule](https://github.com/kerpopule)).
- **hashd1ve** ([hashd1ve](https://github.com/hashd1ve)): the change to `model/chat_template.jinja` that writes each piece of the
  prompt out as it is made instead of adding it to one growing string, which cut rank 0's memory growth with long
  conversations ([issue #1](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/1)).
- **Mia's AI Lab**: this recipe's own engine patches, `zig/patches/0001` and `0002` (changes to BertholomusAI's engine, written
  for this recipe, Apache-2.0): the prefill indexer split between the two ranks and a second AIO ring for Engram reads (a third, prompt-cache placement,
  was dropped when the pin moved); also `tools/exact.py` (token-hash replay check) and `scripts/memguard.sh`.
  The idea of splitting the indexer is drowzeys's, from the GLM-5.3-Flash recipe's design (no code of it was used).
  `0004` (`tool_choice` enforcement) is Mia's AI Lab's Zig call gate from its Flash Next CUDA engine, carried into the
  DeepSeek-V4.1 lanes; that gate is a port of the call gate in TensorFold's Python engine (`call_gate.py`), by the
  TensorFold contributors (Apache-2.0).
- **NVIDIA**: the DGX Spark and GB10 hardware, and the PyTorch container (`nvcr.io/nvidia/pytorch:26.07-py3`) the image is
  built on (CUDA, NCCL, cuBLAS; under NVIDIA's container license).

Full list, with commits: [`CREDITS.md`](CREDITS.md).
