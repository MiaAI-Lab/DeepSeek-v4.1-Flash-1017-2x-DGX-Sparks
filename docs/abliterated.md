# Abliterated mode

Optional: `ABLIT=1` applies the matching Keys overlay by drowzeys on top of the default EAQ v1 pack.

`ABLIT` accepts **`0` or `1`** (default `0`). Set it in the environment, `scripts/local.sh` or `.env`.
`ABLIT=0` serves the stock checkpoint (`MODEL_ID`/`MODEL_REVISION`; the default is Mia EAQ v1 2.9 bpw).
`ABLIT=1` keeps the **EAQ v1** 2.9 bpw checkpoint and applies `exl3_wo_b_l10_35_mia29.safetensors` from
[`drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated`](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated);
it is refused when `EAQ_V1_1=1`, so leave `EAQ_V1_1=0` (the default) for it.
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
The stock performance and correctness results in [Status and measurements](measurements.md) do not validate the abliterated mode on this recipe's engine.
