---
name: Bug report
about: The recipe is not behaving as expected, is producing an error, or the replies are wrong.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this recipe! The issue template is adapted from the GLM-5.3-Flash recipe's (Mia's AI Lab).

     If you are looking for support, please check the README and scripts/config.sh first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     This recipe has not served a request yet (docs/measurements.md): a first report that rank 0 or rank 1 exits at
     start is useful, with both ranks' logs.

     If you have found a bug, then fill out the template below.
-->

---

## Environment

- Hardware: <!-- e.g. 2x DGX Spark (GB10, 128 GB unified memory each), one QSFP cable between the CX7 ports -->
- Memory available before launch, on both Sparks: <!-- `free -g`; other GPU workloads running? -->
- Image: <!-- `docker images | grep tensorfold-dsv41`, or the ghcr.io tag you pulled -->
- Engine: <!-- `docker exec dsv41-1017-exl3-tf tensorfold-native --version`, and TF_REF / INPUTS_REF / KIT_REF if you changed them -->
- `start.sh` invocation: <!-- e.g. `./start.sh`, or `PARALLEL=2 ./start.sh restart --max-tokens 16384` -->
- Changed settings: <!-- environment, scripts/local.sh or .env: CONTEXT, PARALLEL, ROUND_ROWS, DRAFTS, RDMA, GRAPHS, SIDE, PREFETCH, ENGRAM_AIO, WORKER_*, TENSORFOLD_* / TF_DS_* -->
- Link: <!-- the "Link:" and "rank N" lines start.sh prints in step 2 (RoCE devices, GID) -->

---

## Steps to Reproduce

1. Run `./start.sh` <!-- describe any overrides and what it printed up to the failure -->
2. ... <!-- describe steps to demonstrate the bug -->
3. ... <!-- for example "a tool call with an array parameter returns a string" -->

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

Add any other context here: a minimal request that reproduces bad output, JSON responses, `docker inspect` output, and so on.

<details>
<summary>Minimal reproduction sample</summary>

<!--
      If the bug is about model output or API behavior, attach a minimal reproducible request below.

      NOTE: the model thinks before it answers, in "reasoning_content" rather than "content". Budget enough
      max_tokens (~2,000 for anything non-trivial), or send "chat_template_kwargs": {"enable_thinking": false}: with a
      small budget the reply is often still in its reasoning and "content" comes back empty on a healthy server.

      To tell an engine bug from model behavior, send the same request with DRAFTS=0 (./start.sh restart) and a fixed
      "seed" or temperature 0: drafted and plain replies must be identical.
-->

```bash
curl -s http://<head-address>:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4.1-Flash-1017",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 2000
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the log output below, and say whether it came from `start.sh`, `scripts/prepare.sh`,
      `docker logs dsv41-1017-exl3-tf` (rank 0, on the head), `ssh <worker> docker logs dsv41-1017-exl3-tf` (rank 1),
      or a client. `stop.sh` saves both ranks' logs in ~/.cache/dsv41-tensorfold/logs.

      Common culprits worth checking before filing:
        * start.sh warns "only N GiB memory available" -> stop other GPU containers on that Spark first (`docker ps`).
        * "no RoCE device or RoCE v2 GID for the link" -> WORKER is reached over another network: set FABRIC_PEER to
          the worker's CX7 address.
        * `start.sh` refuses port 8888 -> something else listens there; set PORT.
        * prepare.sh: "does not see every file over NFS" -> the head's HF_CACHE is not exported to the worker (docs/requirements.md, NFS).
        * prepare.sh: the kit does not check out -> KIT_REF was changed, or the checkpoint's config is not the one the
          RoPE tables are made for.
        * rank 1 exits at start with a missing input -> its weight file, Engram's tables or the token map: prepare.sh
          step 5 and 7.
-->

```

```

</details>
