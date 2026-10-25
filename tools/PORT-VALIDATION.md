# Port validation on the Sparks: tools, pass criteria, run order

The GPU-phase checks for the ported GLM features (spill tier 0088 + spill v2 blocks 0110 + chat continuation 0111,
KV sharding 0118, spill and KV sharding together, spill for prompts with pictures, the broken-pair policy) in the
native dsv41 engine. Lean by design: targeted suites for what changed, each
writing a JSON receipt, compared by strong token hash (`tensorfold.token_sha`) against a baseline recorded on the
**unchanged daily service** a few minutes earlier. No multi-hour reruns.

Every tool is stdlib Python 3, run as `python3 -I tools/<tool>.py` (`-I`: no site packages, no cwd imports), and
reads `API_URL` (default `http://127.0.0.1:8888`) and `API_KEY` / `API_KEY_FILE` like `tools/client.py`.

## The tools

| Tool | What it proves | Modes |
| --- | --- | --- |
| `exact.py` | same greedy outputs as the old image: 11 lengths, a 3-conversation interleave, repeat, resume, **eight concurrent == the same eight solo** (`--streams 4` for the old four) | `record [--like OLD]`, `compare OLD NEW` |
| `spill_check.py` | disk tier end to end, every disk hit proven by per-rank counters; `--checks pictures` and `--pictures` (prerestart / postrestart) for points with a picture | `record`, `run --phase main\|prerestart\|postrestart\|prekill\|postkill\|floor`, `drop` (server stopped), `estimate` |
| `kv_shard_check_e2e.py` | KV sharding on the engine's real counters: owners, mixed rounds, same-part burst, carry, twins, live move, cancel during transfer, rank 1 killed during a carry, idle accounting, window vs pool, over-window refusal, needles | `record`, `run --checks ... [--startup-log F] [--kill-cmd C]` |
| `pair_check.py` | the broken-pair policy: streams decoding, rank 1 killed or stopped by `--kill-cmd`, every stream ends, `/health` goes down (or 503), rank 0's container `exited 75` (`--exit-cmd`, `--exit-code`), `--expect-log CMD::PATTERN`; restarting the pair is the caller's | one run |
| `prefix_fanout.py` | shared system prompt at concurrency 1/2/4/8: duplicate prefill tokens counted (the prefix computed once) | `--out` (record), `--baseline` (run) |
| `gpu_validate.sh` | the whole runbook on gx10-a, `--dry-run` prints every command | `[--phases] [--configs]`, `kill-ranks`, `restore` |
| `checklib.py` | shared client, counters (`/health` + `/metrics` flattened to `prompt_cache.x`; the tier per rank as `spill.x.rankN` and over the ranks as `spill.x`; `kv_shard.x` and every part's rows as `kv_shard.partN.row`), receipts, per-rank file ops | library |
| `fake_server.py` | a scripted stand-in server in the engine's shapes (`prompt_cache.spill` an array of per-rank objects, `spill_*{rank="N"}`, the window only on `/tokenize`, presence from an in-memory index, the store's floor rule; real files, real timing, `--mutate FAULT`) | `--port P ...` |
| `selftest.py` | every tool against the fake: must PASS on a correct fake and FAIL on the fault it exists to catch | `[--only ...] [--keep]` |

Existing tools still used by the runbook: `smoke.py` (restore health), `toolcheck.py`.

## Before the Sparks: self-test here

```
python3 -I tools/selftest.py            # py_compile of every tool, exact, spill, kv, prefix, pair, runbook (~2 min)
bash -n tools/gpu_validate.sh
GPU_VALIDATE_HOST_OK=1 WORKER=gx10-b.invalid SESS=$PWD/.sess bash tools/gpu_validate.sh --dry-run
```

Result on 2026-10-20 (after the round-1 review fixes): `selftest: PASS (78 expectations, 0 unmet)`. Besides each
tool passing on a correct fake, it covers: checklib reading engine-shaped `/health` and `/metrics` text and every
TierCounts field per rank from the fake; planted faults the spill check catches (restore-wrong, no-disk, ignore-crc
on block and small-state files, ignore-presence for missing-rank1), the floor admitting F1 and a margin above F1's
write; for kv shard-wrong, a wrong `--expect-pool`, admit-over-window, and a missing pool as SKIP (FAIL with
`--strict`); for prefix duplicate prefill (a fake whose prefills overlap) and one fill (`--serial-prefill`, the native
admission), a warm server judged cold; for exact replies differing above
four concurrent streams and a record without `token_sha`; and the runbook's `restore` against stubbed
systemctl/docker/ssh/curl (a healthy daily left alone, an active oneshot with no containers restarted, a port
container stopped by its label, the daily's own container never touched by the port stop.sh, an early exit before the
stop removing only the hold, `kill-ranks --resolve`, `kill-ranks --rank 1 --signal STOP`).

Result on 2026-10-22 with spill-shard, spill-pictures and the pair checks: `selftest: PASS (115 expectations, 0
unmet)`. New in it: a picture point kept on both ranks (`pictures_kept`), restored from disk after the eviction and
after a restart, and a fake that never writes picture points caught (`--mutate no-pictures`); `pair_check` against a
fake that a kill ends as a broken pair ends rank 0 (every stream ends, `/health` goes, the container shows
`exited 75`, the 75 line in the log), a rank 0 that ends with 137 caught by the exit check alone, and a kill that
changes nothing caught; the dry run of every new configuration (for `pair-*`: `--exit-code 75`, `start.sh restart`,
the saved log). `scripts/port_test.sh` checks that `start.sh` hands a container it finds, running or exited, to
`stop.sh` before the launch, and that `stop.sh` saves each rank's log before its `docker rm`.

## Pass criteria

Every check reports PASS / FAIL / SKIP (SKIP = not exercised, with why). A tool exits 1 on any FAIL, or on any SKIP
with `--strict`. The receipt JSON holds each check's detail, counter deltas and the rows behind it.

**exact.py** (`compare` exits 0): every case's `token_sha` equals the baseline's; `record`'s own checks pass (repeat ==
first, resume suffix replayed, `conc_k == solo_k` for k = 1..8). A record without `token_sha` (text-hash fallback) is
refused by `compare` unless `--allow-weak`. A 4-stream record still serves as `--like` and as either side of compare.

**spill_check.py** (strong hashes vs the daily baseline for every reply):

| Phase / check | Pass |
| --- | --- |
| main `evict` | memory evicted by fillers (`prompt_cache.evicted`), the hit restores from disk: `disk_hits_committed +1`, `restored_tokens > 0`, reply == baseline |
| seeds | every item's reply == baseline and its point committed on **every** rank (`spill.writes_committed.rankN`), nothing pending |
| `corrupt-block`, `corrupt-small` | one byte flipped on rank 0: no disk hit, `crc_errors`/`size_errors` or `restores_failed` counted, reply == baseline (recomputed) |
| `truncate-block`, `truncate-small` | file cut in half on rank 1: as above |
| `remove-small` | rank 1's `.st` removed while running (presence comes from the store's in-memory index, so the read fails): `read_failed`/`restores_failed`, no disk hit, reply == baseline |
| `cancel` | X1 (500K tokens) restoring, the client leaves at the first sign of it (`restores_begun`, `restores_active`, blocks left): `restores_cancelled +1`, nothing active after, the same request again == baseline; SKIP if the restore finished first |
| `dual-restore` | two restores at once while a third stream decodes: both from disk, all three == baseline; the decoder's worst inter-token gap is in the receipt |
| `sizes` | restore TTFT per `SPILL_SIZES` size, each from disk and == baseline |
| `settled` | no restore active, nothing parked, no queued write |
| prerestart / postrestart | R1 and MM committed on every rank before a clean stop.sh; `drop` removes rank 1's manifest of MM while stopped; after start.sh every rank recovered points, R1 restores from disk == baseline; `missing-rank1`: MM not restored (presence refused, or rank 1 holds fewer points), == baseline; no temporaries |
| `pictures` | P1 (a 64x64 PNG before 16K tokens of text): its seed == baseline, committed, and `spill.pictures_kept +1` on every rank (FAIL when the server reports no `pictures_kept`); after the eviction its hit restores from disk == baseline |
| prerestart / postrestart `--pictures` | P2, a picture conversation, committed before the stop; after start.sh it restores from disk == baseline |
| prekill / postkill | K1 committed; both PIDs resolved; K2 (200K tokens) streamed and both ranks `kill -9`'d the moment its write is queued and pending; after start K1 restores from disk, K2 == baseline (disk or recomputed; SKIP of the in-flight case when nothing was pending at the kill), no temporaries left on either rank |
| floor | on an empty store, both nodes' free space equalised (a ballast beside SPILL_DIR), floor = free - margin with margin = half of F1's estimated write (F1 200K tokens: ~183 MB on DSV4.1, `spill_check.py estimate`): `no_space` on every rank, nothing committed, no new files, reply == baseline, memory still serves the hit; `floor: sizing` FAILs if the margin is not below the estimate |

**kv_shard_check_e2e.py** (counters as shard-lanes merged them: `/health kv_shard{...lanes.kv_parts.Stats,
per_part[{part, free, live_rows, kept_rows, twin_rows, hold_rows, live}]}`, `/metrics tensorfold:kv_shard_*` and
`kv_shard_part_*{part="N"}`; per-part gauges sampled every `--sample-ms` while a check runs):

| Check | Pass |
| --- | --- |
| `limits: window` | `/tokenize` max_model_len (the only endpoint the engine reports it on), agreeing with the start-up line; with `KV_SHARD=1 POOL=2101248` at least 1,048,576 (shard.plan: min(CONTEXT, part - rows - 2); the unsharded daily pool reports 1,048,558) |
| `limits: pool` | `/health kv_shard` parts x part_rows, agreeing with rank 0's start-up line `dsv41: pool P positions, MODE PART a part; ...` (saved from `docker logs` as `<config>/startup-pool.txt`) and its part: window <= pool, pool == 2,101,248 / 524,288; SKIP when nothing reports it |
| `limits: over-window` | a prompt of window + 1024 tokens is refused with HTTP 400 context_length while the pool is about twice the window |
| `owners` | two long decodes at once: == baseline; both parts held a live stream (sampled) and `fetch_rounds + owner_rounds` grew |
| `mixed` | eight at once, varied sizes: == baseline; samples with both parts live, sharded rounds grew |
| `burst` | four at once on one 24K prompt (same-part copies): == baseline, `transfers_failed` unchanged |
| `carry` | kept conversation + fillers on a small pool: every hit == baseline, `carries +1`; `moved_bytes / moved_tokens` in the receipt (720 B expected) |
| `twins` | a long fresh prompt: == baseline, `twins_made +1`, `prompt_fetches` unchanged, every part's `twin_rows` 0 after |
| `move` | two decodes, then a ~256K prompt that fits in no part beside one: all three == baseline, `moves +1` (SKIP when none) |
| `cancel-transfer` | a carry whose client leaves while it runs (`carries +1` or `hold_rows` seen): healthy, every part's `hold_rows` and `live` back to 0, again == baseline |
| `kill-rank1` | rank 1 `kill -9`'d during a carry: the request in flight ends within `KILL_DEADLINE` (90 s; xfer_ns is 30 s), no hang |
| `idle` | nothing running: 2 parts, no `live` / `live_rows` / `twin_rows` / `hold_rows`, free + kept_rows <= part_rows, `transfers_failed` did not grow |
| `needle` | three hidden values at 10/50/90% of a 524,288-token prompt and of the largest admitted prompt (`max`): all found |

Every numeric `kv_shard` field lands in the receipt; evidence it cannot read is a SKIP, not a PASS.

**pair_check.py** (a broken pair, end to end; times in the receipt are seconds after the kill):

| Check | Pass |
| --- | --- |
| `decoding` | every stream (`--streams`, 8) had its first token and none had ended at the kill (the stall monitor counts only a busy pair) |
| `kill` | `kill-ranks --rank 1 [--signal STOP]` exits 0 |
| `streams end` | every stream ends (failed, cut, or finished inside the 5 s drain) within `--end-deadline`: `PAIR_KILL_DEADLINE` 60 s for kill -9, `PAIR_STOP_DEADLINE` 200 s for SIGSTOP (stall 120 s + abort 15 s + grace) |
| `down` | `/health` stops answering ok (gone, or 503 `stalled` / `broken`) within the same deadline |
| `exited` | rank 0's container is `exited 75` (`docker inspect -f '{{.State.Status}} {{.State.ExitCode}}'`) within the same deadline; another code (a memguard kill is 137) FAILs |
| `log` | `docker logs` of the exited rank 0 has the engine's `exiting with status 75` |

Then the suite restarts the pair itself (`./start.sh restart`, as a supervisor or a person would): `<config>-saved-log`
passes when the rank-0 log that restart's `stop.sh` saved into `PORT_LOG_DIR` has `exiting with status 75` (the
evidence survives the restart), and `exact` runs on the restarted pair.

**prefix_fanout.py**: every request answered and == baseline; level 1 (sequential) reuses the 2048-token cut. The runbook
runs it in the `off` configuration with `PREFIX_FLAGS=--expect-one-fill` (the default): every level, up to 8
concurrent, must compute the shared prefix once, as the native engine's one-prompt-pass-at-a-time admission does
(`PREFIX_FLAGS=` empty only measures; on the pair, 8-way: 0 duplicate prefill tokens). `--strict-cold` fails a level
that was already warm.

## Run order on the Sparks (gx10-a, rank 0)

Nothing in the runbook writes under `/tmp`; everything lives under `~/sess/dsv41-port/` (checkout, state, spill,
receipts). Run it with `bash` (scp drops exec bits):

```
# 0. the recipe branch (it carries the tools since ec937bb) somewhere gx10-a can fetch it. On the Mac:
#      git -C ~/GitHub/deepseek1017-recipe-port bundle create port.bundle spill-kvshard
#      scp port.bundle gx10-a:sess/dsv41-port/
#    on gx10-a, a first checkout to run the runbook from (its checkout phase then refetches PORT_REF into PORT_DIR,
#    whose tools/ every suite runs; TOOLS_REF=<ref> uses another ref's tools in TOOLS_DIR instead):
#      git clone -q -b spill-kvshard ~/sess/dsv41-port/port.bundle ~/sess/dsv41-port/runbook
export PORT_REPO=~/sess/dsv41-port/port.bundle     # PORT_REF=spill-kvshard by default
# 1. read every command first
bash ~/sess/dsv41-port/runbook/tools/gpu_validate.sh --dry-run
# 2. the run (about an hour of serving outage per four configurations; the daily service is restored at the end, on
#    any error, and on ^C)
bash ~/sess/dsv41-port/runbook/tools/gpu_validate.sh
```

Phases, in order: `preflight` (host gx10-a, `sudo -n` on both nodes, ssh to the worker, SPILL_DIR under SESS) →
`checkout` (PORT_DIR at PORT_REF, the daily `scripts/local.sh` copied, every configuration's setting names checked
against the port's start.sh / config.sh / port.sh; PORT_DIR's `tools/` every suite runs, or TOOLS_DIR at TOOLS_REF when set) → `hold` (`touch ~/.deepseek1017-hold`: the watchdog stands down;
from here any exit restores) → `baseline` (on the running daily service: exact, spill, kv and fanout records) →
`stop` (`sudo systemctl stop deepseek1017.service`, no container left on either Spark) → `build`
(`scripts/prepare.sh` in PORT_DIR with its own STATE_DIR, so the daily prepared marker is untouched; it shares
DATA_DIR and WORKER_CACHE_DIR with the daily, so the daily's weight files on both nodes and its served folder are
listed before and after and the run stops, and restores, if they changed) → `dropcache` (both nodes; again before
every configuration) → `suites` → `restore`.

`restore` (also on any exit after `hold`, and `gpu_validate.sh restore`): the port's stop.sh runs only when this run
started a port server or a container carrying the port's `tf.recipe.port` label is up (the daily's containers have
the same name and no label); if the daily service was never stopped, only the hold goes; else, when the oneshot unit
is still active but nothing serves, `systemctl restart`, otherwise `systemctl start`; then `/health` ok, smoke, an
exact record compared with the morning's baseline, the hold removed. A port container still up after its stop.sh
leaves the hold in place rather than starting the service over it.

Configurations (`--configs`; default `off,spill,spill-mismatch,shard,shard-copies,shard-pressure,shard-check,
shard-mismatch,shard-pool-mismatch,shard-kill,spill-shard,spill-pictures,spill-pictures-mismatch,pair-kill,pair-stop`;
opt-in: `shard-p2,shard-p4,shard-nodraft,shard-perf,shard-perf-fetch,spill-shard-pressure`), each started by the
port's start.sh with these settings (override with `CFG_<name>`). Every configuration names the switch it does not
test as off, so it runs the same where a fork turns spill or KV sharding on by default (Robert's main does): `off` is
`SPILL_GIB=0 KV_SHARD=0`, `spill` adds `KV_SHARD=0`, and every `shard*` one adds `SPILL_GIB=0` (shown below without
them); `spill-shard` and `pair-stop` run both, the daily configuration, `pair-kill` neither. S1..S6, K1..K9,
SS1..SS10, SP1..SP3 and R1..R6 are the numbered "GPU validation" items of `docs/port/spill-lanes.md`,
`shard-lanes.md`, `spill-shard.md`, `spill-pictures.md` and `robustness.md` (engine repo); the script's header has
the map. A suite that ends a rank restarts the pair itself.

| Configuration | Settings | Suites | Items |
| --- | --- | --- | --- |
| `off` | `SPILL_GIB=0 KV_SHARD=0` | startup (`a broken rank pair ends this server with status 75`); exact, smoke, toolcheck, prefix_fanout (`--strict-cold --expect-one-fill`); start-up lines saved (BENCH=1: bench) | K1, R1 |
| `spill` | `SPILL_DIR=WORKER_SPILL_DIR=~/sess/dsv41-port/spill SPILL_GIB=200 SPILL_FLOOR_GIB=50` (wiped first) | startup (`the ranks agreed`, `staging 128 MiB`); exact; spill main; prerestart → stop → `drop` → start → postrestart; prekill → start → postkill; wipe → ballast → floor | S1-S6 |
| `spill-mismatch` | `spill` + `RANK1_EXTRA_ENV=TF_DS_SPILL_MIN_TOKENS=4096` | the start must fail, a rank's log with the engine's `rank 1's configuration differs from rank 0's in features` (start.sh's own lines do not count) | S1 |
| `shard` | `KV_SHARD=1 POOL=2101248 KV_SHARD_CARRY=0 KV_SHARD_TWINS=0` | exact; kv limits, owners, mixed, needle, idle | K2 |
| `shard-p2`, `shard-p4` (opt-in) | `shard` + `PARALLEL=2` / `4` | exact; kv owners, mixed, idle | K2 |
| `shard-nodraft` (opt-in) | `shard` + `DRAFTS=0` | exact (against the drafted baseline) | K2 |
| `shard-copies` | `KV_SHARD=1 POOL=2101248` (carry, twins on) | exact; kv burst, idle | K3 |
| `shard-pressure` | `KV_SHARD=1 POOL=524288 CONTEXT=260096`, carry and twins on | kv limits, carry, twins, move, cancel-transfer, idle | K4-K6 |
| `shard-check` | `shard-pressure` + `KV_SHARD_CHECK=1` | kv carry, idle; exact | K7 |
| `shard-mismatch` | `KV_SHARD=1 POOL=2101248 RANK1_EXTRA_ENV=TF_DS_KV_SHARD_CARRY=0` | the start must fail, a rank's log with `differs from rank 0's in features`, `KvShardMismatch` or `ShardLayoutMismatch` (or their explanations) | K7 |
| `shard-pool-mismatch` | `KV_SHARD=1 POOL=2101248 RANK1_EXTRA_ENV=TF_DS_WINDOW=1044480` (rank 1 reads 0008's `TF_DS_WINDOW`; every rank's pool and window are compared at load) | the start must fail, a rank's log with `KvShardMismatch` / `ShardLayoutMismatch` (or their explanations) | K7 |
| `shard-kill` | `shard-pressure` | kv carry, kill-rank1 (`kill-ranks --rank 1`); restart; exact, kv idle | K7 |
| `shard-perf`, `shard-perf-fetch` (opt-in) | `KV_SHARD=1 POOL=2101248`, twins on / off | bench (prefill, decode, multi at 1/2/4/8), compared with the daily's | K8, K9 |
| `spill-shard` | `spill` + `KV_SHARD=1 POOL=2101248` (the daily configuration) | startup (`the ranks agreed`); exact; kv limits, owners, mixed, idle; spill main (all its checks); each rank's store size (`stores.txt`, `du`: SS3's 890 / 170 B a token); prerestart → stop → `drop` → start → postrestart | SS1-SS5, SS9 |
| `spill-shard-pressure` (opt-in) | `spill` + the `shard-pressure` settings | kv carry, twins, move, idle; spill evict, dual-restore, settled; store sizes (`kv_shard_staged`, `forced_staging`, `writes_failed` in the snapshots) | SS6-SS8 |
| `spill-pictures` | `spill` | startup; spill `pictures`, settled; prerestart `--pictures` → stop → `drop` → start → postrestart `--pictures` | SP1 |
| `spill-pictures-mismatch` | `spill` + `RANK1_EXTRA_ENV=TF_DS_IMAGE_CACHE=0` | the start must fail, a rank's log with `differs from rank 0's in features` | SP3 |
| `pair-kill` | as `off` | pair_check with rank 1 `kill -9`'d; `./start.sh restart`; the saved rank-0 log; exact | R2 |
| `pair-stop` | as `spill-shard` (wiped first) | the same with rank 1 SIGSTOPped (`kill-ranks --rank 1 --signal STOP`) | R3 |

`KV_SHARD=1` with `SPILL_GIB>0` (the daily's settings) is `spill-shard`'s configuration. Not covered at the server level: S2's tf-dsv41-lanes `drop_memory` runs, K6's per-layer `ch.o` equality (kit
tooling), K8's per-transfer `Dev.wait` cost on its own; SS4's `drop_memory` restores and the part a restore lands in,
SS10's packing costs; SP2's first logits of a cut straddling a picture (kit tooling); SP3's both-ranks
`TF_DS_IMAGE_CACHE=0` (no picture point written) beyond the mismatch; R4 (a stop of both ranks: exit 0; every
configuration ends with one, not judged), R5 (`TF_DS_PAIR_BROKEN=degrade`: no start-up line), R6 (the fence and
`restores_deferred`: counters and logs of the spill suites).

The spill phases pass their state (files, the drop, the kill) in `receipts/<stamp>/spill/state.json`. Port runs save
rank 0's logs in their own `PORT_LOG_DIR` (`~/sess/dsv41-port/logs`) with `LOG_KEEP=1000`, so they never rotate the
daily's logs out (rank 1's go to the worker's fixed `~/.cache/dsv41-tensorfold/logs`); each configuration's
receipts get only the logs written since it started.

Partial runs: `--phases suites --configs spill` (with the daily service already stopped and the hold set by hand),
`--no-restore` to keep the port server's slot for a rerun, `bash tools/gpu_validate.sh restore` to put the daily
service back and remove the hold. `bash tools/gpu_validate.sh kill-ranks [--resolve F | --pids F] [--rank 1]` kills both ranks'
engine processes by the containers' PIDs (`docker inspect .State.Pid`, `sudo -n kill -9`), never by `pkill -f`
pattern; `--resolve` only writes them, `--pids` kills those, so the kill is one quick step; `--rank 1` kills rank 1's
alone (shard-kill, pair-kill); `--rank 1 --signal STOP` stops it instead (pair-stop; the restart's stop.sh then
kills the stopped container).

Receipts: `~/sess/dsv41-port/receipts/<stamp>/`: `summary.json` (every suite's exit code, `failed` list), one
`<suite>.log` and receipt JSON per suite, health/metrics snapshots and rank 0's log per configuration,
`port-commit.txt`. Copy that folder into the deployment's `validation-receipts/`.

## Known limits

- **floor:** the engine refuses a write only when evicting every stored state could not make room
  (`floor + need + queued > free + pending_free`, store.zig `floorShort`), so the suite wipes the store, equalises
  both nodes' free space with a `fallocate` ballast beside SPILL_DIR on the roomier one (removed after the suite and
  on any exit), and sets `SPILL_FLOOR_GIB` to that free space less `FLOOR_MARGIN_BYTES` (default half of F1's
  estimated write). Free space that drops before the write only makes the refusal surer; free space that grows by more
  than need - margin (~90 MB) before it would admit F1. `floor-free.txt` holds the numbers.
- **counters:** with `API_KEY=1` the engine's `/health` answers only `{"status": "ok"}`; checklib then reads every
  counter from `/metrics` (with the key), labelled per rank (`spill_*{rank}`) and per part (`kv_shard_part_*{part}`).
  The per-part sampler falls back to `/metrics` the same way.
- **fake vs engine:** the fake reproduces the counter names, shapes (per-rank array, labelled metrics), file layout,
  in-memory presence and floor rule of port/cache-tier and spill-store, not the engine's timing or eviction order
  (its floor refuses without evicting first); its KV parts are a simplified planner (placement by fewer live streams,
  overcommitted parts) that serves the counters' names and shapes, not the engine's placement.
- **pair exit:** rank 0's exit code reaches `docker inspect` only if the image's entrypoint passes it through; an
  `exited` FAIL with rank 0's log showing `exiting with status 75` points there. The suite's restart after it reloads
  both ranks (minutes; `start.sh`'s own wait, then `/health` for up to 2400 s).
- **baseline drift:** the baseline is recorded on the live daily service at the start of the run; the restore phase
  re-records exact.py after the daily service is back and compares, so a daily image that changed under the run shows.
