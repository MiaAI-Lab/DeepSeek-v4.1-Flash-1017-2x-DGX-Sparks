# Deploying the port fork on gx10-a

How the fork checkout that serves Robert's traffic is brought to a newer `spill-kvshard` (this branch, with the engine
exported as `zig/patches/0007-spill-kvshard.patch`), and how it is rolled back. GPU work and anything on the Sparks is the
coordinator's: these are the steps it runs. Nothing here changes the stock checkout.

## What serves now

| | |
| --- | --- |
| unit | `deepseek1017.service` (oneshot, `RemainAfterExit=yes`, `TimeoutStartSec=7200`), enabled; `deepseek1017-watchdog.timer` restarts it after 3 failed `/health` checks, never while `~/.deepseek1017-hold` exists |
| drop-in | `/etc/systemd/system/deepseek1017.service.d/fork.conf`: `WorkingDirectory`, `ExecStart` (`start.sh`), `ExecStartPost` (memguard from the fork dir) and `ExecStop` (`stop.sh`) point at **`~/DeepSeek-port`** |
| fork checkout | `~/DeepSeek-port` on gx10-a: the round-1 tree tested in `~/sess/port-r1/recipe` (the stock tree with its two local fixes, `zig/patches/0007-glm-port-r1.patch` = engine `b7dd06b`, the daily `scripts/local.sh`) |
| image | `tensorfold-dsv41:exl3-tp2-99f490fc5c18` on both Sparks |
| stock checkout | `~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`, untouched (the unit's own paths) |

`spill-kvshard` already carries both local fixes of the stock tree (`compiler-mirror.patch`: `ZIG_MIRRORS`;
`worker-helper-image.patch`: the worker's helper containers use `$IMAGE`), so the fork needs no working-tree edits
any more: only `scripts/local.sh`, which git ignores.

## Update: a new fork from this branch

Make the new tree beside the old one and swap the two folders, so the drop-in's path never changes and the old tree
stays whole for a rollback.

1. **On the Mac (no Spark involved).** Export the engine and commit it on `spill-kvshard`, then bundle the branch:

   ```
   cd ~/GitHub/deepseek1017-recipe-port
   scripts/port-patches.sh --dry-run && scripts/port-patches.sh     # zig/patches/0007-spill-kvshard.patch from port/integration
   bash scripts/port_test.sh && python3 -I tools/selftest.py      # the golden start.sh test skips here: step 4
   git add zig/patches && git commit -m "zig/patches: 0007-spill-kvshard.patch = port/integration <hash>"
   git bundle create ~/port-recipe.bundle spill-kvshard
   scp ~/port-recipe.bundle gx10-a:
   ```

   `head -1 zig/patches/0007-spill-kvshard.patch` names the engine branch and commit it was exported from.

2. **On gx10-a: hold, record what serves.**

   ```
   touch ~/.deepseek1017-hold                                         # the watchdog leaves the stack alone
   d=~/dsv41-rollback/fork-$(date +%Y%m%d-%H%M); mkdir -p $d
   docker inspect -f '{{.Config.Image}}' dsv41-1017-exl3-tf | tee $d/image
   ssh <worker> docker inspect -f '{{.Config.Image}}' dsv41-1017-exl3-tf      # the same tag
   docker tag "$(cat $d/image)" tensorfold-dsv41:fork-rollback
   ssh <worker> docker tag "$(cat $d/image)" tensorfold-dsv41:fork-rollback   # no prune can take it now
   systemctl cat deepseek1017.service > $d/unit.txt
   ```

3. **The new tree**, a clean checkout of the bundle, with the fork's own settings:

   ```
   git clone --quiet --branch spill-kvshard ~/port-recipe.bundle ~/DeepSeek-port.next
   cp -p ~/DeepSeek-port/scripts/local.sh ~/DeepSeek-port.next/scripts/local.sh
   [ -f ~/DeepSeek-port/.env ] && cp -p ~/DeepSeek-port/.env ~/DeepSeek-port.next/.env
   ls ~/DeepSeek-port.next/zig/patches/       # 0001 0002 0004 0005 0006 0007-spill-kvshard.patch: never also an r1 patch
   ```

   Only one engine export may be in `zig/patches` (the round-1 `0007-glm-port-r1.patch` and the new
   `0007-spill-kvshard.patch` touch the same files and would not apply together). The daily config
   (`SPILL_GIB=64 KV_SHARD=1`, `POOL=2101248` derived from `CONTEXT=1048576`; spill and KV sharding
   together, `PORT_SHARD_SPILL_READY=1` in `scripts/port.sh`) is now the recipe's **default**: the three lines in the
   fork's `scripts/local.sh` may stay (the same values; an explicit `SPILL_GIB=64` keeps a bad spill folder a refusal
   rather than a warning and a start without the tier) or go. To start with a feature off, set it there:
   `SPILL_GIB=0`, `KV_SHARD=0`; both for the recipe before the port. Each Spark's `SPILL_DIR` / `WORKER_SPILL_DIR`
   must be on its own NVMe (docs/PORT-RECIPE.md).

4. **Test the new tree on gx10-a, then look before switching.** The Mac has only bash 3.2, so `scripts/port_test.sh`
   skips its start.sh DRY_RUN golden test there: that test, the only one comparing start.sh's full docker commands
   (all off, KV shard with and without an explicit pool, the defaults, a failing default spill folder against an
   explicit one, spill on) with the stock recipe (`e65785c`, in the bundle's history), runs here,
   with bash 5 and on Linux, on a disk folder (never `/tmp` on a Spark):

   ```
   cd ~/DeepSeek-port.next && mkdir -p ~/sess/tmp
   f=~/sess/port_test-$(date +%Y%m%d-%H%M).txt
   TMPDIR=~/sess/tmp bash scripts/port_test.sh | tee $f | tail -3
   grep -cE '^ok   golden: ' $f; grep -E '^(FAIL|skip golden|skip start.sh)' $f    # 7, and nothing
   ```

   **Do not swap the folders unless it ends in `N of N checks passed` with all seven golden cases `ok`** (a `skip
   golden` line or a `FAIL` stops the deploy; record the result in the deploy notes). Then
   `DRY_RUN=1 ./start.sh restart` prints both ranks' docker
   commands and changes nothing (the image tag in them is the new one: the patch hash changed). With spill on, check
   the identity variables (`TF_DS_ENGINE_ID`, `TF_DS_RECIPE_PATCHES`, `TF_DS_MODEL_REVISION`, `TF_DS_ENGRAM_REVISION`;
   `TF_DS_RANK_CACHE_SHA` appears once its digest is cached, after the first real start). A real start with spill on
   refuses (before stopping anything) when either rank's weight digest cannot be had: the digest names the store, so
   a rank without it would open an empty one. `SPILL_WEIGHTS_SHA=0` runs without digests on purpose, under another
   store name too.

5. **Swap and restart through the unit** (it keeps owning the server):

   ```
   mv ~/DeepSeek-port ~/DeepSeek-port.prev-$(date +%Y%m%d-%H%M) && mv ~/DeepSeek-port.next ~/DeepSeek-port
   sudo systemctl restart deepseek1017.service       # ./stop.sh, then start.sh: prepare.sh builds the new image
   journalctl -u deepseek1017.service -f              # ~6 min build + docker save | load to the worker, then the load
   ```

   The build and the copy to the worker leave pages parked on the per-CPU free lists; `start.sh` now counts them as
   the engine does (`scripts/memavail.sh`) and waits up to `MIN_AVAIL_WAIT_S` (120 s) before refusing, so the round-1
   restore failure (110.3 < 112 GiB right after the load) does not recur. A first start with spill on also hashes
   each rank's ~106 GB weight file once (about a minute a Spark; cached beside the file afterwards).

6. **Check, then release the hold:**

   ```
   cd ~/DeepSeek-port
   python3 -I tools/smoke.py && python3 -I tools/exact.py record --like <baseline.json> --out new.json && \
     python3 -I tools/exact.py compare <baseline.json> new.json
   curl -s localhost:8888/health | python3 -m json.tool | grep -A3 -E 'kv_shard|spill'
   docker logs dsv41-1017-exl3-tf 2>&1 | grep -E 'dsv41: pool|dsv41 spill|CUDA budget'
   journalctl -u deepseek1017.service -n 300 | grep 'Pair restart on'      # the pair-restart guard (below) runs
   tail -2 ~/.cache/dsv41-tensorfold/logs/pairwatch.log
   rm ~/.deepseek1017-hold
   ```

   For the full GPU runbook (every configuration, receipts, restore) use `tools/gpu_validate.sh` instead
   (`tools/PORT-VALIDATION.md`); it works in its own checkout under `~/sess` and never touches `~/DeepSeek-port`.

## A broken pair restarts itself (status 75)

With spill or KV sharding on (the daily config has both), the engine's broken-pair policy defaults to `exit`. When
rank 1 dies or hangs, or a spill-tier or KV-transfer exchange breaks, rank 0 ends with status 75 (tensorfold
`docs/port/robustness.md`). The unit is a oneshot with detached containers, so systemd never sees that exit. On its
own, only `deepseek1017-watchdog.timer` would act, after 3 failed `/health` checks 2 minutes apart: 4 to 6 minutes of
outage before the reload even starts.

This branch closes that gap in the recipe itself: `scripts/pairwatch.sh` (docs/PORT-RECIPE.md "When the rank pair
breaks") is started on rank 0 with the memory guard. When rank 0's container is `exited 75`, it runs
`./start.sh restart` within ~2 s. The server answers again once the load finishes. The watchdog backs off while that
`start.sh` runs (its `pgrep`), so the two never restart at once.

**What to apply on gx10-a:**

1. **The unit, its drop-in, the watchdog and its timer stay as they are.** The guard starts from `start.sh` (ExecStart)
   and again from the drop-in's ExecStartPost, which calls `memguard_start` from the fork dir. It runs as `robert`
   inside the unit's cgroup, so `systemctl stop` ends it with everything else. **Do not** also add the watchdog fast
   path that robustness.md sketches ("For the recipe", option 1). It would restart through `systemctl restart` while
   the guard restarts through `start.sh restart`, and the start lock would fail one of the two. Option 2 (a
   `Type=simple` unit with `FOREGROUND=1`, `Restart=on-failure`, `RestartForceExitStatus=75`) is also unneeded. With
   `FOREGROUND=1` the guard stands down by design, and the watchdog's `pgrep ... start.sh` would then always match and
   never act, so that route would need a watchdog change as well.
2. **One line in `~/DeepSeek-port/scripts/local.sh`**, so the hold also stops pair restarts during manual work:

   ```
   PAIR_RESTART_HOLD=$HOME/.deepseek1017-hold
   ```

   (`tools/gpu_validate.sh` clears it for its `pair-kill` / `pair-stop` configurations; every other configuration
   runs with `PAIR_RESTART=0`.)
3. **Check once, on the first deploy of this tree**, that the guard runs (step 6's two extra lines). Then check that a
   real break restarts the pair: `tools/gpu_validate.sh --configs pair-kill` (tools/PORT-VALIDATION.md). That run also
   confirms the container's exit code is the engine's 75.

`PAIR_RESTART=0` in `scripts/local.sh` turns the guard off. A broken pair then waits for the watchdog's 3 strikes
again. `PAIR_RESTART_MAX` (6 an hour) bounds a pair that keeps breaking, and past it the guard logs and ends. Until
this tree serves, the running fork has no guard: a broken pair is the watchdog's, 4 to 6 minutes.

## Rollback

Hold first (`touch ~/.deepseek1017-hold`), remove it at the end.

- **A feature off, same image:** set it off in `~/DeepSeek-port/scripts/local.sh` (`SPILL_GIB=0`, `KV_SHARD=0`,
  `POOL=`), `sudo systemctl restart deepseek1017.service`. Everything is **on** by default now, so leaving a line out
  turns it back on; with `SPILL_GIB=0 KV_SHARD=0` this is the port image with the port unused.
- **The previous fork, exactly:** swap the folders back and restart. Its image is still on both Sparks (and tagged
  `fork-rollback`), so `prepare.sh` finds it and rebuilds nothing:

  ```
  mv ~/DeepSeek-port ~/DeepSeek-port.bad-$(date +%Y%m%d-%H%M) && mv ~/DeepSeek-port.prev-<stamp> ~/DeepSeek-port
  sudo systemctl restart deepseek1017.service
  ```

  `source scripts/config.sh; echo $IMAGE` in the restored tree must print the tag in `$d/image`.
- **The previous image from the new tree** (quickest, no folder swap): in `scripts/local.sh` set every port setting
  off explicitly (`SPILL_GIB=0 KV_SHARD=0`: the defaults are on, and an image without the port refuses the spill flags)
  plus `IMAGE=<the tag in $d/image>` and `PREPARE=0`, restart the unit. `PREPARE=0` keeps `prepare.sh` from
  rebuilding under that tag; never combine an old `IMAGE` with `PREPARE=1` or `auto`. Take both lines out again after.
- **Back to the stock server:** `sudo rm /etc/systemd/system/deepseek1017.service.d/fork.conf && sudo systemctl
  daemon-reload && sudo systemctl restart deepseek1017.service`. The stock checkout and its image were never changed.
- **Spilled state** stays in each Spark's spill folder (`<SPILL_DIR>/native-spill-<compat>/`), owned by you after a
  stop; a new image or checkpoint gets a new `<compat>` folder and never reads the old one. Reclaim it with the
  server stopped: `rm -rf <SPILL_DIR>/native-spill-*` on each Spark.

Old trees (`~/DeepSeek-port.prev-*`) and the `fork-rollback` tag can go once the new fork has served for a while
(`docker image rm tensorfold-dsv41:fork-rollback` removes the tag only while another tag names the same image).
