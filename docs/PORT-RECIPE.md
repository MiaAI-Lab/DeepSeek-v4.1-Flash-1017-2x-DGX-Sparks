# The GLM port in the recipe: settings, engine mapping, deploy and rollback

This recipe carries the port of Robert Wojciechowski's ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) GLM-5.3-Flash patches into
the native DeepSeek-V4.1 engine: the spill tier (`0088`, PR #78) and spill v2 blocks (`0110`, PR #124, on
[BadAd84](https://github.com/BadAd84)'s block-tier design), chat continuation (`0111`, PR #128), KV sharding with prompt
borrowing and the ring exchange (`0118`, PR #152) and the parked-page startup budget (`0119`, PR #149). The engine side
lives on the port's engine branch (`port/series-pair-370daf0`, one commit a patch, over `baseline-370daf0` = `TF_REF`
plus this recipe's hand-written patches `0001`-`0010`) and reaches the image as `zig/patches/0011-...` to `0022-...`,
one patch per feature (`zig/patches/README.md`). This page covers the recipe side: one image and one `scripts/local.sh` turn the features
on for **both** Sparks.

**Status.** Validated on the pair before the rebase onto `0008`/`0010` (replies byte-identical to the stock engine in
every configuration; disk restores after eviction, a restart and damaged files). The rebased build has passed the
host tests; its GPU checks are listed in `tools/PORT-VALIDATION.md`. Disk numbers on this page are calculated from
the engine's layout unless marked measured.

**Everything is off by default** (`PORT_SPILL_DEFAULT_GIB=0`, `PORT_SHARD_DEFAULT=0` in `scripts/port.sh`; `POOL`
empty): the recipe and the engine run as before the port, but for one change that acts only after a failure: a broken
rank pair ends rank 0 with status 75 instead of hanging it ([below](#when-the-rank-pair-breaks-status-75)). Turn the
features on in `scripts/local.sh`; `SPILL_GIB=64 KV_SHARD=1` (with `POOL` empty, derived from `CONTEXT`: 2,101,248 at
`CONTEXT=1048576` and 16-row rounds) is what the pair was validated with.

| To turn on | Set | Then |
| --- | --- | --- |
| spill v2 | `SPILL_GIB=64` | the spill flags, variables, identity and each Spark's `/spill` mount |
| KV sharding | `KV_SHARD=1` | the same `POOL` cut into one part a Spark; empty, it is derived (two whole-window parts) |
| a larger pool | `POOL=<positions>` | `zig/patches/0008`: more kept prompts stay resident; sharded, each Spark holds only its part's compressed KV |
| everything off | nothing (or `SPILL_GIB=0 KV_SHARD=0`) | the recipe before the port: no engine flag, variable or mount, rank 1's `--pool` is `CONTEXT`, and the only difference in the docker commands is a `tf.recipe.port=off` label on both containers (used to check that the two ranks agree) |

**Set explicitly, a setting that cannot run refuses the start**, before anything is stopped:
- *Spill:* it needs a usable local folder on **both** Sparks (`SPILL_DIR` here, `WORKER_SPILL_DIR` on the worker; the
  checks are below): missing and cannot be created, not writable, root-owned, on NFS or another network or memory file
  system, inside the NFS export or a model folder, under an archive folder, below `SPILL_FLOOR_GIB` free, an unusable
  path, or a worker probe that does not answer all refuse.
- *KV sharding:* two Sparks only (`TP=2`), and a `POOL` it can cut into two whole-window parts (`POOL=1048576`
  refuses, naming the pool it needs).
- Typed values are never "defaults": an invalid `SPILL_QUOTA_GIB`, `SPILL_MIN_TOKENS`, `KV_SHARD_CHECK` and so on
  refuse the start.

**A fork that turns a default on degrades instead, for that default only.** With `PORT_SPILL_DEFAULT_GIB` above 0, a
spill folder that fails on either Spark makes `start.sh` print `SPILL TIER OFF for this start: <reason>` and start the
pair with the tier off on both ranks (no flag, variable, mount, identity or spill label, `tf.recipe.port=...,spill=0`);
with `PORT_SHARD_DEFAULT=1`, a `TP` other than 2 or an explicit `POOL` sharding cannot use prints
`KV SHARDING OFF for this start` and starts unsharded. The same settings given explicitly still refuse. The spill
folders are checked first, before the flags, the variables and the `tf.recipe.port` label are made, so a degraded
start is labelled as what it runs, and `./start.sh` on a running pair compares against the same decision.

## Settings

All of them in `scripts/port.sh` (sourced by `scripts/config.sh`), set from the environment, `scripts/local.sh` or
`.env` like every other setting. **The mapping to the engine is the function `port_map` in `scripts/port.sh` and
nothing else** (and `port_ident_map` for the identity): when the engine renames a switch, only that function and the table below change.

| Setting | Default | What it does | Rank 0 (`tensorfold-native serve`) | Rank 1 (`tf-dsv41-lanes`) |
| --- | --- | --- | --- | --- |
| `SPILL_GIB` | `0` (off; `64` validated) | spill v2: kept prompt states written to each Spark's own disk when kept, up to this many GiB a Spark; read back instead of a new prefill, also after a restart or a crash | `--spill-gib N`, `--spill-dir /spill` | env `TF_DS_SPILL_GIB=N`, `TF_DS_SPILL_DIR=/spill` |
| `SPILL_DIR` | `$(dirname HF_CACHE)/dsv41-tensorfold-spill` | rank 0's spill folder on its local NVMe, mounted read-write at `/spill` | `-v <dir>:/spill` | — |
| `WORKER_SPILL_DIR` | `SPILL_DIR` | rank 1's, on the worker (a leading literal `$HOME/` is the worker's home) | — | `-v <dir>:/spill` |
| `SPILL_FLOOR_GIB` | `50` | free disk the tier leaves on the spill folder's file system; past it writes are refused | `--spill-min-free-gib F` | `TF_DS_SPILL_MIN_FREE_GIB=F` |
| `SPILL_QUOTA_GIB` | `0` (none) | what one conversation may hold before its own old turns and forks go first | `--spill-quota Q`, Q = `SPILL_QUOTA_GIB / SPILL_GIB` (a share 0 to 1; left out at 0) | `TF_DS_SPILL_QUOTA=Q` |
| `SPILL_MIN_TOKENS` | `8192` | shorter kept states are not written. **Must be equal on both ranks** (it is in the fingerprint's features): the recipe passes the one value to both | `--spill-min-tokens T` | `TF_DS_SPILL_MIN_TOKENS=T` |
| `SPILL_STAGING_MIB` | `128` | pinned staging each rank's tier works in, allocated before the pool (at least 16) | `TF_DS_SPILL_STAGING_MIB` | the same |
| `SPILL_TURNS` | `4` | old turns a conversation keeps on disk past its newest | `TF_SPILL_TURNS` | the same |
| `SPILL_FLUSH_MS` | `60000` | what a clean stop gives the tier to write what still waits (`STOP_TIMEOUT` follows it) | `TF_SPILL_FLUSH_MS` | the same |
| `SPILL_WEIGHTS_SHA` | `1` | each rank's weight file's SHA-256 as its `TF_DS_RANK_CACHE_SHA` (below); `0`: the checkpoint revision alone. It is part of the store's name: switching it either way opens another, empty store (the kept states stay in the old folder, unread) | `TF_DS_RANK_CACHE_SHA` (its own) | its own |
| `MODEL_QUANT` | from `MODEL_ID` (`exl3-2.9bpw`) | the quant's name in `TF_DS_MODEL_REVISION` | — | — |
| `KV_SHARD` | `0` (off) | each conversation's compressed latents stored once, on the Spark whose part of the pool holds it (`0118`): the same `POOL` costs about half the KV memory a Spark (measured at 2,101,248 positions: 1,062 MiB a rank against 1,783 MiB replicated), for 1-4% decode | env `TF_DS_KV_SHARD=1` | the same |
| `KV_SHARD_DECODE` | empty (engine default `fetch`) | `fetch` (the owner sends the rows decode picks; the reference) or `owner` (owner-computed decode) | env `TF_DS_KV_SHARD_DECODE` | the same |
| `KV_SHARD_CARRY` | `1` | carries of a kept state into the other part and live moves; `0`: every hit stays in its part | `TF_DS_KV_SHARD_CARRY` (with `KV_SHARD=1`) | the same |
| `KV_SHARD_TWINS` | `1` | borrowed twins for prompts that run from 0; `0`: every prompt chunk fetches | `TF_DS_KV_SHARD_TWINS` | the same |
| `KV_SHARD_CHECK` | `1024` | decode rounds between the ranks' planner digest checks (`0`: at start only) | `TF_DS_KV_SHARD_CHECK` | the same |
| `POOL` | empty: with `KV_SHARD=1` the pool it needs, derived from `CONTEXT` (2,101,248 at 1,048,576); with `KV_SHARD=0` `CONTEXT` | `zig/patches/0008`'s setting (`scripts/config.sh`), the only pool setting: total pool positions, every stream's extent together; `CONTEXT` stays the window one request may use. With `KV_SHARD=1` the engine cuts it into one part a rank, each holding the whole window (else it refuses, `PartTooSmall`) | env `TF_DS_POOL=P`, `TF_DS_WINDOW=CONTEXT` (start.sh's 0008 mapping, set or derived) | the same env and `--pool P` (`CONTEXT` when there is none, as before) |
| `RANK1_EXTRA_ENV` | empty | **tests only**: `NAME=value` words added last to rank 1's environment (a deliberately different rank 1, for `tools/gpu_validate.sh`'s refusal cases) | — | appended |
| `MIN_AVAIL_WAIT_S` | `120` | how long a start waits for a Spark short of `MIN_AVAIL_GIB` before refusing (below) | — | — |

Every engine name above is the one the port's engine branch (`port/series-pair-370daf0`) reads: rank 0's flags in `server/cli.zig` and
`TF_DS_SPILL_STAGING_MIB` in `native.spillSettings`; rank 1's variables in `spill_tier.Settings.follower`
(tf-dsv41-lanes also takes them as `--spill-*` flags; the recipe uses the variables so rank 1's command line stays the
one before the port); the shard knobs in `kv_lanes.Settings.fromEnv`. Both ranks get the whole set through
`ENGINE_ENV`, identical. The round-1 placeholders `TF_SPILL_DIR`, `TF_SPILL_GIB`, `TF_SPILL_MIN_FREE_GIB` and
`TF_SPILL_QUOTA` are gone: exported, each warns and reaches no rank.

Each setting is also taken from its engine variable when only that one is exported (`TF_DS_SPILL_GIB`,
`TF_DS_SPILL_MIN_FREE_GIB`, `TF_DS_SPILL_QUOTA` as the engine's share of the cap, `TF_DS_SPILL_MIN_TOKENS`,
`TF_DS_SPILL_STAGING_MIB`, `TF_SPILL_TURNS`, `TF_SPILL_FLUSH_MS`, `TF_DS_KV_SHARD*`), and the mapped names are not
passed a second time. `TF_DS_POOL` and `TF_DS_WINDOW` are the exception: `POOL` is the one pool setting (0008), so
hand-exported values of those two are ignored with a warning and both ranks always get the same pool and window. An exported `TF_DS_SPILL_DIR` is ignored with a warning: it is the
folder inside the container (always `/spill`); set `SPILL_DIR` / `WORKER_SPILL_DIR`. Rank 0's extra arguments
(`./start.sh restart ARGS`) may not set `--spill-*`: rank 1 would not get them, so `start.sh` refuses them.

### The spill tier's identity

With the tier on, the engine refuses to start unless every rank names the engine, the patch series and the weights
(`rank_identity.Extra.env`: `NoEngineId`, `NoRecipePatches`, `NoWeightsId`, `NoEngramId`). They enter the ranks'
fingerprint and each store's compatibility digest, so another image, checkpoint, quant or Engram revision reads none of
the kept states an older one wrote. `scripts/port.sh` derives them; nothing is typed:

| Variable | Value | Ranks |
| --- | --- | --- |
| `TF_DS_RECIPE_PATCHES` | the image's `tf.patches` label: the hash of `zig/patches`, `zig/Dockerfile` and the pins it was built from (`image_hash` before the image exists) | both, identical |
| `TF_DS_ENGINE_ID` | `tensorfold@<tf.ref label, 12 hex>+<that patch hash>` | both, identical |
| `TF_DS_MODEL_REVISION` | `<MODEL_ID>@<checkpoint revision>,quant=<MODEL_QUANT>,ablit=<0 or 1@ABLIT_REVISION>` | both, identical |
| `TF_DS_ENGRAM_REVISION` | `<ENGRAM_ID>@<Engram revision>` (Engram is always on here) | both, identical |
| `TF_DS_RANK_CACHE_SHA` | `sha256:<digest of that rank's rank<N>of2-*.bin>`, computed once (O_DIRECT reads, ~1 minute a Spark) and cached in `.<file>.sha256` beside it with the file's size, mtime and inode; recomputed when the file changes. When a rank's digest cannot be had (the worker's probe is tried twice) a real start refuses, before anything is stopped: a rank without it would name another store and lose every kept state; DRY_RUN only warns. `SPILL_WEIGHTS_SHA=0` is the only way to run without it | each its own |

An exported `TF_DS_ENGINE_ID`, `TF_DS_RECIPE_PATCHES`, `TF_DS_MODEL_REVISION` or `TF_DS_ENGRAM_REVISION` wins;
`TF_DS_BUILD_FLAGS` (optional) passes through when exported. With an older `IMAGE=` tag the labels of that image are
used, so its kept states stay its own.

Engine knobs without a recipe setting reach both ranks as they are when exported, as `TENSORFOLD_*` and `TF_DS_*`
always did: `start.sh` also passes `TF_SPILL_*` (rank 0's lane host: `TF_SPILL_LOAD_MS` 40, `TF_SPILL_LOAD_WAIT_MS`
30000).

`STOP_TIMEOUT` defaults to `SPILL_FLUSH_MS` rounded up to seconds plus 30 (90 s at the engine's default 60000; 30
before) when `SPILL_GIB` is on, or when the running container was started with the tier, so a clean stop can write what
still waits. A `SPILL_FLUSH_MS` that is not a whole number of milliseconds is refused. `MEMGUARD_GIB`,
`MEMGUARD_HOLD_S` and `MEMGUARD_HARD_GIB` mean what they did.

**`MIN_AVAIL_GIB` counts parked pages** (GLM `0119`, the engine's `native/cuda_memory.zig` rule, in
`scripts/memavail.sh`): available = MemAvailable plus the pages on the kernel's per-CPU free lists beyond their resting
level (every `count:` under a zone's `pagesets` less every `high_min:`; 512 MiB when the kernel prints none), at most
MemTotal, on both Sparks. An image build or a `docker save | load` leaves several GiB there for a minute or two, which
MemAvailable does not count: the round-1 GPU restore was refused at 110.3 < 112 GiB right after one. A Spark still
short waits up to `MIN_AVAIL_WAIT_S` (120 s) before the refusal; the message names the parked share. A Spark whose
`/proc/meminfo` cannot be read (ssh fails, or no `MemAvailable`) is tried twice and then refused at once ("could not
read /proc/meminfo on <host>"): that is not a low reading and does not drain away.

## What `start.sh` checks, before anything is stopped

- **Forms and combinations** (`port_validate`): numbers are numbers, switches are 0 or 1, `POOL >= CONTEXT`,
  `SPILL_QUOTA_GIB <= SPILL_GIB`, spill folders are absolute paths of `A-Z a-z 0-9 . _ / -`. `POOL` above 4x `CONTEXT`
  warns and above 16x is refused, with the estimate of its state on each Spark (~890 bytes a position, calculated from
  the spill block layout, not measured: ~1.7 GiB at `POOL=2101248`). With `KV_SHARD=1`: two Sparks; `POOL` in whole
  4,096-position parts; and **each part holds a whole `CONTEXT` window plus a round's rows and two** (the engine's
  `shard.plan` rule; at `CONTEXT=65536` that is `POOL=135168`), so the
  window is never halved silently: at `CONTEXT=1048576` and 16-row rounds that is `POOL=2101248` (`2097152` would cut a
  conversation to 1,048,558 tokens; `start.sh` names the value to set). `KV_SHARD=1` with `SPILL_GIB>0` runs since
  `port/spill-shard` put the owner layout and part into the spill identity (`PORT_SHARD_SPILL_READY=1` in
  `scripts/port.sh`; validated on the pair as the daily config, PORT-STATUS r5). The spill settings'
  forms: `SPILL_MIN_TOKENS` a count (off the 2,048 grid warns), `SPILL_STAGING_MIB` at least 16, `SPILL_TURNS` and
  `SPILL_FLUSH_MS` whole numbers; `KV_SHARD_CARRY` / `KV_SHARD_TWINS` 0 or 1, `KV_SHARD_CHECK` a count.
- **The spill identity** (`port_ident_map`, `port_weights_check`, with the tier on): the checkpoint's and Engram's
  revisions must be known (`MODEL_REVISION` / `ENGRAM_REVISION`, else the cache's `refs/main`; an exported
  `TF_DS_MODEL_REVISION` / `TF_DS_ENGRAM_REVISION` stands in), or the start is refused: an empty one would let two
  revisions share one identity and one store. With `SPILL_WEIGHTS_SHA=1` both ranks' weight digests must be had
  (the worker's probe is tried twice), or the start is refused; `DRY_RUN=1` only warns.
- **Rank 0's flags against the image** (`scripts/zig_flags.py` on `capabilities --json`): an image without the port
  refuses `--spill-*` here, not after the old server is stopped.
- **Each Spark's spill folder** (`port_spill_check`, with `SPILL_GIB` on; run before the flags and the label are
  made. With the default `SPILL_GIB` a failure below turns the tier off for the start with a warning, with
  `SPILL_GIB` set it refuses): created when missing, as you (docker would
  create a missing bind source as root); on a local disk (NFS, CIFS, sshfs, 9p, Ceph, Gluster, Lustre, rclone and
  other network mounts refused, and so are tmpfs, ramfs, devtmpfs, overlay, squashfs and a `/dev/zram*` device); not
  inside an NFS export listed in the kernel's `/var/lib/nfs/etab`, `/etc/exports` or `/etc/exports.d/*.exports`; not
  overlapping `HF_CACHE`,
  `NFS_PATH`, `HEAD_ENGRAM_DIR` or the served data folder (rank 0), or `WORKER_ENGRAM_DIR`, `WORKER_CACHE_DIR` or the
  worker's Hugging Face cache (rank 1), in either direction; not `/` or your home itself; no path component containing
  `archive`; not owned by root; at least `SPILL_FLOOR_GIB` free (a warning under `SPILL_GIB + SPILL_FLOOR_GIB`).
  `DRY_RUN=1` creates nothing and describes the nearest existing parent.
- **Both ranks agree.** Both get the same variables from the same settings (`port_map`). Both containers carry
  `tf.recipe.port=<settings>` (with the tier on, also a checksum of `SPILL_DIR` and `WORKER_SPILL_DIR`: another folder
  is another set of kept states); `./start.sh` on a running server refuses to adopt ranks whose labels differ from each
  other or from the settings asked for (containers from before the port read as `off`). At load the engine checks again
  on its own: `Model.open` all-gathers the shard mode, pool, window and round rows and refuses a mismatch
  (`KvShardMismatch`); with the tier on, the ranks exchange their full fingerprint (rank_wire hello) and a
  difference ends the start naming the rank and field; with `KV_SHARD=1` the shard's own agreement compares the
  carry/twins switches and the layout (`ShardLayoutMismatch`).

## Spill folder, ownership, disk use

Each rank's folder is its own Spark's local disk, mounted at `/spill`; the store keeps
`/spill/native-spill-<compat>/rank<N>/{owner.lock, blocks/, points/}` (directories 0700, files 0600), a new
`native-spill-<compat>` folder for every model/engine/layout identity, so an image change never reads another
layout's files.

The ranks' containers run as root (the GPU, `IPC_LOCK`, `/dev/infiniband`), and the store writes as root, so the
recipe gives the files back to you: before a start (the leftovers of a crash) and at every stop (`stop.sh`, by the
container's `tf.recipe.spill` label, even after `SPILL_GIB` was set back to 0). Changing a file's owner away from root
needs root (`CAP_CHOWN`), which is why a throwaway container of the serving image does it (no other image is pulled),
and why that container is kept as narrow as it can be (`spill_fix_target` / `spill_fix_owner` in `scripts/port.sh`):
- **only a folder `start.sh` accepted:** after `port_spill_check` passes, `start.sh` creates `.dsv41-tensorfold-spill`
  in each Spark's spill folder, as you. The folder must be an absolute, canonical path (no symlink anywhere on it), a
  real directory, owned by you (not root), and hold that marker, owned by you and not a symlink; anything else (a
  label naming another folder, a folder swapped for a symlink, a root-owned folder) is refused, with a warning, and
  left untouched;
- **only that folder** is mounted (at `/spill`), nothing else of the host;
- **only root-owned regular files and directories under it**: never the folder itself (`-mindepth 1`), never a symlink
  (`find` does not follow them, `-type f -o -type d` leaves them out, `chown --no-dereference`), never across a file
  system (`-xdev`); they become yours (`id -u`:`id -g`), whatever the folder's metadata says;
- the container has **no network, a read-only root, `no-new-privileges`**, and of root's capabilities only `CHOWN` and
  `DAC_READ_SEARCH` (to walk the store's 0700 folders): `--cap-drop ALL --cap-add CHOWN --cap-add DAC_READ_SEARCH`.

Files written while a rank runs belong to root until it stops. Running the serving containers as your own uid would
remove the need, but the engine's GPU, pinned-memory and RDMA setup is only validated as root; a store that `chown`s as
it writes (GLM `0110`'s behaviour) is the engine-side alternative (open items).

Disk use, calculated from the engine's layout (`port/dsv41-state`): **1,822,720 bytes a 2,048-token block a Spark**
(890 bytes a token: the compressed codes and scales, index keys, window and raw state), plus **6,684,672 bytes of small
state a kept point a Spark**. A 100K-token conversation is ~85 MiB of blocks a Spark, the 1M window ~890 MiB; blocks a
later turn shares with an earlier one are stored once. `SPILL_GIB=64` holds ~77M tokens of distinct blocks a Spark. The
cap and the floor are each Spark's own; nothing is written to the NFS export, the model folders or any archive.

With KV sharding on as well the planes are no longer replicated: the Spark that owns a conversation's
part writes its blocks at ~890 bytes a token, the other Spark ~170 bytes a token (its share of the state); without
sharding both write ~890 bytes a token. The cap is per Spark either way: **at most `SPILL_GIB` (64 GiB at the validated setting) of each
Spark's local disk**, plus `SPILL_FLOOR_GIB` (50 GiB) always left free there; the tier stops writing at the floor
before it reaches its cap. `SPILL_GIB=0` uses no disk at all.

## When the rank pair breaks: status 75

With two ranks the engine ends a broken pair instead of hanging on it (`zig/patches/0020`, the broken-pair policy
`TF_DS_PAIR_BROKEN`; tensorfold `docs/port/robustness.md`). When rank 1 dies or hangs, or a spill-tier or KV-transfer
exchange breaks, rank 0 fails what waits and decodes, logs `rank pair broken (<cause>) ... exiting with status 75`
and its process ends with **status 75**. A dead rank 1 is seen at once (about 1.3 s: rank 0 watches its link to the
follower), a hung one by the stall monitor (120 s without progress while requests run), and after a break rank 0 has
at most 15 s to end (the drain deadline); both bounds are fixed. Meanwhile `/health` answers 503 (`stalled`, then
`broken`) and `/metrics` counts the pair (`tensorfold:pair_*`). The stock engine keeps rank 0 waiting forever instead,
its requests stuck, and no supervisor can restart a process that never exits.

This holds whatever the port's settings are, because it acts only after a failure: replies are unchanged, and a
single rank has no pair to break. `TF_DS_PAIR_BROKEN=degrade` (exported, it reaches both ranks) opts out: rank 0 then
stays up after a break, failing requests one by one with the spill tier stopped, and whatever watches `/health`
decides. Any other value refuses the start.

**Restarting is the supervisor's job.** `start.sh` runs detached containers (`docker run -d`, no `--restart`), so
without a supervisor the server stays down after the exit, cleanly (`/health` does not answer), until
`./start.sh restart` (or `./start.sh`) starts both ranks again. To have that happen by itself, run the recipe
under its systemd user service, **[docs/systemd.md](systemd.md)**: with `FOREGROUND=1` the launcher stays attached to
rank 0 and exits with its code, stopping rank 1, and `Restart=on-failure` starts both ranks again 60 s later
(`ExecStop`, `./stop.sh`, runs first), at most three starts in 30 minutes. Keep that one supervisor for the pair.

**The failed run's logs are kept** however the pair is stopped or restarted: `stop.sh` saves each rank's log, an
exited container's too, before it removes the container (`LOG_DIR/<date>-<time>-rank0.log.gz` here, rank 1's in the
worker's `~/.cache/dsv41-tensorfold/logs`, the newest `LOG_KEEP` of each), and `start.sh` hands any container it finds,
running or exited, to `stop.sh` before it launches. The newest `-rank0` file then ends with the engine's
`exiting with status 75`. `tools/gpu_validate.sh`'s `pair-kill` and `pair-stop` configurations test the whole path
on the Sparks: rank 1 killed, or SIGSTOPped, while 8 streams decode.

## Exporting the engine branch: `scripts/port-patches.sh`

```
scripts/port-patches.sh --branch REF --dry-run        # REF of ~/GitHub/tensorfold-dsv41, one patch a first-parent commit
scripts/port-patches.sh --branch REF                  # the same, written into zig/patches and its README
scripts/port-patches.sh --branch REF --mode squash --name SLUG --title TEXT   # one patch instead
scripts/port-patches.sh --engine DIR --base REF --exclude PATH --keep         # another repository, base or exclusion
```

`--base` defaults to the engine branch that `zig/patches` reproduces (`--help` names it), so `REF` is engine work on
top of it, exported as the next numbers. It writes nothing unless a clean `TF_REF` checkout plus the hand-written
patches equals the base, every new patch passes `git apply --check` and applies there in filename order (as
`zig/Dockerfile` applies them), and the result equals the branch outside the port's own notes (`PORT.md`,
`docs/port/`). A rerun replaces the patches it wrote before (first line `# port-patches.sh:`) and the README rows it
owns (`<!-- port-patches -->`). The image tag is a hash of the patches and the Dockerfile, so the next
`./start.sh restart` builds a new image; the old one stays until you remove it.

## Deploy

The fork that serves today runs from `~/DeepSeek-port` on gx10-a through a systemd drop-in: **docs/FORK-DEPLOY.md**
is the procedure for that checkout (a new tree from this branch beside the old one, swapped in, restarted through the
unit; rolled back by swapping back). The steps below are the general ones for a checkout that serves directly.

1. **Engine branch ready.** Engine work after the series (a branch on top of the one `zig/patches` reproduces:
   `port/series-pair-370daf0` with the series alone) checked as `PORT.md` asks (host tests, the Linux cross-build;
   GPU checks on the pair by the coordinator). With nothing after the series, go on at step 3.
2. **Export.** `scripts/port-patches.sh --branch <engine branch> --dry-run`, then without `--dry-run`; commit
   `zig/patches` on the recipe branch.
3. **Record what serves now**, to come back to. A live checkout may carry working-tree edits, and its image tag is a
   hash of `zig/patches` and `zig/Dockerfile` as they are on disk, so record the commit **and** the diff. In the live
   checkout, before anything changes there:

   ```
   mkdir -p ~/dsv41-rollback && d=~/dsv41-rollback/$(date +%Y%m%d-%H%M)
   docker inspect -f '{{.Config.Image}}' dsv41-1017-exl3-tf | tee $d.image          # the serving tag, e.g. tensorfold-dsv41:exl3-tp2-<hash>
   ssh <worker> docker inspect -f '{{.Config.Image}}' dsv41-1017-exl3-tf             # the same tag on rank 1
   git rev-parse HEAD > $d.commit; git status --porcelain > $d.status
   git diff --binary HEAD > $d.diff                                                 # the applied patches, exactly
   bash -c 'source scripts/config.sh; echo $IMAGE'                                 # must print the serving tag
   ```

   Keep that image on **both** Sparks (no `docker image prune`; `docker tag <tag> tensorfold-dsv41:rollback` on each
   also protects it from a careless prune by tag).
4. **Same behavior first, all settings off** (the defaults; `SPILL_GIB=0 KV_SHARD=0` in `scripts/local.sh` on a fork
   that turned them on).
   `./start.sh restart` (it runs `scripts/prepare.sh`: the new image is built
   on the head and copied to the worker; the compiler comes from the mirrors with bounded retries). Then compare replies
   with the previous image: `tools/exact.py` record on the old image, compare on the new one (drafts on and off, 4 and 8
   streams, images, resumes). Feature-off must be identical.
5. **One feature at a time**: `SPILL_GIB=64` in `scripts/local.sh` (`SPILL_QUOTA_GIB`, `SPILL_DIR` /
   `WORKER_SPILL_DIR` if the defaults are not each Spark's NVMe), then `KV_SHARD=1` (the derived `POOL=2101248`).
   `DRY_RUN=1 ./start.sh restart` prints both ranks' docker commands (the `/spill` mounts, the variables, rank 0's flags,
   rank 1's `--pool`) and changes nothing; then `./start.sh restart`. A supervisor that runs `start.sh` (a systemd unit
   with `FOREGROUND=1` or a oneshot) restarts through that unit instead, so it keeps owning the server.
6. **Watch** `/health` and `/metrics` (`prompt_cache`, `spill_*`, `chats_continued`, `kv_shard`, `pair`), `memguard.log`,
   free disk on both Sparks, and the start-up line `dsv41: pool ... a conversation may use up to ...` with `POOL` set.

## Rollback

- **A feature off, same image:** set it back (`SPILL_GIB=0`, `KV_SHARD=0`, `POOL=` empty) and
  `./start.sh restart`. With `SPILL_GIB=0 KV_SHARD=0` this is the image with the port present and unused, but for the
  broken-pair exit (above).
- **The previous image, quickest:** from this tree, with every port setting off **explicitly** (in case a fork turned a default on):
  `SPILL_GIB=0 KV_SHARD=0 IMAGE=<the tag noted in step 3> PREPARE=0 ./start.sh restart`. `PREPARE=0` keeps `scripts/prepare.sh` from running, so nothing is rebuilt under that
  tag; the image must still exist on both Sparks (`docker image inspect <tag>` here and on the worker). With all
  settings off the two ranks' docker commands are the old ones but for the `tf.recipe.port=off` label
  (`scripts/port_test.sh`'s golden test), and `zig_flags.py` refuses any port flag the old image does not serve (so a
  rollback that forgets `SPILL_GIB=0` refuses before anything is stopped).
  **Never** combine `IMAGE=<old tag>` with `PREPARE=1` or `auto`: `prepare.sh` would rebuild that tag from the current
  patches.
- **The previous image, exact tree:** restore the tree it was built from, in the live checkout or a fresh one:
  `git checkout $(cat $d.commit) && git apply $d.diff` (`$d.status` lists what else was there, such as
  `scripts/local.sh`), check that `source scripts/config.sh; echo $IMAGE` prints the noted tag, then
  `./start.sh restart`: `scripts/prepare.sh` finds that image built and rebuilds nothing. Checking out the commit
  **without** the diff does not give the serving tag back: it names a different image, which `prepare.sh` would build.
- **Spilled state** stays in each Spark's spill folder for inspection, owned by you after `./stop.sh`; nothing else reads
  it. It is never reinterpreted under another layout (its folder is named by the compatibility digest). To reclaim it:
  stop, then `rm -rf <SPILL_DIR>/native-spill-*` on each Spark.
- No weight, Engram, kit or NFS change is involved in either direction; the memory guards are unchanged.

## Validation tools

`tools/` (merged from `port/recipe-tools`): `exact.py`, `spill_check.py`, `kv_shard_check_e2e.py`, `prefix_fanout.py`,
`pair_check.py`, `bench.py`, and `tools/gpu_validate.sh`, the coordinator's runbook (configurations mapped one to one to
the GPU validation lists of `docs/port/spill-lanes.md`, `shard-lanes.md`, `spill-shard.md`, `spill-pictures.md` and
`robustness.md`; `tools/PORT-VALIDATION.md`). The KV
checks read the engine's real counters: `/health` `kv_shard{...counters, per_part[...]}` and `/metrics`
`tensorfold:kv_shard_*`, `tensorfold:kv_shard_part_*{part="N"}`. `python3 -I tools/selftest.py` runs every tool against
`tools/fake_server.py` (which serves the engine's shapes) and against planted faults.

**API key.** The `checklib` tools (`spill_check.py`, `kv_shard_check_e2e.py`, `resume_bench.py`, `pair_check.py`,
`prefix_fanout.py`) read the server's key (`API_KEY=1`) from the `API_KEY` environment variable, or from the file a
`--api-key-file PATH` flag names (before the subcommand), and from nowhere else: no default path, no `API_KEY_FILE`
variable, no home folder. The key goes into the `Authorization` header only; receipts and printed lines are scrubbed
of it (`checklib.redact`). `tools/gpu_validate.sh` passes `--api-key-file $STATE_DIR/api-key` when the port tree has
one. `python3 -I tools/test_checklib_credentials.py` checks this.

## Open items for the recipe

- The store `chown`s as it writes (GLM `0110` behaviour) so files are never root-owned while a rank runs; until then
  the recipe's narrowed chown pass at start and stop stays (above).
- Rank 1's logs still land in the worker's fixed `~/.cache/dsv41-tensorfold/logs` (`LOG_DIR` is rank 0's only).
- Owner-computed decode (`KV_SHARD_DECODE=owner`) at 8 streams and 16-row rounds needs its kernel variants in the kit
  (`MissingTritonVariant` is what a missing one looks like); `fetch` uses none. Kit `abe07c7` stays pinned; the RB
  17..32 variants (`ROUND_ROWS=32` in one launch) are a later `KIT_REF` change.
- Measure what this page only calculates: disk use, the pool's memory at `POOL=2101248`, restore times.
- The DRY_RUN golden test of `start.sh` (`scripts/port_test.sh`) needs bash 4: the Mac's bash 3.2 skips it, so it
  has not run on this round's start.sh yet. `docs/FORK-DEPLOY.md` step 4 runs it on gx10-a in the new tree and stops
  the deploy unless every golden case passes (compared with MiaAI-Lab main `370daf0`: the base, all off, KV shard
  against the base's own `POOL=2101248`, KV shard with the derived pool, the defaults, an explicit failing spill
  folder refusing (a fork with defaults on: its degrade too), and spill on).
