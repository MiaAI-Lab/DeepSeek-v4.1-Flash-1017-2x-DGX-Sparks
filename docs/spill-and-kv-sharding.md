# Spill v2 and KV sharding (ported, off by default)



The port of Robert Wojciechowski's ([wojo](https://github.com/wojo) on GitHub, [@rwojo](https://x.com/rwojo) on X) GLM-5.3-Flash patches into this engine: the spill
tier and its v2 block store (GLM `0088` and `0110`, PRs #78 and #124; the block layout, lineage, write-at-keep ring and
fsync order are [BadAd84](https://github.com/BadAd84)'s block-tier design), chat continuation (`0111`, PR #128), KV
sharding with prompt borrowing and the ring exchange (`0118`, PR #152) and the parked-page startup budget (`0119`, PR
#149). The engine changes come in as `zig/patches/0011-...` to `0022-...`, one patch per feature, after `0008` and
`0010` ([zig/patches/README.md](../zig/patches/README.md)). **Everything is off by default**, in the engine and in the recipe: with
`SPILL_GIB` and `KV_SHARD` unset (or `0`) and `POOL` empty the recipe runs as before the port, with no flag, variable
or mount it did not have (one engine change acts regardless, after a failure only: a broken rank pair ends the
server, below). Turn them on in `scripts/local.sh`: `SPILL_GIB=64 KV_SHARD=1` is what the pair was validated
with (replies byte-identical to the stock engine in every configuration: all off, `POOL`, KV sharding at 1 and 8
streams, spill, spill with KV sharding; restore after eviction, a restart and damaged files). Full detail:
[docs/PORT-RECIPE.md](PORT-RECIPE.md).

**Set explicitly, a feature that cannot run refuses the start:** a spill folder that cannot be used (missing and not
creatable, not yours, on NFS or tmpfs, inside the export or a model folder, below `SPILL_FLOOR_GIB` free) on either
Spark, or `KV_SHARD=1` with a `POOL` it cannot cut into two whole-window parts (such as `POOL=1048576`). A fork that
turns the defaults on (`PORT_SPILL_DEFAULT_GIB` / `PORT_SHARD_DEFAULT` in `scripts/port.sh`) gets the other behaviour
for those defaults: the start goes on **without** the feature, with a loud `SPILL TIER OFF for this start` /
`KV SHARDING OFF for this start` warning.

- **Spill v2** (`SPILL_GIB`): each kept prompt state is written, when it is kept, to each Spark's own local disk in
  2,048-token blocks that later turns of the conversation share, and read back instead of a new prefill: after
  eviction, after a restart, after a crash. Room is made by conversation (old turns past the newest few, abandoned
  forks, whole conversations least recently used, shared prefixes last), with an optional per-conversation quota.
- **KV sharding** (`KV_SHARD=1`): builds on `zig/patches/0008`'s pool. There is one `POOL` setting, the total pool
  positions (`CONTEXT` stays the window one request may use); with `KV_SHARD=1` that same pool is cut into one part a
  Spark, and each conversation's compressed latents are stored once, on the Spark whose part holds it, instead of on
  both. Each part must hold a whole window (`CONTEXT` plus a round's rows and two, in whole 2,048-position blocks), so
  at the 1,048,576-token window `POOL=2101248`: left empty, it is derived; a `POOL` that would cut a conversation's
  window is refused, naming the value. **The same `POOL` costs about half the KV memory a Spark** (owner-only planes):
  measured at 2,101,248 positions, 1,062 MiB a rank against 1,783 MiB replicated, for 1-4% decode. Runs together with
  spill.

**The ranks' Engram folders must list the same files.** With the tier on, the rank fingerprint covers the Engram
folder's regular files by name and size (`zig/patches/0016`, `fingerprint.zig`'s `listing()`, which skips symlinks).
With `HEAD_ENGRAM_DIR` empty rank 0 reads the HF snapshot, whose entries are symlinks into `blobs/`, so it lists nothing
while `WORKER_ENGRAM_DIR` lists its four files: the pair then refuses to load with `FingerprintMismatch` in `engram`
(issue #61). `start.sh` compares the two listings before anything is stopped and refuses with both of them and the fix;
put the same files on the head's disk and point `HEAD_ENGRAM_DIR` at them (hardlinks cost no extra disk), which also
gives rank 0 local Engram reads instead of the HF cache.

| Switch | Default | |
| --- | --- | --- |
| `SPILL_GIB` | `0` (off) | the spill tier's cap a Spark, GiB (`64` validated). A Spark whose folder fails the checks refuses the start |
| `SPILL_DIR` / `WORKER_SPILL_DIR` | `$(dirname HF_CACHE)/dsv41-tensorfold-spill` / `SPILL_DIR` | each rank's own folder on its local NVMe, mounted at `/spill`. Never inside `HF_CACHE` (it is the NFS export the worker reads), never on a network file system, never under an archive folder: `start.sh` checks this on both Sparks, creates the folder as you, and refuses a root-owned one |
| `SPILL_FLOOR_GIB` | `50` | free disk the tier always leaves on that file system |
| `SPILL_QUOTA_GIB` | `0` (none) | one conversation's share of `SPILL_GIB` before its own old turns and forks go first |
| `SPILL_MIN_TOKENS`, `SPILL_STAGING_MIB`, `SPILL_TURNS`, `SPILL_FLUSH_MS` | `8192`, `128`, `4`, `60000` | the tier's shortest written state, pinned staging, turns kept, stop flush bound: the same on both ranks |
| `KV_SHARD`, `KV_SHARD_DECODE` | `0` (off), empty (`fetch`) | KV sharding; `owner` for owner-computed decode |
| `KV_SHARD_CARRY`, `KV_SHARD_TWINS`, `KV_SHARD_CHECK` | `1`, `1`, `1024` | carries and live moves, borrowed twins, rounds between the ranks' planner checks |
| `POOL` | empty (with `KV_SHARD=1`: two owner parts of a whole `CONTEXT` window each, 2101248 at 1048576; with `KV_SHARD=0`: `CONTEXT`) | `zig/patches/0008`'s setting (both ranks' `TF_DS_POOL`, `TF_DS_WINDOW` and rank 1's `--pool`); `TF_DS_POOL` exported by hand is ignored |

Both ranks get the same settings (the engine's flags and variables are mapped in one place, `scripts/port.sh`); both
containers are labelled with them, and `./start.sh` will not adopt two running ranks that disagree. With the spill tier
on, `start.sh` also gives both ranks the identity the engine requires (the image's patch hash and ref, the checkpoint's
and Engram's revisions, each rank's weight-file digest, computed once and cached beside the file). **Disk use**
(calculated): ~890 bytes a token a Spark replicated, 1,822,720 bytes per 2,048-token block (~85 MiB for a 100K-token
conversation, ~890 MiB for the 1M window) plus 6.4 MiB a kept point a Spark; with KV sharding the owner
Spark writes ~890 and the other ~170 bytes a token. Blocks a later turn shares are stored once. The tier holds at most
`SPILL_GIB` (64 GiB) of each Spark's own disk and always leaves `SPILL_FLOOR_GIB` (50 GiB) free there. The engine runs as root; spill
files are given back to the folder's owner before each start and at every stop, by a container that can do nothing
else (only the folder start.sh accepted and marked, only root-owned files and folders in it, no symlinks, no network,
only the CHOWN and DAC_READ_SEARCH capabilities; `scripts/port.sh`, `spill_fix_owner`). With the tier on, `./stop.sh` gives
the ranks 90 s (`SPILL_FLUSH_MS` plus 30 s) to write what still waits.

**A broken rank pair ends the server** (`zig/patches/0020`): when rank 1 dies or hangs, or a spill-tier or KV-transfer
exchange breaks, rank 0 fails its requests and exits with status 75 instead of hanging forever (a dead rank 1 is seen
within seconds, a hung one after 120 s without progress). This one is on with two ranks whatever the settings, since
it acts only after a failure (replies are unchanged); `TF_DS_PAIR_BROKEN=degrade` opts out. To have both ranks start
again by themselves, run under the [systemd user service](systemd.md); without a supervisor the server stays
down until `./start.sh restart`. Either way the failed run's logs are saved first (`stop.sh` saves each rank's log,
an exited container's too, and `start.sh` hands any container it finds to `stop.sh` before it launches). Details:
[docs/PORT-RECIPE.md](PORT-RECIPE.md) "When the rank pair breaks".

**Rollback:** `SPILL_GIB=0` / `KV_SHARD=0` (and `POOL` empty) with `./start.sh restart` turns a feature
off on the same image. The previous image comes back with `SPILL_GIB=0 KV_SHARD=0 IMAGE=<its tag> PREPARE=0 ./start.sh
restart` (every port setting off, explicitly, in case a fork turned a default on), or by restoring the exact tree it was built from, commit **and** working-tree diff (the image tag is a
hash of `zig/patches` and `zig/Dockerfile` as they are on disk), then `./start.sh restart`: record both at deploy, as
[docs/PORT-RECIPE.md](PORT-RECIPE.md) describes. Spilled files stay in each Spark's folder for inspection
(`rm -rf <SPILL_DIR>/native-spill-*` reclaims them); no weights, Engram or NFS change in either direction, and the
memory guards are unchanged.

For agent jobs that loop at `reasoning_effort: "max"`, try `"high"` with the same request and compare both reasoning and tool-call output (#33, #37). The pinned engine maps `high` to 75, `xhigh` to 88 and `max` to 100; the HTTP field accepts the named efforts, not arbitrary integers. No universal `THINKING_BUDGET` is validated for agent tasks: closing the reasoning block does not guarantee a tool call or stop repetition, and a small budget can cut a plan short. A client's explicit `max_tokens` / `max_completion_tokens` takes precedence over the server default; bound it in the request when needed. These reports alone do not establish that quantization caused the loop.
