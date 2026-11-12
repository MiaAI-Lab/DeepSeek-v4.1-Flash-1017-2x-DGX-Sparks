# Optional spill-v2 expiry and host-memory admission

`SPILL_TTL_H` and `SPILL_MEM_FLOOR_GIB` default to zero and are mapped to both
ranks only when spill is enabled and the corresponding policy is positive.
Aliases `TF_DS_SPILL_TTL_H` and `TF_DS_SPILL_MEM_FLOOR_GIB` are accepted; recipe
settings take precedence. Disabled defaults preserve the existing launch
arguments and mapped environment. No pool, context, draft, reasoning or
compaction setting changes.

Example in `scripts/local.sh`:

```sh
SPILL_GIB=16
SPILL_TTL_H=24
SPILL_MEM_FLOOR_GIB=3.5
```

TTL is absolute age since creation, persisted in the existing point timestamp.
Lookup and begin-read reject expired points. On cache access, unpinned expired
points are pruned; active reads remain pinned and may finish. This does not
provide timed background deletion or secure erasure. There is no manifest or
cache-format change. A repeated keep can create a fresh point after expiration.

The memory floor uses Linux host `MemAvailable`, plus the configured staging
reserve. Below that sum, new writes, lookups and reads are refused; unreadable
memory information fails closed. Equality at the sum is admitted. In-flight
operations finish. It is admission policy, not a reservation against competing
processes or a guarantee against OOM. Disk-free/cap and existing retention
policies remain independent. Fractional limits are accepted; negative,
nonfinite and out-of-range settings are refused.

## Validation and limits

- Applies independently on `c7f2b851` / `fbbc971` plus all 53 current patches;
  no guard, first-token or prefix-preservation PR dependency.
- `zig build -j2 test --summary all`: 23/23 steps succeed in a CPU-only Linux
  ARM64 container with the pinned compiler. Successful cached targets were
  reused on the non-root rerun; the existing permission-fault suite needs an
  unprivileged user.
- Store regressions cover expiry across reopen, fresh replacement, one byte
  below and exactly at memory floor plus staging reserve, unknown memory,
  disabled-policy defaults, and expiry while a read is already pinned.
  Family parser tests cover matching rank settings, defaults and invalid limits.
- Recipe mapping/refusal suite: 152/152 checks pass as an unprivileged user,
  including 14 new cases for defaults, alias mapping, spill-off and invalid values.
  The existing DRY_RUN historical golden check skips when its historical commit
  cannot be resolved inside the test container.
- Prior combined deployment testing with these protections enabled recovered
  an exact 78,915-token retrieval after graceful restart and an abrupt head
  exit; 77,824 prefix tokens were reused, with no CRC/size/read failures.
  The crash followed a committed point; no torn-write injection was performed.

Live recovery evidence belongs to the combined deployed image, not an isolated
GPU test of this branch. A standalone two-Spark build, drafted/plain and
resumed/concurrent equivalence, upstream-default memory headroom, a 24-hour
live expiry soak and deliberate host-memory exhaustion remain untested. This
branch does not change or restart production.

Positive policies participate in the running-pair configuration identity: changed limits require an explicit `./start.sh restart`; disabled defaults retain existing labels.
