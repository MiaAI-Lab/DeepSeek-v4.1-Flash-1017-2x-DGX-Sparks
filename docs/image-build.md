# Native image builds and compiler caches

`zig/Dockerfile` requires a BuildKit-enabled Docker builder with support for
Dockerfile heredocs and cache mounts. `scripts/prepare.sh` still supplies its
small build context and pinned build arguments; the Dockerfile and patches
remain inputs to the image hash. Stop inference before building on a Spark.

## Stages and retained files

The pinned NVIDIA base provides the CUDA toolchain. A compiler stage adds the
checksum-verified Zig archive. The serving builder applies the ordered engine
patches and installs the native engine; the inputs builder derives from it,
keeping the two all-core builds sequential. The final stage starts from the
same NVIDIA base and copies the complete `/opt/tensorfold` install tree,
including preparation tools and license notices.

The downloaded Zig compiler, checkouts, patches and compiler caches stay in
builder stages. The named compiler and inputs-builder stages can be selected
as build targets with the same context and pins for development. NVIDIA's base
still includes its own CUDA compiler and PyTorch;
this is not a minimal serving image. The existing pinned Python dependencies
install after the native outputs, so changing them does not recompile the engine.
NVIDIA's entrypoint, runtime environment and the native CPU/SM121 targets remain
unchanged. No model, serving or memory configuration changes are required.

## Compiler cache behavior

BuildKit keeps a shared global Zig cache and separate serving/inputs local
caches outside image layers. Locked mounts protect them during concurrent
image builds. Their contents are disposable: a build must also work with empty
caches. Changing source or patches normally reruns the affected instructions
while allowing Zig to reuse unchanged work. An unchanged build can reuse whole
Docker layers without invoking Zig at all.

`--no-cache` and `--no-cache-filter` may reset cache mounts. Measure warm caches
with a real source/patch edit instead of treating those flags as layer-only
invalidation. Builder pruning can remove caches; the runtime image does not
need them. Intermediate stages and caches consume builder disk separately, so
a smaller runtime image does not imply smaller total build-host storage.

## Build measurements

A two-Spark ARM64 build experiment using an already-local NVIDIA 26.10 base
reported 24,856,153,689 bytes for the runtime image, versus 25,278,792,693 bytes
before this refactor: about 423 MB / 1.67% smaller (Docker-reported sizes, not
compressed transfer sizes or inference memory). The large NVIDIA base limits
size savings.

Single samples: first build 203 s; normal rebuild after a comment-only build.zig
patch 125 s; returning to the unchanged context 1 s through layer caching. These
are exploratory cases, not a matched old/new Dockerfile benchmark or a promise
that arbitrary source edits rebuild at the same rate. No inference speedup is
claimed.

## Build-refactor qualification

The staged image was loaded on both Sparks and compared with the retained
NVIDIA 26.10 / Pillow 13 / Zig 0.17.1 image. All 31 greedy cases matched prompt,
server token and complete text hashes/text; request errors and failed internal
checks were zero. JPEG, WebP and GIF inference also passed. Offline validation
included 32 script fixtures, ten decoder cases and the preparation dtype fixture.

| Setting or measurement | Scope |
| --- | --- |
| Configured window | `CONTEXT=1048576` |
| Configured pool | `POOL=1048576`, shared across all requests |
| Configured stream limit | `PARALLEL=4` |
| Actual corpus input lengths | 5–99,892 tokens |
| Concurrent subset inputs | 597–2,387 tokens per request |
| Read-ahead and fill tuning | Upstream default read-ahead; no chunk overrides |
| Sampled minimum MemAvailable | 7.37 GiB/head, 9.78 GiB/worker |

Configured limits are not actual input lengths. Those memory minima describe
loading, this bounded corpus and vision checks; they are not a guarantee at a
fully occupied pool. Four independent 1M-token histories were not tested and
cannot fit in the shared 1M pool.

Hardware qualification used the staged image before the presentation-only
heredoc formatting update. All eight shell ASTs and parser-backed canonical
forms stayed equivalent; sixteen shell syntax checks passed, and an actual
BuildKit `--check` reported no warnings. Dockerfile byte hashing gives a rebuilt
formatted image a different tag. The formatted image was not separately served.

The original image, private configuration and prepared marker were restored after
qualification. Recorded swap usage was zero, container OOM flags false, and
selected new kernel fault-pattern checks passed through restoration. Memory
sampling was approximately two seconds and can miss short transients. Arbitrary
workloads, optional modes, Podman, sustained real-agent/thermal soak and outage
recovery are outside this build-refactor qualification. No runtime tuning or
inference-throughput change is proposed here.
