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

## Initial measurements

A two-Spark ARM64 build experiment using an already-local NVIDIA26.10 base
reported 24,856,153,689 bytes for the runtime image, versus25,278,792,693 bytes
before this refactor: about423MB /1.67% smaller (Docker-reported sizes, not
compressed transfer sizes or inference memory). The large NVIDIA base limits
size savings.

Single samples: first build203s; normal rebuild after a comment-only build.zig
patch125s; returning to the unchanged context1s through layer caching. These
are exploratory cases, not a matched old/new Dockerfile benchmark or a promise
that arbitrary source edits rebuild at the same rate. No inference speedup is
claimed.

## Serving validation

The staged image was loaded on both Sparks and compared with the retained
NVIDIA26.10/Pillow13/Zig0.17.1 image using the same four-stream greedy corpus.
All31 cases matched prompt, server token and full text hashes/text, with no
request errors or failed internal checks. Inputs ranged5–99892 tokens,
covering cache reuse, interleaved conversations and concurrent/solo equality.
JPEG, WebP and GIF vision requests also returned the expected answer.

The run used the upstream1M context/shared pool and4 streams, default read-ahead
and no chunk overrides. Sampled MemAvailable minima were7.37GiB/head and
9.78GiB/worker, swap0. No new kernel fault-pattern matches or container OOM
state were observed across loading, tests and restoration. This is a bounded
hardware check; arbitrary workloads, optional modes, Podman and outage recovery
were not qualified by it. The original serving image/configuration was restored.
