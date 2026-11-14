# syntax=docker/dockerfile:1
# The engine image (both ranks): TensorFold's native (Zig) engine with the DeepSeek-V4.1-Flash family of BertholomusAI's
# fork, branch deepseek-v41-zig (TF_REPO at TF_REF), plus zig/patches/*.patch if there are any (git apply in the
# checkout's root, filename order). The inputs checkout below gets zig/patches-inputs/*.patch the same way (the
# mixed-pack rank-cache planner). It builds on NVIDIA's PyTorch container, which every Spark already holds: its CUDA
# toolchain compiles the sm_121 fatbins, its NCCL, cuBLAS and libibverbs are what the engine loads at run time,
# its PyTorch makes the kit's RoPE tables, and its Python with Pillow decodes the picture formats other than PNG (the
# engine starts a python3 child for them). No GPU is needed to build.
#
# The two-node branch writes a rank's weight file and Engram's token map only from its Python engine on a first start.
# So a second checkout, the fork's branch deepseek-v41-zig-tp4 at INPUTS_REF, is built only for what it adds, and only
# that is kept: tf-dsv41-rank-cache (the weight file from the checkpoint alone, no GPU, no Python; the same file format,
# which both engines read), three standalone Python tools, and its license.
# Built by scripts/prepare.sh, never by hand: it passes the build arguments and a build context of this file and
# zig/patches plus zig/patches-inputs only.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.10-py3@sha256:6d347d649d5dd108b4652f1b9cea6a3c859692ac408617a9108616da40e3da1f
FROM ${BASE_IMAGE} AS base

FROM base AS compiler
ARG ZIG_VERSION
ARG ZIG_SHA256
# The compiler from the first of ZIG_MIRRORS (community mirrors: <base>/zig-aarch64-linux-<version>.tar.xz) that answers
# with the pinned sha256, then ziglang.org itself as the last resort (the Zig project asks downloads to prefer mirrors, and
# ziglang.org stalled a build on the pair). Each source gets bounded tries (connect 20 s, 300 s a try, 3 retries, at
# most 600 s of retrying); a source that fails or serves other bytes is skipped. An empty ZIG_MIRRORS (scripts/config.sh
# passes what you set, empty included) means ziglang.org only. ZIG_MIRRORS is not part of the image hash: the bytes
# are pinned.
ARG ZIG_MIRRORS="https://zig.linus.dev/zig"
RUN <<'EOF'
set -eu
ok=0
for base in ${ZIG_MIRRORS} "https://ziglang.org/download/${ZIG_VERSION}"; do
    url="${base%/}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz"
    echo "zig: trying ${url}"
    if curl --connect-timeout 20 --max-time 300 --retry 3 --retry-delay 5 --retry-max-time 600 -fsSL -o /tmp/zig.tar.xz "${url}" &&
       echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c -; then
        ok=1
        break
    fi
    rm -f /tmp/zig.tar.xz
done
[ "$ok" = 1 ] || { echo "zig ${ZIG_VERSION}: no source served the pinned sha256" >&2; exit 1; }
mkdir -p /opt/zig
tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1
rm /tmp/zig.tar.xz
/opt/zig/zig version
EOF
ARG SM=121

# Cache mounts accelerate builds after source or patch changes. A --no-cache
# build may reset them; caches are disposable, never required image contents.
FROM compiler AS engine-builder
ARG TF_REPO
ARG TF_REF
RUN <<'EOF'
set -eu
git init -q /tensorfold
cd /tensorfold
git remote add origin "${TF_REPO}"
git fetch -q --depth 1 origin "${TF_REF}"
git checkout -q --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "${TF_REF}"
EOF
COPY zig/patches /opt/tf-patches
RUN <<'EOF'
set -eu
cd /tensorfold
for p in /opt/tf-patches/*.patch; do
    [ -e "$p" ] || continue
    echo "applying $p"
    git apply --check "$p"
    git apply "$p"
done
EOF
# Align the checkout's strict compiler pin only for the verified patch release.
# Global Zig artifacts are shared; each checkout has its own local cache. Locked
# mounts also protect them when separate image builds use this builder together.
RUN --mount=type=cache,id=tf-zig-global-${ZIG_VERSION}-${SM},target=/root/.cache/zig,sharing=locked \
    --mount=type=cache,id=tf-zig-engine-${ZIG_VERSION}-${SM},target=/tensorfold/.zig-cache,sharing=locked <<'EOF'
set -eu
cd /tensorfold
if [ "$ZIG_VERSION" = 0.17.1 ]; then
    test "$(cat .zig-version)" = 0.17.0
    printf '0.17.1\n' > .zig-version
fi
/opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Doptimize=ReleaseFast \
    --prefix /opt/tensorfold -j"$(nproc)" install native
test -x /opt/tensorfold/native/bin/tensorfold-native
test -x /opt/tensorfold/bin/tf-dsv41-lanes
mkdir -p /opt/tensorfold/share/doc/tensorfold
for f in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do
    if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/; fi
done
if [ -e tools/dsv41-zig/ATTRIBUTION.md ]; then
    cp -a tools/dsv41-zig/ATTRIBUTION.md /opt/tensorfold/share/doc/tensorfold/DSV41-ATTRIBUTION.md
fi
EOF

# Deriving from engine-builder serializes the two all-core Zig builds on a Spark.
# Nothing from this checkout but the rank-cache tool, three Python tools and license is kept.
FROM engine-builder AS inputs-builder
ARG INPUTS_REF
RUN <<'EOF'
set -eu
git init -q /tf-inputs
cd /tf-inputs
git remote add origin "${TF_REPO}"
git fetch -q --depth 1 origin "${INPUTS_REF}"
git checkout -q --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "${INPUTS_REF}"
EOF
COPY zig/patches-inputs /opt/tf-inputs-patches
RUN <<'EOF'
set -eu
cd /tf-inputs
for p in /opt/tf-inputs-patches/*.patch; do
    [ -e "$p" ] || continue
    echo "applying $p"
    git apply --check "$p"
    git apply "$p"
done
EOF
# Align the checkout's strict compiler pin only for the verified patch release.
RUN --mount=type=cache,id=tf-zig-global-${ZIG_VERSION}-${SM},target=/root/.cache/zig,sharing=locked \
    --mount=type=cache,id=tf-zig-inputs-${ZIG_VERSION}-${SM},target=/tf-inputs/.zig-cache,sharing=locked <<'EOF'
set -eu
cd /tf-inputs
if [ "$ZIG_VERSION" = 0.17.1 ]; then
    test "$(cat .zig-version)" = 0.17.0
    printf '0.17.1\n' > .zig-version
fi
/opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Doptimize=ReleaseFast \
    --prefix /opt/inputs -j"$(nproc)" install
test -x /opt/inputs/bin/tf-dsv41-rank-cache
cp /opt/inputs/bin/tf-dsv41-rank-cache /opt/tensorfold/bin/
mkdir -p /opt/tensorfold/tools/inputs
cp tools/dsv41-zig/inputs/make_token_map.py tools/dsv41-zig/kit/make_rope.py tools/dsv41-zig/kit/make_bias_vl.py \
    /opt/tensorfold/tools/inputs/
cp LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSE-inputs-branch
EOF

# Copy the complete install tree; Zig, checkouts, patches and build caches stay in builders.
# Inherit NVIDIA's runtime environment, entrypoint and command from the same pinned base.
FROM base AS runtime
COPY --from=inputs-builder /opt/tensorfold /opt/tensorfold
# Keep Python dependency updates independent of the native build stages.
# Optional ABLIT preparation uses the image's PyTorch and safetensors on CPU.
# Non-PNG vision input is decoded by an isolated Python child.
RUN <<'EOF'
set -eu
python3 -m pip install --no-cache-dir safetensors==0.8.0
python3 -m pip install --no-cache-dir --only-binary=:all: Pillow==13.0.0
EOF
ARG TF_REF
ARG INPUTS_REF
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.engine=zig tf.ref=${TF_REF} tf.inputs=${INPUTS_REF}
ENV PATH=/opt/tensorfold/native/bin:/opt/tensorfold/bin:${PATH}
WORKDIR /workspace
