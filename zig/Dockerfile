# The engine image (both ranks): TensorFold's native (Zig) engine with the DeepSeek-V4.1-Flash family of BertholomusAI's
# fork, branch deepseek-v41-zig (TF_REPO at TF_REF), plus zig/patches/*.patch if there are any (git apply in the
# checkout's root, filename order). The inputs checkout below gets zig/patches-inputs/*.patch the same way (the
# mixed-pack rank-cache planner). It builds on NVIDIA's PyTorch container, which every Spark already holds: its CUDA
# toolchain compiles the sm_121 fatbins, its NCCL 2.30.7, cuBLAS and libibverbs are what the engine loads at run time,
# its PyTorch makes the kit's RoPE tables, and its Python with Pillow decodes the picture formats other than PNG (the
# engine starts a python3 child for them). No GPU is needed to build.
#
# The two-node branch writes a rank's weight file and Engram's token map only from its Python engine on a first start.
# So a second checkout, the fork's branch deepseek-v41-zig-tp4 at INPUTS_REF, is built only for what it adds, and only
# that is kept: tf-dsv41-rank-cache (the weight file from the checkpoint alone, no GPU, no Python; the same file format,
# which both engines read) and the standalone make_token_map.py and make_rope.py.
# Built by scripts/prepare.sh, never by hand: it passes the build arguments and a build context of this file and
# zig/patches only.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
# Optional ABLIT preparation uses the image's PyTorch and safetensors on CPU.
RUN python3 -m pip install --no-cache-dir safetensors
ARG ZIG_VERSION
ARG ZIG_SHA256
# The compiler from the first of ZIG_MIRRORS (community mirrors: <base>/zig-aarch64-linux-<version>.tar.xz) that answers
# with the pinned sha256, then ziglang.org itself as the last resort (the Zig project asks downloads to prefer mirrors, and
# ziglang.org stalled a build on the pair). Each source gets bounded tries (connect 20 s, 300 s a try, 3 retries, at
# most 600 s of retrying); a source that fails or serves other bytes is skipped. An empty ZIG_MIRRORS (scripts/config.sh
# passes what you set, empty included) means ziglang.org only. ZIG_MIRRORS is not part of the image hash: the bytes
# are pinned.
ARG ZIG_MIRRORS="https://zig.linus.dev/zig"
RUN ok=0; \
    for base in ${ZIG_MIRRORS} "https://ziglang.org/download/${ZIG_VERSION}"; do \
      url="${base%/}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz"; echo "zig: trying ${url}"; \
      if curl --connect-timeout 20 --max-time 300 --retry 3 --retry-delay 5 --retry-max-time 600 -fsSL -o /tmp/zig.tar.xz "${url}" && \
         echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c -; then ok=1; break; fi; \
      rm -f /tmp/zig.tar.xz; \
    done; \
    [ "$ok" = 1 ] || { echo "zig ${ZIG_VERSION}: no source served the pinned sha256" >&2; exit 1; }; \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG SM=121

# the serving engine
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY zig/patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Doptimize=ReleaseFast \
      --prefix /opt/tensorfold -j"$(nproc)" install native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && test -x /opt/tensorfold/bin/tf-dsv41-lanes && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold && \
    for f in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done && \
    if [ -e tools/dsv41-zig/ATTRIBUTION.md ]; then cp -a tools/dsv41-zig/ATTRIBUTION.md /opt/tensorfold/share/doc/tensorfold/DSV41-ATTRIBUTION.md; fi && \
    rm -rf /tensorfold /root/.cache/zig

# the inputs' tools (not the serving engine: nothing of this checkout but the four files below is kept)
ARG INPUTS_REF
RUN git init -q /tf-inputs && cd /tf-inputs && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${INPUTS_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${INPUTS_REF}"
COPY zig/patches-inputs /opt/tf-inputs-patches
RUN cd /tf-inputs && \
    for p in /opt/tf-inputs-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
RUN cd /tf-inputs && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Doptimize=ReleaseFast \
      --prefix /opt/inputs -j"$(nproc)" install && \
    test -x /opt/inputs/bin/tf-dsv41-rank-cache && \
    cp /opt/inputs/bin/tf-dsv41-rank-cache /opt/tensorfold/bin/ && \
    mkdir -p /opt/tensorfold/tools/inputs && \
    cp tools/dsv41-zig/inputs/make_token_map.py tools/dsv41-zig/kit/make_rope.py tools/dsv41-zig/kit/make_bias_vl.py \
       /opt/tensorfold/tools/inputs/ && \
    cp LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSE-inputs-branch && \
    rm -rf /tf-inputs /opt/inputs /root/.cache/zig

ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.engine=zig tf.ref=${TF_REF} tf.inputs=${INPUTS_REF}
ENV PATH=/opt/tensorfold/native/bin:/opt/tensorfold/bin:${PATH}
WORKDIR /workspace
