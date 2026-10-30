# ds41-gateway: the DeepSeek API front end for TensorFold's token-in/token-out route. Built by scripts/prepare.sh
# with API_GATEWAY=1, never by hand; the build context is this folder only. No GPU, no Python, no OpenCV: the image
# carries the gateway and the parity comparator, and talks to the engine over HTTP on the host.
#
# The builder is Rust 1.97 (rust-toolchain.toml pins 1.97.1); the runtime is Debian slim with ca-certificates for
# image URLs over https. The image is the same on aarch64 and x86_64; cargo builds for the builder's architecture.
# Both base images are pinned by their multi-arch manifest digests (Docker Hub, resolved 2026-10-23).
FROM rust:1.97-bookworm@sha256:0e2bcaef56d041a486784e54104a81aebe0da44bd03019bd70bc0401e42e4a97 AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
RUN cargo fetch --locked
COPY src ./src
COPY tests ./tests
# ds41-paridad (the comparator) runs the battery from the image, so the case files travel with it.
COPY casos ./casos
RUN cargo build --release --locked --bin ds41-gateway --bin ds41-paridad

FROM debian:bookworm-slim@sha256:7ff6ba31527b9e336162fa061746c366c4b2032f4d2cff2078bea43f78efe3f8
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 10001 ds41 \
 && useradd --system --uid 10001 --gid ds41 --no-create-home --shell /usr/sbin/nologin ds41
COPY --from=builder /src/target/release/ds41-gateway /usr/local/bin/ds41-gateway
COPY --from=builder /src/target/release/ds41-paridad /usr/local/bin/ds41-paridad
COPY --from=builder /src/casos /casos
# The MIT notice of Sufyan's (@sfxnz) parity cases, which ship in /casos.
COPY NOTICE /usr/share/doc/ds41-gateway/NOTICE
ARG GATEWAY_HASH=unknown
LABEL tf.gateway=${GATEWAY_HASH}
USER ds41
ENTRYPOINT ["ds41-gateway"]
