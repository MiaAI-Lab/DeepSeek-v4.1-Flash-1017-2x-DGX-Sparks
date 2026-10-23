# ds41-gateway: the DeepSeek API front end for TensorFold's token-in/token-out route. Built by scripts/prepare.sh
# with API_GATEWAY=1, never by hand; the build context is this folder only. No GPU, no Python, no OpenCV: the image
# carries the gateway and the parity comparator, and talks to the engine over HTTP on the host.
#
# The builder is Rust 1.97 (rust-toolchain.toml pins 1.97.1); the runtime is Debian slim with ca-certificates for
# image URLs over https. The image is the same on aarch64 and x86_64; cargo builds for the builder's architecture.
FROM rust:1.97-bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
RUN cargo fetch --locked
COPY src ./src
COPY tests ./tests
# ds41-paridad is the comparator; it is named here even while a gateway checkout without it is being integrated.
RUN cargo build --release --locked --bin ds41-gateway --bin ds41-paridad

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 10001 ds41 \
 && useradd --system --uid 10001 --gid ds41 --no-create-home --shell /usr/sbin/nologin ds41
COPY --from=builder /src/target/release/ds41-gateway /usr/local/bin/ds41-gateway
COPY --from=builder /src/target/release/ds41-paridad /usr/local/bin/ds41-paridad
ARG GATEWAY_HASH=unknown
LABEL tf.gateway=${GATEWAY_HASH}
USER ds41
ENTRYPOINT ["ds41-gateway"]
