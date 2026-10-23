# ds41-gateway

The DeepSeek API front end for TensorFold's token-in/token-out route. It is a Rust service (`src/main.rs`) on top of
DeepSeek's `deepseek-recipe` 0.2.0 and the vLLM token-in contract; the recipe starts it with `API_GATEWAY=1` and the
engine on loopback. It is off by default and has no GPU, Python or OpenCV in its image.

## Build

Rust 1.97.1 is pinned in `rust-toolchain.toml`. On a host with a C toolchain:

```sh
cargo build --release --locked --bin ds41-gateway
```

`scripts/prepare.sh` builds the container image from `Dockerfile` (multi-stage `rust:1.97-bookworm` to
`debian:bookworm-slim`, unprivileged user, `ca-certificates`, both binaries) with the tag `GATEWAY_IMAGE`, which carries
a hash of this folder. The Dockerfile also names `ds41-paridad`; a checkout without `src/bin/ds41-paridad.rs` builds the
gateway alone and the comparator is verified when task E6 is integrated.

## Run with the recipe (GPU)

Put `API_GATEWAY=1` in `scripts/local.sh` and run `./start.sh` (or `./start.sh restart`). The recipe builds the gateway
image if needed, starts the engine on `127.0.0.1:ENGINE_PORT` (default 8887) with `TENSORFOLD_TOKEN_ROUTE=1`, starts
`ds41-gateway` on `HOST:PORT`, waits for `/health`, and runs the smoke tests. `./stop.sh` removes the gateway first and
saves its log. `API_GATEWAY=0` is the recipe as before.

## Run without a GPU

The gateway only needs an engine that speaks `POST /inference/v1/generate`. For a CPU check, start a fake engine and
then the gateway (the fake harness lives in `tests/`; `fake_serve` is added with the comparator when E6 is integrated):

```sh
TF_FAKE_TOKENIZER=tok/ TENSORFOLD_TOKEN_ROUTE=1 TF_FAKE_CONTEXT=65536 ./target/release/fake_serve --port 8887 &
./target/release/ds41-gateway --listen 127.0.0.1:8888 --engine http://127.0.0.1:8887 \
  --tokenizer tok/tokenizer.json --served-name DeepSeek-V4.1-Flash-1017
curl -s http://127.0.0.1:8888/health
```

The gateway serves `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/messages/count_tokens` and
`/ds41/render/*`; other routes are proxied to the engine. `/inference/*` is not exposed. Responses from the API path
carry `x-ds41-gateway: api`; fallback responses carry `x-ds41-gateway: fallback` and one log line with the reason.
`GATEWAY_STRICT=1` returns the API's 400 instead of falling back.

## Settings

The defaults are in `scripts/config.sh`; the flags are in `src/main.rs`. The important ones:

| Setting | Flag | Default |
| --- | --- | --- |
| `HOST`, `PORT` | `--listen` | `0.0.0.0:8888` |
| `ENGINE_PORT` | `--engine` | `http://127.0.0.1:8887` |
| `SERVED_NAME` | `--served-name` | `DeepSeek-V4.1-Flash-1017` |
| `THINKING`, `REASONING_EFFORT` | `--default-thinking`, `--default-effort` | 1, empty |
| `CONTEXT` | `--max-model-len` | auto from the engine, 65536 if it does not answer |
| `GATEWAY_IMAGE_TOKEN` | `--image-token` | `<｜end_of_query｜>` |
| `GATEWAY_STRICT` | `--strict` | 0 |
| `GATEWAY_BLOCK_PRIVATE_URLS` | `--block-private-urls` | 1 |
| `GATEWAY_API_SAMPLING` | `--api-sampling` | 1 |
| `GATEWAY_MAX_IMAGES`, `GATEWAY_MAX_IMAGE_MB` | `--max-images`, `--max-image-mb` | 600, 64 |
| `GATEWAY_LOG` | `--log` | info |
| `API_KEY` | `--require-key` | 0 |

With `API_KEY=1` the recipe mounts the key at `/run/secrets/key` and passes `--require-key`; the gateway forwards
`Authorization` / `x-api-key` to the engine and lets it decide.

## Parity battery

When `src/bin/ds41-paridad.rs` is present (task E6), the comparator runs against the gateway and the engine:

```sh
ds41-paridad casos --casos casos
ds41-paridad render --casos casos --zig http://127.0.0.1:ENGINE_PORT \
  --gateway http://127.0.0.1:PORT --tokenizer tok/tokenizer.json --salida informes/cpu
ds41-paridad e2e --gateway http://127.0.0.1:PORT
```

`casos` validates the case files; `render` compares the API path with the native route and reports the first differing
index, ids and the responsible class/file; `e2e` checks the message, usage, stream, Messages, Responses and context-error
invariants. The CPU run uses `fake_serve` and no model weights; the report is `informes/cpu/informe.{json,html}`. The
contract is summarized in `../docs/api-gateway.md`.
