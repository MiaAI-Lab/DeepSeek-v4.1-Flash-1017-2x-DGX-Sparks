# ds41-gateway

The DeepSeek API front end for TensorFold's token-in/token-out route. It is a Rust service (`src/main.rs`) on top of
DeepSeek's `deepseek-recipe` 0.2.0 and the vLLM token-in contract; the recipe starts it with `API_GATEWAY=1` and the
engine on loopback. It is off by default and has no GPU, Python or OpenCV in its image.

## Build

Rust 1.97.1 is pinned in `rust-toolchain.toml`. On a host with a C toolchain:

```sh
cargo build --release --locked --bin ds41-gateway --bin ds41-paridad
```

The tests that need a tokenizer read `DS41_TOKENIZER` (a `tokenizer.json` path); without it they fall back to
`../tok/tokenizer.json` next to this folder, so `cargo test --locked` needs either the variable or a `tok/` checkout.

`scripts/prepare.sh` builds the container image from `Dockerfile` (multi-stage `rust:1.97-bookworm` to
`debian:bookworm-slim`, unprivileged user, `ca-certificates`, both binaries) with the tag `GATEWAY_IMAGE`, which carries
a hash of this folder. The Dockerfile builds `ds41-gateway` and `ds41-paridad` and copies `casos/`, so the parity battery
runs from the image too.

## Run with the recipe (GPU)

Put `API_GATEWAY=1` in `scripts/local.sh` and run `./start.sh` (or `./start.sh restart`). The recipe builds the gateway
image if needed, starts the engine on `127.0.0.1:ENGINE_PORT` (default 8887) with `TENSORFOLD_TOKEN_ROUTE=1`, starts
`ds41-gateway` on `HOST:PORT`, waits for `/health`, and runs the smoke tests. `./stop.sh` removes the gateway first and
saves its log. `API_GATEWAY=0` is the recipe as before.

## Run without a GPU

The gateway only needs an engine that speaks `POST /inference/v1/generate`. For a CPU check, start a fake token-in
engine and then the gateway; the fake engine is the TensorFold checkout's `fake_serve` (`zig/tests/server/build.sh`,
then `zig-out/server/fake_serve serve zig/tests/server/fixtures`):

```sh
TF_FAKE_TOKENIZER=tok/ TENSORFOLD_TOKEN_ROUTE=1 TF_FAKE_CONTEXT=65536 zig-out/server/fake_serve serve zig/tests/server/fixtures &
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
| `GATEWAY_IMAGE_TOKEN` | `--image-token` | `<｜deepseek_image｜>` |
| `GATEWAY_STRICT` | `--strict` | 0 |
| `GATEWAY_BLOCK_PRIVATE_URLS` | `--block-private-urls` | 1 |
| `GATEWAY_API_SAMPLING` | `--api-sampling` | 1 |
| `GATEWAY_MAX_IMAGES`, `GATEWAY_MAX_IMAGE_MB` | `--max-images`, `--max-image-mb` | 600, 64 |
| `GATEWAY_LOG` | `--log` | info |
| `GATEWAY_MAX_CONCURRENT` | `--max-concurrent` | 3 |
| `GATEWAY_MAX_CONNECTIONS` | `--max-connections` | 256 |
| `GATEWAY_MAX_CONNECTIONS_PER_PEER` | `--max-connections-per-peer` | 32 |
| `GATEWAY_HEADER_TIMEOUT` | `--header-timeout` (seconds) | 30 |
| `GATEWAY_BODY_TIMEOUT` | `--body-timeout` (seconds) | 60 |
| `GATEWAY_WRITE_TIMEOUT` | `--write-timeout` (seconds) | 60 |
| `GATEWAY_MAX_PROMPT_MB` | `--max-prompt-mb` | 8 |
| `API_KEY` | `--require-key` | 0 |

With `API_KEY=1` the gateway loads the keys from `API_KEY_FILE` once at startup (one key, or `label: key`, a line;
an empty or unreadable file stops it) and checks `Authorization: Bearer <key>` or `x-api-key: <key>` itself, in constant
time, before reading the body, on every route except `GET /health`; a missing or wrong key is the API's 401. An
`API_KEY` that is neither 1/0, true/false nor yes/no stops it, rather than leaving the API open. The recipe mounts the
key at `/run/secrets/key`, passes `--require-key` and runs the container as the uid:gid of the user who runs `start.sh`,
the owner of the mode-600 key file (never root: `start.sh` refuses). A key change needs a restart.

## Images

`image_url`, `input_image` and Messages image blocks are fetched by the gateway and counted with DeepSeek's
`ImageTokenSpec::v41`; only their dimensions are decoded, never their pixels. With `GATEWAY_BLOCK_PRIVATE_URLS=1` (the
default) the destination filter rejects, for IPv4: `0.0.0.0/8`, `10.0.0.0/8`, `100.64.0.0/10`, `127.0.0.0/8`,
`169.254.0.0/16`, `172.16.0.0/12`, `192.0.0.0/24`, `192.0.2.0/24`, `192.31.196.0/24`, `192.52.193.0/24`,
`192.88.99.0/24`, `192.168.0.0/16`, `192.175.48.0/24`, `198.18.0.0/15`, `198.51.100.0/24`, `203.0.113.0/24`,
`224.0.0.0/4`, `240.0.0.0/4` and the broadcast address; for IPv6: everything outside global unicast
`2000::/3`, and within it `2001::/23`, `2001:db8::/32`, `3fff::/20`; IPv4-mapped (`::ffff:0:0/96`), 6to4 (`2002::/16`)
and NAT64 (`64:ff9b::/96`) addresses are checked through their embedded IPv4 address, and `64:ff9b:1::/48` is
rejected. These are the loopback, private, carrier-grade NAT, link-local, multicast, broadcast, reserved, documentation,
benchmarking, discard, translation and tunnelling ranges. The fetch uses the gateway's own HTTP client (the image crate
only streams the body under the size limits): environment proxies are ignored, the resolver hands the client only the
addresses the filter allows, so a second DNS answer cannot swap in another one, an address spelled in the URL (decimal,
hexadecimal, IPv4-mapped) is checked as the address it parses to, and every redirect goes through the same checks, at
most five times. So an image URL cannot make the gateway fetch a private or non-routable destination; the tests in
`src/images.rs` run that production client against a loopback server, with a positive control. `GATEWAY_MAX_IMAGES`
(600), `GATEWAY_MAX_IMAGE_MB` (64) and 8192-character URLs still apply.

## Parity battery

`ds41-paridad` compares the gateway's API rendering with TensorFold's token route and checks the API's shape against a
scripted engine. It needs no model weights:

```sh
ds41-paridad casos --casos casos
ds41-paridad render --casos casos --zig http://127.0.0.1:ENGINE_PORT \
  --gateway http://127.0.0.1:PORT --tokenizer tok/tokenizer.json --salida informes/cpu
ds41-paridad e2e --casos casos --gateway http://127.0.0.1:PORT \
  --zig http://127.0.0.1:ENGINE_PORT --tokenizer tok/tokenizer.json --salida informes/cpu
```

`casos` validates the case files (ids, fields, expectations); `render` compares the API path with the native
`/tokenize` route and reports the first differing index, the ids, the class and the responsible file; `e2e` checks the
message, usage, stream, Messages, Responses and context-error invariants and the token hashes. Each run writes
`informes/cpu/informe.json` and `informes/cpu/informe.html`; open the HTML for the outcome table. The exit code is 0 when
everything passes, 1 for divergences and 2 for a setup error. On the Spark pair, run the same three commands against
`--gateway http://127.0.0.1:8888 --zig http://127.0.0.1:8887` from the gateway container instead of `fake_serve`. The
contract is summarized in `docs/api-gateway.md` (the recipe ships the same page under its `docs/`).
