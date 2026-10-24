# The DeepSeek API gateway

`gateway/` builds `ds41-gateway`, a Rust front end that gives TensorFold's token-in/token-out route the DeepSeek API's
surface. It is off by default (`API_GATEWAY=0`); `API_GATEWAY=1` starts it on `HOST:PORT` and moves the engine to
`127.0.0.1:ENGINE_PORT`. The engine image is unchanged apart from `zig/patches/0011-token-route.patch`.

## Routes

| Route | Who answers | How |
| --- | --- | --- |
| `POST /v1/chat/completions`, `/v1/messages`, `/v1/responses` | the gateway or the engine | convert, render, encode, call `POST /inference/v1/generate`, stream the tokens back as the API's body or SSE |
| `POST /v1/messages/count_tokens` | the gateway | convert and encode; `{"input_tokens": N}`; conversion errors use Anthropic's error shape |
| `POST /ds41/render/{chat,messages,responses}` | the gateway | the rendered prompt, ids, image count, thinking, effort, sampling, limits and `fallback_reason`, without generating |
| `GET /health` | the gateway | its own status, the engine's status and the engine's `/health`; 200 only while the engine is 200 |
| `/inference/*` | the gateway | never exposed: 404 `{"detail": "Not Found"}` |
| everything else (`/v1/models`, `/tokenize`, `/detokenize`, `/metrics`, `/v1/completions`, ...) | the engine | transparent proxy: method, path, query and headers except hop-by-hop ones; the body as it is |

Responses from the API path carry `x-ds41-gateway: api`; fallback responses carry `x-ds41-gateway: fallback`. Non-streaming
API responses also carry `x-ds41-token-sha` (the ids the gateway delivered) and `x-ds41-token-sha-raw` (the ids the engine
emitted, including a trailing EOS), which the comparator uses against the native route's `tensorfold.token_sha`.

## Fallback and strict mode

A request goes to the engine unchanged when the API conversion cannot represent it faithfully, or when it uses a field
only the Zig server knows. The gateway logs one line with the reason and marks the response. Examples: a conversion error
(unknown model, invalid role, `n != 1`, required/named `tool_choice` with thinking, an invalid strict schema), a
`chat_template_kwargs` key other than `thinking`/`enable_thinking`/`reasoning_effort`, `prompt`, `draft`,
`thinking_budget`, `priority`, `return_token_ids`, `guided_json`/`guided_regex`/`guided_choice`/`grammar`,
`structured_outputs`, a `response_format` the API does not know, `logprobs`/`top_logprobs` while the engine reports
`logprobs: false`, `previous_response_id`/`store`/`conversation`, `modalities`/`audio`/`video_url`. `GATEWAY_STRICT=1`
returns the API's 400 instead.

The API path is used, on purpose, for fields the Zig server handles differently: `prefix`, `thinking: {type}`,
`minimal`/`medium`/`xhigh`/`ultra` efforts, `stop` (content only), the API's default `max_tokens`, `response_format`
`json_object`/`json_schema`/`regex` (written into the prompt, without grammar), `tool_choice: none`, and
`parallel_tool_calls` (the API ignores it).

## Images

`image_url` data URLs and http(s) URLs, `input_image` and Messages image blocks are decoded to dimensions only (never to
pixels) and counted with DeepSeek's `ImageTokenSpec::v41`. The destination filter (`GATEWAY_BLOCK_PRIVATE_URLS=1`)
rejects, for IPv4: `0.0.0.0/8`, `10.0.0.0/8`, `100.64.0.0/10`, `127.0.0.0/8`, `169.254.0.0/16`, `172.16.0.0/12`,
`192.0.0.0/24`, `192.0.2.0/24`, `192.31.196.0/24`, `192.52.193.0/24`, `192.88.99.0/24`, `192.168.0.0/16`,
`192.175.48.0/24`, `198.18.0.0/15`, `198.51.100.0/24`, `203.0.113.0/24`, `224.0.0.0/4`, `240.0.0.0/4` and the
broadcast address; for IPv6: `::/128`, `::1/128`, `::ffff:0:0/96`, `64:ff9b::/96`, `64:ff9b:1::/48`, `100::/64`,
`2001::/23`, `2001:db8::/32`, `2002::/16`, `3fff::/20`, `fc00::/7`, `fe80::/10`, `fec0::/10` and `ff00::/8`.
IPv4-mapped and 6to4 addresses are checked through their embedded IPv4 address. These are the loopback, private,
carrier-grade NAT, link-local, multicast, broadcast, reserved, documentation, benchmarking, discard, translation and
tunnelling ranges, and every redirect and resolved address is checked, so an image URL cannot make the gateway fetch a
private or non-routable destination. The API's limits apply: `GATEWAY_MAX_IMAGES` (600), `GATEWAY_MAX_IMAGE_MB` (64) and
8192-character URLs; PNG, JPEG, WebP and GIF are accepted.
The served tokenizer's image-token spelling is given by `GATEWAY_IMAGE_TOKEN`; the default is
`<｜deepseek_image｜>`.

## Known limits

- A named or required `tool_choice` with thinking falls back: the API's conversion cannot place the tool choice in the
  thinking prompt faithfully.
- Grammar-constrained decoding (`guided_json`, `guided_regex`, `guided_choice`, `grammar`, `structured_outputs`) falls
  back; `json_object`/`json_schema`/`regex` are written into the prompt without a grammar.
- `logprobs` / `top_logprobs` fall back until the engine serves logprobs on the token route.
- The image token count is the API's own estimate; the engine's prompt length can differ by the tokenizer's special
  tokens. The comparator reports both.
- `n > 1` is not converted; it falls back.
- The gateway does not retry a failed engine call and does not drain requests on stop: `stop.sh` stops it first.
- No Python is in the serving path. The Python in `scripts/` is for the recipe's checks and tests.

## Running it

With the recipe: `API_GATEWAY=1` in `scripts/local.sh`, then `./start.sh` (or `./start.sh restart`). `scripts/prepare.sh`
builds `GATEWAY_IMAGE` from `gateway/Dockerfile`; `start.sh` starts it after the engine answers, waits for `/health` and
runs both smoke tests. `./stop.sh` removes the gateway first and saves its log.

By hand, from `gateway/`:

```sh
cargo build --release --locked --bin ds41-gateway
./target/release/ds41-gateway --listen 127.0.0.1:8888 --engine http://127.0.0.1:8887 \
  --tokenizer /served/tokenizer.json --served-name DeepSeek-V4.1-Flash-1017
```

The flags and their defaults are in `gateway/src/main.rs` and `gateway/README.md`. The container runs as an unprivileged
user, read-only, with `--memory` and `--cpus` from `GATEWAY_MEMORY` / `GATEWAY_CPUS`, and no GPU.

## Parity battery without a GPU

The comparator (`ds41-paridad`) builds from `gateway/` and needs no model weights. The CPU harness is a fake token-in
engine: the TensorFold checkout's `fake_serve` (`zig/tests/server/build.sh`, then
`zig-out/server/fake_serve serve zig/tests/server/fixtures`) with `TF_FAKE_TOKENIZER=tok/`,
`TENSORFOLD_TOKEN_ROUTE=1` and `TF_FAKE_CONTEXT=65536`. Start it on `ENGINE_PORT`, start `ds41-gateway` against it, then

```sh
ds41-paridad casos --casos gateway/casos
ds41-paridad render --casos gateway/casos --zig http://127.0.0.1:ENGINE_PORT \
  --gateway http://127.0.0.1:PORT --tokenizer tok/tokenizer.json --salida informes/cpu
ds41-paridad e2e --casos gateway/casos --gateway http://127.0.0.1:PORT \
  --zig http://127.0.0.1:ENGINE_PORT --tokenizer tok/tokenizer.json --salida informes/cpu
```

`casos` validates the case files; `render` compares the API path with the native `/tokenize` route and reports the first
differing index, the ids, the class and the responsible file; `e2e` checks the message, usage, stream, Messages,
Responses and context-error invariants and the token hashes. Each run writes `informes/cpu/informe.json` and
`informes/cpu/informe.html`; open the HTML for the outcome table. The exit code is 0 when everything passes, 1 for
divergences and 2 for a setup error. On the Spark pair the same three commands run against
`--gateway http://127.0.0.1:8888 --zig http://127.0.0.1:8887` from the gateway container (no fake engine).
