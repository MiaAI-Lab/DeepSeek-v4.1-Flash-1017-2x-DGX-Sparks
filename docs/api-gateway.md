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
rejects loopback, RFC 1918, ULA, link-local, multicast, reserved and unspecified destinations, including every
redirect and resolved address. The API's limits apply: `GATEWAY_MAX_IMAGES` (600), `GATEWAY_MAX_IMAGE_MB` (64) and
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

The comparator (`ds41-paridad`) and the CPU harness live in `gateway/` when task E6 is integrated. The intended CPU run
is: a fake token-in engine (`fake_serve`, `TF_FAKE_TOKENIZER=tok/`, `TENSORFOLD_TOKEN_ROUTE=1`, `TF_FAKE_CONTEXT=65536`),
then `ds41-gateway`, then

```sh
ds41-paridad casos --casos gateway/casos
ds41-paridad render --casos gateway/casos --zig http://127.0.0.1:ENGINE_PORT \
  --gateway http://127.0.0.1:PORT --tokenizer tok/tokenizer.json --salida informes/cpu
ds41-paridad e2e --gateway http://127.0.0.1:PORT
```

`render` compares the API path with the native route (first differing index, ids, classes and responsible file); `e2e`
checks the API's message, usage, stream, Messages, Responses and context-error invariants against the gateway. The CPU
report is saved as `informes/cpu/informe.{json,html}` and needs no GPU or model weights. A checkout without
`src/bin/ds41-paridad.rs` still builds `ds41-gateway`; the comparator is verified when E6 lands.
