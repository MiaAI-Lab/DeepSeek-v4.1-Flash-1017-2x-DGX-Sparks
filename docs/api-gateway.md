# The DeepSeek API gateway

`gateway/` builds `ds41-gateway`, a Rust front end that gives TensorFold's token-in/token-out route the DeepSeek API's
surface. It is off by default (`API_GATEWAY=0`); `API_GATEWAY=1` starts it on `HOST:PORT` and moves the engine to
`127.0.0.1:ENGINE_PORT`. The engine image is unchanged apart from `zig/patches/0035-token-route.patch`.

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

## Authentication and admission

With `API_KEY=1`, `start.sh` passes `--require-key --api-key-file /run/secrets/key` and sets `API_KEY_FILE` to that
read-only mount. The gateway loads the key once at startup and refuses to start if it is missing, unreadable or empty;
restart it after rotating the file. `--require-key` without a usable `API_KEY_FILE`/`--api-key-file` is likewise a startup
error, not an authentication bypass.

`/health` is public. Every other route, including the API, render, count-tokens, `/inference/*` and transparent engine
proxy routes, requires `Authorization: Bearer <key>` or `x-api-key: <key>`. Middleware checks the actual key with a
constant-time comparison before reading or parsing the body. A missing or wrong key therefore gets an API-shaped 401,
not a 400 for malformed JSON and not a proxy response from the engine. The key is never logged. `API_KEY=0` leaves the
API open, as the native recipe does by default.

`GATEWAY_MAX_CONCURRENT` (default 3) admits that many requests and queues the rest instead of rejecting them at the cap.
A permit covers the whole response body, including an SSE stream, and is released on completion or client cancellation.
The default comes from measured peaks of the release binary at the recipe's 1M-token context (three runs each, fresh
process; median and range). Encoding a prompt keeps memory for every byte of it and more for every token: about 120
bytes a byte for ordinary text, and up to about 360 at one token a byte (digits and punctuation such as `1,1,1`), the
most there can be. So one request encodes at a time (image fetches run outside that slot), and two checks come first.
A rendered prompt over `GATEWAY_MAX_PROMPT_MB` (8 MiB; a million tokens of ordinary English is about 7 MiB) is a 400.
A prompt over 1 MiB is then counted a piece at a time, with the encoder's own tokenizer, in pieces of up to 256 KiB cut
where the tokenizer itself splits (after a newline, at the end of a run of digits, before the space after a letter), so
that the count is exact there; a cut elsewhere is allowed 64 tokens. A prompt counted at more than twice the context is
a 400 without the encode; up to that it is encoded whole, and an overflow gets the API's exact context 400 with its
exact count.

| admitted requests | peak (idle: 57 MiB) |
| --- | --- |
| one 188,000-token prompt (0.7 MiB) | 206 MiB (205-207) |
| one 8 MiB prompt of ordinary text | 993 MiB (990-1,030) |
| one 64 MiB body of images | 536 MiB (536-536) |
| one 8 MiB prompt and 55 MiB of images | 1,247 MiB (1,231-1,247) |
| three 8 MiB prompts at once | 1,091 MiB (1,078-1,098) |
| three 64 MiB image bodies at once | 1,015 MiB (920-1,021) |
| three 8 MiB prompts with 55 MiB of images each, at once | 1,670 MiB (1,625-1,701) |
| one 2 MiB prompt at one token a byte, 2.08 million tokens (the most the count admits) | 808 MiB (806-808) |
| three of those with 55 MiB of images each, at once | 1,586 MiB (1,566-1,587) |
| one 8 MiB prompt at one token a byte, 8.2 million tokens, refused by the count | 249 MiB (237-252) |
| three of those with 55 MiB of images each, at once | 882 MiB (856-908) |
| a 64 MiB body of text, refused by the prompt limit before encoding | 441 MiB (441-441) |

These peaks are what the gateway holds to prepare a request: the measurement runs without an engine, so each request
ends in a 400 or a 502 after its body, JSON, encode and images. Generation adds the reply as it streams, a few bytes a
token. Three at once stay under the `3g` container default with room to spare. Without the token count, the 8 MiB prompt at
one token a byte took 2,926 MiB (2,918-2,935) alone and 3,991 MiB (3,980-4,011) three at once with images, over the
cap. Without the single encode slot the three ordinary prompts with images took 3,574 MiB (3,505-3,600), and before the
prompt limit a 64 MiB body of text was encoded whole: 6,537 MiB and 26 s only to be refused for its length. Each
further admitted request adds about 200 MiB (its body, JSON and images). The count costs time only for a prompt over
1 MiB: the 2 MiB prompt above takes 6.8 s to count and encode, the refused 8 MiB one 3.6 s.

The body of an admitted request is read by the same middleware, within `GATEWAY_BODY_TIMEOUT` (60 s): a client that has
not sent its whole body by then gets a 408 and its permit goes back. A `content-length` over 64 MiB is a 413 before
anything is read; a chunked body is a 413 when it crosses 64 MiB. Frames are copied into one buffer as they arrive, and
the buffer never grows past 64 MiB (while it grows, old and new allocations coexist for a moment, under 128 MiB), so a
body split into tiny chunks costs no more than a whole one. A body holding more than 1,048,576 JSON values (counted in
one pass over its bytes, object keys included) is a 400 before it is parsed: a parsed value takes 32 bytes or more, so a
64 MiB body of one-digit array elements would otherwise parse into gigabytes. Real requests put their weight in strings
and hold far fewer values. The parsed body is not cloned on the way to the encoder. Connections: `GATEWAY_MAX_CONNECTIONS` (256) open at most,
`GATEWAY_MAX_CONNECTIONS_PER_PEER` (32) from one address (more are closed at once, so one client cannot take every
slot; behind a reverse proxy every client has the proxy's address, so raise it to `GATEWAY_MAX_CONNECTIONS` there), and `GATEWAY_HEADER_TIMEOUT` (30 s) for a request header, on a new connection and between keep-alive requests, so
a silent, slow or idle connection gives its slot back. A reply that has not moved for `GATEWAY_WRITE_TIMEOUT` (60 s)
because the client stopped reading it (the kernel's buffers are full) closes its connection, and the response's permit
goes back with it. The time counts from the last write that went through, as nginx's `send_timeout` does: a client
that keeps reading, however slowly, keeps its reply. Queued requests wait for admission without a deadline; each
running one ends, or loses its client to these deadlines. The gateway speaks HTTP/1.1.

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
broadcast address; for IPv6: everything outside global unicast `2000::/3`, and within it `2001::/23`,
`2001:db8::/32`, `3fff::/20`; IPv4-mapped (`::ffff:0:0/96`), 6to4 (`2002::/16`) and NAT64 (`64:ff9b::/96`) addresses
are checked through their embedded IPv4 address, and `64:ff9b:1::/48` is rejected. These are the loopback, private,
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
- Patch 0035 serves `logprobs` / `top_logprobs` on the token route. They fall back only if the running engine's health
  reports `logprobs: false` (for example, an engine without that patch).
- The image token count is the API's own estimate; the engine's prompt length can differ by the tokenizer's special
  tokens. The comparator reports both.
- `n > 1` is not converted; it falls back.
- The gateway does not retry a failed engine call and does not drain requests on stop: `stop.sh` stops it first.
- No Python is in the serving path. The Python in `scripts/` is for the recipe's checks and tests.

## Running it

With the recipe: `API_GATEWAY=1` in `scripts/local.sh`, then `./start.sh` (or `./start.sh restart`). `scripts/prepare.sh`
builds `GATEWAY_IMAGE` from `gateway/Dockerfile`; `start.sh` starts it after the engine answers, waits for `/health` and
runs both smoke tests. `./stop.sh` removes the gateway first and saves its log. The gateway's container carries a
digest of its settings (`tf.recipe.gateway`): with the ranks already up, `./start.sh` starts a stopped gateway again,
and asks for `./start.sh restart` when the gateway is gone or was started with other settings. With `FOREGROUND=1` a
gateway that exits is started again; if that fails the server stops with a failure, so the supervisor restarts it.

By hand, from `gateway/`:

```sh
cargo build --release --locked --bin ds41-gateway
./target/release/ds41-gateway --listen 127.0.0.1:8888 --engine http://127.0.0.1:8887 \
  --tokenizer /served/tokenizer.json --served-name DeepSeek-V4.1-Flash-1017
```

The flags and their defaults are in `gateway/src/main.rs` and `gateway/README.md`. The container runs as an unprivileged
user, read-only, with no capabilities (`--cap-drop ALL`, `--security-opt no-new-privileges`), `--memory` and `--cpus`
from `GATEWAY_MEMORY` / `GATEWAY_CPUS`, and no GPU. With `API_KEY=1` it runs as the uid and gid of the user who runs
`start.sh`, who owns the mode-600 key file it mounts, and `start.sh` refuses to do that as root; otherwise it runs as the
image's unprivileged `ds41` user (uid 10001). Docker's `userns-remap` is not supported with a key: the uid inside is not
the host's, and the key stays unreadable. The memory default is `3g`, sized for three admitted requests as above.

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
