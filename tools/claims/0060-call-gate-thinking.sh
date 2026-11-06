#!/usr/bin/env bash
# Written by markyu0401 for zig/patches/0060-call-gate-thinking.patch (issue #80).
# Apache-2.0. The patch's promise, as assertions against a running server.
# Usage: tools/claims/0060-call-gate-thinking.sh API MODEL. Exit 0: the claim holds. 1: it does not. 2: the script could not judge.
# Exits 1 on an image built before zig/patches/0060-call-gate-thinking.patch.
# claim: a thinking request does not take the call gate, so `tool_choice: "required"` on one runs as `auto`:
#   - its first token is the one `auto` returns, and does not climb across requests (before the patch it is the
#     backend's per-request draw handle: 506, 607, ... 1109 over 600 requests, independent of the prompt);
#   - a 400-token `required` reply equals the `auto` reply (same `token_sha`, content and `finish_reason`);
#   - the gate still holds for a reply that starts outside a think block: `required` with
#     `chat_template_kwargs: {"enable_thinking": false}` returns a tool call, with the same first token every time.
# `auto` and the non-thinking paths are the controls: if they fail, the build cannot be judged on this claim.
# Needs `return_token_ids` (the reply's `tensorfold.token_ids`). Not run against an image yet: the patch's own
# measurements on a pair (PR #81, image tensorfold-dsv41:exl3-tp2-b8825cd47726) are the manual equivalent.
API=$1; MODEL=$2
[ -n "$API" ] && [ -n "$MODEL" ] || { echo "usage: claim.sh API MODEL [LOGCMD]"; echo "CLAIM: error missing arguments"; exit 2; }
exec python3 - "$API" "$MODEL" <<'PY'
import http.client, json, socket, sys, urllib.parse

api, model = sys.argv[1].rstrip("/"), sys.argv[2]
u = urllib.parse.urlsplit(api)
if u.scheme != "http":
    print("CLAIM: error only plain http APIs are supported"); sys.exit(2)
host, port, prefix = u.hostname, u.port or 80, u.path.rstrip("/")

TOOLS = [{"type": "function", "function": {
    "name": "get_time", "description": "The current time.", "parameters": {"type": "object", "properties": {}}}}]
PROMPT = "What is 17 times 23? Think it through step by step, then call get_time."


def post(body, timeout=900):
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    c.request("POST", f"{prefix}/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse(); text = r.read().decode("utf-8", "replace"); c.close()
    try: j = json.loads(text)
    except Exception: j = None
    return r.status, j, text


def ask(choice, thinking, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "tools": TOOLS,
            "tool_choice": choice, "temperature": 0, "max_tokens": max_tokens, "return_token_ids": True}
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return post(body)


def ids(j):
    tf = (j or {}).get("tensorfold") or {}
    got = tf.get("token_ids")
    return got if isinstance(got, list) and got else None


def first_choice(j):
    return ((j or {}).get("choices") or [{}])[0]


def sha(j):
    return ((j or {}).get("tensorfold") or {}).get("token_sha")


def content(j):
    return (first_choice(j).get("message") or {}).get("content") or ""


def finish(j):
    return first_choice(j).get("finish_reason")


try:
    socket.create_connection((host, port), timeout=60).close()
except OSError as e:
    print(f"CLAIM: error cannot connect to the server: {e}"); sys.exit(2)

# control: `auto` with thinking on answers, and the reply carries its ids
status, j, text = ask("auto", True, 1)
if status != 200 or ids(j) is None:
    print(f"CLAIM: error `auto` + thinking on did not return token ids (status {status}): {text[:200]}"); sys.exit(2)
a0 = ids(j)[0]
print(f"control auto + thinking on: first token {a0}")

# 1. required + thinking on: the same first token, every time (before the patch it is the draw handle, climbing)
ok = True
got = []
for _ in range(5):
    status, j, text = ask("required", True, 1)
    i = ids(j)[0] if status == 200 and ids(j) else None
    got.append(i)
    good = i == a0
    ok = ok and good
    print(f"required + thinking on: first token {i} [{'as promised' if good else 'WRONG'}]")
if len(set(got)) != 1:
    print(f"required + thinking on: the first token changed across requests: {got}"); ok = False

# 2. required + thinking on equals auto for a real reply
sa, ja, ta = ask("auto", True, 400)
sr, jr, tr = ask("required", True, 400)
if sa != 200 or sr != 200 or ids(ja) is None or ids(jr) is None:
    print(f"CLAIM: error a 400-token reply failed (auto {sa}, required {sr})"); sys.exit(2)
same = (sha(ja) == sha(jr) and content(ja) == content(jr) and finish(ja) == finish(jr))
ok = ok and same
print(f"required + thinking on: token_sha {sha(jr)}, finish {finish(jr)} [{'same as auto' if same else 'DIFFERS from auto'}]")
print(f"  reply: {content(jr)[:160]!r}")
if not same:
    print(f"  auto: token_sha {sha(ja)}, finish {finish(ja)}, reply {content(ja)[:160]!r}")

# 3. the gate still holds for a reply that starts outside a think block
opener_ids = []
for _ in range(3):
    status, j, text = ask("required", False, 64)
    i = ids(j)[0] if status == 200 and ids(j) else None
    opener_ids.append(i)
    good = status == 200 and finish(j) == "tool_calls" and i is not None
    ok = ok and good
    print(f"required + thinking off: first token {i}, finish {finish(j)} [{'as promised' if good else 'WRONG'}]")
if len(set(opener_ids)) != 1:
    print(f"required + thinking off: the opener changed across requests: {opener_ids}"); ok = False

print("CLAIM: holds" if ok else "CLAIM: does not hold")
sys.exit(0 if ok else 1)
PY
