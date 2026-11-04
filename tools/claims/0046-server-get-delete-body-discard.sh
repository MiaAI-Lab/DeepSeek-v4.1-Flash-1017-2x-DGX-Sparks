#!/usr/bin/env bash
# Written by Hector Isaza (hecisaza, https://github.com/hecisaza) with the verify step of inference-agent-kit
# (https://github.com/hecisaza/inference-agent-kit), for the 4-Spark recipe, where the patch is 0044; here it is 0046.
# Apache-2.0. Below this header it is
# https://github.com/hecisaza/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/blob/10c4c7c21679f920aa2780d25a0019989ff284af/claims/MiaAI-Lab-DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold-pr-37.sh
# unchanged, except that Mia's AI Lab added the connection check before the controls: a server that is down exits 2, not 1.
# Usage: tools/claims/0046-server-get-delete-body-discard.sh API MODEL. Exit 0: the claim holds. 1: it does not. 2: the script could not judge.
# Exits 1 on an image built before zig/patches/0046-server-get-delete-body-discard.patch.
# claim: patches: 0044 GET and DELETE request bodies are drained, so a kept connection cannot hand one request's bytes
# to the next. PR.md: "a body-carrying GET ... leaves its bytes in the socket ... at the next keep-alive reuse those
# bytes are glued onto the next request's request line"; with 0044 the body is read and dropped before the route, and
# "a bodyless request reads nothing". So the claim is two-sided: a GET or DELETE with a Content-Length body is answered
# exactly as the same request without a body, and the next request on the same connection is parsed from its own line.
# Exercised by: raw HTTP/1.1 keep-alive sockets (python3; curl cannot pin a body-carrying GET and a follow-up to one
# socket). Each case sends a sequence on ONE connection and asserts the status AND content of every reply:
#   controls (no body): GET /models -> GET /models; DELETE /responses/<unknown> (404) -> GET /models. These prove
#     keep-alive reuse and both routes work on the build at all, and fix the replies the body cases must reproduce.
#   GET /models + body -> 200 with the same model list as the control, then GET /models -> 200 with the model.
#   DELETE /responses/<unknown> + body -> the same 404 "no stored response" error as the control, then GET /models.
#   DELETE /responses/<stored id> + body -> 200 {"id": <id>, "deleted": true}, then GET /responses/<id> on the same
#     socket -> 404 (the delete really took effect, and that reply was parsed from its own line), then GET /models.
# A build that rejects or closes on a body-carrying GET/DELETE fails, as does one that keeps the connection but
# answers the body request differently from the bodyless one. The stored response is created first with one
# deterministic POST /responses (temperature 0, 8 output tokens). No log is needed; LOGCMD ($3) is ignored.
# From REVIEW.md §2: "a Content-Length-bearing GET /models and DELETE /responses/<unknown-id>, each followed by GET
# /models on the same socket"; the audit asked that the first reply's status and content be asserted too, and that
# bodyless DELETE and a successful stored-response deletion be covered.
API=$1; MODEL=$2
[ -n "$API" ] && [ -n "$MODEL" ] || { echo "usage: claim.sh API MODEL [LOGCMD]"; echo "CLAIM: error missing arguments"; exit 2; }
exec python3 - "$API" "$MODEL" <<'PY'
import http.client, json, socket, sys, urllib.parse

api, model = sys.argv[1].rstrip("/"), sys.argv[2]
u = urllib.parse.urlsplit(api)
if u.scheme != "http":
    print("CLAIM: error only plain http APIs are supported"); sys.exit(2)
host, port, prefix = u.hostname, u.port or 80, u.path.rstrip("/")
PROBE = b"probe=0044&scanner=1"   # no CR/LF: left in the socket it becomes the start of the next request line
UNKNOWN = "resp_claim0044_does_not_exist"


def req(method, path, body=b""):
    head = f"{method} {prefix}{path} HTTP/1.1\r\nHost: api\r\nAccept: application/json\r\nConnection: keep-alive\r\n"
    if body:
        head += f"Content-Type: application/x-www-form-urlencoded\r\nContent-Length: {len(body)}\r\n"
    return head.encode() + b"\r\n" + body


class Reader:
    def __init__(self, s): self.s, self.buf = s, b""
    def fill(self):
        d = self.s.recv(65536)
        if not d: raise EOFError("connection closed by server")
        self.buf += d
    def line(self):
        while b"\r\n" not in self.buf: self.fill()
        l, self.buf = self.buf.split(b"\r\n", 1); return l
    def take(self, n):
        while len(self.buf) < n: self.fill()
        d, self.buf = self.buf[:n], self.buf[n:]; return d
    def response(self):
        status = self.line().decode("latin-1")
        hdrs = {}
        while True:
            l = self.line()
            if not l: break
            k, _, v = l.decode("latin-1").partition(":"); hdrs[k.strip().lower()] = v.strip()
        if "chunked" in hdrs.get("transfer-encoding", "").lower():
            body = b""
            while True:
                n = int(self.line().split(b";")[0], 16)
                if n == 0:
                    while self.line(): pass
                    break
                body += self.take(n); self.take(2)
        else:
            body = self.take(int(hdrs.get("content-length", "0")))
        parts = status.split()
        code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return code, body.decode("utf-8", "replace")


def sequence(reqs):
    """Send each request in turn on one socket, reading each reply before the next. Returns a list of
    (code, text); code is None when the reply never came (the rest are then not sent)."""
    s = socket.create_connection((host, port), timeout=60)
    r, out = Reader(s), []
    try:
        for q in reqs:
            try:
                s.sendall(q); out.append(r.response())
            except (EOFError, OSError, ValueError) as e:
                out.append((None, f"no reply on this connection: {e}"))
                out += [(None, "not sent")] * (len(reqs) - len(out))
                break
        return out
    finally:
        s.close()


def js(text):
    try: return json.loads(text)
    except Exception: return None


def models_ok(rep):
    j = js(rep[1])
    return rep[0] == 200 and isinstance(j, dict) and any(m.get("id") == model for m in j.get("data", []))


def model_ids(rep):
    return sorted(m.get("id") for m in js(rep[1]).get("data", []))


def not_stored(rid):
    def check(rep):
        j = js(rep[1])
        return (rep[0] == 404 and isinstance(j, dict) and isinstance(j.get("error"), dict)
                and f"no stored response has id '{rid}'" in str(j["error"].get("message")))
    return check


def deleted(rid):
    def check(rep):
        j = js(rep[1])
        return rep[0] == 200 and isinstance(j, dict) and j.get("id") == rid and j.get("deleted") is True
    return check


def run(label, steps):
    """steps: list of (request description, raw request, check). Prints every reply; True when all checks pass."""
    reps = sequence([q for _, q, _ in steps])
    ok = True
    for i, ((what, _, check), rep) in enumerate(zip(steps, reps), 1):
        good = rep[0] is not None and check(rep)
        print(f"{label}: #{i} {what} -> {rep[0]} {rep[1][:150]!r} [{'as promised' if good else 'WRONG'}]")
        if not good:
            ok = False
            msg = [l.strip() for l in rep[1].splitlines() if "Message:" in l]
            if msg: print(f"{label}: #{i} error page says: {msg[0][:200]}")
            if PROBE.decode()[:10] in rep[1]: print(f"{label}: #{i} the reply carries the previous request's body bytes")
    return ok, reps


GET_M = ("GET /models", req("GET", "/models"), models_ok)

try:
    socket.create_connection((host, port), timeout=60).close()
except OSError as e:
    print(f"CLAIM: error cannot connect to the server: {e}"); sys.exit(2)

# controls: no body anywhere; if these fail the build cannot be judged on this claim
c1, reps = run("control GET", [GET_M, GET_M])
c2, _ = run("control DELETE", [(f"DELETE /responses/<unknown>", req("DELETE", f"/responses/{UNKNOWN}"), not_stored(UNKNOWN)), GET_M])
if not (c1 and c2):
    print("CLAIM: error bodyless keep-alive reuse or the DELETE route does not work, so the body cases cannot be judged"); sys.exit(2)
ids = model_ids(reps[0])

# a stored response to delete for real (one deterministic generation on its own connection)
try:
    hc = http.client.HTTPConnection(host, port, timeout=300)
    hc.request("POST", f"{prefix}/responses", json.dumps({
        "model": model, "input": "Say ok.", "temperature": 0, "max_output_tokens": 8, "store": True}),
        {"content-type": "application/json"})
    hr = hc.getresponse(); created = js(hr.read().decode("utf-8", "replace")); hc.close()
    rid = created.get("id") if hr.status == 200 and isinstance(created, dict) else None
except (OSError, http.client.HTTPException) as e:
    print(f"CLAIM: error could not create a stored response: {e}"); sys.exit(2)
if not rid:
    print(f"CLAIM: error could not create a stored response (status {hr.status})"); sys.exit(2)
print(f"stored response created for the delete case: {rid}")

same_list = lambda rep: models_ok(rep) and model_ids(rep) == ids
cases = [
    ("GET with body", [("GET /models + body", req("GET", "/models", PROBE), same_list), GET_M]),
    ("DELETE unknown with body", [("DELETE /responses/<unknown> + body", req("DELETE", f"/responses/{UNKNOWN}", PROBE), not_stored(UNKNOWN)), GET_M]),
    ("DELETE stored with body", [("DELETE /responses/<stored> + body", req("DELETE", f"/responses/{rid}", PROBE), deleted(rid)),
                                 ("GET /responses/<stored>", req("GET", f"/responses/{rid}"), not_stored(rid)), GET_M]),
]
ok = True
for label, steps in cases:
    good, _ = run(label, steps)
    print(f"{label}: " + ("every reply on the kept connection is what the bodyless request gets"
                           if good else "a reply on the kept connection is not what the pull request promises"))
    ok = ok and good

print("CLAIM: holds" if ok else "CLAIM: does not hold")
sys.exit(0 if ok else 1)
PY

# Run against the base build (one run, 2026-11-01):
#   control GET: #1 GET /models -> 200 '{"object": "list", "data": [{"id": "DeepSeek-V4.1-Flash-1017", "object": "model", "created": 1793568591, "owned_by": "tensorfold"}]}' [as promised]
#   control GET: #2 GET /models -> 200 '{"object": "list", "data": [{"id": "DeepSeek-V4.1-Flash-1017", "object": "model", "created": 1793568591, "owned_by": "tensorfold"}]}' [as promised]
#   control DELETE: #1 DELETE /responses/<unknown> -> 404 '{"error": {"message": "no stored response has id \'resp_claim0044_does_not_exist\'", "type": "invalid_request_error", "param": "response_id", "code": nu' [as promised]
#   control DELETE: #2 GET /models -> 200 '{"object": "list", "data": [{"id": "DeepSeek-V4.1-Flash-1017", "object": "model", "created": 1793568591, "owned_by": "tensorfold"}]}' [as promised]
#   stored response created for the delete case: resp_b050cf1d72e27601032c85e91af6da4b
#   GET with body: #1 GET /models + body -> 200 '{"object": "list", "data": [{"id": "DeepSeek-V4.1-Flash-1017", "object": "model", "created": 1793568591, "owned_by": "tensorfold"}]}' [as promised]
#   GET with body: #2 GET /models -> 501 '<!DOCTYPE HTML>\n<html lang="en">\n    <head>\n        <meta charset="utf-8">\n        <style type="text/css">\n            :root {\n                color-s' [WRONG]
#   GET with body: #2 error page says: <p>Message: Unsupported method ('probe=0044&amp;scanner=1GET').</p>
#   GET with body: #2 the reply carries the previous request's body bytes
#   GET with body: a reply on the kept connection is not what the pull request promises
#   DELETE unknown with body: #1 DELETE /responses/<unknown> + body -> 404 '{"error": {"message": "no stored response has id \'resp_claim0044_does_not_exist\'", "type": "invalid_request_error", "param": "response_id", "code": nu' [as promised]
#   DELETE unknown with body: #2 GET /models -> 501 '<!DOCTYPE HTML>\n<html lang="en">\n    <head>\n        <meta charset="utf-8">\n        <style type="text/css">\n            :root {\n                color-s' [WRONG]
#   DELETE unknown with body: #2 error page says: <p>Message: Unsupported method ('probe=0044&amp;scanner=1GET').</p>
#   DELETE unknown with body: #2 the reply carries the previous request's body bytes
#   DELETE unknown with body: a reply on the kept connection is not what the pull request promises
#   DELETE stored with body: #1 DELETE /responses/<stored> + body -> 200 '{"id": "resp_b050cf1d72e27601032c85e91af6da4b", "object": "response", "deleted": true}' [as promised]
#   DELETE stored with body: #2 GET /responses/<stored> -> 501 '<!DOCTYPE HTML>\n<html lang="en">\n    <head>\n        <meta charset="utf-8">\n        <style type="text/css">\n            :root {\n                color-s' [WRONG]
#   DELETE stored with body: #2 error page says: <p>Message: Unsupported method ('probe=0044&amp;scanner=1GET').</p>
#   DELETE stored with body: #2 the reply carries the previous request's body bytes
#   DELETE stored with body: #3 GET /models -> None 'no reply on this connection: connection closed by server' [WRONG]
#   DELETE stored with body: a reply on the kept connection is not what the pull request promises
#   CLAIM: does not hold
#   exit=1
