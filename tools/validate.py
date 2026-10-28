"""Check a running server end to end: text, thinking, tool calls, an image, a video, a long prompt, and that greedy
replies sent several at once equal the same replies sent one at a time.

    python3 tools/validate.py [base url, default http://127.0.0.1:8888] [--model NAME] [--save FILE] [--compare FILE]

--save writes every greedy reply to FILE; --compare checks them against a FILE saved earlier (a server started with
DRAFTS=0, say: drafted replies must equal undrafted ones). Needs only the Python standard library, and PyAV for the
video clip it makes (skipped without it).
"""
import base64
import concurrent.futures as cf
import io
import json
import struct
import sys
import time
import urllib.request
import zlib

args = sys.argv[1:]


def arg(name, default):
    if name not in args:
        return default
    i = args.index(name) + 1
    if i >= len(args) or args[i].startswith("--"):
        sys.exit(f"{name} needs a value")
    return args[i]


URL = args[0].rstrip("/") if args and not args[0].startswith("--") else "http://127.0.0.1:8888"
MODEL = arg("--model", "DeepSeek-V4.1-Flash-1017")
SAVE = arg("--save", None)
COMPARE = arg("--compare", None)
failed = []


def post(body, timeout=900):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace")}


def chat(messages, max_tokens=200, thinking=False, **kw):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": thinking}, **kw}
    return post(body)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        failed.append(name)


def message(r):
    """The reply's assistant message when r is a completion, else None: an HTTP error, an error object, no choices or
    a message that is not the assistant's is never read as the model's reply."""
    if not isinstance(r, dict) or "error" in r or not isinstance(r.get("choices"), list) or not r["choices"]:
        return None
    m = r["choices"][0].get("message") if isinstance(r["choices"][0], dict) else None
    if not isinstance(m, dict) or m.get("role", "assistant") != "assistant":
        return None
    return m


def text(r):
    """The reply's text content, "" when there is none (a tool call's empty content, or no completion at all)."""
    c = (message(r) or {}).get("content")
    return c if isinstance(c, str) else ""


def reply_text(r):
    """The reply's text when r is a completion with non-empty text content, else None."""
    return text(r) or None


def prompt_tokens(r):
    usage = r.get("usage") if isinstance(r, dict) else None
    return usage.get("prompt_tokens") if isinstance(usage, dict) else None


def strict_json(s):
    """json.loads without the NaN / Infinity extensions, which are not JSON."""
    def refuse(name):
        raise ValueError(f"{name} is not JSON")
    return json.loads(s, parse_constant=refuse)


def well_formed_call(c):
    """An OpenAI tool call: an id, type "function", and a function with a name and JSON-object arguments."""
    if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"] or c.get("type") != "function":
        return False
    f = c.get("function")
    if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"] or not isinstance(f.get("arguments"), str):
        return False
    try:
        return isinstance(strict_json(f["arguments"]), dict)
    except ValueError:
        return False


def tool_calls_ok(r, name, city):
    """Every call in the reply is well formed and calls the one tool offered, `name`, with its required string `city`;
    the first asks for `city`. Returns the calls only when they can be sent back as the conversation's next turn."""
    calls = (message(r) or {}).get("tool_calls")
    if not isinstance(calls, list) or not calls or not all(well_formed_call(c) for c in calls):
        return False, []
    got = [strict_json(c["function"]["arguments"]).get("city") for c in calls]
    if any(c["function"]["name"] != name for c in calls) or not all(isinstance(g, str) and g for g in got):
        return False, []
    return city in got[0].lower(), calls


def png(w, h, pixel):
    raw = b"".join(b"\0" + b"".join(bytes(pixel(x, y)) for x in range(w)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def clip():
    try:
        import av
        import numpy as np
    except ImportError:
        return None
    buf = io.BytesIO()
    c = av.open(buf, "w", format="mp4")
    s = c.add_stream("libx264", rate=10)
    s.width, s.height, s.pix_fmt = 320, 240, "yuv420p"
    for i in range(40):
        a = np.zeros((240, 320, 3), np.uint8)
        a[:] = (220, 30, 30) if i < 20 else (30, 30, 220)
        a[100:140, 10 + i * 7:50 + i * 7] = 255
        for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    return buf.getvalue()


# ---------------------------------------------------------------- features
t = time.time()
r = chat([{"role": "user", "content": "What is the capital of France? One word."}], 20)
check("text", "paris" in text(r).lower(), repr(text(r)[:60]) + f" {time.time() - t:.1f}s")

r = chat([{"role": "user", "content": "What is 17 * 23? Answer with the number."}], 2000, thinking=True)
m = message(r) or {}
reasoning = m.get("reasoning_content") if isinstance(m.get("reasoning_content"), str) else ""
check("thinking", "391" in text(r) and len(reasoning) > 0, f"reasoning {len(reasoning)} chars, answer {text(r)[:40]!r}")

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = chat([{"role": "user", "content": "What's the weather in Tokyo? Use the tool."}], 300, tools=tools)
ok, calls = tool_calls_ok(r, "get_weather", "tokyo")
check("tool call", ok, json.dumps(calls)[:120] if calls else json.dumps(r)[:160])
if ok:
    msgs = [{"role": "user", "content": "What's the weather in Tokyo? Use the tool."},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": calls[0].get("id", "0"), "content": json.dumps({"temp_c": 21, "sky": "clear"})}]
    r = chat(msgs, 200, tools=tools)
    check("tool result", "21" in text(r), repr(text(r)[:80]))

img = png(64, 64, lambda x, y: (240, 200, 0) if (x - 32) ** 2 + (y - 32) ** 2 < 400 else (255, 255, 255))
r = chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}},
                                        {"type": "text", "text": "What shape and color is in this picture? Short answer."}]}], 60)
check("image", "yellow" in text(r).lower() or "circle" in text(r).lower(), repr(text(r)[:80]))

v = clip()
if v is None:
    print("SKIP  video (no PyAV here)")
else:
    r = chat([{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(v).decode()}},
                                            {"type": "text", "text": "What color is the background at first and what color later? Short answer."}]}], 80)
    tx = text(r).lower()
    check("video", "red" in tx and "blue" in tx, repr(text(r)[:100]))

long = " ".join(f"Item {i}: fox {i} jumps over dog {i * 7}." for i in range(1500))
t = time.time()
r = chat([{"role": "user", "content": long + "\nWhat is item 1234's dog number? The number only."}], 20)
check("long prompt", "8638" in text(r), f"{prompt_tokens(r)} tokens, {text(r)[:20]!r}, {time.time() - t:.1f}s")

# ---------------------------------------------------------------- one at a time == several at once
prompts = ["Write a haiku about the sea.", "List the first ten prime numbers.", "Explain TCP in two sentences.",
           "Write a Python function that reverses a string.", "Name five rivers in Europe.",
           "What causes the seasons on Earth? Two sentences.", "Translate 'good morning' into French, Spanish and German.",
           "Give three tips for writing clear documentation."]
req = lambda p: chat([{"role": "user", "content": p}], 160)
solo = [reply_text(req(p)) or "" for p in prompts]
with cf.ThreadPoolExecutor(len(prompts)) as ex:
    burst = [reply_text(r) or "" for r in ex.map(req, prompts)]
same = sum(bool(a) and a == b for a, b in zip(solo, burst))
check("concurrent == solo", same == len(prompts), f"{same} of {len(prompts)} equal")
if SAVE:
    json.dump(solo, open(SAVE, "w"))
if COMPARE:
    # every reply and its saved reference: as many as the prompts, non-empty text, equal exactly
    try:
        ref = json.load(open(COMPARE))
        why = ""
    except (OSError, ValueError) as e:
        ref, why = [], f"cannot read {COMPARE}: {e}"
    if why:
        pass
    elif not isinstance(ref, list):
        why = f"{COMPARE} holds a {type(ref).__name__}, not a list of replies"
    elif len(ref) != len(prompts):
        why = f"{COMPARE} holds {len(ref)} replies, not {len(prompts)}"
    elif not all(isinstance(x, str) and x for x in ref):
        why = f"{COMPARE} has an empty or non-text reply"
    elif not all(solo):
        why = "an empty reply here"
    same = 0 if why else sum(a == b for a, b in zip(solo, ref))
    check("equal to the saved replies", not why and same == len(prompts), why or f"{same} of {len(prompts)} equal")

print(f"\n{'all passed' if not failed else 'FAILED: ' + ', '.join(failed)}")
sys.exit(1 if failed else 0)
