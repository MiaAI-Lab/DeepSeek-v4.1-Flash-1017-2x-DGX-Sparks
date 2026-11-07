#!/usr/bin/env python3
"""Measure silence and argument delivery in a streamed OpenAI tool call."""
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request


def api_key():
    key = os.environ.get("API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get(
        "API_KEY_FILE", os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key")
    )
    try:
        with open(path) as handle:
            line = next(
                (row.strip() for row in handle if row.strip() and not row.startswith("#")),
                "",
            )
    except OSError:
        return ""
    return line.split(":", 1)[1].strip() if ":" in line else line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("API_URL", "http://127.0.0.1:8888"))
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    parser.add_argument("--lines", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--fail-gap", type=float, default=1.0)
    parser.add_argument("--min-argument-bytes", type=int, default=40000)
    parser.add_argument("--expect-sha256")
    args = parser.parse_args()

    tools = [{
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a complete file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    }]
    prompt = (
        f"Call write exactly once. Write /tmp/index.html as a complete HTML document "
        f"containing exactly {args.lines} rows in this exact format, with four-digit "
        f"numbers: <p>0001: stream flush verification row</p>. Increment the number "
        f"on each row. Do not summarize."
    )
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "write"}},
        "parallel_tool_calls": False,
        "max_tokens": 16384,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key():
        headers["Authorization"] = "Bearer " + api_key()
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(),
        headers,
    )

    started = time.monotonic()
    previous = None
    first_event = first_args = None
    max_data_gap = max_active_gap = 0.0
    data_events = heartbeat_events = argument_events = argument_bytes = 0
    finish_reason = None
    arguments = {}
    names = {}
    done = usage_seen = False
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                now = time.monotonic()
                if payload == "[DONE]":
                    if previous is not None:
                        gap = now - previous
                        max_data_gap = max(max_data_gap, gap)
                        if first_args is not None:
                            max_active_gap = max(max_active_gap, gap)
                    done = True
                    break
                if previous is not None:
                    gap = now - previous
                    max_data_gap = max(max_data_gap, gap)
                    if first_args is not None:
                        max_active_gap = max(max_active_gap, gap)
                previous = now
                first_event = first_event or now
                data_events += 1
                event = json.loads(payload)
                usage_seen = usage_seen or isinstance(event.get("usage"), dict)
                for choice in event.get("choices") or []:
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    if not delta and choice.get("finish_reason") is None:
                        heartbeat_events += 1
                    for call in delta.get("tool_calls") or []:
                        index = call.get("index", 0)
                        function = call.get("function") or {}
                        if function.get("name"):
                            names[index] = function["name"]
                        fragment = function.get("arguments") or ""
                        if fragment:
                            first_args = first_args or now
                            argument_events += 1
                            argument_bytes += len(fragment.encode())
                            arguments[index] = arguments.get(index, "") + fragment
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP {error.code}: {error.read().decode(errors='replace')[:500]}")

    ended = time.monotonic()
    argument_text = arguments.get(0, "")
    argument_sha256 = hashlib.sha256(argument_text.encode()).hexdigest()
    parsed = None
    parse_error = None
    try:
        parsed = json.loads(argument_text)
    except (TypeError, json.JSONDecodeError) as error:
        parse_error = str(error)
    content = parsed.get("content", "") if isinstance(parsed, dict) else ""
    path = parsed.get("path") if isinstance(parsed, dict) else None
    expected_rows = [
        f"<p>{number:04d}: stream flush verification row</p>"
        for number in range(1, args.lines + 1)
    ]
    rows_valid = all(row in content for row in expected_rows)
    checks = {
        "multiple_argument_events": argument_events > 1,
        "active_gap_below_limit": max_active_gap < args.fail_gap,
        "argument_size": argument_bytes >= args.min_argument_bytes,
        "single_write_call": set(arguments) == {0} and names.get(0) == "write",
        "valid_json": parse_error is None,
        "path": path == "/tmp/index.html",
        "rows": rows_valid,
        "finish_reason": finish_reason == "tool_calls",
        "done": done,
        "usage": usage_seen,
        "expected_sha256": (
            args.expect_sha256 is None or argument_sha256 == args.expect_sha256
        ),
    }
    result = {
        "lines": args.lines,
        "total_s": round(ended - started, 3),
        "first_data_s": round(first_event - started, 3) if first_event else None,
        "first_arguments_s": round(first_args - started, 3) if first_args else None,
        "max_data_gap_s": round(max_data_gap, 3),
        "max_active_gap_s": round(max_active_gap, 3),
        "data_events": data_events,
        "heartbeat_events": heartbeat_events,
        "argument_events": argument_events,
        "argument_bytes": argument_bytes,
        "argument_sha256": argument_sha256,
        "parse_error": parse_error,
        "finish_reason": finish_reason,
        "done": done,
        "usage_seen": usage_seen,
        "checks": checks,
        "pass": all(checks.values()),
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
