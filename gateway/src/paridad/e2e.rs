//! End-to-end checks: the gateway's non-stream and stream replies, the native
//! completion hash, and the invariants ported from Sufyan's `tests/api_eval.py`.
//!
//! The checks are deliberately structural.  They do not compare the text with a
//! golden model answer: a scripted engine is allowed to answer anything, but the
//! API's shape, ordering, usage accounting and token hash must be exact.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::paridad::Protocol;
use crate::paridad::caso::{Caso, E2eExpect};
use crate::paridad::diff::first_divergence;

/// One end-to-end result, as it appears in the report.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct E2eResult {
    pub finish_api: Option<String>,
    pub finish_zig: Option<String>,
    pub sha_api: Option<String>,
    pub sha_zig: Option<String>,
    pub violaciones: Vec<String>,
    pub status: Option<u16>,
    pub gateway: Option<String>,
    pub same_ids: Option<bool>,
}

/// The outcome of one HTTP call.
#[derive(Debug, Clone)]
pub struct Reply {
    status: u16,
    gateway: Option<String>,
    sha: Option<String>,
    sha_raw: Option<String>,
    body: Value,
    text: String,
}

fn clean(text: &str) -> bool {
    !text.contains("<|im_start|>")
        && !text.contains("<|im_end|>")
        && !text.contains("<|endoftext|>")
        && !text.contains("<｜end_of_query｜>")
}

fn json_object(text: &str) -> Option<Value> {
    let value: Value = serde_json::from_str(text).ok()?;
    value.is_object().then_some(value)
}

fn call_id_ok(id: &str) -> bool {
    id.starts_with("call_")
        && id.len() >= 10
        && id.as_bytes()[5..7].iter().all(u8::is_ascii_digit)
        && id.as_bytes()[7] == b'_'
}

/// The HTTP/header invariants of a gateway reply.
pub fn path_violations(reply: &Reply) -> Vec<String> {
    let mut violations = Vec::new();
    if reply.status != 200 {
        violations.push(format!(
            "HTTP {}: {}",
            reply.status,
            truncate(&reply.text, 200)
        ));
    } else if reply.gateway.as_deref() != Some("api") {
        violations.push(format!("x-ds41-gateway: {:?} (want api)", reply.gateway));
    }
    violations
}

/// Chat message invariants, shared by full and streamed replies.
pub fn message_violations(message: &Value, finish: Option<&str>, thinking: bool) -> Vec<String> {
    let mut violations = Vec::new();
    let calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !matches!(finish, Some("stop" | "length" | "tool_calls")) {
        violations.push(format!("finish_reason {finish:?}"));
    }
    if finish == Some("tool_calls") && calls.is_empty() {
        violations.push("finish_reason tool_calls without tool_calls".to_string());
    }
    if finish == Some("stop") && !calls.is_empty() {
        violations.push("tool_calls with finish_reason stop".to_string());
    }
    if finish == Some("tool_calls") {
        for call in &calls {
            let function = call.get("function").unwrap_or(&Value::Null);
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            if !call_id_ok(id)
                || call.get("type").and_then(Value::as_str) != Some("function")
                || function
                    .get("name")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
            {
                violations.push(format!(
                    "tool call shape {}",
                    truncate(&call.to_string(), 160)
                ));
            }
            if function
                .get("arguments")
                .and_then(Value::as_str)
                .and_then(json_object)
                .is_none()
            {
                violations.push(format!(
                    "arguments not a JSON object: {}",
                    truncate(
                        function
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or(""),
                        80
                    )
                ));
            }
        }
    }
    let ids: Vec<&str> = calls
        .iter()
        .filter_map(|call| call.get("id").and_then(Value::as_str))
        .collect();
    if ids.len() != ids.iter().collect::<HashSet<_>>().len() {
        violations.push("duplicate tool call ids".to_string());
    }
    if !clean(message.get("content").and_then(Value::as_str).unwrap_or("")) {
        violations.push(format!(
            "markup in content: {:?}",
            truncate(
                message.get("content").and_then(Value::as_str).unwrap_or(""),
                120
            )
        ));
    }
    if message.get("reasoning").is_some() {
        violations.push("vLLM field `reasoning` (API: reasoning_content)".to_string());
    }
    if thinking
        && !message
            .get("reasoning_content")
            .is_some_and(Value::is_string)
    {
        violations.push("thinking on but no reasoning_content".to_string());
    }
    if !thinking
        && message
            .get("reasoning_content")
            .is_some_and(|value| !value.is_null())
    {
        violations.push("reasoning_content with thinking off".to_string());
    }
    violations
}

/// Usage invariants: prompt/completion/total, cache hit/miss and reasoning tokens.
pub fn usage_violations(usage: &Value, thinking: bool, reasoning: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let prompt = usage.get("prompt_tokens").and_then(Value::as_i64);
    let completion = usage.get("completion_tokens").and_then(Value::as_i64);
    let total = usage.get("total_tokens").and_then(Value::as_i64);
    if prompt.is_none_or(|value| value <= 0)
        || completion.is_none()
        || total != prompt.map(|p| p + completion.unwrap_or(0))
    {
        violations.push(format!("usage counts {usage}"));
    }
    let hit = usage.get("prompt_cache_hit_tokens").and_then(Value::as_i64);
    let miss = usage
        .get("prompt_cache_miss_tokens")
        .and_then(Value::as_i64);
    if hit.is_none()
        || miss.is_none()
        || hit.unwrap_or(0) + miss.unwrap_or(0) != prompt.unwrap_or(-1)
    {
        violations.push(format!(
            "prompt_cache_hit/miss_tokens {hit:?}/{miss:?} vs prompt_tokens {prompt:?}"
        ));
    }
    if thinking {
        let reasoning_tokens = usage
            .get("completion_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(Value::as_i64);
        if reasoning_tokens.is_none()
            || (reasoning_tokens.unwrap_or(0) <= 0 && !reasoning.is_empty())
        {
            violations.push(format!(
                "reasoning_tokens {reasoning_tokens:?} for {} reasoning chars",
                reasoning.chars().count()
            ));
        }
    }
    violations
}

/// Chat invariants for a full response.
pub fn chat_violations(reply: &Reply, thinking: bool) -> Vec<String> {
    let mut violations = path_violations(reply);
    if reply.status != 200 {
        return violations;
    }
    let Some(choice) = reply
        .body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    else {
        violations.push(format!(
            "not a chat.completion: {}",
            truncate(&reply.body.to_string(), 200)
        ));
        return violations;
    };
    let message = choice.get("message").cloned().unwrap_or(Value::Null);
    let finish = choice.get("finish_reason").and_then(Value::as_str);
    let reasoning = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .unwrap_or("");
    violations.extend(message_violations(&message, finish, thinking));
    violations.extend(usage_violations(
        reply.body.get("usage").unwrap_or(&Value::Null),
        thinking,
        reasoning,
    ));
    violations
}

/// Parse an SSE body into `(event, data)` pairs; `[DONE]` has no event name.
fn parse_sse(text: &str) -> Vec<(Option<String>, String)> {
    let mut events = Vec::new();
    let mut event = None;
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("event: ") {
            event = Some(name.to_string());
        } else if let Some(data) = line.strip_prefix("data:") {
            events.push((
                event.clone(),
                data.strip_prefix(' ').unwrap_or(data).to_string(),
            ));
            event = None;
        }
    }
    events
}

/// Every SSE `data:` that is not `[DONE]` must be a JSON value; a malformed line
/// is a violation, never silently dropped.
pub fn sse_data_violations(text: &str) -> Vec<String> {
    let mut violations = Vec::new();
    for (_, data) in parse_sse(text) {
        if data != "[DONE]" && serde_json::from_str::<Value>(&data).is_err() {
            violations.push(format!("SSE data is not JSON: {:?}", truncate(&data, 80)));
        }
    }
    violations
}

/// Chat stream invariants: one id, role first, reasoning before content, exactly
/// one finish at the last chunk, usage in the last chunk, and `[DONE]`.
pub fn chat_stream_violations(text: &str, thinking: bool) -> Vec<String> {
    let mut violations = Vec::new();
    let events = parse_sse(text);
    let datas: Vec<&str> = events.iter().map(|(_, data)| data.as_str()).collect();
    if datas.last() != Some(&"[DONE]") {
        violations.push("stream does not end with data: [DONE]".to_string());
    }
    let mut chunks = Vec::new();
    for data in datas.iter().filter(|data| **data != "[DONE]") {
        match serde_json::from_str::<Value>(data) {
            Ok(chunk) => chunks.push(chunk),
            Err(error) => violations.push(format!(
                "SSE data is not JSON ({}): {:?}",
                error,
                truncate(data, 80)
            )),
        }
    }
    if chunks.is_empty() {
        violations.push("no chunks".to_string());
        return violations;
    }
    let ids: HashSet<&str> = chunks
        .iter()
        .filter_map(|chunk| chunk.get("id").and_then(Value::as_str))
        .collect();
    if ids.len() != 1
        || chunks.iter().any(|chunk| {
            chunk.get("object").and_then(Value::as_str) != Some("chat.completion.chunk")
        })
    {
        violations.push("chunks differ in id or are not chat.completion.chunk".to_string());
    }
    let first_delta = chunks
        .first()
        .and_then(|chunk| chunk.get("choices"))
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"));
    if first_delta
        .and_then(|delta| delta.get("role"))
        .and_then(Value::as_str)
        != Some("assistant")
    {
        violations.push("first delta has no role assistant".to_string());
    }
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut in_content = false;
    let mut finishes = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let usage = chunk.get("usage");
        if index + 1 < chunks.len() && usage.is_some_and(|value| !value.is_null()) {
            violations.push(format!("chunk {index}: usage before the last chunk"));
        }
        for choice in chunk
            .get("choices")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            if let Some(finish) = choice.get("finish_reason").and_then(Value::as_str) {
                finishes.push((index, finish.to_string()));
            }
            let delta = choice.get("delta").unwrap_or(&Value::Null);
            if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str)
                && !text.is_empty()
            {
                if in_content {
                    violations.push(format!("chunk {index}: reasoning delta after content"));
                }
                reasoning.push_str(text);
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str)
                && !text.is_empty()
            {
                in_content = true;
                content.push_str(text);
            }
        }
    }
    let finish_indexes: Vec<usize> = finishes.iter().map(|(index, _)| *index).collect();
    if finish_indexes != [chunks.len() - 1] {
        violations.push(format!(
            "finish_reason on chunks {finish_indexes:?} of {} (want only the last)",
            chunks.len()
        ));
    }
    let last_usage = chunks.last().and_then(|chunk| chunk.get("usage"));
    if !last_usage.is_some_and(Value::is_object) {
        violations.push("last chunk carries no usage".to_string());
    }
    if thinking && reasoning.is_empty() {
        violations.push("thinking on but stream has no reasoning".to_string());
    }
    if !thinking && !reasoning.is_empty() {
        violations.push("stream has reasoning with thinking off".to_string());
    }
    let _ = content;
    violations
}

/// Messages API invariants.
pub fn messages_violations(reply: &Reply, thinking: bool) -> Vec<String> {
    let mut violations = path_violations(reply);
    if reply.status != 200 {
        return violations;
    }
    let body = &reply.body;
    let Some(blocks) = body.get("content").and_then(Value::as_array) else {
        violations.push(format!(
            "not a message: {}",
            truncate(&body.to_string(), 200)
        ));
        return violations;
    };
    if body.get("type").and_then(Value::as_str) != Some("message")
        || body.get("role").and_then(Value::as_str) != Some("assistant")
    {
        violations.push(format!(
            "not a message: {}",
            truncate(&body.to_string(), 200)
        ));
        return violations;
    }
    let stop = body.get("stop_reason").and_then(Value::as_str);
    let kinds: Vec<&str> = blocks
        .iter()
        .filter_map(|block| block.get("type").and_then(Value::as_str))
        .collect();
    let uses: Vec<&Value> = blocks
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
        .collect();
    if !matches!(
        stop,
        Some("end_turn" | "max_tokens" | "stop_sequence" | "tool_use")
    ) {
        violations.push(format!("stop_reason {stop:?}"));
    }
    if (stop == Some("tool_use")) != !uses.is_empty() && stop != Some("max_tokens") {
        violations.push(format!(
            "stop_reason {stop:?} with {} tool_use blocks",
            uses.len()
        ));
    }
    if thinking
        && (kinds.first() != Some(&"thinking")
            || !blocks
                .first()
                .and_then(|block| block.get("signature"))
                .is_some_and(Value::is_string))
    {
        violations.push(format!(
            "thinking on: blocks {kinds:?} (want a signed thinking block first)"
        ));
    }
    if !thinking && kinds.contains(&"thinking") {
        violations.push("thinking block with thinking disabled".to_string());
    }
    for block in &uses {
        if block
            .get("id")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
            || block
                .get("name")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || !block.get("input").is_some_and(Value::is_object)
        {
            violations.push(format!(
                "tool_use shape {}",
                truncate(&block.to_string(), 160)
            ));
        }
    }
    let ids: Vec<&str> = uses
        .iter()
        .filter_map(|block| block.get("id").and_then(Value::as_str))
        .collect();
    if ids.len() != ids.iter().collect::<HashSet<_>>().len() {
        violations.push("duplicate tool_use ids".to_string());
    }
    let text: String = blocks
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect();
    if !clean(&text) {
        violations.push("markup in text blocks".to_string());
    }
    let usage = body.get("usage").unwrap_or(&Value::Null);
    if !usage
        .get("input_tokens")
        .and_then(Value::as_i64)
        .is_some_and(|value| value > 0)
        || usage.get("output_tokens").and_then(Value::as_i64).is_none()
    {
        violations.push(format!("usage {usage}"));
    }
    violations
}

/// Responses API invariants.
pub fn responses_violations(reply: &Reply, thinking: bool) -> Vec<String> {
    let mut violations = path_violations(reply);
    if reply.status != 200 {
        return violations;
    }
    let body = &reply.body;
    let Some(items) = body.get("output").and_then(Value::as_array) else {
        violations.push(format!(
            "not a response: {}",
            truncate(&body.to_string(), 200)
        ));
        return violations;
    };
    if body.get("object").and_then(Value::as_str) != Some("response") {
        violations.push(format!(
            "not a response: {}",
            truncate(&body.to_string(), 200)
        ));
        return violations;
    }
    let kinds: Vec<&str> = items
        .iter()
        .filter_map(|item| item.get("type").and_then(Value::as_str))
        .collect();
    if !matches!(
        body.get("status").and_then(Value::as_str),
        Some("completed" | "incomplete")
    ) {
        violations.push(format!("status {:?}", body.get("status")));
    }
    let has_reasoning = kinds.contains(&"reasoning");
    if thinking && !has_reasoning {
        violations.push(format!(
            "output {kinds:?}: thinking on but no reasoning item"
        ));
    } else if thinking && kinds.first() != Some(&"reasoning") {
        violations.push(format!("output {kinds:?}: reasoning item not first"));
    } else if !thinking && has_reasoning {
        violations.push(format!(
            "output {kinds:?}: reasoning item with thinking off"
        ));
    }
    let calls: Vec<&Value> = items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
        .collect();
    for call in &calls {
        if call
            .get("call_id")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
            || call
                .get("name")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || call
                .get("arguments")
                .and_then(Value::as_str)
                .and_then(json_object)
                .is_none()
        {
            violations.push(format!(
                "function_call shape {}",
                truncate(&call.to_string(), 160)
            ));
        }
    }
    let ids: Vec<&str> = calls
        .iter()
        .filter_map(|call| call.get("call_id").and_then(Value::as_str))
        .collect();
    if ids.len() != ids.iter().collect::<HashSet<_>>().len() {
        violations.push("duplicate call_ids".to_string());
    }
    let text: String = items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .flat_map(|item| {
            item.get("content")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[])
        })
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect();
    if !clean(&text) {
        violations.push("markup in output_text".to_string());
    }
    let usage = body.get("usage").unwrap_or(&Value::Null);
    let input = usage.get("input_tokens").and_then(Value::as_i64);
    let output = usage.get("output_tokens").and_then(Value::as_i64);
    let total = usage.get("total_tokens").and_then(Value::as_i64);
    let counts_ok = matches!(
        (input, output, total),
        (Some(input), Some(output), Some(total))
            if input > 0 && output >= 0 && total == input + output
    );
    if !counts_ok {
        violations.push(format!("usage {usage}"));
    }
    if thinking
        && usage
            .get("output_tokens_details")
            .and_then(|details| details.get("reasoning_tokens"))
            .and_then(Value::as_i64)
            .is_none()
    {
        violations.push("no output_tokens_details.reasoning_tokens".to_string());
    }
    violations
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        text.to_string()
    } else {
        text.chars().take(limit).collect::<String>() + "…"
    }
}

/// The thinking mode a case asks for.  The gateway defaults to thinking on, so a
/// case without an explicit switch is treated as thinking on.
pub fn case_thinking(caso: &Caso) -> bool {
    let body = &caso.body;
    if let Some(thinking) = body.get("thinking") {
        match thinking {
            Value::Bool(on) => return *on,
            Value::Object(object) => {
                if let Some(kind) = object.get("type").and_then(Value::as_str) {
                    return kind == "enabled" || kind == "adaptive";
                }
            }
            _ => {}
        }
    }
    // Chat's shim maps only these `chat_template_kwargs` keys; the two switches
    // are ORed and an effort of `none` turns thinking off, as in the gateway.
    let ctk = if caso.protocol == "chat" {
        body.get("chat_template_kwargs").and_then(Value::as_object)
    } else {
        None
    };
    let ctk_switch = ctk.and_then(|object| {
        let mut seen = false;
        let mut on = false;
        for key in ["thinking", "enable_thinking"] {
            if let Some(value) = object.get(key).and_then(Value::as_bool) {
                seen = true;
                on |= value;
            }
        }
        seen.then_some(on)
    });
    let ctk_effort = ctk
        .and_then(|object| object.get("reasoning_effort"))
        .and_then(Value::as_str);
    let effort = body
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or(ctk_effort)
        .or_else(|| {
            body.get("reasoning")
                .and_then(|reasoning| reasoning.get("effort"))
                .and_then(Value::as_str)
        });
    if let Some(on) = ctk_switch {
        return on && effort != Some("none");
    }
    if effort == Some("none") {
        return false;
    }
    if effort.is_some() {
        return true;
    }
    true
}

/// The max token field and value a case asks for, with a small default for a
/// scripted engine.
fn max_tokens(protocol: Protocol, body: &Value) -> (String, u64) {
    let field = match protocol {
        Protocol::Chat | Protocol::Messages => "max_tokens",
        Protocol::Responses => "max_output_tokens",
    };
    let value = body.get(field).and_then(Value::as_u64).unwrap_or(32);
    (field.to_string(), value)
}

/// Fix the comparison to greedy sampling.  Responses has `temperature` but no
/// `seed` field in the recipe's schema; temperature 0 already makes it greedy.
fn force_greedy(protocol: Protocol, body: &mut Value) {
    if let Value::Object(object) = body {
        object.insert("temperature".to_string(), json!(0.0));
        if protocol != Protocol::Responses {
            object.insert("seed".to_string(), json!(7));
        }
    }
}

/// Run one e2e case.  Every request is local; `gateway` and `zig` are base URLs.
pub async fn run_case(caso: &Caso, gateway: &str, zig: &str, served_name: &str) -> E2eResult {
    let mut result = E2eResult::default();
    let protocol = match caso.protocol.as_str() {
        "chat" => Protocol::Chat,
        "messages" => Protocol::Messages,
        _ => Protocol::Responses,
    };
    let mut body = crate::paridad::caso::with_model(&caso.body, served_name);
    crate::paridad::caso::expand_images(&mut body);
    let (max_field, max_value) = max_tokens(protocol, &body);
    if let Value::Object(object) = &mut body {
        object.insert(max_field.clone(), json!(max_value));
    }
    force_greedy(protocol, &mut body);
    let thinking = case_thinking(caso);
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            result
                .violaciones
                .push(format!("cannot build HTTP client: {error}"));
            return result;
        }
    };
    let path = match protocol {
        Protocol::Chat => "/v1/chat/completions",
        Protocol::Messages => "/v1/messages",
        Protocol::Responses => "/v1/responses",
    };
    let non_stream = call_json(&client, &format!("{gateway}{path}"), body.clone()).await;
    if let E2eExpect::Object {
        error: Some(expected),
        ..
    } = &caso.expect.e2e
    {
        result.status = Some(non_stream.status);
        result.gateway = non_stream.gateway.clone();
        let message = non_stream
            .body
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if non_stream.status != expected.status || !message.contains(&expected.message_contains) {
            result.violaciones.push(format!(
                "expected error {} containing {:?}, got HTTP {} {:?}",
                expected.status, expected.message_contains, non_stream.status, message
            ));
        }
        return result;
    }
    result.status = Some(non_stream.status);
    result.gateway = non_stream.gateway.clone();
    result.finish_api = non_stream
        .body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            non_stream
                .body
                .get("stop_reason")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    result.sha_api = non_stream
        .sha_raw
        .clone()
        .or_else(|| non_stream.sha.clone());
    let violations = match protocol {
        Protocol::Chat => chat_violations(&non_stream, thinking),
        Protocol::Messages => messages_violations(&non_stream, thinking),
        Protocol::Responses => responses_violations(&non_stream, thinking),
    };
    result.violaciones.extend(violations);

    // The streaming request, assembled and checked for the chat protocol.  The
    // other protocols still get the basic `[DONE]`/no-error check.
    let mut stream_body = body.clone();
    if let Value::Object(object) = &mut stream_body {
        object.insert("stream".to_string(), json!(true));
    }
    let stream_reply = call_text(&client, &format!("{gateway}{path}"), stream_body).await;
    if stream_reply.status != 200 {
        result.violaciones.push(format!(
            "stream HTTP {}: {}",
            stream_reply.status,
            truncate(&stream_reply.text, 200)
        ));
    } else {
        match protocol {
            Protocol::Chat => {
                result
                    .violaciones
                    .extend(chat_stream_violations(&stream_reply.text, thinking));
            }
            Protocol::Messages | Protocol::Responses => {
                let events = parse_sse(&stream_reply.text);
                if events.last().is_none_or(|(_, data)| data != "[DONE]") {
                    result
                        .violaciones
                        .push("stream does not end with data: [DONE]".to_string());
                }
                if stream_reply.text.contains("\"error\"") {
                    result
                        .violaciones
                        .push("stream carries an error event".to_string());
                }
                result
                    .violaciones
                    .extend(sse_data_violations(&stream_reply.text));
            }
        }
    }

    // The render endpoint gives the prompt ids; the native completion route gives
    // the engine's own token hash for the same ids.
    let render = call_json(
        &client,
        &format!(
            "{gateway}/ds41/render/{}",
            crate::paridad::api::protocol_name(protocol)
        ),
        body.clone(),
    )
    .await;
    if render.status == 200
        && let Some(ids) = render
            .body
            .get("prompt_token_ids")
            .and_then(Value::as_array)
    {
        let prompt: Vec<u64> = ids.iter().filter_map(Value::as_u64).collect();
        let completion = json!({
            "prompt": prompt,
            "max_tokens": max_value,
            "temperature": 0.0,
            "seed": 7,
            "return_token_ids": true,
            "chat_template_kwargs": {"enable_thinking": false}
        });
        let native = call_json(&client, &format!("{zig}/v1/completions"), completion).await;
        result.sha_zig = native
            .body
            .get("tensorfold")
            .and_then(|runtime| runtime.get("token_sha"))
            .and_then(Value::as_str)
            .map(str::to_string);
        if native.status != 200 {
            result.violaciones.push(format!(
                "native completion HTTP {}: {}",
                native.status,
                truncate(&native.text, 200)
            ));
        }
    } else if render.status != 200 {
        result.violaciones.push(format!(
            "render HTTP {}: {}",
            render.status,
            truncate(&render.text, 200)
        ));
    }
    result.same_ids = match (&result.sha_api, &result.sha_zig) {
        (Some(api), Some(engine)) => Some(api == engine),
        _ => None,
    };
    result
}

async fn call_json(client: &reqwest::Client, url: &str, body: Value) -> Reply {
    let response = client.post(url).json(&body).send().await;
    match response {
        Ok(response) => {
            let status = response.status().as_u16();
            let gateway = response
                .headers()
                .get("x-ds41-gateway")
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let sha = response
                .headers()
                .get("x-ds41-token-sha")
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let sha_raw = response
                .headers()
                .get("x-ds41-token-sha-raw")
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let text = response.text().await.unwrap_or_default();
            let body = serde_json::from_str(&text).unwrap_or(Value::Null);
            Reply {
                status,
                gateway,
                sha,
                sha_raw,
                body,
                text,
            }
        }
        Err(error) => Reply {
            status: 0,
            gateway: None,
            sha: None,
            sha_raw: None,
            body: Value::Null,
            text: format!("request failed: {error}"),
        },
    }
}

async fn call_text(client: &reqwest::Client, url: &str, body: Value) -> Reply {
    let response = client.post(url).json(&body).send().await;
    match response {
        Ok(response) => {
            let status = response.status().as_u16();
            let gateway = response
                .headers()
                .get("x-ds41-gateway")
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let text = response.text().await.unwrap_or_default();
            Reply {
                status,
                gateway,
                sha: None,
                sha_raw: None,
                body: Value::Null,
                text,
            }
        }
        Err(error) => Reply {
            status: 0,
            gateway: None,
            sha: None,
            sha_raw: None,
            body: Value::Null,
            text: format!("request failed: {error}"),
        },
    }
}

/// Whether two id sequences are identical; used by the hash comparison and tests.
pub fn same_ids(left: &[u32], right: &[u32]) -> bool {
    first_divergence(left, right).is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(status: u16, gateway: &str, body: Value) -> Reply {
        Reply {
            status,
            gateway: Some(gateway.to_string()),
            sha: None,
            sha_raw: None,
            text: body.to_string(),
            body,
        }
    }

    #[test]
    fn a_chat_reply_with_a_template_mark_is_detected() {
        let body = json!({
            "choices": [{"message": {"role": "assistant", "content": "hola<|im_end|>"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                       "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 5}
        });
        let violations = chat_violations(&reply(200, "api", body), false);
        assert!(violations.iter().any(|v| v.contains("markup in content")));
    }

    #[test]
    fn a_tool_call_with_a_bad_id_is_detected() {
        let message = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        });
        let violations = message_violations(&message, Some("tool_calls"), false);
        assert!(violations.iter().any(|v| v.contains("tool call shape")));
    }

    #[test]
    fn a_stream_without_done_is_detected() {
        let text = "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
        let violations = chat_stream_violations(text, false);
        assert!(violations.iter().any(|v| v.contains("[DONE]")));
        assert!(violations.iter().any(|v| v.contains("last chunk")));
    }

    #[test]
    fn a_non_json_sse_data_line_is_a_violation() {
        let text = "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\ndata: not-json\n\ndata: [DONE]\n\n";
        let violations = chat_stream_violations(text, false);
        assert!(
            violations.iter().any(|v| v.contains("not JSON")),
            "violations: {violations:?}"
        );
        assert!(
            sse_data_violations(text)
                .iter()
                .any(|v| v.contains("not JSON"))
        );
    }

    #[test]
    fn usage_without_cache_hit_miss_is_detected() {
        let usage = json!({"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7});
        let violations = usage_violations(&usage, false, "");
        assert!(
            violations
                .iter()
                .any(|v| v.contains("prompt_cache_hit/miss"))
        );
    }

    #[test]
    fn messages_without_a_signed_thinking_block_are_detected() {
        let body = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hola"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2}
        });
        let violations = messages_violations(&reply(200, "api", body), true);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("signed thinking block"))
        );
    }

    #[test]
    fn responses_with_thinking_require_a_reasoning_item_first() {
        let body = json!({
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hola"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7,
                      "output_tokens_details": {"reasoning_tokens": 1}}
        });
        let violations = responses_violations(&reply(200, "api", body), true);
        assert!(
            violations.iter().any(|v| v.contains("no reasoning item")),
            "violations: {violations:?}"
        );

        let body = json!({
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hola"}]},
                       {"type": "reasoning"}],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        });
        let violations = responses_violations(&reply(200, "api", body), false);
        assert!(
            violations.iter().any(|v| v.contains("thinking off")),
            "violations: {violations:?}"
        );
    }

    #[test]
    fn responses_require_output_tokens_and_the_sum() {
        let body = json!({
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hola"}]}],
            "usage": {"input_tokens": 5, "total_tokens": 5}
        });
        let violations = responses_violations(&reply(200, "api", body), false);
        assert!(
            violations.iter().any(|v| v.contains("usage")),
            "violations: {violations:?}"
        );

        let body = json!({
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hola"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 9}
        });
        let violations = responses_violations(&reply(200, "api", body), false);
        assert!(
            violations.iter().any(|v| v.contains("usage")),
            "violations: {violations:?}"
        );
    }

    #[test]
    fn responses_without_reasoning_tokens_are_detected() {
        let body = json!({
            "object": "response",
            "status": "completed",
            "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "hola"}]}],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        });
        let violations = responses_violations(&reply(200, "api", body), true);
        assert!(violations.iter().any(|v| v.contains("reasoning_tokens")));
    }

    fn caso(body: Value) -> Caso {
        serde_json::from_value(json!({
            "id": "mock-thinking",
            "protocol": "chat",
            "body": body,
            "expect": {"render": "identical", "e2e": "n/a"}
        }))
        .expect("case")
    }

    #[test]
    fn chat_template_kwargs_decide_thinking_like_the_gateway() {
        assert!(!case_thinking(&caso(json!({
            "messages": [{"role": "user", "content": "Hola"}],
            "chat_template_kwargs": {"enable_thinking": false}
        }))));
        assert!(!case_thinking(&caso(json!({
            "messages": [{"role": "user", "content": "Hola"}],
            "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "none"}
        }))));
        assert!(case_thinking(&caso(json!({
            "messages": [{"role": "user", "content": "Hola"}],
            "chat_template_kwargs": {"reasoning_effort": "low"}
        }))));
        assert!(!case_thinking(&caso(json!({
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "disabled"},
            "chat_template_kwargs": {"enable_thinking": true}
        }))));
    }

    #[test]
    fn responses_requests_are_greedy_too() {
        let mut body = json!({"input": "hola"});
        force_greedy(Protocol::Responses, &mut body);
        assert_eq!(body["temperature"], 0.0);
        assert!(body.get("seed").is_none(), "Responses has no seed field");

        let mut body = json!({"messages": [{"role": "user", "content": "hola"}]});
        force_greedy(Protocol::Chat, &mut body);
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["seed"], 7);
    }

    #[test]
    fn same_ids_uses_the_first_divergence() {
        assert!(same_ids(&[1, 2, 3], &[1, 2, 3]));
        assert!(!same_ids(&[1, 2, 3], &[1, 2, 4]));
    }
}
