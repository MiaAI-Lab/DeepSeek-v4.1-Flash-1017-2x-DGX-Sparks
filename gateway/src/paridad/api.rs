//! The two sides of a render comparison: the gateway's own conversion (through
//! `deepseek-recipe`) and TensorFold's `/tokenize`, plus the protocol translation
//! that lets the comparator ask the Zig server the same prompt.
//!
//! The gateway side calls the same `parity::prepare` the production handlers use,
//! so a divergence is a real divergence and not an artefact of the comparator.

use std::path::Path;
use std::sync::Arc;

use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use crate::parity::{self, Chat, GatewayConfig, Messages, Protocol, Responses};
use crate::{Gateway, GatewaySettings};

/// The gateway's side of one case.
#[derive(Debug, Clone)]
pub enum LadoApi {
    Rendered {
        ids: Vec<u32>,
        prompt: String,
        history_len: u32,
        shared_prefixes: Vec<u32>,
        max_tokens: Option<u32>,
        thinking: bool,
        effort: Option<String>,
        image_count: usize,
    },
    Fallback {
        reason: String,
    },
    Error {
        status: u16,
        message: String,
    },
}

/// The Zig server's side of one case.
#[derive(Debug, Clone)]
pub struct LadoZig {
    pub ids: Vec<u32>,
    pub token_strs: Vec<String>,
    pub history_len: u32,
    pub shared_prefixes: Vec<u32>,
}

/// A translation that cannot be made faithfully (for example, a Responses item the
/// chat template cannot represent).
#[derive(Debug, Clone)]
pub enum ZigSide {
    Body(Value),
    NoTranslation(String),
}

/// Build the gateway with the served name and tokenizer the run uses.
pub fn build_gateway(
    tokenizer: &Path,
    served_name: &str,
    default_thinking: bool,
    image_token: &str,
) -> Result<Gateway, String> {
    let tokenizer = Tokenizer::from_file(tokenizer)
        .map_err(|error| format!("cannot load {}: {error}", tokenizer.display()))?;
    Gateway::new(
        Arc::new(tokenizer),
        GatewaySettings {
            served_name: served_name.to_string(),
            default_thinking,
            image_token: image_token.to_string(),
            aliases: Vec::new(),
        },
    )
    .map_err(|error| error.to_string())
}

/// Prepare a case with the gateway's own path.  The body must already carry the
/// served model and expanded image URLs.
pub async fn prepare_local(
    gateway: &Gateway,
    config: &GatewayConfig,
    protocol: Protocol,
    body: Value,
) -> LadoApi {
    if let Some(reason) = parity::detect_fallback(&body, protocol, config.engine_logprobs) {
        return LadoApi::Fallback { reason };
    }
    let prepared = match protocol {
        Protocol::Chat => parity::prepare::<Chat>(gateway, config, protocol, body).await,
        Protocol::Messages => parity::prepare::<Messages>(gateway, config, protocol, body).await,
        Protocol::Responses => parity::prepare::<Responses>(gateway, config, protocol, body).await,
    };
    match prepared {
        Ok(prepared) => LadoApi::Rendered {
            ids: prepared.encoded.ids.clone(),
            prompt: prepared.encoded.prompt.clone(),
            history_len: prepared.encoded.history_len,
            shared_prefixes: prepared.encoded.shared_prefixes.clone(),
            max_tokens: prepared.max_tokens,
            thinking: prepared.converted.conversation.thinking_mode,
            effort: prepared
                .converted
                .conversation
                .reasoning_effort
                .map(|effort| format!("{effort:?}").to_lowercase()),
            image_count: prepared.encoded.image_count,
        },
        Err(error) => {
            if let Some(reason) = error.fallback_reason.clone() {
                LadoApi::Fallback { reason }
            } else {
                LadoApi::Error {
                    status: error.status,
                    message: error.to_string(),
                }
            }
        }
    }
}

/// Translate a case body to the `/tokenize` request.  Messages and Responses are
/// reduced to equivalent chat messages; when that reduction is not faithful the
/// case is marked `n/a` by the caller.
pub fn zig_body(protocol: Protocol, body: &Value) -> ZigSide {
    match protocol {
        Protocol::Chat => ZigSide::Body(chat_tokenize_body(body)),
        Protocol::Messages => match translate_messages(body) {
            Ok(translated) => ZigSide::Body(translated),
            Err(reason) => ZigSide::NoTranslation(reason),
        },
        Protocol::Responses => match translate_responses(body) {
            Ok(translated) => ZigSide::Body(translated),
            Err(reason) => ZigSide::NoTranslation(reason),
        },
    }
}

fn chat_tokenize_body(body: &Value) -> Value {
    let mut out = Map::new();
    if let Some(messages) = body.get("messages") {
        out.insert("messages".to_string(), messages.clone());
    }
    for key in ["tools", "tool_choice", "response_format"] {
        if let Some(value) = body.get(key) {
            out.insert(key.to_string(), value.clone());
        }
    }
    let mut kwargs = body
        .get("chat_template_kwargs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(thinking) = body.get("thinking") {
        match thinking {
            Value::Bool(on) => {
                kwargs.insert("enable_thinking".to_string(), Value::Bool(*on));
            }
            Value::Object(object) => {
                if let Some(kind) = object.get("type").and_then(Value::as_str) {
                    kwargs.insert(
                        "enable_thinking".to_string(),
                        Value::Bool(kind == "enabled" || kind == "adaptive"),
                    );
                }
            }
            _ => {}
        }
    }
    if let Some(effort) = body.get("reasoning_effort").and_then(Value::as_str) {
        kwargs.insert(
            "reasoning_effort".to_string(),
            Value::String(effort.to_string()),
        );
    }
    if !kwargs.is_empty() {
        out.insert("chat_template_kwargs".to_string(), Value::Object(kwargs));
    }
    out.insert("add_generation_prompt".to_string(), Value::Bool(true));
    out.insert("return_token_strs".to_string(), Value::Bool(true));
    Value::Object(out)
}

fn translate_messages(body: &Value) -> Result<Value, String> {
    let mut messages = Vec::new();
    if let Some(system) = body.get("system") {
        let text = match system {
            Value::String(text) => text.clone(),
            Value::Array(blocks) => {
                let parts: Vec<&str> = blocks
                    .iter()
                    .filter_map(|block| {
                        if block.get("type").and_then(Value::as_str) == Some("text") {
                            block.get("text").and_then(Value::as_str)
                        } else {
                            None
                        }
                    })
                    .collect();
                parts.join("\n\n")
            }
            _ => return Err("system is neither a string nor a block list".to_string()),
        };
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }
    let Some(list) = body.get("messages").and_then(Value::as_array) else {
        return Err("messages is missing".to_string());
    };
    for message in list {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        let content = message.get("content");
        match role {
            "user" => match content {
                Some(Value::String(text)) => {
                    messages.push(json!({"role": "user", "content": text}))
                }
                Some(Value::Array(blocks)) => {
                    for block in blocks {
                        match block.get("type").and_then(Value::as_str) {
                            Some("text") => messages.push(json!({
                                "role": "user",
                                "content": block.get("text").cloned().unwrap_or(Value::Null)
                            })),
                            Some("image_url") => messages.push(json!({
                                "role": "user",
                                "content": [block.clone()]
                            })),
                            Some("tool_result") => {
                                let tool_call_id = block
                                    .get("tool_use_id")
                                    .and_then(Value::as_str)
                                    .unwrap_or("");
                                let text = match block.get("content") {
                                    Some(Value::String(text)) => text.clone(),
                                    Some(Value::Array(parts)) => parts
                                        .iter()
                                        .filter_map(|part| {
                                            if part.get("type").and_then(Value::as_str)
                                                == Some("text")
                                            {
                                                part.get("text").and_then(Value::as_str)
                                            } else {
                                                None
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n\n"),
                                    _ => String::new(),
                                };
                                messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": text
                                }));
                            }
                            Some("document") => messages.push(json!({
                                "role": "user",
                                "content": block.get("source").cloned().unwrap_or(Value::Null)
                            })),
                            Some(other) => {
                                return Err(format!("unsupported user content block {other}"));
                            }
                            None => return Err("content block without type".to_string()),
                        }
                    }
                }
                _ => return Err("user content is neither a string nor a block list".to_string()),
            },
            "assistant" => {
                let mut text = String::new();
                let mut reasoning = String::new();
                let mut calls = Vec::new();
                match content {
                    Some(Value::String(value)) => text = value.clone(),
                    Some(Value::Array(blocks)) => {
                        for block in blocks {
                            match block.get("type").and_then(Value::as_str) {
                                Some("text") => {
                                    if !text.is_empty() {
                                        text.push_str("\n\n");
                                    }
                                    text.push_str(
                                        block.get("text").and_then(Value::as_str).unwrap_or(""),
                                    );
                                }
                                Some("thinking") => {
                                    if !reasoning.is_empty() {
                                        reasoning.push_str("\n\n");
                                    }
                                    reasoning.push_str(
                                        block.get("thinking").and_then(Value::as_str).unwrap_or(""),
                                    );
                                }
                                Some("tool_use") => {
                                    let arguments =
                                        block.get("input").cloned().unwrap_or(json!({}));
                                    calls.push(json!({
                                        "id": block.get("id").cloned().unwrap_or(Value::Null),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name").cloned().unwrap_or(Value::Null),
                                            "arguments": serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string())
                                        }
                                    }));
                                }
                                Some("server_tool_use") | Some("web_search_tool_result") => {}
                                Some(other) => {
                                    return Err(format!("unsupported assistant block {other}"));
                                }
                                None => return Err("content block without type".to_string()),
                            }
                        }
                    }
                    None => {}
                    _ => {
                        return Err(
                            "assistant content is neither a string nor a block list".to_string()
                        );
                    }
                }
                let mut message = Map::new();
                message.insert("role".to_string(), Value::String("assistant".to_string()));
                message.insert("content".to_string(), Value::String(text));
                if !reasoning.is_empty() {
                    message.insert("reasoning_content".to_string(), Value::String(reasoning));
                }
                if !calls.is_empty() {
                    message.insert("tool_calls".to_string(), Value::Array(calls));
                }
                messages.push(Value::Object(message));
            }
            "system" => {
                let text = match content {
                    Some(Value::String(text)) => text.clone(),
                    Some(Value::Array(blocks)) => blocks
                        .iter()
                        .filter_map(|block| block.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                    _ => String::new(),
                };
                messages.push(json!({"role": "system", "content": text}));
            }
            other => return Err(format!("unsupported message role {other}")),
        }
    }
    let mut out = Map::new();
    out.insert("messages".to_string(), Value::Array(messages));
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let tools: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.get("name").cloned().unwrap_or(Value::Null),
                        "description": tool.get("description").cloned().unwrap_or(Value::Null),
                        "parameters": tool.get("input_schema").cloned().unwrap_or(json!({}))
                    }
                })
            })
            .collect();
        out.insert("tools".to_string(), Value::Array(tools));
    }
    if let Some(choice) = body.get("tool_choice") {
        out.insert(
            "tool_choice".to_string(),
            translate_anthropic_choice(choice)?,
        );
    }
    let mut kwargs = Map::new();
    if let Some(thinking) = body.get("thinking").and_then(Value::as_object) {
        let on = thinking.get("type").and_then(Value::as_str) == Some("enabled");
        kwargs.insert("enable_thinking".to_string(), Value::Bool(on));
    }
    if let Some(effort) = body
        .get("output_config")
        .and_then(|config| config.get("effort"))
        .and_then(Value::as_str)
    {
        kwargs.insert(
            "reasoning_effort".to_string(),
            Value::String(effort.to_string()),
        );
    }
    if !kwargs.is_empty() {
        out.insert("chat_template_kwargs".to_string(), Value::Object(kwargs));
    }
    out.insert("add_generation_prompt".to_string(), Value::Bool(true));
    out.insert("return_token_strs".to_string(), Value::Bool(true));
    Ok(Value::Object(out))
}

fn translate_anthropic_choice(choice: &Value) -> Result<Value, String> {
    match choice {
        Value::String(value) => match value.as_str() {
            "auto" => Ok(Value::String("auto".to_string())),
            "any" => Ok(Value::String("required".to_string())),
            "none" => Ok(Value::String("none".to_string())),
            other => Err(format!("unsupported Anthropic tool_choice {other}")),
        },
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("tool") {
                Ok(json!({
                    "type": "function",
                    "function": {"name": object.get("name").cloned().unwrap_or(Value::Null)}
                }))
            } else {
                Err("unsupported Anthropic tool_choice object".to_string())
            }
        }
        _ => Err("tool_choice is neither a string nor an object".to_string()),
    }
}

fn translate_responses(body: &Value) -> Result<Value, String> {
    let mut messages = Vec::new();
    if let Some(instructions) = body.get("instructions").and_then(Value::as_str)
        && !instructions.is_empty()
    {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    match body.get("input") {
        Some(Value::String(text)) => messages.push(json!({"role": "user", "content": text})),
        Some(Value::Array(items)) => {
            for item in items {
                let kind = item
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("message");
                match kind {
                    "message" => {
                        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                        let text = responses_content_text(item.get("content"));
                        if role == "assistant" {
                            messages.push(json!({"role": "assistant", "content": text}));
                        } else {
                            messages.push(json!({"role": role, "content": text}));
                        }
                    }
                    "reasoning" => {
                        let text = responses_content_text(item.get("content"));
                        if let Some(Value::Object(last)) = messages.last_mut()
                            && last.get("role").and_then(Value::as_str) == Some("assistant")
                        {
                            last.insert("reasoning_content".to_string(), Value::String(text));
                        } else {
                            messages.push(json!({
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": text
                            }));
                        }
                    }
                    "function_call" | "custom_tool_call" => {
                        let arguments = item.get("arguments").cloned().unwrap_or(json!({}));
                        messages.push(json!({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": item.get("call_id").cloned().unwrap_or(Value::Null),
                                "type": "function",
                                "function": {
                                    "name": item.get("name").cloned().unwrap_or(Value::Null),
                                    "arguments": serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string())
                                }
                            }]
                        }));
                    }
                    "function_call_output" | "custom_tool_call_output" => {
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
                            "content": item.get("output").cloned().unwrap_or(Value::Null)
                        }));
                    }
                    "item_reference" | "web_search_call" => {}
                    other => return Err(format!("unsupported Responses input item {other}")),
                }
            }
        }
        _ => return Err("input is neither a string nor an item list".to_string()),
    }
    let mut out = Map::new();
    out.insert("messages".to_string(), Value::Array(messages));
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let tools: Vec<Value> = tools
            .iter()
            .filter_map(|tool| {
                if tool.get("type").and_then(Value::as_str) == Some("function") {
                    Some(json!({
                        "type": "function",
                        "function": {
                            "name": tool.get("name").cloned().unwrap_or(Value::Null),
                            "description": tool.get("description").cloned().unwrap_or(Value::Null),
                            "parameters": tool.get("parameters").cloned().unwrap_or(json!({}))
                        }
                    }))
                } else {
                    None
                }
            })
            .collect();
        out.insert("tools".to_string(), Value::Array(tools));
    }
    if let Some(choice) = body.get("tool_choice") {
        out.insert(
            "tool_choice".to_string(),
            translate_responses_choice(choice)?,
        );
    }
    let mut kwargs = Map::new();
    if let Some(effort) = body
        .get("reasoning")
        .and_then(|reasoning| reasoning.get("effort"))
        .and_then(Value::as_str)
    {
        kwargs.insert(
            "reasoning_effort".to_string(),
            Value::String(effort.to_string()),
        );
        kwargs.insert("enable_thinking".to_string(), Value::Bool(effort != "none"));
    }
    if !kwargs.is_empty() {
        out.insert("chat_template_kwargs".to_string(), Value::Object(kwargs));
    }
    if let Some(format) = body.get("text").and_then(|text| text.get("format"))
        && let Some(converted) = translate_response_format(format)
    {
        out.insert("response_format".to_string(), converted);
    }
    out.insert("add_generation_prompt".to_string(), Value::Bool(true));
    out.insert("return_token_strs".to_string(), Value::Bool(true));
    Ok(Value::Object(out))
}

fn responses_content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                if matches!(
                    part.get("type").and_then(Value::as_str),
                    Some("input_text")
                        | Some("output_text")
                        | Some("text")
                        | Some("reasoning_text")
                ) {
                    part.get("text").and_then(Value::as_str)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

fn translate_responses_choice(choice: &Value) -> Result<Value, String> {
    match choice {
        Value::String(value) => match value.as_str() {
            "auto" => Ok(Value::String("auto".to_string())),
            "required" => Ok(Value::String("required".to_string())),
            "none" => Ok(Value::String("none".to_string())),
            other => Err(format!("unsupported Responses tool_choice {other}")),
        },
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("function") {
                Ok(json!({
                    "type": "function",
                    "function": {"name": object.get("name").cloned().unwrap_or(Value::Null)}
                }))
            } else {
                Err("unsupported Responses tool_choice object".to_string())
            }
        }
        _ => Err("tool_choice is neither a string nor an object".to_string()),
    }
}

fn translate_response_format(format: &Value) -> Option<Value> {
    match format.get("type").and_then(Value::as_str) {
        Some("text") => None,
        Some("json_object") => Some(json!({"type": "json_object"})),
        Some("json_schema") => {
            let schema = format.get("json_schema").cloned().unwrap_or(json!({}));
            Some(json!({"type": "json_schema", "json_schema": schema}))
        }
        Some("regex") => Some(json!({
            "type": "regex",
            "pattern": format.get("pattern").cloned().unwrap_or(Value::Null)
        })),
        _ => None,
    }
}

/// A small HTTP client for the Zig server's `/tokenize`.
#[derive(Clone)]
pub struct ZigClient {
    base: String,
    client: reqwest::Client,
}

impl ZigClient {
    pub fn new(base: &str) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            base: base.trim_end_matches('/').to_string(),
            client,
        })
    }

    /// Ask the Zig server to render and encode the translated body.
    pub async fn tokenize(&self, body: Value) -> Result<LadoZig, String> {
        let response = self
            .client
            .post(format!("{}/tokenize", self.base))
            .json(&body)
            .send()
            .await
            .map_err(|error| format!("tokenize request failed: {error}"))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| format!("cannot read tokenize response: {error}"))?;
        if !status.is_success() {
            return Err(format!("HTTP {status}: {text}"));
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| format!("invalid tokenize JSON: {error}"))?;
        let ids = value
            .get("tokens")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_u64)
                    .map(|id| id as u32)
                    .collect()
            })
            .ok_or_else(|| "tokenize response has no tokens".to_string())?;
        let token_strs = value
            .get("token_strs")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|item| item.as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default();
        let history_len = value
            .get("history_len")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let shared_prefixes = value
            .get("shared_prefixes")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_u64)
                    .map(|cut| cut as u32)
                    .collect()
            })
            .unwrap_or_default();
        Ok(LadoZig {
            ids,
            token_strs,
            history_len,
            shared_prefixes,
        })
    }
}

/// Ask the gateway to render a body; used as the SC-001 control when `--gateway`
/// is given.
pub async fn render_gateway(base: &str, protocol: Protocol, body: Value) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|error| error.to_string())?;
    let path = format!(
        "{}/ds41/render/{}",
        base.trim_end_matches('/'),
        protocol_name(protocol)
    );
    let response = client
        .post(path)
        .json(&body)
        .send()
        .await
        .map_err(|error| format!("render request failed: {error}"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|error| format!("cannot read render response: {error}"))?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {text}"));
    }
    serde_json::from_str(&text).map_err(|error| format!("invalid render JSON: {error}"))
}

pub fn protocol_name(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::Chat => "chat",
        Protocol::Messages => "messages",
        Protocol::Responses => "responses",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_translation_carries_thinking_and_effort() {
        let body = json!({
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "medium"
        });
        let value = chat_tokenize_body(&body);
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(value["chat_template_kwargs"]["reasoning_effort"], "medium");
        assert_eq!(value["add_generation_prompt"], true);
    }

    #[test]
    fn messages_translation_maps_tools_and_tool_results() {
        let body = json!({
            "system": "sos",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hola"}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "voy"},
                    {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"x": 1}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
                ]}
            ],
            "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
            "tool_choice": "any",
            "thinking": {"type": "enabled"},
            "output_config": {"effort": "low"}
        });
        let value = translate_messages(&body).expect("translate");
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(
            value["messages"][2]["tool_calls"][0]["function"]["name"],
            "f"
        );
        assert_eq!(value["messages"][3]["role"], "tool");
        assert_eq!(value["tool_choice"], "required");
        assert_eq!(value["chat_template_kwargs"]["reasoning_effort"], "low");
    }

    #[test]
    fn responses_translation_maps_input_and_function_calls() {
        let body = json!({
            "instructions": "sos",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hola"}]},
                {"type": "function_call", "call_id": "call_00_x", "name": "f", "arguments": {"x": 1}},
                {"type": "function_call_output", "call_id": "call_00_x", "output": "ok"}
            ],
            "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
            "reasoning": {"effort": "medium"}
        });
        let value = translate_responses(&body).expect("translate");
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(
            value["messages"][2]["tool_calls"][0]["function"]["name"],
            "f"
        );
        assert_eq!(value["messages"][3]["role"], "tool");
        assert_eq!(value["chat_template_kwargs"]["reasoning_effort"], "medium");
    }
}
