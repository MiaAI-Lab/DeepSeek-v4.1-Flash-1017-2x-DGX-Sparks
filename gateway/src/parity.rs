//! Shared request preparation for the API path and `/ds41/render`.
//!
//! Both endpoints must produce exactly the same prompt, token IDs, sampling parameters
//! and cache marks, so the preparation lives here and neither endpoint can drift.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{ConversationRequest, ProtocolRequest};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use crate::engine::{GenerateRequest, SamplingParams, StreamOptions};
use crate::images::{self, ImageSettings, ResolvedImages};
use crate::security::ApiKeys;
use crate::{ApiError, EncodedPrompt, Gateway};

/// The public protocol of a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Chat,
    Messages,
    Responses,
}

impl Protocol {
    pub fn path(self) -> &'static str {
        match self {
            Self::Chat => "/v1/chat/completions",
            Self::Messages => "/v1/messages",
            Self::Responses => "/v1/responses",
        }
    }

    pub fn id_prefix(self) -> &'static str {
        match self {
            Self::Chat => "chatcmpl-",
            Self::Messages => "msg_",
            Self::Responses => "resp_",
        }
    }
}

/// Runtime policy from the CLI.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Context window used by `resolve_max_tokens`.
    pub max_model_len: u32,
    /// Whether sampling comes from deepseek-recipe (`true`) or the client (`false`).
    pub api_sampling: bool,
    /// Operator default effort injected into requests without one.
    pub default_effort: Option<String>,
    /// Return the API's 400 instead of falling back (the fallback itself arrives in T022).
    pub strict: bool,
    /// Require a local, constant-time `Authorization`/`x-api-key` check before any body handling.
    pub require_key: bool,
    /// Digests loaded once at startup; redacted by this config's `Debug` output.
    pub api_keys: Option<ApiKeys>,
    /// Reject private destinations for image URLs.
    pub block_private_urls: bool,
    /// Maximum images per request.
    pub max_images: usize,
    /// Maximum total image megabytes per request.
    pub max_image_mb: usize,
    /// Log level.
    pub log_level: String,
    /// The public model name.
    pub served_name: String,
    /// Whether the engine reports logprobs support, when known.
    pub engine_logprobs: Option<bool>,
    /// Test-only host-to-address overrides for image fetches.
    pub dns_overrides: HashMap<String, IpAddr>,
    /// How long an admitted request may take to send its body (`None`: no limit).
    pub body_timeout: Option<Duration>,
    /// The largest rendered prompt the gateway encodes, in bytes; a larger one is a 400
    /// before encoding, which costs memory in proportion to the prompt's bytes.
    pub max_prompt_bytes: usize,
    /// One encode at a time (see `prepare`), shared by every request of this gateway.
    pub encode_slot: Arc<Semaphore>,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            max_model_len: 65_536,
            api_sampling: true,
            default_effort: None,
            strict: false,
            require_key: false,
            api_keys: None,
            block_private_urls: true,
            max_images: 600,
            max_image_mb: 64,
            log_level: "info".to_string(),
            served_name: "DeepSeek-V4.1-Flash-1017".to_string(),
            engine_logprobs: None,
            dns_overrides: HashMap::new(),
            body_timeout: None,
            max_prompt_bytes: DEFAULT_MAX_PROMPT_BYTES,
            encode_slot: Arc::new(Semaphore::new(1)),
        }
    }
}

/// The default of `GatewayConfig::max_prompt_bytes` (`GATEWAY_MAX_PROMPT_MB=8`): a million
/// tokens of ordinary English renders to about 7 MiB (0.145 tokens a byte, measured).
pub const DEFAULT_MAX_PROMPT_BYTES: usize = 8 * 1024 * 1024;

/// One prepared request.
#[derive(Debug)]
pub struct Prepared {
    pub protocol: Protocol,
    pub converted: ConversationRequest,
    pub model: String,
    pub id: String,
    pub encoded: EncodedPrompt,
    pub prompt_tokens: u32,
    /// `None` when the prompt does not leave room for one token.
    pub max_tokens: Option<u32>,
    /// The exact context-limit 400, raised before the engine is touched.
    pub context_error: Option<ApiError>,
    pub sampling: Value,
    pub fallback_reason: Option<String>,
    pub logprobs_top_n: Option<u8>,
    pub images: ResolvedImages,
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate an API response id: prefix plus 32 lowercase hex characters.
pub fn response_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let counter = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut hasher = Sha256::new();
    hasher.update(nanos.to_le_bytes());
    hasher.update(counter.to_le_bytes());
    hasher.update(std::process::id().to_le_bytes());
    let digest = hasher.finalize();
    let hex: String = digest
        .iter()
        .take(16)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("{prefix}{hex}")
}

/// Prepare a request body for the engine.  `R` is the protocol request type.
pub async fn prepare<R: ProtocolRequest + DeserializeOwned>(
    gateway: &Gateway,
    config: &GatewayConfig,
    protocol: Protocol,
    body: Value,
) -> Result<Prepared, ApiError> {
    let Value::Object(mut object) = body else {
        return Err(ApiError::bad_request(
            "the request body must be a JSON object",
        ));
    };
    let mut fallback_reason = fallback_fields(&object, protocol, config.engine_logprobs);
    if protocol == Protocol::Chat {
        fallback_reason = fallback_reason.or_else(|| shim_chat(&mut object));
    }
    if fallback_reason.is_some() && config.strict {
        return Err(ApiError::bad_request(fallback_reason.clone().unwrap()));
    }
    if protocol == Protocol::Messages {
        check_context_management(&object)?;
    }
    if object.get("model").is_none_or(Value::is_null) {
        object.insert("model".to_string(), json!(config.served_name));
    }
    inject_default_effort(&mut object, protocol, config.default_effort.as_deref());
    // Deserialized from a reference, not a clone: the body's strings (images included) are not copied twice.
    let value = Value::Object(object);
    let parsed = R::deserialize(&value);
    let Value::Object(object) = value else {
        unreachable!("built as an object just above")
    };
    let request: R = parsed.map_err(|error| {
        let message = error.to_string();
        // The native chat route can infer a missing tool_call_id from the preceding
        // tool calls; preserve that engine behavior unless strict mode is enabled.
        // Only fall back when the missing id is genuinely the reason the request did
        // not deserialize, not when an unrelated error merely mentions the field.
        if protocol == Protocol::Chat
            && !config.strict
            && missing_tool_call_id_is_the_only_error::<R>(&object)
        {
            ApiError::fallback(message)
        } else {
            ApiError::bad_request(message)
        }
    })?;
    let converted = match gateway.try_convert(request) {
        Ok(converted) => converted,
        Err(error) if config.strict => return Err(ApiError::from(error)),
        Err(error) => return Err(ApiError::fallback(error.to_string())),
    };
    let model = converted_model(&object, &config.served_name);
    // Encoding takes memory for the prompt's bytes and tokens (about 120 bytes a byte of ordinary text, up to 360 at
    // one token a byte), so one request encodes at a time: three admitted requests each with a prompt at the byte limit
    // would otherwise encode at once. The slot covers only the checks, the render and the encode (seconds, no network);
    // image fetches run outside it.
    let encoded = {
        let _slot = config
            .encode_slot
            .acquire()
            .await
            .map_err(|_| ApiError::internal("the encode slot is closed"))?;
        let rendered = gateway.render(&converted.conversation);
        let prompt_bytes = rendered.prompt.len();
        if prompt_bytes > config.max_prompt_bytes {
            return Err(ApiError::bad_request(format!(
                "The rendered prompt is {prompt_bytes} bytes; this gateway encodes prompts of at most {} bytes \
                 (GATEWAY_MAX_PROMPT_MB). Please reduce the length of the messages.",
                config.max_prompt_bytes
            )));
        }
        // Memory also grows with the tokens: up to one a byte. A prompt counted at more than twice the context is
        // refused here; up to that it is encoded whole and an overflow gets the API's exact context 400 below.
        let context = config.max_model_len as usize;
        if gateway.prompt_tokens_over(&rendered, context.saturating_mul(2))? {
            return Err(ApiError::bad_request(format!(
                "This model's maximum context length is {context} tokens. However, the messages hold more than \
                 twice that ({} tokens). Please reduce the length of the messages.",
                context.saturating_mul(2)
            )));
        }
        drop(rendered);
        gateway.prepare_conversation(&converted.conversation)?
    };
    let images = images::resolve(
        &encoded.image_sources,
        &ImageSettings {
            max_images: config.max_images,
            max_total_bytes: config.max_image_mb.saturating_mul(1024 * 1024),
            block_private_urls: config.block_private_urls,
            dns_overrides: config.dns_overrides.clone(),
        },
    )
    .await?;
    let prompt_tokens = encoded.ids.len() as u32 + images.adjustment as u32;
    let (max_tokens, context_error) =
        match converted.resolve_max_tokens(prompt_tokens, config.max_model_len) {
            Ok(max_tokens) => (Some(max_tokens), None),
            Err(error) => (None, Some(ApiError::from(error))),
        };
    let inference = &converted.inference_options;
    let (temperature, top_p) = if config.api_sampling {
        (inference.temperature, inference.top_p)
    } else {
        (
            object
                .get("temperature")
                .and_then(Value::as_f64)
                .map(|v| v as f32),
            object
                .get("top_p")
                .and_then(Value::as_f64)
                .map(|v| v as f32),
        )
    };
    let logprobs_top_n = if inference.logprobs {
        Some(inference.top_logprobs.unwrap_or(0))
    } else {
        None
    };
    let mut sampling = Map::new();
    if let Some(max_tokens) = max_tokens {
        sampling.insert("max_tokens".to_string(), json!(max_tokens));
    }
    if let Some(temperature) = temperature {
        sampling.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = top_p {
        sampling.insert("top_p".to_string(), json!(top_p));
    }
    if let Some(seed) = inference.seed {
        sampling.insert("seed".to_string(), json!(seed));
    }
    if let Some(top) = logprobs_top_n {
        sampling.insert("logprobs".to_string(), json!(top + 1));
    }
    sampling.insert("detokenize".to_string(), json!(false));
    let sampling = Value::Object(sampling);

    Ok(Prepared {
        protocol,
        model,
        id: response_id(protocol.id_prefix()),
        encoded,
        prompt_tokens,
        max_tokens,
        context_error,
        sampling,
        fallback_reason,
        logprobs_top_n,
        images,
        converted,
    })
}

/// Whether `R` fails to deserialize only because a `tool` message lacks its
/// `tool_call_id`.  The native chat route infers the id from the preceding tool
/// calls, so the gateway falls back to the engine for exactly that case; any other
/// deserialization error stays the API's 400.
fn missing_tool_call_id_is_the_only_error<R: DeserializeOwned>(
    object: &Map<String, Value>,
) -> bool {
    let Some(messages) = object.get("messages").and_then(Value::as_array) else {
        return false;
    };
    if !messages.iter().any(|message| {
        message.get("role").and_then(Value::as_str) == Some("tool")
            && message.get("tool_call_id").is_none()
    }) {
        return false;
    }
    let mut patched = object.clone();
    let Some(messages) = patched.get_mut("messages").and_then(Value::as_array_mut) else {
        return false;
    };
    for message in messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        if message.get("role").and_then(Value::as_str) == Some("tool")
            && message.get("tool_call_id").is_none()
        {
            message.insert(
                "tool_call_id".to_string(),
                Value::String("__missing__".to_string()),
            );
        }
    }
    serde_json::from_value::<R>(Value::Object(patched)).is_ok()
}

/// The public model name the client asked for, or the served one if absent.
fn converted_model(object: &Map<String, Value>, served_name: &str) -> String {
    object
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(served_name)
        .to_string()
}

/// Chat fields that vLLM/DeepSeek clients send with today's meaning.  Returns a fallback
/// reason for `chat_template_kwargs` keys the API cannot map.
/// Every field `shim_chat` reads or writes.
const SHIM_CHAT_FIELDS: [&str; 5] = [
    "max_tokens",
    "max_completion_tokens",
    "chat_template_kwargs",
    "thinking",
    "reasoning_effort",
];

fn shim_chat(object: &mut Map<String, Value>) -> Option<String> {
    if object.get("max_tokens").is_none_or(Value::is_null)
        && let Some(max) = object.get("max_completion_tokens").cloned()
    {
        object.insert("max_tokens".to_string(), max);
    }
    object.remove("max_completion_tokens");
    let ctk = object.remove("chat_template_kwargs")?;
    let Value::Object(ctk) = ctk else {
        return Some("chat_template_kwargs is not an object".to_string());
    };
    let allowed = ["thinking", "enable_thinking", "reasoning_effort"];
    if let Some(key) = ctk.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Some(format!("chat_template_kwargs field {key}"));
    }
    for key in ["thinking", "enable_thinking"] {
        if ctk.get(key).is_some_and(|value| !value.is_boolean()) {
            return Some(format!("chat_template_kwargs {key} is not a boolean"));
        }
    }
    let effort = match ctk.get("reasoning_effort") {
        None | Some(Value::Null) => None,
        Some(Value::String(effort)) => Some(effort.clone()),
        Some(_) => {
            return Some("chat_template_kwargs reasoning_effort is not a string".to_string());
        }
    };
    let has_thinking = object.get("thinking").is_some_and(|value| !value.is_null());
    let has_effort = object
        .get("reasoning_effort")
        .is_some_and(|value| !value.is_null());
    let has_switch = ["thinking", "enable_thinking"]
        .iter()
        .any(|key| ctk.contains_key(*key));
    if !has_thinking && has_switch {
        // The two switches are ORed as vLLM does; the effort is mapped separately.
        let on = ["thinking", "enable_thinking"]
            .iter()
            .any(|key| ctk.get(*key).and_then(Value::as_bool).unwrap_or(false))
            && effort.as_deref() != Some("none");
        object.insert(
            "thinking".to_string(),
            json!({"type": if on { "enabled" } else { "disabled" }}),
        );
    }
    // An effort alone does not decide thinking: the request or --default-thinking
    // does, so no `thinking` field is inserted here.
    if let Some(effort) = effort
        && !has_effort
    {
        object.insert("reasoning_effort".to_string(), Value::String(effort));
    }
    None
}

/// Detect the fallback fields of a raw body, including Chat's `chat_template_kwargs`.
pub fn detect_fallback(
    body: &Value,
    protocol: Protocol,
    engine_logprobs: Option<bool>,
) -> Option<String> {
    let Value::Object(object) = body else {
        return None;
    };
    let mut reason = fallback_fields(object, protocol, engine_logprobs);
    if protocol == Protocol::Chat {
        // shim_chat reads and rewrites only these fields: copy them, not the whole body.
        let mut fields: Map<String, Value> = SHIM_CHAT_FIELDS
            .iter()
            .filter_map(|key| {
                object
                    .get(*key)
                    .map(|value| ((*key).to_string(), value.clone()))
            })
            .collect();
        reason = reason.or_else(|| shim_chat(&mut fields));
    }
    reason
}

/// Fields the Zig engine owns that the API path would drop, so the request must go to
/// the engine untouched (the proxy itself is T022; the render endpoint already reports
/// the reason).
fn fallback_fields(
    object: &Map<String, Value>,
    protocol: Protocol,
    engine_logprobs: Option<bool>,
) -> Option<String> {
    let present = |key: &str| object.get(key).is_some_and(|value| !value.is_null());
    if engine_logprobs == Some(false)
        && (object.get("logprobs").and_then(Value::as_bool) == Some(true)
            || present("top_logprobs"))
    {
        return Some("vLLM logprobs while /health says logprobs: false".to_string());
    }
    for key in [
        "prompt",
        "draft",
        "thinking_budget",
        "priority",
        "guided_json",
        "guided_regex",
        "guided_choice",
        "guided_grammar",
        "structured_outputs",
    ] {
        if present(key) {
            return Some(format!("vLLM field {key}"));
        }
    }
    if object
        .get("return_token_ids")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some("vLLM field return_token_ids".to_string());
    }
    if protocol == Protocol::Responses {
        for key in ["previous_response_id", "store", "conversation", "video_url"] {
            if present(key) {
                return Some(format!("vLLM field {key}"));
            }
        }
        if object
            .get("modalities")
            .and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty())
        {
            return Some("vLLM field modalities".to_string());
        }
        if present("audio") {
            return Some("vLLM field audio".to_string());
        }
    }
    if let Some(response_format) = object
        .get("response_format")
        .filter(|value| !value.is_null())
    {
        match response_format.get("type").and_then(Value::as_str) {
            Some("text" | "json_object" | "json_schema" | "regex") => {}
            Some(format) => return Some(format!("vLLM field response_format {format}")),
            None => return Some("vLLM field response_format".to_string()),
        }
    }
    None
}

/// `context_management` on `/v1/messages`.  Any `clear_thinking_*` edit (either API
/// date, any `keep`) is accepted, as the cloud accepts it, and the conversion renders
/// the history it is given unchanged; any other edit is refused with a 400.
fn check_context_management(object: &Map<String, Value>) -> Result<(), ApiError> {
    let Some(value) = object
        .get("context_management")
        .filter(|value| !value.is_null())
    else {
        return Ok(());
    };
    let Value::Object(value) = value else {
        return Err(ApiError::bad_request(
            "context_management must be an object",
        ));
    };
    let edits = match value.get("edits") {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Array(edits)) => edits,
        Some(_) => {
            return Err(ApiError::bad_request(
                "context_management.edits must be a list",
            ));
        }
    };
    for edit in edits {
        let kind = edit.get("type").and_then(Value::as_str).unwrap_or("");
        if !kind.starts_with("clear_thinking_") {
            return Err(ApiError::bad_request(
                "context_management supports clear_thinking edits only",
            ));
        }
    }
    Ok(())
}

/// Inject the operator's default effort before conversion, per protocol.
fn inject_default_effort(
    object: &mut Map<String, Value>,
    protocol: Protocol,
    effort: Option<&str>,
) {
    let Some(effort) = effort else {
        return;
    };
    match protocol {
        Protocol::Chat => {
            if object.get("reasoning_effort").is_none_or(Value::is_null) {
                object.insert(
                    "reasoning_effort".to_string(),
                    Value::String(effort.to_string()),
                );
            }
        }
        Protocol::Messages => {
            let config = object
                .entry("output_config".to_string())
                .or_insert_with(|| json!({}));
            if config.is_null() {
                *config = json!({});
            }
            if let Value::Object(config) = config
                && config.get("effort").is_none_or(Value::is_null)
            {
                config.insert("effort".to_string(), Value::String(effort.to_string()));
            }
        }
        Protocol::Responses => {
            let reasoning = object
                .entry("reasoning".to_string())
                .or_insert_with(|| json!({}));
            if reasoning.is_null() {
                *reasoning = json!({});
            }
            if let Value::Object(reasoning) = reasoning
                && reasoning.get("effort").is_none_or(Value::is_null)
            {
                reasoning.insert("effort".to_string(), Value::String(effort.to_string()));
            }
        }
    }
}

/// Build the token-route request for a prepared request.
pub fn generate_request(prepared: &Prepared) -> GenerateRequest {
    let sampling = &prepared.sampling;
    GenerateRequest {
        request_id: prepared.id.clone(),
        model: prepared.model.clone(),
        token_ids: prepared.encoded.ids.clone(),
        sampling_params: SamplingParams {
            max_tokens: sampling
                .get("max_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(1) as u32,
            temperature: sampling
                .get("temperature")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            top_p: sampling
                .get("top_p")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            seed: sampling.get("seed").and_then(Value::as_u64),
            logprobs: sampling
                .get("logprobs")
                .and_then(Value::as_u64)
                .map(|value| value as u8),
            detokenize: false,
        },
        stream: true,
        stream_options: StreamOptions {
            include_usage: true,
        },
        content_parts: prepared.images.parts.clone(),
        history_len: prepared.encoded.history_len,
        shared_prefixes: prepared.encoded.shared_prefixes.clone(),
        thinking: prepared.converted.conversation.thinking_mode,
        logprobs_top_n: prepared.logprobs_top_n,
    }
}

/// The `/ds41/render` body.
pub fn render_info(prepared: &Prepared) -> Value {
    if let Some(reason) = &prepared.fallback_reason {
        return json!({
            "prompt": null,
            "prompt_token_ids": null,
            "image_count": null,
            "thinking": null,
            "effort": null,
            "sampling": null,
            "max_tokens": null,
            "history_len": null,
            "shared_prefixes": null,
            "fallback_reason": reason,
        });
    }
    json!({
        "prompt": prepared.encoded.prompt,
        "prompt_token_ids": prepared.encoded.ids,
        "image_count": prepared.encoded.image_count,
        "thinking": prepared.converted.conversation.thinking_mode,
        "effort": prepared.converted.conversation.reasoning_effort.map(|effort| format!("{effort:?}").to_lowercase()),
        "sampling": prepared.sampling,
        "max_tokens": prepared.max_tokens,
        "history_len": prepared.encoded.history_len,
        "shared_prefixes": prepared.encoded.shared_prefixes,
        "fallback_reason": null,
    })
}

/// Protocol-specific parsing helper used by the handlers.
pub fn parse<R: DeserializeOwned>(body: Value) -> Result<R, ApiError> {
    serde_json::from_value(body).map_err(|error| ApiError::bad_request(error.to_string()))
}

/// Re-export the request types for the handlers.
pub type Chat = ChatCompletionRequest;
pub type Messages = MessagesRequest;
pub type Responses = ResponsesRequest;
