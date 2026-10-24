//! Client for the engine's token-in/token-out route (`POST /inference/v1/generate`).
//!
//! It sends a streaming request, parses the engine's SSE events, and maps them to the
//! deepseek-recipe [`InferenceChunk`] stream the protocol processor consumes.  Dropping
//! the stream closes the upstream response, which makes the engine cancel the request.

use std::collections::VecDeque;
use std::time::Duration;

use deepseek_recipe::stream::{
    InferenceChunk, InferenceFinishReason, InferenceTokenLogprob, PromptUsage,
};
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ApiError;

/// Sampling parameters sent to the token route.  `top_k`, `min_p` and `stop` are
/// deliberately absent: the server owns them and the gateway matches stops in content.
#[derive(Debug, Clone, Serialize)]
pub struct SamplingParams {
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u8>,
    pub detokenize: bool,
}

/// A token-in generation request.
#[derive(Debug, Clone, Serialize)]
pub struct GenerateRequest {
    pub request_id: String,
    pub model: String,
    pub token_ids: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub stream: bool,
    pub stream_options: StreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_parts: Option<Vec<Value>>,
    pub history_len: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub shared_prefixes: Vec<u32>,
    pub thinking: bool,
    /// Number of top alternatives to keep in the API response.  Not serialized: the
    /// engine receives `logprobs: top + 1`.
    #[serde(skip)]
    pub logprobs_top_n: Option<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

/// One parsed engine event and the raw token IDs it carried before the gateway trims the
/// terminal EOS from the API stream.
#[derive(Debug)]
pub struct EngineEvent {
    pub chunk: InferenceChunk,
    pub raw_ids: Vec<u32>,
}

/// A live engine stream.  Dropping it closes the HTTP response and cancels the request.
pub struct EngineStream {
    response: reqwest::Response,
    buffer: Vec<u8>,
    pending: VecDeque<EngineEvent>,
    finished: bool,
    done: bool,
    logprobs_top_n: Option<u8>,
    eos: u32,
}

impl std::fmt::Debug for EngineStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineStream")
            .field("finished", &self.finished)
            .field("done", &self.done)
            .finish()
    }
}

/// The HTTP client for the token route.
#[derive(Clone)]
pub struct EngineClient {
    base: String,
    client: reqwest::Client,
}

impl EngineClient {
    /// Build a client with the contract's 10-second connection timeout and no read
    /// timeout (long prompts take minutes).
    pub fn new(base: impl Into<String>) -> Result<Self, ApiError> {
        let base = base.into().trim_end_matches('/').to_string();
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|error| {
                ApiError::internal(format!("cannot build the engine client: {error}"))
            })?;
        Ok(Self { base, client })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    /// A cheap `/v1/models` request used by `--require-key` to let the engine decide
    /// whether the client's credentials are acceptable.
    pub async fn models(&self, headers: &HeaderMap) -> Result<(u16, Value), ApiError> {
        let mut builder = self.client.get(format!("{}/v1/models", self.base));
        for name in ["authorization", "x-api-key"] {
            if let Some(value) = headers.get(name) {
                builder = builder.header(name, value);
            }
        }
        let response = builder.send().await.map_err(|error| {
            ApiError::bad_gateway(format!("the model server is unreachable: {error}"))
        })?;
        let status = response.status().as_u16();
        let body = response.json().await.unwrap_or(Value::Null);
        Ok((status, body))
    }

    /// The shared HTTP client, used by the transparent proxy.
    pub fn http(&self) -> &reqwest::Client {
        &self.client
    }

    /// Fetch the engine's `/health`; returns its status and JSON body, or a transport
    /// error message.
    pub async fn health(&self) -> Result<(u16, Value), String> {
        let response = self
            .client
            .get(format!("{}/health", self.base))
            .send()
            .await
            .map_err(|error| error.to_string())?;
        let status = response.status().as_u16();
        let body = response.json().await.unwrap_or(Value::Null);
        Ok((status, body))
    }

    /// Start a streaming generation.  HTTP errors from the engine are translated into the
    /// API's error shape before any output is emitted.
    pub async fn start(
        &self,
        request: &GenerateRequest,
        eos: u32,
        headers: &HeaderMap,
    ) -> Result<EngineStream, ApiError> {
        let url = format!("{}/inference/v1/generate", self.base);
        let mut builder = self.client.post(&url).json(request);
        for name in ["authorization", "x-api-key"] {
            if let Some(value) = headers.get(name) {
                builder = builder.header(name, value);
            }
        }
        let response = builder.send().await.map_err(|error| {
            ApiError::bad_gateway(format!("the model server is unreachable: {error}"))
        })?;
        let status = response.status();
        if !status.is_success() {
            let body: Value = response.json().await.unwrap_or(Value::Null);
            return Err(upstream_error(status.as_u16(), &body));
        }
        Ok(EngineStream {
            response,
            buffer: Vec::new(),
            pending: VecDeque::new(),
            finished: false,
            done: false,
            logprobs_top_n: request.logprobs_top_n,
            eos,
        })
    }
}

/// Read the status of an engine error event.  The engine sends a number or a
/// string; known string codes that mean a bad request become 400.
fn error_status(code: Option<&Value>) -> u16 {
    match code {
        Some(Value::Number(number)) => number.as_u64().unwrap_or(500).min(599) as u16,
        Some(Value::String(code)) => code.parse().unwrap_or(match code.as_str() {
            "context_length_exceeded" | "invalid_request_error" => 400,
            _ => 500,
        }),
        _ => 500,
    }
}

/// Translate an engine error body into the API's shape: 4xx becomes a 400
/// `invalid_request_error`, anything else a 502 `internal_error`.
fn upstream_error(status: u16, body: &Value) -> ApiError {
    let message = body
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| body.to_string().chars().take(2000).collect());
    if (400..500).contains(&status) {
        ApiError::bad_request(message)
    } else {
        ApiError::bad_gateway(message)
    }
}

impl EngineStream {
    /// Read the next mapped event.  Returns `None` after a complete `[DONE]`.
    ///
    /// An engine error event, a malformed event, a broken connection, or a stream that
    /// ends without a stop/length finish is an [`ApiError`].
    pub async fn next_event(&mut self) -> Result<Option<EngineEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }
            if self.done {
                if !self.finished {
                    return Err(ApiError::bad_gateway(
                        "the model server ended the stream without a stop or length finish",
                    ));
                }
                return Ok(None);
            }
            let Some(chunk) = self.response.chunk().await.map_err(|error| {
                ApiError::bad_gateway(format!("the model server connection failed: {error}"))
            })?
            else {
                // A clean EOF without [DONE] is still an incomplete stream.
                self.done = true;
                continue;
            };
            self.buffer.extend_from_slice(&chunk);
            while let Some(newline) = self.buffer.iter().position(|&byte| byte == b'\n') {
                let line: Vec<u8> = self.buffer.drain(..=newline).collect();
                let line = std::str::from_utf8(&line[..line.len().saturating_sub(1)])
                    .map_err(|error| ApiError::bad_gateway(format!("invalid SSE line: {error}")))?
                    .trim_end_matches('\r');
                if line.is_empty() {
                    continue;
                }
                if let Some(data) = line.strip_prefix("data:") {
                    let data = data.trim_start();
                    if data == "[DONE]" {
                        self.done = true;
                        break;
                    }
                    let value: Value = serde_json::from_str(data).map_err(|error| {
                        ApiError::bad_gateway(format!(
                            "the model server sent a malformed event: {error}"
                        ))
                    })?;
                    if !value.is_object() {
                        return Err(ApiError::bad_gateway(
                            "the model server sent a non-object event",
                        ));
                    }
                    if let Some(error) = value.get("error") {
                        let status = error_status(error.get("code"));
                        return Err(upstream_error(status, &value));
                    }
                    let mapped = self.map_event(&value)?;
                    if mapped
                        .iter()
                        .any(|event| matches!(event.chunk, InferenceChunk::Finish { .. }))
                    {
                        self.finished = true;
                    }
                    self.pending.extend(mapped);
                }
            }
        }
    }

    /// Map one engine event object into zero or more inference chunks.
    fn map_event(&self, value: &Value) -> Result<Vec<EngineEvent>, ApiError> {
        let mut events = Vec::new();
        let usage = value.get("usage").filter(|usage| !usage.is_null());
        let choices = value
            .get("choices")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        // The first usage event precedes any token and supplies prompt usage.
        if let Some(usage) = usage {
            let prompt_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let cached = usage
                .get("prompt_tokens_details")
                .and_then(|details| details.get("cached_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            events.push(EngineEvent {
                chunk: InferenceChunk::Ready {
                    system_fingerprint: None,
                    prompt_usage: PromptUsage {
                        prompt_tokens,
                        prompt_cache_hit_tokens: cached,
                    },
                },
                raw_ids: Vec::new(),
            });
        }

        for choice in choices {
            let token_ids = choice
                .get("token_ids")
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .map(|id| {
                            id.as_u64().map(|id| id as u32).ok_or_else(|| {
                                ApiError::bad_gateway(
                                    "the model server sent a non-integer token id",
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();
            let finish = choice.get("finish_reason").and_then(Value::as_str);
            let logprobs = if token_ids.is_empty() {
                None
            } else {
                match choice.get("logprobs") {
                    Some(logprobs) if !logprobs.is_null() => {
                        Some(self.parse_logprobs(logprobs, token_ids.len())?)
                    }
                    _ => None,
                }
            };
            let raw_ids = token_ids.clone();
            let (mut token_ids, mut logprobs) = (token_ids, logprobs);
            if finish == Some("stop") && token_ids.last() == Some(&self.eos) {
                token_ids.pop();
                if let Some(logprobs) = logprobs.as_mut() {
                    logprobs.pop();
                }
            }
            let mut finish_raw = Vec::new();
            if !token_ids.is_empty() {
                events.push(EngineEvent {
                    chunk: InferenceChunk::Tokens {
                        token_ids,
                        logprobs,
                    },
                    raw_ids,
                });
            } else {
                // The only token was the terminal EOS: keep its raw ids on the finish
                // event for the token hash, without emitting an empty token chunk.
                finish_raw = raw_ids;
            }
            if let Some(finish) = finish {
                let finish_reason = match finish {
                    "stop" => InferenceFinishReason::Stop,
                    "length" => InferenceFinishReason::Length,
                    other => {
                        return Err(ApiError::bad_gateway(format!(
                            "the model server ended with finish_reason {other:?}"
                        )));
                    }
                };
                events.push(EngineEvent {
                    chunk: InferenceChunk::Finish { finish_reason },
                    raw_ids: finish_raw,
                });
            }
        }
        Ok(events)
    }

    /// Parse engine logprobs, tolerating both `token_id` integers and the vLLM
    /// `token: "token_id:N"` spelling.  Keep at most the requested number of entries.
    fn parse_logprobs(
        &self,
        value: &Value,
        expected: usize,
    ) -> Result<Vec<InferenceTokenLogprob>, ApiError> {
        let content = value
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| ApiError::bad_gateway("the model server sent malformed logprobs"))?;
        if content.len() != expected {
            return Err(ApiError::bad_gateway(format!(
                "the model server sent {} logprobs for {expected} tokens",
                content.len()
            )));
        }
        let mut parsed = Vec::with_capacity(content.len());
        for entry in content {
            let logprob = entry
                .get("logprob")
                .and_then(Value::as_f64)
                .ok_or_else(|| ApiError::bad_gateway("logprob without a number"))?
                as f32;
            let mut alternatives = Vec::new();
            if let Some(top) = entry.get("top_logprobs").and_then(Value::as_array) {
                for alternative in top {
                    let id = token_id(alternative)?;
                    let lp = alternative
                        .get("logprob")
                        .and_then(Value::as_f64)
                        .ok_or_else(|| ApiError::bad_gateway("top logprob without a number"))?
                        as f32;
                    alternatives.push((id, lp));
                }
            }
            alternatives.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let keep = self.logprobs_top_n.unwrap_or(0) as usize;
            alternatives.truncate(keep);
            parsed.push(InferenceTokenLogprob {
                logprob,
                top_logprobs: alternatives,
            });
        }
        Ok(parsed)
    }
}

/// Return the token ID of a logprob entry, accepting an integer or `token_id:N`.
fn token_id(value: &Value) -> Result<u32, ApiError> {
    if let Some(id) = value.get("token_id").and_then(Value::as_u64) {
        return Ok(id as u32);
    }
    if let Some(token) = value.get("token").and_then(Value::as_str) {
        if let Some(id) = token.strip_prefix("token_id:") {
            return id
                .parse()
                .map_err(|_| ApiError::bad_gateway(format!("invalid token id {token:?}")));
        }
        if let Ok(id) = token.parse() {
            return Ok(id);
        }
    }
    Err(ApiError::bad_gateway("logprob entry without a token id"))
}

/// Build the JSON value the engine expects.  Kept for the render endpoint and tests.
pub fn generate_body(request: &GenerateRequest) -> Value {
    serde_json::to_value(request).unwrap_or_else(|_| json!({}))
}

/// Deserialize an engine event body for diagnostics.
pub fn event_value(value: &Value) -> Option<&Value> {
    value.get("choices").map(|_| value)
}

#[derive(Debug, Deserialize)]
struct _EngineError {
    message: String,
}
