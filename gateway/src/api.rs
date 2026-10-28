//! The three DeepSeek API routes on the gateway's API path.
//!
//! The handlers convert with deepseek-recipe, encode with the real tokenizer, send the
//! token route request, and turn the engine's token stream into the API's body or SSE
//! stream through the library's `StreamProcessor`.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use deepseek_recipe::request::ProtocolRequest;
use deepseek_recipe::response::{Accumulate, ProtocolResponse};
use deepseek_recipe::stream::{
    ChunkGenerator, InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::engine::{EngineClient, EngineEvent, EngineStream};
use crate::parity::{GatewayConfig, Prepared, Protocol, generate_request, prepare, render_info};
use crate::{ApiError, Gateway};

/// The protocol chunk type of `R`.
#[allow(type_alias_bounds)]
type ChunkOf<R: ProtocolRequest> =
    <<R::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk;

/// Shared state of the gateway's HTTP layer.
#[derive(Clone)]
pub struct AppState {
    pub gateway: Arc<Gateway>,
    pub engine: EngineClient,
    pub config: Arc<GatewayConfig>,
    pub semaphore: Arc<Semaphore>,
}

/// The API routes, with the contract's 64 MiB body limit.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/responses", post(responses))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}

async fn chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Response {
    handle::<crate::parity::Chat>(state, Protocol::Chat, headers, uri, body).await
}

async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Response {
    handle::<crate::parity::Messages>(state, Protocol::Messages, headers, uri, body).await
}

async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Response {
    handle::<crate::parity::Responses>(state, Protocol::Responses, headers, uri, body).await
}

/// `POST /v1/messages/count_tokens`: the converted prompt's input tokens.
async fn count_tokens(State(state): State<AppState>, body: Bytes) -> Response {
    let value = match parse_body(&body) {
        Ok(value) => value,
        Err(error) => return anthropic_error(&error),
    };
    match prepare::<crate::parity::Messages>(
        &state.gateway,
        &state.config,
        Protocol::Messages,
        value,
    )
    .await
    {
        Ok(prepared) => {
            let mut response = (
                StatusCode::OK,
                Json(json!({"input_tokens": prepared.prompt_tokens})),
            )
                .into_response();
            response
                .headers_mut()
                .insert("x-ds41-gateway", HeaderValue::from_static("api"));
            response
        }
        Err(error) => anthropic_error(&error),
    }
}

/// Per-request token accounting for the two token-hash headers and the log line.
#[derive(Debug, Default, Clone)]
struct Trace {
    delivered: Vec<u32>,
    raw: Vec<u32>,
    cached: usize,
    finish: Option<String>,
    prompt_tokens: usize,
}

/// The most JSON values a request body may hold. A parsed value takes 32 bytes or more whatever its text (a
/// one-digit array element is 2 bytes on the wire), so a 64 MiB body of tiny values would parse into gigabytes.
/// Real requests put their weight in strings (text, base64 images) and hold far fewer values: a conversation of ten
/// thousand messages with tools is in the tens of thousands.
pub const MAX_JSON_VALUES: usize = 1 << 20;

/// An upper bound on the number of JSON values (object keys included) in `body`, from one pass over its bytes:
/// every value but the first follows a `[`, `{`, `,` or `:` outside a string.
pub fn json_values_upper_bound(body: &[u8]) -> usize {
    let mut count = 1usize;
    let mut in_string = false;
    let mut escaped = false;
    for &byte in body {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
        } else {
            match byte {
                b'"' => in_string = true,
                b'[' | b'{' | b',' | b':' => count += 1,
                _ => {}
            }
        }
    }
    count
}

/// Read and validate the JSON body.
fn parse_body(body: &Bytes) -> Result<Value, ApiError> {
    let values = json_values_upper_bound(body);
    if values > MAX_JSON_VALUES {
        return Err(ApiError::bad_request(format!(
            "the request body holds about {values} JSON values; this gateway takes at most {MAX_JSON_VALUES}"
        )));
    }
    let value: Value = serde_json::from_slice(body)
        .map_err(|error| ApiError::bad_request(format!("invalid JSON body: {error}")))?;
    if !value.is_object() {
        return Err(ApiError::bad_request(
            "the request body must be a JSON object",
        ));
    }
    Ok(value)
}

/// Serve one protocol request.
async fn handle<R>(
    state: AppState,
    protocol: Protocol,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Response
where
    R: ProtocolRequest + DeserializeOwned,
    ChunkOf<R>: Serialize + Send,
{
    let started = Instant::now();
    match serve::<R>(&state, protocol, &headers, &uri, &body, started).await {
        Ok(response) => response,
        Err(error) => error_response(&error),
    }
}

async fn serve<R>(
    state: &AppState,
    protocol: Protocol,
    headers: &HeaderMap,
    uri: &Uri,
    body: &Bytes,
    started: Instant,
) -> Result<Response, ApiError>
where
    R: ProtocolRequest + DeserializeOwned,
    ChunkOf<R>: Serialize + Send,
{
    let value = parse_body(body)?;
    if let Some(reason) =
        crate::parity::detect_fallback(&value, protocol, state.config.engine_logprobs)
    {
        if state.config.strict {
            return Err(ApiError::bad_request(reason));
        }
        return Ok(fallback_response(state, protocol, headers, uri, body, &reason, started).await);
    }
    let prepared = match prepare::<R>(&state.gateway, &state.config, protocol, value).await {
        Ok(prepared) => prepared,
        Err(error) => {
            if let Some(reason) = error.fallback_reason.clone() {
                return Ok(fallback_response(
                    state, protocol, headers, uri, body, &reason, started,
                )
                .await);
            }
            return Err(error);
        }
    };
    if let Some(reason) = prepared.fallback_reason.clone() {
        return Ok(fallback_response(state, protocol, headers, uri, body, &reason, started).await);
    }
    if let Some(error) = prepared.context_error.clone() {
        log_line(&prepared, &Trace::default(), None, started.elapsed());
        return Err(error);
    }
    let request = generate_request(&prepared);
    let mut stream = state
        .engine
        .start(&request, state.gateway.eos, headers)
        .await?;
    let first = match stream.next_event().await? {
        Some(event) => event,
        None => {
            return Err(ApiError::bad_gateway(
                "the model server ended the stream without output",
            ));
        }
    };
    let mut initial_cached = 0;
    let mut initial_prompt = prepared.prompt_tokens as usize;
    let mut initial_events = Vec::new();
    if let InferenceChunk::Ready { prompt_usage, .. } = &first.chunk {
        initial_cached = prompt_usage.prompt_cache_hit_tokens;
        initial_prompt = prompt_usage.prompt_tokens;
        initial_events.push(first);
    } else {
        // The contract puts usage first; tolerate an engine that starts with tokens by
        // prepending a Ready with the prompt usage the gateway computed, keeping the
        // original token event.
        initial_events.push(EngineEvent {
            chunk: InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage {
                    prompt_tokens: prepared.prompt_tokens as usize,
                    prompt_cache_hit_tokens: 0,
                },
            },
            raw_ids: Vec::new(),
        });
        initial_events.push(first);
    }

    let trace = Arc::new(Mutex::new(Trace {
        cached: initial_cached,
        prompt_tokens: initial_prompt,
        ..Trace::default()
    }));
    let error_slot = Arc::new(Mutex::new(None::<ApiError>));
    let (inference, engine_task) = inference_stream(
        stream,
        initial_events,
        Arc::clone(&trace),
        Arc::clone(&error_slot),
    );

    let generator = R::chunk_generator(
        &prepared.converted,
        prepared.id.clone(),
        prepared.model.clone(),
    );
    let processor = StreamProcessor::new(generator, prepared.converted.parsing_options.clone())
        .with_tokenizer(state.gateway.tokenizer.clone());
    let protocol_stream = processor.into_protocol_stream(inference);

    let wants_stream = prepared.converted.stream;
    let cached = trace.lock().expect("trace mutex poisoned").cached;
    if wants_stream {
        return stream_response::<R>(
            protocol,
            &prepared,
            protocol_stream,
            trace,
            error_slot,
            engine_task,
            started,
            cached,
        )
        .await;
    }

    let mut response = R::Response::new(
        prepared.id.clone(),
        prepared.model.clone(),
        unix_now(),
        prepared.prompt_tokens as usize,
        cached,
    );
    let mut protocol_stream = Box::pin(protocol_stream);
    while let Some(item) = protocol_stream.next().await {
        match item {
            Ok(chunk) => response.accumulate(chunk),
            Err(error) => {
                log_line(&prepared, &trace.lock().unwrap(), None, started.elapsed());
                return Err(ApiError::bad_gateway(error.to_string()));
            }
        }
    }
    // Wait for the engine task to drain the final usage snapshot before reading the
    // cache-hit trace.
    let _ = engine_task.await;
    if let Some(error) = error_slot.lock().expect("error mutex poisoned").take() {
        log_line(&prepared, &trace.lock().unwrap(), None, started.elapsed());
        return Err(error);
    }
    let trace = trace.lock().expect("trace mutex poisoned").clone();
    let mut value = serde_json::to_value(&response)
        .map_err(|error| ApiError::internal(format!("cannot serialize the response: {error}")))?;
    patch_usage(&mut value, protocol, trace.cached, trace.prompt_tokens);
    let mut http = (StatusCode::OK, Json(value)).into_response();
    http.headers_mut()
        .insert("x-ds41-gateway", HeaderValue::from_static("api"));
    http.headers_mut().insert(
        "x-ds41-token-sha",
        HeaderValue::from_str(&token_sha(&trace.delivered))
            .map_err(|error| ApiError::internal(error.to_string()))?,
    );
    http.headers_mut().insert(
        "x-ds41-token-sha-raw",
        HeaderValue::from_str(&token_sha(&trace.raw))
            .map_err(|error| ApiError::internal(error.to_string()))?,
    );
    log_line(
        &prepared,
        &trace,
        trace.finish.as_deref(),
        started.elapsed(),
    );
    Ok(http)
}

/// Update the per-request trace from one engine event.
fn record_event(trace: &Arc<Mutex<Trace>>, event: &EngineEvent) {
    let mut trace = trace.lock().expect("trace mutex poisoned");
    trace.raw.extend_from_slice(&event.raw_ids);
    match &event.chunk {
        InferenceChunk::Tokens { token_ids, .. } => {
            trace.delivered.extend_from_slice(token_ids);
        }
        InferenceChunk::Ready { prompt_usage, .. } => {
            trace.cached = prompt_usage.prompt_cache_hit_tokens;
            trace.prompt_tokens = prompt_usage.prompt_tokens;
        }
        InferenceChunk::Finish { finish_reason } => {
            trace.finish = Some(match finish_reason {
                InferenceFinishReason::Stop => "stop".to_string(),
                InferenceFinishReason::Length => "length".to_string(),
                InferenceFinishReason::ContentFilter => "content_filter".to_string(),
            });
        }
        InferenceChunk::Text { .. } => {}
    }
}

/// Turn the engine event stream into inference chunks, recording token ids and the last
/// cached usage for the response headers and log line.
fn inference_stream(
    mut stream: EngineStream,
    initial_events: Vec<EngineEvent>,
    trace: Arc<Mutex<Trace>>,
    error_slot: Arc<Mutex<Option<ApiError>>>,
) -> (
    impl tokio_stream::Stream<Item = InferenceChunk> + Send,
    tokio::task::JoinHandle<()>,
) {
    let (tx, rx) = mpsc::channel::<Result<EngineEvent, ApiError>>(32);
    let error_slot_task = Arc::clone(&error_slot);
    let handle = tokio::spawn(async move {
        for event in initial_events {
            let finished = matches!(event.chunk, InferenceChunk::Finish { .. });
            record_event(&trace, &event);
            if tx.send(Ok(event)).await.is_err() {
                return;
            }
            if finished {
                drain_final(&mut stream, &trace, &error_slot_task).await;
                return;
            }
        }
        loop {
            tokio::select! {
                _ = tx.closed() => return,
                event = stream.next_event() => match event {
                    Ok(Some(event)) => {
                        let finished = matches!(event.chunk, InferenceChunk::Finish { .. });
                        record_event(&trace, &event);
                        if tx.send(Ok(event)).await.is_err() {
                            return;
                        }
                        if finished {
                            // The final usage snapshot follows the finish event.  Drain it
                            // for the cache-hit patch even though the processor already
                            // stopped consuming the stream.
                            drain_final(&mut stream, &trace, &error_slot_task).await;
                            return;
                        }
                    }
                    Ok(None) => return,
                    Err(error) => {
                        let _ = tx.send(Err(error)).await;
                        return;
                    }
                }
            }
        }
    });
    let stream = ReceiverStream::new(rx).filter_map(move |item| match item {
        Ok(event) => Some(event.chunk),
        Err(error) => {
            *error_slot.lock().expect("error mutex poisoned") = Some(error);
            None
        }
    });
    (stream, handle)
}

/// Drain the engine events that follow a finish chunk, recording them for the trace.
/// A late engine error is recorded into `error_slot` instead of being dropped, so the
/// request does not close as successful.
async fn drain_final(
    stream: &mut EngineStream,
    trace: &Arc<Mutex<Trace>>,
    error_slot: &Arc<Mutex<Option<ApiError>>>,
) {
    loop {
        match stream.next_event().await {
            Ok(Some(event)) => record_event(trace, &event),
            Ok(None) => return,
            Err(error) => {
                *error_slot.lock().expect("error mutex poisoned") = Some(error);
                return;
            }
        }
    }
}

/// Build the SSE response, waiting up to two seconds for the first protocol frame so an
/// engine rejection is still a normal HTTP error.
#[allow(clippy::too_many_arguments)]
async fn stream_response<R>(
    protocol: Protocol,
    prepared: &Prepared,
    protocol_stream: impl tokio_stream::Stream<
        Item = Result<ChunkOf<R>, deepseek_recipe::stream::StreamError>,
    > + Send
    + 'static,
    trace: Arc<Mutex<Trace>>,
    error_slot: Arc<Mutex<Option<ApiError>>>,
    engine_task: tokio::task::JoinHandle<()>,
    started: Instant,
    _cached: usize,
) -> Result<Response, ApiError>
where
    R: ProtocolRequest,
    ChunkOf<R>: Serialize + Send,
{
    let (frame_tx, mut frame_rx) = mpsc::channel::<Result<Event, ApiError>>(16);
    let trace_task = Arc::clone(&trace);
    let error_task = Arc::clone(&error_slot);
    let id = prepared.id.clone();
    let route = prepared.protocol.path().to_string();
    let prompt_tokens = prepared.prompt_tokens;
    let max_tokens = prepared.max_tokens.unwrap_or(0);
    tokio::spawn(async move {
        let mut protocol_stream = Box::pin(protocol_stream);
        let mut sent = false;
        let mut failed = false;
        loop {
            tokio::select! {
                _ = frame_tx.closed() => break,
                item = protocol_stream.next() => match item {
                    Some(Ok(chunk)) => {
                        let event = protocol_event::<R>(&chunk);
                        if frame_tx.send(Ok(event)).await.is_err() {
                            break;
                        }
                        sent = true;
                    }
                    Some(Err(error)) => {
                        let error = ApiError::bad_gateway(error.to_string());
                        let _ = if sent {
                            frame_tx.send(Ok(error_event(protocol, &error))).await
                        } else {
                            frame_tx.send(Err(error)).await
                        };
                        failed = true;
                        break;
                    }
                    None => break,
                }
            }
        }
        // Drop the protocol stream, and with it the receiver the engine task sends to: a
        // task still sending (the client left mid-reply, or the processor stopped early)
        // then sees the channel closed and ends, closing the engine stream, instead of
        // waiting on a channel nobody reads while this task waits on it.
        drop(protocol_stream);
        // Wait for the engine task to finish draining after the finish chunk, so a
        // late engine error is not masked by [DONE] or by a successful close; but a
        // client that leaves meanwhile ends that drain too (aborting the task closes the
        // engine stream), since nobody is left to tell.
        let mut engine_task = engine_task;
        tokio::select! {
            _ = &mut engine_task => {}
            _ = frame_tx.closed() => {
                engine_task.abort();
                let _ = engine_task.await;
            }
        }
        let error = error_task.lock().expect("error mutex poisoned").take();
        if let Some(error) = error {
            let _ = if sent {
                frame_tx.send(Ok(error_event(protocol, &error))).await
            } else {
                frame_tx.send(Err(error)).await
            };
        } else if !failed {
            // The gateway terminates all three SSE protocols with [DONE], but only
            // when nothing already reported an error.
            // deepseek-recipe only provides the marker explicitly for Chat.
            let done = R::Response::sse_done_data().unwrap_or_else(|| "[DONE]".to_string());
            let _ = frame_tx.send(Ok(Event::default().data(done))).await;
        }
        let trace = trace_task.lock().expect("trace mutex poisoned");
        eprintln!(
            "{} {} camino=api prompt_tokens={} cached={} max_tokens={} finish={} ms={}",
            id,
            route,
            prompt_tokens,
            trace.cached,
            max_tokens,
            trace.finish.as_deref().unwrap_or("cut"),
            started.elapsed().as_millis()
        );
    });

    let sse = |rx: mpsc::Receiver<Result<Event, ApiError>>| {
        ReceiverStream::new(rx).map(move |item| match item {
            Ok(event) => Ok::<Event, Infallible>(event),
            Err(error) => Ok(error_event(protocol, &error)),
        })
    };

    // Wait for the first frame, but do not hold a long prefill: after two seconds the
    // headers go out and the stream continues.  A rejection before the first frame is
    // still a normal HTTP error, not an error inside a 200 SSE stream.  Every stream that
    // goes out keeps itself alive with a comment every 15 s, the slow start included.
    let first = tokio::time::timeout(Duration::from_secs(2), frame_rx.recv()).await;
    let mut response = match first {
        Ok(Some(Ok(frame))) => {
            let stream = tokio_stream::once(Ok::<Event, Infallible>(frame)).chain(sse(frame_rx));
            Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response()
        }
        Ok(Some(Err(error))) => {
            return Err(error);
        }
        Ok(None) => {
            if let Some(error) = error_slot.lock().expect("error mutex poisoned").take() {
                return Err(error);
            }
            Sse::new(sse(frame_rx))
                .keep_alive(KeepAlive::default())
                .into_response()
        }
        Err(_) => Sse::new(sse(frame_rx))
            .keep_alive(KeepAlive::default())
            .into_response(),
    };
    response
        .headers_mut()
        .insert("x-ds41-gateway", HeaderValue::from_static("api"));
    response
        .headers_mut()
        .insert("cache-control", HeaderValue::from_static("no-cache"));
    Ok(response)
}

/// Serialize one protocol chunk as an SSE event with the protocol's event name.
fn protocol_event<R>(chunk: &ChunkOf<R>) -> Event
where
    R: ProtocolRequest,
    ChunkOf<R>: Serialize + Send,
{
    let data = serde_json::to_string(chunk).unwrap_or_else(|_| "{}".to_string());
    let mut event = Event::default();
    if let Some(name) = R::Response::sse_event_name(chunk) {
        event = event.event(name);
    }
    event.data(data)
}

/// The API-shaped error frame for a stream that already sent its headers.
fn error_event(protocol: Protocol, error: &ApiError) -> Event {
    let data = error.body.to_string();
    match protocol {
        Protocol::Chat => Event::default().data(data),
        Protocol::Messages | Protocol::Responses => Event::default().event("error").data(data),
    }
}

/// Patch the usage snapshot with the engine's prefix-cache hits.
fn patch_usage(value: &mut Value, protocol: Protocol, cached: usize, prompt: usize) {
    let cached = cached.min(prompt);
    let usage = match value.get_mut("usage") {
        Some(usage) => usage,
        None => return,
    };
    match protocol {
        Protocol::Chat => {
            usage["prompt_cache_hit_tokens"] = Value::from(cached);
            usage["prompt_cache_miss_tokens"] = Value::from(prompt.saturating_sub(cached));
            usage["prompt_tokens_details"]["cached_tokens"] = Value::from(cached);
        }
        Protocol::Messages => {
            usage["cache_read_input_tokens"] = Value::from(cached);
            usage["input_tokens"] = Value::from(prompt.saturating_sub(cached));
        }
        Protocol::Responses => {
            usage["input_tokens_details"]["cached_tokens"] = Value::from(cached);
        }
    }
}

/// SHA-256 of the comma-joined token IDs, as the token route computes `token_sha`.
fn token_sha(ids: &[u32]) -> String {
    let joined = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let digest = Sha256::digest(joined.as_bytes());
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    hex[..12].to_string()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn error_response(error: &ApiError) -> Response {
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut response = (status, Json(error.body.clone())).into_response();
    response
        .headers_mut()
        .insert("x-ds41-gateway", HeaderValue::from_static("api"));
    response
}

/// The Anthropic error shape used by `count_tokens`.
fn anthropic_error(error: &ApiError) -> Response {
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let message = error
        .body
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("invalid request")
        .to_string();
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "api_error"
    };
    let mut response = (
        status,
        Json(json!({
            "type": "error",
            "error": {"type": kind, "message": message}
        })),
    )
        .into_response();
    response
        .headers_mut()
        .insert("x-ds41-gateway", HeaderValue::from_static("api"));
    response
}

/// Send a request that must go to the engine untouched and mark the response.
async fn fallback_response(
    state: &AppState,
    protocol: Protocol,
    headers: &HeaderMap,
    uri: &Uri,
    body: &Bytes,
    reason: &str,
    started: Instant,
) -> Response {
    eprintln!(
        "- {} camino=fallback motivo={} ms={}",
        protocol.path(),
        log_text(reason),
        started.elapsed().as_millis()
    );
    let path_and_query = uri
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_else(|| protocol.path().to_string());
    crate::proxy::forward(
        state,
        Method::POST,
        &path_and_query,
        headers.clone(),
        body.clone(),
        Some(("x-ds41-gateway", "fallback")),
    )
    .await
}

/// One stderr line per request: no request content and no URLs with credentials.
fn log_line(prepared: &Prepared, trace: &Trace, finish: Option<&str>, elapsed: Duration) {
    eprintln!(
        "{} {} camino=api prompt_tokens={} cached={} max_tokens={} finish={} ms={}",
        prepared.id,
        prepared.protocol.path(),
        trace.prompt_tokens,
        trace.cached,
        prepared.max_tokens.unwrap_or(0),
        finish.unwrap_or("cut"),
        elapsed.as_millis()
    );
}

/// Log a render that falls back before conversion, without request content.
fn log_render_fallback(protocol: Protocol, reason: &str) {
    eprintln!(
        "- {} camino=fallback render motivo={}",
        protocol.path(),
        log_text(reason)
    );
}

/// The longest piece of client text a log line carries.
const LOG_TEXT_CHARS: usize = 200;

/// Client-controlled text (a fallback reason can name a request field) as one bounded
/// log line: control characters and quotes escaped, cut at [`LOG_TEXT_CHARS`].
fn log_text(text: &str) -> String {
    let mut out: String = text
        .chars()
        .take(LOG_TEXT_CHARS)
        .flat_map(char::escape_debug)
        .collect();
    if text.chars().nth(LOG_TEXT_CHARS).is_some() {
        out.push_str("...");
    }
    out
}

/// The render body when a request must go to the engine untouched.
fn render_fallback(reason: &str) -> Value {
    json!({
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
    })
}

/// Render endpoint helper shared with T021.
pub async fn render_handler(
    state: &AppState,
    protocol: Protocol,
    body: &Bytes,
) -> Result<Value, ApiError> {
    let value = parse_body(body)?;
    if let Some(reason) =
        crate::parity::detect_fallback(&value, protocol, state.config.engine_logprobs)
    {
        log_render_fallback(protocol, &reason);
        if state.config.strict {
            return Err(ApiError::bad_request(reason));
        }
        return Ok(render_fallback(&reason));
    }
    let prepared = match match protocol {
        Protocol::Chat => {
            prepare::<crate::parity::Chat>(&state.gateway, &state.config, protocol, value).await
        }
        Protocol::Messages => {
            prepare::<crate::parity::Messages>(&state.gateway, &state.config, protocol, value).await
        }
        Protocol::Responses => {
            prepare::<crate::parity::Responses>(&state.gateway, &state.config, protocol, value)
                .await
        }
    } {
        Ok(prepared) => prepared,
        Err(error) => {
            if let Some(reason) = error.fallback_reason.clone() {
                log_render_fallback(protocol, &reason);
                return Ok(render_fallback(&reason));
            }
            return Err(error);
        }
    };
    let info = render_info(&prepared);
    eprintln!(
        "{} {} camino=api render prompt_tokens={} max_tokens={} fallback={}",
        prepared.id,
        prepared.protocol.path(),
        prepared.prompt_tokens,
        prepared
            .max_tokens
            .map_or_else(|| "null".to_string(), |value| value.to_string()),
        log_text(prepared.fallback_reason.as_deref().unwrap_or("none")),
    );
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GatewaySettings;

    #[test]
    fn client_text_in_a_log_line_is_one_bounded_line() {
        assert_eq!(
            log_text("chat_template_kwargs field x"),
            "chat_template_kwargs field x"
        );
        let forged = log_text("field a\n- /v1/messages camino=api forged=1\r");
        assert!(!forged.contains('\n') && !forged.contains('\r'), "{forged}");
        assert!(forged.contains("\\n"), "{forged}");
        let long = log_text(&"k".repeat(10_000));
        assert_eq!(long.len(), LOG_TEXT_CHARS + 3);
        assert!(long.ends_with("..."));
    }
    use deepseek_recipe::stream::InferenceTokenLogprob;

    fn test_gateway() -> Gateway {
        Gateway::new(
            Arc::new(crate::test_tokenizer()),
            GatewaySettings {
                served_name: "DeepSeek-V4.1-Flash-1017".to_string(),
                default_thinking: true,
                image_token: "<｜deepseek_image｜>".to_string(),
                aliases: Vec::new(),
            },
        )
        .expect("build gateway")
    }

    /// Drive `stream_response` with a processor error after output has already started,
    /// returning the raw SSE body.
    async fn sse_text_after_a_processor_error<R>(protocol: Protocol, body: Value) -> String
    where
        R: ProtocolRequest + DeserializeOwned,
        ChunkOf<R>: Serialize + Send,
    {
        let gateway = test_gateway();
        let config = GatewayConfig {
            api_sampling: true,
            ..GatewayConfig::default()
        };
        let prepared = prepare::<R>(&gateway, &config, protocol, body)
            .await
            .expect("prepare");
        let generator = R::chunk_generator(
            &prepared.converted,
            prepared.id.clone(),
            prepared.model.clone(),
        );
        let processor = StreamProcessor::new(generator, prepared.converted.parsing_options.clone())
            .with_tokenizer(gateway.tokenizer.clone());
        // A Ready chunk emits the start frame (so `sent` becomes true); the second
        // Tokens chunk carries more log probabilities than token ids, which makes the
        // processor yield a StreamError after output has already begun.
        let inference = tokio_stream::iter(vec![
            InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage {
                    prompt_tokens: 1,
                    prompt_cache_hit_tokens: 0,
                },
            },
            InferenceChunk::Tokens {
                token_ids: vec![0],
                logprobs: Some(vec![
                    InferenceTokenLogprob {
                        logprob: 0.0,
                        top_logprobs: Vec::new(),
                    },
                    InferenceTokenLogprob {
                        logprob: 0.0,
                        top_logprobs: Vec::new(),
                    },
                ]),
            },
        ]);
        let protocol_stream = processor.into_protocol_stream(inference);
        let trace = Arc::new(Mutex::new(Trace::default()));
        let error_slot = Arc::new(Mutex::new(None::<ApiError>));
        let engine_task = tokio::spawn(async {});
        let response = stream_response::<R>(
            protocol,
            &prepared,
            protocol_stream,
            trace,
            error_slot,
            engine_task,
            Instant::now(),
            0,
        )
        .await
        .expect("stream_response");
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024 * 1024)
            .await
            .expect("sse body");
        String::from_utf8_lossy(&bytes).to_string()
    }

    #[tokio::test]
    async fn messages_stream_does_not_emit_done_after_a_processor_error() {
        let text = sse_text_after_a_processor_error::<crate::parity::Messages>(
            Protocol::Messages,
            json!({
                "model": "DeepSeek-V4.1-Flash-1017",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Hola"}],
                "thinking": {"type": "disabled"}
            }),
        )
        .await;
        assert!(text.contains("error"), "no error frame in stream: {text}");
        assert!(
            !text.contains("[DONE]"),
            "stream emitted [DONE] after the error: {text}"
        );
    }

    #[tokio::test]
    async fn responses_stream_does_not_emit_done_after_a_processor_error() {
        let text = sse_text_after_a_processor_error::<crate::parity::Responses>(
            Protocol::Responses,
            json!({
                "model": "DeepSeek-V4.1-Flash-1017",
                "input": "Hola",
                "max_output_tokens": 64,
                "reasoning": {"effort": "none"}
            }),
        )
        .await;
        assert!(text.contains("error"), "no error frame in stream: {text}");
        assert!(
            !text.contains("[DONE]"),
            "stream emitted [DONE] after the error: {text}"
        );
    }
}
