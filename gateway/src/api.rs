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
use axum::http::{HeaderValue, StatusCode};
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
use tokio::sync::mpsc;
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
}

/// The API routes, with the contract's 64 MiB body limit.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/messages", post(messages))
        .route("/v1/responses", post(responses))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}

async fn chat(State(state): State<AppState>, body: Bytes) -> Response {
    handle::<crate::parity::Chat>(state, Protocol::Chat, body).await
}

async fn messages(State(state): State<AppState>, body: Bytes) -> Response {
    handle::<crate::parity::Messages>(state, Protocol::Messages, body).await
}

async fn responses(State(state): State<AppState>, body: Bytes) -> Response {
    handle::<crate::parity::Responses>(state, Protocol::Responses, body).await
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

/// Read and validate the JSON body.
fn parse_body(body: &Bytes) -> Result<Value, ApiError> {
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
async fn handle<R>(state: AppState, protocol: Protocol, body: Bytes) -> Response
where
    R: ProtocolRequest + DeserializeOwned,
    ChunkOf<R>: Serialize + Send,
{
    let started = Instant::now();
    match serve::<R>(&state, protocol, &body, started).await {
        Ok(response) => response,
        Err(error) => error_response(&error),
    }
}

async fn serve<R>(
    state: &AppState,
    protocol: Protocol,
    body: &Bytes,
    started: Instant,
) -> Result<Response, ApiError>
where
    R: ProtocolRequest + DeserializeOwned,
    ChunkOf<R>: Serialize + Send,
{
    let value = parse_body(body)?;
    let prepared = prepare::<R>(&state.gateway, &state.config, protocol, value)?;
    if let Some(error) = prepared.context_error.clone() {
        log_line(&prepared, &Trace::default(), None, started.elapsed());
        return Err(error);
    }
    let request = generate_request(&prepared);
    let mut stream = state.engine.start(&request, state.gateway.eos).await?;
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
    let handle = tokio::spawn(async move {
        for event in initial_events {
            let finished = matches!(event.chunk, InferenceChunk::Finish { .. });
            record_event(&trace, &event);
            if tx.send(Ok(event)).await.is_err() {
                return;
            }
            if finished {
                while let Ok(Some(event)) = stream.next_event().await {
                    record_event(&trace, &event);
                }
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
                            while let Ok(Some(event)) = stream.next_event().await {
                                record_event(&trace, &event);
                            }
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
    started: Instant,
    _cached: usize,
) -> Result<Response, ApiError>
where
    R: ProtocolRequest,
    ChunkOf<R>: Serialize + Send,
{
    let (frame_tx, mut frame_rx) = mpsc::channel::<Result<Event, Infallible>>(16);
    let trace_task = Arc::clone(&trace);
    let error_task = Arc::clone(&error_slot);
    let id = prepared.id.clone();
    let route = prepared.protocol.path().to_string();
    let prompt_tokens = prepared.prompt_tokens;
    let max_tokens = prepared.max_tokens.unwrap_or(0);
    tokio::spawn(async move {
        let mut protocol_stream = Box::pin(protocol_stream);
        loop {
            tokio::select! {
                _ = frame_tx.closed() => break,
                item = protocol_stream.next() => match item {
                    Some(Ok(chunk)) => {
                        let event = protocol_event::<R>(&chunk);
                        if frame_tx.send(Ok(event)).await.is_err() {
                            break;
                        }
                    }
                    Some(Err(error)) => {
                        let _ = frame_tx.send(Ok(error_event(protocol, &ApiError::bad_gateway(error.to_string())))).await;
                        break;
                    }
                    None => break,
                }
            }
        }
        let error = error_task.lock().expect("error mutex poisoned").take();
        if let Some(error) = error {
            let _ = frame_tx.send(Ok(error_event(protocol, &error))).await;
        } else if let Some(done) = R::Response::sse_done_data() {
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
            started.elapsed().as_millis(),
        );
    });

    // Wait for the first frame, but do not hold a long prefill: after two seconds the
    // headers go out and the stream continues.
    let first = tokio::time::timeout(Duration::from_secs(2), frame_rx.recv()).await;
    let mut response = match first {
        Ok(Some(frame)) => {
            let stream = tokio_stream::once(frame).chain(ReceiverStream::new(frame_rx));
            Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response()
        }
        Ok(None) => {
            if let Some(error) = error_slot.lock().expect("error mutex poisoned").take() {
                log_line(prepared, &trace.lock().unwrap(), None, started.elapsed());
                return Err(error);
            }
            Sse::new(ReceiverStream::new(frame_rx)).into_response()
        }
        Err(_) => Sse::new(ReceiverStream::new(frame_rx)).into_response(),
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
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
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
pub fn render_handler(
    state: &AppState,
    protocol: Protocol,
    body: &Bytes,
) -> Result<Value, ApiError> {
    let value = parse_body(body)?;
    if let Some(reason) = crate::parity::detect_fallback(&value, protocol) {
        return Ok(render_fallback(&reason));
    }
    let prepared = match protocol {
        Protocol::Chat => {
            prepare::<crate::parity::Chat>(&state.gateway, &state.config, protocol, value)?
        }
        Protocol::Messages => {
            prepare::<crate::parity::Messages>(&state.gateway, &state.config, protocol, value)?
        }
        Protocol::Responses => {
            prepare::<crate::parity::Responses>(&state.gateway, &state.config, protocol, value)?
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
        prepared.fallback_reason.as_deref().unwrap_or("none"),
    );
    Ok(info)
}
