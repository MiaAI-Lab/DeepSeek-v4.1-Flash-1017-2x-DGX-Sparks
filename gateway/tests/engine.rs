//! T019: engine client mapping, SSE parsing, errors and cancellation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::post;
use ds41_gateway::engine::{EngineClient, GenerateRequest, SamplingParams, StreamOptions};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

fn request() -> GenerateRequest {
    GenerateRequest {
        request_id: "chatcmpl-test".to_string(),
        model: "DeepSeek-V4.1-Flash-1017".to_string(),
        token_ids: vec![0, 128799, 77666],
        sampling_params: SamplingParams {
            max_tokens: 64,
            temperature: Some(0.0),
            top_p: Some(1.0),
            seed: None,
            logprobs: None,
            detokenize: false,
        },
        stream: true,
        stream_options: StreamOptions {
            include_usage: true,
        },
        content_parts: None,
        history_len: 3,
        shared_prefixes: vec![512],
        thinking: true,
        logprobs_top_n: None,
    }
}

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    format!("http://{address}")
}

fn sse(events: &[Value], done: bool) -> String {
    let mut text = String::new();
    for event in events {
        text.push_str(&format!("data: {}\n\n", event));
    }
    if done {
        text.push_str("data: [DONE]\n\n");
    }
    text
}

async fn mock(
    events: Vec<Value>,
    done: bool,
    seen: Arc<tokio::sync::Mutex<Option<Value>>>,
) -> Router {
    let handler = move |State(seen): State<Arc<tokio::sync::Mutex<Option<Value>>>>, body: Bytes| async move {
        let value: Value = serde_json::from_slice(&body).expect("json request");
        *seen.lock().await = Some(value);
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from(sse(&events, done)))
            .expect("response")
    };
    Router::new()
        .route("/inference/v1/generate", post(handler))
        .with_state(seen)
}

#[tokio::test]
async fn maps_rounds_usage_and_trims_the_terminal_eos() {
    let seen = Arc::new(tokio::sync::Mutex::new(None));
    let events = vec![
        json!({"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 0, "total_tokens": 40, "prompt_tokens_details": {"cached_tokens": 32}}}),
        json!({"choices": [{"index": 0, "finish_reason": null, "token_ids": [15, 27, 33], "logprobs": null}]}),
        json!({"choices": [{"index": 0, "finish_reason": "stop", "token_ids": [9, 1], "logprobs": null}]}),
        json!({"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45, "prompt_tokens_details": {"cached_tokens": 32}}}),
    ];
    let base = spawn(mock(events, true, Arc::clone(&seen)).await).await;
    let client = EngineClient::new(base).expect("client");
    let mut stream = client
        .start(&request(), 1, &HeaderMap::new())
        .await
        .expect("start");

    let mut chunks = Vec::new();
    let mut raw = Vec::new();
    while let Some(event) = stream.next_event().await.expect("event") {
        raw.extend(event.raw_ids);
        chunks.push(event.chunk);
    }
    assert!(matches!(
        chunks[0],
        deepseek_recipe::stream::InferenceChunk::Ready { prompt_usage, .. }
            if prompt_usage.prompt_tokens == 40 && prompt_usage.prompt_cache_hit_tokens == 32
    ));
    assert!(matches!(
        chunks[1],
        deepseek_recipe::stream::InferenceChunk::Tokens { ref token_ids, .. } if token_ids == &[15, 27, 33]
    ));
    assert!(matches!(
        chunks[2],
        deepseek_recipe::stream::InferenceChunk::Tokens { ref token_ids, .. } if token_ids == &[9]
    ));
    assert!(matches!(
        chunks[3],
        deepseek_recipe::stream::InferenceChunk::Finish {
            finish_reason: deepseek_recipe::stream::InferenceFinishReason::Stop
        }
    ));
    assert_eq!(raw, vec![15, 27, 33, 9, 1]);

    let body = seen.lock().await.clone().expect("body");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["history_len"], 3);
    assert_eq!(body["shared_prefixes"], json!([512]));
    assert_eq!(body["thinking"], true);
    assert!(body["sampling_params"].get("top_k").is_none());
    assert!(body["sampling_params"].get("stop").is_none());
}

#[tokio::test]
async fn maps_length_without_trimming() {
    let seen = Arc::new(tokio::sync::Mutex::new(None));
    let events = vec![
        json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}}),
        json!({"choices": [{"index": 0, "finish_reason": "length", "token_ids": [7, 1], "logprobs": null}]}),
    ];
    let base = spawn(mock(events, true, seen).await).await;
    let client = EngineClient::new(base).expect("client");
    let mut stream = client
        .start(&request(), 1, &HeaderMap::new())
        .await
        .expect("start");
    let mut token_ids = Vec::new();
    let mut finish = None;
    while let Some(event) = stream.next_event().await.expect("event") {
        match event.chunk {
            deepseek_recipe::stream::InferenceChunk::Tokens { token_ids: ids, .. } => {
                token_ids.extend(ids)
            }
            deepseek_recipe::stream::InferenceChunk::Finish { finish_reason } => {
                finish = Some(finish_reason)
            }
            _ => {}
        }
    }
    assert_eq!(token_ids, vec![7, 1]);
    assert_eq!(
        finish,
        Some(deepseek_recipe::stream::InferenceFinishReason::Length)
    );
}

#[tokio::test]
async fn an_error_event_becomes_an_api_error() {
    let seen = Arc::new(tokio::sync::Mutex::new(None));
    let events = vec![
        json!({"error": {"message": "bad logprobs", "type": "invalid_request_error", "code": 400}}),
    ];
    let base = spawn(mock(events, true, seen).await).await;
    let client = EngineClient::new(base).expect("client");
    let mut stream = client
        .start(&request(), 1, &HeaderMap::new())
        .await
        .expect("start");
    let error = stream.next_event().await.expect_err("error");
    assert_eq!(error.status, 400);
    assert_eq!(error.body["error"]["message"], "bad logprobs");
}

#[tokio::test]
async fn a_cut_stream_without_finish_is_a_bad_gateway() {
    let seen = Arc::new(tokio::sync::Mutex::new(None));
    let events = vec![
        json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}}),
        json!({"choices": [{"index": 0, "finish_reason": null, "token_ids": [7], "logprobs": null}]}),
    ];
    let base = spawn(mock(events, false, seen).await).await;
    let client = EngineClient::new(base).expect("client");
    let mut stream = client
        .start(&request(), 1, &HeaderMap::new())
        .await
        .expect("start");
    let _ = stream.next_event().await.expect("ready");
    let _ = stream.next_event().await.expect("tokens");
    let error = stream.next_event().await.expect_err("cut stream");
    assert_eq!(error.status, 502);
}

#[tokio::test]
async fn http_500_becomes_a_bad_gateway() {
    async fn fail() -> Response {
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header("content-type", "application/json")
            .body(Body::from(r#"{"error":{"message":"engine exploded"}}"#))
            .expect("response")
    }
    let router = Router::new().route("/inference/v1/generate", post(fail));
    let base = spawn(router).await;
    let client = EngineClient::new(base).expect("client");
    let error = client
        .start(&request(), 1, &HeaderMap::new())
        .await
        .expect_err("http error");
    assert_eq!(error.status, 502);
    assert_eq!(error.body["error"]["message"], "engine exploded");
}

#[tokio::test]
async fn dropping_the_stream_closes_the_engine_connection() {
    let closed = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&closed);
    let handler = move || async move {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(8);
        let flag = Arc::clone(&flag);
        tokio::spawn(async move {
            let events = [
                "data: {\"choices\": [], \"usage\": {\"prompt_tokens\": 1}}\n\n".to_string(),
                "data: {\"choices\": [{\"finish_reason\": null, \"token_ids\": [7]}]}\n\n"
                    .to_string(),
                "data: {\"choices\": [{\"finish_reason\": \"stop\", \"token_ids\": [1]}]}\n\n"
                    .to_string(),
                "data: [DONE]\n\n".to_string(),
            ];
            for event in events {
                tokio::select! {
                    _ = tx.closed() => {
                        flag.store(true, Ordering::SeqCst);
                        return;
                    }
                    _ = tokio::time::sleep(Duration::from_millis(150)) => {}
                }
                if tx.send(Ok(Bytes::from(event))).await.is_err() {
                    flag.store(true, Ordering::SeqCst);
                    return;
                }
            }
        });
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(ReceiverStream::new(rx)))
            .expect("response")
    };
    let router = Router::new().route("/inference/v1/generate", post(handler));
    let base = spawn(router).await;
    let client = EngineClient::new(base).expect("client");
    let mut stream = client
        .start(&request(), 1, &HeaderMap::new())
        .await
        .expect("start");
    let _ = stream.next_event().await.expect("ready");
    drop(stream);
    tokio::time::timeout(Duration::from_secs(3), async {
        while !closed.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the engine saw the close");
}

#[tokio::test]
async fn tolerant_logprob_token_spellings_are_accepted_and_trimmed() {
    let seen = Arc::new(tokio::sync::Mutex::new(None));
    let events = vec![
        json!({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2}}),
        json!({"choices": [{"index": 0, "finish_reason": "stop", "token_ids": [5],
        "logprobs": {"content": [{"token": "token_id:5", "logprob": -0.1, "rank": 1,
            "top_logprobs": [
                {"token": "token_id:5", "logprob": -0.1},
                {"token_id": 9, "logprob": -0.2},
                {"token_id": 3, "logprob": -0.3}
            ]}]}}]}),
    ];
    let base = spawn(mock(events, true, seen).await).await;
    let client = EngineClient::new(base).expect("client");
    let mut request = request();
    request.sampling_params.logprobs = Some(2);
    request.logprobs_top_n = Some(1);
    let mut stream = client
        .start(&request, 1, &HeaderMap::new())
        .await
        .expect("start");
    let mut logprobs = None;
    while let Some(event) = stream.next_event().await.expect("event") {
        if let deepseek_recipe::stream::InferenceChunk::Tokens {
            logprobs: Some(found),
            ..
        } = event.chunk
        {
            logprobs = Some(found);
        }
    }
    let logprobs = logprobs.expect("logprobs");
    assert_eq!(logprobs[0].top_logprobs.len(), 1);
    assert_eq!(logprobs[0].top_logprobs[0].0, 5);
}

#[tokio::test]
async fn unreachable_engine_is_a_bad_gateway() {
    let client = EngineClient::new("http://127.0.0.1:1").expect("client");
    let error = client
        .start(&request(), 1, &HeaderMap::new())
        .await
        .expect_err("unreachable");
    assert_eq!(error.status, 502);
    assert!(
        error.body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unreachable")
    );
}

#[tokio::test]
async fn an_eos_only_final_event_does_not_emit_an_empty_token_chunk() {
    let seen = Arc::new(tokio::sync::Mutex::new(None));
    let events = vec![
        json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}}),
        json!({"choices": [{"index": 0, "finish_reason": "stop", "token_ids": [1], "logprobs": null}]}),
        json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}),
    ];
    let base = spawn(mock(events, true, seen).await).await;
    let client = EngineClient::new(base).expect("client");
    let mut stream = client
        .start(&request(), 1, &HeaderMap::new())
        .await
        .expect("start");
    let mut raw = Vec::new();
    let mut chunks = Vec::new();
    while let Some(event) = stream.next_event().await.expect("event") {
        raw.extend(event.raw_ids);
        chunks.push(event.chunk);
    }
    assert_eq!(raw, vec![1]);
    assert!(chunks.iter().all(|chunk| !matches!(
        chunk,
        deepseek_recipe::stream::InferenceChunk::Tokens { token_ids, .. } if token_ids.is_empty()
    )));
    assert!(chunks.iter().any(|chunk| matches!(
        chunk,
        deepseek_recipe::stream::InferenceChunk::Finish {
            finish_reason: deepseek_recipe::stream::InferenceFinishReason::Stop
        }
    )));
}

#[tokio::test]
async fn a_context_length_error_code_string_is_a_bad_request() {
    let seen = Arc::new(tokio::sync::Mutex::new(None));
    let events = vec![json!({"error": {"message": "too long", "code": "context_length_exceeded"}})];
    let base = spawn(mock(events, true, seen).await).await;
    let client = EngineClient::new(base).expect("client");
    let mut stream = client
        .start(&request(), 1, &HeaderMap::new())
        .await
        .expect("start");
    let error = stream.next_event().await.expect_err("error");
    assert_eq!(error.status, 400);
    assert_eq!(error.body["error"]["message"], "too long");
}
