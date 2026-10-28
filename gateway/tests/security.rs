//! Router-wide authentication and admission-control regressions.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use axum::{Json, Router};
use ds41_gateway::api::AppState;
use ds41_gateway::engine::EngineClient;
use ds41_gateway::health;
use ds41_gateway::parity::GatewayConfig;
use ds41_gateway::security::ApiKeys;
use ds41_gateway::{Gateway, GatewaySettings};
use serde_json::{Value, json};
use tokio::sync::{Notify, Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;

const KEY: &str = "test-secret";
const MODEL: &str = "DeepSeek-V4.1-Flash-1017";

fn gateway() -> Gateway {
    Gateway::new(
        Arc::new(ds41_gateway::test_tokenizer()),
        GatewaySettings {
            served_name: MODEL.to_string(),
            default_thinking: true,
            image_token: "<｜deepseek_image｜>".to_string(),
            aliases: Vec::new(),
        },
    )
    .expect("gateway")
}

fn secured_config() -> GatewayConfig {
    GatewayConfig {
        require_key: true,
        api_keys: Some(ApiKeys::parse(&format!("{KEY}\n")).expect("key")),
        ..GatewayConfig::default()
    }
}

async fn serve(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    format!("http://{address}")
}

async fn spawn_gateway(engine: &str, config: GatewayConfig, semaphore: Arc<Semaphore>) -> String {
    let state = AppState {
        gateway: Arc::new(gateway()),
        engine: EngineClient::new(engine).expect("engine client"),
        config: Arc::new(config),
        semaphore,
    };
    serve(health::router(state)).await
}

fn usage(completion: u64) -> Value {
    json!({
        "choices": [],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": completion,
            "total_tokens": 1 + completion
        }
    })
}

fn tokens(finish: Option<&str>) -> Value {
    json!({
        "choices": [{
            "index": 0,
            "token_ids": [1],
            "finish_reason": finish
        }]
    })
}

fn event_text(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
}

fn events(events: &[Value]) -> String {
    let mut body = event_text(events);
    body.push_str("data: [DONE]\n\n");
    body
}

fn complete_sse() -> String {
    events(&[usage(0), tokens(Some("stop")), usage(1)])
}

async fn spawn_basic_engine() -> (String, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let handler = |State(calls): State<Arc<AtomicUsize>>, request: Request| async move {
        calls.fetch_add(1, Ordering::SeqCst);
        if request.uri().path() == "/health" {
            return Json(json!({"status": "ok"})).into_response();
        }
        if request.uri().path() == "/inference/v1/generate" {
            return Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from(complete_sse()))
                .expect("SSE response");
        }
        Json(json!({"data": [], "ok": true})).into_response()
    };
    let router = Router::new()
        .fallback(any(handler))
        .with_state(Arc::clone(&calls));
    (serve(router).await, calls)
}

fn chat_body() -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hola"}],
        "thinking": {"type": "disabled"}
    })
}

fn messages_body() -> Value {
    json!({
        "model": MODEL,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Hola"}]
    })
}

fn responses_body() -> Value {
    json!({
        "model": MODEL,
        "input": "Hola",
        "max_output_tokens": 8,
        "reasoning": {"effort": "none"}
    })
}

fn post_cases() -> [(&'static str, Value); 5] {
    [
        ("/v1/chat/completions", chat_body()),
        ("/v1/messages", messages_body()),
        ("/v1/messages/count_tokens", messages_body()),
        ("/v1/responses", responses_body()),
        ("/ds41/render/chat", chat_body()),
    ]
}

async fn expect_status(response: reqwest::Response, expected: StatusCode) {
    let actual = response.status();
    let body = response.text().await.unwrap_or_default();
    assert_eq!(actual, expected, "response body: {body}");
}

#[tokio::test]
async fn valid_bearer_and_x_api_key_cover_every_route_kind() {
    let (engine, _calls) = spawn_basic_engine().await;
    let url = spawn_gateway(&engine, secured_config(), Arc::new(Semaphore::new(4))).await;
    let client = reqwest::Client::new();
    for (path, body) in post_cases() {
        let response = client
            .post(format!("{url}{path}"))
            .bearer_auth(KEY)
            .json(&body)
            .send()
            .await
            .expect(path);
        expect_status(response, StatusCode::OK).await;
    }
    let response = client
        .get(format!("{url}/v1/models"))
        .header("x-api-key", KEY)
        .send()
        .await
        .expect("proxy");
    expect_status(response, StatusCode::OK).await;
}

#[tokio::test]
async fn missing_and_wrong_keys_are_401_before_every_handler() {
    let (engine, calls) = spawn_basic_engine().await;
    let url = spawn_gateway(&engine, secured_config(), Arc::new(Semaphore::new(4))).await;
    let client = reqwest::Client::new();
    for (path, body) in post_cases() {
        for credential in [None, Some("wrong")] {
            let mut request = client.post(format!("{url}{path}")).json(&body);
            if let Some(credential) = credential {
                request = request.bearer_auth(credential);
            }
            let response = request.send().await.expect(path);
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
            let value: Value = response.json().await.expect("error JSON");
            assert_eq!(value["error"]["type"], "authentication_error");
        }
    }
    for credential in [None, Some("wrong")] {
        let mut request = client.get(format!("{url}/v1/models"));
        if let Some(credential) = credential {
            request = request.header("x-api-key", credential);
        }
        let response = request.send().await.expect("proxy");
        expect_status(response, StatusCode::UNAUTHORIZED).await;
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn invalid_json_with_a_bad_key_is_401_not_400() {
    let (engine, calls) = spawn_basic_engine().await;
    let url = spawn_gateway(&engine, secured_config(), Arc::new(Semaphore::new(4))).await;
    let client = reqwest::Client::new();
    for (path, _body) in post_cases() {
        let response = client
            .post(format!("{url}{path}"))
            .bearer_auth("wrong")
            .header("content-type", "application/json")
            .body("{not JSON")
            .send()
            .await
            .expect(path);
        expect_status(response, StatusCode::UNAUTHORIZED).await;
    }
    let response = client
        .post(format!("{url}/v1/echo"))
        .bearer_auth("wrong")
        .body("{not JSON")
        .send()
        .await
        .expect("proxy");
    expect_status(response, StatusCode::UNAUTHORIZED).await;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn only_get_health_bypasses_authentication() {
    let (engine, _calls) = spawn_basic_engine().await;
    let url = spawn_gateway(&engine, secured_config(), Arc::new(Semaphore::new(1))).await;
    let client = reqwest::Client::new();
    let response = client
        .get(format!("{url}/health"))
        .send()
        .await
        .expect("health");
    expect_status(response, StatusCode::OK).await;
    let response = client
        .get(format!("{url}/health"))
        .bearer_auth("wrong")
        .send()
        .await
        .expect("health with a bad key");
    expect_status(response, StatusCode::OK).await;
    let response = client
        .post(format!("{url}/health"))
        .send()
        .await
        .expect("POST health");
    expect_status(response, StatusCode::UNAUTHORIZED).await;
}

#[tokio::test]
async fn unauthorized_response_does_not_wait_for_the_request_body() {
    let (engine, calls) = spawn_basic_engine().await;
    let url = spawn_gateway(&engine, secured_config(), Arc::new(Semaphore::new(1))).await;
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(1);
    let body = reqwest::Body::wrap_stream(ReceiverStream::new(receiver));
    let request = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .bearer_auth("wrong")
        .header("content-type", "application/json")
        .body(body)
        .send();
    let response = tokio::time::timeout(Duration::from_secs(2), request)
        .await
        .expect("401 must not wait for EOF on the body")
        .expect("response");
    expect_status(response, StatusCode::UNAUTHORIZED).await;
    drop(sender);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

struct QueueEngine {
    calls: AtomicUsize,
    active: AtomicUsize,
    peak: AtomicUsize,
    first_entered: Notify,
    release_first: Notify,
}

async fn spawn_queue_engine() -> (String, Arc<QueueEngine>) {
    let state = Arc::new(QueueEngine {
        calls: AtomicUsize::new(0),
        active: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
        first_entered: Notify::new(),
        release_first: Notify::new(),
    });
    let handler = |State(state): State<Arc<QueueEngine>>, _body: Bytes| async move {
        let ordinal = state.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
        state.peak.fetch_max(active, Ordering::SeqCst);
        if ordinal == 1 {
            state.first_entered.notify_one();
            state.release_first.notified().await;
        }
        state.active.fetch_sub(1, Ordering::SeqCst);
        Json(json!({"ok": true})).into_response()
    };
    let router = Router::new()
        .fallback(any(handler))
        .with_state(Arc::clone(&state));
    (serve(router).await, state)
}

fn spawn_proxy_request(
    client: reqwest::Client,
    url: String,
) -> tokio::task::JoinHandle<StatusCode> {
    tokio::spawn(async move {
        let response = client
            .post(url)
            .json(&json!({"value": "queued"}))
            .send()
            .await
            .expect("request");
        let status = response.status();
        response.bytes().await.expect("response body");
        status
    })
}

#[tokio::test]
async fn excess_requests_queue_and_never_overlap_in_the_engine() {
    let (engine, state) = spawn_queue_engine().await;
    let semaphore = Arc::new(Semaphore::new(1));
    let url = spawn_gateway(&engine, GatewayConfig::default(), Arc::clone(&semaphore)).await;
    let client = reqwest::Client::new();
    let first = spawn_proxy_request(client.clone(), format!("{url}/v1/echo"));
    tokio::time::timeout(Duration::from_secs(2), state.first_entered.notified())
        .await
        .expect("first request enters the engine");
    let second = spawn_proxy_request(client, format!("{url}/v1/echo"));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!second.is_finished());
    assert_eq!(state.calls.load(Ordering::SeqCst), 1);

    state.release_first.notify_one();
    let first = tokio::time::timeout(Duration::from_secs(5), first)
        .await
        .expect("first completes")
        .expect("first task");
    let second = tokio::time::timeout(Duration::from_secs(5), second)
        .await
        .expect("queued request completes")
        .expect("second task");
    assert_eq!(first, StatusCode::OK);
    assert_eq!(second, StatusCode::OK);
    assert_eq!(state.calls.load(Ordering::SeqCst), 2);
    assert_eq!(state.peak.load(Ordering::SeqCst), 1);
    assert_eq!(semaphore.available_permits(), 1);
}

struct StreamEngine {
    calls: AtomicUsize,
    first_entered: Notify,
    release_first: Notify,
}

async fn spawn_stream_engine() -> (String, Arc<StreamEngine>) {
    let state = Arc::new(StreamEngine {
        calls: AtomicUsize::new(0),
        first_entered: Notify::new(),
        release_first: Notify::new(),
    });
    let handler = |State(state): State<Arc<StreamEngine>>| async move {
        let ordinal = state.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if ordinal == 1 {
            state.first_entered.notify_one();
        }
        if ordinal > 1 {
            return Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from(complete_sse()))
                .expect("complete SSE");
        }
        let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(8);
        let task_state = Arc::clone(&state);
        tokio::spawn(async move {
            let first = events(&[usage(0), tokens(None)]);
            if sender.send(Ok(Bytes::from(first))).await.is_err() {
                return;
            }
            tokio::select! {
                _ = task_state.release_first.notified() => {
                    let finish = events(&[tokens(Some("stop")), usage(2)]);
                    let _ = sender.send(Ok(Bytes::from(finish))).await;
                }
                _ = sender.closed() => {}
            }
        });
        Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(ReceiverStream::new(receiver)))
            .expect("pending SSE")
    };
    let router = Router::new()
        .fallback(any(handler))
        .with_state(Arc::clone(&state));
    (serve(router).await, state)
}

fn spawn_stream_request(
    client: reqwest::Client,
    url: String,
) -> tokio::task::JoinHandle<StatusCode> {
    tokio::spawn(async move {
        let mut body = chat_body();
        body["stream"] = Value::Bool(true);
        let response = client.post(url).json(&body).send().await.expect("stream");
        let status = response.status();
        response.bytes().await.expect("stream body");
        status
    })
}

async fn check_sse_permit_release(cancel: bool) {
    let (engine, state) = spawn_stream_engine().await;
    let semaphore = Arc::new(Semaphore::new(1));
    let url = spawn_gateway(&engine, GatewayConfig::default(), Arc::clone(&semaphore)).await;
    let client = reqwest::Client::new();
    let mut body = chat_body();
    body["stream"] = Value::Bool(true);
    let mut first = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("first stream");
    assert_eq!(first.status(), StatusCode::OK);
    assert!(first.chunk().await.expect("first chunk").is_some());
    // The limiter admits the first request only once its handler is running; wait for that
    // signal instead of assuming it from the response headers.
    tokio::time::timeout(Duration::from_secs(5), state.first_entered.notified())
        .await
        .expect("the first request enters the engine");

    // Drain the first body in the background: a response nobody polls can be torn down by
    // the transport, which releases the permit early and hides what this test measures.
    let mut drain = if !cancel {
        None
    } else {
        Some(tokio::spawn(async move {
            while first.chunk().await.expect("remaining stream").is_some() {}
        }))
    };
    let second = spawn_stream_request(client, format!("{url}/v1/chat/completions"));
    // Whether the second request is already parked on the limiter at this instant is timing
    // dependent; what this test measures is that the slot is released when the first body
    // finishes or is dropped, which the completion below proves.

    if cancel {
        state.release_first.notify_one();
        if let Some(task) = drain.take() {
            task.abort();
        }
    } else {
        state.release_first.notify_one();
        if let Some(task) = drain.take() {
            task.await.expect("the drained body finishes");
        }
    }
    let second = tokio::time::timeout(Duration::from_secs(5), second)
        .await
        .expect("second stream starts after release")
        .expect("second task");
    assert_eq!(second, StatusCode::OK);
    assert_eq!(state.calls.load(Ordering::SeqCst), 2);
    assert_eq!(semaphore.available_permits(), 1);
}

#[tokio::test]
async fn finishing_an_sse_body_releases_its_slot() {
    check_sse_permit_release(false).await;
}

#[tokio::test]
async fn cancelling_an_sse_body_releases_its_slot() {
    check_sse_permit_release(true).await;
}

/// Send `head` and then `body` in pieces over raw TCP, and return the status line of
/// the reply, or `None` when no reply comes within `wait`.
async fn raw_request(
    url: &str,
    head: &str,
    body: &[u8],
    pause: Duration,
    wait: Duration,
) -> Option<String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let address = url.trim_start_matches("http://");
    let mut stream = tokio::net::TcpStream::connect(address)
        .await
        .expect("connect");
    stream.write_all(head.as_bytes()).await.expect("head");
    if !body.is_empty() {
        tokio::time::sleep(pause).await;
        let _ = stream.write_all(body).await;
    }
    let mut reply = vec![0; 512];
    let read = tokio::time::timeout(wait, stream.read(&mut reply))
        .await
        .ok()?
        .ok()?;
    let reply = String::from_utf8_lossy(&reply[..read]).to_string();
    reply.lines().next().map(str::to_owned)
}

fn timed_config(timeout: Duration) -> GatewayConfig {
    GatewayConfig {
        body_timeout: Some(timeout),
        ..GatewayConfig::default()
    }
}

#[tokio::test]
async fn a_slow_body_is_408_and_gives_its_slot_back() {
    let (engine, _) = spawn_basic_engine().await;
    let semaphore = Arc::new(Semaphore::new(1));
    let url = spawn_gateway(
        &engine,
        timed_config(Duration::from_millis(300)),
        Arc::clone(&semaphore),
    )
    .await;
    let head = "POST /v1/messages HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: 20\r\n\r\n";
    // Control: the same 20 bytes sent at once are read (and refused as JSON, 400).
    let status = raw_request(
        &url,
        head,
        b"{\"not\": \"a request\"}",
        Duration::ZERO,
        Duration::from_secs(5),
    )
    .await
    .expect("a reply");
    assert!(status.contains(" 400 "), "{status}");
    // The body never comes: 408 at the timeout, not a hang.
    let started = std::time::Instant::now();
    let status = raw_request(&url, head, b"", Duration::ZERO, Duration::from_secs(5))
        .await
        .expect("a reply");
    assert!(status.contains(" 408 "), "{status}");
    assert!(started.elapsed() < Duration::from_secs(3));
    // The slot came back: the one permit is free again.
    assert_eq!(semaphore.available_permits(), 1);
}

#[tokio::test]
async fn a_body_over_the_limit_is_413_before_it_is_read() {
    let (engine, _) = spawn_basic_engine().await;
    let url = spawn_gateway(
        &engine,
        timed_config(Duration::from_secs(5)),
        Arc::new(Semaphore::new(1)),
    )
    .await;
    let over = ds41_gateway::security::MAX_BODY_BYTES + 1;
    let head = format!(
        "POST /v1/messages HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {over}\r\n\r\n"
    );
    let status = raw_request(&url, &head, b"", Duration::ZERO, Duration::from_secs(2))
        .await
        .expect("a reply without waiting for the body");
    assert!(status.contains(" 413 "), "{status}");
}

#[tokio::test]
async fn the_listener_keeps_at_most_max_connections_open() {
    use ds41_gateway::listener::LimitedListener;
    let router = Router::new().route("/", any(|| async { "ok" }));
    let listener = LimitedListener::new(
        tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind"),
        1,
    );
    let address = axum::serve::Listener::local_addr(&listener).expect("address");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    let url = format!("http://{address}");
    let head = "GET / HTTP/1.1\r\nhost: x\r\n\r\n";
    // Control: with the slot free, a request is answered.
    let status = raw_request(&url, head, b"", Duration::ZERO, Duration::from_secs(2)).await;
    assert!(status.is_some_and(|line| line.contains(" 200 ")));
    tokio::time::sleep(Duration::from_millis(100)).await;
    // An idle connection takes the only slot; the next client waits to be accepted.
    let idle = tokio::net::TcpStream::connect(address).await.expect("idle");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let waiting = tokio::spawn({
        let url = url.clone();
        async move { raw_request(&url, head, b"", Duration::ZERO, Duration::from_secs(5)).await }
    });
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        !waiting.is_finished(),
        "a second connection was served while the first held the slot"
    );
    drop(idle);
    let status = waiting.await.expect("join");
    assert!(status.is_some_and(|line| line.contains(" 200 ")));
}

#[tokio::test]
async fn a_chunked_body_over_the_limit_is_413() {
    let (engine, _) = spawn_basic_engine().await;
    let url = spawn_gateway(
        &engine,
        timed_config(Duration::from_secs(20)),
        Arc::new(Semaphore::new(1)),
    )
    .await;
    // No content-length: the limit is found while reading, one MiB chunk at a time.
    let chunk = vec![0x20_u8; 1024 * 1024];
    let chunks = ds41_gateway::security::MAX_BODY_BYTES / chunk.len() + 1;
    let body = chunk_stream(chunk, chunks);
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/messages"))
        .header("content-type", "application/json")
        .body(reqwest::Body::wrap_stream(body))
        .send()
        .await;
    match response {
        Ok(response) => assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE),
        // The server may answer and close before the client finishes sending.
        Err(error) => panic!("no reply: {error}"),
    }
}

fn chunk_stream(
    chunk: Vec<u8>,
    count: usize,
) -> impl tokio_stream::Stream<Item = Result<Vec<u8>, std::io::Error>> {
    tokio_stream::iter((0..count).map(move |_| Ok(chunk.clone())))
}
