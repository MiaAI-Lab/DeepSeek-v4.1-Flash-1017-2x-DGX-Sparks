//! T022: transparent proxy, API fallback, hop-by-hop headers, and `--require-key`.

use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use ds41_gateway::api::AppState;
use ds41_gateway::engine::EngineClient;
use ds41_gateway::health;
use ds41_gateway::parity::GatewayConfig;
use ds41_gateway::security::ApiKeys;
use ds41_gateway::{Gateway, GatewaySettings};
use serde_json::{Value, json};

fn gateway() -> Gateway {
    let tokenizer = ds41_gateway::test_tokenizer();
    Gateway::new(
        Arc::new(tokenizer),
        GatewaySettings {
            served_name: "DeepSeek-V4.1-Flash-1017".to_string(),
            default_thinking: true,
            image_token: "<｜deepseek_image｜>".to_string(),
            aliases: Vec::new(),
        },
    )
    .expect("build gateway")
}

#[derive(Clone, Debug)]
struct Record {
    method: String,
    path: String,
    headers: HeaderMap,
    body: Bytes,
}

struct Mock {
    records: Mutex<Vec<Record>>,
    require_key: bool,
}

async fn spawn_mock(require_key: bool) -> (String, Arc<Mock>) {
    let mock = Arc::new(Mock {
        records: Mutex::new(Vec::new()),
        require_key,
    });
    let handler = |State(mock): State<Arc<Mock>>, request: Request| async move {
        let method = request.method().to_string();
        let path = request
            .uri()
            .path_and_query()
            .map(|value| value.as_str().to_string())
            .unwrap_or_default();
        let headers = request.headers().clone();
        let body = axum::body::to_bytes(request.into_body(), 64 * 1024 * 1024)
            .await
            .expect("body");
        mock.records.lock().unwrap().push(Record {
            method,
            path: path.clone(),
            headers: headers.clone(),
            body: body.clone(),
        });
        let has_key = headers.contains_key("authorization") || headers.contains_key("x-api-key");
        if mock.require_key && path.starts_with("/v1/models") && !has_key {
            return (
                StatusCode::UNAUTHORIZED,
                axum::Json(
                    json!({"error": {"message": "missing key", "type": "invalid_request_error"}}),
                ),
            )
                .into_response();
        }
        let mut response = if path.starts_with("/v1/echo") {
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/octet-stream")
                .header("x-engine", "yes")
                .body(axum::body::Body::from(body))
                .expect("response")
        } else {
            (
                StatusCode::OK,
                axum::Json(json!({"ok": true, "path": path})),
            )
                .into_response()
        };
        response.headers_mut().insert(
            "connection",
            axum::http::HeaderValue::from_static("keep-alive"),
        );
        response
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let router = Router::new()
        .fallback(any(handler))
        .with_state(mock.clone());
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    (format!("http://{address}"), mock)
}

async fn spawn_gateway(engine: &str, config: GatewayConfig) -> String {
    let state = AppState {
        gateway: Arc::new(gateway()),
        engine: EngineClient::new(engine).expect("engine client"),
        config: Arc::new(config),
        semaphore: Arc::new(tokio::sync::Semaphore::new(16)),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        axum::serve(listener, health::router(state))
            .await
            .expect("serve");
    });
    format!("http://{address}")
}

fn tool_result_without_id() -> Value {
    let case: Value = serde_json::from_str(include_str!("../casos/chat-tool-result-sin-id.json"))
        .expect("valid parity case");
    let mut body = case["body"].clone();
    body["model"] = json!("DeepSeek-V4.1-Flash-1017");
    body
}

#[tokio::test]
async fn a_five_megabyte_body_is_proxied_byte_for_byte() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(&engine, GatewayConfig::default()).await;
    let body = vec![b'x'; 5 * 1024 * 1024];
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/echo"))
        .body(body.clone())
        .send()
        .await
        .expect("proxy");
    assert_eq!(response.status(), 200);
    let received = response.bytes().await.expect("body");
    assert_eq!(received.as_ref(), body.as_slice());
    let record = mock
        .records
        .lock()
        .unwrap()
        .last()
        .cloned()
        .expect("record");
    assert_eq!(record.method, "POST");
    assert_eq!(record.path, "/v1/echo");
    assert_eq!(record.body.as_ref(), body.as_slice());
}

#[tokio::test]
async fn proxy_keeps_client_headers_and_drops_hop_by_hop_ones() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(&engine, GatewayConfig::default()).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/echo?q=1"))
        .header("authorization", "Bearer sk-test")
        .header("x-api-key", "key-test")
        .header("host", "client.example")
        .header("connection", "keep-alive")
        .header("proxy-authorization", "Basic x")
        .header("te", "trailers")
        .header("x-custom", "kept")
        .body("hello")
        .send()
        .await
        .expect("proxy");
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-engine"], "yes");
    assert!(!response.headers().contains_key("connection"));
    let record = mock
        .records
        .lock()
        .unwrap()
        .last()
        .cloned()
        .expect("record");
    assert_eq!(record.path, "/v1/echo?q=1");
    assert_eq!(record.headers["authorization"], "Bearer sk-test");
    assert_eq!(record.headers["x-api-key"], "key-test");
    assert_eq!(record.headers["host"], "client.example");
    assert_eq!(record.headers["x-custom"], "kept");
    assert!(!record.headers.contains_key("connection"));
    assert!(!record.headers.contains_key("proxy-authorization"));
    assert!(!record.headers.contains_key("te"));
    assert_eq!(
        record.headers["content-length"],
        record.body.len().to_string().as_str()
    );
}

#[tokio::test]
async fn internal_inference_routes_are_not_exposed() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(&engine, GatewayConfig::default()).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/inference/v1/generate"))
        .json(&json!({"token_ids": [1]}))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 404);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["detail"], "Not Found");
    assert!(mock.records.lock().unwrap().is_empty());
}

#[tokio::test]
async fn other_engine_routes_are_proxied() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(&engine, GatewayConfig::default()).await;
    let response = reqwest::get(format!("{url}/v1/models"))
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["path"], "/v1/models");
    let record = mock
        .records
        .lock()
        .unwrap()
        .last()
        .cloned()
        .expect("record");
    assert_eq!(record.method, "GET");
}

#[tokio::test]
async fn require_key_rejects_locally_without_touching_the_engine() {
    let (engine, mock) = spawn_mock(true).await;
    let url = spawn_gateway(
        &engine,
        GatewayConfig {
            require_key: true,
            api_keys: Some(ApiKeys::parse("sk-test\n").expect("key")),
            ..GatewayConfig::default()
        },
    )
    .await;
    let response = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}]
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 401);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["error"]["message"], "missing or invalid API key");
    assert!(mock.records.lock().unwrap().is_empty());

    let response = reqwest::Client::new()
        .get(format!("{url}/v1/models"))
        .header("authorization", "Bearer sk-test")
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn api_fallback_fields_go_to_the_engine_with_the_fallback_header() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(&engine, GatewayConfig::default()).await;
    let cases = [
        ("/v1/chat/completions", json!({"draft": {"tokens": [1]}})),
        ("/v1/chat/completions", json!({"prompt": "raw"})),
        ("/v1/chat/completions", json!({"thinking_budget": 32})),
        ("/v1/chat/completions", json!({"priority": 1})),
        (
            "/v1/chat/completions",
            json!({"guided_json": {"type": "object"}}),
        ),
        (
            "/v1/chat/completions",
            json!({"structured_outputs": {"json": {}}}),
        ),
        ("/v1/chat/completions", json!({"return_token_ids": true})),
        (
            "/v1/chat/completions",
            json!({"chat_template_kwargs": {"other": 1}}),
        ),
        (
            "/v1/chat/completions",
            json!({"response_format": {"type": "xml"}}),
        ),
        ("/v1/responses", json!({"previous_response_id": "resp_1"})),
        ("/v1/responses", json!({"store": true})),
        ("/v1/responses", json!({"conversation": "conv_1"})),
        ("/v1/responses", json!({"modalities": ["text", "audio"]})),
        ("/v1/responses", json!({"audio": {"voice": "alloy"}})),
        (
            "/v1/responses",
            json!({"video_url": "https://example.com/a.mp4"}),
        ),
    ];
    for (path, extra) in cases {
        let mut body = if path.ends_with("/responses") {
            json!({"model": "DeepSeek-V4.1-Flash-1017", "input": "Hola"})
        } else {
            json!({
                "model": "DeepSeek-V4.1-Flash-1017",
                "messages": [{"role": "user", "content": "Hola"}]
            })
        };
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let response = reqwest::Client::new()
            .post(format!("{url}{path}"))
            .json(&body)
            .send()
            .await
            .expect("request");
        assert_eq!(response.status(), 200, "{path} {body}");
        assert_eq!(response.headers()["x-ds41-gateway"], "fallback");
        let record = mock
            .records
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("record");
        assert_eq!(record.path, path);
        assert_eq!(
            serde_json::from_slice::<Value>(&record.body).expect("json"),
            body
        );
    }
}

#[tokio::test]
async fn strict_returns_the_api_400_instead_of_falling_back() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(
        &engine,
        GatewayConfig {
            strict: true,
            ..GatewayConfig::default()
        },
    )
    .await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "draft": {"tokens": [1]}
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 400);
    assert_eq!(response.headers()["x-ds41-gateway"], "api");
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"].is_string());
    assert!(mock.records.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_conversion_error_falls_back_to_the_engine_unless_strict() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(&engine, GatewayConfig::default()).await;
    let body = json!({
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "Hola"}]
    });
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-ds41-gateway"], "fallback");
    let record = mock
        .records
        .lock()
        .unwrap()
        .last()
        .cloned()
        .expect("record");
    assert_eq!(
        serde_json::from_slice::<Value>(&record.body).expect("json"),
        body
    );
}

#[tokio::test]
async fn tool_result_without_id_falls_back_with_the_body_unchanged() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(&engine, GatewayConfig::default()).await;
    let body = tool_result_without_id();
    let client = reqwest::Client::new();

    let response = client
        .post(format!("{url}/ds41/render/chat"))
        .json(&body)
        .send()
        .await
        .expect("render");
    assert_eq!(response.status(), 200);
    let render: Value = response.json().await.expect("render body");
    assert!(
        render["fallback_reason"]
            .as_str()
            .is_some_and(|reason| reason.contains("missing field `tool_call_id`"))
    );
    assert!(mock.records.lock().unwrap().is_empty());

    let response = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-ds41-gateway"], "fallback");
    let records = mock.records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].path, "/v1/chat/completions");
    assert_eq!(
        serde_json::from_slice::<Value>(&records[0].body).expect("json"),
        body
    );
}

#[tokio::test]
async fn tool_result_without_id_is_an_api_400_in_strict_mode() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(
        &engine,
        GatewayConfig {
            strict: true,
            ..GatewayConfig::default()
        },
    )
    .await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&tool_result_without_id())
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 400);
    assert_eq!(response.headers()["x-ds41-gateway"], "api");
    let body: Value = response.json().await.expect("body");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("missing field `tool_call_id`"))
    );
    assert!(mock.records.lock().unwrap().is_empty());
}

#[tokio::test]
async fn an_unrelated_error_mentioning_tool_call_id_is_not_a_fallback() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(&engine, GatewayConfig::default()).await;
    // `stream` must be a boolean; a string that happens to spell the missing-field
    // message is a plain 400, not the tool_call_id fallback.
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "stream": "missing field `tool_call_id`"
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 400);
    assert_eq!(response.headers()["x-ds41-gateway"], "api");
    assert!(mock.records.lock().unwrap().is_empty());
}

#[tokio::test]
async fn logprobs_fall_back_only_when_health_says_they_are_unsupported() {
    let (engine, mock) = spawn_mock(false).await;
    let url = spawn_gateway(
        &engine,
        GatewayConfig {
            engine_logprobs: Some(false),
            ..GatewayConfig::default()
        },
    )
    .await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "logprobs": true,
            "top_logprobs": 3
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-ds41-gateway"], "fallback");
    assert_eq!(mock.records.lock().unwrap().len(), 1);
}
