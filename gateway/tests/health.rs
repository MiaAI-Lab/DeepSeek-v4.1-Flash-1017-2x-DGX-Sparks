//! T021: `/health`, `/ds41/render/*` and the gateway router.

use std::path::PathBuf;
use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::ProtocolRequest;
use ds41_gateway::api::AppState;
use ds41_gateway::engine::EngineClient;
use ds41_gateway::health;
use ds41_gateway::parity::GatewayConfig;
use ds41_gateway::{Gateway, GatewaySettings};
use serde_json::{Value, json};
use tokenizers::Tokenizer;

fn tokenizer_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tok/tokenizer.json")
}

fn gateway() -> Gateway {
    let tokenizer = Tokenizer::from_file(tokenizer_path()).expect("load tokenizer");
    Gateway::new(
        Arc::new(tokenizer),
        GatewaySettings {
            served_name: "DeepSeek-V4.1-Flash-1017".to_string(),
            default_thinking: true,
            image_token: "<｜end_of_query｜>".to_string(),
            aliases: Vec::new(),
        },
    )
    .expect("build gateway")
}

async fn spawn_gateway(engine: &str, gateway: Gateway, config: GatewayConfig) -> String {
    let state = AppState {
        gateway: Arc::new(gateway),
        engine: EngineClient::new(engine).expect("engine client"),
        config: Arc::new(config),
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

async fn spawn_health(status: StatusCode, body: Value) -> String {
    let handler = move || async move {
        let mut response: Response = (status, Json(body.clone())).into_response();
        response.headers_mut().insert(
            "content-type",
            axum::http::HeaderValue::from_static("application/json"),
        );
        response
    };
    let router = Router::new().route("/health", get(handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    format!("http://{address}")
}

#[tokio::test]
async fn health_is_200_when_the_engine_is_healthy() {
    let engine = spawn_health(
        StatusCode::OK,
        json!({"status": "ok", "live": {"connections": 1}}),
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), GatewayConfig::default()).await;
    let response = reqwest::get(format!("{url}/health")).await.expect("health");
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["gateway"], "ok");
    assert_eq!(body["upstream"], "ok");
    assert_eq!(body["engine"]["live"]["connections"], 1);
}

#[tokio::test]
async fn health_is_503_when_the_engine_is_unhealthy() {
    let engine = spawn_health(
        StatusCode::SERVICE_UNAVAILABLE,
        json!({"status": "loading"}),
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), GatewayConfig::default()).await;
    let response = reqwest::get(format!("{url}/health")).await.expect("health");
    assert_eq!(response.status(), 503);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["gateway"], "ok");
    assert_eq!(body["upstream"], "status 503");
    assert_eq!(body["engine"]["status"], "loading");
}

#[tokio::test]
async fn health_is_503_when_the_engine_is_unreachable() {
    let url = spawn_gateway("http://127.0.0.1:1", gateway(), GatewayConfig::default()).await;
    let response = reqwest::get(format!("{url}/health")).await.expect("health");
    assert_eq!(response.status(), 503);
    let body: Value = response.json().await.expect("body");
    assert!(
        body["upstream"]
            .as_str()
            .unwrap()
            .starts_with("unreachable:")
    );
    assert!(body["engine"].is_null());
}

#[tokio::test]
async fn render_chat_returns_the_same_ids_and_marks_as_the_api_path() {
    let gateway = gateway();
    let body = json!({
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": "Hola"}],
        "thinking": {"type": "disabled"}
    });
    let converted = serde_json::from_value::<ChatCompletionRequest>(body.clone())
        .expect("request")
        .convert(gateway.options.clone())
        .expect("convert");
    let expected = gateway
        .prepare_conversation(&converted.conversation)
        .expect("encode");
    let url = spawn_gateway(
        "http://127.0.0.1:1",
        gateway,
        GatewayConfig {
            max_model_len: 65536,
            ..GatewayConfig::default()
        },
    )
    .await;
    let response = reqwest::Client::new()
        .post(format!("{url}/ds41/render/chat"))
        .json(&body)
        .send()
        .await
        .expect("render");
    assert_eq!(response.status(), 200);
    let value: Value = response.json().await.expect("body");
    assert_eq!(value["prompt_token_ids"], json!(expected.ids));
    assert_eq!(value["history_len"], expected.history_len);
    assert_eq!(value["shared_prefixes"], json!(expected.shared_prefixes));
    assert_eq!(value["max_tokens"], 8192);
    assert!(value["fallback_reason"].is_null());
    assert!(value["prompt"].as_str().unwrap().contains("Hola"));
}

#[tokio::test]
async fn render_reports_a_fallback_reason_without_touching_the_engine() {
    let url = spawn_gateway("http://127.0.0.1:1", gateway(), GatewayConfig::default()).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/ds41/render/chat"))
        .json(&json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "draft": true
        }))
        .send()
        .await
        .expect("render");
    assert_eq!(response.status(), 200);
    let value: Value = response.json().await.expect("body");
    assert_eq!(value["fallback_reason"], "vLLM field draft");
    assert!(value["prompt"].is_null());
    assert!(value["prompt_token_ids"].is_null());
    assert!(value["max_tokens"].is_null());
}

#[tokio::test]
async fn render_injects_the_operator_default_effort() {
    let url = spawn_gateway(
        "http://127.0.0.1:1",
        gateway(),
        GatewayConfig {
            default_effort: Some("low".to_string()),
            ..GatewayConfig::default()
        },
    )
    .await;
    let response = reqwest::Client::new()
        .post(format!("{url}/ds41/render/chat"))
        .json(&json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}]
        }))
        .send()
        .await
        .expect("render");
    assert_eq!(response.status(), 200);
    let value: Value = response.json().await.expect("body");
    assert_eq!(value["effort"], "low");
    assert_eq!(value["thinking"], true);
}

#[tokio::test]
async fn render_messages_and_responses_work_without_an_engine() {
    let url = spawn_gateway("http://127.0.0.1:1", gateway(), GatewayConfig::default()).await;
    for (protocol, body) in [
        (
            "messages",
            json!({
                "model": "DeepSeek-V4.1-Flash-1017",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Hola"}]
            }),
        ),
        (
            "responses",
            json!({
                "model": "DeepSeek-V4.1-Flash-1017",
                "input": "Hola",
                "max_output_tokens": 64
            }),
        ),
    ] {
        let response = reqwest::Client::new()
            .post(format!("{url}/ds41/render/{protocol}"))
            .json(&body)
            .send()
            .await
            .expect("render");
        assert_eq!(response.status(), 200, "{protocol}");
        let value: Value = response.json().await.expect("body");
        assert!(!value["prompt_token_ids"].as_array().unwrap().is_empty());
        assert!(value["prompt"].as_str().unwrap().contains("Hola"));
        assert!(value["max_tokens"].as_u64().unwrap() > 0);
    }
}

#[tokio::test]
async fn inference_routes_are_hidden_behind_a_404() {
    let url = spawn_gateway("http://127.0.0.1:1", gateway(), GatewayConfig::default()).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/inference/v1/generate"))
        .json(&json!({}))
        .send()
        .await
        .expect("404");
    assert_eq!(response.status(), 404);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["detail"], "Not Found");
}

#[tokio::test]
async fn render_rejects_a_non_object_body_with_the_api_error() {
    let url = spawn_gateway("http://127.0.0.1:1", gateway(), GatewayConfig::default()).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/ds41/render/chat"))
        .header("content-type", "application/json")
        .body("[]")
        .send()
        .await
        .expect("render");
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}
