//! T023: `/v1/messages/count_tokens`.

use std::sync::Arc;

use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::request::ProtocolRequest;
use ds41_gateway::api::AppState;
use ds41_gateway::engine::EngineClient;
use ds41_gateway::health;
use ds41_gateway::parity::GatewayConfig;
use ds41_gateway::{Gateway, GatewaySettings};
use serde_json::{Value, json};

fn gateway() -> Gateway {
    let tokenizer = ds41_gateway::test_tokenizer();
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

async fn spawn_gateway(gateway: Gateway) -> String {
    let state = AppState {
        gateway: Arc::new(gateway),
        engine: EngineClient::new("http://127.0.0.1:1").expect("engine client"),
        config: Arc::new(GatewayConfig::default()),
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

#[tokio::test]
async fn count_tokens_is_the_encoded_prompt_length() {
    let body = json!({
        "model": "DeepSeek-V4.1-Flash-1017",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let gateway = gateway();
    let converted = serde_json::from_value::<MessagesRequest>(body.clone())
        .expect("request")
        .convert(gateway.options.clone())
        .expect("convert");
    let expected = gateway
        .prepare_conversation(&converted.conversation)
        .expect("encode")
        .ids
        .len();
    let url = spawn_gateway(gateway).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/messages/count_tokens"))
        .json(&body)
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-ds41-gateway"], "api");
    let value: Value = response.json().await.expect("body");
    assert_eq!(value["input_tokens"], expected);
}

#[tokio::test]
async fn an_invalid_role_is_the_anthropic_400() {
    let url = spawn_gateway(gateway()).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/messages/count_tokens"))
        .json(&json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "max_tokens": 64,
            "messages": [{"role": "robot", "content": "Hello"}]
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 400);
    assert_eq!(response.headers()["x-ds41-gateway"], "api");
    let value: Value = response.json().await.expect("body");
    assert_eq!(value["type"], "error");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert!(value["error"]["message"].is_string());
}
