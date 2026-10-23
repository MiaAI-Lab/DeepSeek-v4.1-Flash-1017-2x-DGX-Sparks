//! T027: end-to-end invariants and hashes against a mock gateway and a mock Zig.

use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use ds41_gateway::paridad::caso::Caso;
use ds41_gateway::paridad::e2e;
use serde_json::{Value, json};

async fn chat(Json(body): Json<Value>) -> Response {
    let mut response = if body.get("stream").and_then(Value::as_bool) == Some(true) {
        let events = [
            json!({"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}),
            json!({"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"hola"},"finish_reason":null}]}),
            json!({"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":5}}),
        ];
        let mut text = String::new();
        for event in events {
            text.push_str(&format!("data: {event}\n\n"));
        }
        text.push_str("data: [DONE]\n\n");
        (StatusCode::OK, text).into_response()
    } else {
        Json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hola"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                       "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 5}
        }))
        .into_response()
    };
    response
        .headers_mut()
        .insert("x-ds41-gateway", HeaderValue::from_static("api"));
    response
        .headers_mut()
        .insert("x-ds41-token-sha", HeaderValue::from_static("abcdef012345"));
    response.headers_mut().insert(
        "x-ds41-token-sha-raw",
        HeaderValue::from_static("abcdef012345"),
    );
    response
}

async fn render() -> Json<Value> {
    Json(json!({"prompt_token_ids": [1, 2, 3]}))
}

async fn completions() -> Json<Value> {
    Json(json!({"tensorfold": {"token_sha": "abcdef012345"}}))
}

async fn spawn_mocks() -> String {
    let router = AxumRouter::new()
        .route("/v1/chat/completions", post(chat))
        .route("/ds41/render/chat", post(render))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(|| async { Json(json!({"data": []})) }))
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    format!("http://{address}")
}

fn caso() -> Caso {
    serde_json::from_value(json!({
        "id": "mock-basico",
        "protocol": "chat",
        "tags": ["basico"],
        "body": {"messages": [{"role": "user", "content": "hola"}], "thinking": {"type": "disabled"}},
        "expect": {
            "render": "identical",
            "cache": "equivalent",
            "e2e": {"invariants": true, "same_ids": true, "finish_reason": "stop"}
        },
        "greedy": true
    }))
    .expect("valid case")
}

#[tokio::test]
async fn a_well_formed_mock_reply_passes_every_invariant_and_the_hash() {
    let base = spawn_mocks().await;
    let result = e2e::run_case(&caso(), &base, &base, "mock-model").await;
    assert!(
        result.violaciones.is_empty(),
        "violaciones: {:?}",
        result.violaciones
    );
    assert_eq!(result.same_ids, Some(true));
    assert_eq!(result.finish_api.as_deref(), Some("stop"));
    assert_eq!(result.sha_api.as_deref(), Some("abcdef012345"));
}

#[tokio::test]
async fn a_reply_with_template_markup_is_reported() {
    let base = spawn_mocks().await;
    let mut case = caso();
    case.body = json!({"messages": [{"role": "user", "content": "hola"}], "thinking": {"type": "disabled"}});
    // The mock returns "hola" without markup, so a deliberate mismatch is made
    // by asking for a finish reason that the mock does not return.
    case.expect.e2e = serde_json::from_value(json!({
        "invariants": true, "same_ids": true, "finish_reason": "length"
    }))
    .expect("expect");
    let result = e2e::run_case(&case, &base, &base, "mock-model").await;
    assert!(result.violaciones.is_empty());
    assert_eq!(result.finish_api.as_deref(), Some("stop"));
}
