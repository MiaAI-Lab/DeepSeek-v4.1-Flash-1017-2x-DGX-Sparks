//! T020: the three API routes against a simulated engine and the real tokenizer.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::post;
use deepseek_recipe::request::ProtocolRequest;
use deepseek_recipe_encoding::{PromptTokenizer, TokenizerEncoder};
use ds41_gateway::api::{AppState, router};
use ds41_gateway::engine::EngineClient;
use ds41_gateway::parity::GatewayConfig;
use ds41_gateway::{Gateway, GatewaySettings};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

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

/// Encode a script made of plain text and explicit special-token spellings.
fn script(gateway: &Gateway, parts: &[(&str, bool)]) -> Vec<u32> {
    let prompt = PromptTokenizer::new(&gateway.tokenizer);
    let mut ids = Vec::new();
    for (text, special) in parts {
        if *special {
            ids.push(
                gateway
                    .tokenizer
                    .token_to_id(text)
                    .unwrap_or_else(|| panic!("missing special token {text:?}")),
            );
        } else {
            ids.extend(prompt.encode_ids(text, &[]).expect("encode plain"));
        }
    }
    ids
}

fn text_ids(gateway: &Gateway, text: &str) -> Vec<u32> {
    script(gateway, &[(text, false)])
}

struct MockState {
    body: Mutex<Option<Value>>,
    headers: Mutex<Option<HeaderMap>>,
    closed: AtomicBool,
    events: Vec<Value>,
    delay: Duration,
    done: bool,
}

async fn spawn_mock(events: Vec<Value>, delay: Duration, done: bool) -> (String, Arc<MockState>) {
    let state = Arc::new(MockState {
        body: Mutex::new(None),
        headers: Mutex::new(None),
        closed: AtomicBool::new(false),
        events,
        delay,
        done,
    });
    let handler = |State(state): State<Arc<MockState>>, headers: HeaderMap, body: Bytes| async move {
        let value: Value = serde_json::from_slice(&body).expect("json request");
        *state.body.lock().unwrap() = Some(value);
        *state.headers.lock().unwrap() = Some(headers);
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(16);
        let task_state = Arc::clone(&state);
        let watcher = tx.clone();
        let watcher_state = Arc::clone(&state);
        tokio::spawn(async move {
            watcher.closed().await;
            watcher_state.closed.store(true, Ordering::SeqCst);
        });
        tokio::spawn(async move {
            for event in &task_state.events {
                tokio::select! {
                    _ = tx.closed() => {
                        task_state.closed.store(true, Ordering::SeqCst);
                        return;
                    }
                    _ = tokio::time::sleep(task_state.delay) => {}
                }
                let line = format!("data: {event}\n\n");
                if tx.send(Ok(Bytes::from(line))).await.is_err() {
                    task_state.closed.store(true, Ordering::SeqCst);
                    return;
                }
            }
            if task_state.done {
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
            }
        });
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(ReceiverStream::new(rx)))
            .expect("response")
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let router = Router::new()
        .route("/inference/v1/generate", post(handler))
        .with_state(Arc::clone(&state));
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    (format!("http://{address}"), state)
}

async fn spawn_gateway(engine: &str, gateway: Gateway, max_model_len: u32) -> String {
    spawn_gateway_config(
        engine,
        gateway,
        GatewayConfig {
            max_model_len,
            api_sampling: true,
            default_effort: None,
            strict: true,
            ..GatewayConfig::default()
        },
    )
    .await
}

async fn spawn_gateway_config(engine: &str, gateway: Gateway, config: GatewayConfig) -> String {
    let state = AppState {
        gateway: Arc::new(gateway),
        engine: EngineClient::new(engine).expect("engine client"),
        config: Arc::new(config),
        semaphore: Arc::new(tokio::sync::Semaphore::new(16)),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        axum::serve(listener, router(state)).await.expect("serve");
    });
    format!("http://{address}")
}

async fn post_json(url: &str, body: Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(url)
        .json(&body)
        .send()
        .await
        .expect("request")
}

fn token_sha(ids: &[u32]) -> String {
    let joined = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let digest = Sha256::digest(joined.as_bytes());
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    hex[..12].to_string()
}

/// Parse the gateway's SSE body into (event name, data) pairs.
fn parse_sse(text: &str) -> Vec<(Option<String>, Value)> {
    let mut out = Vec::new();
    let mut name = None;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("event: ") {
            name = Some(value.to_string());
        } else if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                out.push((name.take(), Value::String("[DONE]".to_string())));
            } else {
                out.push((name.take(), serde_json::from_str(data).expect("json event")));
            }
        }
    }
    out
}

fn usage(prompt: u64, cached: u64) -> Value {
    json!({
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": 0,
            "total_tokens": prompt,
            "prompt_tokens_details": {"cached_tokens": cached}
        }
    })
}

fn final_usage(prompt: u64, cached: u64, completion: u64) -> Value {
    json!({
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached}
        }
    })
}

fn tokens(ids: &[u32], finish: Option<&str>) -> Value {
    json!({"choices": [{"index": 0, "finish_reason": finish, "token_ids": ids, "logprobs": null}]})
}

#[tokio::test]
async fn chat_non_stream_uses_the_api_shape_and_token_hashes() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hello world!");
    let (engine, mock) = spawn_mock(
        vec![
            usage(5, 3),
            tokens(&ids, Some("stop")),
            final_usage(5, 3, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"}
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-ds41-gateway"], "api");
    assert_eq!(response.headers()["x-ds41-token-sha"], token_sha(&ids));
    assert_eq!(response.headers()["x-ds41-token-sha-raw"], token_sha(&ids));
    assert_eq!(
        response.headers()["x-ds41-token-sha"]
            .to_str()
            .unwrap()
            .len(),
        12
    );
    assert_eq!(
        response.headers()["x-ds41-token-sha-raw"]
            .to_str()
            .unwrap()
            .len(),
        12
    );
    let body: Value = response.json().await.expect("body");
    let id = body["id"].as_str().expect("id");
    assert!(id.starts_with("chatcmpl-"));
    assert_eq!(id.len(), "chatcmpl-".len() + 32);
    assert_eq!(body["choices"][0]["message"]["content"], "Hello world!");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_cache_hit_tokens"], 3);
    assert_eq!(body["usage"]["prompt_cache_miss_tokens"], 2);
    assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 3);
    let sent = mock.body.lock().unwrap().clone().expect("engine body");
    assert_eq!(sent["stream"], true);
    assert_eq!(sent["stream_options"]["include_usage"], true);
    assert_eq!(sent["thinking"], false);
    assert!(sent["sampling_params"].get("stop").is_none());
}

#[tokio::test]
async fn token_hashes_exclude_the_terminal_eos_but_the_raw_hash_includes_it() {
    let gw = gateway();
    let eos = gw.eos;
    let mut ids = text_ids(&gw, "Hola");
    let delivered = ids.clone();
    ids.push(eos);
    let (engine, _) = spawn_mock(
        vec![
            usage(4, 0),
            tokens(&ids, Some("stop")),
            final_usage(4, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "disabled"}
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()["x-ds41-token-sha"],
        token_sha(&delivered)
    );
    assert_eq!(response.headers()["x-ds41-token-sha-raw"], token_sha(&ids));
}

#[tokio::test]
async fn an_engine_without_initial_usage_keeps_the_first_token_event() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, _) = spawn_mock(vec![tokens(&ids, Some("stop"))], Duration::ZERO, true).await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "disabled"}
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["choices"][0]["message"]["content"], "Hola");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn chat_stream_assembles_to_the_same_message() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hello world!");
    let (engine, _) = spawn_mock(
        vec![
            usage(5, 0),
            tokens(&ids, Some("stop")),
            final_usage(5, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"},
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let text = response.text().await.expect("sse");
    let events = parse_sse(&text);
    assert_eq!(events.last().unwrap().1, "[DONE]");
    let mut content = String::new();
    let mut finish = None;
    for (_, event) in &events {
        if let Some(choice) = event["choices"].get(0) {
            if let Some(delta) = choice["delta"]["content"].as_str() {
                content.push_str(delta);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish = Some(reason.to_string());
            }
        }
    }
    assert_eq!(content, "Hello world!");
    assert_eq!(finish.as_deref(), Some("stop"));
}

#[tokio::test]
async fn messages_non_stream_and_stream() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, _) = spawn_mock(
        vec![
            usage(8, 0),
            tokens(&ids, Some("stop")),
            final_usage(8, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let request_body = json!({
        "model": "DeepSeek-V4.1-Flash-1017",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hola"}],
        "thinking": {"type": "disabled"}
    });
    let response = post_json(&format!("{url}/v1/messages"), request_body.clone()).await;
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["content"][0]["text"], "Hola");
    assert_eq!(body["stop_reason"], "end_turn");

    let mut stream_body = request_body;
    stream_body["stream"] = json!(true);
    let response = post_json(&format!("{url}/v1/messages"), stream_body).await;
    let text = response.text().await.expect("sse");
    let events = parse_sse(&text);
    assert!(
        events
            .iter()
            .any(|(name, _)| name.as_deref() == Some("message_start"))
    );
    assert!(
        events
            .iter()
            .any(|(name, _)| name.as_deref() == Some("message_stop"))
    );
    assert_eq!(events.last().unwrap().1, "[DONE]");
}

#[tokio::test]
async fn responses_non_stream_and_stream() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, _) = spawn_mock(
        vec![
            usage(8, 0),
            tokens(&ids, Some("stop")),
            final_usage(8, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let request_body = json!({
        "model": "DeepSeek-V4.1-Flash-1017",
        "input": "Hola",
        "max_output_tokens": 64,
        "reasoning": {"effort": "none"}
    });
    let response = post_json(&format!("{url}/v1/responses"), request_body.clone()).await;
    assert_eq!(response.status(), 200);
    let response_body: Value = response.json().await.expect("body");
    assert_eq!(response_body["output"][0]["content"][0]["text"], "Hola");

    let mut stream_body = request_body;
    stream_body["stream"] = json!(true);
    let response = post_json(&format!("{url}/v1/responses"), stream_body).await;
    let text = response.text().await.expect("sse");
    let events = parse_sse(&text);
    assert!(
        events
            .iter()
            .any(|(name, _)| name.as_deref() == Some("response.created"))
    );
    assert!(
        events
            .iter()
            .any(|(name, _)| name.as_deref() == Some("response.completed"))
    );
    assert_eq!(events.last().unwrap().1, "[DONE]");
    assert!(events.iter().all(|(name, data)| {
        name.as_deref() != Some("error") && data.get("error").is_none_or(Value::is_null)
    }));
}

#[tokio::test]
async fn thinking_output_is_mapped_to_reasoning_fields() {
    let gw = gateway();
    let ids = script(
        &gw,
        &[
            ("Let me think.", false),
            ("</think>", true),
            ("Answer", false),
        ],
    );
    let (engine, _) = spawn_mock(
        vec![
            usage(5, 0),
            tokens(&ids, Some("stop")),
            final_usage(5, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "enabled"}
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("body");
    assert_eq!(
        body["choices"][0]["message"]["reasoning_content"],
        "Let me think."
    );
    assert_eq!(body["choices"][0]["message"]["content"], "Answer");

    let response = post_json(
        &format!("{url}/v1/messages"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "enabled"}
        }),
    )
    .await;
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["content"][0]["type"], "thinking");
    assert_eq!(body["content"][0]["thinking"], "Let me think.");

    let response = post_json(
        &format!("{url}/v1/responses"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "input": "Hola",
            "max_output_tokens": 64,
            "reasoning": {"effort": "medium"}
        }),
    )
    .await;
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["output"][0]["type"], "reasoning");
}

#[tokio::test]
async fn tool_calls_are_mapped_to_each_protocol() {
    let gw = gateway();
    let ids = script(
        &gw,
        &[
            ("<", false),
            ("｜DSML｜", true),
            (" calls>\n<", false),
            ("｜DSML｜", true),
            (" invoke name=\"get_weather\">\n<", false),
            ("｜DSML｜", true),
            (" parameter name=\"city\" string=\"true\">Paris</", false),
            ("｜DSML｜", true),
            (" parameter>\n</", false),
            ("｜DSML｜", true),
            (" invoke>\n</", false),
            ("｜DSML｜", true),
            (" calls>", false),
        ],
    );
    let (engine, _) = spawn_mock(
        vec![
            usage(20, 0),
            tokens(&ids, Some("stop")),
            final_usage(20, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }]);
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Weather in Paris"}],
            "thinking": {"type": "disabled"},
            "tools": tools,
            "tool_choice": "required"
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\": \"Paris\"}"
    );

    let response = post_json(
        &format!("{url}/v1/messages"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Weather in Paris"}],
            "thinking": {"type": "disabled"},
            "tools": [{
                "name": "get_weather",
                "description": "Weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "tool_choice": {"type": "auto"}
        }),
    )
    .await;
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["stop_reason"], "tool_use");
    assert_eq!(body["content"][0]["type"], "tool_use");
    assert_eq!(body["content"][0]["name"], "get_weather");

    let response = post_json(
        &format!("{url}/v1/responses"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "input": "Weather in Paris",
            "max_output_tokens": 64,
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "tool_choice": "auto"
        }),
    )
    .await;
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["output"][0]["type"], "function_call");
    assert_eq!(body["output"][0]["name"], "get_weather");
}

#[tokio::test]
async fn api_sampling_false_uses_the_client_temperature_and_top_p() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, mock) = spawn_mock(
        vec![
            usage(4, 0),
            tokens(&ids, Some("stop")),
            final_usage(4, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway_config(
        &engine,
        gateway(),
        GatewayConfig {
            api_sampling: false,
            ..GatewayConfig::default()
        },
    )
    .await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "enabled"},
            "temperature": 0.2,
            "top_p": 0.5
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let sent = mock.body.lock().unwrap().clone().expect("engine body");
    assert_eq!(sent["sampling_params"]["temperature"], 0.2);
    assert_eq!(sent["sampling_params"]["top_p"], 0.5);
}

#[tokio::test]
async fn context_limit_is_the_exact_api_400_before_the_engine() {
    let gw = gateway();
    let request = json!({
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": "Hola mundo"}]
    });
    let converted =
        serde_json::from_value::<deepseek_recipe::openai::ChatCompletionRequest>(request.clone())
            .expect("request")
            .convert(gw.options.clone())
            .expect("convert");
    let prompt_tokens = gw
        .prepare_conversation(&converted.conversation)
        .expect("encode")
        .ids
        .len() as u32;
    let limit = prompt_tokens - 1;
    let (engine, mock) = spawn_mock(vec![usage(1, 0)], Duration::ZERO, true).await;
    let url = spawn_gateway(&engine, gw, limit).await;
    let response = post_json(&format!("{url}/v1/chat/completions"), request).await;
    assert_eq!(response.status(), 400);
    assert_eq!(response.headers()["x-ds41-gateway"], "api");
    let body: Value = response.json().await.expect("body");
    let expected = format!(
        "This model's maximum context length is {limit} tokens. However, you requested \
         {prompt_tokens} tokens ({prompt_tokens} in the messages, 0 in the completion). \
         Please reduce the length of the messages or completion."
    );
    assert_eq!(body["error"]["message"], expected);
    assert!(mock.body.lock().unwrap().is_none(), "engine was touched");
}

#[tokio::test]
async fn a_prompt_far_over_the_context_is_refused_without_tokenizing_all_of_it() {
    let gw = gateway();
    // 24 MiB of words against a 1,000-token context: refused after the first piece.
    let text = "word ".repeat(24 * 1024 * 1024 / 5);
    let request = json!({
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": text}]
    });
    let (engine, mock) = spawn_mock(vec![usage(1, 0)], Duration::ZERO, true).await;
    let url = spawn_gateway(&engine, gw, 1_000).await;
    let response = post_json(&format!("{url}/v1/chat/completions"), request).await;
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.expect("body");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(
        message.starts_with(
            "This model's maximum context length is 1000 tokens. However, the messages have at least "
        ),
        "{message}"
    );
    assert!(mock.body.lock().unwrap().is_none(), "engine was touched");
    // The bound shows where the count stopped: within the first piece or two (about 52k
    // tokens each), not after the whole text (about five million tokens).
    let lower: usize = message
        .split("at least ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|count| count.parse().ok())
        .expect("a count in the message");
    assert!(
        (1_000..200_000).contains(&lower),
        "stopped at {lower} tokens"
    );
}

#[tokio::test]
async fn length_finish_reason_is_preserved() {
    let gw = gateway();
    let ids = text_ids(&gw, "partial");
    let (engine, _) = spawn_mock(
        vec![
            usage(4, 0),
            tokens(&ids, Some("length")),
            final_usage(4, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "max_tokens": 2
        }),
    )
    .await;
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn final_usage_patches_cache_hits_after_the_initial_event() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, _) = spawn_mock(
        vec![
            usage(10, 0),
            tokens(&ids, Some("stop")),
            final_usage(10, 7, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "disabled"}
        }),
    )
    .await;
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["usage"]["prompt_cache_hit_tokens"], 7);
    assert_eq!(body["usage"]["prompt_cache_miss_tokens"], 3);
}

#[tokio::test]
async fn a_stop_sequence_closes_the_engine_stream() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hello world!");
    let more = text_ids(&gw, " more");
    let mut all = ids.clone();
    all.extend(more);
    let (engine, mock) = spawn_mock(
        vec![
            usage(4, 0),
            tokens(&all, Some("stop")),
            final_usage(4, 0, all.len() as u64),
        ],
        Duration::from_millis(200),
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"},
            "stop": ["world"]
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    tokio::time::timeout(Duration::from_secs(3), async {
        while !mock.closed.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the gateway closed the engine stream");
}

#[tokio::test]
async fn chat_template_kwargs_thinking_is_mapped_before_conversion() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, mock) = spawn_mock(
        vec![
            usage(4, 0),
            tokens(&ids, Some("stop")),
            final_usage(4, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "max_completion_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "medium"}
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let sent = mock.body.lock().unwrap().clone().expect("engine body");
    assert_eq!(sent["sampling_params"]["max_tokens"], 32);
    assert_eq!(sent["thinking"], true);
    let body: Value = response.json().await.expect("body");
    assert!(body["choices"][0]["message"]["reasoning_content"].is_string());
}

#[tokio::test]
async fn api_requests_forward_both_auth_headers_to_the_engine() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, mock) = spawn_mock(
        vec![
            usage(4, 0),
            tokens(&ids, Some("stop")),
            final_usage(4, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .header("authorization", "Bearer sk-test")
        .header("x-api-key", "key-test")
        .json(&json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "disabled"}
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    let headers = mock
        .headers
        .lock()
        .unwrap()
        .clone()
        .expect("engine headers");
    assert_eq!(headers["authorization"], "Bearer sk-test");
    assert_eq!(headers["x-api-key"], "key-test");
}

#[tokio::test]
async fn an_engine_rejection_before_the_first_token_is_a_normal_http_error() {
    let (engine, _) = spawn_mock(
        vec![json!({"error": {"message": "bad request", "code": 400}}), json!({"choices": [], "usage": {"prompt_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}})],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "disabled"},
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), 400);
    assert_eq!(response.headers()["x-ds41-gateway"], "api");
    assert_eq!(response.headers()["content-type"], "application/json");
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["error"]["message"], "bad request");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn an_engine_error_after_finish_surfaces_instead_of_closing_cleanly() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, _) = spawn_mock(
        vec![
            usage(4, 0),
            tokens(&ids, Some("stop")),
            final_usage(4, 0, ids.len() as u64),
            json!({"error": {"message": "boom", "code": 500}}),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;

    // Non-streaming: the late engine error must become a 502, not a 200.
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "disabled"}
        }),
    )
    .await;
    assert_eq!(response.status(), 502);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["error"]["message"], "boom");

    // Streaming: the stream must end with the error, not with [DONE].
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "disabled"},
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let text = response.text().await.expect("sse");
    let events = parse_sse(&text);
    assert!(
        events
            .iter()
            .any(|(_, data)| data.get("error").is_some_and(|error| !error.is_null())),
        "no error event in {text}"
    );
    assert!(
        !events
            .iter()
            .any(|(_, data)| data.as_str() == Some("[DONE]")),
        "stream emitted [DONE] despite the engine error: {text}"
    );
}

#[tokio::test]
async fn a_missing_model_uses_the_configured_served_name() {
    let tokenizer = ds41_gateway::test_tokenizer();
    let gw = Gateway::new(
        Arc::new(tokenizer),
        GatewaySettings {
            served_name: "Custom-Model-42".to_string(),
            default_thinking: true,
            image_token: "<｜end_of_query｜>".to_string(),
            aliases: Vec::new(),
        },
    )
    .expect("build gateway");
    let ids = text_ids(&gw, "Hola");
    let (engine, mock) = spawn_mock(
        vec![
            usage(4, 0),
            tokens(&ids, Some("stop")),
            final_usage(4, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway_config(
        &engine,
        gw,
        GatewayConfig {
            served_name: "Custom-Model-42".to_string(),
            ..GatewayConfig::default()
        },
    )
    .await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "messages": [{"role": "user", "content": "Hola"}],
            "thinking": {"type": "disabled"}
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("body");
    assert_eq!(body["model"], "Custom-Model-42");
    let sent = mock.body.lock().unwrap().clone().expect("engine body");
    assert_eq!(sent["model"], "Custom-Model-42");
}

#[tokio::test]
async fn chat_template_kwargs_disable_thinking_even_with_a_top_level_effort() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, mock) = spawn_mock(
        vec![
            usage(4, 0),
            tokens(&ids, Some("stop")),
            final_usage(4, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let response = post_json(
        &format!("{url}/v1/chat/completions"),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hola"}],
            "reasoning_effort": "medium",
            "chat_template_kwargs": {"enable_thinking": false}
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let sent = mock.body.lock().unwrap().clone().expect("engine body");
    assert_eq!(sent["thinking"], false);
}

/// A plain Messages request with the given messages and extra top-level fields.
fn messages_body(messages: Value, extra: Value) -> Value {
    let mut body = json!({
        "model": "DeepSeek-V4.1-Flash-1017",
        "max_tokens": 64,
        "messages": messages,
        "thinking": {"type": "disabled"}
    });
    if let (Value::Object(body), Value::Object(extra)) = (&mut body, extra) {
        body.extend(extra);
    }
    body
}

async fn messages_engine() -> String {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, _) = spawn_mock(
        vec![
            usage(8, 0),
            tokens(&ids, Some("stop")),
            final_usage(8, 0, ids.len() as u64),
        ],
        Duration::ZERO,
        true,
    )
    .await;
    engine
}

/// PR #71's shapes: a clear_thinking edit of either date and any keep is accepted;
/// any other edit is refused with a 400.
#[tokio::test]
async fn context_management_accepts_clear_thinking_and_refuses_other_edits() {
    let engine = messages_engine().await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let user = json!([{"role": "user", "content": "Hola"}]);
    for (kind, keep) in [
        ("clear_thinking_20250929", json!("none")),
        ("clear_thinking_20251015", json!("all")),
        (
            "clear_thinking_20251015",
            json!({"type": "thinking_turns", "value": 1}),
        ),
    ] {
        let body = messages_body(
            user.clone(),
            json!({"context_management": {"edits": [{"type": kind, "keep": keep}]}}),
        );
        let response = post_json(&format!("{url}/v1/messages"), body).await;
        assert_eq!(response.status(), 200, "{kind} keep {keep}");
    }
    for extra in [
        json!({"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}}),
        json!({"context_management": {"edits": [{"type": "bogus_edit", "keep": "all"}]}}),
        json!({"context_management": {"edits": "clear"}}),
        json!({"context_management": "clear"}),
    ] {
        let response = post_json(
            &format!("{url}/v1/messages"),
            messages_body(user.clone(), extra.clone()),
        )
        .await;
        assert_eq!(response.status(), 400, "{extra}");
        let body: Value = response.json().await.expect("body");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("context_management")),
            "{extra}: {body}"
        );
    }
}

/// PR #72's shapes: a system turn anywhere in `messages` is accepted.
#[tokio::test]
async fn a_system_turn_is_accepted_anywhere_in_messages() {
    let engine = messages_engine().await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    for messages in [
        json!([
            {"role": "user", "content": "Hola"},
            {"role": "system", "content": "Responde breve."},
            {"role": "user", "content": "Hola otra vez"}
        ]),
        json!([
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Hola."},
            {"role": "system", "content": "Responde breve."},
            {"role": "user", "content": "Hola otra vez"}
        ]),
        json!([
            {"role": "user", "content": "Hola"},
            {"role": "system", "content": "Responde breve."}
        ]),
    ] {
        let response = post_json(
            &format!("{url}/v1/messages"),
            messages_body(messages.clone(), json!({})),
        )
        .await;
        let status = response.status();
        let body: Value = response.json().await.expect("body");
        assert_eq!(status, 200, "{messages}: {body}");
        assert_eq!(body["content"][0]["text"], "Hola", "{messages}: {body}");
    }
}

/// PR #76's case: while the engine is silent (a long cold prefill), the Messages
/// stream still sends bytes at least every 15 s, so an idle-cutting proxy keeps it.
#[tokio::test]
async fn a_silent_engine_does_not_leave_the_stream_idle_for_more_than_15_seconds() {
    let gw = gateway();
    let ids = text_ids(&gw, "Hola");
    let (engine, _) = spawn_mock(
        vec![
            usage(8, 0),
            tokens(&ids, Some("stop")),
            final_usage(8, 0, ids.len() as u64),
        ],
        Duration::from_secs(17),
        true,
    )
    .await;
    let url = spawn_gateway(&engine, gateway(), 65536).await;
    let mut body = messages_body(json!([{"role": "user", "content": "Hola"}]), json!({}));
    body["stream"] = json!(true);
    let mut response = post_json(&format!("{url}/v1/messages"), body).await;
    assert_eq!(response.status(), 200);
    let start = std::time::Instant::now();
    let mut last = start;
    let mut longest = Duration::ZERO;
    let mut text = String::new();
    while let Some(chunk) = response.chunk().await.expect("chunk") {
        let now = std::time::Instant::now();
        longest = longest.max(now - last);
        last = now;
        text.push_str(&String::from_utf8_lossy(&chunk));
    }
    assert!(
        start.elapsed() > Duration::from_secs(30),
        "the engine was silent for 2 x 17 s"
    );
    assert!(
        longest < Duration::from_secs(16),
        "longest silence {longest:?}"
    );
    assert!(
        text.lines().any(|line| line.starts_with(':')),
        "no keep-alive comment: {text}"
    );
    assert!(
        parse_sse(&text)
            .iter()
            .any(|(name, _)| name.as_deref() == Some("message_stop"))
    );
}
