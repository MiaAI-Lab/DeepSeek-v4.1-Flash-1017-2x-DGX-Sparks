//! T024: image destination filter, fetching, limits, token adjustment and
//! `content_parts` order.

use std::io::Cursor;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use base64::Engine as _;
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::request::ProtocolRequest;
use deepseek_recipe_core::multimodal::{ImageDetail, ImageSource, ImageTokenSpec};
use ds41_gateway::api::AppState;
use ds41_gateway::engine::EngineClient;
use ds41_gateway::health;
use ds41_gateway::images::{ImageSettings, destination_allowed, resolve};
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
            image_token: "<｜deepseek_image｜>".to_string(),
            aliases: Vec::new(),
        },
    )
    .expect("build gateway")
}

fn png(width: u32, height: u32, color: [u8; 4]) -> Vec<u8> {
    let image = image::RgbaImage::from_pixel(width, height, image::Rgba(color));
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("encode PNG");
    bytes
}

fn data_url(bytes: &[u8]) -> String {
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn source_data(bytes: &[u8], path: &str) -> ImageSource {
    ImageSource::DataUrl {
        data_url: data_url(bytes),
        detail: ImageDetail::High,
        path: path.to_string(),
    }
}

fn source_url(url: &str, path: &str) -> ImageSource {
    ImageSource::Url {
        url: url.to_string(),
        detail: ImageDetail::High,
        path: path.to_string(),
    }
}

#[test]
fn destination_filter_rejects_the_api_table() {
    let rejected = [
        "0.1.2.3",
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.1.1",
        "100.64.0.1",
        "100.127.255.254",
        "192.0.0.1",
        "192.0.2.1",
        "192.31.196.1",
        "192.52.193.1",
        "192.88.99.1",
        "192.175.48.1",
        "198.18.0.1",
        "198.19.255.254",
        "198.51.100.1",
        "203.0.113.1",
        "224.0.0.1",
        "239.255.255.255",
        "240.0.0.1",
        "255.255.255.254",
        "255.255.255.255",
        "::",
        "::1",
        "fc00::1",
        "fdff::1",
        "fe80::1",
        "febf::1",
        "ff02::1",
        "64:ff9b::7f00:1",
        "64:ff9b::a00:1",
        "64:ff9b:1::1",
        "2001::1",
        "2001:db8::1",
        "2002:7f00:1::",
        "100::1",
        "fec0::1",
        "3fff::1",
        "::ffff:127.0.0.1",
    ];
    for address in rejected {
        assert!(
            !destination_allowed("example.test", address.parse().unwrap()),
            "{address} should be rejected"
        );
    }
    let accepted = [
        "8.8.8.8",
        "1.1.1.1",
        "2001:4860:4860::8888",
        "2606:4700::1111",
        "64:ff9b::1.1.1.1",
        "2002:0101:0101::",
        "::ffff:8.8.8.8",
    ];
    for address in accepted {
        assert!(
            destination_allowed("example.test", address.parse().unwrap()),
            "{address} should be accepted"
        );
    }
}

#[tokio::test]
async fn dns_override_does_not_bypass_the_destination_filter() {
    for address in ["10.0.0.1", "127.0.0.1", "100.100.100.100"] {
        let settings = ImageSettings {
            max_images: 600,
            max_total_bytes: 64 * 1024 * 1024,
            block_private_urls: true,
            dns_overrides: [("fake.test".to_string(), address.parse::<IpAddr>().unwrap())]
                .into_iter()
                .collect(),
        };
        let error = resolve(
            &[source_url(
                "http://fake.test:1/image.png",
                "messages.0.content[0]",
            )],
            &settings,
        )
        .await
        .expect_err("private destination");
        assert_eq!(error.status, 400, "{address}: {}", error.body);
        let message = error.body["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("the destination filter rejects fake.test"),
            "{address}: {message}"
        );
    }
}

#[tokio::test]
async fn a_non_overridden_name_is_still_filtered_when_overrides_are_present() {
    let settings = ImageSettings {
        max_images: 600,
        max_total_bytes: 64 * 1024 * 1024,
        block_private_urls: true,
        dns_overrides: [(
            "other.test".to_string(),
            "1.1.1.1".parse::<IpAddr>().unwrap(),
        )]
        .into_iter()
        .collect(),
    };
    let error = resolve(
        &[source_url(
            "http://localhost:1/image.png",
            "messages.0.content[0]",
        )],
        &settings,
    )
    .await
    .expect_err("loopback destination");
    assert_eq!(error.status, 400, "{}", error.body);
    let message = error.body["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("the destination filter rejects localhost"),
        "{message}"
    );
}

async fn spawn_image_server(bytes: Vec<u8>) -> String {
    let bytes = Arc::new(bytes);
    let image = {
        let bytes = Arc::clone(&bytes);
        move || async move {
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "image/png")],
                bytes.as_ref().clone(),
            )
                .into_response()
        }
    };
    let redirect = || async {
        (
            StatusCode::FOUND,
            [(header::LOCATION, "/image.png")],
            axum::body::Body::empty(),
        )
            .into_response()
    };
    let router = Router::new()
        .route("/image.png", get(image))
        .route("/redirect", get(redirect));
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
async fn a_local_png_is_fetched_counted_and_returned_in_content_parts() {
    let bytes = png(16, 16, [255, 0, 0, 255]);
    let server = spawn_image_server(bytes.clone()).await;
    let port = server.rsplit(':').next().unwrap();
    let url = format!("http://fake.test:{port}/image.png");
    let settings = ImageSettings {
        max_images: 600,
        max_total_bytes: 64 * 1024 * 1024,
        block_private_urls: false,
        dns_overrides: [(
            "fake.test".to_string(),
            "127.0.0.1".parse::<IpAddr>().unwrap(),
        )]
        .into_iter()
        .collect(),
    };
    let resolved = resolve(&[source_url(&url, "messages.0.content[0]")], &settings)
        .await
        .expect("resolve");
    assert_eq!(resolved.count, 1);
    let expected = ImageTokenSpec::v41()
        .calc_token_count(16, 16)
        .expect("tokens")
        - 1;
    assert_eq!(resolved.adjustment, expected);
    let parts = resolved.parts.expect("parts");
    let encoded = parts[0]["url"].as_str().expect("url");
    let body = encoded
        .strip_prefix("data:application/octet-stream;base64,")
        .expect("data URL");
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(body)
            .expect("base64"),
        bytes
    );
}

#[tokio::test]
async fn a_redirect_is_followed_with_the_filter_off() {
    let bytes = png(16, 16, [0, 255, 0, 255]);
    let server = spawn_image_server(bytes.clone()).await;
    let port = server.rsplit(':').next().unwrap();
    let url = format!("http://fake.test:{port}/redirect");
    let settings = ImageSettings {
        max_images: 600,
        max_total_bytes: 64 * 1024 * 1024,
        block_private_urls: false,
        dns_overrides: [(
            "fake.test".to_string(),
            "127.0.0.1".parse::<IpAddr>().unwrap(),
        )]
        .into_iter()
        .collect(),
    };
    let resolved = resolve(&[source_url(&url, "messages.0.content[0]")], &settings)
        .await
        .expect("resolve");
    let parts = resolved.parts.expect("parts");
    let body = parts[0]["url"]
        .as_str()
        .unwrap()
        .strip_prefix("data:application/octet-stream;base64,")
        .unwrap();
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(body)
            .expect("base64"),
        bytes
    );
}

#[tokio::test]
async fn two_images_keep_the_marker_order() {
    let first = png(16, 16, [255, 0, 0, 255]);
    let second = png(16, 16, [0, 0, 255, 255]);
    let settings = ImageSettings {
        max_images: 600,
        max_total_bytes: 64 * 1024 * 1024,
        block_private_urls: true,
        dns_overrides: Default::default(),
    };
    let resolved = resolve(
        &[
            source_data(&first, "messages.0.content[0]"),
            source_data(&second, "messages.0.content[1]"),
        ],
        &settings,
    )
    .await
    .expect("resolve");
    let parts = resolved.parts.expect("parts");
    for (part, expected) in parts.iter().zip([first, second]) {
        let body = part["url"]
            .as_str()
            .unwrap()
            .strip_prefix("data:application/octet-stream;base64,")
            .unwrap();
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(body)
                .expect("base64"),
            expected
        );
    }
}

#[tokio::test]
async fn a_private_url_is_rejected_without_fetching() {
    let settings = ImageSettings {
        max_images: 600,
        max_total_bytes: 64 * 1024 * 1024,
        block_private_urls: true,
        dns_overrides: Default::default(),
    };
    let error = resolve(
        &[source_url(
            "http://127.0.0.1:1/image.png",
            "messages.0.content[0]",
        )],
        &settings,
    )
    .await
    .expect_err("private destination");
    assert_eq!(error.status, 400);
    let message = error.body["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("the destination filter rejects 127.0.0.1"),
        "{message}"
    );
}

#[tokio::test]
async fn an_invalid_image_is_the_api_400() {
    let settings = ImageSettings {
        max_images: 600,
        max_total_bytes: 64 * 1024 * 1024,
        block_private_urls: true,
        dns_overrides: Default::default(),
    };
    let error = resolve(
        &[source_data(b"not a png", "messages.0.content[0]")],
        &settings,
    )
    .await
    .expect_err("invalid image");
    assert_eq!(error.status, 400);
    assert!(
        error.body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unsupported image")
    );
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

#[tokio::test]
async fn count_tokens_includes_the_image_adjustment() {
    let bytes = png(16, 16, [0, 255, 255, 255]);
    let body = json!({
        "model": "DeepSeek-V4.1-Flash-1017",
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64::engine::general_purpose::STANDARD.encode(&bytes)}}
            ]
        }]
    });
    let gateway = gateway();
    let converted = serde_json::from_value::<MessagesRequest>(body.clone())
        .expect("request")
        .convert(gateway.options.clone())
        .expect("convert");
    let ids = gateway
        .prepare_conversation(&converted.conversation)
        .expect("encode")
        .ids
        .len();
    let expected = ids
        + ImageTokenSpec::v41()
            .calc_token_count(16, 16)
            .expect("tokens")
        - 1;
    let url = spawn_gateway("http://127.0.0.1:1", GatewayConfig::default()).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/messages/count_tokens"))
        .json(&body)
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    let value: Value = response.json().await.expect("body");
    assert_eq!(value["input_tokens"], expected);
}

#[tokio::test]
async fn too_many_images_are_the_api_400() {
    let bytes = png(1, 1, [0, 0, 0, 255]);
    let mut content = Vec::new();
    for _ in 0..601 {
        content.push(json!({
            "type": "image_url",
            "image_url": {"url": data_url(&bytes)}
        }));
    }
    let body = json!({
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": content}]
    });
    let url = spawn_gateway("http://127.0.0.1:1", GatewayConfig::default()).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/ds41/render/chat"))
        .json(&body)
        .send()
        .await
        .expect("render");
    assert_eq!(response.status(), 400);
    assert_eq!(response.headers()["x-ds41-gateway"], "api");
    let value: Value = response.json().await.expect("body");
    assert_eq!(value["error"]["type"], "invalid_request_error");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Too many images")
    );
}

struct Mock {
    body: Mutex<Option<Value>>,
}

async fn spawn_engine() -> (String, Arc<Mock>) {
    let mock = Arc::new(Mock {
        body: Mutex::new(None),
    });
    let handler = |State(mock): State<Arc<Mock>>, body: Bytes| async move {
        let value: Value = serde_json::from_slice(&body).expect("json");
        *mock.body.lock().unwrap() = Some(value);
        let events = [
            json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}}),
            json!({"choices": [{"index": 0, "finish_reason": "stop", "token_ids": [1], "logprobs": null}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}),
        ];
        let mut text = String::new();
        for event in events {
            text.push_str(&format!("data: {event}\n\n"));
        }
        text.push_str("data: [DONE]\n\n");
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(axum::body::Body::from(text))
            .expect("response")
    };
    let router = Router::new()
        .route("/inference/v1/generate", post(handler))
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    (format!("http://{address}"), mock)
}

#[tokio::test]
async fn the_api_sends_the_original_image_bytes_as_content_parts() {
    let bytes = png(16, 16, [255, 128, 0, 255]);
    let (engine, mock) = spawn_engine().await;
    let url = spawn_gateway(&engine, GatewayConfig::default()).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": data_url(&bytes)}}
                ]
            }],
            "thinking": {"type": "disabled"}
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    let sent = mock.body.lock().unwrap().clone().expect("engine body");
    let parts = sent["content_parts"].as_array().expect("content_parts");
    assert_eq!(parts.len(), 1);
    let encoded = parts[0]["url"].as_str().expect("url");
    let body = encoded
        .strip_prefix("data:application/octet-stream;base64,")
        .expect("data URL");
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(body)
            .expect("base64"),
        bytes
    );
}
