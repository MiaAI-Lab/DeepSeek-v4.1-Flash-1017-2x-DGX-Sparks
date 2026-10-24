//! `ds41-gateway`: the DeepSeek API front end for TensorFold's token-in/token-out route.

use std::collections::HashMap;
use std::error::Error;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use ds41_gateway::api::AppState;
use ds41_gateway::engine::EngineClient;
use ds41_gateway::health;
use ds41_gateway::parity::GatewayConfig;
use ds41_gateway::{Gateway, GatewaySettings};
use serde_json::Value;
use tokio::net::TcpListener;

#[derive(Debug, Parser)]
#[command(name = "ds41-gateway", about = "DeepSeek API parity gateway")]
struct Cli {
    /// Address to listen on (`HOST:PORT`).
    #[arg(long)]
    listen: Option<String>,
    /// Engine URL.
    #[arg(long)]
    engine: Option<String>,
    /// Public model name served by this gateway.
    #[arg(long)]
    served_name: Option<String>,
    /// Additional accepted model alias (repeatable).
    #[arg(long = "alias")]
    aliases: Vec<String>,
    /// Served `tokenizer.json`.
    #[arg(long, default_value = "/served/tokenizer.json")]
    tokenizer: PathBuf,
    /// Spelling of the image token in the served tokenizer.
    #[arg(long)]
    image_token: Option<String>,
    /// Default thinking mode for requests that do not specify one.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    default_thinking: Option<bool>,
    /// Effort injected into requests without one.
    #[arg(long)]
    default_effort: Option<String>,
    /// Context limit; auto from `/v1/models` when absent.
    #[arg(long)]
    max_model_len: Option<u32>,
    /// Return the API's 400 instead of falling back.
    #[arg(long)]
    strict: bool,
    /// Reject private image destinations (default true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    block_private_urls: Option<bool>,
    /// Resolve sampling with deepseek-recipe (default true).
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    api_sampling: Option<bool>,
    /// Maximum images per request.
    #[arg(long)]
    max_images: Option<usize>,
    /// Maximum total image megabytes per request.
    #[arg(long)]
    max_image_mb: Option<usize>,
    /// Require `Authorization`/`x-api-key` before rendering.
    #[arg(long)]
    require_key: bool,
    /// Log level.
    #[arg(long)]
    log: Option<String>,
    /// Image DNS override for tests (`HOST=IP`, repeatable).
    #[arg(long = "dns-override", value_name = "HOST=IP")]
    dns_overrides: Vec<String>,
}

const DEFAULT_IMAGE_TOKEN: &str = "<｜deepseek_image｜>";

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn env_bool(name: &str) -> Option<bool> {
    env(name).and_then(|value| match value.as_str() {
        "1" | "true" | "TRUE" | "yes" => Some(true),
        "0" | "false" | "FALSE" | "no" => Some(false),
        _ => None,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let host = env("HOST").unwrap_or_else(|| "0.0.0.0".to_string());
    let port = env("PORT").unwrap_or_else(|| "8888".to_string());
    let listen = cli.listen.unwrap_or_else(|| format!("{host}:{port}"));
    let engine = cli.engine.unwrap_or_else(|| {
        env("ENGINE_PORT")
            .map(|port| format!("http://127.0.0.1:{port}"))
            .unwrap_or_else(|| "http://127.0.0.1:8887".to_string())
    });
    let served_name = cli
        .served_name
        .or_else(|| env("SERVED_NAME"))
        .unwrap_or_else(|| "DeepSeek-V4.1-Flash-1017".to_string());
    let image_token = cli
        .image_token
        .or_else(|| env("GATEWAY_IMAGE_TOKEN"))
        .unwrap_or_else(|| DEFAULT_IMAGE_TOKEN.to_string());
    let default_thinking = cli
        .default_thinking
        .or_else(|| env_bool("THINKING"))
        .unwrap_or(true);
    let default_effort = cli.default_effort.or_else(|| env("REASONING_EFFORT"));
    let strict = cli.strict || env_bool("GATEWAY_STRICT").unwrap_or(false);
    let block_private_urls = cli
        .block_private_urls
        .or_else(|| env_bool("GATEWAY_BLOCK_PRIVATE_URLS"))
        .unwrap_or(true);
    let api_sampling = cli
        .api_sampling
        .or_else(|| env_bool("GATEWAY_API_SAMPLING"))
        .unwrap_or(true);
    let max_images = cli
        .max_images
        .or_else(|| env("GATEWAY_MAX_IMAGES").and_then(|value| value.parse().ok()))
        .unwrap_or(600);
    let max_image_mb = cli
        .max_image_mb
        .or_else(|| env("GATEWAY_MAX_IMAGE_MB").and_then(|value| value.parse().ok()))
        .unwrap_or(64);
    let require_key = cli.require_key || env("API_KEY").is_some();
    let log_level = cli
        .log
        .or_else(|| env("GATEWAY_LOG"))
        .unwrap_or_else(|| "info".to_string());
    let dns_overrides = parse_dns_overrides(&cli.dns_overrides)?;

    let gateway = Gateway::from_file(
        &cli.tokenizer,
        GatewaySettings {
            served_name: served_name.clone(),
            default_thinking,
            image_token,
            aliases: cli.aliases,
        },
    )?;
    let engine_client = EngineClient::new(engine)?;
    let max_model_len = match cli.max_model_len {
        Some(limit) => limit,
        None => auto_max_model_len(&engine_client).await.unwrap_or(65_536),
    };
    let engine_logprobs = auto_engine_logprobs(&engine_client).await;
    let config = GatewayConfig {
        max_model_len,
        api_sampling,
        default_effort,
        strict,
        require_key,
        block_private_urls,
        max_images,
        max_image_mb,
        log_level,
        served_name: served_name.clone(),
        engine_logprobs,
        dns_overrides,
    };
    let state = AppState {
        gateway: Arc::new(gateway),
        engine: engine_client,
        config: Arc::new(config),
    };
    let listener = TcpListener::bind(&listen).await?;
    eprintln!(
        "ds41-gateway listening on {listen} (engine {}, model {served_name}, max_model_len {max_model_len})",
        state.engine.base()
    );
    axum::serve(listener, health::router(state)).await?;
    Ok(())
}

fn parse_dns_overrides(values: &[String]) -> Result<HashMap<String, IpAddr>, String> {
    let mut overrides = HashMap::new();
    for value in values {
        let (host, address) = value
            .split_once('=')
            .ok_or_else(|| format!("--dns-override {value:?}: want HOST=IP"))?;
        let address = address
            .parse()
            .map_err(|_| format!("--dns-override {value:?}: invalid IP address"))?;
        overrides.insert(host.to_string(), address);
    }
    Ok(overrides)
}

/// Ask the engine whether it reports logprobs support; unknown means "allow".
async fn auto_engine_logprobs(engine: &EngineClient) -> Option<bool> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .ok()?;
    let response = client
        .get(format!("{}/health", engine.base()))
        .send()
        .await
        .ok()?;
    if response.status() != reqwest::StatusCode::OK {
        return None;
    }
    let body: Value = response.json().await.ok()?;
    body.get("logprobs")
        .or_else(|| body.get("engine").and_then(|engine| engine.get("logprobs")))
        .and_then(Value::as_bool)
}

/// Ask the engine for its context limit; a short timeout keeps startup under two seconds.
async fn auto_max_model_len(engine: &EngineClient) -> Option<u32> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .ok()?;
    let value: Value = client
        .get(format!("{}/v1/models", engine.base()))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    value
        .get("data")
        .and_then(Value::as_array)
        .and_then(|data| data.first())
        .and_then(|model| {
            model
                .get("max_model_len")
                .or_else(|| model.get("max_model_length"))
                .or_else(|| model.get("context_length"))
        })
        .or_else(|| value.get("max_model_len"))
        .and_then(Value::as_u64)
        .map(|value| value as u32)
}

#[cfg(test)]
mod tests {
    use super::DEFAULT_IMAGE_TOKEN;

    #[test]
    fn the_default_image_token_is_the_served_spelling() {
        assert_eq!(DEFAULT_IMAGE_TOKEN, "<｜deepseek_image｜>");
        assert_ne!(DEFAULT_IMAGE_TOKEN, "<｜end_of_query｜>");
    }
}
