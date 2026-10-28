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
use ds41_gateway::listener::LimitedListener;
use ds41_gateway::parity::GatewayConfig;
use ds41_gateway::security::ApiKeys;
use ds41_gateway::{Gateway, GatewaySettings};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

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
    /// Require a valid `Authorization: Bearer` or `x-api-key` credential.
    #[arg(long)]
    require_key: bool,
    /// API key file used with `--require-key` (or API_KEY_FILE).
    #[arg(long, value_name = "PATH")]
    api_key_file: Option<PathBuf>,
    /// Maximum admitted requests; additional requests wait (default: 3).
    #[arg(long, value_parser = parse_positive_usize)]
    max_concurrent: Option<usize>,
    /// Maximum open client connections; further clients wait to be accepted (default: 64).
    #[arg(long, value_parser = parse_positive_usize)]
    max_connections: Option<usize>,
    /// Seconds an admitted request may take to send its body (default: 60).
    #[arg(long, value_parser = parse_positive_usize)]
    body_timeout: Option<usize>,
    /// Log level.
    #[arg(long)]
    log: Option<String>,
    /// Image DNS override for tests (`HOST=IP`, repeatable).
    #[arg(long = "dns-override", value_name = "HOST=IP")]
    dns_overrides: Vec<String>,
}

const DEFAULT_IMAGE_TOKEN: &str = "<｜deepseek_image｜>";
// Open connections: admitted requests, the ones queued behind them and idle keep-alives.
const DEFAULT_MAX_CONNECTIONS: usize = 64;
// A 64 MiB body at about 1 MiB/s; a slower client gives its admission slot back.
const DEFAULT_BODY_TIMEOUT_SECS: usize = 60;
// Measured peaks of the release binary: 57 MiB idle (64 here), up to 784 MiB an admitted request (800 here), against
// the recipe's 3 GiB container default (GATEWAY_MEMORY=3g in scripts/config.sh).
const BUDGET_MIB: usize = 3072;
const IDLE_MIB: usize = 64;
const REQUEST_MIB: usize = 800;
const DEFAULT_MAX_CONCURRENT: usize = (BUDGET_MIB - IDLE_MIB) / REQUEST_MIB;

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("must be a positive integer, got {value:?}"))?;
    if parsed == 0 {
        return Err("must be greater than zero".to_string());
    }
    Ok(parsed)
}

fn load_api_keys(require_key: bool, path: Option<PathBuf>) -> Result<Option<ApiKeys>, String> {
    if !require_key {
        return Ok(None);
    }
    let path = path
        .ok_or_else(|| "--require-key requires --api-key-file PATH or API_KEY_FILE".to_string())?;
    let contents = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read API key file {}: {error}", path.display()))?;
    ApiKeys::parse(&contents)
        .map(Some)
        .map_err(|error| format!("invalid API key file {}: {error}", path.display()))
}

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
    let require_key = cli.require_key || env_bool("API_KEY").unwrap_or(false);
    let api_key_file = cli
        .api_key_file
        .or_else(|| env("API_KEY_FILE").map(PathBuf::from));
    let api_keys = load_api_keys(require_key, api_key_file)?;
    let max_concurrent = match cli.max_concurrent {
        Some(value) => value,
        None => env("GATEWAY_MAX_CONCURRENT")
            .map(|value| parse_positive_usize(&value))
            .transpose()?
            .unwrap_or(DEFAULT_MAX_CONCURRENT),
    };
    let max_connections = match cli.max_connections {
        Some(value) => value,
        None => env("GATEWAY_MAX_CONNECTIONS")
            .map(|value| parse_positive_usize(&value))
            .transpose()?
            .unwrap_or(DEFAULT_MAX_CONNECTIONS),
    };
    let body_timeout = match cli.body_timeout {
        Some(value) => value,
        None => env("GATEWAY_BODY_TIMEOUT")
            .map(|value| parse_positive_usize(&value))
            .transpose()?
            .unwrap_or(DEFAULT_BODY_TIMEOUT_SECS),
    };
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
        api_keys,
        block_private_urls,
        max_images,
        max_image_mb,
        log_level,
        served_name: served_name.clone(),
        engine_logprobs,
        dns_overrides,
        body_timeout: Some(Duration::from_secs(body_timeout as u64)),
    };
    let state = AppState {
        gateway: Arc::new(gateway),
        engine: engine_client,
        config: Arc::new(config),
        semaphore: Arc::new(Semaphore::new(max_concurrent)),
    };
    let listener = LimitedListener::new(TcpListener::bind(&listen).await?, max_connections);
    eprintln!(
        "ds41-gateway listening on {listen} (engine {}, model {served_name}, max_model_len {max_model_len}, \
         {max_concurrent} admitted requests, {max_connections} connections, body timeout {body_timeout} s)",
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
    use super::{
        BUDGET_MIB, DEFAULT_IMAGE_TOKEN, DEFAULT_MAX_CONCURRENT, IDLE_MIB, REQUEST_MIB,
        load_api_keys, parse_positive_usize,
    };

    #[test]
    fn the_default_image_token_is_the_served_spelling() {
        assert_eq!(DEFAULT_IMAGE_TOKEN, "<｜deepseek_image｜>");
        assert_ne!(DEFAULT_IMAGE_TOKEN, "<｜end_of_query｜>");
    }

    #[test]
    fn require_key_needs_a_readable_nonempty_key_file() {
        let error = load_api_keys(true, None).expect_err("missing path");
        assert!(error.contains("requires --api-key-file"));

        let missing =
            std::env::temp_dir().join(format!("ds41-gateway-missing-key-{}", std::process::id()));
        let error = load_api_keys(true, Some(missing)).expect_err("missing file");
        assert!(error.contains("cannot read API key file"));

        let empty =
            std::env::temp_dir().join(format!("ds41-gateway-empty-key-{}", std::process::id()));
        std::fs::write(&empty, "\n  \n").expect("empty key fixture");
        let error = load_api_keys(true, Some(empty.clone())).expect_err("empty file");
        let _ = std::fs::remove_file(empty);
        assert!(error.contains("contains no API keys"));
    }

    #[test]
    fn concurrency_default_and_parser_enforce_the_memory_budget() {
        assert_eq!(DEFAULT_MAX_CONCURRENT, 3);
        // The budget is a property of the constants, so it is checked at compile time.
        const { assert!(IDLE_MIB + DEFAULT_MAX_CONCURRENT * REQUEST_MIB <= BUDGET_MIB) };
        const { assert!(IDLE_MIB + (DEFAULT_MAX_CONCURRENT + 1) * REQUEST_MIB > BUDGET_MIB) };
        assert_eq!(parse_positive_usize("4"), Ok(4));
        assert!(parse_positive_usize("0").is_err());
        assert!(parse_positive_usize("not-a-number").is_err());
    }
}
