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
use ds41_gateway::listener::{self, LimitedListener};
use ds41_gateway::parity::{DEFAULT_MAX_PROMPT_BYTES, GatewayConfig};
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
    #[arg(long, value_parser = parse_positive_usize)]
    max_images: Option<usize>,
    /// Maximum total image megabytes per request.
    #[arg(long, value_parser = parse_positive_usize)]
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
    /// Maximum open client connections; further clients wait to be accepted (default: 256).
    #[arg(long, value_parser = parse_positive_usize)]
    max_connections: Option<usize>,
    /// Maximum open connections from one IP address; more are closed at once (default: 32).
    #[arg(long, value_parser = parse_positive_usize)]
    max_connections_per_peer: Option<usize>,
    /// Seconds a connection has to send a request header, new or kept alive (default: 30).
    #[arg(long, value_parser = parse_positive_usize)]
    header_timeout: Option<usize>,
    /// Seconds an admitted request may take to send its body (default: 60).
    #[arg(long, value_parser = parse_positive_usize)]
    body_timeout: Option<usize>,
    /// Seconds a connection's writes may stay blocked (a client not reading its reply) before it is closed (default: 60).
    #[arg(long, value_parser = parse_positive_usize)]
    write_timeout: Option<usize>,
    /// Largest rendered prompt the gateway encodes, in MiB; a larger one is a 400 (default: 8).
    #[arg(long, value_parser = parse_positive_usize)]
    max_prompt_mb: Option<usize>,
    /// Log level.
    #[arg(long)]
    log: Option<String>,
    /// Image DNS override for tests (`HOST=IP`, repeatable).
    #[arg(long = "dns-override", value_name = "HOST=IP")]
    dns_overrides: Vec<String>,
}

const DEFAULT_IMAGE_TOKEN: &str = "<｜deepseek_image｜>";
// Open connections: admitted requests, the ones queued behind them and idle keep-alives; and from one address.
const DEFAULT_MAX_CONNECTIONS: usize = 256;
const DEFAULT_MAX_CONNECTIONS_PER_PEER: usize = 32;
// hyper's own default for the request header; it also closes keep-alive connections idle that long.
const DEFAULT_HEADER_TIMEOUT_SECS: usize = 30;
// A 64 MiB body at about 1 MiB/s; a slower client gives its admission slot back.
const DEFAULT_BODY_TIMEOUT_SECS: usize = 60;
// A client that has not taken a byte of its reply for this long is gone or stalling; its permit goes back.
const DEFAULT_WRITE_TIMEOUT_SECS: usize = 60;
const DEFAULT_MAX_IMAGES: usize = 600;
const DEFAULT_MAX_IMAGE_MB: usize = 64;
// Admitted requests. Measured peaks of the release binary at a 1M-token context, one encode at a time (three runs
// each, the highest): 57 MiB idle; one request with an 8 MiB prompt and 55 MiB of images 1,247 MiB; three of them at
// once 1,701 MiB, under the recipe's 3 GiB container cap (GATEWAY_MEMORY=3g in scripts/config.sh). At one token a byte
// the token count keeps three with images at 1,587 MiB (4,011 MiB without it). Each further
// admitted request adds about 200 MiB (its body, JSON and images); the encode itself is not repeated.
const DEFAULT_MAX_CONCURRENT: usize = 3;

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

/// `API_KEY` from the environment. Unset is off; any value that is not a clear on or off (empty, unknown, not text)
/// is an error, since reading it as "off" would leave the API open.
fn api_key_setting(value: Result<String, std::env::VarError>) -> Result<bool, String> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => Err("API_KEY is not valid text".to_string()),
        Ok(value) => parse_bool(&value).ok_or_else(|| {
            format!("API_KEY is 1 or 0 (true/false, yes/no, any case), not {value:?}")
        }),
    }
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// 1/0, true/false or yes/no, in any case.
fn parse_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Some(true),
        "0" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// A boolean from the command line, else from `name` in the environment, else `default`. An empty variable is unset,
/// as everywhere here; any other value that is not a clear on or off stops the gateway instead of being ignored.
fn bool_setting(cli: Option<bool>, name: &str, default: bool) -> Result<bool, String> {
    match cli {
        Some(value) => Ok(value),
        None => env_bool(name, std::env::var(name)).map(|value| value.unwrap_or(default)),
    }
}

fn env_bool(name: &str, value: Result<String, std::env::VarError>) -> Result<Option<bool>, String> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{name} is not valid text")),
        Ok(value) if value.is_empty() => Ok(None),
        Ok(value) => parse_bool(&value).map(Some).ok_or_else(|| {
            format!("{name} is 1 or 0 (true/false, yes/no, any case), not {value:?}")
        }),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let host = env("HOST").unwrap_or_else(|| "0.0.0.0".to_string());
    let port = env("PORT").unwrap_or_else(|| "8888".to_string());
    // An IPv6 HOST goes in brackets in HOST:PORT.
    let listen = cli.listen.unwrap_or_else(|| {
        if host.contains(':') {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        }
    });
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
    let default_thinking = bool_setting(cli.default_thinking, "THINKING", true)?;
    let default_effort = cli.default_effort.or_else(|| env("REASONING_EFFORT"));
    let strict = cli.strict || bool_setting(None, "GATEWAY_STRICT", false)?;
    let block_private_urls =
        bool_setting(cli.block_private_urls, "GATEWAY_BLOCK_PRIVATE_URLS", true)?;
    let api_sampling = bool_setting(cli.api_sampling, "GATEWAY_API_SAMPLING", true)?;
    let max_images = positive_setting(cli.max_images, "GATEWAY_MAX_IMAGES", DEFAULT_MAX_IMAGES)?;
    let max_image_mb = positive_setting(
        cli.max_image_mb,
        "GATEWAY_MAX_IMAGE_MB",
        DEFAULT_MAX_IMAGE_MB,
    )?;
    let require_key = cli.require_key || api_key_setting(std::env::var("API_KEY"))?;
    let api_key_file = cli
        .api_key_file
        .or_else(|| env("API_KEY_FILE").map(PathBuf::from));
    let api_keys = load_api_keys(require_key, api_key_file)?;
    let max_concurrent = positive_setting(
        cli.max_concurrent,
        "GATEWAY_MAX_CONCURRENT",
        DEFAULT_MAX_CONCURRENT,
    )?;
    let max_connections = positive_setting(
        cli.max_connections,
        "GATEWAY_MAX_CONNECTIONS",
        DEFAULT_MAX_CONNECTIONS,
    )?;
    let max_connections_per_peer = positive_setting(
        cli.max_connections_per_peer,
        "GATEWAY_MAX_CONNECTIONS_PER_PEER",
        DEFAULT_MAX_CONNECTIONS_PER_PEER,
    )?;
    let header_timeout = positive_setting(
        cli.header_timeout,
        "GATEWAY_HEADER_TIMEOUT",
        DEFAULT_HEADER_TIMEOUT_SECS,
    )?;
    let body_timeout = positive_setting(
        cli.body_timeout,
        "GATEWAY_BODY_TIMEOUT",
        DEFAULT_BODY_TIMEOUT_SECS,
    )?;
    let write_timeout = positive_setting(
        cli.write_timeout,
        "GATEWAY_WRITE_TIMEOUT",
        DEFAULT_WRITE_TIMEOUT_SECS,
    )?;
    let max_prompt_mb = positive_setting(
        cli.max_prompt_mb,
        "GATEWAY_MAX_PROMPT_MB",
        DEFAULT_MAX_PROMPT_BYTES / (1024 * 1024),
    )?;
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
        max_prompt_bytes: max_prompt_mb.saturating_mul(1024 * 1024),
        encode_slot: Arc::new(Semaphore::new(1)),
    };
    let state = AppState {
        gateway: Arc::new(gateway),
        engine: engine_client,
        config: Arc::new(config),
        semaphore: Arc::new(Semaphore::new(max_concurrent)),
    };
    let listener = LimitedListener::new(
        TcpListener::bind(&listen).await?,
        max_connections,
        max_connections_per_peer,
    );
    eprintln!(
        "ds41-gateway listening on {listen} (engine {}, model {served_name}, max_model_len {max_model_len}, \
         {max_concurrent} admitted requests, {max_connections} connections ({max_connections_per_peer} a peer), \
         header timeout {header_timeout} s, body timeout {body_timeout} s, write timeout {write_timeout} s, prompts up to {max_prompt_mb} MiB)",
        state.engine.base()
    );
    listener::serve(
        listener,
        health::router(state),
        Duration::from_secs(header_timeout as u64),
        Duration::from_secs(write_timeout as u64),
    )
    .await?;
    Ok(())
}

/// A positive count from the command line, else from `name` in the environment, else `default`.
/// An empty variable is unset, as everywhere here; one that is not text is an error, not the default.
fn positive_setting(cli: Option<usize>, name: &str, default: usize) -> Result<usize, String> {
    if cli.is_some() {
        return count_setting(cli, name, None, default);
    }
    let value = match std::env::var(name) {
        Ok(value) => Some(value).filter(|value| !value.is_empty()),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => return Err(format!("{name} is not valid text")),
    };
    count_setting(cli, name, value, default)
}

/// [`positive_setting`] with the variable's value given: an invalid one is an error, never the default.
fn count_setting(
    cli: Option<usize>,
    name: &str,
    value: Option<String>,
    default: usize,
) -> Result<usize, String> {
    match cli {
        Some(value) => Ok(value),
        None => value
            .map(|value| parse_positive_usize(&value).map_err(|error| format!("{name}: {error}")))
            .transpose()
            .map(|value| value.unwrap_or(default)),
    }
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
        DEFAULT_IMAGE_TOKEN, DEFAULT_MAX_CONCURRENT, api_key_setting, count_setting, env_bool,
        load_api_keys, parse_bool, parse_positive_usize,
    };

    #[test]
    fn api_key_is_on_off_or_an_error() {
        use std::env::VarError;
        assert_eq!(api_key_setting(Err(VarError::NotPresent)), Ok(false));
        assert_eq!(api_key_setting(Ok("1".into())), Ok(true));
        assert_eq!(api_key_setting(Ok("yes".into())), Ok(true));
        assert_eq!(api_key_setting(Ok("0".into())), Ok(false));
        for (spelling, on) in [
            ("YES", true),
            ("True", true),
            ("No", false),
            ("FALSE", false),
        ] {
            assert_eq!(api_key_setting(Ok(spelling.into())), Ok(on), "{spelling:?}");
            assert_eq!(parse_bool(spelling), Some(on), "{spelling:?}");
        }
        for bad in ["", "enabled", "on", "2"] {
            assert!(api_key_setting(Ok(bad.into())).is_err(), "{bad:?}");
        }
        let not_text = std::ffi::OsString::from("x");
        assert!(api_key_setting(Err(VarError::NotUnicode(not_text))).is_err());
    }

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
        assert_eq!(parse_positive_usize("4"), Ok(4));
        assert!(parse_positive_usize("0").is_err());
        assert!(parse_positive_usize("not-a-number").is_err());
    }

    #[test]
    fn a_boolean_setting_that_is_not_on_or_off_stops_startup() {
        use std::env::VarError;
        let name = "GATEWAY_STRICT";
        assert_eq!(env_bool(name, Err(VarError::NotPresent)), Ok(None));
        assert_eq!(env_bool(name, Ok(String::new())), Ok(None));
        assert_eq!(env_bool(name, Ok("YES".into())), Ok(Some(true)));
        assert_eq!(env_bool(name, Ok("0".into())), Ok(Some(false)));
        for bad in ["on", "typo", "2"] {
            let error = env_bool(name, Ok(bad.into())).expect_err(bad);
            assert!(error.starts_with("GATEWAY_STRICT is 1 or 0"), "{error}");
        }
        let not_text = std::ffi::OsString::from("x");
        assert!(env_bool(name, Err(VarError::NotUnicode(not_text))).is_err());
    }

    #[test]
    fn a_count_setting_that_is_not_a_positive_integer_stops_startup() {
        // The command line wins; without it the default stands only when the variable is unset.
        let set = |value: &str| Some(value.to_string());
        assert_eq!(
            count_setting(Some(5), "GATEWAY_MAX_IMAGES", set("9"), 600),
            Ok(5)
        );
        assert_eq!(
            count_setting(None, "GATEWAY_MAX_IMAGES", None, 600),
            Ok(600)
        );
        assert_eq!(
            count_setting(None, "GATEWAY_MAX_IMAGES", set("9"), 600),
            Ok(9)
        );
        for bad in ["64g", "600foo", "0", "-1"] {
            let error = count_setting(None, "GATEWAY_MAX_IMAGE_MB", set(bad), 64).expect_err(bad);
            assert!(error.starts_with("GATEWAY_MAX_IMAGE_MB:"), "{error}");
        }
    }
}
