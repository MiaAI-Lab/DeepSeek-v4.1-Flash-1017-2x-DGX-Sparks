//! Authentication and admission control for the complete public router.
//!
//! This middleware runs before route extractors, so an unauthorized or queued request's
//! body is not read by a handler. An admitted request's body is read here, under the
//! API's 64 MiB limit and the configured body timeout, so a client that sends its body
//! slowly gives its slot back instead of holding it. The semaphore permit lives in the
//! response body: an SSE response therefore keeps its slot until the stream ends or the
//! client disconnects.

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http_body::{Body as HttpBody, Frame, SizeHint};
use pin_project_lite::pin_project;
use sha2::{Digest, Sha256};
use subtle::{Choice, ConstantTimeEq};
use tokio::sync::OwnedSemaphorePermit;

use crate::ApiError;
use crate::api::AppState;

/// The parsed contents of an API key file. Only fixed-size digests are retained, and
/// the custom `Debug` implementation prevents accidental secret disclosure.
#[derive(Clone)]
pub struct ApiKeys {
    hashes: Vec<[u8; 32]>,
}

impl ApiKeys {
    /// Parse one `key` or `label: key` per non-empty line.
    pub fn parse(contents: &str) -> Result<Self, String> {
        let mut hashes = Vec::new();
        for line in contents.lines().map(str::trim) {
            if line.is_empty() {
                continue;
            }
            let key = line
                .split_once(':')
                .map_or(line, |(_label, key)| key.trim());
            if key.is_empty() {
                return Err("the file contains an empty key".to_string());
            }
            hashes.push(Sha256::digest(key.as_bytes()).into());
        }
        if hashes.is_empty() {
            return Err("the file contains no API keys".to_string());
        }
        Ok(Self { hashes })
    }

    /// Compare a presented key against every configured digest without an early exit.
    fn matches(&self, presented: &str) -> bool {
        let presented = Sha256::digest(presented.as_bytes());
        self.hashes
            .iter()
            .fold(Choice::from(0), |matched, expected| {
                matched | expected.as_slice().ct_eq(presented.as_slice())
            })
            .into()
    }
}

impl fmt::Debug for ApiKeys {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ApiKeys")
            .field("count", &self.hashes.len())
            .field("material", &"[REDACTED]")
            .finish()
    }
}

/// Authenticate, then asynchronously queue for an admission slot. `GET /health` is
/// deliberately outside both checks so liveness probes remain available under load.
/// `GET /v1/models` and `GET /metrics` are authenticated but take no slot: they are cheap
/// reads, and behind long generations they waited 8-19 s, so monitors saw the server as down.
pub async fn guard(State(state): State<AppState>, request: Request, next: Next) -> Response {
    if request.method() == Method::GET && request.uri().path() == "/health" {
        return next.run(request).await;
    }

    if state.config.require_key && !authorized(request.headers(), state.config.api_keys.as_ref()) {
        return error_response(ApiError::new(
            StatusCode::UNAUTHORIZED.as_u16(),
            "missing or invalid API key",
            "authentication_error",
        ));
    }

    if request.method() == Method::GET && is_cheap_read(request.uri().path()) {
        return next.run(request).await;
    }

    let permit = match Arc::clone(&state.semaphore).acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => {
            return error_response(ApiError::internal("the request limiter is closed"));
        }
    };
    let request = match read_body(request, state.config.body_timeout).await {
        Ok(request) => request,
        Err(error) => return error_response(error),
    };
    let response = next.run(request).await;
    hold_permit(response, permit)
}

/// The largest request body any route takes (the API's 64 MiB).
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Buffer the body of an admitted request within the size limit and the timeout.
async fn read_body(request: Request, timeout: Option<Duration>) -> Result<Request, ApiError> {
    let too_large = || {
        ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE.as_u16(),
            format!("the request body is larger than {MAX_BODY_BYTES} bytes"),
            "request_too_large",
        )
    };
    // A declared length over the limit is refused before any of the body is read.
    let declared = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok());
    if declared.is_some_and(|length| length > MAX_BODY_BYTES as u64) {
        return Err(too_large());
    }
    let (parts, body) = request.into_parts();
    // At most the declared length up front (a lie only costs a regrow), never the limit.
    let capacity = declared.map_or(0, |length| length as usize);
    let read = collect_within(body, capacity, MAX_BODY_BYTES);
    let bytes = match timeout {
        Some(limit) => tokio::time::timeout(limit, read).await.map_err(|_| {
            ApiError::new(
                StatusCode::REQUEST_TIMEOUT.as_u16(),
                format!(
                    "the request body was not received within {} s",
                    limit.as_secs_f64()
                ),
                "invalid_request_error",
            )
        })?,
        None => read.await,
    };
    let bytes = bytes.map_err(|error| match error {
        BodyError::TooLarge => too_large(),
        BodyError::Read(error) => {
            ApiError::bad_request(format!("cannot read the request body: {error}"))
        }
    })?;
    Ok(Request::from_parts(parts, Body::from(bytes)))
}

enum BodyError {
    TooLarge,
    Read(axum::Error),
}

/// Copy the body's data frames into one buffer as they arrive, dropping each frame at
/// once, so the memory is the buffer itself however finely the client splits the body
/// (a collector that keeps every frame until the end would hold one allocation each).
/// The buffer grows by doubling but never past `limit`; while it grows, the old and the
/// new allocation can coexist for a moment, so the peak stays under twice the limit.
async fn collect_within(body: Body, capacity: usize, limit: usize) -> Result<Bytes, BodyError> {
    use http_body_util::BodyExt;
    let mut body = body;
    let mut buffer: Vec<u8> = Vec::with_capacity(capacity.min(limit));
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(BodyError::Read)?;
        if let Ok(data) = frame.into_data() {
            let needed = buffer.len() + data.len();
            if needed > limit {
                return Err(BodyError::TooLarge);
            }
            if needed > buffer.capacity() {
                let target = needed.max(buffer.capacity().saturating_mul(2)).min(limit);
                buffer.reserve_exact(target - buffer.len());
            }
            buffer.extend_from_slice(&data);
        }
    }
    Ok(Bytes::from(buffer))
}

fn authorized(headers: &HeaderMap, keys: Option<&ApiKeys>) -> bool {
    let Some(keys) = keys else {
        return false;
    };
    let bearer_valid = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(bearer_key)
        .is_some_and(|key| keys.matches(key));
    let x_api_key_valid = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .is_some_and(|key| keys.matches(key));
    bearer_valid | x_api_key_valid
}

fn bearer_key(value: &str) -> Option<&str> {
    let (scheme, key) = value.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| key.trim())
}

fn error_response(error: ApiError) -> Response {
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(error.body)).into_response()
}

/// Reads that answer from the engine's state at once and never generate: outside the admission limit.
fn is_cheap_read(path: &str) -> bool {
    matches!(path, "/v1/models" | "/metrics")
}

fn hold_permit(response: Response, permit: OwnedSemaphorePermit) -> Response {
    let (parts, body) = response.into_parts();
    let body = Body::new(PermitBody {
        body,
        permit: Some(permit),
    });
    Response::from_parts(parts, body)
}

pin_project! {
    /// An HTTP body whose owned permit is released on EOF or when the body is dropped.
    struct PermitBody {
        #[pin]
        body: Body,
        permit: Option<OwnedSemaphorePermit>,
    }
}

impl HttpBody for PermitBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.project();
        let polled = this.body.poll_frame(context);
        if let Poll::Ready(None) = &polled {
            this.permit.take();
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::ApiKeys;
    use crate::parity::GatewayConfig;

    #[test]
    fn keys_accept_labels_and_debug_is_redacted() {
        let keys = ApiKeys::parse("first: secret-one\nsecret-two\n").expect("keys");
        assert!(keys.matches("secret-one"));
        assert!(keys.matches("secret-two"));
        assert!(!keys.matches("secret"));
        let config = GatewayConfig {
            require_key: true,
            api_keys: Some(keys),
            ..GatewayConfig::default()
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("REDACTED"));
        assert!(!debug.contains("secret-one"));
        assert!(!debug.contains("secret-two"));
    }

    #[test]
    fn empty_key_files_are_rejected() {
        assert!(ApiKeys::parse("\n \n").is_err());
        assert!(ApiKeys::parse("label:  \n").is_err());
    }
}
