//! Authentication and admission control for the complete public router.
//!
//! This middleware runs before route extractors, so an unauthorized or queued request's
//! body is not read by a handler. The semaphore permit lives in the response body: an
//! SSE response therefore keeps its slot until the stream ends or the client disconnects.

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

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

    let permit = match Arc::clone(&state.semaphore).acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => {
            return error_response(ApiError::internal("the request limiter is closed"));
        }
    };
    let response = next.run(request).await;
    hold_permit(response, permit)
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
