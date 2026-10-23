//! `/health`, `/ds41/render/*`, and the gateway's router assembly.

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use crate::api::{self, AppState};
use crate::parity::Protocol;

/// Assemble the full gateway router.
pub fn router(state: AppState) -> Router {
    let api = api::router(state.clone());
    let extra = Router::new()
        .route("/health", get(health))
        .route("/ds41/render/{protocol}", post(render))
        .route("/inference", any(inference_not_found))
        .route("/inference/{*path}", any(inference_not_found))
        .fallback(any(crate::proxy::proxy_handler))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state);
    api.merge(extra)
}

async fn health(State(state): State<AppState>) -> Response {
    match state.engine.health().await {
        Ok((status, engine)) => {
            let upstream = if status == 200 {
                "ok".to_string()
            } else {
                format!("status {status}")
            };
            let body = json!({
                "gateway": "ok",
                "upstream": upstream,
                "engine": engine,
            });
            let http_status = if status == 200 {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            (http_status, Json(body)).into_response()
        }
        Err(error) => {
            let body = json!({
                "gateway": "ok",
                "upstream": format!("unreachable: {error}"),
                "engine": Value::Null,
            });
            (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
        }
    }
}

async fn render(
    State(state): State<AppState>,
    Path(protocol): Path<String>,
    body: Bytes,
) -> Response {
    let protocol = match protocol.as_str() {
        "chat" => Protocol::Chat,
        "messages" => Protocol::Messages,
        "responses" => Protocol::Responses,
        _ => {
            return (StatusCode::NOT_FOUND, Json(json!({"detail": "Not Found"}))).into_response();
        }
    };
    match api::render_handler(&state, protocol, &body).await {
        Ok(value) => {
            let mut response = (StatusCode::OK, Json(value)).into_response();
            response
                .headers_mut()
                .insert("x-ds41-gateway", HeaderValue::from_static("api"));
            response
        }
        Err(error) => {
            let status =
                StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let mut response = (status, Json(error.body)).into_response();
            response
                .headers_mut()
                .insert("x-ds41-gateway", HeaderValue::from_static("api"));
            response
        }
    }
}

async fn inference_not_found() -> Response {
    (StatusCode::NOT_FOUND, Json(json!({"detail": "Not Found"}))).into_response()
}
