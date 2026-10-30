//! Transparent proxy for the engine routes the gateway does not interpret.
//!
//! The request method, raw path and query, headers (except hop-by-hop ones and
//! `content-length`), and body go to the engine unchanged.  The response status,
//! headers, and streaming body come back the same way.  The gateway's API
//! fallback uses the same hop and adds `x-ds41-gateway: fallback`.

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use tokio_stream::StreamExt;

use crate::ApiError;
use crate::api::AppState;

/// The body limit of the proxy hop (the API's 64 MiB).
const MAX_BODY: usize = 64 * 1024 * 1024;

/// Proxy one request whose path the gateway does not own.
pub async fn proxy_handler(State(state): State<AppState>, request: Request) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let headers = request.headers().clone();
    let body = match axum::body::to_bytes(request.into_body(), MAX_BODY).await {
        Ok(body) => body,
        Err(error) => {
            return api_error_response(&ApiError::bad_request(format!(
                "cannot read the request body: {error}"
            )));
        }
    };
    let path_and_query = uri
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_else(|| uri.path().to_string());
    forward(&state, method, &path_and_query, headers, body, None).await
}

/// Send one request to the engine and stream its response back.
pub async fn forward(
    state: &AppState,
    method: Method,
    path_and_query: &str,
    headers: HeaderMap,
    body: Bytes,
    extra: Option<(&'static str, &'static str)>,
) -> Response {
    let url = format!(
        "{}{}",
        state.engine.base(),
        if path_and_query.starts_with('/') {
            path_and_query.to_string()
        } else {
            format!("/{path_and_query}")
        }
    );
    let mut outgoing = HeaderMap::new();
    for (name, value) in &headers {
        if is_hop_by_hop(name) || name == "content-length" {
            continue;
        }
        outgoing.insert(name.clone(), value.clone());
    }
    let method =
        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET);
    let response = state
        .engine
        .http()
        .request(method, &url)
        .headers(outgoing)
        .body(body)
        .send()
        .await;
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            let mut response = api_error_response(&ApiError::bad_gateway(format!(
                "the model server is unreachable: {error}"
            )));
            add_extra(&mut response, extra);
            return response;
        }
    };

    let status = StatusCode::from_u16(response.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut builder = Response::builder().status(status);
    for (name, value) in response.headers() {
        if is_hop_by_hop(name) {
            continue;
        }
        builder = builder.header(name, value);
    }
    let stream = response
        .bytes_stream()
        .map(|item| item.map_err(|error| std::io::Error::other(error.to_string())));
    let mut response = match builder.body(Body::from_stream(stream)) {
        Ok(response) => response,
        Err(error) => api_error_response(&ApiError::internal(format!(
            "cannot build the response: {error}"
        ))),
    };
    add_extra(&mut response, extra);
    response
}

fn api_error_response(error: &ApiError) -> Response {
    let status = StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, axum::Json(error.body.clone())).into_response()
}

fn add_extra(response: &mut Response, extra: Option<(&'static str, &'static str)>) {
    if let Some((name, value)) = extra
        && let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        )
    {
        response.headers_mut().insert(name, value);
    }
}

/// Hop-by-hop headers that must not cross the proxy hop.
fn is_hop_by_hop(name: &HeaderName) -> bool {
    let name = name.as_str();
    name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("keep-alive")
        || name.eq_ignore_ascii_case("te")
        || name.eq_ignore_ascii_case("trailer")
        || name.eq_ignore_ascii_case("transfer-encoding")
        || name.eq_ignore_ascii_case("upgrade")
        || name.to_ascii_lowercase().starts_with("proxy-")
}
