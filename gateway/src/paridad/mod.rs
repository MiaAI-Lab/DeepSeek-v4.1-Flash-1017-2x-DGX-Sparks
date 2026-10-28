//! The `ds41-paridad` comparator: cases, render comparison, end-to-end checks and
//! the report.  The binary is a thin wrapper around this module.

pub mod api;
pub mod caso;
pub mod diff;
pub mod e2e;
pub mod informe;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Gateway;
use crate::parity::{GatewayConfig, Protocol};

use self::api::{LadoApi, ZigClient, ZigSide};
use self::caso::{
    CacheExpect, Caso, E2eExpect, RenderExpect, expand_images, load_cases, selected, with_model,
};
use self::diff::{classify_window, divergence, first_divergence, responsible_file};
use self::e2e::E2eResult;
use self::informe::{Informe, build_meta};

/// The cache marks of both sides.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheReport {
    pub history_len_api: u32,
    pub history_len_zig: u32,
    pub shared_api: Vec<u32>,
    pub shared_zig: Vec<u32>,
}

/// One render result.
#[derive(Debug, Clone)]
pub struct RenderOutcome {
    pub caso: Caso,
    pub veredicto: String,
    pub cumple: bool,
    pub divergencia: Option<diff::Divergencia>,
    pub cache: Option<CacheReport>,
    pub nota: String,
}

/// One end-to-end result.
#[derive(Debug, Clone)]
pub struct E2eOutcome {
    pub id: String,
    pub protocol: String,
    /// Whether the case was actually measured.  A case declared `n/a` is not a
    /// pass and not a failure: it has its own counter and never enters the
    /// percentages.
    pub medido: bool,
    pub cumple: bool,
    /// The case's e2e expectation, carried into the report instead of `null`.
    pub esperado: Value,
    pub resultado: E2eResult,
    pub nota: String,
}

/// Build the gateway used by the local side.
fn gateway(tokenizer: &Path, served_name: &str, default_thinking: bool) -> Result<Gateway, String> {
    api::build_gateway(
        tokenizer,
        served_name,
        default_thinking,
        "<｜deepseek_image｜>",
    )
}

fn config(served_name: &str) -> GatewayConfig {
    GatewayConfig {
        max_model_len: 65_536,
        api_sampling: true,
        default_effort: None,
        strict: false,
        require_key: false,
        api_keys: None,
        block_private_urls: false,
        max_images: 600,
        max_image_mb: 64,
        log_level: "info".to_string(),
        served_name: served_name.to_string(),
        engine_logprobs: None,
        dns_overrides: HashMap::new(),
        body_timeout: None,
        max_prompt_bytes: crate::parity::DEFAULT_MAX_PROMPT_BYTES,
        encode_slot: std::sync::Arc::new(tokio::sync::Semaphore::new(1)),
    }
}

/// A quick liveness check: `/v1/models` on the Zig side and `/health` on the
/// gateway.  A connection failure is a "could not measure" (exit 2), not a case
/// failure.
async fn alive(url: &str, path: &str) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };
    match client
        .get(format!("{}{path}", url.trim_end_matches('/')))
        .send()
        .await
    {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// The API-side text around the first divergent token, not the start of the
/// whole prompt.  Six ids before and six after `at`, decoded together and
/// capped at 60 characters without splitting a UTF-8 sequence.
fn api_text_window(tokenizer: &tokenizers::Tokenizer, ids: &[u32], at: usize) -> String {
    let start = at.saturating_sub(6);
    let end = (at + 6).min(ids.len());
    tokenizer
        .decode(&ids[start..end], false)
        .unwrap_or_default()
        .chars()
        .take(60)
        .collect()
}

fn protocol_of(caso: &Caso) -> Protocol {
    match caso.protocol.as_str() {
        "messages" => Protocol::Messages,
        "responses" => Protocol::Responses,
        _ => Protocol::Chat,
    }
}

fn cache_matches(
    expect: &CacheExpect,
    api: u32,
    zig: u32,
    api_shared: &[u32],
    zig_shared: &[u32],
) -> bool {
    match expect {
        CacheExpect::Equivalent => api == zig && api_shared == zig_shared,
        CacheExpect::DifferentByDesign | CacheExpect::Na => true,
    }
}

/// Run the render comparison.  The result is the process exit code (0 all cases
/// meet their expectation, 1 at least one does not) or an error for exit 2.
#[allow(clippy::too_many_arguments)]
pub async fn run_render(
    cases_dir: &Path,
    filters: &[String],
    zig_url: &str,
    gateway_url: Option<&str>,
    tokenizer: &Path,
    served_name: &str,
    default_thinking: bool,
    salida: &Path,
    motor_dir: Option<&Path>,
) -> Result<i32, String> {
    if !tokenizer.is_file() {
        return Err(format!("tokenizer not found: {}", tokenizer.display()));
    }
    if !alive(zig_url, "/v1/models").await {
        return Err(format!(
            "the Zig server at {zig_url} does not answer /v1/models"
        ));
    }
    if let Some(gateway_url) = gateway_url
        && !alive(gateway_url, "/health").await
    {
        return Err(format!(
            "the gateway at {gateway_url} does not answer /health"
        ));
    }
    let gateway = gateway(tokenizer, served_name, default_thinking)?;
    let zig = ZigClient::new(zig_url)?;
    let cases = load_cases(cases_dir).map_err(|errors| errors.join("; "))?;
    let config = config(served_name);
    let mut outcomes = Vec::new();
    for case in cases.iter().filter(|case| selected(case, filters)) {
        let mut body = with_model(&case.body, served_name);
        expand_images(&mut body);
        let protocol = protocol_of(case);
        let local = api::prepare_local(&gateway, &config, protocol, body.clone()).await;
        let mut nota = case.nota.clone().unwrap_or_default();
        let mut cumple;
        let veredicto;
        let mut divergencia = None;
        let mut cache = None;
        match &local {
            LadoApi::Fallback { reason } => {
                veredicto = "fallback".to_string();
                cumple = matches!(case.expect.render, RenderExpect::Simple(ref value) if value == "fallback")
                    || matches!(case.expect.render, RenderExpect::Simple(ref value) if value == "n/a");
                if !cumple {
                    nota = format!("{nota} fallback inesperado: {reason}")
                        .trim()
                        .to_string();
                }
                if let Some(gateway_url) = gateway_url {
                    match api::render_gateway(gateway_url, protocol, body.clone()).await {
                        Ok(value) => {
                            let got = value
                                .get("fallback_reason")
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            if got != reason {
                                cumple = false;
                                nota = format!("{nota} el control del gateway no coincide: {got:?} vs {reason:?}")
                                    .trim()
                                    .to_string();
                            }
                        }
                        Err(error) => {
                            cumple = false;
                            nota = format!("{nota} control del gateway: {error}")
                                .trim()
                                .to_string();
                        }
                    }
                }
            }
            LadoApi::Error { status, message } => {
                veredicto = "error".to_string();
                cumple =
                    matches!(case.expect.render, RenderExpect::Simple(ref value) if value == "n/a");
                nota = format!("{nota} el camino API devolvió {status}: {message}")
                    .trim()
                    .to_string();
            }
            LadoApi::Rendered {
                ids,
                prompt: _,
                history_len,
                shared_prefixes,
                max_tokens,
                thinking,
                effort,
                image_count,
            } => {
                let expected_na =
                    matches!(case.expect.render, RenderExpect::Simple(ref value) if value == "n/a");
                if expected_na {
                    veredicto = "no-medido".to_string();
                    cumple = true;
                    nota = format!(
                        "{nota} render Zig no medido ({}); API: {} ids, history_len {}, max_tokens {:?}, thinking {}, effort {:?}",
                        case.nota.clone().unwrap_or_else(|| "n/a".to_string()),
                        ids.len(),
                        history_len,
                        max_tokens,
                        thinking,
                        effort
                    )
                    .trim()
                    .to_string();
                } else {
                    match api::zig_body(protocol, &body) {
                        ZigSide::NoTranslation(reason) => {
                            veredicto = "no-medido".to_string();
                            cumple = expected_na;
                            nota = format!("{nota} sin traducción fiel al Zig: {reason}")
                                .trim()
                                .to_string();
                        }
                        ZigSide::Body(zig_body) => match zig.tokenize(zig_body).await {
                            Err(error) => {
                                // A refusal about effort is itself the effort divergence;
                                // otherwise it is a measurement error.
                                let expected_class = match &case.expect.render {
                                    RenderExpect::Class { class, note } => {
                                        Some((class.clone(), note.clone()))
                                    }
                                    _ => None,
                                };
                                if let Some((expected_class_name, expected_note)) = expected_class
                                    && (error.to_lowercase().contains("effort")
                                        || error.to_lowercase().contains("reasoning"))
                                {
                                    // The class is already known from the case; the
                                    // responsible file is the effort table, never the
                                    // generic generation header.
                                    let archivo = responsible_file(&expected_class_name);
                                    let api_window =
                                        api_text_window(&gateway.tokenizer, ids, ids.len());
                                    divergencia = Some(divergence(
                                        ids,
                                        &[],
                                        &api_window,
                                        &error,
                                        ids.len(),
                                        &expected_class_name,
                                        archivo,
                                        &format!("{expected_note} Zig: {error}"),
                                    ));
                                    veredicto = "divergente".to_string();
                                    cumple = true;
                                    nota = format!("{nota} el Zig rechazó el caso: {error}")
                                        .trim()
                                        .to_string();
                                } else {
                                    veredicto = "error".to_string();
                                    cumple = false;
                                    nota = format!("{nota} el Zig devolvió: {error}")
                                        .trim()
                                        .to_string();
                                }
                            }
                            Ok(zig_side) => {
                                let cache_report = CacheReport {
                                    history_len_api: *history_len,
                                    history_len_zig: zig_side.history_len,
                                    shared_api: shared_prefixes.clone(),
                                    shared_zig: zig_side.shared_prefixes.clone(),
                                };
                                let cache_ok = cache_matches(
                                    &case.expect.cache,
                                    *history_len,
                                    zig_side.history_len,
                                    shared_prefixes,
                                    &zig_side.shared_prefixes,
                                );
                                if !cache_ok {
                                    nota = format!(
                                        "{nota} marcas de caché distintas: API history_len {} shared {:?}; Zig history_len {} shared {:?}",
                                        history_len, shared_prefixes, zig_side.history_len, zig_side.shared_prefixes
                                    )
                                    .trim()
                                    .to_string();
                                }
                                match first_divergence(ids, &zig_side.ids) {
                                    None => {
                                        veredicto = "identico".to_string();
                                        cumple = matches!(case.expect.render, RenderExpect::Simple(ref value) if value == "identical")
                                            && cache_ok;
                                        if !cumple {
                                            nota = format!("{nota} se esperaba una divergencia de clase conocida").trim().to_string();
                                        }
                                    }
                                    Some(at) => {
                                        let start = at.saturating_sub(6);
                                        let api_window =
                                            api_text_window(&gateway.tokenizer, ids, at);
                                        let zig_end = (at + 6).min(zig_side.token_strs.len());
                                        let zig_window = zig_side
                                            .token_strs
                                            .get(start.min(zig_side.token_strs.len())..zig_end)
                                            .unwrap_or(&[])
                                            .join("");
                                        // `largo` is only when one full side is an exact
                                        // prefix of the other; a near-the-end index by
                                        // itself proves nothing.
                                        let exact_prefix = ids.len() != zig_side.ids.len()
                                            && (ids.starts_with(&zig_side.ids)
                                                || zig_side.ids.starts_with(ids))
                                            && !case
                                                .tags
                                                .iter()
                                                .any(|tag| tag == "prefijo" || tag == "prefix");
                                        let (class, archivo) = if exact_prefix {
                                            ("largo", "model/chat_template.jinja")
                                        } else {
                                            classify_window(
                                                &case.tags,
                                                &api_window,
                                                &zig_window,
                                                *image_count,
                                            )
                                        };
                                        let expected = match &case.expect.render {
                                            RenderExpect::Class {
                                                class: expected,
                                                note,
                                            } => Some((expected.clone(), note.clone())),
                                            _ => None,
                                        };
                                        divergencia = Some(divergence(
                                            ids,
                                            &zig_side.ids,
                                            &api_window,
                                            &zig_window,
                                            at,
                                            class,
                                            archivo,
                                            expected
                                                .as_ref()
                                                .map(|(_, note)| note.clone())
                                                .unwrap_or_default()
                                                .as_str(),
                                        ));
                                        veredicto = "divergente".to_string();
                                        cumple = expected
                                            .as_ref()
                                            .is_some_and(|(expected, _)| expected == class)
                                            && cache_ok;
                                        if !cumple {
                                            nota = format!(
                                                "{nota} clase {:?}, esperado {:?}",
                                                class,
                                                expected.as_ref().map(|(class, _)| class)
                                            )
                                            .trim()
                                            .to_string();
                                        }
                                    }
                                }
                                cache = Some(cache_report);
                            }
                        },
                    }
                }
                if let Some(gateway_url) = gateway_url {
                    match api::render_gateway(gateway_url, protocol, body.clone()).await {
                        Ok(value) => {
                            let gateway_ids = value
                                .get("prompt_token_ids")
                                .and_then(Value::as_array)
                                .map(|items| {
                                    items
                                        .iter()
                                        .filter_map(Value::as_u64)
                                        .map(|id| id as u32)
                                        .collect::<Vec<_>>()
                                });
                            let gateway_history = value
                                .get("history_len")
                                .and_then(Value::as_u64)
                                .map(|value| value as u32);
                            let gateway_shared = value
                                .get("shared_prefixes")
                                .and_then(Value::as_array)
                                .map(|items| {
                                    items
                                        .iter()
                                        .filter_map(Value::as_u64)
                                        .map(|cut| cut as u32)
                                        .collect::<Vec<_>>()
                                });
                            let mismatch = gateway_ids.as_deref() != Some(ids.as_slice())
                                || gateway_history != Some(*history_len)
                                || gateway_shared.as_deref() != Some(shared_prefixes.as_slice());
                            if mismatch {
                                cumple = false;
                                nota = format!("{nota} el control /ds41/render no coincide con la conversión local").trim().to_string();
                            }
                        }
                        Err(error) => {
                            cumple = false;
                            nota = format!("{nota} control /ds41/render: {error}")
                                .trim()
                                .to_string();
                        }
                    }
                }
            }
        }
        outcomes.push(RenderOutcome {
            caso: case.clone(),
            veredicto,
            cumple,
            divergencia,
            cache,
            nota,
        });
    }
    let meta = build_meta(zig_url, gateway_url.unwrap_or("—"), tokenizer, motor_dir).await;
    let report = Informe::from_render(meta, &outcomes, None);
    report.write(salida)?;
    let failed = outcomes.iter().any(|outcome| !outcome.cumple);
    println!(
        "render: {} casos, {} cumplen, {} no cumplen; informe en {}",
        outcomes.len(),
        outcomes.iter().filter(|outcome| outcome.cumple).count(),
        outcomes.iter().filter(|outcome| !outcome.cumple).count(),
        salida.display()
    );
    Ok(if failed { 1 } else { 0 })
}

/// Run the end-to-end checks.  The report is merged with a render report in the
/// same directory when one exists.
#[allow(clippy::too_many_arguments)]
pub async fn run_e2e(
    cases_dir: &Path,
    filters: &[String],
    gateway_url: &str,
    zig_url: &str,
    tokenizer: &Path,
    served_name: &str,
    paralelo: usize,
    salida: &Path,
    motor_dir: Option<&Path>,
) -> Result<i32, String> {
    if !alive(zig_url, "/v1/models").await {
        return Err(format!(
            "the Zig server at {zig_url} does not answer /v1/models"
        ));
    }
    if !alive(gateway_url, "/health").await {
        return Err(format!(
            "the gateway at {gateway_url} does not answer /health"
        ));
    }
    let cases = load_cases(cases_dir).map_err(|errors| errors.join("; "))?;
    let selected_cases: Vec<Caso> = cases
        .into_iter()
        .filter(|case| selected(case, filters))
        .collect();
    let mut outcomes = Vec::new();
    let mut pending = Vec::new();
    for case in selected_cases {
        match &case.expect.e2e {
            E2eExpect::Simple(value) if value == "n/a" => outcomes.push(E2eOutcome {
                id: case.id.clone(),
                protocol: case.protocol.clone(),
                medido: false,
                cumple: false,
                esperado: serde_json::to_value(&case.expect.e2e).unwrap_or(Value::Null),
                resultado: E2eResult::default(),
                nota: case.nota.clone().unwrap_or_else(|| "e2e n/a".to_string()),
            }),
            _ => pending.push(case),
        }
    }
    let parallel = paralelo.max(1);
    for chunk in pending.chunks(parallel) {
        let mut set = tokio::task::JoinSet::new();
        for case in chunk {
            let case = case.clone();
            let gateway_url = gateway_url.to_string();
            let zig_url = zig_url.to_string();
            let served_name = served_name.to_string();
            set.spawn(async move {
                let result = e2e::run_case(&case, &gateway_url, &zig_url, &served_name).await;
                (case, result)
            });
        }
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok((case, result)) => {
                    let mut cumple = result.violaciones.is_empty();
                    let mut nota = case.nota.clone().unwrap_or_default();
                    if let E2eExpect::Object {
                        invariants,
                        same_ids,
                        finish_reason,
                        error,
                    } = &case.expect.e2e
                    {
                        if *invariants && !result.violaciones.is_empty() {
                            cumple = false;
                        }
                        if *same_ids && result.same_ids != Some(true) {
                            cumple = false;
                            nota = format!("{nota} hashes distintos o ausentes")
                                .trim()
                                .to_string();
                        }
                        if let Some(finish) = finish_reason
                            && result.finish_api.as_deref() != Some(finish.as_str())
                        {
                            cumple = false;
                            nota = format!(
                                "{nota} finish_reason {:?}, esperado {finish}",
                                result.finish_api
                            )
                            .trim()
                            .to_string();
                        }
                        if let Some(error) = error
                            && result.status != Some(error.status)
                        {
                            cumple = false;
                            nota = format!(
                                "{nota} HTTP {:?}, esperado {}",
                                result.status, error.status
                            )
                            .trim()
                            .to_string();
                        }
                    }
                    outcomes.push(E2eOutcome {
                        id: case.id.clone(),
                        protocol: case.protocol.clone(),
                        medido: true,
                        cumple,
                        esperado: serde_json::to_value(&case.expect.e2e).unwrap_or(Value::Null),
                        resultado: result,
                        nota,
                    });
                }
                Err(error) => {
                    outcomes.push(E2eOutcome {
                        id: "?".to_string(),
                        protocol: "?".to_string(),
                        medido: true,
                        cumple: false,
                        esperado: Value::Null,
                        resultado: E2eResult {
                            violaciones: vec![format!("task failed: {error}")],
                            ..E2eResult::default()
                        },
                        nota: String::new(),
                    });
                }
            }
        }
    }
    let meta = build_meta(zig_url, gateway_url, tokenizer, motor_dir).await;
    let existing = std::fs::read_to_string(salida.join("informe.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Informe>(&text).ok());
    let report = match existing {
        Some(report) => report.with_e2e(&outcomes),
        None => Informe::from_render(meta, &[], Some(&outcomes)),
    };
    report.write(salida)?;
    let failed = outcomes
        .iter()
        .any(|outcome| outcome.medido && !outcome.cumple);
    println!(
        "e2e: {} casos, {} medidos, {} cumplen, {} no cumplen, {} no medidos; informe en {}",
        outcomes.len(),
        outcomes.iter().filter(|outcome| outcome.medido).count(),
        outcomes
            .iter()
            .filter(|outcome| outcome.medido && outcome.cumple)
            .count(),
        outcomes
            .iter()
            .filter(|outcome| outcome.medido && !outcome.cumple)
            .count(),
        outcomes.iter().filter(|outcome| !outcome.medido).count(),
        salida.display()
    );
    Ok(if failed { 1 } else { 0 })
}

/// Validate the case files without touching either server.
pub fn run_casos(cases_dir: &Path) -> Result<i32, String> {
    match load_cases(cases_dir) {
        Ok(cases) => {
            println!("casos: {} válidos en {}", cases.len(), cases_dir.display());
            Ok(0)
        }
        Err(errors) => {
            for error in errors {
                eprintln!("caso inválido: {error}");
            }
            Ok(1)
        }
    }
}

/// Resolve a path relative to the current directory, without requiring an
/// absolute path in the tests.
pub fn resolve(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

/// The default motor directory, if it exists next to the gateway.
pub fn default_motor_dir() -> Option<PathBuf> {
    for candidate in [Path::new("../tensorfold"), Path::new("tensorfold")] {
        let candidate = resolve(candidate);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// A small helper for the report: the JSON value of a case expectation.
pub fn expected_value(case: &Caso) -> Value {
    serde_json::to_value(&case.expect).unwrap_or(json!({}))
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_api_text_window_is_around_the_divergence_not_the_prompt_start() {
        let tokenizer = crate::test_tokenizer();
        let prompt = format!("{}FINAL", "hola ".repeat(120));
        let ids = tokenizer
            .encode(prompt, false)
            .expect("encode")
            .get_ids()
            .to_vec();
        let window = api_text_window(&tokenizer, &ids, ids.len().saturating_sub(1));
        assert!(window.contains("FINAL"), "window: {window:?}");
        assert!(!window.starts_with("hola hola"), "window: {window:?}");
    }

    #[test]
    fn the_api_text_window_is_capped_at_sixty_unicode_characters() {
        let vocab = [("é漢".repeat(40), 0)].into_iter().collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("build the synthetic tokenizer");
        let tokenizer = tokenizers::Tokenizer::new(model);
        let ids = vec![0; 12];
        let window = api_text_window(&tokenizer, &ids, 6);
        assert_eq!(window.chars().count(), 60, "window: {window:?}");
    }

    #[test]
    fn a_missing_test_tokenizer_explains_ds41_tokenizer() {
        let missing = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".missing-test-tokenizer/tokenizer.json");
        assert!(!missing.exists(), "the missing-path fixture must not exist");
        let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .arg(
                "paridad::tests::the_api_text_window_is_around_the_divergence_not_the_prompt_start",
            )
            .arg("--exact")
            .arg("--nocapture")
            .env("DS41_TOKENIZER", &missing)
            .output()
            .expect("run the window test with a missing DS41_TOKENIZER");
        assert!(
            !output.status.success(),
            "the window test ignored DS41_TOKENIZER"
        );
        let logs = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(logs.contains("set DS41_TOKENIZER"), "output: {logs}");
        assert!(logs.contains("tokenizer.json"), "output: {logs}");
    }
}
