//! The JSON and self-contained HTML report.
//!
//! The JSON is the machine-readable record; the HTML is the same data with inline
//! CSS, summary cards, one row per case and a `<details>` block with the two
//! windows around a divergence.  No external JavaScript or stylesheet is used.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::paridad::diff::Divergencia;
use crate::paridad::e2e::E2eResult;
use crate::paridad::{CacheReport, E2eOutcome, RenderOutcome};

/// The complete report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Informe {
    pub meta: Meta,
    pub resumen: Resumen,
    pub casos: Vec<CasoInforme>,
    pub que_corregir: Vec<QueCorregir>,
}

/// Provenance of the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub fecha: String,
    pub zig: String,
    pub gateway: String,
    pub motor_commit: String,
    pub motor_health: Value,
    pub motor_models: Value,
    pub recipe: String,
    pub tokenizer_sha: String,
    pub comparador: String,
}

/// Counters by mode and protocol.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Resumen {
    pub render: RenderSummary,
    pub e2e: E2eSummary,
    pub por_protocolo: BTreeMap<String, ProtocolSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RenderSummary {
    pub casos: usize,
    pub identicos: usize,
    pub divergentes: BTreeMap<String, usize>,
    pub fallbacks: usize,
    pub errores: usize,
    pub no_medidos: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct E2eSummary {
    pub casos: usize,
    pub medidos: usize,
    pub invariantes_ok: usize,
    pub hashes_iguales: usize,
    pub hashes_distintos: usize,
    pub errores: usize,
    pub no_medidos: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProtocolSummary {
    pub casos: usize,
    pub identicos: usize,
    pub divergentes: usize,
    pub fallbacks: usize,
    pub errores: usize,
    pub no_medidos: usize,
}

/// One row of the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CasoInforme {
    pub id: String,
    pub modo: String,
    pub protocolo: String,
    pub veredicto: String,
    pub esperado: Value,
    pub cumple_esperado: bool,
    pub divergencia: Option<Divergencia>,
    pub cache: Option<CacheReport>,
    pub e2e: Option<E2eResult>,
    pub nota: String,
}

/// One group of the "what to fix in the Zig" section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueCorregir {
    pub clase: String,
    pub casos: usize,
    pub archivo: String,
    pub ejemplo: String,
}

impl Informe {
    /// Build a render-only report.
    pub fn from_render(meta: Meta, outcomes: &[RenderOutcome], e2e: Option<&[E2eOutcome]>) -> Self {
        let mut casos = Vec::new();
        let mut resumen = Resumen::default();
        let mut que_corregir: BTreeMap<String, QueCorregir> = BTreeMap::new();
        for outcome in outcomes {
            let protocol = outcome.caso.protocol.clone();
            let entry = resumen.por_protocolo.entry(protocol).or_default();
            entry.casos += 1;
            resumen.render.casos += 1;
            match outcome.veredicto.as_str() {
                "identico" => {
                    resumen.render.identicos += 1;
                    entry.identicos += 1;
                }
                "divergente" => {
                    let class = outcome
                        .divergencia
                        .as_ref()
                        .map(|divergencia| divergencia.clase.clone())
                        .unwrap_or_else(|| "otro".to_string());
                    *resumen.render.divergentes.entry(class.clone()).or_insert(0) += 1;
                    entry.divergentes += 1;
                    let item = que_corregir
                        .entry(class.clone())
                        .or_insert_with(|| QueCorregir {
                            clase: class,
                            casos: 0,
                            archivo: outcome
                                .divergencia
                                .as_ref()
                                .map(|divergencia| divergencia.archivo.clone())
                                .unwrap_or_else(|| "—".to_string()),
                            ejemplo: outcome
                                .divergencia
                                .as_ref()
                                .map(|divergencia| {
                                    format!(
                                        "{}: {:?} vs {:?}",
                                        outcome.caso.id,
                                        divergencia.texto_api,
                                        divergencia.texto_zig
                                    )
                                })
                                .unwrap_or_default(),
                        });
                    item.casos += 1;
                }
                "fallback" => {
                    resumen.render.fallbacks += 1;
                    entry.fallbacks += 1;
                }
                "no-medido" => {
                    resumen.render.no_medidos += 1;
                    entry.no_medidos += 1;
                }
                _ => {
                    resumen.render.errores += 1;
                    entry.errores += 1;
                }
            }
            casos.push(CasoInforme {
                id: outcome.caso.id.clone(),
                modo: "render".to_string(),
                protocolo: outcome.caso.protocol.clone(),
                veredicto: outcome.veredicto.clone(),
                esperado: serde_json::to_value(&outcome.caso.expect).unwrap_or(Value::Null),
                cumple_esperado: outcome.cumple,
                divergencia: outcome.divergencia.clone(),
                cache: outcome.cache.clone(),
                e2e: None,
                nota: outcome.nota.clone(),
            });
        }
        // The effort cases are always listed, even when the recipe and the Zig
        // template happen to agree: the report then says so explicitly.
        let effort: Vec<&RenderOutcome> = outcomes
            .iter()
            .filter(|outcome| outcome.caso.tags.iter().any(|tag| tag == "effort"))
            .collect();
        if !effort.is_empty() {
            let entry = que_corregir
                .entry("esfuerzo".to_string())
                .or_insert_with(|| QueCorregir {
                    clase: "esfuerzo".to_string(),
                    casos: 0,
                    archivo: "families/dsv41/text.zig (tabla), model/chat_template.jinja"
                        .to_string(),
                    ejemplo: String::new(),
                });
            entry.casos = effort.len();
            entry.ejemplo = effort
                .iter()
                .map(|outcome| format!("{}: {}", outcome.caso.id, outcome.veredicto))
                .collect::<Vec<_>>()
                .join("; ");
        }
        if let Some(e2e) = e2e {
            for outcome in e2e {
                resumen.e2e.casos += 1;
                if !outcome.medido {
                    resumen.e2e.no_medidos += 1;
                } else {
                    resumen.e2e.medidos += 1;
                    if outcome.cumple {
                        resumen.e2e.invariantes_ok += 1;
                    }
                    match outcome.resultado.same_ids {
                        Some(true) => resumen.e2e.hashes_iguales += 1,
                        Some(false) => resumen.e2e.hashes_distintos += 1,
                        None => {}
                    }
                    if outcome.resultado.status.is_some_and(|status| status >= 400) {
                        resumen.e2e.errores += 1;
                    }
                }
                casos.push(CasoInforme {
                    id: outcome.id.clone(),
                    modo: "e2e".to_string(),
                    protocolo: outcome.protocol.clone(),
                    veredicto: if !outcome.medido {
                        "no-medido".to_string()
                    } else if outcome.cumple {
                        "cumple".to_string()
                    } else {
                        "no-cumple".to_string()
                    },
                    esperado: outcome.esperado.clone(),
                    cumple_esperado: outcome.cumple,
                    divergencia: None,
                    cache: None,
                    e2e: Some(outcome.resultado.clone()),
                    nota: outcome.nota.clone(),
                });
            }
        }
        Self {
            meta,
            resumen,
            casos,
            que_corregir: que_corregir.into_values().collect(),
        }
    }

    /// Merge e2e outcomes into a report previously written by `render`.
    pub fn with_e2e(mut self, outcomes: &[E2eOutcome]) -> Self {
        self.casos.retain(|case| case.modo != "e2e");
        self.resumen.e2e = E2eSummary::default();
        for outcome in outcomes {
            self.resumen.e2e.casos += 1;
            if !outcome.medido {
                self.resumen.e2e.no_medidos += 1;
            } else {
                self.resumen.e2e.medidos += 1;
                if outcome.cumple {
                    self.resumen.e2e.invariantes_ok += 1;
                }
                match outcome.resultado.same_ids {
                    Some(true) => self.resumen.e2e.hashes_iguales += 1,
                    Some(false) => self.resumen.e2e.hashes_distintos += 1,
                    None => {}
                }
                if outcome.resultado.status.is_some_and(|status| status >= 400) {
                    self.resumen.e2e.errores += 1;
                }
            }
            self.casos.push(CasoInforme {
                id: outcome.id.clone(),
                modo: "e2e".to_string(),
                protocolo: outcome.protocol.clone(),
                veredicto: if !outcome.medido {
                    "no-medido".to_string()
                } else if outcome.cumple {
                    "cumple".to_string()
                } else {
                    "no-cumple".to_string()
                },
                esperado: outcome.esperado.clone(),
                cumple_esperado: outcome.cumple,
                divergencia: None,
                cache: None,
                e2e: Some(outcome.resultado.clone()),
                nota: outcome.nota.clone(),
            });
        }
        self
    }

    /// Write `informe.json` and `informe.html` under `dir`.
    pub fn write(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir)
            .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
        let json_text = serde_json::to_string_pretty(self)
            .map_err(|error| format!("cannot serialize report: {error}"))?;
        std::fs::write(dir.join("informe.json"), json_text)
            .map_err(|error| format!("cannot write informe.json: {error}"))?;
        std::fs::write(dir.join("informe.html"), self.html())
            .map_err(|error| format!("cannot write informe.html: {error}"))?;
        Ok(())
    }

    /// The self-contained HTML report.
    pub fn html(&self) -> String {
        let mut out = String::new();
        out.push_str("<!DOCTYPE html>\n<html lang=\"es\">\n<head>\n<meta charset=\"utf-8\"/>\n");
        out.push_str("<title>Paridad ds41</title>\n<style>\n");
        out.push_str(
            "body{font-family:system-ui,sans-serif;margin:2rem;color:#222;background:#fafafa}",
        );
        out.push_str("h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:2rem}");
        out.push_str(".cards{display:flex;gap:1rem;flex-wrap:wrap}.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1rem;min-width:10rem}");
        out.push_str("table{border-collapse:collapse;width:100%;background:#fff}th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:left;vertical-align:top;font-size:.9rem}");
        out.push_str(
            "th{background:#eee}.ok{color:#1a7f37}.bad{color:#b42318}.warn{color:#9a6700}",
        );
        out.push_str("details{margin:.25rem 0}pre{white-space:pre-wrap;font-size:.8rem;background:#f4f4f4;padding:.5rem}");
        out.push_str("</style>\n</head>\n<body>\n");
        out.push_str(&format!(
            "<h1>Paridad ds41 — {}</h1>\n",
            escape(&self.meta.fecha)
        ));
        out.push_str("<section class=\"cards\">\n");
        out.push_str(&card("Casos render", self.resumen.render.casos));
        out.push_str(&card("Idénticos", self.resumen.render.identicos));
        out.push_str(&card(
            "Divergentes",
            self.resumen.render.divergentes.values().sum::<usize>(),
        ));
        out.push_str(&card("Fallbacks", self.resumen.render.fallbacks));
        out.push_str(&card("Errores", self.resumen.render.errores));
        out.push_str(&card("No medidos", self.resumen.render.no_medidos));
        out.push_str(&card("E2E casos", self.resumen.e2e.casos));
        out.push_str(&card("E2E medidos", self.resumen.e2e.medidos));
        out.push_str(&card("E2E invariantes", self.resumen.e2e.invariantes_ok));
        out.push_str(&card("E2E no medidos", self.resumen.e2e.no_medidos));
        out.push_str(&card("Hashes iguales", self.resumen.e2e.hashes_iguales));
        out.push_str(&card("Hashes distintos", self.resumen.e2e.hashes_distintos));
        out.push_str("</section>\n");
        out.push_str("<h2>Meta</h2>\n<table>\n");
        for (key, value) in [
            ("fecha", self.meta.fecha.clone()),
            ("zig", self.meta.zig.clone()),
            ("gateway", self.meta.gateway.clone()),
            ("commit del motor", self.meta.motor_commit.clone()),
            ("deepseek-recipe", self.meta.recipe.clone()),
            ("sha256 del tokenizador", self.meta.tokenizer_sha.clone()),
            ("comparador", self.meta.comparador.clone()),
        ] {
            out.push_str(&format!(
                "<tr><th>{}</th><td>{}</td></tr>\n",
                escape(key),
                escape(&value)
            ));
        }
        out.push_str("</table>\n");
        out.push_str("<h2>Casos</h2>\n<table>\n<tr><th>id</th><th>modo</th><th>protocolo</th><th>veredicto</th><th>esperado</th><th>clase</th><th>posición</th><th>archivo</th></tr>\n");
        for case in &self.casos {
            let (class, position, file) = match &case.divergencia {
                Some(divergencia) => (
                    divergencia.clase.clone(),
                    divergencia
                        .posicion
                        .map(|value| value.to_string())
                        .unwrap_or_default(),
                    divergencia.archivo.clone(),
                ),
                None => (String::new(), String::new(), String::new()),
            };
            let verdict_class = if case.veredicto == "no-medido" {
                "warn"
            } else if case.cumple_esperado {
                "ok"
            } else {
                "bad"
            };
            out.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td class=\"{}\">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
                escape(&case.id),
                escape(&case.modo),
                escape(&case.protocolo),
                verdict_class,
                escape(&case.veredicto),
                escape(&case.esperado.to_string()),
                escape(&class),
                escape(&position),
                escape(&file),
            ));
            if let Some(divergencia) = &case.divergencia {
                out.push_str(&format!(
                    "<tr><td colspan=\"8\"><details><summary>Divergencia {}: {:?} vs {:?}</summary><pre>ids API: {:?}\nids Zig: {:?}\ntexto API: {:?}\ntexto Zig: {:?}\nnota: {}</pre></details></td></tr>\n",
                    escape(&case.id),
                    escape(&divergencia.texto_api),
                    escape(&divergencia.texto_zig),
                    divergencia.ids_api,
                    divergencia.ids_zig,
                    escape(&divergencia.texto_api),
                    escape(&divergencia.texto_zig),
                    escape(&divergencia.nota),
                ));
            }
            if let Some(e2e) = &case.e2e {
                out.push_str(&format!(
                    "<tr><td colspan=\"8\"><details><summary>E2E {}: finish {:?} / {:?}, sha {:?} / {:?}</summary><pre>violaciones:\n{}</pre></details></td></tr>\n",
                    escape(&case.id),
                    escape(&format!("{:?}", e2e.finish_api)),
                    escape(&format!("{:?}", e2e.finish_zig)),
                    escape(&format!("{:?}", e2e.sha_api)),
                    escape(&format!("{:?}", e2e.sha_zig)),
                    escape(&e2e.violaciones.join("\n")),
                ));
            }
            if !case.nota.is_empty() {
                out.push_str(&format!(
                    "<tr><td colspan=\"8\"><em>{}</em></td></tr>\n",
                    escape(&case.nota)
                ));
            }
        }
        out.push_str("</table>\n");
        out.push_str("<h2>Qué corregir en el Zig</h2>\n");
        for item in &self.que_corregir {
            out.push_str(&format!(
                "<section><h3>{}</h3><p>{} caso(s) · {}</p><pre>{}</pre></section>\n",
                escape(&item.clase),
                item.casos,
                escape(&item.archivo),
                escape(&item.ejemplo)
            ));
        }
        out.push_str("</body>\n</html>\n");
        out
    }
}

fn card(label: &str, value: usize) -> String {
    format!(
        "<div class=\"card\"><strong>{}</strong><br/>{}</div>\n",
        escape(label),
        value
    )
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Build the report metadata from the local environment and the two servers.
pub async fn build_meta(
    zig: &str,
    gateway: &str,
    tokenizer: &Path,
    motor_dir: Option<&Path>,
) -> Meta {
    let tokenizer_sha = std::fs::read(tokenizer)
        .map(|bytes| {
            let digest = Sha256::digest(&bytes);
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        })
        .unwrap_or_else(|_| "no disponible".to_string());
    let motor_commit = motor_dir
        .and_then(|dir| {
            std::process::Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        })
        .unwrap_or_else(|| "desconocido".to_string());
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok();
    let motor_health = match &client {
        Some(client) => match client.get(format!("{zig}/health")).send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                match response.json::<Value>().await {
                    Ok(body) => json!({"status": status, "body": body}),
                    Err(_) => Value::Null,
                }
            }
            Err(_) => Value::Null,
        },
        None => Value::Null,
    };
    let motor_models = match &client {
        Some(client) => match client.get(format!("{zig}/v1/models")).send().await {
            Ok(response) => response.json::<Value>().await.unwrap_or(Value::Null),
            Err(_) => Value::Null,
        },
        None => Value::Null,
    };
    Meta {
        fecha: utc_now(),
        zig: zig.to_string(),
        gateway: gateway.to_string(),
        motor_commit,
        motor_health,
        motor_models,
        recipe: "deepseek-recipe 0.2.0".to_string(),
        tokenizer_sha,
        comparador: format!("ds41-paridad {}", env!("CARGO_PKG_VERSION")),
    }
}

/// A UTC timestamp without an external date crate.
fn utc_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let days = seconds / 86_400;
    let day_seconds = seconds % 86_400;
    let (year, month, day) = civil_from_days(days as i64);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        day_seconds / 3600,
        (day_seconds % 3600) / 60,
        day_seconds % 60
    )
}

/// Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    (y + i64::from(m <= 2), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quick_xml::Reader;
    use quick_xml::events::Event;

    fn sample_report() -> Informe {
        Informe {
            meta: Meta {
                fecha: "2026-01-01T00:00:00Z".to_string(),
                zig: "http://127.0.0.1:1".to_string(),
                gateway: "http://127.0.0.1:2".to_string(),
                motor_commit: "abc".to_string(),
                motor_health: json!({"status": 200}),
                motor_models: json!({"data": []}),
                recipe: "deepseek-recipe 0.2.0".to_string(),
                tokenizer_sha: "00".to_string(),
                comparador: "ds41-paridad 0.1.0".to_string(),
            },
            resumen: Resumen {
                render: RenderSummary {
                    casos: 2,
                    identicos: 1,
                    divergentes: BTreeMap::from([("esfuerzo".to_string(), 1)]),
                    fallbacks: 0,
                    errores: 0,
                    no_medidos: 0,
                },
                e2e: E2eSummary {
                    casos: 1,
                    medidos: 1,
                    invariantes_ok: 1,
                    hashes_iguales: 1,
                    hashes_distintos: 0,
                    errores: 0,
                    no_medidos: 0,
                },
                por_protocolo: BTreeMap::new(),
            },
            casos: vec![CasoInforme {
                id: "chat-think-medium".to_string(),
                modo: "render".to_string(),
                protocolo: "chat".to_string(),
                veredicto: "divergente".to_string(),
                esperado: json!({"class": "esfuerzo", "note": "75 vs 50"}),
                cumple_esperado: true,
                divergencia: Some(Divergencia {
                    posicion: Some(3),
                    ids_api: vec![1, 2, 3],
                    ids_zig: vec![1, 2, 4],
                    texto_api: "Reasoning Effort: 75".to_string(),
                    texto_zig: "Reasoning Effort: 50".to_string(),
                    clase: "esfuerzo".to_string(),
                    archivo: "families/dsv41/text.zig".to_string(),
                    nota: "nota".to_string(),
                }),
                cache: None,
                e2e: None,
                nota: String::new(),
            }],
            que_corregir: vec![QueCorregir {
                clase: "esfuerzo".to_string(),
                casos: 1,
                archivo: "families/dsv41/text.zig".to_string(),
                ejemplo: "75 vs 50".to_string(),
            }],
        }
    }

    #[test]
    fn the_html_is_well_formed_xml() {
        let html = sample_report().html();
        let fragment = html
            .split_once("<body>")
            .and_then(|(_, rest)| rest.split_once("</body>"))
            .map(|(body, _)| body)
            .expect("body");
        let mut reader = Reader::from_str(fragment);
        reader.config_mut().check_end_names = true;
        loop {
            match reader.read_event() {
                Ok(Event::Eof) => break,
                Ok(_) => {}
                Err(error) => panic!("malformed HTML fragment: {error}"),
            }
        }
    }

    #[test]
    fn the_summary_counts_a_divergence_and_an_e2e_hash() {
        let report = sample_report();
        assert_eq!(report.resumen.render.divergentes.get("esfuerzo"), Some(&1));
        assert_eq!(report.resumen.e2e.hashes_iguales, 1);
    }
    #[test]
    fn the_e2e_row_carries_the_expectation() {
        let meta = sample_report().meta;
        let outcome = E2eOutcome {
            id: "chat-con-expectativa".to_string(),
            protocol: "chat".to_string(),
            medido: true,
            cumple: true,
            esperado: json!({"invariants": true, "same_ids": true, "finish_reason": "stop"}),
            resultado: E2eResult::default(),
            nota: String::new(),
        };
        let report = Informe::from_render(meta, &[], Some(&[outcome]));
        assert_eq!(
            report.casos[0].esperado,
            json!({"invariants": true, "same_ids": true, "finish_reason": "stop"})
        );
    }

    #[test]
    fn an_unmeasured_e2e_case_has_its_own_counter_and_is_not_a_pass() {
        let meta = sample_report().meta;
        let outcomes = vec![
            E2eOutcome {
                id: "chat-sin-medir".to_string(),
                protocol: "chat".to_string(),
                medido: false,
                cumple: false,
                esperado: json!("n/a"),
                resultado: E2eResult::default(),
                nota: "e2e n/a".to_string(),
            },
            E2eOutcome {
                id: "chat-medido".to_string(),
                protocol: "chat".to_string(),
                medido: true,
                cumple: true,
                esperado: json!({"invariants": true}),
                resultado: E2eResult::default(),
                nota: String::new(),
            },
        ];
        let report = Informe::from_render(meta, &[], Some(&outcomes));
        assert_eq!(report.resumen.e2e.casos, 2);
        assert_eq!(report.resumen.e2e.medidos, 1);
        assert_eq!(report.resumen.e2e.no_medidos, 1);
        assert_eq!(report.resumen.e2e.invariantes_ok, 1);
        assert_eq!(report.casos[0].veredicto, "no-medido");
        assert!(!report.casos[0].cumple_esperado);
        assert_eq!(report.casos[1].veredicto, "cumple");
        assert_eq!(report.casos[0].esperado, json!("n/a"));
        assert_eq!(report.casos[1].esperado, json!({"invariants": true}));
    }
}
