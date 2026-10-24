//! Parity cases: the on-disk format, validation, and the images the comparator
//! generates for the image cases.
//!
//! A case is one request exactly as a client would send it.  The comparator adds
//! the served model name and expands the `ds41-image://` placeholders into PNG data
//! URLs, so no case depends on an external image or an absolute path.

use std::collections::HashSet;
use std::io::Cursor;
use std::path::Path;

use image::{ImageFormat, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The divergence classes of `contracts/comparador.md`.
pub const CLASSES: [&str; 11] = [
    "esfuerzo",
    "herramientas",
    "formato",
    "thinking",
    "prefijo",
    "resultados",
    "sistema",
    "grafias",
    "imagenes",
    "largo",
    "otro",
];

/// What the render side must find.  `identical` and `fallback` are the two
/// simple outcomes; `{"class": ..., "note": ...}` is a predicted divergence;
/// `n/a` means the case cannot be measured (for example, a server without vision).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RenderExpect {
    Simple(String),
    Class {
        class: String,
        #[serde(default)]
        note: String,
    },
}

impl Default for RenderExpect {
    fn default() -> Self {
        Self::Simple("identical".to_string())
    }
}

/// Whether the cache marks must match.  `different-by-design` is allowed for a
/// case that deliberately asks for a different history mark.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CacheExpect {
    #[default]
    Equivalent,
    DifferentByDesign,
    #[serde(rename = "n/a")]
    Na,
}

/// The expected end-to-end shape.  `n/a` skips the case; the object carries the
/// invariants the run must satisfy and, optionally, the exact finish reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum E2eExpect {
    Simple(String),
    Object {
        #[serde(default)]
        invariants: bool,
        #[serde(default)]
        same_ids: bool,
        #[serde(default)]
        finish_reason: Option<String>,
        #[serde(default)]
        error: Option<E2eError>,
    },
}

impl Default for E2eExpect {
    fn default() -> Self {
        Self::Simple("n/a".to_string())
    }
}

/// An expected error: the HTTP status and a substring of the API message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2eError {
    pub status: u16,
    #[serde(default)]
    pub message_contains: String,
}

/// One case file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Caso {
    pub id: String,
    pub protocol: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub body: Value,
    pub expect: Expect,
    #[serde(default = "default_greedy")]
    pub greedy: bool,
    /// Free-form note for the report when the case cannot meet its expectation.
    #[serde(default)]
    pub nota: Option<String>,
}

fn default_greedy() -> bool {
    true
}

/// The `expect` object of a case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expect {
    pub render: RenderExpect,
    #[serde(default)]
    pub cache: CacheExpect,
    #[serde(default)]
    pub e2e: E2eExpect,
}

/// Load and validate every `*.json` case in `dir`.  The error vector is empty on
/// success and contains one line per broken case otherwise.
pub fn load_cases(dir: &Path) -> Result<Vec<Caso>, Vec<String>> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|error| vec![format!("cannot read {}: {error}", dir.display())])?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(vec![format!("no *.json cases in {}", dir.display())]);
    }
    let mut cases = Vec::new();
    let mut errors = Vec::new();
    for path in paths {
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) => {
                errors.push(format!("{}: cannot read: {error}", path.display()));
                continue;
            }
        };
        let case: Caso = match serde_json::from_str(&text) {
            Ok(case) => case,
            Err(error) => {
                errors.push(format!("{}: invalid JSON: {error}", path.display()));
                continue;
            }
        };
        let mut found = validate(&case);
        for error in &found {
            errors.push(format!("{}: {error}", path.display()));
        }
        if found.is_empty() {
            cases.push(case);
        } else {
            found.clear();
        }
    }
    let mut ids = HashSet::new();
    for case in &cases {
        if !ids.insert(case.id.clone()) {
            errors.push(format!("duplicate case id: {}", case.id));
        }
    }
    if errors.is_empty() {
        Ok(cases)
    } else {
        Err(errors)
    }
}

/// Validate one case; return one message per problem.
pub fn validate(case: &Caso) -> Vec<String> {
    let mut errors = Vec::new();
    if case.id.trim().is_empty() {
        errors.push("id is empty".to_string());
    }
    if !matches!(case.protocol.as_str(), "chat" | "messages" | "responses") {
        errors.push(format!(
            "protocol {} is not chat, messages or responses",
            case.protocol
        ));
    }
    if !case.body.is_object() {
        errors.push("body must be a JSON object".to_string());
    }
    if let Some(model) = case.body.get("model")
        && !model.is_string()
    {
        errors.push("body.model must be a string when present".to_string());
    }
    match &case.expect.render {
        RenderExpect::Simple(value) => {
            if !matches!(value.as_str(), "identical" | "fallback" | "n/a") {
                errors.push(format!(
                    "expect.render {value} is not identical, fallback or n/a"
                ));
            }
        }
        RenderExpect::Class { class, .. } => {
            if !CLASSES.contains(&class.as_str()) {
                errors.push(format!("expect.render class {class} is unknown"));
            }
        }
    }
    match &case.expect.e2e {
        E2eExpect::Simple(value) => {
            if value != "n/a" {
                errors.push(format!("expect.e2e {value} is not n/a"));
            }
        }
        E2eExpect::Object {
            finish_reason,
            error,
            ..
        } => {
            if let Some(finish) = finish_reason
                && !matches!(
                    finish.as_str(),
                    "stop"
                        | "length"
                        | "tool_calls"
                        | "end_turn"
                        | "max_tokens"
                        | "stop_sequence"
                        | "tool_use"
                        | "completed"
                        | "incomplete"
                )
            {
                errors.push(format!("expect.e2e.finish_reason {finish} is unknown"));
            }
            if let Some(error) = error
                && !(100..=599).contains(&error.status)
            {
                errors.push(format!(
                    "expect.e2e.error.status {} is not an HTTP status",
                    error.status
                ));
            }
        }
    }
    if !case.tags.iter().all(|tag| !tag.trim().is_empty()) {
        errors.push("tags must not contain empty strings".to_string());
    }
    errors
}

/// Apply the `--filtro` values: a case matches when any filter is a substring of
/// its id or one of its tags.  No filters means every case.
pub fn selected(case: &Caso, filters: &[String]) -> bool {
    filters.is_empty()
        || filters.iter().any(|filter| {
            case.id.contains(filter.as_str())
                || case.tags.iter().any(|tag| tag.contains(filter.as_str()))
        })
}

/// Replace every `ds41-image://NAME` string with a generated PNG data URL.  The
/// generated image is a solid 16x16 PNG, the same shape Sufyan's fixtures use.
pub fn expand_images(value: &mut Value) {
    match value {
        Value::String(text) => {
            if let Some(name) = text.strip_prefix("ds41-image://")
                && let Some(url) = generated_image(name)
            {
                *text = url;
            } else if let Some(name) = text.strip_prefix("ds41-b64://")
                && let Some(data) = generated_image_base64(name)
            {
                *text = data;
            }
        }
        Value::Array(items) => {
            for item in items {
                expand_images(item);
            }
        }
        Value::Object(object) => {
            for (_, item) in object.iter_mut() {
                expand_images(item);
            }
        }
        _ => {}
    }
}

/// A solid 16x16 PNG as a data URL, or `None` for an unknown name.
pub fn generated_image(name: &str) -> Option<String> {
    Some(format!(
        "data:image/png;base64,{}",
        generated_image_base64(name)?
    ))
}

/// The bare base64 of the same generated PNG, for Messages' `source.data`.
pub fn generated_image_base64(name: &str) -> Option<String> {
    let color = match name {
        "red16" => Rgba([220, 32, 32, 255]),
        "blue16" => Rgba([32, 64, 220, 255]),
        "green16" => Rgba([32, 180, 64, 255]),
        _ => return None,
    };
    let image = RgbaImage::from_pixel(16, 16, color);
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .ok()?;
    Some(base64(&bytes))
}

/// Standard base64 (no line breaks), enough for the small PNGs above.
fn base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let a = chunk[0] as u32;
        let b = chunk.get(1).copied().unwrap_or(0) as u32;
        let c = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (a << 16) | (b << 8) | c;
        out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((n >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(n & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// The served model name is inserted into every body unless the case already names
/// one.  The comparator never sends a body without a model to the gateway.
pub fn with_model(body: &Value, served_name: &str) -> Value {
    let mut object = body.as_object().cloned().unwrap_or_else(Map::new);
    object
        .entry("model".to_string())
        .or_insert_with(|| Value::String(served_name.to_string()));
    Value::Object(object)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(id: &str) -> Caso {
        Caso {
            id: id.to_string(),
            protocol: "chat".to_string(),
            tags: vec!["basico".to_string()],
            body: serde_json::json!({"messages": [{"role": "user", "content": "Hola"}]}),
            expect: Expect {
                render: RenderExpect::Simple("identical".to_string()),
                cache: CacheExpect::Equivalent,
                e2e: E2eExpect::Simple("n/a".to_string()),
            },
            greedy: true,
            nota: None,
        }
    }

    #[test]
    fn a_valid_case_passes_validation() {
        assert!(validate(&case("chat-basico")).is_empty());
    }

    #[test]
    fn a_broken_case_reports_the_reason() {
        let mut broken = case("chat-basico");
        broken.expect.render = RenderExpect::Simple("igual".to_string());
        let errors = validate(&broken);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("expect.render igual"));
    }

    #[test]
    fn a_generated_png_is_a_png_data_url() {
        let url = generated_image("red16").expect("red image");
        assert!(url.starts_with("data:image/png;base64,"));
        let raw = url.trim_start_matches("data:image/png;base64,");
        assert!(raw.starts_with("iVBORw0KGgo"));
    }

    #[test]
    fn images_are_expanded_recursively() {
        let mut body = serde_json::json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "mira"},
                {"type": "image_url", "image_url": {"url": "ds41-image://blue16"}}
            ]}]
        });
        expand_images(&mut body);
        let url = body["messages"][0]["content"][1]["image_url"]["url"]
            .as_str()
            .unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
    }
}
