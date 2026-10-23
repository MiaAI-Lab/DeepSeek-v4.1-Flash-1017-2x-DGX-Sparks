//! Token-level divergence: first index, the two windows around it, the class of
//! the difference and the Zig file responsible.
//!
//! The classifier follows the table in `contracts/comparador.md`.  It only reads
//! the two rendered prompts and the case tags; it never guesses from the ids.

use serde::{Deserialize, Serialize};

/// One divergence as it appears in the report.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Divergencia {
    pub posicion: Option<usize>,
    pub ids_api: Vec<u32>,
    pub ids_zig: Vec<u32>,
    pub texto_api: String,
    pub texto_zig: String,
    pub clase: String,
    pub archivo: String,
    #[serde(default)]
    pub nota: String,
}

/// The first index at which two id sequences differ, or the length of the shorter
/// one when one is a prefix of the other.
pub fn first_divergence(left: &[u32], right: &[u32]) -> Option<usize> {
    let common = left.len().min(right.len());
    for index in 0..common {
        if left[index] != right[index] {
            return Some(index);
        }
    }
    if left.len() == right.len() {
        None
    } else {
        Some(common)
    }
}

/// Six ids from each side around `at`, and up to 60 characters of decoded text
/// around the first character difference.
#[allow(clippy::too_many_arguments)]
pub fn divergence(
    left: &[u32],
    right: &[u32],
    left_text: &str,
    right_text: &str,
    at: usize,
    class: &str,
    archivo: &str,
    nota: &str,
) -> Divergencia {
    let start = at.saturating_sub(6);
    let left_ids = left
        .get(start..(at + 6).min(left.len()))
        .unwrap_or(&[])
        .to_vec();
    let right_ids = right
        .get(start..(at + 6).min(right.len()))
        .unwrap_or(&[])
        .to_vec();
    let (texto_api, texto_zig) = text_windows(left_text, right_text);
    Divergencia {
        posicion: Some(at),
        ids_api: left_ids,
        ids_zig: right_ids,
        texto_api,
        texto_zig,
        clase: class.to_string(),
        archivo: archivo.to_string(),
        nota: nota.to_string(),
    }
}

/// A window of at most 60 characters around the first character difference.
pub fn text_windows(left: &str, right: &str) -> (String, String) {
    let left_chars: Vec<char> = left.chars().collect();
    let right_chars: Vec<char> = right.chars().collect();
    let common = left_chars.len().min(right_chars.len());
    let mut at = 0;
    while at < common && left_chars[at] == right_chars[at] {
        at += 1;
    }
    let start = at.saturating_sub(30);
    let end_left = (at + 30).min(left_chars.len());
    let end_right = (at + 30).min(right_chars.len());
    let api: String = left_chars[start..end_left].iter().collect();
    let zig: String = right_chars[start..end_right].iter().collect();
    (api, zig)
}

/// The class of a divergence and the Zig file(s) responsible.  The windows are
/// the text around the first difference; the suffix after it decides the marker,
/// so an extra newline before `## Tools` is a tools divergence even when the same
/// window still shows the effort line before it.
pub fn classify_window(
    tags: &[String],
    left_window: &str,
    right_window: &str,
    image_count: usize,
) -> (&'static str, &'static str) {
    let has = |tag: &str| tags.iter().any(|value| value == tag);
    if has("grafias") {
        return ("grafias", "server/hf_text.zig");
    }
    if image_count > 0 || has("imagenes") {
        return ("imagenes", "server/prompt.zig, vision.prepare");
    }
    if has("prefijo") || has("prefix") {
        return ("prefijo", "server/messages.zig");
    }
    let left_chars: Vec<char> = left_window.chars().collect();
    let right_chars: Vec<char> = right_window.chars().collect();
    let common = left_chars.len().min(right_chars.len());
    let mut at = 0;
    while at < common && left_chars[at] == right_chars[at] {
        at += 1;
    }
    let suffix = format!(
        "{}\n{}",
        left_chars[at..].iter().collect::<String>(),
        right_chars[at..].iter().collect::<String>()
    );
    let prefix = format!(
        "{}\n{}",
        left_chars[..at].iter().collect::<String>(),
        right_chars[..at].iter().collect::<String>()
    );
    let marker = |text: &str, needle: &str| text.contains(needle);
    if marker(&suffix, "## Tools")
        || marker(&suffix, "### Available Tool Schemas")
        || marker(&suffix, "\"parameters\":")
        || marker(&prefix, "## Tools")
    {
        return (
            "herramientas",
            "model/chat_template.jinja, server/tool_specs.zig",
        );
    }
    if marker(&suffix, "## Response Format") || marker(&prefix, "## Response Format") {
        return ("formato", "model/chat_template.jinja, server/grammar.zig");
    }
    if marker(&suffix, "Reasoning Effort:") || marker(&prefix, "Reasoning Effort:") {
        return (
            "esfuerzo",
            "families/dsv41/text.zig (tabla), model/chat_template.jinja",
        );
    }
    if marker(&suffix, "thinking")
        || marker(&suffix, "response")
        || marker(&prefix, "thinking")
        || marker(&prefix, "response")
    {
        return ("thinking", "server/prompt.zig, server/fields.zig");
    }
    if marker(&suffix, "<tool_result>") || marker(&prefix, "<tool_result>") {
        return (
            "resultados",
            "model/chat_template.jinja, server/messages.zig",
        );
    }
    if has("sistema")
        || has("system")
        || marker(&suffix, "<｜end_of_query｜>")
        || marker(&suffix, "<｜User｜>")
    {
        return ("sistema", "model/chat_template.jinja, server/messages.zig");
    }
    ("otro", "—")
}

/// The Zig file(s) responsible for a class, straight from the contract table.
/// Used when the class is already known (for example, a rejected effort).
pub fn responsible_file(class: &str) -> &'static str {
    match class {
        "esfuerzo" => "families/dsv41/text.zig (tabla), model/chat_template.jinja",
        "herramientas" => "model/chat_template.jinja, server/tool_specs.zig",
        "formato" => "model/chat_template.jinja, server/grammar.zig",
        "thinking" => "server/prompt.zig, server/fields.zig",
        "prefijo" => "server/messages.zig",
        "resultados" => "model/chat_template.jinja, server/messages.zig",
        "sistema" => "model/chat_template.jinja, server/messages.zig",
        "grafias" => "server/hf_text.zig",
        "imagenes" => "server/prompt.zig, vision.prepare",
        "largo" => "model/chat_template.jinja",
        _ => "—",
    }
}

/// Classify two full prompts; kept for the synthetic tests and callers that do not
/// have the divergence window.
pub fn classify(
    tags: &[String],
    left_prompt: &str,
    right_prompt: &str,
    image_count: usize,
) -> (&'static str, &'static str) {
    // `largo` is only the exact-prefix case of the two full sides; a shared
    // window prefix is not enough (the window may simply be cut).
    if left_prompt.starts_with(right_prompt) || right_prompt.starts_with(left_prompt) {
        return ("largo", "model/chat_template.jinja");
    }
    classify_window(tags, left_prompt, right_prompt, image_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn first_divergence_reports_a_prefix_as_a_length_difference() {
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2]), Some(2));
        assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 4]), Some(2));
        assert_eq!(first_divergence(&[1, 2], &[1, 2]), None);
    }

    #[test]
    fn every_class_has_a_synthetic_example() {
        let cases = [
            (
                "esfuerzo",
                tags(&["thinking"]),
                "Reasoning Effort: 50\n",
                "Reasoning Effort: 75\n",
                0,
            ),
            (
                "herramientas",
                tags(&["tools"]),
                "## Tools\nx\n",
                "## Tools\ny\n",
                0,
            ),
            (
                "formato",
                tags(&["formato"]),
                "## Response Format\nx\n",
                "## Response Format\ny\n",
                0,
            ),
            (
                "thinking",
                tags(&["thinking"]),
                " thinking\nx\n",
                " response\ny\n",
                0,
            ),
            ("prefijo", tags(&["prefix"]), "a", "b", 0),
            (
                "resultados",
                tags(&["tools"]),
                "<tool_result>\nx\n",
                "<tool_result>\ny\n",
                0,
            ),
            (
                "sistema",
                tags(&["sistema"]),
                "<｜end_of_query｜> x",
                "<｜end_of_query｜> y",
                0,
            ),
            ("grafias", tags(&["grafias"]), "a", "b", 0),
            ("imagenes", tags(&["imagenes"]), "a", "b", 1),
            ("largo", tags(&["basico"]), "abc", "abcd", 0),
        ];
        for (want, tags, left, right, images) in cases {
            let (class, archivo) = classify(&tags, left, right, images);
            assert_eq!(class, want, "prompt {left:?} vs {right:?}");
            assert!(!archivo.is_empty());
        }
    }

    #[test]
    fn an_effort_rejection_names_the_effort_table() {
        let (class, archivo) = classify_window(&tags(&[]), "Reasoning Effort:", "", 0);
        assert_eq!(class, "esfuerzo");
        assert_eq!(archivo, responsible_file("esfuerzo"));
        assert!(archivo.contains("text.zig"));
    }

    #[test]
    fn a_shared_window_prefix_alone_is_not_largo() {
        assert_eq!(
            classify_window(&tags(&[]), "abc", "abcd", 0).0,
            "otro",
            "a window that happens to be a prefix must not claim largo"
        );
        assert_eq!(classify(&tags(&[]), "abc", "abcd", 0).0, "largo");
    }

    #[test]
    fn an_unknown_difference_is_otro() {
        assert_eq!(classify(&[], "abc", "abd", 0).0, "otro");
    }

    #[test]
    fn divergence_keeps_six_ids_per_side_and_a_short_text_window() {
        let left: Vec<u32> = (0..20).collect();
        let mut right = left.clone();
        right[10] = 99;
        let divergence = divergence(
            &left,
            &right,
            "0123456789abcdefghij",
            "0123456789Xbcdefghij",
            10,
            "otro",
            "—",
            "",
        );
        assert_eq!(divergence.posicion, Some(10));
        assert_eq!(divergence.ids_api.len(), 12);
        assert_eq!(divergence.ids_zig.len(), 12);
        assert!(divergence.texto_api.chars().count() <= 60);
        assert!(divergence.texto_zig.chars().count() <= 60);
    }
}
