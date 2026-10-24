//! T018: real-tokenizer rendering, offsets and cache marks.

use std::sync::Arc;

use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::ProtocolRequest;
use deepseek_recipe_core::special_tokens::{SpecialToken, SpecialTokens};
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer, TokenizerEncoder};
use ds41_gateway::{Gateway, GatewaySettings};
use serde_json::{Value, json};

fn gateway() -> Gateway {
    let tokenizer = ds41_gateway::test_tokenizer();
    Gateway::new(
        Arc::new(tokenizer),
        GatewaySettings {
            served_name: "DeepSeek-V4.1-Flash-1017".to_string(),
            default_thinking: true,
            image_token: "<｜deepseek_image｜>".to_string(),
            aliases: Vec::new(),
        },
    )
    .expect("build gateway")
}

fn request(body: Value) -> ChatCompletionRequest {
    serde_json::from_value(body).expect("valid chat request")
}

fn prepare(gateway: &Gateway, body: Value) -> ds41_gateway::EncodedPrompt {
    let converted = request(body)
        .convert(gateway.options.clone())
        .expect("conversion");
    gateway
        .prepare_conversation(&converted.conversation)
        .expect("render and encode")
}

#[test]
fn simple_chat_history_len_is_before_the_assistant_header() {
    let gateway = gateway();
    let prepared = prepare(
        &gateway,
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hello"}],
        }),
    );
    assert_eq!(prepared.history_len as usize, prepared.ids.len() - 2);
    assert!(prepared.ids.len() > 2);
}

#[test]
fn required_tool_choice_moves_history_len_before_the_dsml_prefix() {
    let gateway = gateway();
    let prepared = prepare(
        &gateway,
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "thinking": {"type": "disabled"},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather by city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }],
            "tool_choice": "required",
        }),
    );
    // The prompt ends with the assistant header, the thinking marker and the DSML
    // tool-call prefix; history_len excludes all of them.
    let dsml = gateway
        .special_tokens()
        .token(SpecialToken::Dsml)
        .to_string();
    let prefix = format!("\n\n<{dsml} calls>\n");
    let ranges: Vec<_> = prefix
        .match_indices(&dsml)
        .map(|(start, _)| start..start + dsml.len())
        .collect();
    let prefix_ids = gateway
        .encode_fragment(&prefix, &ranges)
        .expect("encode prefix");
    assert_eq!(
        prepared.history_len as usize,
        prepared.ids.len() - 2 - prefix_ids.len()
    );
}

#[test]
fn prefix_continuation_has_no_history_mark() {
    let gateway = gateway();
    let prepared = prepare(
        &gateway,
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [
                {"role": "user", "content": "Write a sentence"},
                {"role": "assistant", "content": "The", "prefix": true}
            ],
        }),
    );
    assert_eq!(prepared.history_len, 0);
}

#[test]
fn long_system_prompt_yields_three_shared_prefix_cuts() {
    let gateway = gateway();
    // 600 groups of five words, about 3000 tokens: enough for s - 2048 >= 512.
    let system = "uno dos tres cuatro cinco ".repeat(600);
    let prepared = prepare(
        &gateway,
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "Hola"}
            ],
        }),
    );
    let s = prepared
        .ids
        .iter()
        .position(|&id| {
            // First user header: find it through the special-token spellings.
            id == gateway
                .tokenizer
                .token_to_id(gateway.special_tokens().token(SpecialToken::User))
                .expect("user token")
        })
        .expect("user header");
    assert!(s >= 512, "user header at {s}");
    let expected: Vec<u32> = [s.saturating_sub(2048), s.saturating_sub(512), s]
        .into_iter()
        .filter(|&cut| cut >= 512)
        .map(|cut| cut as u32)
        .collect();
    assert_eq!(prepared.shared_prefixes, expected);
    assert_eq!(prepared.shared_prefixes.len(), 3);
}

#[test]
fn offsets_encoder_matches_the_recipe_encoder_for_readme_prompts() {
    let gateway = gateway();
    let cases = [
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Hello"}],
        }),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi", "reasoning_content": "greet"},
                {"role": "user", "content": "How are you?"}
            ],
            "thinking": {"type": "enabled"}
        }),
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{"role": "user", "content": "Use the tool"}],
            "thinking": {"type": "disabled"},
            "tools": [{
                "type": "function",
                "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}
            }],
            "tool_choice": "auto"
        }),
    ];
    let prompt_tokenizer = PromptTokenizer::new(&gateway.tokenizer);
    for body in cases {
        let converted = request(body)
            .convert(gateway.options.clone())
            .expect("convert");
        let rendered = gateway
            .encoding
            .render_conversation(&converted.conversation);
        let recipe_ids = prompt_tokenizer
            .encode_ids(&rendered.prompt, &rendered.special_ranges)
            .expect("recipe encode");
        let prepared = gateway
            .prepare_conversation(&converted.conversation)
            .expect("gateway encode");
        assert_eq!(prepared.ids, recipe_ids);
    }
}

#[test]
fn image_token_override_is_used_by_conversion() {
    let gateway = gateway();
    let prepared = prepare(
        &gateway,
        json!({
            "model": "DeepSeek-V4.1-Flash-1017",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]
            }]
        }),
    );
    assert_eq!(prepared.image_count, 1);
    let image_id = gateway
        .tokenizer
        .token_to_id(gateway.special_tokens().token(SpecialToken::Image))
        .expect("image token in vocabulary");
    assert_eq!(prepared.ids.iter().filter(|&&id| id == image_id).count(), 1);
}
