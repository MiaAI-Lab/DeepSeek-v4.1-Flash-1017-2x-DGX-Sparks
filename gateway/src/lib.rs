//! Core of the `ds41-gateway`: prompt conversion, rendering, token encoding with offsets,
//! and the cache marks (`history_len`, `shared_prefixes`) that the token route consumes.
//!
//! The gateway is a thin, faithful front end for deepseek-recipe 0.2.0.  It keeps the
//! tokenizer loaded once, renders and encodes with the official library, and hands the
//! resulting token IDs to the engine's token-in/token-out route.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use deepseek_recipe::request::{ConversionError, ConversionOptions, ModelVersion, ProtocolRequest};
use deepseek_recipe_core::conversation::Conversation;
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::special_tokens::{SpecialToken, SpecialTokens, SpecialTokensMap};
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer, RenderedPrompt, TokenizerEncoder};
use serde::Serialize;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

pub mod api;
pub mod engine;
pub mod health;
pub mod parity;

/// The error shape of the public API.  `status` is the HTTP status, and `body` is the
/// JSON body (`{"error": {"message", ...}}`) that the API returns.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: u16,
    pub body: Value,
}

impl ApiError {
    /// Build an error with the API's shape.
    pub fn new(status: u16, message: impl Into<String>, kind: &str) -> Self {
        let kind = kind.to_string();
        Self {
            status,
            body: json!({
                "error": {
                    "message": message.into(),
                    "type": kind,
                    "param": null,
                    "code": kind,
                }
            }),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(400, message, "invalid_request_error")
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(500, message, "internal_error")
    }

    pub fn bad_gateway(message: impl Into<String>) -> Self {
        Self::new(502, message, "internal_error")
    }

    /// The error body as JSON text.
    pub fn body_text(&self) -> String {
        self.body.to_string()
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
            .body
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
        {
            Some(message) => write!(formatter, "{}: {message}", self.status),
            None => write!(formatter, "{}", self.status),
        }
    }
}

impl std::error::Error for ApiError {}

impl From<ConversionError> for ApiError {
    fn from(error: ConversionError) -> Self {
        let status = error.status_code();
        let kind = if status >= 500 {
            "internal_error"
        } else {
            "invalid_request_error"
        };
        Self::new(status, error.to_string(), kind)
    }
}

/// A converted request, its rendered prompt and the cache marks the route needs.
#[derive(Debug, Clone)]
pub struct EncodedPrompt {
    pub prompt: String,
    pub ids: Vec<u32>,
    pub history_len: u32,
    pub shared_prefixes: Vec<u32>,
    pub image_count: usize,
}

/// Settings needed to build a [`Gateway`].
#[derive(Debug, Clone)]
pub struct GatewaySettings {
    /// The model name this gateway serves.  It is accepted together with the official
    /// DeepSeek aliases.
    pub served_name: String,
    /// Whether requests without an explicit thinking mode default to thinking.
    pub default_thinking: bool,
    /// Spelling of the image token in the served tokenizer.
    pub image_token: String,
    /// Additional accepted model aliases.
    pub aliases: Vec<String>,
}

/// The tokenizer and the deepseek-recipe options shared by every request.
pub struct Gateway {
    pub options: ConversionOptions,
    pub tokenizer: Arc<Tokenizer>,
    pub encoding: DeepseekV41Encoding,
    pub eos: u32,
    special_tokens: Arc<dyn SpecialTokens>,
}

impl Gateway {
    /// Load a tokenizer file and build the gateway.
    pub fn from_file(path: &std::path::Path, settings: GatewaySettings) -> Result<Self, ApiError> {
        let tokenizer = Tokenizer::from_file(path).map_err(|error| {
            ApiError::internal(format!("cannot load {}: {error}", path.display()))
        })?;
        Self::new(Arc::new(tokenizer), settings)
    }

    /// Build the gateway from an already loaded tokenizer.
    pub fn new(tokenizer: Arc<Tokenizer>, settings: GatewaySettings) -> Result<Self, ApiError> {
        let tokens = SpecialTokensMap {
            image: settings.image_token.clone(),
            ..SpecialTokensMap::default()
        };
        let special_tokens: Arc<dyn SpecialTokens> = Arc::new(tokens);
        let mut options = ConversionOptions::new()
            .with_default_thinking_mode(settings.default_thinking)
            .with_special_tokens(Arc::clone(&special_tokens));
        for name in std::iter::once(settings.served_name.clone())
            .chain([
                "DeepSeek-V4.1-Flash".to_string(),
                "deepseek-flash".to_string(),
            ])
            .chain(settings.aliases)
        {
            options = options.with_model_alias(name, ModelVersion::DeepseekV41Flash);
        }
        let encoding = DeepseekV41Encoding::new().with_special_tokens(Arc::clone(&special_tokens));
        let eos = tokenizer
            .encode(SpecialToken::Eos.default_str(), false)
            .map_err(|error| ApiError::internal(format!("cannot encode EOS: {error}")))?
            .get_ids()
            .first()
            .copied()
            .ok_or_else(|| ApiError::internal("the tokenizer has no EOS token"))?;
        Ok(Self {
            options,
            tokenizer,
            encoding,
            eos,
            special_tokens,
        })
    }

    /// The special-token mapping used for rendering and conversion.
    pub fn special_tokens(&self) -> Arc<dyn SpecialTokens> {
        Arc::clone(&self.special_tokens)
    }

    /// Convert a protocol request with the gateway's options.
    pub fn convert<R: ProtocolRequest>(
        &self,
        request: R,
    ) -> Result<deepseek_recipe::request::ConversationRequest, ApiError> {
        request
            .convert(self.options.clone())
            .map_err(ApiError::from)
    }

    /// Render the conversation, then encode it while recording where every explicit
    /// special token starts.
    pub fn encode_conversation(
        &self,
        conversation: &Conversation,
    ) -> Result<(RenderedPrompt, Vec<u32>, Vec<usize>), ApiError> {
        let rendered = self.encoding.render_conversation(conversation);
        let encoder = OffsetsEncoder::new(&self.tokenizer);
        let encoding = DeepseekV41Encoding::new()
            .with_special_tokens(Arc::clone(&self.special_tokens))
            .with_tokenizer(encoder.clone());
        let ids = encoding
            .encode(conversation)
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let offsets = encoder.take_offsets();
        Ok((rendered, ids, offsets))
    }

    /// Render and encode a conversation, deriving `history_len` and `shared_prefixes`.
    pub fn prepare_conversation(
        &self,
        conversation: &Conversation,
    ) -> Result<EncodedPrompt, ApiError> {
        let (rendered, ids, offsets) = self.encode_conversation(conversation)?;
        let history_len = history_len(
            conversation,
            &rendered,
            &offsets,
            self.special_tokens.as_ref(),
        );
        let shared_prefixes = shared_prefixes(&rendered, &offsets, self.special_tokens.as_ref());
        Ok(EncodedPrompt {
            prompt: rendered.prompt,
            ids,
            history_len,
            shared_prefixes,
            image_count: rendered.image_sources.len(),
        })
    }

    /// Encode an arbitrary prompt fragment with explicit special-token ranges.  Used by
    /// tests and by the render endpoint for diagnostics.
    pub fn encode_fragment(
        &self,
        text: &str,
        special_ranges: &[Range<usize>],
    ) -> Result<Vec<u32>, ApiError> {
        let encoder = OffsetsEncoder::new(&self.tokenizer);
        encoder
            .encode_ids(text, special_ranges)
            .map_err(ApiError::internal)
    }
}

/// A tokenizer encoder that behaves exactly like [`PromptTokenizer`] but records the token
/// index at which each explicit special-token range starts.
#[derive(Clone)]
pub struct OffsetsEncoder {
    inner: PromptTokenizer,
    offsets: Arc<Mutex<Vec<usize>>>,
}

impl OffsetsEncoder {
    /// Wrap the recipe's prompt tokenizer, which keeps the plain tokenizer and the
    /// added-vocabulary spellings exactly as deepseek-recipe does.
    pub fn new(tokenizer: &Tokenizer) -> Self {
        Self {
            inner: PromptTokenizer::new(tokenizer),
            offsets: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Take the recorded offsets, leaving the encoder ready for another prompt.
    pub fn take_offsets(&self) -> Vec<usize> {
        std::mem::take(&mut *self.offsets.lock().expect("offsets mutex poisoned"))
    }
}

impl TokenizerEncoder for OffsetsEncoder {
    fn encode_ids(&self, text: &str, special_ranges: &[Range<usize>]) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();
        let mut offsets = Vec::with_capacity(special_ranges.len());
        let mut start = 0;
        for range in special_ranges {
            ids.extend(self.inner.encode_ids(&text[start..range.start], &[])?);
            offsets.push(ids.len());
            // Encode the spelling as an explicit special token: one added-vocabulary
            // id and no surrounding text.
            let special = &text[range.clone()];
            let special_range = 0..special.len();
            ids.extend(
                self.inner
                    .encode_ids(special, std::slice::from_ref(&special_range))?,
            );
            start = range.end;
        }
        ids.extend(self.inner.encode_ids(&text[start..], &[])?);
        *self.offsets.lock().expect("offsets mutex poisoned") = offsets;
        Ok(ids)
    }
}

/// Whether the final message is a prefix continuation.
fn is_prefix_continuation(conversation: &Conversation) -> bool {
    matches!(
        conversation.messages.last(),
        Some(InputMessage::Assistant { prefix: true, .. })
    )
}

/// `history_len` is the token index of the final assistant header; 0 for a prefix
/// continuation, as the token route expects.
fn history_len(
    conversation: &Conversation,
    rendered: &RenderedPrompt,
    offsets: &[usize],
    tokens: &dyn SpecialTokens,
) -> u32 {
    if is_prefix_continuation(conversation) {
        return 0;
    }
    let assistant = tokens.token(SpecialToken::Assistant);
    let mut result = 0;
    for (index, range) in rendered.special_ranges.iter().enumerate() {
        if &rendered.prompt[range.start..range.end] == assistant {
            result = offsets[index] as u32;
        }
    }
    result
}

/// The shared prefix cuts `[s - 2048, s - 512, s]` (only values >= 512) where `s` is
/// the token index of the first user header, matching the native chat route.
fn shared_prefixes(
    rendered: &RenderedPrompt,
    offsets: &[usize],
    tokens: &dyn SpecialTokens,
) -> Vec<u32> {
    let user = tokens.token(SpecialToken::User);
    let Some(s) = rendered
        .special_ranges
        .iter()
        .enumerate()
        .find(|(_, range)| &rendered.prompt[range.start..range.end] == user)
        .map(|(index, _)| offsets[index])
    else {
        return Vec::new();
    };
    let s = s as u32;
    if s < 512 {
        return Vec::new();
    }
    [s.saturating_sub(2048), s.saturating_sub(512), s]
        .into_iter()
        .filter(|&cut| cut >= 512)
        .collect()
}

/// A model alias table exposed for diagnostics and tests.
pub fn accepted_model_names(options: &ConversionOptions) -> Vec<String> {
    let mut names: Vec<String> = options.model_versions_by_name.keys().cloned().collect();
    names.sort();
    names
}

/// Serialize a protocol response into a JSON value.
pub fn response_json<T: Serialize>(response: &T) -> Value {
    serde_json::to_value(response).unwrap_or(Value::Null)
}
