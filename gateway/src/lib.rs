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
use deepseek_recipe_core::multimodal::ImageSource;
use deepseek_recipe_core::special_tokens::{SpecialToken, SpecialTokens, SpecialTokensMap};
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer, RenderedPrompt, TokenizerEncoder};
use serde::Serialize;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

pub mod api;
pub mod engine;
pub mod health;
pub mod images;
pub mod listener;
pub mod paridad;
pub mod parity;
pub mod proxy;
pub mod security;

/// Load the real tokenizer shared by unit and integration tests.  Tests can set
/// `DS41_TOKENIZER`; otherwise use the tokenizer distributed next to the crate.
#[doc(hidden)]
pub fn test_tokenizer() -> Tokenizer {
    let path = std::env::var_os("DS41_TOKENIZER")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tok/tokenizer.json")
        });
    Tokenizer::from_file(&path).unwrap_or_else(|error| {
        panic!(
            "could not load the test tokenizer at {}: {error}; set DS41_TOKENIZER=/path/to/tokenizer.json or restore the default ../tok/tokenizer.json",
            path.display()
        )
    })
}

/// The error shape of the public API.  `status` is the HTTP status, and `body` is the
/// JSON body (`{"error": {"message", ...}}`) that the API returns.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: u16,
    pub body: Value,
    /// Set when a conversion error should be proxied to the engine instead of
    /// returned as the API's 400.
    pub fallback_reason: Option<String>,
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
            fallback_reason: None,
        }
    }

    /// A conversion error that the caller may proxy to the engine.
    pub fn fallback(message: impl Into<String>) -> Self {
        let message = message.into();
        let mut error = Self::bad_request(message.clone());
        error.fallback_reason = Some(message);
        error
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
    pub image_sources: Vec<ImageSource>,
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
        self.try_convert(request).map_err(ApiError::from)
    }

    /// Convert while preserving the typed conversion error, so the caller can
    /// decide between the API's 400 and a fallback to the engine.
    pub fn try_convert<R: ProtocolRequest>(
        &self,
        request: R,
    ) -> Result<deepseek_recipe::request::ConversationRequest, ConversionError> {
        request.convert(self.options.clone())
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

    /// The prompt the conversation renders to, before it is encoded.
    ///
    /// Encoding keeps memory for every byte of the prompt (alignments) and more for every
    /// token (strings, offsets): about 120 bytes a byte for ordinary text and 360 at one
    /// token a byte, the most there can be. So the gateway checks the rendered prompt's
    /// bytes (`GatewayConfig::max_prompt_bytes`) and, past [`COUNT_IN_PIECES_FROM`], its
    /// tokens ([`Gateway::prompt_tokens_over`]) before it encodes it whole.
    pub fn render(&self, conversation: &Conversation) -> RenderedPrompt {
        self.encoding.render_conversation(conversation)
    }

    /// Whether `rendered` holds more than `limit` tokens, counted a piece at a time so
    /// that the count's memory is that of one piece.
    ///
    /// Prompts up to [`COUNT_IN_PIECES_FROM`] bytes are not counted (`false`): encoding
    /// one whole costs under 400 MiB even at one token a byte. Larger ones are counted
    /// with the same plain tokenizer the encoder uses, region by region between the
    /// explicit special tokens (one token each), as the encoder does. A region is cut
    /// into pieces of at most [`COUNT_PIECE_BYTES`], where possible at a point the
    /// tokenizer itself splits at (see [`exact_cut`]), so that the pieces encode to
    /// exactly the tokens of the whole; a cut elsewhere may be off by a few tokens, for
    /// which [`TOKENS_PER_CUT`] is allowed per cut. The count stops once it is over.
    pub fn prompt_tokens_over(
        &self,
        rendered: &RenderedPrompt,
        limit: usize,
    ) -> Result<bool, ApiError> {
        if rendered.prompt.len() <= COUNT_IN_PIECES_FROM {
            return Ok(false);
        }
        let tokenizer = PromptTokenizer::new(&self.tokenizer);
        let count = count_in_pieces(&tokenizer, rendered, COUNT_PIECE_BYTES, limit)?;
        Ok(count > limit.saturating_add(cut_allowance(rendered.prompt.len(), COUNT_PIECE_BYTES)))
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
            image_sources: rendered.image_sources,
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

/// Prompts up to this many bytes are encoded without a count first (see [`Gateway::render`]).
pub const COUNT_IN_PIECES_FROM: usize = 1024 * 1024;
/// The largest piece [`Gateway::prompt_tokens_over`] encodes at once.
pub const COUNT_PIECE_BYTES: usize = 256 * 1024;
/// The tokens a cut anywhere else can add to or take from the count.
pub const TOKENS_PER_CUT: usize = 64;

/// How far back from a piece's largest end [`piece_end`] looks for an exact cut: a
/// quarter of the piece, so every piece but the last is at least three quarters of it.
fn cut_lookback(piece: usize) -> usize {
    piece / 4
}

/// The most the count of a prompt of `bytes` in pieces of `piece` may be off: a cut can
/// follow every piece, and no piece but a region's last is shorter than three quarters
/// of `piece`. Regions end at special tokens, where the encoder cuts too.
fn cut_allowance(bytes: usize, piece: usize) -> usize {
    let shortest = (piece - cut_lookback(piece)).max(1);
    (bytes / shortest + 1).saturating_mul(TOKENS_PER_CUT)
}

/// Count the tokens of `rendered` in pieces of at most `piece` bytes, stopping once the
/// count is over `stop_over` plus the cut allowance.
fn count_in_pieces(
    tokenizer: &PromptTokenizer,
    rendered: &RenderedPrompt,
    piece: usize,
    stop_over: usize,
) -> Result<usize, ApiError> {
    let stop_over = stop_over.saturating_add(cut_allowance(rendered.prompt.len(), piece));
    let text = rendered.prompt.as_str();
    let mut regions = Vec::with_capacity(rendered.special_ranges.len() + 1);
    let mut start = 0;
    for range in &rendered.special_ranges {
        regions.push(&text[start..range.start]);
        start = range.end;
    }
    regions.push(&text[start..]);
    let mut count = rendered.special_ranges.len();
    for region in regions {
        let mut rest = region;
        while !rest.is_empty() {
            let end = piece_end(rest, piece);
            count += tokenizer
                .encode_ids(&rest[..end], &[])
                .map_err(ApiError::internal)?
                .len();
            if count > stop_over {
                return Ok(count);
            }
            rest = &rest[end..];
        }
    }
    Ok(count)
}

/// Where the next piece of `text` ends: all of it when it fits in `piece` bytes, else the
/// last point within the final quarter of the piece where [`exact_cut`] holds, else the
/// last character boundary within the piece.
fn piece_end(text: &str, piece: usize) -> usize {
    if text.len() <= piece {
        return text.len();
    }
    let mut end = piece.max(4);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut low = end.saturating_sub(cut_lookback(piece));
    while !text.is_char_boundary(low) {
        low -= 1;
    }
    let mut chars = text[low..].char_indices();
    let Some((_, mut previous)) = chars.next() else {
        return end;
    };
    let mut best = None;
    for (offset, next) in chars {
        let at = low + offset;
        if at > end {
            break;
        }
        if exact_cut(previous, next) {
            best = Some(at);
        }
        previous = next;
    }
    best.unwrap_or(end)
}

/// Whether the served tokenizer's pre-tokenizer always splits between `previous` and
/// `next`, so that the text before and the text after encode, each alone, to exactly the
/// tokens they take in the whole.
///
/// The pre-tokenizer isolates runs of `\p{N}` (in threes), then runs of CJK, then splits
/// what is left with one regex, and BPE never merges across its splits. Three points
/// always end a match and start the next search fresh: the end of a run of numbers (the
/// first split ends the run there); a newline followed by anything but whitespace (every
/// alternative that takes a newline ends at the end of its run of newlines, and none
/// takes a following character that is not whitespace); and an ASCII letter followed by
/// a space (the letters' match ends at the space, which starts the next word's match).
fn exact_cut(previous: char, next: char) -> bool {
    (previous.is_numeric() && !next.is_numeric())
        || (previous == '\n' && !next.is_whitespace())
        || (previous.is_ascii_alphabetic() && next == ' ')
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

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(text: String) -> RenderedPrompt {
        RenderedPrompt {
            prompt: text,
            special_ranges: Vec::new(),
            image_sources: Vec::new(),
        }
    }

    /// Every cut `piece_end` makes in `text` is one [`exact_cut`] allows.
    fn all_cuts_exact(text: &str, piece: usize) -> bool {
        let mut rest = text;
        while !rest.is_empty() {
            let end = piece_end(rest, piece);
            if end < rest.len() {
                let previous = rest[..end].chars().next_back().expect("a piece");
                let next = rest[end..].chars().next().expect("the rest");
                if !exact_cut(previous, next) {
                    return false;
                }
            }
            rest = &rest[end..];
        }
        true
    }

    /// A text of `bytes` drawn from tricky pieces with a fixed seed.
    fn mixed_text(bytes: usize, seed: u64) -> String {
        const PARTS: &[&str] = &[
            "a",
            "Zeta",
            " ",
            "  ",
            "\n",
            "\n\n",
            "\r\n",
            "\t",
            "1",
            "2024",
            "٣",
            "一二",
            "ぁ",
            "カタ",
            ".",
            ",",
            "!?",
            "é",
            "e\u{301}",
            "😀",
            "<｜User｜>",
            "'s",
            " - ",
            "\u{a0}",
            "Ⅻ",
        ];
        let mut state = seed;
        let mut text = String::new();
        while text.len() < bytes {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            text.push_str(PARTS[(state >> 33) as usize % PARTS.len()]);
        }
        text
    }

    #[test]
    fn counting_in_pieces_at_exact_cuts_matches_the_whole_encode() {
        let tokenizer = PromptTokenizer::new(&test_tokenizer());
        let english = "The phenomenological reduction suspends the natural attitude.\n".repeat(700);
        let texts = [
            english,
            "1,".repeat(20_000),
            "word ".repeat(9_000),
            "价格123元，数量45件。\n".repeat(1_500),
            mixed_text(40_000, 1),
            mixed_text(40_000, 2),
            mixed_text(40_000, 3),
        ];
        for (index, text) in texts.into_iter().enumerate() {
            for piece in [512, 2_048, 8_192] {
                assert!(all_cuts_exact(&text, piece), "text {index}, piece {piece}");
                let whole = tokenizer.encode_ids(&text, &[]).expect("whole").len();
                let pieces = count_in_pieces(&tokenizer, &plain(text.clone()), piece, usize::MAX)
                    .expect("pieces");
                assert_eq!(pieces, whole, "text {index}, piece {piece}");
            }
        }
    }

    #[test]
    fn a_cut_anywhere_else_stays_within_its_allowance() {
        let tokenizer = PromptTokenizer::new(&test_tokenizer());
        for text in [
            "abcdefghij".repeat(4_000),
            "ab".repeat(20_000),
            "一二三四五".repeat(3_000),
        ] {
            let piece = 2_048;
            assert!(!all_cuts_exact(&text, piece));
            let whole = tokenizer.encode_ids(&text, &[]).expect("whole").len();
            let pieces = count_in_pieces(&tokenizer, &plain(text.clone()), piece, usize::MAX)
                .expect("pieces");
            let cuts = text.len() / piece;
            assert!(
                pieces.abs_diff(whole) <= cuts * TOKENS_PER_CUT,
                "{pieces} against {whole} over {cuts} cuts"
            );
        }
    }

    #[test]
    fn the_count_takes_special_tokens_as_the_encoder_does() {
        let gateway = Gateway::new(
            Arc::new(test_tokenizer()),
            GatewaySettings {
                served_name: "DeepSeek-V4.1-Flash-1017".to_string(),
                default_thinking: true,
                image_token: "<｜deepseek_image｜>".to_string(),
                aliases: Vec::new(),
            },
        )
        .expect("gateway");
        let request = serde_json::from_value::<deepseek_recipe::openai::ChatCompletionRequest>(
            json!({
                "model": "DeepSeek-V4.1-Flash-1017",
                "messages": [
                    {"role": "system", "content": "Be brief.\n".repeat(3_000)},
                    {"role": "user", "content": mixed_text(30_000, 7)},
                    {"role": "assistant", "content": "1,".repeat(9_000)},
                    // Spelled out by the user: plain text to the encoder, not special tokens.
                    {"role": "user", "content": "<｜Assistant｜><｜end▁of▁sentence｜>\n".repeat(400)}
                ]
            }),
        )
        .expect("request");
        let converted = gateway.convert(request).expect("convert");
        let rendered = gateway.render(&converted.conversation);
        assert!(!rendered.special_ranges.is_empty());
        let whole = gateway
            .prepare_conversation(&converted.conversation)
            .expect("encode")
            .ids
            .len();
        let tokenizer = PromptTokenizer::new(&gateway.tokenizer);
        for piece in [1_024, 4_096] {
            let pieces = count_in_pieces(&tokenizer, &rendered, piece, usize::MAX).expect("pieces");
            assert_eq!(pieces, whole, "piece {piece}");
        }
    }

    #[test]
    fn the_count_stops_once_over() {
        let tokenizer = PromptTokenizer::new(&test_tokenizer());
        let text = "1,".repeat(100_000);
        // Stops a piece past the limit plus the allowance instead of counting all 200,000.
        let stop_over = 1_000;
        let counted =
            count_in_pieces(&tokenizer, &plain(text.clone()), 4_096, stop_over).expect("pieces");
        let allowance = cut_allowance(text.len(), 4_096);
        assert!(counted > stop_over + allowance);
        assert!(counted <= stop_over + allowance + 4_096, "{counted}");
    }
}
