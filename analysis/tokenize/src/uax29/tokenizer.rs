//! OpenNLP's UAX #29 `WordTokenizer`: the word segments of a text that
//! are words, each with its word type.
//!
//! A segment is a token when [`word_type`] gives it a type; spaces and
//! punctuation are dropped. A token longer than the maximum length, in
//! UTF-16 code units as in the Java, is cut into consecutive pieces of at
//! most that length that each carry the type of the whole word. A cut
//! never falls inside a character; when the first character of a piece
//! is longer than the maximum (a character outside the BMP with a maximum
//! of one), it is a piece on its own.

use std::num::NonZeroU32;

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_tokenize_layers::{WordType, WordTypes};

use super::segmenter::for_each_segment;
use super::word_type::word_type;
use crate::unit::tokenize_units;
use crate::utf8::decode;

/// Appends the word tokens of `text` to `spans`, as byte spans of `text`,
/// and the type of each to `types`. `max_token_length` is in UTF-16 code
/// units.
pub fn word_tokens(text: &str, max_token_length: NonZeroU32, spans: &mut Vec<Span>, types: &mut Vec<WordType>) {
    for_each_segment(text, |start, end| push_word(text, start, end, max_token_length, spans, types));
}

/// Appends the segment `text[start..end]` to `spans` and its type to
/// `types` when it is a word, cut into pieces of at most `max` UTF-16
/// code units.
#[inline]
pub(crate) fn push_word(
    text: &str,
    start: u32,
    end: u32,
    max: NonZeroU32,
    spans: &mut Vec<Span>,
    types: &mut Vec<WordType>,
) {
    let max = max.get();
    let segment = &text[start as usize..end as usize];
    let Some(ty) = word_type(segment) else { return };
    // A UTF-16 code unit takes at least one UTF-8 byte, so a segment
    // no longer than the maximum in bytes needs no cut.
    if end - start <= max {
        spans.push(Span { start, end });
        types.push(ty);
    } else {
        cut(segment, start, max, ty, spans, types);
    }
}

/// Emits `segment`, which starts at byte `base`, as pieces of at most
/// `max` UTF-16 code units, never cutting a character.
#[cold]
fn cut(segment: &str, base: u32, max: u32, ty: WordType, spans: &mut Vec<Span>, types: &mut Vec<WordType>) {
    let bytes = segment.as_bytes();
    let mut from = 0;
    while from < bytes.len() {
        let mut units = 0;
        let mut at = from;
        while at < bytes.len() {
            let (cp, len) = decode(bytes, at);
            let width = if cp >= 0x1_0000 { 2 } else { 1 };
            if units + width > max {
                break;
            }
            units += width;
            at += len;
        }
        if at == from {
            at += decode(bytes, from).1;
        }
        spans.push(Span { start: base + from as u32, end: base + at as u32 });
        types.push(ty);
        from = at;
    }
}

/// The UAX #29 word tokenizer as an annotator: writes the tokens and the
/// word type of each.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WordTokenizer {
    max_token_length: NonZeroU32,
}

impl WordTokenizer {
    /// OpenNLP's default maximum token length, in UTF-16 code units.
    pub const DEFAULT_MAX_TOKEN_LENGTH: NonZeroU32 = NonZeroU32::new(255).unwrap();

    pub fn new() -> WordTokenizer {
        WordTokenizer { max_token_length: Self::DEFAULT_MAX_TOKEN_LENGTH }
    }

    /// A tokenizer that cuts tokens longer than `max_token_length` UTF-16
    /// code units.
    pub fn with_max_token_length(max_token_length: NonZeroU32) -> WordTokenizer {
        WordTokenizer { max_token_length }
    }

    pub fn max_token_length(&self) -> NonZeroU32 {
        self.max_token_length
    }

    fn tokenize(&self, view: &View, sentences: Option<&Sentences>, types: &mut Vec<WordType>) -> Result<Tokens, Error> {
        let max = self.max_token_length;
        tokenize_units(Self::ID, view, sentences, |text, _, out| word_tokens(text, max, out, types))
    }

    fn hash(&self) -> u64 {
        crate::fnv1a(&self.max_token_length.get().to_le_bytes())
    }
}

impl Default for WordTokenizer {
    fn default() -> WordTokenizer {
        WordTokenizer::new()
    }
}

impl Annotator for WordTokenizer {
    const ID: &'static str = "turbo-tokenize:uax29";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens, WordTypes);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens, WordTypes), Error> {
        let mut kind = Vec::new();
        let tokens = self.tokenize(view, sentences, &mut kind)?;
        Ok((tokens, WordTypes { kind }))
    }

    fn config_hash(&self) -> u64 {
        self.hash()
    }
}

/// The UAX #29 word tokenizer writing the tokens alone, for a pipeline
/// that does not need the word types. Its tokens are those of
/// [`WordTokenizer`] with the same maximum length.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct UntypedWordTokenizer(pub WordTokenizer);

impl Annotator for UntypedWordTokenizer {
    const ID: &'static str = "turbo-tokenize:uax29-untyped";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens,);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens,), Error> {
        let mut kind = Vec::new();
        Ok((self.0.tokenize(view, sentences, &mut kind)?,))
    }

    fn config_hash(&self) -> u64 {
        self.0.hash()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(text: &str, max: u32) -> Vec<(&str, WordType)> {
        let (mut spans, mut types) = (Vec::new(), Vec::new());
        word_tokens(text, NonZeroU32::new(max).unwrap(), &mut spans, &mut types);
        assert_eq!(spans.len(), types.len());
        spans.iter().zip(types).map(|(s, t)| (&text[s.range()], t)).collect()
    }

    use WordType::*;

    #[test]
    fn words_are_kept_and_typed() {
        assert_eq!(
            tokens("Hello, world! 3.14 \u{1f600} \u{65e5}\u{672c}", 255),
            [
                ("Hello", Alphanumeric),
                ("world", Alphanumeric),
                ("3.14", Numeric),
                ("\u{1f600}", Emoji),
                ("\u{65e5}", Ideographic),
                ("\u{672c}", Ideographic),
            ]
        );
        assert_eq!(tokens(" ... ", 255), []);
    }

    #[test]
    fn long_tokens_are_cut_in_utf16_units() {
        assert_eq!(tokens("abcdefg", 3), [("abc", Alphanumeric), ("def", Alphanumeric), ("g", Alphanumeric)]);
        // U+00E9 is two bytes but one unit: "\u{e9}\u{e9}\u{e9}" fits in three.
        assert_eq!(
            tokens("\u{e9}\u{e9}\u{e9}\u{e9}", 3),
            [("\u{e9}\u{e9}\u{e9}", Alphanumeric), ("\u{e9}", Alphanumeric)]
        );
        // The cut would fall inside the pair of U+10400: it moves before it.
        assert_eq!(tokens("ab\u{10400}c", 3), [("ab", Alphanumeric), ("\u{10400}c", Alphanumeric)]);
        // A maximum of one cannot hold U+10400, which is then a piece of its own.
        assert_eq!(tokens("a\u{10400}b", 1), [("a", Alphanumeric), ("\u{10400}", Alphanumeric), ("b", Alphanumeric)]);
        // Every piece carries the type of the whole word.
        // Hangul syllables are ALetter, so "ab" and U+D55C are one Hangul word.
        assert_eq!(tokens("ab\u{d55c}", 2), [("ab", Hangul), ("\u{d55c}", Hangul)]);
    }

    #[test]
    fn default_maximum() {
        assert_eq!(WordTokenizer::default().max_token_length().get(), 255);
        let long = "a".repeat(300);
        let cut: Vec<usize> = tokens(&long, 255).iter().map(|(t, _)| t.len()).collect();
        assert_eq!(cut, [255, 45]);
        let short = WordTokenizer::with_max_token_length(NonZeroU32::new(8).unwrap());
        assert_ne!(WordTokenizer::new().config_hash(), short.config_hash());
        assert_eq!(WordTokenizer::new().config_hash(), UntypedWordTokenizer::default().config_hash());
    }
}
