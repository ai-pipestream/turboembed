//! The dictionary word tokenizer: the UAX #29 word tokenizer, with the
//! words of Thai, Lao, Khmer and Burmese runs found by their dictionaries.
//!
//! The segments are those of [`ComplexSegmenter`](super::ComplexSegmenter)
//! with the bundled dictionaries. As in [`WordTokenizer`](crate::WordTokenizer),
//! a segment is a token when it has a word type (a segment of a complex
//! script run has the type `SoutheastAsian`), and a token longer than the
//! maximum length in UTF-16 code units is cut into pieces.

use std::num::NonZeroU32;

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_tokenize_layers::{WordType, WordTypes};

use super::bundled;
use crate::uax29::tokenizer::push_word;
use crate::unit::tokenize_units;

/// Appends the word tokens of `text` to `spans`, as byte spans of `text`,
/// and the type of each to `types`, with the bundled dictionaries.
/// `max_token_length` is in UTF-16 code units.
pub fn dictionary_word_tokens(
    text: &str,
    max_token_length: NonZeroU32,
    spans: &mut Vec<Span>,
    types: &mut Vec<WordType>,
) {
    bundled().for_each_segment(text, |start, end| push_word(text, start, end, max_token_length, spans, types));
}

/// The dictionary word tokenizer as an annotator: writes the tokens and
/// the word type of each.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DictionaryWordTokenizer {
    max_token_length: NonZeroU32,
}

impl DictionaryWordTokenizer {
    /// The same default as the UAX #29 tokenizer: 255 UTF-16 code units.
    pub const DEFAULT_MAX_TOKEN_LENGTH: NonZeroU32 = crate::WordTokenizer::DEFAULT_MAX_TOKEN_LENGTH;

    pub fn new() -> DictionaryWordTokenizer {
        DictionaryWordTokenizer { max_token_length: Self::DEFAULT_MAX_TOKEN_LENGTH }
    }

    /// A tokenizer that cuts tokens longer than `max_token_length` UTF-16
    /// code units.
    pub fn with_max_token_length(max_token_length: NonZeroU32) -> DictionaryWordTokenizer {
        DictionaryWordTokenizer { max_token_length }
    }

    pub fn max_token_length(&self) -> NonZeroU32 {
        self.max_token_length
    }

    fn tokenize(&self, view: &View, sentences: Option<&Sentences>, types: &mut Vec<WordType>) -> Result<Tokens, Error> {
        let max = self.max_token_length;
        tokenize_units(Self::ID, view, sentences, |text, _, out| dictionary_word_tokens(text, max, out, types))
    }

    fn hash(&self) -> u64 {
        crate::fnv1a(&self.max_token_length.get().to_le_bytes())
    }
}

impl Default for DictionaryWordTokenizer {
    fn default() -> DictionaryWordTokenizer {
        DictionaryWordTokenizer::new()
    }
}

impl Annotator for DictionaryWordTokenizer {
    const ID: &'static str = "turbo-tokenize:uax29-dictionary";
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

/// The dictionary word tokenizer writing the tokens alone. Its tokens are
/// those of [`DictionaryWordTokenizer`] with the same maximum length.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct UntypedDictionaryWordTokenizer(pub DictionaryWordTokenizer);

impl Annotator for UntypedDictionaryWordTokenizer {
    const ID: &'static str = "turbo-tokenize:uax29-dictionary-untyped";
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
        dictionary_word_tokens(text, NonZeroU32::new(max).unwrap(), &mut spans, &mut types);
        assert_eq!(spans.len(), types.len());
        spans.iter().zip(types).map(|(s, t)| (&text[s.range()], t)).collect()
    }

    use WordType::*;

    #[test]
    fn complex_runs_are_words() {
        assert_eq!(
            tokens("ราคา 1,250 บาท, Thai: ภาษาไทย", 255),
            [
                ("ราคา", SoutheastAsian),
                ("1,250", Numeric),
                ("บาท", SoutheastAsian),
                ("Thai", Alphanumeric),
                ("ภาษา", SoutheastAsian),
                ("ไทย", SoutheastAsian),
            ]
        );
    }

    #[test]
    fn long_words_are_cut() {
        assert_eq!(tokens("ประเทศไทย", 3), [("ประ", SoutheastAsian), ("เทศ", SoutheastAsian), ("ไทย", SoutheastAsian)]);
    }

    #[test]
    fn hashes() {
        let short = DictionaryWordTokenizer::with_max_token_length(NonZeroU32::new(8).unwrap());
        assert_ne!(DictionaryWordTokenizer::new().config_hash(), short.config_hash());
        assert_eq!(
            DictionaryWordTokenizer::new().config_hash(),
            UntypedDictionaryWordTokenizer::default().config_hash()
        );
        assert_eq!(DictionaryWordTokenizer::default().max_token_length().get(), 255);
    }
}
