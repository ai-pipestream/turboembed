//! Word tokenizers ported from OpenNLP 3.x onto the analysis API.
//!
//! Each tokenizer reads a `View` and, when the document has one, a
//! `Sentences` layer, and writes `Tokens`: with sentences, each sentence
//! is tokenized on its own and every token records its sentence
//! ([`unit`]). Every offset is a UTF-8 byte offset into the view, on a
//! character boundary.
//!
//! - [`simple`]: `SimpleTokenizer`, runs of one character class.
//! - [`whitespace`]: `WhitespaceTokenizer`, runs of non-whitespace.
//! - [`uax29`]: the UAX #29 word segmenter and `WordTokenizer`, which
//!   also writes the word type of each token.
//! - [`unigram`]: `UnigramSegmenter`, the frequency lexicon segmenter
//!   for Chinese and other text written without spaces.
//! - [`complex`]: the UAX #29 word tokenizer with dictionaries for Thai,
//!   Lao, Khmer and Burmese runs, from ICU4X's segmenter data.
//! - [`lattice`]: `LatticeTokenizer`, the Viterbi segmenter over a MeCab
//!   dictionary for Japanese and Korean, which also writes the part of
//!   speech of each token, and `LatticeAnalyzer`, which also writes each
//!   token's reading, pronunciation and lemma for a known dictionary
//!   format.
//! - [`me`]: the maxent tokenizer, `TokenizerME`, which reads an OpenNLP
//!   tokenizer model and writes a confidence per token; the model file's
//!   SHA-256 from turbo-maxent identifies it in the provenance.
//!
//! Each rule tokenizer is also a plain function over a `&str` that appends
//! spans to a caller's buffer, for use outside a pipeline.
//!
//! [`detok`] is OpenNLP's `DictionaryDetokenizer`, which rebuilds text
//! from tokens with a detokenizer dictionary. It is not an annotator.
//!
//! docs/segmentation.md describes the design and every place the ports
//! differ from OpenNLP on purpose.

pub mod complex;
pub mod detok;
pub mod lattice;
pub mod me;
pub mod simple;
pub mod uax29;
pub mod unigram;
pub mod unit;
pub mod whitespace;

mod lines;
mod utf8;

pub use complex::{
    ComplexScript, ComplexSegmenter, DictionaryWordTokenizer, UntypedDictionaryWordTokenizer, dictionary_word_tokens,
};
pub use detok::{Detokenizer, DetokenizerDictionary, DictionaryError, DictionaryOperation, Operation};
pub use lattice::{DictionaryFormat, LatticeAnalyzer, LatticeTokenizer, MecabDictionary};
pub use simple::{SimpleTokenizer, simple_spans};
pub use turbo_tokenize_layers::{Readings, WordType, WordTypes};
pub use uax29::{UntypedWordTokenizer, WordTokenizer, word_boundaries, word_tokens};
pub use unigram::{LexiconError, UnigramLexicon, UnigramMode, UnigramSegmenter};
pub use unit::{UnitToken, tokenize_units, tokenize_units_with};
pub use whitespace::{WhitespaceTokenizer, whitespace_spans};

/// The options the simple, whitespace and maxent tokenizers share. For
/// the maxent tokenizer they decide the whitespace units the model sees.
/// Whitespace is always the Unicode `White_Space` property
/// ([`turbo_unicode::whitespace::is_whitespace`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TokenizerOptions {
    /// Emit every line feed and carriage return as a token of its own.
    pub keep_newlines: bool,
}

impl TokenizerOptions {
    /// The options as a configuration hash, for provenance: 64-bit FNV-1a
    /// over [`WHITESPACE_DEFINITION`] and one byte per option, the same on
    /// every platform.
    pub fn config_hash(self) -> u64 {
        fnv1a(&[WHITESPACE_DEFINITION, u8::from(self.keep_newlines)])
    }
}

/// The whitespace definition every tokenizer uses, as the byte a
/// configuration hash folds in: 1 is the Unicode `White_Space` property.
/// It changes only if that definition ever does, so a hash recorded for a
/// tokenizer stays valid.
pub(crate) const WHITESPACE_DEFINITION: u8 = 1;

/// The 64-bit FNV-1a hash of `bytes`: small, fixed and the same on every
/// platform, which is all a configuration hash needs.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    turbo_analysis::fnv1a(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options() {
        let o = TokenizerOptions::default();
        assert!(!o.keep_newlines);
    }

    #[test]
    fn config_hashes_differ_per_option() {
        let plain = TokenizerOptions { keep_newlines: false };
        let newlines = TokenizerOptions { keep_newlines: true };
        assert_ne!(plain.config_hash(), newlines.config_hash());
        // Pinned, so a recorded provenance hash stays valid.
        assert_eq!(plain.config_hash(), fnv1a(&[1, 0]));
        assert_eq!(newlines.config_hash(), fnv1a(&[1, 1]));
    }

    #[test]
    fn fnv1a_reference_values() {
        assert_eq!(fnv1a(b""), turbo_analysis::FNV1A_OFFSET_BASIS);
        assert_eq!(fnv1a(b"a"), 0xAF63_DC4C_8601_EC8C);
    }
}
