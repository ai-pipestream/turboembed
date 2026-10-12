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
//!
//! - [`detok`]: `DictionaryDetokenizer`, which rebuilds text from
//!   tokens with a detokenizer dictionary.
//!
//! Each tokenizer is also a plain function over a `&str` that appends
//! spans to a caller's buffer, for use outside a pipeline. The maxent
//! tokenizer is a later addition to this crate. docs/segmentation.md describes the design and every place the
//! ports differ from OpenNLP on purpose.

pub mod detok;
pub mod simple;
pub mod uax29;
pub mod unit;
pub mod whitespace;

mod utf8;

pub use detok::{Detokenizer, DetokenizerDictionary, DictionaryError, Operation};
pub use simple::{SimpleTokenizer, simple_spans};
pub use turbo_tokenize_layers::{WordType, WordTypes};
pub use turbo_unicode::WhitespaceMode;
pub use uax29::{UntypedWordTokenizer, WordTokenizer, word_boundaries, word_tokens};
pub use unit::tokenize_units;
pub use whitespace::{WhitespaceTokenizer, whitespace_spans};

/// The options the simple and whitespace tokenizers share, as OpenNLP's
/// `WhitespaceMode` and `AbstractTokenizer.setKeepNewLines` set them.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TokenizerOptions {
    /// Which characters are whitespace.
    pub whitespace: WhitespaceMode,
    /// Emit every line feed and carriage return as a token of its own.
    pub keep_newlines: bool,
}

impl TokenizerOptions {
    /// The options as a configuration hash, for provenance.
    pub(crate) fn config_hash(self) -> u64 {
        let mode = match self.whitespace {
            WhitespaceMode::Legacy => 0,
            WhitespaceMode::Unicode => 1,
        };
        fnv1a(&[mode, u8::from(self.keep_newlines)])
    }
}

/// The 64-bit FNV-1a hash of `bytes`: small, fixed and the same on every
/// platform, which is all a configuration hash needs.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options() {
        let o = TokenizerOptions::default();
        assert_eq!(o.whitespace, WhitespaceMode::Unicode);
        assert!(!o.keep_newlines);
    }

    #[test]
    fn config_hashes_differ_per_option() {
        let all = [
            TokenizerOptions { whitespace: WhitespaceMode::Unicode, keep_newlines: false },
            TokenizerOptions { whitespace: WhitespaceMode::Unicode, keep_newlines: true },
            TokenizerOptions { whitespace: WhitespaceMode::Legacy, keep_newlines: false },
            TokenizerOptions { whitespace: WhitespaceMode::Legacy, keep_newlines: true },
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.config_hash(), b.config_hash());
            }
        }
    }

    #[test]
    fn fnv1a_reference_values() {
        assert_eq!(fnv1a(b""), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xAF63_DC4C_8601_EC8C);
    }
}
