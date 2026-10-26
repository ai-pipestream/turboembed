//! The tokenizer (README rule 6): WordPiece, SentencePiece's Unigram or
//! byte-level BPE as the bundle describes it, built from the upstream
//! tokenizer.json, or SentencePiece's own tokenizer built from its
//! `.model` file, and checked against the reference's ids on every load
//! (docs/bundle.md loader rule 5).

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Range;

use serde_json::Value;
use turbo_unicode::normalize::{Form, aligned_chars};

use crate::bpe::Bpe;
use crate::bundle::Bundle;
use crate::manifest::{Normalizer, PromptRole, SpecialRole, Truncation};
use crate::safetensors::{self, Dtype};
use crate::sentencepiece::{Algorithm, SentencePiece};
use crate::status::{CAPACITY, Error, INVALID_ARGUMENT, Result, invalid};
use crate::unigram::Unigram;

/// FxHash, as rustc uses it: a multiply and a rotate per word. The vocab
/// is fixed and comes from the bundle, so the keys need no defence against
/// collisions; SipHash, the std default, was most of a short text's time.
#[derive(Default)]
pub(crate) struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for &w in words {
            self.add(u64::from_le_bytes(w));
        }
        let mut tail = 0u64;
        for (i, &b) in rest.iter().enumerate() {
            tail |= (b as u64) << (8 * i);
        }
        self.add(tail);
    }

    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

impl Fx {
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

/// How a static model turns a text into ids, beside the encoding itself
/// (docs/static.md): when truncating at the right, the text is cut to
/// max_tokens times `median_chars` characters and its encoding to
/// max_tokens; then the unknown token is dropped. The tokenizer file's
/// own truncation and padding are never used, so a text's ids do not
/// depend on its batch.
struct StaticRules {
    /// The median length of the vocabulary's entries in characters,
    /// rounded down.
    median_chars: usize,
}

type Vocab = HashMap<String, u32, BuildHasherDefault<Fx>>;

/// The models the core runs.
enum Kind {
    WordPiece {
        vocab: Vocab,
        /// The entries that start with the continuing prefix, under the
        /// rest of their text, so a piece inside a word is looked up as
        /// it stands in the text.
        continuing: Vocab,
        /// The longest entry in bytes, and the longest in `continuing`.
        longest: usize,
        longest_continuing: usize,
        normalizer: Normalizer,
        /// What the normalizer makes of each ASCII character.
        ascii: [u8; 128],
        continuing_prefix: String,
        max_chars_per_word: usize,
    },
    Unigram(Unigram),
    Bpe(Bpe),
    /// A SentencePiece `.model`, run as Google's library runs it.
    SentencePiece(Box<SentencePiece>),
}

/// A special token as it is matched in raw text, or in normalized text.
pub(crate) struct Special {
    pub(crate) content: String,
    pub(crate) id: i32,
    /// The whitespace before it is part of the match.
    pub(crate) lstrip: bool,
    /// Matched only with no word character on either side.
    single_word: bool,
}

pub struct Tokenizer {
    pub(crate) rerank: Option<crate::manifest::Rerank>,
    kind: Kind,
    /// Special tokens as they are matched in raw text: longest first.
    specials: Vec<Special>,
    /// Special tokens matched in each piece of normalized text, their
    /// content normalized: longest first.
    normalized_specials: Vec<Special>,
    /// The bytes a special token starts with, so text_ids looks for a
    /// match only where one can begin.
    special_starts: [bool; 256],
    /// The same for the normalized special tokens, in normalized text.
    normalized_starts: [bool; 256],
    /// The row layout: Some(id) for a special token, None for the text.
    template: Vec<Option<i32>>,
    truncation: Truncation,
    pub max_seq: u32,
    prefix_query: String,
    prefix_document: String,
    pub pad_id: i32,
    pub bos_id: i32,
    pub eos_id: i32,
    pub unk_id: i32,
    /// A static model's rules (manifest static_embedding), None for an
    /// encoder.
    stat: Option<StaticRules>,
    /// How ids are written back as text.
    decoder: Decoder,
    /// Each id's piece, for decode.
    pieces: Vec<Box<str>>,
    /// The ids of the special tokens, which decode can leave out.
    special_ids: Vec<i32>,
    /// SHA-256 of the tokenizer file.
    pub sha256: String,
    /// SHA-256 of the manifest the tokenizer was made from.
    pub manifest_sha256: String,
}

/// How to lay out one row. `max_tokens` includes the special tokens when
/// they are added.
#[derive(Debug, Clone, Copy)]
pub struct Encode {
    pub add_special_tokens: bool,
    pub truncation: Truncation,
    pub max_tokens: u32,
    pub prompt: PromptRole,
}

/// Where a token came from: the bytes `[start, end)` of the caller's
/// text, with any whitespace at either end left out unless the token is
/// nothing else. A token that came from no character of the text, a
/// special token the template adds or a token of the prompt's prefix,
/// is [`NOWHERE`].
pub type Span = [u32; 2];

/// The span of a token that came from no character of the text.
pub const NOWHERE: Span = [0, 0];

/// Where a row's tokens go: their ids alone, or their ids and spans. The
/// tokenizer is generic over it, so a row of ids alone makes no span and
/// pays nothing for them.
pub(crate) trait Sink {
    /// Whether spans are kept; when not, `span` is never called.
    const SPANS: bool;
    fn len(&self) -> usize;
    fn push(&mut self, id: i32, span: impl FnOnce() -> Span);
    /// The last token's span, widened to cover `span` as well.
    fn widen_last(&mut self, span: impl FnOnce() -> Span);
    fn truncate(&mut self, len: usize);
    fn drain(&mut self, r: Range<usize>);
    fn reverse_from(&mut self, at: usize);
    /// The tokens from `at` on whose id `keep` refuses are dropped.
    fn retain_from(&mut self, at: usize, keep: impl Fn(i32) -> bool);
    /// The tokens from `at` on, taken out: their ids and, when kept,
    /// their spans.
    fn split_off(&mut self, at: usize) -> (Vec<i32>, Vec<Span>);
    fn extend_with(&mut self, ids: &[i32], spans: &[Span]);
}

mod sink;
pub(crate) use sink::*;

mod load;

mod encode;

mod pieces;

mod decode;
use decode::*;

mod normalize;
pub use normalize::pre_words;
pub(crate) use normalize::*;

#[cfg(test)]
mod tests;

mod pair;
pub use pair::PairWindow;

#[cfg(test)]
mod pair_tests;
