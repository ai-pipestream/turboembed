//! SentencePiece's Unigram model as upstream `tokenizers` runs it, from
//! the upstream tokenizer.json: the precompiled character map it carries
//! (SentencePiece's own normalizer, `nmt_nfkc` for XLM-RoBERTa), the
//! collapse of repeated spaces, the metaspace in place of each space and
//! in front of each piece of text, the cut into words at each metaspace,
//! and each word's segmentation into the pieces of highest total score.
//! A character no piece covers is the unknown token, and a run of them
//! is one unknown token, as upstream fuses them.

use serde_json::Value;
use unicode_segmentation::UnicodeSegmentation;

use crate::manifest;
use crate::status::{Result, invalid};
use crate::tokenizer::{Sink, Span, covering};

/// Upstream's penalty under the lowest score for a character no piece
/// covers.
const UNK_PENALTY: f64 = 10.0;

pub struct Unigram {
    /// Each piece, by id.
    pieces: Vec<Box<str>>,
    /// Every piece, for the lattice's walk from each position.
    trie: Trie,
    /// Each piece's log probability, by id.
    scores: Vec<f64>,
    unk_score: f64,
    /// Maximum magnitude used to rescale only lattices that could overflow.
    max_score: f64,
    unk_id: u32,
    charsmap: Option<Charsmap>,
    collapse_spaces: bool,
    space_punctuation: bool,
    collapse_whitespace: bool,
    strip: bool,
    whole_text: bool,
    metaspace: char,
    add_prefix_space: bool,
}

mod added;
mod base64;
mod charsmap;
mod encode;
mod lattice;
mod parse;
#[cfg(test)]
mod tests;
mod trie;
use base64::base64;
use charsmap::Charsmap;
pub(crate) use trie::Trie;
impl Unigram {
    pub fn vocab_size(&self) -> u32 {
        self.scores.len() as u32
    }

    /// Every piece of the vocabulary.
    pub fn pieces(&self) -> impl Iterator<Item = &str> {
        self.pieces.iter().map(|p| &**p)
    }

    /// The id of `piece`, if it is one.
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.trie.get(piece.as_bytes())
    }
}
#[cfg(test)]
mod boundary_tests;
