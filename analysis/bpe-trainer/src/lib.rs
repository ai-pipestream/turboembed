//! BPE trainers: a vocabulary and its merges learned from a corpus, in
//! the forms turboembed's tokenizers and OpenNLP read. Training is a tool
//! run before a model is used, never part of encoding. docs/bpe-trainer.md
//! describes the three trainers, what each is compared with and where
//! each differs from its reference.
//!
//! - [`ByteLevelBpe`]: upstream `tokenizers`' `BpeTrainer` with GPT-2's
//!   byte-level alphabet, written as a `tokenizer.json` for the core's
//!   byte-level BPE.
//! - [`SentencePieceBpe`]: the sentencepiece library's BPE trainer with
//!   the identity normalization, written as a `.model` file for the
//!   core's SentencePiece BPE.
//! - [`OpenNlpBpe`]: OpenNLP's `BPETokenizerTrainer`, whose merges
//!   OpenNLP's `BPETokenizer` ([`OpenNlpTokenizer`]) applies.
//!
//! The three share one merge loop ([`merger`]), which keeps every pair's
//! count as merges are made rather than counting again, and orders ties
//! by a key each trainer gives, so the same corpus and settings always
//! give the same model. OpenNLP's default tie order, the JVM's, instead
//! counts every pair again before each merge as the JVM does
//! ([`TieOrder`]).

pub mod byte_level;
pub mod merger;
pub mod opennlp;
pub mod pieces;
mod proto;
pub mod sentencepiece;

use std::fmt;

pub use byte_level::{ByteLevelBpe, ByteLevelConfig, PostProcessor, SpecialToken};
pub use opennlp::{OpenNlpBpe, OpenNlpTokenizer, TieOrder};
pub use sentencepiece::{PieceKind, SentencePieceBpe, SentencePieceConfig, TrainedPiece};

/// One merge: the two pieces joined, and how often the pair occurred in
/// the corpus when it was merged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Merge {
    pub left: String,
    pub right: String,
    pub count: u64,
}

/// Why a model could not be trained.
#[derive(Clone, Debug, PartialEq)]
pub enum TrainError {
    /// OpenNLP: `numMerges` must be positive.
    NoMerges,
    /// A special token with no text.
    EmptySpecialToken,
    /// SentencePiece: no sentence is left to train on.
    EmptyCorpus,
    /// SentencePiece: `character_coverage` is not in (0, 1].
    Coverage(f32),
    /// SentencePiece: the vocabulary cannot hold the meta pieces and the
    /// characters kept.
    VocabTooSmall { asked: usize, least: usize },
    /// SentencePiece: the corpus has run out of pairs to merge before the
    /// vocabulary is full; `most` is the largest it can make.
    VocabTooLarge { asked: usize, most: usize },
    /// A line of a merges file that is not two symbols and a space.
    BadMergeLine { line: usize },
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainError::NoMerges => write!(f, "the number of merges must be positive"),
            TrainError::EmptySpecialToken => write!(f, "a special token is empty"),
            TrainError::EmptyCorpus => write!(f, "no sentence of the corpus is left to train on"),
            TrainError::Coverage(c) => write!(f, "character coverage {c} is not in (0, 1]"),
            TrainError::VocabTooSmall { asked, least } => {
                write!(f, "vocabulary size {asked} cannot hold the meta pieces and characters, {least} pieces")
            }
            TrainError::VocabTooLarge { asked, most } => {
                write!(f, "vocabulary size {asked} is more than the corpus gives; it gives at most {most}")
            }
            TrainError::BadMergeLine { line } => write!(f, "merges line {line}: expected two symbols and a space"),
        }
    }
}

impl std::error::Error for TrainError {}

/// `s` as a JSON string, escaped as serde_json escapes it.
pub(crate) fn json_string(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}
