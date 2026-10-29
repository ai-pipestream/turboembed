//! Corpus readers and typed training samples with exact source spans.
//!
//! Every token, tag and lemma is a [`Field`] with its UTF-8 byte span, so
//! samples map back to the source text. Kept readers cover CoNLL-U with
//! multiword tokens, MASC, CoNLL-2002/2003, WNUT-17, MultiCoNER v2,
//! GermEval 2014, LitBank, brat and supporting sentence corpora. Errors
//! are typed and malformed input never supplies invented source spans.
//!
//! The crate also holds document export, evaluation and pinned corpus
//! recipes. MASC XML parsing is local, and no JVM, model archive or
//! constituency-parser dependency is used.

pub mod conll;
pub mod conllu;
pub mod doc_export;
pub mod encoding;
pub mod error;
pub mod evaluate;
pub mod legal;
pub mod masc;
pub mod ner;
pub mod recipes;
pub mod sample;
pub mod source;
pub mod vocabulary;
pub(crate) mod xml;

pub use encoding::{DecodeError, Encoding, decode};
pub use error::FormatError;
pub use sample::{
    ChunkSample, DependencySample, DocumentSample, LanguageSample, LemmaSample, Name, NameSample, Phrase, PosSample,
    SentenceSample, SplitMarker, TokenSample,
};
pub use source::{Field, Line, Lines, Piece};
