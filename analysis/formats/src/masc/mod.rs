//! MASC text and GrAF stand-off annotation readers with exact source spans.
//!
//! A document header names its text, segmentation quarks, sentence regions,
//! Penn tokens and named entities. Annotation offsets are UTF-16 code units;
//! [`MascDocument::span`] converts them to checked UTF-8 byte spans, refusing
//! out-of-bounds offsets and offsets inside a surrogate pair.
//!
//! [`MascCorpus`] walks documents deterministically. [`MascDocument::parse`]
//! reads one document, and [`samples`] streams typed part-of-speech, token,
//! sentence and name samples whose fields borrow the original text. Empty
//! token sentences are skipped without discarding later documents.
//!
//! Optional noun and verb chunks are read by [`MascCorpus::read_with_chunks`].
//! [`MascCorpus::read_with_logical`] also reads logical markup, and
//! [`MascDocument::structure`] projects document, heading and paragraph blocks
//! into a `Structure` layer over the source text.
//!
//! Named entities preserve their complete token ranges. Overlapping display
//! names keep the longest annotation, while [`MascSentence::entities`] retains
//! the original token membership. Entities outside all kept sentences remain
//! available through [`MascDocument::unplaced_entities`].
//!
//! Test documents are synthetic fixtures in the MASC layout; no MASC corpus
//! text is committed.

mod chunks;
mod corpus;
mod document;
mod error;
mod logical;
pub mod samples;
pub(crate) mod text;

pub use chunks::{ChunkSkip, MascChunk, MascChunkFiles, MascChunkKind, MascChunkRead, MascChunkReader};
pub use corpus::{
    ChunkFiles, MascCorpus, SkippedDocument, read_document, read_document_with, read_document_with_logical,
};
pub use document::{
    MascDocument, MascEntity, MascFiles, MascName, MascSentence, MascToken, MascUnplacedEntity, MascWarning, Quark,
    Unplaced,
};
pub use error::{AnnotationFile, MascError, MascStream};
pub use logical::MascLogical;
pub use samples::{MascNameReader, MascPosReader, MascSample, MascSentenceReader, MascSentenceSample, MascTokenReader};
