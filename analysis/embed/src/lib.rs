//! The embedding model as analysis steps:
//!
//! - `SubwordAnnotator` writes the model's subword pieces of a text view
//!   (`Subwords`) with turboembed's own tokenizer, each with its exact
//!   span, so the chunker counts every chunk in the tokens the model will
//!   see.
//! - `VectorAnnotator` embeds windows of those pieces once and pools a
//!   vector per sentence and per chunk from the encoder's states
//!   (`Vectors<Sentences>`, `Vectors<Chunks>`), through the core's late
//!   pooling. `SentenceVectors`, `ChunkVectors` and `SentenceChunkVectors`
//!   are the steps that run it.
//! - `LevelVectors` runs it over `Levels` (chunk, section, document) and
//!   writes a local and a contextual vector per unit, as two named
//!   instances of `Vectors<Levels>` (`VectorKind::key`).
//!
//! This is the one analysis crate that depends on the core crate, `turbo`.

pub mod aspect;
pub mod centroids;
pub mod encoder;
pub mod heads;
pub mod levels;
pub mod rerank;
pub use rerank::{RerankAnnotator, Reranked, Shortlist};
pub mod qa;
pub use qa::{Answers, ExtractiveQaAnnotator, Question};
pub mod subwords;
pub mod vectors;
pub mod window;

pub use aspect::{BIO_ASPECT_OPINION_LABELS, LabeledAbsaDoc, LearnedAspectSentiment, bio_aspect_opinion_tagset};
pub use centroids::{CentroidNorm, LabelCentroids, add_categories, category_sentence_centroids, centroid};
pub use encoder::{Device, Encoder, EncoderInfo, EncoderOptions, Precision, Rows, normalize};
pub use heads::{
    AspectSpanHead, ChunkSpanHead, ClassifierHead, DenseLayer, HeadMode, HiddenActivation, IncivilityHeadAnnotator,
    OpinionSpanHead, PragmaticsHeadAnnotator, ScoringHead, SentenceSpanHead, VectorHeadAnnotator,
};
pub use levels::LevelVectors;
pub use subwords::{SubwordAnnotator, SubwordsProvider, align};
pub use vectors::{
    ChunkVectors, ContextualTokens, Input, Output, SentenceChunkVectors, SentenceVectors, SubwordVectors,
    VectorAnnotator, VectorOptions, VectorsProvider,
};

use turbo::bundle::Bundle;
use turbo_analysis::ModelRef;

/// The model a bundle holds, as every layer made with it names it: its
/// id, its revision and the SHA-256 of its manifest.
pub(crate) fn model_ref(bundle: &Bundle) -> ModelRef {
    let m = &bundle.manifest.model;
    ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: bundle.manifest_sha256.clone() }
}
