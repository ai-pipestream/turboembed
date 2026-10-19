//! Coreference resolution for `turbo-analysis` documents, ported from
//! OpenNLP's `opennlp.tools.coref`.
//!
//! - [`layer`]: the typed [`Mentions`] layer (`opennlp:mentions`), the
//!   optional [`Speakers`] input layer, and the codecs that carry them,
//!   their chains (`Groups<Mentions>`) and their antecedent links
//!   (`Edges<Mentions, Mentions>`) through the document graph.
//! - [`CorefAnnotator`]: mention detection from entities, parse phrases or
//!   noun phrase chunks and pronouns; the precision-ranked sieves of Lee
//!   et al. (2013); or a [`CorefModel`] ranker or pair classifier.
//! - [`lexicon`]: the English word knowledge detection and the sieves use.
//! - [`WordVectors`] and [`TokenVectors`]: vector providers for the
//!   ranker's similarity and contextual span features.
//! - [`train`]: pair classifier and mention ranking training.
//! - [`scorer`]: MUC, B-cubed, CEAF and CoNLL scores.

mod annotator;
mod clusters;
mod detect;
mod features;
mod hash;
mod input;
pub mod layer;
pub mod lexicon;
mod mention;
mod model;
pub mod scorer;
mod sieve;
pub mod train;
mod validate;
mod vectors;

pub use annotator::{CorefAnnotator, DEFAULT_THRESHOLD, Resolution, sieve_tags};
pub use detect::UNTYPED;
pub use features::{DIFFERENCE, NEW_CHAIN, PRODUCT};
pub use input::CorefInput;
pub use layer::codec::{codecs, register_codecs};
pub use layer::{Agreement, Animacy, Gender, MentionKind, Mentions, Number, Origin, Person, Speakers};
pub use model::{
    APART, COMPONENT_NAME, CorefModel, LINK, MODEL_ENTRY, RANKING_PROPERTY, TOKEN_VECTOR_DIMENSION_PROPERTY,
};
pub use scorer::{CorefScorer, CorefScores, Score};
pub use sieve::Sieve;
pub use vectors::{MapWordVectors, TokenVectors, WordVectors};
