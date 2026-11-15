//! Coreference resolution for `turbo-analysis` documents.
//!
//! - [`layer`]: the typed [`Mentions`] layer (`turbo:mentions`), the
//!   optional [`Speakers`] input layer, and the codecs that carry them,
//!   their chains (`Groups<Mentions>`) and their antecedent links
//!   (`Edges<Mentions, Mentions>`) through the document graph.
//! - [`CorefAnnotator`]: mention detection from entities, parse phrases or
//!   noun phrase chunks and pronouns, then the precision-ranked sieves of
//!   Lee et al. (2013).
//! - [`lexicon`]: the English word knowledge detection and the sieves use.
//! - [`scorer`]: MUC, B-cubed, CEAF and CoNLL scores.

mod annotator;
mod clusters;
mod detect;
mod hash;
mod input;
pub mod layer;
pub mod lexicon;
mod mention;
mod quotation;
mod quotes;
pub mod scorer;
mod sieve;
mod validate;

pub use annotator::{CorefAnnotator, Resolution, sieve_tags};
pub use detect::UNTYPED;
pub use input::CorefInput;
pub use layer::codec::{codecs, register_codecs};
pub use layer::{Agreement, Animacy, Gender, MentionKind, Mentions, Number, Origin, Person, Speakers};
pub use quotation::{QuotationAnnotator, QuotationConfig};
pub use scorer::{CorefScorer, CorefScores, Score};
pub use sieve::Sieve;
