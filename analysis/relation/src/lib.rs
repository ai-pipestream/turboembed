//! Dependency-path relation extraction: declarative
//! [`RelationPattern`] rules matched against the shortest
//! dependency path between entity head words, writing typed
//! relation mentions (`opennlp:relations`) and the same triples
//! as entity-to-entity edges.
//!
//! A pattern is a path like `<nsubj >obj` (up steps from the
//! subject's head, then down steps to the object's head), an
//! optional pivot trigger, and optional restrictions on the
//! entity kinds and the pivot's part of speech or lemma. An
//! entity head is the first overlapping word whose dependency
//! head leaves the entity (or is a root). The annotator is
//! generic over the token and syntactic-word anchors, like the
//! dependency parser it builds on, and reads sentences, the
//! anchor and token layers, dependencies, entities, and
//! optionally POS tags and lemmas.
//!
//! [`zeroshot`] extracts relations for labels the caller supplies at
//! request time instead, scoring mention pairs with a model behind
//! [`RelationScorer`], and writes the same layers.

pub mod annotator;
pub mod defaults;
pub mod discover;
mod evidence;
pub mod explore;
pub mod export;
pub mod layer;
pub mod paths;
pub mod pattern;
mod prepare;
pub mod zeroshot;

pub use annotator::RelationAnnotator;
pub use discover::RelationPathAnnotator;
pub use explore::{DistantSupervisionMiner, MinedRule, RelationPathExplorer, Seed, SignatureKey, SignatureRow};
pub use layer::Relations;
pub use paths::{DependencyPath, RelationPaths};
pub use pattern::{RelationMention, RelationPattern, Step};
pub use zeroshot::{
    LexicalRelationScorer, Mention, RelationError, RelationLabel, RelationScorer, ZeroShotRelationAnnotator,
    ZeroShotRelationConfig,
};
