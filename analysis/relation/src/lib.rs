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

pub mod annotator;
pub mod export;
pub mod layer;
pub mod pattern;

pub use annotator::RelationAnnotator;
pub use layer::Relations;
pub use pattern::{RelationMention, RelationPattern, Step};
