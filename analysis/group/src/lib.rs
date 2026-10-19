//! Near-duplicate grouping, version links and boilerplate detection over
//! the units of a document (structure blocks, sentences, chunks) and
//! across a batch of documents.
//!
//! `Sketcher<L>` sketches each item of `L` from its words: an exact hash
//! of the word sequence, a shape with numbers folded, a SimHash and a
//! MinHash of its shingles (`Sketches<L>`). `NearDuplicates<L>` links
//! the items of one document from their sketches into `Groups<L>` and
//! `Edges<L, L>`; `cross_duplicates` does the same across a batch.
//! `BoilerplateDetector<L>` flags the items repeated across a batch or
//! inside one document (`Boilerplate<L>`, in turbo-analysis-layers), and
//! `BoilerplateProjector` carries structure-level flags to sentences or
//! chunks, so retrieval can exclude or down-weight them. Every hash is
//! fixed, so the same text sketches the same everywhere.

mod boilerplate;
mod dupes;
pub mod export;
pub mod hash;
mod link;
mod project;
mod sketch;

pub use boilerplate::{BoilerplateDetector, BoilerplateUnit};
pub use dupes::{CrossDuplicates, CrossLink, Member, NearDuplicates, cross_duplicates, duplicate_kinds};
pub use link::{DuplicateKind, Thresholds};
pub use project::BoilerplateProjector;
pub use sketch::{Scratch, Sketcher, Sketches, item_spans};
