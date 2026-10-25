//! Sentence boundaries from a SaT model (Segment any Text, wtpsplit) on
//! turboembed's own encoder: `SatAnnotator` writes `opennlp:sentences`
//! from a boundary score per token, read by the bundle's classifier from
//! the encoder's final states.
//!
//! - `cuts` turns the scores into sentences as wtpsplit's `split` does,
//!   then keeps every boundary on a grapheme cluster boundary.
//! - `head` is the classifier the bundle's manifest declares.
//!
//! docs/neural-annotators.md describes the design and what is not built
//! yet; analysis/sat/README.md the reference fixtures.

pub mod annotator;
pub mod cuts;
pub mod fallback;
pub mod head;

pub use annotator::{SatAnnotator, SatOptions, SatProvider, Scored};
pub use fallback::{rule_detector_with_sat, rule_splitter_with_sat};
pub use head::{BOUNDARY_LABEL, Head};

use turbo::bundle::Bundle;
use turbo_analysis::ModelRef;

/// The model a bundle holds, as every layer made with it names it: its
/// id, its revision and the SHA-256 of its manifest.
pub(crate) fn model_ref(bundle: &Bundle) -> ModelRef {
    let m = &bundle.manifest.model;
    ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: bundle.manifest_sha256.clone() }
}
