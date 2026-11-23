//! The function bodies that were in two files of one crate when the check
//! of `duplicates.rs` was written (#2459), each by the first place it is
//! written. Folding one into a single function removes its line here; no
//! line is ever added.

/// The number of entries below. It is lowered with each entry removed and
/// never raised: a new copy is fixed, not listed.
pub const KNOWN_AT_MOST: usize = 12;

pub const KNOWN: &[&str] = &[
    "analysis/doccat/src/categorizer.rs::all_results",
    "analysis/encoding/src/detect/cjk/chinese.rs::maybe_set_as_pending",
    "analysis/namefind/src/consensus.rs::compute",
    "analysis/train-components/src/chunk.rs::package_chunker_model",
    "analysis/train-components/src/sentdetect.rs::package_sentence_model",
    "analysis/train-recipes/src/litbank.rs::files",
    "core/src/cpu/deberta.rs::config",
    "core/src/ffi.rs::new_error",
    "core/src/levelzero.rs::device_count",
    "core/src/qa/session.rs::drop",
    "trainer/src/lemma.rs::language_row",
    "trainer/src/relation.rs::new",
];
