//! The function bodies that were in two files of one crate when the check
//! of `duplicates.rs` was written (#2459), each by the first place it is
//! written. Folding one into a single function removes its line here; no
//! line is ever added.

/// The number of entries below. It is lowered with each entry removed and
/// never raised: a new copy is fixed, not listed.
pub const KNOWN_AT_MOST: usize = 0;

pub const KNOWN: &[&str] = &[];
