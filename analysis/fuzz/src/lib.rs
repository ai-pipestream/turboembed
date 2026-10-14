//! Differential fuzzing of the analysis crates against OpenNLP on the JVM.
//!
//! `turbo_goldens::fuzz` makes a seeded corpus of texts. The golden
//! generator (`turbo-goldens-tool`, in its container) runs every annotator
//! over them and writes goldens as it does for the checked in inputs.
//! This crate holds the Rust side: for every run the generator has
//! (`runs`), the annotators that reproduce it and the comparison the
//! crate's golden tests make; the walk over a tree of goldens that
//! compares each with the Rust result and checks the offset invariants of
//! every layer the Rust side wrote (`differential`); and the reduction of
//! a diverging text to a minimal one by delta debugging (`reduce`). The
//! `fuzz` subcommand of `turbo-goldens` drives the three. docs/analysis.md,
//! "Differential fuzzing", describes the whole.

pub mod differential;
pub mod reduce;
pub mod resources;
pub mod runs;

pub use differential::{Finding, Report, compare_tree, read_golden};
pub use reduce::{Oracle, reduce};
pub use resources::{Resources, Unavailable};
pub use runs::{Checked, Outcome, Run};
