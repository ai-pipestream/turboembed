//! Golden layer files: what OpenNLP 3.x computes on the JVM for a fixed
//! input and model, written as typed layer files, and the comparison of
//! Rust layers against them.
//!
//! `Golden::read` parses a file into typed values (`format`), every error
//! naming its line. `matches_golden` compares a layer on a `Document` with
//! its golden layer and returns the first difference as a `Mismatch` a
//! person can read; `assert_matches_golden` panics with it. SHA-256 and
//! SHA-512 (`sha`) check input texts and model files. docs/analysis.md,
//! "Golden files", describes the format.

pub mod compare;
pub mod escape;
pub mod format;
pub mod hex;
pub mod read;
pub mod sha;
pub mod write;

use std::path::PathBuf;

pub use compare::{
    Cx, GoldenCompare, Item, Mismatch, Shape, Tolerance, assert_matches_golden, assert_matches_golden_key,
    known_shapes, matches_golden, matches_golden_key, shape_of,
};
pub use format::{
    AnnotatorRef, Column, ColumnData, ColumnSpec, ColumnType, ConfigEntry, Generator, Golden, GoldenAnchor,
    GoldenLayer, Header, Input, Jdk, Model, OpenNlp, UnicodeVersion,
};
pub use read::GoldenError;
pub use sha::{Sha256, Sha512, sha256, sha512};

/// The checked-in golden data, `analysis/goldens/data`: the input texts
/// under `inputs/<set>/` and a directory per annotator and model. Tests
/// in other crates find the goldens here.
pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}
