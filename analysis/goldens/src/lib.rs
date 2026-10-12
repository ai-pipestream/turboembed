//! Golden layer files: what OpenNLP 3.x computes on the JVM for a fixed
//! input and model, written as typed layer files, and the comparison of
//! Rust layers against them. SHA-256 and SHA-512 (`sha`) check input
//! texts and model files.

pub mod hex;
pub mod sha;

use std::path::PathBuf;

pub use sha::{Sha256, Sha512, sha256, sha512};

/// The checked-in golden data, `analysis/goldens/data`: the input texts
/// under `inputs/<set>/` and a directory per annotator and model. Tests
/// in other crates find the goldens here.
pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}
