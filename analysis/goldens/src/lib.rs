//! Golden layer files: what OpenNLP 3.x computes on the JVM for a fixed
//! input and model, written as typed layer files, and the comparison of
//! Rust layers against them.
//!
//! `Golden::read` parses a file into typed values (`format`), every error
//! naming its line. `matches_golden` compares a layer on a `Document` with
//! its golden layer and returns the first difference as a `Mismatch` a
//! person can read; `assert_matches_golden` panics with it. SHA-256 and
//! SHA-512 (`sha`) check input texts and model files. `load_document` and
//! `load_layer` (`load`) build a document from golden layers, under the
//! key name `golden`, so one Rust annotator can run on the JVM's own
//! input layers. `Dump::read` (`dump`) parses a decision point dump, the
//! contexts and probabilities of every maxent decision beside a golden.
//! `fuzz` makes a seeded corpus of texts that stress the annotators,
//! `invariants` checks the offsets of every layer a document holds, and
//! `expect` holds the JVM behaviours the golden tests of several crates
//! allow for. docs/analysis.md, "Golden files" and "Decision point dumps", describes
//! the formats.

pub mod compare;
pub mod dump;
pub mod escape;
pub mod expect;
pub mod format;
pub mod fuzz;
pub mod hex;
pub mod invariants;
pub mod load;
pub mod read;
pub mod sha;
pub mod write;

use std::path::{Path, PathBuf};

pub use compare::{
    Cx, GoldenCompare, InputFound, Item, Mismatch, Shape, Tolerance, assert_matches_golden, assert_matches_golden_key,
    check_input, check_shape, known_shapes, matches_golden, matches_golden_key, segmentation_shapes, shape_of,
};
pub use dump::{Dump, Outcome, Point, Unit};
pub use format::{
    AnnotatorRef, Column, ColumnData, ColumnSpec, ColumnType, ConfigEntry, Generator, Golden, GoldenAnchor,
    GoldenLayer, Header, Input, Jdk, JvmError, Model, OpenNlp, UnicodeVersion,
};
pub use invariants::{Violation, assert_invariants, check_document};
pub use load::{GOLDEN_KEY, GoldenLoad, LoadCx, LoadError, golden_key, load_document, load_layer};
pub use read::GoldenError;
pub use sha::{Sha256, Sha512, sha256, sha512};

/// The checked-in golden data, `analysis/goldens/data`: the input texts
/// under `inputs/<set>/` and a directory per annotator and model. Tests
/// in other crates find the goldens here.
pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}

/// A published OpenNLP model where `turbo-goldens fetch-models` puts it:
/// the directory in `TURBO_GOLDENS_MODELS`, or
/// `target/turbo-goldens/models` under the repository root. The models
/// are trained on CC BY-SA treebanks and are not checked in, so a test
/// that needs one says so and returns `None` when it is missing, unless
/// `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, which makes a missing model
/// fail it, as CI sets it. Give `TURBO_GOLDENS_MODELS` as an absolute
/// path: cargo runs each test in its own crate's directory, so a relative
/// one names a different directory for each crate.
pub fn published_model(file: &str) -> Option<PathBuf> {
    let dir = match std::env::var_os("TURBO_GOLDENS_MODELS") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/turbo-goldens/models"),
    };
    let path = dir.join(file);
    if path.is_file() {
        Some(path)
    } else if std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1") {
        panic!("{} not found and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1", path.display());
    } else {
        eprintln!("{} not found: skipping the test that needs it", path.display());
        None
    }
}
