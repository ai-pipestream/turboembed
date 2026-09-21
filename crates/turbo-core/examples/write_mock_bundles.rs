//! Write the committed mock bundle fixtures under `testdata/bundles/mock/<kind>/`.
//!
//! Run from the repository root:
//! `cargo run -p turbo-core --example write_mock_bundles`
//!
//! The fixtures are deterministic, so re-running produces no diff unless the
//! mock contract changes; CI checks that with `git diff --exit-code`.

use std::path::PathBuf;

use turbo_core::mock::{write_mock_bundle, MockBundleKind};

fn main() {
    let root = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .and_then(|p| p.parent().and_then(|p| p.parent()).map(PathBuf::from))
        .expect("CARGO_MANIFEST_DIR is set by cargo run");
    let kinds = [
        ("embedding", MockBundleKind::Embedding),
        ("reranker", MockBundleKind::Reranker),
        ("classifier", MockBundleKind::Classifier),
        ("token-classifier", MockBundleKind::TokenClassifier),
        ("generative", MockBundleKind::Generative),
        ("generic", MockBundleKind::Generic),
    ];
    for (name, kind) in kinds {
        let dir = root.join("testdata").join("bundles").join("mock").join(name);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
        write_mock_bundle(&dir, kind).unwrap_or_else(|e| panic!("write {}: {e}", dir.display()));
        println!("wrote {}", dir.display());
    }
}
