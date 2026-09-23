//! Write the committed mock bundle fixtures under `testdata/bundles/mock/<kind>/`
//! and the manifest of the MiniLM tokenizer bundle under
//! `testdata/bundles/minilm-tokenizer/` (its `tokenizer.json` is committed).
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
    write_tokenizer_bundle(&root.join("testdata").join("bundles").join("minilm-tokenizer"));
}

/// A tokenizer-only bundle: no model artifact, just the pinned
/// `sentence-transformers/all-MiniLM-L6-v2` tokenizer.json and its contract.
fn write_tokenizer_bundle(dir: &std::path::Path) {
    use turbo_core::bundle::{sha256_file, MANIFEST_NAME};
    let tok = dir.join("tokenizer.json");
    let sha = sha256_file(&tok).unwrap_or_else(|e| panic!("hash {}: {e}", tok.display()));
    let manifest = serde_json::json!({
        "bundle_version": 2,
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "license": "Apache-2.0",
        "task": "tokenize",
        "kind": "embedding",
        "modality": "text",
        "family": "bert",
        "tokenizer": { "kind": "wordpiece", "files": { "tokenizer.json": { "path": "tokenizer.json", "sha256": sha } } },
        "contract": { "pooling": "mean", "normalize": "l2", "max_seq": 256, "dim": 384, "prompts": { "query": "", "document": "" } },
        "artifacts": {},
        "limits": { "max_batch": 32 }
    });
    let text = serde_json::to_string_pretty(&manifest).expect("serialize");
    std::fs::write(dir.join(MANIFEST_NAME), text).unwrap_or_else(|e| panic!("write {}: {e}", dir.display()));
    turbo_core::Bundle::open(dir).unwrap_or_else(|e| panic!("verify {}: {e}", dir.display()));
    println!("wrote {}", dir.display());
}
