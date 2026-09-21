//! Group `bundle`, Rust layer, part two: manifest identity and escapes.
//!
//! Pins the fixes for H5 and the contract-parsing items in
//! `docs/reviews/2026-09-21-p0-p2.md`: a manifest has a hash the caller can
//! pin, a symlink cannot lead the loader outside the bundle, and a manifest
//! with unknown fields or an unparseable contract value never opens.

use std::fs;

use turbo::abi::*;
use turbo::bundle::Bundle;
use turbo_conformance::fixtures::{copy_of, Scratch};
use turbo_conformance::{BundleKind, Target};

fn embedding_bundle() -> Scratch {
    copy_of(&Target::from_env().bundle(BundleKind::Embedding))
}

#[test]
fn bundle_manifest_sha256_identifies_the_contract() {
    let a = embedding_bundle();
    let before = Bundle::open(a.path()).unwrap().manifest_sha256().to_string();
    assert_eq!(before.len(), 64, "hex SHA-256");
    assert_eq!(Bundle::open(a.path()).unwrap().manifest_sha256(), before, "stable across opens");
    a.patch_manifest(|m| m["contract"]["normalize"] = serde_json::json!("none"));
    let after = Bundle::open(a.path()).unwrap().manifest_sha256().to_string();
    assert_ne!(before, after, "a contract edit changes the manifest hash even though every file hash still matches");
}

#[test]
fn bundle_symlink_escaping_the_directory_is_rejected() {
    let scratch = embedding_bundle();
    let (format, inside) = scratch.artifact_paths().into_iter().next().expect("an artifact");
    // Move the artifact outside the bundle and point a symlink at it: every
    // hash still matches, only the location lies.
    let outside = tempfile::tempdir().unwrap();
    let moved = outside.path().join("moved");
    fs::rename(&inside, &moved).unwrap();
    std::os::unix::fs::symlink(&moved, &inside).unwrap();
    let e = Bundle::open(scratch.path()).unwrap_err();
    assert_eq!(e.code(), TURBO_E_BUNDLE_INVALID, "{e} (artifact `{format}`)");
    assert!(e.message().contains("outside the bundle directory"), "{e}");
}

#[test]
fn bundle_symlink_inside_the_directory_is_allowed() {
    let scratch = embedding_bundle();
    let (_, inside) = scratch.artifact_paths().into_iter().next().expect("an artifact");
    let real = inside.with_file_name("real-artifact");
    fs::rename(&inside, &real).unwrap();
    std::os::unix::fs::symlink(&real, &inside).unwrap();
    Bundle::open(scratch.path()).expect("a symlink that stays inside the bundle is fine");
}

#[test]
fn bundle_unknown_manifest_fields_are_rejected() {
    let scratch = embedding_bundle();
    scratch.patch_manifest(|m| m["contract"]["truncate_dimz"] = serde_json::json!([4]));
    let e = Bundle::open(scratch.path()).unwrap_err();
    assert_eq!(e.code(), TURBO_E_BUNDLE_INVALID, "{e}");
    assert!(e.message().contains("truncate_dimz"), "the unknown field is named: {e}");
}

#[test]
fn bundle_unparseable_contract_values_never_open() {
    for (field, value) in [("pooling", "banana"), ("normalize", "L2"), ("aggregation", "avg"), ("activation", "relu")] {
        let scratch = embedding_bundle();
        scratch.patch_manifest(|m| m["contract"][field] = serde_json::json!(value));
        let e = Bundle::open(scratch.path()).unwrap_err();
        assert_eq!(e.code(), TURBO_E_BUNDLE_INVALID, "{field} = {value}: {e}");
    }
}

#[test]
fn bundle_scored_models_must_declare_their_activation() {
    let t = Target::from_env();
    for kind in [BundleKind::Reranker, BundleKind::Classifier, BundleKind::TokenClassifier] {
        let scratch = copy_of(&t.bundle(kind));
        scratch.patch_manifest(|m| {
            m["contract"].as_object_mut().unwrap().remove("activation");
        });
        let e = Bundle::open(scratch.path()).unwrap_err();
        assert_eq!(e.code(), TURBO_E_BUNDLE_INVALID, "{kind:?}: {e}");
        assert!(e.message().contains("activation"), "{kind:?}: {e}");
    }
}
