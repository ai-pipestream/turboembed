//! Group `bundle`, Rust layer: a bundle is verified before anything loads
//! (PLAN.md principle 6 and section 6). Every failure mode has its own code,
//! and none of them is a silent fallback.

use std::fs;

use turbo::abi::*;
use turbo::bundle::{sha256_bytes, Bundle};
use turbo::provider::ModelDesc;
use turbo_conformance::{assert_err, fixtures, BundleKind, Target};

fn load(t: &Target, dir: &std::path::Path) -> turbo::Result<std::sync::Arc<turbo::handles::Model>> {
    t.context().load_model(dir, &ModelDesc::default())
}

#[test]
fn bundle_missing_directory_is_not_found() {
    let t = Target::from_env();
    let scratch = fixtures::empty();
    let missing = scratch.path().join("not-a-bundle");
    assert_err!(Bundle::open(&missing), TURBO_E_BUNDLE_NOT_FOUND);
    assert_err!(load(&t, &missing), TURBO_E_BUNDLE_NOT_FOUND);
    // A file where a directory is expected is equally refused.
    let file = scratch.path().join("file.txt");
    fs::write(&file, b"not a bundle").expect("write");
    assert_err!(Bundle::open(&file), TURBO_E_BUNDLE_NOT_FOUND);
}

#[test]
fn bundle_missing_manifest_is_not_found() {
    let t = Target::from_env();
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    fs::remove_file(scratch.path().join("bundle.json")).expect("remove the manifest");
    assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_NOT_FOUND);
    assert_err!(load(&t, scratch.path()), TURBO_E_BUNDLE_NOT_FOUND);
}

#[test]
fn bundle_malformed_manifest_is_invalid() {
    let t = Target::from_env();
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    fs::write(scratch.path().join("bundle.json"), b"{ this is not json").expect("write");
    assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_INVALID);
    assert_err!(load(&t, scratch.path()), TURBO_E_BUNDLE_INVALID);
    // A manifest that is valid JSON but not a manifest is equally refused.
    fs::write(scratch.path().join("bundle.json"), b"[1, 2, 3]").expect("write");
    assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_INVALID);
}

#[test]
fn bundle_wrong_version_is_invalid() {
    let t = Target::from_env();
    for version in [1u32, 3, 99] {
        let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
        scratch.patch_manifest(|m| m["bundle_version"] = serde_json::json!(version));
        let e = assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_INVALID);
        assert!(e.message().contains(&version.to_string()), "the message must name the version: {}", e.message());
        assert_err!(load(&t, scratch.path()), TURBO_E_BUNDLE_INVALID);
    }
}

#[test]
fn bundle_tampered_artifact_is_integrity() {
    let t = Target::from_env();
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    let (format, path) = scratch.artifact_paths().into_iter().next().expect("the bundle lists an artifact");
    let mut body = fs::read(&path).expect("read the artifact");
    body.push(b'\n');
    fs::write(&path, &body).expect("tamper with the artifact");
    let e = assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_INTEGRITY);
    assert!(e.message().contains(&format) || e.message().contains("hashes to"), "{}", e.message());
    assert_err!(load(&t, scratch.path()), TURBO_E_BUNDLE_INTEGRITY);

    // Recording the new hash makes it load again: the check is the hash, not the name.
    scratch.patch_manifest(|m| {
        m["artifacts"][&format]["sha256"] = serde_json::json!(sha256_bytes(&body));
    });
    Bundle::open(scratch.path()).expect("a bundle whose hashes match opens");
}

#[test]
fn bundle_a_missing_listed_file_is_not_found() {
    let t = Target::from_env();
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    let (_, path) = scratch.artifact_paths().into_iter().next().expect("an artifact");
    fs::remove_file(&path).expect("remove the artifact");
    assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_NOT_FOUND);
    assert_err!(load(&t, scratch.path()), TURBO_E_BUNDLE_NOT_FOUND);
}

#[test]
fn bundle_path_escape_is_invalid() {
    let t = Target::from_env();
    for escape in ["../outside.bin", "/etc/passwd", "sub/../../outside.bin"] {
        let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
        let (format, _) = scratch.artifact_paths().into_iter().next().expect("an artifact");
        scratch.patch_manifest(|m| {
            m["artifacts"][&format]["path"] = serde_json::json!(escape);
        });
        let e = assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_INVALID);
        assert!(
            e.message().contains("inside the bundle") || e.message().contains(escape),
            "the message must explain the refusal: {}",
            e.message()
        );
        assert_err!(load(&t, scratch.path()), TURBO_E_BUNDLE_INVALID);
    }
}

#[test]
fn bundle_missing_contract_fields_are_invalid_per_kind() {
    let t = Target::from_env();
    // Embedders need dim, pooling and normalize; classifiers need labels;
    // every kind but the generic one needs max_seq.
    let cases: [(BundleKind, &str); 6] = [
        (BundleKind::Embedding, "dim"),
        (BundleKind::Embedding, "pooling"),
        (BundleKind::Embedding, "normalize"),
        (BundleKind::Embedding, "max_seq"),
        (BundleKind::Classifier, "labels"),
        (BundleKind::TokenClassifier, "labels"),
    ];
    for (kind, field) in cases {
        let scratch = fixtures::copy_of(&t.bundle(kind));
        scratch.patch_manifest(|m| {
            m["contract"].as_object_mut().expect("a contract object").remove(field);
        });
        let e = assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_INVALID);
        assert!(
            e.message().contains(field),
            "removing contract.{field} from a {} bundle must be reported by name: {}",
            kind.dir_name(),
            e.message()
        );
        assert_err!(load(&t, scratch.path()), TURBO_E_BUNDLE_INVALID);
    }
}

#[test]
fn bundle_inconsistent_contract_values_are_invalid() {
    let t = Target::from_env();
    // A Matryoshka dimension larger than the model's is a contradiction.
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    scratch.patch_manifest(|m| {
        let dim = m["contract"]["dim"].as_u64().expect("dim");
        m["contract"]["truncate_dims"] = serde_json::json!([dim + 1]);
    });
    assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_INVALID);

    // Unknown enumeration names in the contract are refused, never defaulted.
    for (field, value) in [("pooling", "sideways"), ("normalize", "l3"), ("aggregation", "sometimes")] {
        let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
        scratch.patch_manifest(|m| m["contract"][field] = serde_json::json!(value));
        let bundle = Bundle::open(scratch.path());
        match bundle {
            Ok(b) => {
                // Opening may defer the parse; reading the value must then fail.
                let parsed = match field {
                    "pooling" => b.pooling().err(),
                    "normalize" => b.normalize().err(),
                    _ => b.aggregation().err(),
                };
                let e = parsed.unwrap_or_else(|| panic!("contract.{field} = {value:?} was accepted"));
                assert_eq!(e.code(), TURBO_E_BUNDLE_INVALID);
            }
            Err(e) => assert_eq!(e.code(), TURBO_E_BUNDLE_INVALID),
        }
        assert_err!(load(&t, scratch.path()), TURBO_E_BUNDLE_INVALID);
    }
}

#[test]
fn bundle_unknown_task_kind_or_modality_is_invalid() {
    let t = Target::from_env();
    for (field, value) in [("task", "teleport"), ("kind", "oracle"), ("modality", "smell")] {
        let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
        scratch.patch_manifest(|m| m[field] = serde_json::json!(value));
        let e = assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_INVALID);
        assert!(e.message().contains(value), "the message must name the value: {}", e.message());
        assert_err!(load(&t, scratch.path()), TURBO_E_BUNDLE_INVALID);
    }
    // An empty model id is not an identity.
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    scratch.patch_manifest(|m| m["model_id"] = serde_json::json!("   "));
    assert_err!(Bundle::open(scratch.path()), TURBO_E_BUNDLE_INVALID);
}

#[test]
fn bundle_without_an_artifact_this_provider_can_load_is_no_artifact() {
    let t = Target::from_env();
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    let body = b"a format no provider in this build knows";
    fs::write(scratch.path().join("foreign.bin"), body).expect("write");
    scratch.patch_manifest(|m| {
        m["artifacts"] = serde_json::json!({
            "not_a_real_format": { "path": "foreign.bin", "sha256": sha256_bytes(body) }
        });
    });
    // The bundle itself is well formed: the manifest verifies.
    let bundle = Bundle::open(scratch.path()).expect("the manifest is valid");
    assert!(bundle.artifact("not_a_real_format").is_some());
    // The provider refuses it by name, with its own code.
    let e = assert_err!(load(&t, scratch.path()), TURBO_E_BUNDLE_NO_ARTIFACT);
    assert!(!e.message().is_empty(), "the refusal must explain what was missing");
    assert_err!(
        t.runtime.can_run(t.device_index, scratch.path(), bundle.task(), bundle.modality()),
        TURBO_E_BUNDLE_NO_ARTIFACT
    );
}

#[test]
fn bundle_contract_is_reported_verbatim_by_model_info() {
    let t = Target::from_env();
    for kind in BundleKind::ALL {
        let path = t.bundle(*kind);
        if !path.is_dir() {
            continue;
        }
        let bundle = Bundle::open(&path).expect("a committed bundle opens");
        let Ok(model) = load(&t, &path) else { continue };
        let info = model.info();
        let manifest = bundle.manifest();
        assert_eq!(info.model_id, manifest.model_id, "model_id must come from the bundle");
        assert_eq!(info.revision, manifest.revision, "revision must come from the bundle");
        assert_eq!(info.task, bundle.task());
        assert_eq!(info.kind, bundle.kind());
        assert_eq!(info.modality, bundle.modality());
        assert_eq!(info.labels, bundle.contract().labels, "labels must come from the bundle");
        assert_eq!(info.pooling, bundle.pooling().expect("pooling"), "pooling must come from the bundle");
        assert_eq!(info.normalize, bundle.normalize().expect("normalize"));
        assert_eq!(info.prefix_query, bundle.contract().prompts.query);
        assert_eq!(info.prefix_document, bundle.contract().prompts.document);
        if bundle.contract().max_seq != 0 {
            assert_eq!(info.max_seq, bundle.contract().max_seq, "max_seq must come from the bundle");
        }
        if bundle.contract().dim != 0 {
            assert_eq!(info.dim, bundle.contract().dim, "dim must come from the bundle");
        }
        if manifest.limits.max_batch != 0 {
            assert_eq!(info.max_batch, manifest.limits.max_batch, "max_batch must come from the bundle");
        }
        assert_eq!(info.provider_id, t.provider_id(), "the model must name the provider that loaded it");
        assert!(info.max_seq > 0 && info.max_batch > 0, "a loaded model declares usable limits");
    }
}

#[test]
fn bundle_every_committed_test_bundle_verifies() {
    let t = Target::from_env();
    for kind in BundleKind::ALL {
        let path = t.bundle(*kind);
        assert!(path.is_dir(), "the suite needs a `{}` bundle under {}", kind.dir_name(), t.bundle_root.display());
        let bundle = Bundle::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(bundle.manifest().bundle_version, 2);
        assert!(!bundle.manifest().model_id.is_empty());
        // Every listed file exists and hashes as recorded (Bundle::open checked
        // it; this asserts the manifest actually lists something).
        assert!(
            !bundle.manifest().artifacts.is_empty(),
            "{}: a bundle with no artifacts cannot be loaded by any provider",
            path.display()
        );
    }
}
