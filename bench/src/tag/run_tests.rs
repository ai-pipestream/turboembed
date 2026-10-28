use super::*;
use turbo_postag::neural::bilstm_model::{UNKNOWN, char_vocabulary_from_strs};
use turbo_postag::neural::lstm::LstmLayer;

#[test]
fn native_bundle_load_preserves_source_identity_across_precisions() {
    let layer = |input| {
        LstmLayer::of_weights(input, 1, vec![vec![0.125; input]; 4], vec![vec![0.25]; 4], vec![0.0625; 4]).unwrap()
    };
    let model = BilstmPosModel::new(
        vec![(UNKNOWN.into(), 0), ("word".into(), 1)],
        char_vocabulary_from_strs([(UNKNOWN, 0), ("w", 1)]),
        vec!["NOUN".into(), "VERB".into()],
        vec![vec![0.25], vec![0.5]],
        vec![vec![0.125], vec![0.375]],
        layer(1),
        layer(1),
        layer(3),
        layer(3),
        None,
        None,
        vec![vec![0.125, 0.25], vec![0.5, 0.75]],
        vec![0.0625, 0.125],
        8,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let dir = std::env::temp_dir().join(format!("turbo-tag-native-bundle-{}", std::process::id()));
    assert!(!dir.exists());
    let source = turbo_bundle::postag::NativeSource { id: "turbo-native:tag-test".into(), revision: "v1".into() };
    turbo_bundle::postag::export(&model, &source, "Apache-2.0", &dir).unwrap();
    let expected_source = sha256_hex(&std::fs::read(dir.join("weights.safetensors")).unwrap());
    let expected_manifest = sha256_hex(&std::fs::read(dir.join(turbo_bundle::postag::MANIFEST)).unwrap());
    for (weights, name) in [(["f32"], "f32"), (["f16"], "f16")] {
        let (loaded, digest, manifest, selected) = load(&dir, &weights).unwrap();
        assert_eq!(digest, expected_source);
        assert_eq!(manifest, expected_manifest);
        assert_eq!(selected, name);
        assert_eq!(loaded.named_tensors(), model.named_tensors());
        assert_eq!(loaded.vocabularies(), model.vocabularies());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn archive_file_argument_is_rejected_before_reading() {
    let path = std::env::temp_dir().join(format!("turbo-tag-absent-{}", std::process::id()));
    assert!(!path.exists());
    let error = load(&path, &["f64", "f32"]).err().unwrap();
    assert!(error.contains("archive loading is retired"), "{error}");
}
