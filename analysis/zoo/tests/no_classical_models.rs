//! The catalogue and custom indices cannot re-enable retired model archives.

use turbo_zoo::{Index, ZooError};

#[test]
fn builtin_contains_no_opennlp_model_archives() {
    let index = Index::builtin();
    assert!(index.list().iter().all(|a| a.format.name() != "FORMAT_OPENNLP_MODEL"));
    assert!(index.get("opennlp-en-ud-ewt-sentence-1.3-2.5.4").is_err());
    assert!(index.get("turbo-es-ud-ancora-2.16-depparse-maxent-1").is_err());
    assert!(index.get("turbo-mul-tatoeba-langdetect-1").is_err());
}

#[test]
fn custom_index_cannot_offer_an_opennlp_model() {
    let builtin: serde_json::Value = serde_json::from_str(turbo_zoo::index::BUILTIN).unwrap();
    let mut artifact = builtin["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["format"] != "FORMAT_OPENNLP_MODEL")
        .unwrap()
        .clone();
    artifact["format"] = "FORMAT_OPENNLP_MODEL".into();
    let input = serde_json::json!({"zoo_version": 1, "artifacts": [artifact]});
    let error = Index::parse(&serde_json::to_vec(&input).unwrap()).unwrap_err();
    assert!(matches!(error, ZooError::Index(_)), "{error}");
    assert!(error.to_string().contains("FORMAT_OPENNLP_MODEL"), "{error}");
}
