//! The manifest's rules for FAMILY_DEBERTA_V2's architecture, on the tiny
//! BERT bundle's manifest made into a DeBERTa-v2 one.

use serde_json::{Value, json};
use turbo::manifest::Manifest;

fn bert() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle/manifest.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn deberta() -> Value {
    let mut m = bert();
    let a = &mut m["architecture"];
    a["family"] = "FAMILY_DEBERTA_V2".into();
    a["position_embedding"] = "POSITION_RELATIVE".into();
    a["token_types"] = 0.into();
    a["relative"] = json!({ "position_buckets": 256, "max_relative_positions": 512 });
    let names = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    names.remove("position_embeddings");
    names.remove("token_type_embeddings");
    names.insert("relative_embeddings".into(), "encoder.rel_embeddings.weight".into());
    names.insert("relative_ln_weight".into(), "encoder.LayerNorm.weight".into());
    names.insert("relative_ln_bias".into(), "encoder.LayerNorm.bias".into());
    m
}

fn refusal(m: &Value) -> String {
    match Manifest::parse(&serde_json::to_vec(m).unwrap()) {
        Ok(_) => panic!("accepted {}", m["architecture"]),
        Err(e) => e.message,
    }
}

#[test]
fn a_deberta_architecture_parses() {
    let m = Manifest::parse(&serde_json::to_vec(&deberta()).unwrap()).unwrap();
    let r = m.architecture.unwrap().relative.unwrap();
    assert_eq!((r.position_buckets, r.max_relative_positions), (256, 512));
}

#[test]
fn a_bert_architecture_takes_no_relative_positions() {
    let mut m = bert();
    m["architecture"]["relative"] = json!({ "position_buckets": 256, "max_relative_positions": 512 });
    assert!(refusal(&m).contains("POSITION_ABSOLUTE"));
    let mut m = bert();
    m["architecture"]["position_embedding"] = "POSITION_RELATIVE".into();
    assert!(refusal(&m).contains("POSITION_ABSOLUTE"));
}

#[test]
fn a_deberta_architecture_needs_its_relative_geometry() {
    let mut m = deberta();
    m["architecture"].as_object_mut().unwrap().remove("relative");
    assert!(refusal(&m).contains("architecture.relative"));
    let mut m = deberta();
    m["architecture"]["position_embedding"] = "POSITION_ABSOLUTE".into();
    assert!(refusal(&m).contains("POSITION_RELATIVE"));
    let mut m = deberta();
    m["architecture"]["token_types"] = 2.into();
    assert!(refusal(&m).contains("no token types"));
    for buckets in [0, 2, 7, 8192] {
        let mut m = deberta();
        m["architecture"]["relative"]["position_buckets"] = buckets.into();
        assert!(refusal(&m).contains("position_buckets"), "{buckets}");
    }
    for max in [129, 65537] {
        let mut m = deberta();
        m["architecture"]["relative"]["max_relative_positions"] = max.into();
        assert!(refusal(&m).contains("max_relative_positions"), "{max}");
    }
}

#[test]
fn a_deberta_model_is_raw_weights_only() {
    let mut m = deberta();
    m["artifacts"][0]["format"] = "FORMAT_ONNX".into();
    let e = refusal(&m);
    assert!(e.contains("FAMILY_DEBERTA_V2 is loaded from FORMAT_SAFETENSORS only"), "{e}");
}

#[test]
fn a_deberta_artifact_names_no_absolute_tables() {
    for role in ["position_embeddings", "token_type_embeddings"] {
        let mut m = deberta();
        m["artifacts"][0]["tensor_names"][role] = "embeddings.unused.weight".into();
        let e = refusal(&m);
        assert!(e.contains(&format!("FAMILY_DEBERTA_V2 has no {role}")), "{e}");
    }
}

#[test]
fn a_deberta_architecture_is_at_most_the_encoders_longest_sequence() {
    let mut m = deberta();
    m["architecture"]["max_positions"] = 8192.into();
    assert!(refusal(&m).contains("architecture.max_positions: 8192"));
}
