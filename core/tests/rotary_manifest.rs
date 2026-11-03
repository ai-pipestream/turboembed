//! The manifest's rules for the rotary families (#1461), FAMILY_MODERNBERT
//! and FAMILY_NOMIC_BERT, on the tiny BERT bundle's manifest made into
//! one of theirs.

use serde_json::{Map, Value, json};
use turbo::manifest::{Family, Manifest, Mlp, Norm};

fn bert() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle/manifest.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn names(pairs: &[(&str, &str)]) -> Value {
    Value::Object(pairs.iter().map(|&(k, v)| (k.to_owned(), Value::from(v))).collect::<Map<_, _>>())
}

fn modernbert() -> Value {
    let mut m = bert();
    let a = &mut m["architecture"];
    a["family"] = "FAMILY_MODERNBERT".into();
    a["position_embedding"] = "POSITION_ROTARY".into();
    a["token_types"] = 0.into();
    a["layer_norm_eps"] = 1e-5.into();
    a["rope"] = json!({ "global_theta": 160000.0, "local_theta": 10000.0 });
    a["attention"] = json!({ "window": 128, "global_every": 3 });
    a["mlp"] = "MLP_GATED".into();
    a["norm"] = "NORM_LAYER_NO_BIAS".into();
    m["artifacts"][0]["tensor_names"] = names(&[
        ("word_embeddings", "model.embeddings.tok_embeddings.weight"),
        ("embeddings_ln_weight", "model.embeddings.norm.weight"),
        ("final_norm_weight", "model.final_norm.weight"),
        ("attn_ln_weight", "model.layers.{layer}.attn_norm.weight"),
        ("qkv_weight", "model.layers.{layer}.attn.Wqkv.weight"),
        ("attn_out_weight", "model.layers.{layer}.attn.Wo.weight"),
        ("ffn_ln_weight", "model.layers.{layer}.mlp_norm.weight"),
        ("ffn_in_weight", "model.layers.{layer}.mlp.Wi.weight"),
        ("ffn_out_weight", "model.layers.{layer}.mlp.Wo.weight"),
    ]);
    m
}

fn nomic() -> Value {
    let mut m = bert();
    let a = &mut m["architecture"];
    a["family"] = "FAMILY_NOMIC_BERT".into();
    a["position_embedding"] = "POSITION_ROTARY".into();
    a["activation"] = "ACTIVATION_SILU".into();
    a["rope"] = json!({ "global_theta": 1000.0 });
    a["mlp"] = "MLP_GATED".into();
    m["artifacts"][0]["tensor_names"] = names(&[
        ("word_embeddings", "embeddings.word_embeddings.weight"),
        ("token_type_embeddings", "embeddings.token_type_embeddings.weight"),
        ("embeddings_ln_weight", "emb_ln.weight"),
        ("embeddings_ln_bias", "emb_ln.bias"),
        ("qkv_weight", "encoder.layers.{layer}.attn.Wqkv.weight"),
        ("attn_out_weight", "encoder.layers.{layer}.attn.out_proj.weight"),
        ("attn_ln_weight", "encoder.layers.{layer}.norm1.weight"),
        ("attn_ln_bias", "encoder.layers.{layer}.norm1.bias"),
        ("ffn_gate_weight", "encoder.layers.{layer}.mlp.fc12.weight"),
        ("ffn_up_weight", "encoder.layers.{layer}.mlp.fc11.weight"),
        ("ffn_out_weight", "encoder.layers.{layer}.mlp.fc2.weight"),
        ("ffn_ln_weight", "encoder.layers.{layer}.norm2.weight"),
        ("ffn_ln_bias", "encoder.layers.{layer}.norm2.bias"),
    ]);
    m
}

fn parse(m: &Value) -> Manifest {
    Manifest::parse(&serde_json::to_vec(m).unwrap()).unwrap()
}

fn refusal(m: &Value) -> String {
    match Manifest::parse(&serde_json::to_vec(m).unwrap()) {
        Ok(_) => panic!("accepted {}", m["architecture"]),
        Err(e) => e.message,
    }
}

#[test]
fn both_rotary_architectures_parse_and_write_back_as_they_were() {
    for (m, family) in [(modernbert(), Family::Modernbert), (nomic(), Family::NomicBert)] {
        let parsed = parse(&m);
        let a = parsed.architecture.as_ref().unwrap();
        assert_eq!((a.family, a.mlp), (family, Mlp::Gated));
        let back: Value = serde_json::from_slice(&parsed.to_json()).unwrap();
        assert_eq!(back["architecture"], m["architecture"]);
    }
    let a = parse(&modernbert()).architecture.unwrap();
    let w = a.attention.unwrap();
    assert_eq!((w.window, w.global_every, a.norm), (128, 3, Norm::LayerNoBias));
    assert!(w.is_global(0) && !w.is_global(1) && !w.is_global(2) && w.is_global(3));
    assert_eq!(a.rope.unwrap().local_theta, Some(10000.0));
}

/// The defaults are not written, so every manifest made before the
/// rotary families writes back byte for byte.
#[test]
fn a_bert_manifest_writes_no_rotary_field() {
    let back: Value = serde_json::from_slice(&parse(&bert()).to_json()).unwrap();
    for f in ["rope", "attention", "mlp", "norm"] {
        assert!(back["architecture"].get(f).is_none(), "{f}");
    }
}

#[test]
fn a_bert_architecture_takes_none_of_the_rotary_parts() {
    for (field, v) in [
        ("rope", json!({ "global_theta": 10000.0 })),
        ("attention", json!({ "window": 128, "global_every": 3 })),
        ("mlp", json!("MLP_GATED")),
        ("norm", json!("NORM_LAYER_NO_BIAS")),
        ("activation", json!("ACTIVATION_SILU")),
    ] {
        let mut m = bert();
        m["architecture"][field] = v;
        assert!(refusal(&m).contains("FAMILY_MODERNBERT's and FAMILY_NOMIC_BERT's"), "{field}");
    }
    let mut m = bert();
    m["artifacts"][0]["tensor_names"]["qkv_weight"] = "fused.{layer}".into();
    assert!(refusal(&m).contains("qkv_weight is FAMILY_MODERNBERT's"));
    let mut m = bert();
    m["architecture"]["position_embedding"] = "POSITION_ROTARY".into();
    assert!(refusal(&m).contains("POSITION_ABSOLUTE"));
}

#[test]
fn a_modernbert_architecture_is_held_to_its_checkpoint() {
    let cases: [(&str, Value, &str); 9] = [
        ("position_embedding", json!("POSITION_ABSOLUTE"), "POSITION_ROTARY"),
        ("activation", json!("ACTIVATION_SILU"), "MLP_GATED with ACTIVATION_GELU_ERF"),
        ("mlp", json!("MLP_DENSE"), "MLP_GATED with ACTIVATION_GELU_ERF"),
        ("norm", json!("NORM_LAYER"), "NORM_LAYER_NO_BIAS"),
        ("norm", json!("NORM_RMS"), "no backend runs NORM_RMS yet"),
        ("token_types", json!(2), "0, no token type table"),
        ("rope", json!({ "global_theta": 160000.0 }), "a local_theta with attention"),
        ("rope", json!({ "global_theta": 1.0, "local_theta": 10000.0 }), "not a base above 1"),
        ("lookahead", json!(8), "no relative, lookahead or position_offset"),
    ];
    for (field, v, want) in cases {
        let mut m = modernbert();
        m["architecture"][field] = v;
        let e = refusal(&m);
        assert!(e.contains(want), "{field}: {e}");
    }
    let mut m = modernbert();
    m["architecture"].as_object_mut().unwrap().remove("attention");
    assert!(refusal(&m).contains("attention { window, global_every }"));
    for (window, every) in [(0, 3), (7, 3), (128, 0), (1024, 3)] {
        let mut m = modernbert();
        m["architecture"]["attention"] = json!({ "window": window, "global_every": every });
        m["architecture"]["max_positions"] = 512.into();
        assert!(refusal(&m).contains("architecture.attention: window"), "{window} {every}");
    }
}

#[test]
fn a_rotary_model_names_only_the_tensors_its_encoder_reads() {
    for (role, name) in [
        ("position_embeddings", "embeddings.position_embeddings.weight"),
        ("token_type_embeddings", "embeddings.token_type_embeddings.weight"),
        ("q_bias", "q.{layer}.bias"),
        ("ffn_gate_weight", "gate.{layer}"),
    ] {
        let mut m = modernbert();
        m["artifacts"][0]["tensor_names"][role] = name.into();
        let e = refusal(&m);
        assert!(e.contains(&format!("FAMILY_MODERNBERT has no {role}")), "{e}");
    }
    for (role, name) in [("position_embeddings", "p"), ("final_norm_weight", "f"), ("ffn_in_bias", "b.{layer}")] {
        let mut m = nomic();
        m["artifacts"][0]["tensor_names"][role] = name.into();
        let e = refusal(&m);
        assert!(e.contains(&format!("FAMILY_NOMIC_BERT has no {role}")), "{e}");
    }
    let mut m = nomic();
    m["artifacts"][0]["format"] = "FORMAT_ONNX".into();
    assert!(refusal(&m).contains("FAMILY_NOMIC_BERT is loaded from FORMAT_SAFETENSORS only"));
}

#[test]
fn a_nomic_architecture_is_held_to_its_checkpoint() {
    let cases: [(&str, Value, &str); 5] = [
        ("activation", json!("ACTIVATION_GELU_ERF"), "MLP_GATED with ACTIVATION_SILU"),
        ("norm", json!("NORM_LAYER_NO_BIAS"), "takes NORM_LAYER"),
        ("attention", json!({ "window": 128, "global_every": 3 }), "no attention"),
        (
            "rope",
            json!({ "global_theta": 1000.0, "local_theta": 10.0 }),
            "rope.local_theta: FAMILY_NOMIC_BERT takes none",
        ),
        ("token_types", json!(0), "architecture.token_types"),
    ];
    for (field, v, want) in cases {
        let mut m = nomic();
        m["architecture"][field] = v;
        let e = refusal(&m);
        assert!(e.contains(want), "{field}: {e}");
    }
    let mut m = nomic();
    m["architecture"].as_object_mut().unwrap().remove("rope");
    assert!(refusal(&m).contains("rope { global_theta }"));
    let mut m = nomic();
    m["architecture"]["max_positions"] = 65536.into();
    assert!(refusal(&m).contains("the longest FAMILY_NOMIC_BERT takes"));
}

#[test]
fn a_rotary_model_carries_no_head_yet() {
    let mut m = nomic();
    m["classifier"] = json!({ "kind": "CLASSIFIER_SEQUENCE", "labels": ["a", "b"] });
    assert!(refusal(&m).contains("carries no classifier, projection or QA head yet"));
}
