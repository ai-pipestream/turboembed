use std::path::Path;

use serde_json::{Value, json};

use super::agrees;
use crate::recipe::Recipe;

/// The pinned checkpoints' config.json, as upstream wrote them.
const MODERNBERT: &str = include_str!("../../tests/data/modernbert-base-config.json");
const NOMIC: &str = include_str!("../../tests/data/nomic-embed-text-v1.5-config.json");
/// The tiny fixtures' config.json, as transformers 5.17 saved them.
const TINY_MODERNBERT: &str = include_str!("../../../testdata/tiny-modernbert-reference/model/config.json");
const TINY_NOMIC: &str = include_str!("../../../testdata/tiny-nomic-reference/model/config.json");

fn parse(s: &str) -> Value {
    serde_json::from_str(s).unwrap()
}

fn recipe(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes").join(name);
    Recipe::load(&path).unwrap().manifest
}

fn modernbert() -> Value {
    recipe("modernbert-base.json")
}

fn nomic() -> Value {
    recipe("nomic-embed-text-v1.5.json")
}

#[test]
fn the_recipes_agree_with_the_pinned_configs() {
    assert_eq!(agrees(&parse(MODERNBERT), &modernbert()), Ok(()));
    assert_eq!(agrees(&parse(NOMIC), &nomic()), Ok(()));
    let mut bert = modernbert();
    bert["architecture"]["family"] = "FAMILY_BERT".into();
    assert_eq!(agrees(&json!({}), &bert), Ok(()), "another family is not checked");
}

#[test]
fn the_transformers_5_form_of_modernbert_s_config_agrees() {
    let mut c = parse(MODERNBERT);
    for key in ["global_attn_every_n_layers", "global_rope_theta", "local_rope_theta"] {
        c.as_object_mut().unwrap().remove(key).unwrap();
    }
    let types: Vec<&str> = (0..22).map(|l| if l % 3 == 0 { "full_attention" } else { "sliding_attention" }).collect();
    c["layer_types"] = json!(types);
    c["rope_parameters"] = json!({
        "full_attention": { "rope_theta": 160000.0, "rope_type": "default" },
        "sliding_attention": { "rope_theta": 10000.0, "rope_type": "default" }
    });
    assert_eq!(agrees(&c, &modernbert()), Ok(()));
    // The tiny fixture, saved by transformers 5.17 with both forms.
    let mut m = modernbert();
    let a = &mut m["architecture"];
    for (k, v) in [("layers", 3), ("hidden", 32), ("heads", 2), ("intermediate", 48), ("vocab_size", 1200)] {
        a[k] = v.into();
    }
    a["max_positions"] = 64.into();
    a["attention"]["window"] = 8.into();
    assert_eq!(agrees(&parse(TINY_MODERNBERT), &m), Ok(()));
}

#[test]
fn the_transformers_5_form_of_nomic_s_config_agrees() {
    let mut m = nomic();
    let a = &mut m["architecture"];
    for (k, v) in [("layers", 2), ("hidden", 32), ("heads", 2), ("intermediate", 48), ("max_positions", 64)] {
        a[k] = v.into();
    }
    assert_eq!(agrees(&parse(TINY_NOMIC), &m), Ok(()));
}

#[test]
fn nomic_is_served_to_n_positions_not_its_trained_length() {
    let c = parse(NOMIC);
    assert_eq!((c["max_position_embeddings"].as_u64(), c["n_positions"].as_u64()), (Some(2048), Some(8192)));
    let mut m = nomic();
    m["architecture"]["max_positions"] = 2048.into();
    let e = agrees(&c, &m).unwrap_err();
    assert!(e.contains("manifest.architecture.max_positions: 2048, the upstream config.json gives 8192"), "{e}");
}

/// Set (or, with `None`, remove) one key of `c`, a `/`-separated path.
fn mutate(c: &Value, path: &str, value: Option<Value>) -> Value {
    let mut c = c.clone();
    let (parent, key) = path.rsplit_once('/').unwrap();
    let obj = c.pointer_mut(parent).unwrap().as_object_mut().unwrap();
    match value {
        Some(v) => obj.insert(key.to_owned(), v),
        None => obj.remove(key),
    };
    c
}

#[test]
fn each_unsupported_modernbert_key_is_refused_by_name() {
    let c = parse(MODERNBERT);
    let five = mutate(
        &c,
        "/rope_parameters",
        Some(json!({ "full_attention": { "rope_theta": 160000.0 }, "sliding_attention": { "rope_theta": 10000.0 } })),
    );
    for (path, value, name) in [
        ("/model_type", Some(json!("bert")), "model_type"),
        ("/model_type", None, "model_type"),
        ("/hidden_activation", Some(json!("gelu_pytorch_tanh")), "hidden_activation"),
        ("/norm_bias", Some(json!(true)), "norm_bias"),
        ("/attention_bias", Some(json!(true)), "attention_bias"),
        ("/mlp_bias", Some(json!(true)), "mlp_bias"),
        ("/rope_scaling", Some(json!({ "rope_type": "linear", "factor": 2.0 })), "rope_scaling"),
        ("/norm_eps", None, "norm_eps"),
        ("/local_attention", None, "local_attention"),
        ("/sliding_window", Some(json!(128)), "sliding_window"),
        ("/global_attn_every_n_layers", None, "global_attn_every_n_layers or layer_types"),
        ("/global_attn_every_n_layers", Some(json!(0)), "global_attn_every_n_layers"),
        ("/global_rope_theta", None, "global_rope_theta or rope_parameters.full_attention.rope_theta"),
        ("/local_rope_theta", Some(json!("10000")), "local_rope_theta"),
        ("/layer_types", Some(json!(vec!["full_attention"; 22])), "layer_types[1]"),
        ("/layer_types", Some(json!(vec!["full_attention"; 3])), "layer_types"),
        ("/head_dim", Some(json!(32)), "head_dim"),
        ("/rope_parameters", Some(json!({ "rope_theta": 160000.0 })), "rope_parameters.rope_theta"),
    ] {
        let e = agrees(&mutate(&c, path, value), &modernbert()).unwrap_err();
        assert!(e.contains(&format!("upstream config.json: {name}")), "{path}: {e}");
    }
    for (path, value, name) in [
        ("/rope_parameters/full_attention/rope_theta", json!(10000.0), "global_rope_theta"),
        ("/rope_parameters/sliding_attention/rope_type", json!("yarn"), "rope_parameters.sliding_attention.rope_type"),
        ("/rope_parameters/full_attention/factor", json!(2.0), "rope_parameters.full_attention.factor"),
    ] {
        let e = agrees(&mutate(&five, path, Some(value)), &modernbert()).unwrap_err();
        assert!(e.contains(&format!("upstream config.json: {name}")), "{path}: {e}");
    }
}

/// Contrastors' grouped keys and partial rotary width may be written as
/// the values they imply for this model.
#[test]
fn nomic_options_at_their_implied_values_agree() {
    let c = mutate(&mutate(&parse(NOMIC), "/num_heads_kv", Some(json!(12))), "/rotary_head_dim", Some(json!(64)));
    assert_eq!(agrees(&c, &nomic()), Ok(()));
}

#[test]
fn each_unsupported_nomic_key_is_refused_by_name() {
    let c = parse(NOMIC);
    for (path, value, name) in [
        ("/model_type", Some(json!("bert")), "model_type"),
        ("/activation_function", Some(json!("gelu")), "activation_function"),
        ("/activation_function", None, "activation_function"),
        ("/hidden_act", Some(json!("gelu")), "hidden_act"),
        ("/rotary_emb_interleaved", Some(json!(true)), "rotary_emb_interleaved"),
        ("/rotary_emb_interleaved", None, "rotary_emb_interleaved"),
        ("/rotary_emb_fraction", Some(json!(0.5)), "rotary_emb_fraction"),
        ("/prenorm", Some(json!(true)), "prenorm"),
        ("/use_rms_norm", Some(json!(true)), "use_rms_norm"),
        ("/qkv_proj_bias", Some(json!(true)), "qkv_proj_bias"),
        ("/mlp_fc1_bias", Some(json!(true)), "mlp_fc1_bias"),
        ("/mlp_fc2_bias", Some(json!(true)), "mlp_fc2_bias"),
        ("/rotary_scaling_factor", Some(json!(2.0)), "rotary_scaling_factor"),
        ("/rotary_emb_scale_base", Some(json!(512)), "rotary_emb_scale_base"),
        ("/rope_scaling", Some(json!({ "type": "dynamic", "factor": 2.0 })), "rope_scaling"),
        ("/num_heads_kv", Some(json!(4)), "num_heads_kv"),
        ("/rotary_head_dim", Some(json!(32)), "rotary_head_dim"),
        ("/causal", Some(json!(true)), "causal"),
        ("/parallel_block", Some(json!(true)), "parallel_block"),
        ("/rotary_emb_base", Some(json!(10000)), "rotary_emb_base"),
        ("/rope_parameters/rope_type", Some(json!("dynamic")), "rope_parameters.rope_type"),
        ("/rope_parameters/factor", Some(json!(2.0)), "rope_parameters.factor"),
        ("/n_inner", Some(json!(2048)), "intermediate_size"),
        ("/layer_norm_epsilon", Some(json!(1e-5)), "layer_norm_epsilon"),
        ("/type_vocab_size", None, "type_vocab_size"),
        ("/head_dim", Some(json!(32)), "head_dim"),
    ] {
        let e = agrees(&mutate(&c, path, value), &nomic()).unwrap_err();
        assert!(e.contains(&format!("upstream config.json: {name}")), "{path}: {e}");
    }
}

#[test]
fn each_architecture_field_the_config_does_not_give_is_refused() {
    let fields = |extra: &[(&'static str, Value)]| {
        let mut f: Vec<(&str, Value)> = vec![
            ("/hidden", json!(1024)),
            ("/heads", json!(16)),
            ("/layers", json!(28)),
            ("/intermediate", json!(2624)),
            ("/vocab_size", json!(50000)),
            ("/max_positions", json!(4096)),
            ("/layer_norm_eps", json!(1e-6)),
            ("/token_types", json!(1)),
            ("/position_offset", json!(2)),
            ("/position_embedding", json!("POSITION_ABSOLUTE")),
            ("/mlp", json!("MLP_DENSE")),
            ("/norm", json!("NORM_RMS")),
            ("/rope/global_theta", json!(500000.0)),
        ];
        f.extend_from_slice(extra);
        f
    };
    let cases = [
        (
            parse(MODERNBERT),
            modernbert(),
            fields(&[
                ("/activation", json!("ACTIVATION_SILU")),
                ("/rope/local_theta", json!(1000.0)),
                ("/attention/window", json!(256)),
                ("/attention/global_every", json!(2)),
            ]),
        ),
        (
            parse(NOMIC),
            nomic(),
            fields(&[
                ("/activation", json!("ACTIVATION_GELU_ERF")),
                ("/rope/local_theta", json!(10000.0)),
                ("/attention", json!({ "window": 128, "global_every": 3 })),
            ]),
        ),
    ];
    for (c, m, fields) in cases {
        for (path, value) in fields {
            let mut m = m.clone();
            let (parent, key) = path.rsplit_once('/').unwrap();
            m.pointer_mut(&format!("/architecture{parent}")).unwrap()[key] = value;
            let e = agrees(&c, &m).unwrap_err();
            let name = path.trim_start_matches('/').replace('/', ".");
            assert!(e.contains(&format!("manifest.architecture.{name}:")), "{path}: {e}");
        }
    }
}
