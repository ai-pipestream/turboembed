//! A DeBERTa-v2 bundle loaded through the C interface on the CPU: its
//! vectors are turbo::deberta's encoder's (held to transformers in
//! tests/deberta.rs), mean pooled and normalized as the manifest says.

#![cfg(feature = "cpu")]

mod common;

use common::*;
use serde_json::json;
use turbo::deberta::{Config, Encoder};

const H: u64 = 8;
const I: u64 = 16;
const LAYERS: usize = 2;
const VOCAB: u64 = 30522;
const BUCKETS: u64 = 8;

/// A DeBERTa-v2 of the tiny BERT's width over the upstream vocabulary,
/// under Hugging Face's names, F32, with values from the element's place.
fn weights() -> Vec<Tensor> {
    let mut shapes: Vec<(String, Vec<u64>)> = vec![
        ("embeddings.word_embeddings.weight".into(), vec![VOCAB, H]),
        ("embeddings.LayerNorm.weight".into(), vec![H]),
        ("embeddings.LayerNorm.bias".into(), vec![H]),
        ("encoder.rel_embeddings.weight".into(), vec![2 * BUCKETS, H]),
        ("encoder.LayerNorm.weight".into(), vec![H]),
        ("encoder.LayerNorm.bias".into(), vec![H]),
    ];
    for l in 0..LAYERS {
        for (s, shape) in [
            ("attention.self.query_proj.weight", vec![H, H]),
            ("attention.self.query_proj.bias", vec![H]),
            ("attention.self.key_proj.weight", vec![H, H]),
            ("attention.self.key_proj.bias", vec![H]),
            ("attention.self.value_proj.weight", vec![H, H]),
            ("attention.self.value_proj.bias", vec![H]),
            ("attention.output.dense.weight", vec![H, H]),
            ("attention.output.dense.bias", vec![H]),
            ("attention.output.LayerNorm.weight", vec![H]),
            ("attention.output.LayerNorm.bias", vec![H]),
            ("intermediate.dense.weight", vec![I, H]),
            ("intermediate.dense.bias", vec![I]),
            ("output.dense.weight", vec![H, I]),
            ("output.dense.bias", vec![H]),
            ("output.LayerNorm.weight", vec![H]),
            ("output.LayerNorm.bias", vec![H]),
        ] {
            shapes.push((format!("encoder.layer.{l}.{s}"), shape));
        }
    }
    let mut k = 0u64;
    shapes
        .into_iter()
        .map(|(name, shape)| {
            let n: u64 = shape.iter().product();
            let data = (0..n)
                .flat_map(|_| {
                    k += 1;
                    (((k * 7919) % 2001) as f32 / 1000.0 - 1.0).to_le_bytes()
                })
                .collect();
            Tensor { name, dtype: "F32", shape, data }
        })
        .collect()
}

fn fixture(name: &str) -> Fixture {
    let mut m = model_manifest();
    let a = &mut m["architecture"];
    a["family"] = "FAMILY_DEBERTA_V2".into();
    a["position_embedding"] = "POSITION_RELATIVE".into();
    a["token_types"] = 0.into();
    a["layer_norm_eps"] = 1e-7.into();
    a["relative"] = json!({ "position_buckets": BUCKETS, "max_relative_positions": 32 });
    let layer = |s: &str| format!("encoder.layer.{{layer}}.{s}");
    m["artifacts"][0]["tensor_names"] = json!({
        "word_embeddings": "embeddings.word_embeddings.weight",
        "embeddings_ln_weight": "embeddings.LayerNorm.weight",
        "embeddings_ln_bias": "embeddings.LayerNorm.bias",
        "relative_embeddings": "encoder.rel_embeddings.weight",
        "relative_ln_weight": "encoder.LayerNorm.weight",
        "relative_ln_bias": "encoder.LayerNorm.bias",
        "q_weight": layer("attention.self.query_proj.weight"),
        "q_bias": layer("attention.self.query_proj.bias"),
        "k_weight": layer("attention.self.key_proj.weight"),
        "k_bias": layer("attention.self.key_proj.bias"),
        "v_weight": layer("attention.self.value_proj.weight"),
        "v_bias": layer("attention.self.value_proj.bias"),
        "attn_out_weight": layer("attention.output.dense.weight"),
        "attn_out_bias": layer("attention.output.dense.bias"),
        "attn_ln_weight": layer("attention.output.LayerNorm.weight"),
        "attn_ln_bias": layer("attention.output.LayerNorm.bias"),
        "ffn_in_weight": layer("intermediate.dense.weight"),
        "ffn_in_bias": layer("intermediate.dense.bias"),
        "ffn_out_weight": layer("output.dense.weight"),
        "ffn_out_bias": layer("output.dense.bias"),
        "ffn_ln_weight": layer("output.LayerNorm.weight"),
        "ffn_ln_bias": layer("output.LayerNorm.bias"),
    });
    let mut f = Fixture::new(name, m);
    f.weights("weights/model.safetensors", &weights());
    f
}

/// The encoder's mean of the live tokens' last states, L2 normalized.
fn expected(encoder: &Encoder, ids: &[i32]) -> Vec<f32> {
    let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let states = encoder.encode(&ids, &vec![true; ids.len()]).unwrap();
    let mut v = vec![0f64; H as usize];
    for t in states.data().chunks(H as usize) {
        for (d, &x) in v.iter_mut().zip(t) {
            *d += x as f64 / ids.len() as f64;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| (x / norm) as f32).collect()
}

#[test]
fn a_deberta_bundle_embeds_as_its_encoder_on_the_cpu() {
    let f = fixture("deberta-load");
    let loaded = f.load().unwrap();
    assert_eq!(loaded.info().dim, H as u32);
    let config = Config {
        hidden: H as usize,
        intermediate: I as usize,
        heads: 2,
        layers: LAYERS,
        vocabulary: VOCAB as usize,
        max_sequence: 512,
        position_buckets: BUCKETS as usize,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let bytes = std::fs::read(f.dir.join("weights/model.safetensors")).unwrap();
    let encoder = Encoder::from_safetensors(config, &bytes).unwrap();
    let texts = ["The quick brown fox jumps over the lazy dog.", "a", "Café naïve RÉSUMÉ, twice: Café naïve RÉSUMÉ"];
    let tok = f.open().unwrap();
    let session = Session::create(loaded.m, None).unwrap();
    let got = session.embed(&texts, None).unwrap();
    for (text, v) in texts.iter().zip(&got) {
        let want = expected(&encoder, &tok.row(text, None).unwrap());
        let error = v.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(error < 1e-5, "{text}: largest difference {error}");
    }
}

#[test]
fn a_deberta_bundle_without_its_relative_table_is_refused() {
    let mut f = fixture("deberta-no-relative");
    f.manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap().remove("relative_embeddings");
    let e = f.load().err().unwrap();
    assert!(e.is(turbo::status::BUNDLE_INVALID, "no relative_embeddings, which a DeBERTa-v2 encoder needs"), "{e:?}");
}
