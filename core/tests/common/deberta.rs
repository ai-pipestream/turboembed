//! Generated DeBERTa-v2 bundles under Hugging Face's names, F32, over the
//! upstream vocabulary, for the tests that load one: the tiny model of
//! tests/deberta_load.rs and one at GLiNER-small's size. Included by
//! `#[path]` beside `mod common`, whose fixtures it builds on.

#![allow(dead_code)]

use serde_json::json;

use crate::common::{Fixture, Tensor, long_text, model_manifest};

/// A DeBERTa-v2's shape, and how its weights are made: `scale` None gives
/// every value from the element's place in [-1, 1]; Some(s) gives the
/// matrices and embeddings that times s, LayerNorm weights near 1 and
/// small biases, so a wide, deep model saturates nothing.
#[derive(Clone, Copy)]
pub struct Shape {
    pub hidden: u64,
    pub heads: u64,
    pub intermediate: u64,
    pub layers: usize,
    pub buckets: u64,
    pub max_relative_positions: u64,
    pub scale: Option<f32>,
}

pub const VOCAB: u64 = 30522;

/// The tiny model: hidden 8, two heads, two layers, 8 buckets.
pub const TINY: Shape =
    Shape { hidden: 8, heads: 2, intermediate: 16, layers: 2, buckets: 8, max_relative_positions: 32, scale: None };

/// GLiNER-small's encoder: hidden 768, 12 heads, 6 layers, 256 buckets.
pub const SMALL: Shape = Shape {
    hidden: 768,
    heads: 12,
    intermediate: 3072,
    layers: 6,
    buckets: 256,
    max_relative_positions: 512,
    scale: Some(0.04),
};

/// Every tensor of `s` in safetensors order.
pub fn weights(s: &Shape) -> Vec<Tensor> {
    let (h, i) = (s.hidden, s.intermediate);
    let mut shapes: Vec<(String, Vec<u64>)> = vec![
        ("embeddings.word_embeddings.weight".into(), vec![VOCAB, h]),
        ("embeddings.LayerNorm.weight".into(), vec![h]),
        ("embeddings.LayerNorm.bias".into(), vec![h]),
        ("encoder.rel_embeddings.weight".into(), vec![2 * s.buckets, h]),
        ("encoder.LayerNorm.weight".into(), vec![h]),
        ("encoder.LayerNorm.bias".into(), vec![h]),
    ];
    for l in 0..s.layers {
        for (stem, shape) in [
            ("attention.self.query_proj.weight", vec![h, h]),
            ("attention.self.query_proj.bias", vec![h]),
            ("attention.self.key_proj.weight", vec![h, h]),
            ("attention.self.key_proj.bias", vec![h]),
            ("attention.self.value_proj.weight", vec![h, h]),
            ("attention.self.value_proj.bias", vec![h]),
            ("attention.output.dense.weight", vec![h, h]),
            ("attention.output.dense.bias", vec![h]),
            ("attention.output.LayerNorm.weight", vec![h]),
            ("attention.output.LayerNorm.bias", vec![h]),
            ("intermediate.dense.weight", vec![i, h]),
            ("intermediate.dense.bias", vec![i]),
            ("output.dense.weight", vec![h, i]),
            ("output.dense.bias", vec![h]),
            ("output.LayerNorm.weight", vec![h]),
            ("output.LayerNorm.bias", vec![h]),
        ] {
            shapes.push((format!("encoder.layer.{l}.{stem}"), shape));
        }
    }
    let mut k = 0u64;
    shapes
        .into_iter()
        .map(|(name, shape)| {
            let n: u64 = shape.iter().product();
            let (norm, bias) = (name.ends_with("LayerNorm.weight"), name.ends_with(".bias"));
            let mut data = Vec::with_capacity(n as usize * 4);
            for _ in 0..n {
                k += 1;
                let v = match s.scale {
                    None => ((k * 7919) % 2001) as f32 / 1000.0 - 1.0,
                    Some(scale) => {
                        // A 64-bit mix of the place, to [-1, 1).
                        let mut x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        x ^= x >> 31;
                        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        x ^= x >> 29;
                        let u = (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
                        if norm {
                            1.0 + 0.1 * u
                        } else if bias {
                            0.02 * u
                        } else {
                            scale * u
                        }
                    }
                };
                data.extend(v.to_le_bytes());
            }
            Tensor { name, dtype: "F32", shape, data }
        })
        .collect()
}

/// A bundle of `s` named `name`, embedding by the mean, L2 normalized, at
/// up to 512 tokens.
pub fn fixture(name: &str, s: &Shape) -> Fixture {
    let mut m = model_manifest();
    let a = &mut m["architecture"];
    a["family"] = "FAMILY_DEBERTA_V2".into();
    a["position_embedding"] = "POSITION_RELATIVE".into();
    a["token_types"] = 0.into();
    a["layer_norm_eps"] = 1e-7.into();
    a["hidden"] = s.hidden.into();
    a["heads"] = s.heads.into();
    a["intermediate"] = s.intermediate.into();
    a["layers"] = s.layers.into();
    a["relative"] = json!({ "position_buckets": s.buckets, "max_relative_positions": s.max_relative_positions });
    m["embed"]["dim"] = s.hidden.into();
    // A row of 512 tokens, with a reference case longer than that.
    m["embed"]["max_seq"] = 512.into();
    let longer = [long_text(), long_text()].join(" ");
    m["reference"]["cases"].as_array_mut().unwrap().push(json!({ "text": longer, "prompt_role": "PROMPT_NONE" }));
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
    f.weights("weights/model.safetensors", &weights(s));
    f
}
