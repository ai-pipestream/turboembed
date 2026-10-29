//! Generated ModernBERT and nomic-bert bundles under Hugging Face's names,
//! F32, over the upstream vocabulary, for the tests that load one (#1461).
//! Included by `#[path]` beside `mod common`, whose fixtures it builds on.

#![allow(dead_code)]

use serde_json::{Value, json};

use crate::common::{Fixture, Tensor, model_manifest};

pub const VOCAB: u64 = 30522;
pub const HIDDEN: u64 = 8;
pub const HEADS: u64 = 2;
pub const INTERMEDIATE: u64 = 12;
/// One global layer and two local ones, with global_every 3.
pub const LAYERS: usize = 3;

/// Values from each element's place in [-1, 1].
fn tensors(shapes: Vec<(String, Vec<u64>)>) -> Vec<Tensor> {
    let mut k = 0u64;
    shapes
        .into_iter()
        .map(|(name, shape)| {
            let n: u64 = shape.iter().product();
            let mut data = Vec::with_capacity(n as usize * 4);
            for _ in 0..n {
                k += 1;
                data.extend((((k * 7919) % 2001) as f32 / 1000.0 - 1.0).to_le_bytes());
            }
            Tensor { name, dtype: "F32", shape, data }
        })
        .collect()
}

/// A ModernBERT's tensors; `first_norm` also writes layer 0's attention
/// norm, which the upstream model does not have.
pub fn modernbert_weights(first_norm: bool) -> Vec<Tensor> {
    let (h, i) = (HIDDEN, INTERMEDIATE);
    let mut shapes: Vec<(String, Vec<u64>)> = vec![
        ("model.embeddings.tok_embeddings.weight".into(), vec![VOCAB, h]),
        ("model.embeddings.norm.weight".into(), vec![h]),
        ("model.final_norm.weight".into(), vec![h]),
    ];
    for l in 0..LAYERS {
        if l > 0 || first_norm {
            shapes.push((format!("model.layers.{l}.attn_norm.weight"), vec![h]));
        }
        for (stem, shape) in [
            ("attn.Wqkv.weight", vec![3 * h, h]),
            ("attn.Wo.weight", vec![h, h]),
            ("mlp_norm.weight", vec![h]),
            ("mlp.Wi.weight", vec![2 * i, h]),
            ("mlp.Wo.weight", vec![h, i]),
        ] {
            shapes.push((format!("model.layers.{l}.{stem}"), shape));
        }
    }
    tensors(shapes)
}

pub fn nomic_weights() -> Vec<Tensor> {
    let (h, i) = (HIDDEN, INTERMEDIATE);
    let mut shapes: Vec<(String, Vec<u64>)> = vec![
        ("embeddings.word_embeddings.weight".into(), vec![VOCAB, h]),
        ("embeddings.token_type_embeddings.weight".into(), vec![2, h]),
        ("emb_ln.weight".into(), vec![h]),
        ("emb_ln.bias".into(), vec![h]),
    ];
    for l in 0..LAYERS {
        for (stem, shape) in [
            ("attn.Wqkv.weight", vec![3 * h, h]),
            ("attn.out_proj.weight", vec![h, h]),
            ("norm1.weight", vec![h]),
            ("norm1.bias", vec![h]),
            ("mlp.fc11.weight", vec![i, h]),
            ("mlp.fc12.weight", vec![i, h]),
            ("mlp.fc2.weight", vec![h, i]),
            ("norm2.weight", vec![h]),
            ("norm2.bias", vec![h]),
        ] {
            shapes.push((format!("encoder.layers.{l}.{stem}"), shape));
        }
    }
    tensors(shapes)
}

fn architecture(m: &mut Value, family: &str) {
    let a = &mut m["architecture"];
    a["family"] = family.into();
    a["position_embedding"] = "POSITION_ROTARY".into();
    a["hidden"] = HIDDEN.into();
    a["heads"] = HEADS.into();
    a["intermediate"] = INTERMEDIATE.into();
    a["layers"] = LAYERS.into();
    a["mlp"] = "MLP_GATED".into();
    m["embed"]["dim"] = HIDDEN.into();
}

fn modernbert_manifest(h: u64, i: u64) -> Value {
    let mut m = model_manifest();
    architecture(&mut m, "FAMILY_MODERNBERT");
    let a = &mut m["architecture"];
    a["hidden"] = h.into();
    a["intermediate"] = i.into();
    a["token_types"] = 0.into();
    a["layer_norm_eps"] = 1e-5.into();
    a["norm"] = "NORM_LAYER_NO_BIAS".into();
    a["rope"] = json!({ "global_theta": 160000.0, "local_theta": 10000.0 });
    a["attention"] = json!({ "window": 4, "global_every": 3 });
    m["embed"]["dim"] = h.into();
    let layer = |s: &str| format!("model.layers.{{layer}}.{s}");
    m["artifacts"][0]["tensor_names"] = json!({
        "word_embeddings": "model.embeddings.tok_embeddings.weight",
        "embeddings_ln_weight": "model.embeddings.norm.weight",
        "final_norm_weight": "model.final_norm.weight",
        "attn_ln_weight": layer("attn_norm.weight"),
        "qkv_weight": layer("attn.Wqkv.weight"),
        "attn_out_weight": layer("attn.Wo.weight"),
        "ffn_ln_weight": layer("mlp_norm.weight"),
        "ffn_in_weight": layer("mlp.Wi.weight"),
        "ffn_out_weight": layer("mlp.Wo.weight"),
    });
    m
}

/// A ModernBERT bundle named `name`: window 4 (two keys each side).
pub fn modernbert(name: &str) -> Fixture {
    let mut f = Fixture::new(name, modernbert_manifest(HIDDEN, INTERMEDIATE));
    f.weights("weights/model.safetensors", &modernbert_weights(false));
    f
}

fn nomic_manifest(h: u64, i: u64) -> Value {
    let mut m = model_manifest();
    architecture(&mut m, "FAMILY_NOMIC_BERT");
    let a = &mut m["architecture"];
    a["hidden"] = h.into();
    a["intermediate"] = i.into();
    a["activation"] = "ACTIVATION_SILU".into();
    a["rope"] = json!({ "global_theta": 1000.0 });
    m["embed"]["dim"] = h.into();
    let layer = |s: &str| format!("encoder.layers.{{layer}}.{s}");
    m["artifacts"][0]["tensor_names"] = json!({
        "word_embeddings": "embeddings.word_embeddings.weight",
        "token_type_embeddings": "embeddings.token_type_embeddings.weight",
        "embeddings_ln_weight": "emb_ln.weight",
        "embeddings_ln_bias": "emb_ln.bias",
        "qkv_weight": layer("attn.Wqkv.weight"),
        "attn_out_weight": layer("attn.out_proj.weight"),
        "attn_ln_weight": layer("norm1.weight"),
        "attn_ln_bias": layer("norm1.bias"),
        "ffn_gate_weight": layer("mlp.fc12.weight"),
        "ffn_up_weight": layer("mlp.fc11.weight"),
        "ffn_out_weight": layer("mlp.fc2.weight"),
        "ffn_ln_weight": layer("norm2.weight"),
        "ffn_ln_bias": layer("norm2.bias"),
    });
    m
}

/// A nomic-bert bundle named `name`.
pub fn nomic(name: &str) -> Fixture {
    let mut f = Fixture::new(name, nomic_manifest(HIDDEN, INTERMEDIATE));
    f.weights("weights/model.safetensors", &nomic_weights());
    f
}

/// Widths that every F16 Metal kernel (small, 32-wide and 64-wide tiles)
/// runs: hidden 32, intermediate 64.
pub const OVERFLOW_HIDDEN: u64 = 32;
pub const OVERFLOW_INTERMEDIATE: u64 = 64;

fn set_f32(t: &mut Tensor, index: usize, value: f32) {
    let at = index * 4;
    t.data[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn fill_f32(t: &mut Tensor, value: f32) {
    for c in t.data.as_chunks_mut::<4>().0 {
        *c = value.to_le_bytes();
    }
}

/// A ModernBERT bundle at width 32 / 64 whose weights all stay in [-30, 30]
/// while layer 0's gated MLP middle product `gelu(gate x) * (up x)` reaches
/// at least 90000 (past F16's 65504 limit) on every token (#1602).
pub fn overflow_modernbert(name: &str) -> Fixture {
    let (h, i) = (OVERFLOW_HIDDEN, OVERFLOW_INTERMEDIATE);
    let mut f = Fixture::new(name, modernbert_manifest(h, i));
    let mut shapes: Vec<(String, Vec<u64>)> = vec![
        ("model.embeddings.tok_embeddings.weight".into(), vec![VOCAB, h]),
        ("model.embeddings.norm.weight".into(), vec![h]),
        ("model.final_norm.weight".into(), vec![h]),
    ];
    for l in 0..LAYERS {
        if l > 0 {
            shapes.push((format!("model.layers.{l}.attn_norm.weight"), vec![h]));
        }
        for (stem, shape) in [
            ("attn.Wqkv.weight", vec![3 * h, h]),
            ("attn.Wo.weight", vec![h, h]),
            ("mlp_norm.weight", vec![h]),
            ("mlp.Wi.weight", vec![2 * i, h]),
            ("mlp.Wo.weight", vec![h, i]),
        ] {
            shapes.push((format!("model.layers.{l}.{stem}"), shape));
        }
    }
    let mut w = tensors(shapes);
    fill_f32(w.iter_mut().find(|t| t.name == "model.layers.0.mlp_norm.weight").unwrap(), 10.0);
    let wi = w.iter_mut().find(|t| t.name == "model.layers.0.mlp.Wi.weight").unwrap();
    fill_f32(wi, 0.0);
    let (hu, iu) = (h as usize, i as usize);
    for c in 0..hu {
        set_f32(wi, c * hu + c, 30.0);
        set_f32(wi, (iu + c) * hu + c, 30.0);
        set_f32(wi, (hu + c) * hu + c, -30.0);
        set_f32(wi, (iu + hu + c) * hu + c, -30.0);
    }
    f.weights("weights/model.safetensors", &w);
    f
}

/// A nomic-bert bundle at width 32 / 64 whose weights all stay in [-30, 30]
/// while layer 0's gated MLP middle product `silu(gate x) * (up x)` reaches
/// at least 90000 (past F16's 65504 limit) on every token (#1602).
pub fn overflow_nomic(name: &str) -> Fixture {
    let (h, i) = (OVERFLOW_HIDDEN, OVERFLOW_INTERMEDIATE);
    let mut f = Fixture::new(name, nomic_manifest(h, i));
    let mut shapes: Vec<(String, Vec<u64>)> = vec![
        ("embeddings.word_embeddings.weight".into(), vec![VOCAB, h]),
        ("embeddings.token_type_embeddings.weight".into(), vec![2, h]),
        ("emb_ln.weight".into(), vec![h]),
        ("emb_ln.bias".into(), vec![h]),
    ];
    for l in 0..LAYERS {
        for (stem, shape) in [
            ("attn.Wqkv.weight", vec![3 * h, h]),
            ("attn.out_proj.weight", vec![h, h]),
            ("norm1.weight", vec![h]),
            ("norm1.bias", vec![h]),
            ("mlp.fc11.weight", vec![i, h]),
            ("mlp.fc12.weight", vec![i, h]),
            ("mlp.fc2.weight", vec![h, i]),
            ("norm2.weight", vec![h]),
            ("norm2.bias", vec![h]),
        ] {
            shapes.push((format!("encoder.layers.{l}.{stem}"), shape));
        }
    }
    let mut w = tensors(shapes);
    fill_f32(w.iter_mut().find(|t| t.name == "encoder.layers.0.norm1.weight").unwrap(), 10.0);
    fill_f32(w.iter_mut().find(|t| t.name == "encoder.layers.0.norm1.bias").unwrap(), 0.0);
    let (hu, _iu) = (h as usize, i as usize);
    for stem in ["encoder.layers.0.mlp.fc11.weight", "encoder.layers.0.mlp.fc12.weight"] {
        let fc = w.iter_mut().find(|t| t.name == stem).unwrap();
        fill_f32(fc, 0.0);
        for c in 0..hu {
            set_f32(fc, c * hu + c, 30.0);
            set_f32(fc, (hu + c) * hu + c, -30.0);
        }
    }
    f.weights("weights/model.safetensors", &w);
    f
}
