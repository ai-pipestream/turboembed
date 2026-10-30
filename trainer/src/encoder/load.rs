//! An encoder read from a bundle: the manifest's architecture as the
//! geometry, and the raw F32 weights artifact's tensors by their roles.

use std::collections::BTreeMap;
use std::path::Path;

use turbo::bundle::Bundle;
use turbo::manifest::{Activation, Family, Format, GraphInput, Mlp, Norm, PositionEmbedding, TensorRole, role_name};
use turbo::safetensors::{Dtype, File};
use turbo::status::{Result, invalid};

use super::{ATTN_NORM, EMB_NORM, Encoder, FFN_IN, FFN_NORM, FFN_OUT, FINAL_NORM, Geometry, OUT, QKV, WORD, slot};
use crate::real::Real;

/// Every base slot's role, layer, slot and shape, for a vocabulary of
/// `v` words, hidden width `h` and intermediate width `i`.
pub(super) fn roles(layers: usize, v: u64, h: u64, i: u64) -> Vec<(TensorRole, Option<usize>, usize, Vec<u64>)> {
    use TensorRole::*;
    let mut out = vec![
        (WordEmbeddings, None, WORD, vec![v, h]),
        (EmbeddingsLnWeight, None, EMB_NORM, vec![h]),
        (FinalNormWeight, None, FINAL_NORM, vec![h]),
    ];
    for l in 0..layers {
        out.extend([
            (AttnLnWeight, Some(l), slot(l, ATTN_NORM), vec![h]),
            (QkvWeight, Some(l), slot(l, QKV), vec![3 * h, h]),
            (AttnOutWeight, Some(l), slot(l, OUT), vec![h, h]),
            (FfnLnWeight, Some(l), slot(l, FFN_NORM), vec![h]),
            (FfnInWeight, Some(l), slot(l, FFN_IN), vec![2 * i, h]),
            (FfnOutWeight, Some(l), slot(l, FFN_OUT), vec![h, i]),
        ]);
    }
    out
}

/// The name of `role` in layer `layer` under `names`.
pub(super) fn name(names: &BTreeMap<TensorRole, String>, role: TensorRole, layer: Option<usize>) -> Result<String> {
    let n = names.get(&role).ok_or_else(|| invalid(format!("encoder: the artifact names no {}", role_name(role))))?;
    Ok(match layer {
        Some(l) => n.replace("{layer}", &l.to_string()),
        None => n.clone(),
    })
}

impl<R: Real> Encoder<R> {
    /// The encoder of the ModernBERT bundle at `dir`, from its raw F32
    /// weights artifact, every file verified against the manifest.
    pub fn load(dir: &Path) -> Result<Self> {
        let bundle = Bundle::open(dir)?;
        let m = &bundle.manifest;
        let a = m.architecture.as_ref().ok_or_else(|| invalid("encoder: the bundle has no architecture"))?;
        if a.family != Family::Modernbert {
            return Err(invalid(format!("encoder: {:?} is not FAMILY_MODERNBERT", a.family)));
        }
        if a.mlp != Mlp::Gated
            || a.norm != Norm::LayerNoBias
            || a.activation != Activation::GeluErf
            || a.position_embedding != PositionEmbedding::Rotary
        {
            return Err(invalid("encoder: a ModernBERT has a gated GELU MLP, bias-free norms and rotary positions"));
        }
        let rope = a.rope.ok_or_else(|| invalid("encoder: no rope bases"))?;
        let geo = Geometry {
            layers: a.layers as usize,
            hidden: a.hidden as usize,
            heads: a.heads as usize,
            intermediate: a.intermediate as usize,
            eps: a.layer_norm_eps,
            window: a.attention.map_or(usize::MAX, |w| w.window as usize / 2),
            global_every: a.attention.map_or(0, |w| w.global_every as usize),
            max_positions: a.max_positions as usize,
            global_theta: rope.global_theta,
            local_theta: rope.local_theta.unwrap_or(rope.global_theta),
        };
        let vocab = a.vocab_size as usize;
        let art = m
            .artifacts
            .iter()
            .find(|x| x.format == Format::Safetensors && x.graph_input == GraphInput::TokenIds)
            .ok_or_else(|| invalid("encoder: the bundle has no raw weights artifact"))?;
        let path = art.files.first().ok_or_else(|| invalid("encoder: the weights artifact lists no file"))?;
        let bytes = bundle.read_verified(path)?;
        let file = File::parse(path, &bytes)?;
        let mut params = Vec::new();
        for (role, layer, s, want) in roles(geo.layers, vocab as u64, geo.hidden as u64, geo.intermediate as u64) {
            debug_assert_eq!(s, params.len());
            if role == TensorRole::AttnLnWeight && layer == Some(0) {
                params.push(Vec::new());
                continue;
            }
            let key = name(&art.tensor_names, role, layer)?;
            let t = file.get(&key, Dtype::F32, want.len())?;
            if t.shape != want {
                return Err(invalid(format!("{path}: {key} is {:?}, not {want:?}", t.shape)));
            }
            params.push(t.f32s().into_iter().map(|v| R::from_f64(v as f64)).collect());
        }
        Self::assemble(geo, vocab, params)
    }
}
