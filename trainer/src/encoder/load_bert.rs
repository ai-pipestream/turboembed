//! BERT and RoBERTa weights packed into the trainer's projection slots.

use turbo::bundle::Bundle;
use turbo::manifest::{Activation, Format, GraphInput, Mlp, Norm, PositionEmbedding, TensorRole as T};
use turbo::safetensors::{Dtype, File};
use turbo::status::{Result, invalid};

use crate::real::Real;

use super::load::name;
use super::{Encoder, Geometry};

pub(super) fn load<R: Real>(bundle: &Bundle) -> Result<Encoder<R>> {
    let m = &bundle.manifest;
    let a = m.architecture.as_ref().expect("the caller checked the architecture");
    if a.mlp != Mlp::Dense || a.norm != Norm::Layer || a.position_embedding != PositionEmbedding::Absolute {
        return Err(invalid("encoder: BERT needs a dense MLP, biased layer norms and absolute positions"));
    }
    if a.activation != Activation::GeluErf || a.lookahead.is_some() {
        return Err(invalid("encoder: BERT training needs erf GELU and full attention"));
    }
    let max_positions = (a.max_positions as usize).saturating_sub(a.position_offset as usize);
    let geo = Geometry {
        layers: a.layers as usize,
        hidden: a.hidden as usize,
        heads: a.heads as usize,
        intermediate: a.intermediate as usize,
        eps: a.layer_norm_eps,
        window: usize::MAX,
        global_every: 0,
        max_positions,
        global_theta: 10_000.0,
        local_theta: 10_000.0,
    };
    let art = m
        .artifacts
        .iter()
        .find(|x| x.format == Format::Safetensors && x.graph_input == GraphInput::TokenIds)
        .ok_or_else(|| invalid("encoder: the bundle has no raw weights artifact"))?;
    let path = art.files.first().ok_or_else(|| invalid("encoder: the weights artifact lists no file"))?;
    let bytes = bundle.read_verified(path)?;
    let file = File::parse(path, &bytes)?;
    let get = |role: T, layer: Option<usize>, shape: &[u64]| -> Result<Vec<R>> {
        let key = name(&art.tensor_names, role, layer)?;
        let tensor = file.get(&key, Dtype::F32, shape.len())?;
        if tensor.shape != shape {
            return Err(invalid(format!("{path}: {key} is {:?}, not {shape:?}", tensor.shape)));
        }
        Ok(tensor.f32s().into_iter().map(|v| R::from_f64(v as f64)).collect())
    };
    let h = geo.hidden as u64;
    let i = geo.intermediate as u64;
    let mut params = Vec::new();
    let mut embeddings = get(T::WordEmbeddings, None, &[a.vocab_size as u64, h])?;
    embeddings.extend(get(T::PositionEmbeddings, None, &[a.max_positions as u64, h])?);
    embeddings.extend(get(T::TokenTypeEmbeddings, None, &[a.token_types as u64, h])?);
    params.push(embeddings);
    params.push([get(T::EmbeddingsLnWeight, None, &[h])?, get(T::EmbeddingsLnBias, None, &[h])?].concat());
    params.push(Vec::new());
    for l in 0..geo.layers {
        let layer = Some(l);
        params.push([get(T::AttnLnWeight, layer, &[h])?, get(T::AttnLnBias, layer, &[h])?].concat());
        let mut qkv = get(T::QWeight, layer, &[h, h])?;
        qkv.extend(get(T::KWeight, layer, &[h, h])?);
        qkv.extend(get(T::VWeight, layer, &[h, h])?);
        qkv.extend(get(T::QBias, layer, &[h])?);
        qkv.extend(get(T::KBias, layer, &[h])?);
        qkv.extend(get(T::VBias, layer, &[h])?);
        params.push(qkv);
        params.push([get(T::AttnOutWeight, layer, &[h, h])?, get(T::AttnOutBias, layer, &[h])?].concat());
        params.push([get(T::FfnLnWeight, layer, &[h])?, get(T::FfnLnBias, layer, &[h])?].concat());
        params.push([get(T::FfnInWeight, layer, &[i, h])?, get(T::FfnInBias, layer, &[i])?].concat());
        params.push([get(T::FfnOutWeight, layer, &[h, i])?, get(T::FfnOutBias, layer, &[h])?].concat());
    }
    Encoder::assemble_bert(
        geo,
        a.vocab_size as usize,
        params,
        a.family,
        a.max_positions as usize,
        a.position_offset as usize,
        a.token_types as usize,
    )
}
