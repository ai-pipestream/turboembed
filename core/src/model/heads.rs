//! The head tensors a raw-weights artifact names, beside the encoder's
//! (mod.rs): the classification head, the masked language model head of
//! `embed.sparse`, and the linear token-projection head. Each lists its
//! tensors with the shapes the architecture and the head's parameters
//! imply, in the order the loader reads them.

use super::Expected;
use crate::manifest::{Architecture, Artifact, Classifier, Projection, TensorRole, role_name};
use crate::status::{Result, invalid};

/// The classification head tensors an artifact names, with the shapes
/// the architecture and the classifier's label count imply.
pub fn classifier_tensors(index: usize, art: &Artifact, a: &Architecture, c: &Classifier) -> Result<Vec<Expected>> {
    let h = a.hidden as u64;
    let labels = c.labels.len() as u64;
    let mut out = Vec::with_capacity(4);
    let pooler_w = art.tensor_names.get(&TensorRole::PoolerWeight);
    let pooler_b = art.tensor_names.get(&TensorRole::PoolerBias);
    if pooler_w.is_some() != pooler_b.is_some() {
        return Err(invalid(format!(
            "manifest.json: artifacts[{index}].tensor_names: pooler_weight and pooler_bias come together, or neither"
        )));
    }
    if let (Some(w), Some(b)) = (pooler_w, pooler_b) {
        out.push(Expected { name: w.clone(), shape: vec![h, h], what: role_name(TensorRole::PoolerWeight) });
        out.push(Expected { name: b.clone(), shape: vec![h], what: role_name(TensorRole::PoolerBias) });
    }
    for (role, shape) in [(TensorRole::ClassifierWeight, vec![labels, h]), (TensorRole::ClassifierBias, vec![labels])] {
        let name = art.tensor_names.get(&role).ok_or_else(|| {
            invalid(format!(
                "manifest.json: artifacts[{index}].tensor_names: no {}, which a classifier needs",
                role_name(role)
            ))
        })?;
        out.push(Expected { name: name.clone(), shape, what: role_name(role) });
    }
    Ok(out)
}

/// The masked language model head tensors an artifact names, with the
/// shapes the architecture implies (mlm.rs): the transform, its
/// LayerNorm, and the decoder, whose weight is optional, tied to the
/// word embeddings when absent, as BERT ties it.
pub fn mlm_tensors(index: usize, art: &Artifact, a: &Architecture) -> Result<Vec<Expected>> {
    let (h, v) = (a.hidden as u64, a.vocab_size as u64);
    let mut out = Vec::with_capacity(6);
    for (role, shape) in [
        (TensorRole::MlmTransformWeight, vec![h, h]),
        (TensorRole::MlmTransformBias, vec![h]),
        (TensorRole::MlmTransformLnWeight, vec![h]),
        (TensorRole::MlmTransformLnBias, vec![h]),
    ] {
        let name = art.tensor_names.get(&role).ok_or_else(|| {
            invalid(format!(
                "manifest.json: artifacts[{index}].tensor_names: no {}, which embed.sparse needs",
                role_name(role)
            ))
        })?;
        out.push(Expected { name: name.clone(), shape, what: role_name(role) });
    }
    let bias = art.tensor_names.get(&TensorRole::MlmDecoderBias).ok_or_else(|| {
        invalid(format!(
            "manifest.json: artifacts[{index}].tensor_names: no {}, which embed.sparse needs",
            role_name(TensorRole::MlmDecoderBias)
        ))
    })?;
    // The decoder bias comes before the optional weight, so a tied head's
    // word embeddings always read at the same place (mlm.rs).
    out.push(Expected { name: bias.clone(), shape: vec![v], what: role_name(TensorRole::MlmDecoderBias) });
    if let Some(w) = art.tensor_names.get(&TensorRole::MlmDecoderWeight) {
        out.push(Expected { name: w.clone(), shape: vec![v, h], what: role_name(TensorRole::MlmDecoderWeight) });
    }
    Ok(out)
}

/// The linear projection head tensors an artifact names, with the shapes
/// the architecture and `projection.dim` imply.
pub fn projection_tensors(index: usize, art: &Artifact, a: &Architecture, p: &Projection) -> Result<Vec<Expected>> {
    let h = a.hidden as u64;
    let dim = p.dim as u64;
    let mut out = Vec::with_capacity(2);
    let w = art.tensor_names.get(&TensorRole::ProjectionWeight).ok_or_else(|| {
        invalid(format!(
            "manifest.json: artifacts[{index}].tensor_names: no {}, which a projection needs",
            role_name(TensorRole::ProjectionWeight)
        ))
    })?;
    out.push(Expected { name: w.clone(), shape: vec![dim, h], what: role_name(TensorRole::ProjectionWeight) });
    if let Some(b) = art.tensor_names.get(&TensorRole::ProjectionBias) {
        out.push(Expected { name: b.clone(), shape: vec![dim], what: role_name(TensorRole::ProjectionBias) });
    }
    Ok(out)
}
