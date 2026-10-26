//! Which tensors a family's encoder has, in the order model_load takes
//! them, and the shape the architecture gives each.

use crate::backend::{TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, TURBO_DEBERTA_EMBEDDING_TENSORS};
use crate::manifest::{Architecture, Artifact, Family, TensorRole, role_name};
use crate::status::{Result, invalid};

/// The shape the architecture implies for a BERT tensor: `[out, in]` for a
/// linear layer's weight, `[out]` for its bias and for LayerNorm.
fn bert_shape(role: TensorRole, a: &Architecture) -> Vec<u64> {
    let (h, i) = (a.hidden as u64, a.intermediate as u64);
    use TensorRole::*;
    match role {
        WordEmbeddings => vec![a.vocab_size as u64, h],
        PositionEmbeddings => vec![a.max_positions as u64, h],
        TokenTypeEmbeddings => vec![a.token_types as u64, h],
        QWeight | KWeight | VWeight | AttnOutWeight => vec![h, h],
        FfnInWeight => vec![i, h],
        FfnInBias => vec![i],
        FfnOutWeight => vec![h, i],
        EmbeddingsLnWeight | EmbeddingsLnBias | QBias | KBias | VBias | AttnOutBias | AttnLnWeight | AttnLnBias
        | FfnOutBias | FfnLnWeight | FfnLnBias | RelativeLnWeight | RelativeLnBias => vec![h],
        RelativeEmbeddings => vec![2 * a.relative.map_or(0, |r| r.position_buckets) as u64, h],
        StaticEmbeddings | StaticWeights | StaticMapping => {
            unreachable!("an encoder's tensor_names has no static role (validate)")
        }
        PoolerWeight | PoolerBias | ClassifierWeight | ClassifierBias => {
            unreachable!("classifier roles are shaped by classifier_tensors")
        }
        MlmTransformWeight | MlmTransformBias | MlmTransformLnWeight | MlmTransformLnBias | MlmDecoderWeight
        | MlmDecoderBias => unreachable!("MLM roles are shaped by mlm_tensors"),
        ProjectionWeight | ProjectionBias => {
            unreachable!("projection roles are shaped by projection_tensors")
        }
    }
}

/// The embedding tensors, at TURBO_BERT_* in turbo_backend.h.
pub const BERT_EMBEDDING_ROLES: [TensorRole; TURBO_BERT_EMBEDDING_TENSORS as usize] = {
    use TensorRole::*;
    [WordEmbeddings, PositionEmbeddings, TokenTypeEmbeddings, EmbeddingsLnWeight, EmbeddingsLnBias]
};

/// Each layer's tensors, at TURBO_BERT_* in turbo_backend.h.
pub const BERT_LAYER_ROLES: [TensorRole; TURBO_BERT_LAYER_TENSORS as usize] = {
    use TensorRole::*;
    [
        QWeight,
        QBias,
        KWeight,
        KBias,
        VWeight,
        VBias,
        AttnOutWeight,
        AttnOutBias,
        AttnLnWeight,
        AttnLnBias,
        FfnInWeight,
        FfnInBias,
        FfnOutWeight,
        FfnOutBias,
        FfnLnWeight,
        FfnLnBias,
    ]
};

/// One tensor a BERT encoder needs: its name in the weights file and the
/// shape the architecture implies.
pub struct Expected {
    pub name: String,
    pub shape: Vec<u64>,
    /// The role and layer, for messages.
    pub what: String,
}

/// A DeBERTa-v2 encoder's embedding tensors, at TURBO_DEBERTA_* in
/// turbo_backend.h; its layers' are BERT_LAYER_ROLES.
pub const DEBERTA_EMBEDDING_ROLES: [TensorRole; TURBO_DEBERTA_EMBEDDING_TENSORS as usize] = {
    use TensorRole::*;
    [WordEmbeddings, EmbeddingsLnWeight, EmbeddingsLnBias, RelativeEmbeddings, RelativeLnWeight, RelativeLnBias]
};

/// Every tensor the family's encoder needs, in the order turbo_backend.h's
/// TURBO_BERT_* (or TURBO_DEBERTA_*) constants give. Every role must be in
/// tensor_names.
pub fn bert_tensors(index: usize, art: &Artifact, a: &Architecture) -> Result<Vec<Expected>> {
    tensors_of(index, art, a, a.layers)
}

/// The embedding tensors alone, for a host lookup: only their roles need
/// be in tensor_names.
pub fn embedding_tensors(index: usize, art: &Artifact, a: &Architecture) -> Result<Vec<Expected>> {
    tensors_of(index, art, a, 0)
}

fn tensors_of(index: usize, art: &Artifact, a: &Architecture, layers: u32) -> Result<Vec<Expected>> {
    let encoder = if a.family == Family::DebertaV2 { "a DeBERTa-v2" } else { "a BERT" };
    let name = |role: TensorRole| {
        art.tensor_names.get(&role).ok_or_else(|| {
            invalid(format!(
                "manifest.json: artifacts[{index}].tensor_names: no {}, which {encoder} encoder needs",
                role_name(role)
            ))
        })
    };
    let embeddings: &[TensorRole] =
        if a.family == Family::DebertaV2 { &DEBERTA_EMBEDDING_ROLES } else { &BERT_EMBEDDING_ROLES };
    let mut out = Vec::with_capacity(embeddings.len() + (layers * TURBO_BERT_LAYER_TENSORS) as usize);
    for &role in embeddings {
        out.push(Expected { name: name(role)?.clone(), shape: bert_shape(role, a), what: role_name(role) });
    }
    for layer in 0..layers {
        for role in BERT_LAYER_ROLES {
            out.push(Expected {
                name: name(role)?.replace("{layer}", &layer.to_string()),
                shape: bert_shape(role, a),
                what: format!("{} of layer {layer}", role_name(role)),
            });
        }
    }
    Ok(out)
}
