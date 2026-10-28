//! The rotary families (#1461): which tensors a ModernBERT or nomic-bert
//! encoder has, in the order model_load takes them, and the shape the
//! architecture gives each; and the fields of turbo_backend_model that
//! only they fill.

use crate::backend::{
    TURBO_ACTIVATION_GELU_ERF, TURBO_ACTIVATION_SILU, TURBO_MLP_DENSE, TURBO_MLP_GATED,
    TURBO_MODERNBERT_EMBEDDING_TENSORS, TURBO_MODERNBERT_LAYER_TENSORS, TURBO_NOMIC_EMBEDDING_TENSORS,
    TURBO_NOMIC_LAYER_TENSORS, TURBO_NORM_LAYER, TURBO_NORM_LAYER_NO_BIAS, TURBO_NORM_RMS, turbo_backend_model,
};
use crate::bundle::AlignedBytes;
use crate::manifest::{Activation, Architecture, Artifact, Family, Mlp, Norm, TensorRole};
use crate::safetensors;
use crate::status::{CAPACITY, Error, Result, invalid};

/// A ModernBERT encoder's embedding tensors, at TURBO_MODERNBERT_* in
/// turbo_backend.h: the final norm is among them, since it is one tensor
/// for the whole encoder.
pub const MODERNBERT_EMBEDDING_ROLES: [TensorRole; TURBO_MODERNBERT_EMBEDDING_TENSORS as usize] = {
    use TensorRole::*;
    [WordEmbeddings, EmbeddingsLnWeight, FinalNormWeight]
};

/// Each ModernBERT layer's tensors, at TURBO_MODERNBERT_*: its attention
/// norm (none in layer 0), the fused QKV and output projections, its MLP
/// norm, and the gated MLP's two projections.
pub const MODERNBERT_LAYER_ROLES: [TensorRole; TURBO_MODERNBERT_LAYER_TENSORS as usize] = {
    use TensorRole::*;
    [AttnLnWeight, QkvWeight, AttnOutWeight, FfnLnWeight, FfnInWeight, FfnOutWeight]
};

/// A nomic-bert encoder's embedding tensors, at TURBO_NOMIC_*.
pub const NOMIC_EMBEDDING_ROLES: [TensorRole; TURBO_NOMIC_EMBEDDING_TENSORS as usize] = {
    use TensorRole::*;
    [WordEmbeddings, TokenTypeEmbeddings, EmbeddingsLnWeight, EmbeddingsLnBias]
};

/// Each nomic-bert layer's tensors, at TURBO_NOMIC_*.
pub const NOMIC_LAYER_ROLES: [TensorRole; TURBO_NOMIC_LAYER_TENSORS as usize] = {
    use TensorRole::*;
    [
        QkvWeight,
        AttnOutWeight,
        AttnLnWeight,
        AttnLnBias,
        FfnGateWeight,
        FfnUpWeight,
        FfnOutWeight,
        FfnLnWeight,
        FfnLnBias,
    ]
};

/// The family's embedding and layer tables; None for a family that is
/// not rotary.
pub(super) fn tables(family: Family) -> Option<(&'static [TensorRole], &'static [TensorRole])> {
    match family {
        Family::Modernbert => Some((&MODERNBERT_EMBEDDING_ROLES, &MODERNBERT_LAYER_ROLES)),
        Family::NomicBert => Some((&NOMIC_EMBEDDING_ROLES, &NOMIC_LAYER_ROLES)),
        Family::Bert | Family::Roberta | Family::DebertaV2 => None,
    }
}

/// The shape of a rotary role whose shape BERT's tables do not give: the
/// fused QKV, ModernBERT's MLP input holding the activated half's rows
/// and then the linear half's, and nomic-bert's separate gate and up.
pub(super) fn rotary_shape(role: TensorRole, a: &Architecture) -> Option<Vec<u64>> {
    let (h, i) = (a.hidden as u64, a.intermediate as u64);
    match role {
        TensorRole::QkvWeight => Some(vec![3 * h, h]),
        TensorRole::FfnInWeight if a.family == Family::Modernbert => Some(vec![2 * i, h]),
        TensorRole::FfnGateWeight | TensorRole::FfnUpWeight => Some(vec![i, h]),
        TensorRole::FinalNormWeight => Some(vec![h]),
        _ => None,
    }
}

/// The index in model_load's tensors of the one tensor a ModernBERT
/// encoder does not have: layer 0's attention norm, whose place is an
/// identity (transformers modeling_modernbert.py, `attn_norm` is
/// `nn.Identity` when `layer_id == 0`). None for every other family.
pub fn absent(family: Family) -> Option<usize> {
    (family == Family::Modernbert).then_some(TURBO_MODERNBERT_EMBEDDING_TENSORS as usize)
}

/// Refuses a ModernBERT artifact whose files hold the tensor `absent`
/// says layer 0 lacks: a manifest that names it and an encoder that never
/// reads it would disagree about the model.
pub(super) fn check_absent(art: &Artifact, files: &[AlignedBytes], a: &Architecture) -> Result<()> {
    if absent(a.family).is_none() {
        return Ok(());
    }
    let Some(name) = art.tensor_names.get(&TensorRole::AttnLnWeight) else {
        return Ok(());
    };
    let name = name.replace("{layer}", "0");
    for (file, bytes) in art.files.iter().zip(files) {
        if safetensors::File::parse(file, &bytes[..])?.tensor(&name).is_some() {
            return Err(invalid(format!(
                "{file}: {name:?} (attn_ln_weight of layer 0): a ModernBERT encoder's first layer has no attention norm"
            )));
        }
    }
    Ok(())
}

/// What a rotary family's encoder is told beyond BERT's geometry: its
/// rotary bases and the core's cos/sin table, and its MLP, activation,
/// norm and attention window as turbo_backend.h numbers them.
pub(super) struct Rotary {
    thetas: [f64; 2],
    table: Vec<f32>,
    pairs: u32,
    /// mlp, activation, norm, attention_window, global_every.
    knobs: [u32; 5],
}

impl Rotary {
    /// The family's fields, with the table built for every position the
    /// architecture has; None for a family that is not rotary.
    pub(super) fn of(a: &Architecture) -> Result<Option<Rotary>> {
        let Some(rope) = a.rope.filter(|_| tables(a.family).is_some()) else {
            return Ok(None);
        };
        let global = rope.global_theta;
        let local = rope.local_theta.unwrap_or(global);
        let pairs = a.hidden / a.heads / 2;
        let table = crate::rope::table(global, local, a.max_positions as usize, pairs as usize).map_err(|bytes| {
            Error::new(CAPACITY, format!("the rotary table of {} positions: {bytes} bytes", a.max_positions))
        })?;
        let mlp = match a.mlp {
            Mlp::Dense => TURBO_MLP_DENSE,
            Mlp::Gated => TURBO_MLP_GATED,
        };
        let activation = match a.activation {
            Activation::GeluErf => TURBO_ACTIVATION_GELU_ERF,
            Activation::Silu => TURBO_ACTIVATION_SILU,
        };
        let norm = match a.norm {
            Norm::Layer => TURBO_NORM_LAYER,
            Norm::LayerNoBias => TURBO_NORM_LAYER_NO_BIAS,
            Norm::Rms => TURBO_NORM_RMS,
        };
        let (window, every) = a.attention.map_or((0, 0), |w| (w.window, w.global_every));
        Ok(Some(Rotary { thetas: [global, local], table, pairs, knobs: [mlp, activation, norm, window, every] }))
    }

    /// Writes the fields into `d`, pointing into `self`'s table.
    pub(super) fn describe(&self, d: &mut turbo_backend_model) {
        [d.rope_global_theta, d.rope_local_theta] = self.thetas;
        d.rope_table = self.table.as_ptr();
        d.rope_pairs = self.pairs;
        [d.mlp, d.activation, d.norm, d.attention_window, d.global_every] = self.knobs;
    }
}
