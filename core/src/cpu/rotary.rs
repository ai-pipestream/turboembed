//! What model_load checks of a ModernBERT or nomic-bert description
//! (#1461) before the encoder (encoder/rotary/) reads it.

use crate::backend::{
    TURBO_ACTIVATION_GELU_ERF, TURBO_ACTIVATION_SILU, TURBO_FAMILY_MODERNBERT, TURBO_FAMILY_NOMIC_BERT,
    TURBO_MLP_GATED, TURBO_MODERNBERT_EMBEDDING_TENSORS, TURBO_MODERNBERT_LAYER_TENSORS, TURBO_NOMIC_EMBEDDING_TENSORS,
    TURBO_NOMIC_LAYER_TENSORS, TURBO_NORM_LAYER, TURBO_NORM_LAYER_NO_BIAS, turbo_backend_model,
};
use crate::status::{INVALID_ARGUMENT, INVALID_STRUCT_SIZE, UNSUPPORTED};

/// Whether `family` is one of the rotary families.
pub(super) fn is(family: u32) -> bool {
    matches!(family, TURBO_FAMILY_MODERNBERT | TURBO_FAMILY_NOMIC_BERT)
}

/// Ok when the CPU's encoder runs the model `d` describes; else the
/// status and why.
pub(super) fn check(d: &turbo_backend_model) -> Result<(), (i32, String)> {
    let need = std::mem::offset_of!(turbo_backend_model, global_every) + size_of::<u32>();
    if (d.struct_size as usize) < need {
        let msg = format!("turbo_backend_model of {} bytes: a rotary model needs {need}", d.struct_size);
        return Err((INVALID_STRUCT_SIZE, msg));
    }
    let modern = d.family == TURBO_FAMILY_MODERNBERT;
    let (emb, per) = if modern {
        (TURBO_MODERNBERT_EMBEDDING_TENSORS, TURBO_MODERNBERT_LAYER_TENSORS)
    } else {
        (TURBO_NOMIC_EMBEDDING_TENSORS, TURBO_NOMIC_LAYER_TENSORS)
    };
    let count = emb as u64 + d.layers as u64 * per as u64;
    if d.tensor_count as u64 != count || d.tensors.is_null() {
        let msg = format!("tensor_count {}: this family's encoder of {} layers has {count}", d.tensor_count, d.layers);
        return Err((INVALID_ARGUMENT, msg));
    }
    let (act, norm) = if modern {
        (TURBO_ACTIVATION_GELU_ERF, TURBO_NORM_LAYER_NO_BIAS)
    } else {
        (TURBO_ACTIVATION_SILU, TURBO_NORM_LAYER)
    };
    if (d.mlp, d.activation, d.norm) != (TURBO_MLP_GATED, act, norm) {
        let msg = format!(
            "mlp {}, activation {}, norm {}: the cpu runs this family with {TURBO_MLP_GATED}, {act}, {norm}",
            d.mlp, d.activation, d.norm
        );
        return Err((UNSUPPORTED, msg));
    }
    let head = d.hidden.checked_div(d.heads).unwrap_or(0);
    if head == 0
        || !d.hidden.is_multiple_of(d.heads)
        || !head.is_multiple_of(2)
        || d.rope_pairs * 2 != head
        || d.rope_table.is_null()
    {
        let msg = format!(
            "hidden {}, heads {}, rope_pairs {}: a rotary head is an even width, twice rope_pairs, with a table",
            d.hidden, d.heads, d.rope_pairs
        );
        return Err((INVALID_ARGUMENT, msg));
    }
    // A model with no window has every layer global.
    if (d.attention_window == 0) != (d.global_every == 0) || (modern && d.attention_window == 0) {
        let msg = format!("attention_window {}, global_every {}", d.attention_window, d.global_every);
        return Err((INVALID_ARGUMENT, msg));
    }
    Ok(())
}
