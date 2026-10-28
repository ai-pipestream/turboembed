//! nomic-ai/nomic-embed-text's config.json: contrastors' NomicBertConfig
//! keys (`n_embd`, `n_head`, `n_layer`, `n_inner`, `rotary_emb_base`,
//! `layer_norm_epsilon`) beside the transformers names (`hidden_size`,
//! `rope_parameters.rope_theta`, `layer_norm_eps`), which transformers 5
//! keeps when it saves the model again. Every pair present must agree.
//!
//! `max_position_embeddings` is the length the model was trained to (2048);
//! `n_positions` (8192) is the length it is served to, with the unscaled
//! cos/sin table (`rotary_scaling_factor` null; text-embeddings-inference
//! builds the table for `n_positions`). `max_positions` is `n_positions`
//! when the config has it.

use serde_json::Value;

use super::{Config, Fields};
use crate::Result;

pub(super) fn architecture(c: &Config) -> Result<Fields> {
    c.must("model_type", "nomic_bert".into())?;
    c.must("activation_function", "swiglu".into())?;
    c.default_is("hidden_act", "silu".into())?;
    c.must("rotary_emb_interleaved", false.into())?;
    c.must("rotary_emb_fraction", 1.into())?;
    c.must("prenorm", false.into())?;
    c.must("use_rms_norm", false.into())?;
    for key in ["qkv_proj_bias", "mlp_fc1_bias", "mlp_fc2_bias"] {
        c.must(key, false.into())?;
    }
    // The NTK-scaled table is not built; absent is contrastors' None.
    c.default_is("rotary_scaling_factor", Value::Null)?;
    c.default_is("rotary_emb_scale_base", Value::Null)?;
    c.default_is("rope_scaling", Value::Null)?;
    for key in ["causal", "parallel_block"] {
        c.default_is(key, false.into())?;
    }
    let hidden = c.number(&["hidden_size", "n_embd"])?;
    let heads = c.number(&["num_attention_heads", "n_head"])?;
    c.head_dim(&hidden, &heads)?;
    // Grouped keys are a contrastors option the encoder does not run: null,
    // or the heads it implies. `rotary_head_dim` is contrastors' flag for
    // xformers' rotary layout (layers/attention.py), not a width: only its
    // off values, null or false, describe this model.
    c.default_is_or_null("num_heads_kv", heads.clone())?;
    c.default_is_or_null("rotary_head_dim", false.into())?;
    let positions = match c.get("n_positions") {
        Some(_) => c.number(&["n_positions"])?,
        None => c.number(&["max_position_embeddings"])?,
    };
    Ok(vec![
        ("/position_embedding", Some("POSITION_ROTARY".into())),
        ("/mlp", Some("MLP_GATED".into())),
        ("/activation", Some("ACTIVATION_SILU".into())),
        ("/norm", Some("NORM_LAYER".into())),
        ("/hidden", Some(hidden)),
        ("/heads", Some(heads)),
        ("/layers", Some(c.number(&["num_hidden_layers", "n_layer"])?)),
        ("/intermediate", Some(c.number(&["intermediate_size", "n_inner"])?)),
        ("/vocab_size", Some(c.number(&["vocab_size"])?)),
        ("/max_positions", Some(positions)),
        ("/layer_norm_eps", Some(c.number(&["layer_norm_epsilon", "layer_norm_eps"])?)),
        ("/token_types", Some(c.number(&["type_vocab_size"])?)),
        ("/position_offset", Some(0.into())),
        ("/rope/global_theta", Some(c.theta("rotary_emb_base", None)?)),
        ("/rope/local_theta", None),
        ("/attention", None),
    ])
}
