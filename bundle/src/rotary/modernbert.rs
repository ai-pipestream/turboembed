//! answerdotai/ModernBERT's config.json. The checkpoint's own file has the
//! keys ModernBERT was released with (`global_attn_every_n_layers`,
//! `global_rope_theta`, `local_rope_theta`); transformers 5 saves the same
//! model with `layer_types` and `rope_parameters { full_attention,
//! sliding_attention }` beside them (transformers
//! configuration_modernbert.py, `__post_init__` and
//! `convert_rope_params_to_dict`). Either form, or both agreeing, is read.

use serde_json::Value;

use super::{Config, Fields};
use crate::Result;

const GLOBAL: &str = "full_attention";
const LOCAL: &str = "sliding_attention";

pub(super) fn architecture(c: &Config) -> Result<Fields> {
    c.must("model_type", "modernbert".into())?;
    // `gelu` is transformers' GELUActivation, the erf form.
    c.default_is("hidden_activation", "gelu".into())?;
    for key in ["norm_bias", "attention_bias", "mlp_bias"] {
        c.default_is(key, false.into())?;
    }
    c.default_is("rope_scaling", Value::Null)?;
    if let Some(p) = c.get("rope_parameters") {
        c.only_members("rope_parameters", p, &[GLOBAL, LOCAL])?;
    }
    let hidden = c.number(&["hidden_size"])?;
    let heads = c.number(&["num_attention_heads"])?;
    c.head_dim(&hidden, &heads)?;
    let layers = c.number(&["num_hidden_layers"])?;
    let global_every = global_every(c, &layers)?;
    let window = c.number(&["local_attention"])?;
    if let (Some(s), Some(w)) = (c.get("sliding_window"), window.as_u64())
        && s.as_u64() != Some(w / 2)
    {
        return Err(format!("upstream config.json: sliding_window is {s}; local_attention {w} gives {}", w / 2));
    }
    Ok(vec![
        ("/position_embedding", Some("POSITION_ROTARY".into())),
        ("/mlp", Some("MLP_GATED".into())),
        ("/activation", Some("ACTIVATION_GELU_ERF".into())),
        ("/norm", Some("NORM_LAYER_NO_BIAS".into())),
        ("/hidden", Some(hidden)),
        ("/heads", Some(heads)),
        ("/layers", Some(layers)),
        ("/intermediate", Some(c.number(&["intermediate_size"])?)),
        ("/vocab_size", Some(c.number(&["vocab_size"])?)),
        ("/max_positions", Some(c.number(&["max_position_embeddings"])?)),
        ("/layer_norm_eps", Some(c.number(&["norm_eps"])?)),
        ("/token_types", Some(0.into())),
        ("/position_offset", Some(0.into())),
        ("/rope/global_theta", Some(c.theta("global_rope_theta", Some(GLOBAL))?)),
        ("/rope/local_theta", Some(c.theta("local_rope_theta", Some(LOCAL))?)),
        ("/attention/window", Some(window)),
        ("/attention/global_every", Some(global_every.into())),
    ])
}

/// Layer l is global iff `l % n == 0`: `n` from `global_attn_every_n_layers`,
/// or read off `layer_types`, which must then be exactly that pattern.
fn global_every(c: &Config, layers: &Value) -> Result<u64> {
    let layers = layers.as_u64().filter(|&n| n > 0).ok_or("upstream config.json: num_hidden_layers is not positive")?;
    let legacy = match c.get("global_attn_every_n_layers") {
        Some(v) => Some(v.as_u64().filter(|&n| n > 0).ok_or_else(|| {
            format!("upstream config.json: global_attn_every_n_layers is {v}, not a positive whole number")
        })?),
        None => None,
    };
    let Some(types) = c.get("layer_types") else {
        return legacy
            .ok_or_else(|| "upstream config.json: global_attn_every_n_layers or layer_types is missing".into());
    };
    let types = types
        .as_array()
        .filter(|t| t.len() as u64 == layers)
        .ok_or_else(|| format!("upstream config.json: layer_types is not a list of {layers} layers"))?;
    // Without the old key, the period is the index of the second global
    // layer, or the layer count when layer 0 is the only one.
    let every =
        legacy.unwrap_or_else(|| types.iter().skip(1).position(|t| t == GLOBAL).map_or(layers, |i| i as u64 + 1));
    for (l, t) in types.iter().enumerate() {
        let want = if (l as u64).is_multiple_of(every) { GLOBAL } else { LOCAL };
        if t != want {
            return Err(format!(
                "upstream config.json: layer_types[{l}] is {t}; the pattern of a global layer every {every} gives \
                 \"{want}\""
            ));
        }
    }
    Ok(every)
}
