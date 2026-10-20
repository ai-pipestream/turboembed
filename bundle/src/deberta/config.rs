use crate::Result;
use serde_json::Value;
use turbo::deberta::Config;

/// Translate only the shared-key DeBERTa-v3 profile. Unsupported architectural
/// switches return named errors instead of silently running BERT equations.
pub fn from_huggingface(v: &Value) -> Result<Config> {
    let integer = |name: &str| -> Result<usize> {
        v.get(name)
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| format!("deberta config: missing/invalid {name}"))
    };
    for (name, expected) in [
        ("model_type", Value::from("deberta-v2")),
        ("hidden_act", Value::from("gelu")),
        ("relative_attention", Value::from(true)),
        ("share_att_key", Value::from(true)),
        ("position_biased_input", Value::from(false)),
        ("type_vocab_size", Value::from(0)),
        ("norm_rel_ebd", Value::from("layer_norm")),
    ] {
        if v.get(name) != Some(&expected) {
            return Err(format!("deberta config: unsupported {name}, expected {expected}"));
        }
    }
    for name in ["conv_kernel_size", "z_steps"] {
        if v.get(name).is_some_and(|x| x.as_u64() != Some(0)) {
            return Err(format!("deberta config: unsupported {name}"));
        }
    }
    let hidden = integer("hidden_size")?;
    let heads = integer("num_attention_heads")?;
    if v.get("embedding_size").is_some_and(|x| x.as_u64() != Some(hidden as u64))
        || v.get("attention_head_size").is_some_and(|x| x.as_u64() != Some((hidden / heads.max(1)) as u64))
    {
        return Err("deberta config: unsupported embedding_size/attention_head_size".into());
    }
    let mut pos: Vec<&str> = match v.get("pos_att_type") {
        Some(Value::String(s)) => s.split('|').collect(),
        Some(Value::Array(a)) => a.iter().map(|x| x.as_str().unwrap_or("")).collect(),
        _ => return Err("deberta config: missing pos_att_type".into()),
    };
    pos.sort_unstable();
    if pos != ["c2p", "p2c"] {
        return Err("deberta config: pos_att_type must contain c2p and p2c exactly".into());
    }
    let max_sequence = integer("max_position_embeddings")?;
    let maximum = match v.get("max_relative_positions").and_then(Value::as_i64) {
        Some(-1) => max_sequence,
        Some(n) if n > 0 => usize::try_from(n).map_err(|_| "deberta relative position overflow")?,
        _ => return Err("deberta config: invalid max_relative_positions".into()),
    };
    let config = Config {
        hidden,
        heads,
        intermediate: integer("intermediate_size")?,
        layers: integer("num_hidden_layers")?,
        vocabulary: integer("vocab_size")?,
        max_sequence,
        position_buckets: integer("position_buckets")?,
        max_relative_positions: maximum,
        layer_norm_eps: v
            .get("layer_norm_eps")
            .and_then(Value::as_f64)
            .ok_or("deberta config: invalid layer_norm_eps")?,
    };
    config.check().map_err(|e| e.message)?;
    Ok(config)
}
