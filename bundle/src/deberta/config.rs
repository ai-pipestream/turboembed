use crate::Result;
use crate::recipe::Recipe;
use serde_json::Value;
use std::{fs, path::Path};
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

/// The recipe's architecture against the upstream config.json of a
/// FAMILY_DEBERTA_V2 model: the config is the shared-key profile
/// `from_huggingface` takes, and every field of the manifest's geometry is
/// the config's. A recipe of another family is not checked.
pub fn agrees(cfg: &Value, manifest: &Value) -> Result<()> {
    let a = &manifest["architecture"];
    if a["family"] != "FAMILY_DEBERTA_V2" {
        return Ok(());
    }
    let c = from_huggingface(cfg)?;
    let pairs: [(&str, Option<f64>, f64); 10] = [
        ("hidden", a["hidden"].as_f64(), c.hidden as f64),
        ("heads", a["heads"].as_f64(), c.heads as f64),
        ("intermediate", a["intermediate"].as_f64(), c.intermediate as f64),
        ("layers", a["layers"].as_f64(), c.layers as f64),
        ("vocab_size", a["vocab_size"].as_f64(), c.vocabulary as f64),
        ("max_positions", a["max_positions"].as_f64(), c.max_sequence as f64),
        ("layer_norm_eps", a["layer_norm_eps"].as_f64(), c.layer_norm_eps),
        ("relative.position_buckets", a["relative"]["position_buckets"].as_f64(), c.position_buckets as f64),
        (
            "relative.max_relative_positions",
            a["relative"]["max_relative_positions"].as_f64(),
            c.max_relative_positions as f64,
        ),
        ("token_types", a["token_types"].as_f64(), 0.0),
    ];
    for (name, got, want) in pairs {
        if got != Some(want) {
            let got = got.map_or("absent".to_owned(), |g| g.to_string());
            return Err(format!(
                "manifest.architecture.{name}: {got}, the upstream DeBERTa-v2 config.json gives {want}"
            ));
        }
    }
    Ok(())
}

/// A FAMILY_DEBERTA_V2 recipe's architecture against the upstream
/// config.json, which such a recipe must have.
pub fn check_profile(recipe: &Recipe, upstream: &Path) -> Result<()> {
    if recipe.manifest["architecture"]["family"] != "FAMILY_DEBERTA_V2" {
        return Ok(());
    }
    let path = upstream.join("config.json");
    let bytes = fs::read(&path).map_err(|e| {
        format!("upstream config.json: {e}; a FAMILY_DEBERTA_V2 recipe's architecture is checked against it")
    })?;
    let cfg: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    agrees(&cfg, &recipe.manifest)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn config() -> Value {
        json!({
            "model_type": "deberta-v2", "hidden_act": "gelu", "relative_attention": true, "share_att_key": true,
            "position_biased_input": false, "type_vocab_size": 0, "norm_rel_ebd": "layer_norm",
            "pos_att_type": ["p2c", "c2p"], "hidden_size": 32, "num_attention_heads": 4,
            "intermediate_size": 64, "num_hidden_layers": 2, "vocab_size": 30522,
            "max_position_embeddings": 512, "position_buckets": 8, "max_relative_positions": -1,
            "layer_norm_eps": 1e-7
        })
    }

    fn manifest() -> Value {
        json!({ "architecture": {
            "family": "FAMILY_DEBERTA_V2", "hidden": 32, "heads": 4, "intermediate": 64, "layers": 2,
            "vocab_size": 30522, "max_positions": 512, "layer_norm_eps": 1e-7, "token_types": 0,
            "relative": { "position_buckets": 8, "max_relative_positions": 512 }
        }})
    }

    #[test]
    fn a_manifest_of_the_config_s_geometry_agrees() {
        assert_eq!(agrees(&config(), &manifest()), Ok(()));
        let mut bert = manifest();
        bert["architecture"]["family"] = "FAMILY_BERT".into();
        assert_eq!(agrees(&json!({}), &bert), Ok(()), "another family is not checked");
    }

    #[test]
    fn each_field_the_config_does_not_give_is_refused() {
        for (path, value) in [
            ("/architecture/hidden", json!(64)),
            ("/architecture/heads", json!(2)),
            ("/architecture/intermediate", json!(128)),
            ("/architecture/layers", json!(3)),
            ("/architecture/vocab_size", json!(128000)),
            ("/architecture/max_positions", json!(256)),
            ("/architecture/layer_norm_eps", json!(1e-12)),
            ("/architecture/token_types", json!(2)),
            ("/architecture/relative/position_buckets", json!(16)),
            ("/architecture/relative/max_relative_positions", json!(300)),
        ] {
            let mut m = manifest();
            *m.pointer_mut(path).unwrap() = value;
            let e = agrees(&config(), &m).unwrap_err();
            let name = path.trim_start_matches("/architecture/").replace('/', ".");
            assert!(e.contains(&format!("manifest.architecture.{name}:")), "{path}: {e}");
        }
    }

    #[test]
    fn a_config_outside_the_profile_is_refused_before_its_geometry() {
        let mut c = config();
        c["share_att_key"] = false.into();
        assert!(agrees(&c, &manifest()).unwrap_err().contains("share_att_key"));
        let mut c = config();
        c["pos_att_type"] = json!(["c2p"]);
        assert!(agrees(&c, &manifest()).unwrap_err().contains("pos_att_type"));
    }
}
