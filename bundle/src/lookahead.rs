//! Limited lookahead from a SaT checkpoint's config.json: wtpsplit's
//! `lookahead` is the keys ahead over all the layers, and each layer
//! takes `lookahead / lookahead_split_layers` of them, or `/ num_hidden_layers`
//! when that is absent, rounded down (wtpsplit `models.py`). The manifest's
//! `architecture.lookahead` is that per-layer value, and sealing holds the
//! recipe to it.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::Result;
use crate::recipe::Recipe;

/// The keys ahead a layer attends to that `cfg` trains, None for none.
pub fn per_layer(cfg: &Value) -> Result<Option<u64>> {
    let Some(total) = cfg.get("lookahead").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let total = total.as_u64().ok_or("upstream config.json: lookahead is not a whole number")?;
    let over = match cfg.get("lookahead_split_layers").filter(|v| !v.is_null()) {
        Some(v) => v.as_u64().ok_or("upstream config.json: lookahead_split_layers is not a whole number")?,
        None => cfg["num_hidden_layers"].as_u64().ok_or("upstream config.json: num_hidden_layers is missing")?,
    };
    match total.checked_div(over) {
        Some(n) if n > 0 => Ok(Some(n)),
        _ => Err(format!("upstream config.json: lookahead {total} over {over} layers leaves no key ahead a layer")),
    }
}

/// The recipe's architecture.lookahead against what `cfg` trains.
pub fn agrees(cfg: &Value, manifest: &Value) -> Result<()> {
    let want = per_layer(cfg)?;
    let got = manifest.pointer("/architecture/lookahead").and_then(Value::as_u64);
    if got == want {
        return Ok(());
    }
    let say = |v: Option<u64>| v.map_or("none".to_owned(), |n| n.to_string());
    Err(format!(
        "manifest.architecture.lookahead: {}, the upstream config.json trains {} a layer (lookahead {} over the layers)",
        say(got),
        say(want),
        say(cfg.get("lookahead").and_then(Value::as_u64))
    ))
}

/// When `upstream` has config.json, the recipe's lookahead is the one it
/// trains.
pub fn check(recipe: &Recipe, upstream: &Path) -> Result<()> {
    let path = upstream.join("config.json");
    if !path.is_file() {
        return Ok(());
    }
    let cfg: Value = serde_json::from_slice(&fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    agrees(&cfg, &recipe.manifest)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn the_lookahead_is_divided_among_the_layers() {
        let sat_3l = json!({ "lookahead": 48, "num_hidden_layers": 3, "lookahead_split_layers": null });
        assert_eq!(per_layer(&sat_3l), Ok(Some(16)));
        assert_eq!(
            per_layer(&json!({ "lookahead": 48, "num_hidden_layers": 12, "lookahead_split_layers": 12 })),
            Ok(Some(4))
        );
        assert_eq!(per_layer(&json!({ "lookahead": 8, "num_hidden_layers": 3 })), Ok(Some(2)));
        assert_eq!(per_layer(&json!({ "lookahead": null, "num_hidden_layers": 1 })), Ok(None));
        assert_eq!(per_layer(&json!({ "num_hidden_layers": 1 })), Ok(None));
        assert!(per_layer(&json!({ "lookahead": 2, "num_hidden_layers": 3 })).unwrap_err().contains("no key ahead"));
        assert!(agrees(&sat_3l, &json!({ "architecture": { "lookahead": 16 } })).is_ok());
        let e = agrees(&sat_3l, &json!({ "architecture": {} })).unwrap_err();
        assert!(e.contains("lookahead: none, the upstream config.json trains 16 a layer (lookahead 48"), "{e}");
        let e = agrees(&json!({ "num_hidden_layers": 1 }), &json!({ "architecture": { "lookahead": 4 } })).unwrap_err();
        assert!(e.contains("lookahead: 4, the upstream config.json trains none"), "{e}");
    }

    #[test]
    fn a_split_inside_the_encoder_cannot_be_one_uniform_band() {
        // wtpsplit switches to causal attention at this layer, even when
        // the initial mask has unrestricted (null or absent) lookahead.
        for split in [0, 1, 6, 11] {
            for total in [json!(48), Value::Null] {
                let cfg = json!({ "lookahead": total, "num_hidden_layers": 12, "lookahead_split_layers": split });
                let e = per_layer(&cfg).unwrap_err();
                assert!(e.contains("per-layer lookahead") && e.contains("not supported"), "{cfg}: {e}");
                assert!(agrees(&cfg, &json!({ "architecture": { "lookahead": 8 } })).is_err());
            }
        }
        assert!(per_layer(&json!({ "num_hidden_layers": 12, "lookahead_split_layers": 6 })).is_err());
    }

    #[test]
    fn an_explicit_split_needs_a_valid_layer_count() {
        for cfg in [
            json!({ "lookahead": 48, "num_hidden_layers": 12, "lookahead_split_layers": 13 }),
            json!({ "lookahead": 48, "lookahead_split_layers": 6 }),
            json!({ "lookahead": 48, "num_hidden_layers": "12", "lookahead_split_layers": 6 }),
        ] {
            assert!(per_layer(&cfg).is_err(), "{cfg}");
        }
    }
}
