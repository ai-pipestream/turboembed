//! Declared Matryoshka widths: the recipe's `matryoshka` field, the widths
//! a model trained with the Matryoshka objective is cut to, which the tool
//! copies into `manifest.embed.output_dims` when the recipe is loaded.
//!
//! The widths are read off the upstream model card by the recipe's author
//! and pinned beside the rest of the model's facts; the tool never guesses
//! them and never parses a card itself. A recipe that declares none, in the
//! field or in the manifest, serves its model's own dim alone. The checks
//! run before anything is fetched: each width a nonzero integer within the
//! model's dim, the list ascending (so unique), and no more of them than
//! `turbo_model_info.output_dims` holds. The core validates the manifest
//! field again when it is parsed at seal.

use serde_json::json;

use crate::Result;
use crate::recipe::Recipe;

/// Fill `manifest.embed.output_dims` from the recipe's `matryoshka` field,
/// or check the field the manifest already carries; either way the recipe
/// is refused, naming the width, when its list is not ascending, not within
/// the model's dim, or too long for the header.
pub fn declare(r: &mut Recipe) -> Result<()> {
    let in_manifest = r.manifest["embed"].get("output_dims").is_some();
    match (&r.matryoshka, in_manifest) {
        (Some(_), true) => Err(
            "the widths are given twice: the recipe has a top-level matryoshka field and manifest.embed.output_dims. \
             Give them once"
                .into(),
        ),
        (Some(widths), false) => {
            check(widths, "matryoshka", dim(r)?)?;
            r.manifest["embed"]["output_dims"] = json!(widths);
            Ok(())
        }
        (None, true) => {
            let field = &r.manifest["embed"]["output_dims"];
            let Some(widths) = field.as_array() else {
                return Err("manifest.embed.output_dims: not a list".into());
            };
            let mut dims = Vec::with_capacity(widths.len());
            for (i, w) in widths.iter().enumerate() {
                let Some(d) = w.as_u64() else {
                    return Err(format!("manifest.embed.output_dims[{i}]: {w} is not an integer"));
                };
                dims.push(u32::try_from(d).map_err(|_| format!("manifest.embed.output_dims[{i}]: {d} is over u32"))?);
            }
            check(&dims, "manifest.embed.output_dims", dim(r)?)?;
            Ok(())
        }
        (None, false) => Ok(()),
    }
}

/// The model's dim, the ceiling every width is checked against.
fn dim(r: &Recipe) -> Result<u64> {
    r.manifest["embed"]["dim"]
        .as_u64()
        .ok_or_else(|| "manifest.embed.dim: missing or not an integer, and matryoshka needs it".to_owned())
}

/// `widths` as the manifest field `name` carries it: every width in 1..=dim,
/// the list ascending, and at most `turbo::TURBO_OUTPUT_DIMS_MAX` entries.
fn check(widths: &[u32], name: &str, dim: u64) -> Result<()> {
    if widths.len() > turbo::TURBO_OUTPUT_DIMS_MAX {
        return Err(format!(
            "{name}: {} widths, and turbo_model_info.output_dims holds {}",
            widths.len(),
            turbo::TURBO_OUTPUT_DIMS_MAX
        ));
    }
    for (i, &d) in widths.iter().enumerate() {
        if d == 0 || d as u64 > dim {
            return Err(format!("{name}[{i}]: {d} is not in 1..={dim}"));
        }
        if i > 0 && d <= widths[i - 1] {
            return Err(format!("{name}[{i}]: {d} follows {}: list the widths ascending", widths[i - 1]));
        }
    }
    Ok(())
}
