//! Declared learned sparse output: the recipe's `sparse` field, the
//! pooling a model with an MLM head serves (`SPARSE_POOLING_MAXLOG`, as
//! SPLADE pools, arXiv 2403.06789), which the tool copies into
//! `manifest.embed.sparse` when the recipe is loaded.
//!
//! Like matryoshka.rs, the declaration is read off the upstream model by
//! the recipe's author and pinned beside the rest of the model's facts;
//! the tool never guesses. The head's tensors are named in the manifest's
//! `artifacts[].tensor_names` by the recipe's author, the way a
//! classifier's are; the checks run before anything is fetched, and the
//! core validates the manifest field again when it is parsed at seal.

use serde::Deserialize;

use crate::Result;
use crate::recipe::Recipe;

/// The recipe's `sparse` field: which pooling the MLM head is pooled by.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseDecl {
    /// Only SPLADE's max pooling exists today; the field is written so a
    /// later pooling is a value here, not a schema change.
    pub pooling: SparsePooling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum SparsePooling {
    #[serde(rename = "SPARSE_POOLING_MAXLOG")]
    MaxLog,
}

impl SparsePooling {
    /// The manifest's name for the value.
    pub fn name(self) -> &'static str {
        match self {
            SparsePooling::MaxLog => "SPARSE_POOLING_MAXLOG",
        }
    }
}

/// Fill `manifest.embed.sparse` from the recipe's `sparse` field, or check
/// the block the manifest already carries; either way the recipe is
/// refused, naming the field, when both carry it.
pub fn declare(r: &mut Recipe) -> Result<()> {
    let in_manifest = r.manifest["embed"].get("sparse").is_some();
    match (&r.sparse, in_manifest) {
        (Some(_), true) => {
            Err("the sparse output is given twice: the recipe has a top-level sparse field and manifest.embed.sparse. \
             Give it once"
                .into())
        }
        (Some(decl), false) => {
            check_mlm_roles(&r.manifest)?;
            r.manifest["embed"]["sparse"] = serde_json::json!({ "pooling": decl.pooling.name() });
            Ok(())
        }
        (None, true) => {
            let field = &r.manifest["embed"]["sparse"];
            let pooling = field["pooling"].as_str().ok_or("manifest.embed.sparse.pooling: missing or not a string")?;
            if pooling != SparsePooling::MaxLog.name() {
                return Err(format!("manifest.embed.sparse.pooling: {pooling:?} is not a SPARSE_POOLING_* value"));
            }
            check_mlm_roles(&r.manifest)
        }
        (None, false) => Ok(()),
    }
}

/// Every raw-weights artifact names the MLM head, as the core checks it
/// when the manifest is parsed; refused here before anything is fetched,
/// so a bad recipe fails at load, not at seal.
fn check_mlm_roles(manifest: &serde_json::Value) -> Result<()> {
    let arts = manifest["artifacts"].as_array().ok_or("manifest.artifacts: missing")?;
    for (i, a) in arts.iter().enumerate() {
        if a["format"].as_str() != Some("FORMAT_SAFETENSORS") {
            continue;
        }
        let tn =
            a["tensor_names"].as_object().ok_or_else(|| format!("manifest.artifacts[{i}].tensor_names: missing"))?;
        for role in [
            "mlm_transform_weight",
            "mlm_transform_bias",
            "mlm_transform_ln_weight",
            "mlm_transform_ln_bias",
            "mlm_decoder_bias",
        ] {
            if !tn.contains_key(role) {
                return Err(format!("manifest.artifacts[{i}].tensor_names: no {role}, which a sparse output needs"));
            }
        }
    }
    Ok(())
}
