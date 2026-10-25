//! A static recipe made from a base bundle alone, for a caller that
//! distils a bundle it already serves with no recipe file: the manifest
//! the bundle's own static recipes have (bundle/recipes/*-static.json),
//! with the base's model source, licence, licence text and tokenizer.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::Spec;
use crate::Result;
use crate::recipe::{Local, Recipe};

/// The quality texts every static recipe is measured on (docs/static.md).
const QUALITY: &str = include_str!("../../recipes/static-quality.jsonl");

/// The recipe whose reference cases a made recipe uses.
const CASES_FROM: &str = include_str!("../../recipes/all-minilm-l6-v2-static.json");

/// What a static model made from a base is.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// The new model's id and revision.
    pub id: String,
    pub revision: String,
    pub pca_dims: u32,
    pub sif_coefficient: f64,
    /// "F32" or "F16".
    pub dtype: String,
    /// The tokens a text is cut to.
    pub max_length: u32,
}

/// The static recipe for distilling the bundle in `base` as `o` says.
/// Its quality texts are written into `work`, which the recipe's local
/// files are read from.
pub fn recipe(base: &Path, o: &Options, work: &Path) -> Result<Recipe> {
    let path = base.join("manifest.json");
    let bm: Value = serde_json::from_slice(&fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let vocab_size = bm["architecture"]["vocab_size"]
        .as_u64()
        .ok_or("base bundle: no architecture.vocab_size; a static model is distilled from an encoder")?;
    let mut tokenizer = bm["tokenizer"].clone();
    tokenizer["file"] = json!("tokenizer.json");
    tokenizer["template"] = json!(["$TEXT"]);
    let cases: Value = serde_json::from_str(CASES_FROM).map_err(|e| e.to_string())?;
    let mut manifest = json!({
        "bundle_version": 1,
        "model": {
            "id": o.id,
            "revision": o.revision,
            "source": bm["model"]["source"],
            "license": bm["model"]["license"],
        },
        "task": "TASK_EMBED",
        "embed": {
            "dim": o.pca_dims,
            "pooling": "POOLING_MEAN",
            "normalize": "NORMALIZE_L2",
            "max_seq": 8192,
            "max_batch": 256,
        },
        "tokenizer": tokenizer,
        "static_embedding": {
            "vocab_size": vocab_size,
            "max_length": o.max_length,
            "quality": { "texts": "quality/texts.jsonl" },
        },
        "artifacts": [{
            "name": format!("table-{}", o.dtype.to_ascii_lowercase()),
            "format": "FORMAT_SAFETENSORS",
            "files": ["weights/static.safetensors"],
            "backends": ["cpu"],
            "graph_input": "INPUT_TOKEN_IDS",
            "graph_output": "OUTPUT_HIDDEN_STATES",
            "tensor_names": { "static_embeddings": "embeddings" },
        }],
        "reference": {
            "file": "reference/reference.safetensors",
            "cases": cases["manifest"]["reference"]["cases"],
        },
    });
    fs::create_dir_all(work).map_err(|e| format!("{}: {e}", work.display()))?;
    crate::fetch::write_atomic(&work.join("static-quality.jsonl"), QUALITY.as_bytes())?;
    let mut local = vec![Local {
        path: "static-quality.jsonl".into(),
        to: "quality/texts.jsonl".into(),
        sha256: turbo::bundle::sha256_hex(QUALITY.as_bytes()),
    }];
    // Carry the verified licence bytes, but reserve their output path:
    // a base path may name a generated table, quality file or report.
    if let Some(l) = bm["model"].get("license_file").filter(|v| !v.is_null()) {
        let l = l.as_str().ok_or("base bundle: model.license_file is not a path")?;
        crate::check_rel(l)?;
        let source = turbo::bundle::Bundle::open(base).map_err(|e| format!("base bundle: {}", e.message))?;
        let text = source.read_verified(l).map_err(|e| format!("base licence: {}", e.message))?;
        crate::fetch::write_atomic(&work.join("base-license"), &text)?;
        let to = "licenses/base-license";
        manifest["model"]["license_file"] = json!(to);
        local.push(Local { path: "base-license".into(), to: to.into(), sha256: turbo::bundle::sha256_hex(&text) });
    }
    Ok(Recipe {
        manifest,
        notice: None,
        upstream: Vec::new(),
        local,
        bpe_tokenizer: None,
        distill: Some(Spec {
            pca_dims: o.pca_dims,
            sif_coefficient: o.sif_coefficient,
            skip: r"\[unused\d+\]".into(),
            dtype: o.dtype.clone(),
        }),
        sparse: None,
        matryoshka: None,
        dir: PathBuf::from(work),
    })
}
