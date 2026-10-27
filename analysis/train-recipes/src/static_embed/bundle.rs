//! Sealing a trained table as a static bundle through the bundle
//! tool's own make path (stage, reference, seal, verify), so the
//! result is a bundle like any other: the recipe is the potion
//! static recipe's manifest with this model's id, source and
//! licence, the BERT tokenizer block of the MiniLM recipe, and the
//! table staged as the upstream `model.safetensors`.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::error::Error;

use super::write_safetensors;

fn fail<T>(m: String) -> Result<T, Error> {
    Err(Error::Model(m))
}

/// What the sealed manifest says about the model.
pub struct Identity<'a> {
    pub id: &'a str,
    pub repository: &'a str,
    pub commit: &'a str,
}

/// The paths a seal reads and writes.
pub struct Paths<'a> {
    pub tokenizer_json: &'a Path,
    pub recipes: &'a Path,
    pub stage: &'a Path,
    pub out: &'a Path,
}

/// Stage `table` ([vocab x dim]) and seal the bundle at `paths.out`
/// (which must not already hold one). `paths.recipes` is the repo's
/// `bundle/recipes` directory, read for the manifest templates.
/// Returns the sealed bundle's manifest SHA-256.
pub fn make_bundle(table: &[f32], vocab: usize, dim: usize, paths: &Paths, who: &Identity) -> Result<String, Error> {
    let (tokenizer_json, recipes, stage, out) = (paths.tokenizer_json, paths.recipes, paths.stage, paths.out);
    if table.len() != vocab * dim {
        return fail(format!("table has {} values for {vocab}x{dim}", table.len()));
    }
    fs::create_dir_all(stage).map_err(|e| Error::Model(format!("{}: {e}", stage.display())))?;
    fs::write(stage.join("model.safetensors"), write_safetensors(&[("embeddings", &[vocab, dim], table.to_vec())]))
        .map_err(|e| Error::Model(format!("model.safetensors: {e}")))?;
    // BERT's upstream tokenizer.json omits model.type, which the core
    // tokenizer requires; spelling it is the one edit to the staged
    // tokenizer, and it changes no vocabulary or rule.
    let tok_bytes = fs::read(tokenizer_json).map_err(|e| Error::Model(format!("tokenizer.json: {e}")))?;
    let mut tok: Value =
        serde_json::from_slice(&tok_bytes).map_err(|e| Error::Model(format!("tokenizer.json: {e}")))?;
    if tok["model"].get("type").is_none_or(Value::is_null) {
        tok["model"]["type"] = "WordPiece".into();
    }
    fs::write(stage.join("tokenizer.json"), serde_json::to_vec(&tok).unwrap())
        .map_err(|e| Error::Model(format!("tokenizer.json: {e}")))?;
    fs::write(stage.join("config.json"), "{\"normalize\": true, \"max_length\": 512}\n")
        .map_err(|e| Error::Model(format!("config.json: {e}")))?;

    let read_recipe = |rel: &str| -> Result<Value, Error> {
        let path = recipes.join(rel);
        let bytes = fs::read(&path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
    };
    let mut recipe = read_recipe("potion/potion-base-8M.json")?;
    let minilm = read_recipe("all-minilm-l6-v2.json")?;
    let m = &mut recipe["manifest"];
    m["model"]["id"] = who.id.into();
    m["model"]["revision"] = "1".into();
    m["model"]["source"] = serde_json::json!({"repository": who.repository, "commit": who.commit});
    m["model"]["license"] = "Apache-2.0".into();
    m["model"]["license_class"] = "CLASS_PERMISSIVE".into();
    m["embed"]["dim"] = dim.into();
    m["static_embedding"]["vocab_size"] = vocab.into();
    m["static_embedding"]["max_length"] = 512.into();
    // The teacher's tokenizer is BERT's: keep the potion block's
    // static template (["$TEXT"]) and take the BERT special-token
    // ids from the MiniLM recipe.
    m["tokenizer"]["special_tokens"] = minilm["manifest"]["tokenizer"]["special_tokens"].clone();
    // No config.json is fetched as an upstream of record; the staged
    // one above only feeds the static reference check through the
    // files the recipe lists, so keep the list to the staged three.
    recipe["upstream"] = serde_json::json!([
        {"path": "tokenizer.json", "to": "tokenizer.json", "sha256": sha_of(&stage.join("tokenizer.json"))?},
        {"path": "model.safetensors", "to": "weights/model.safetensors", "sha256": sha_of(&stage.join("model.safetensors"))?}
    ]);
    let recipe_path = stage.join("recipe.json");
    fs::write(&recipe_path, serde_json::to_vec_pretty(&recipe).unwrap())
        .map_err(|e| Error::Model(format!("recipe.json: {e}")))?;
    let recipe = turbo_bundle::recipe::Recipe::load(&recipe_path).map_err(Error::Model)?;
    turbo_bundle::make(&recipe, stage, out).map_err(Error::Model)?;
    sha_of(&out.join("manifest.json"))
}

fn sha_of(path: &Path) -> Result<String, Error> {
    let bytes = fs::read(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    Ok(turbo_sha256::sha256(&bytes).to_hex())
}
