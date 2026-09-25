//! Putting the bundle together: the upstream files it carries, the
//! `files` list, the manifest, and the check that the core loads it.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use turbo::bundle::{Bundle, sha256_hex};
use turbo::manifest::{Manifest, Normalize};
use turbo::safetensors::{self, Dtype};
use turbo::tokenizer::Tokenizer;

use crate::recipe::Recipe;
use crate::{Result, check_rel};

/// Copy the upstream files the bundle carries from `upstream` into `bundle`.
pub fn stage(recipe: &Recipe, upstream: &Path, bundle: &Path) -> Result<()> {
    for u in &recipe.upstream {
        if let Some(to) = &u.to {
            let bytes = fs::read(upstream.join(&u.path)).map_err(|e| format!("upstream {}: {e}", u.path))?;
            crate::fetch::write_atomic(&bundle.join(to), &bytes)?;
        }
    }
    Ok(())
}

/// Every path the manifest names, which is what `files` must list.
pub fn named_paths(manifest: &Value) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    let mut add = |v: &Value, what: &str| -> Result<()> {
        let p = v.as_str().ok_or(format!("manifest.{what}: not a path"))?;
        check_rel(p)?;
        out.insert(p.to_owned());
        Ok(())
    };
    add(&manifest["tokenizer"]["file"], "tokenizer.file")?;
    add(&manifest["reference"]["file"], "reference.file")?;
    if let Some(l) = manifest["model"].get("license_file") {
        add(l, "model.license_file")?;
    }
    for (i, a) in manifest["artifacts"].as_array().ok_or("manifest.artifacts: missing")?.iter().enumerate() {
        for f in a["files"].as_array().ok_or(format!("manifest.artifacts[{i}].files: missing"))? {
            add(f, &format!("artifacts[{i}].files"))?;
        }
        for f in a["produced_by"]["inputs"].as_array().into_iter().flatten() {
            add(f, &format!("artifacts[{i}].produced_by.inputs"))?;
        }
    }
    Ok(out)
}

/// Write `manifest.json` for the files now in `bundle`, then verify it.
/// `produced_by` is the reference's, from the run, and `converted` each
/// converted artifact's name and `produced_by`, from its run.
pub fn seal(recipe: &Recipe, bundle: &Path, produced_by: Value, converted: Vec<(String, Value)>) -> Result<()> {
    let mut m = recipe.manifest.clone();
    m["reference"]["produced_by"] = produced_by;
    let made: Vec<String> = converted.iter().map(|(n, _)| n.clone()).collect();
    let want: Vec<String> = crate::convert::conversions(recipe)?.into_iter().map(|c| c.name).collect();
    if made != want {
        return Err(format!("the recipe converts {want:?}, and the runs made {made:?}"));
    }
    for (name, pb) in converted {
        let artifacts = m["artifacts"].as_array_mut().ok_or("manifest.artifacts: missing")?;
        let a = artifacts.iter_mut().find(|a| a["name"] == name.as_str()).ok_or(format!("no artifact {name}"))?;
        a["produced_by"] = pb;
    }
    let mut files = Vec::new();
    for p in named_paths(&m)? {
        let bytes = fs::read(bundle.join(&p)).map_err(|e| format!("{p}: {e}"))?;
        files.push(json!({ "path": p, "size": bytes.len(), "sha256": sha256_hex(&bytes) }));
    }
    m["files"] = Value::Array(files);
    // Through the core's own parser, so the tool writes only what the
    // loader accepts, in the one canonical form.
    let parsed = Manifest::parse(&serde_json::to_vec(&m).unwrap()).map_err(|e| e.message)?;
    let mut bytes = serde_json::to_vec_pretty(&parsed).unwrap();
    bytes.push(b'\n');
    crate::fetch::write_atomic(&bundle.join("manifest.json"), &bytes)?;
    verify(bundle)
}

/// Check a bundle the way a machine will: loader rules 1 to 5 through the
/// core (the core's tokenizer must give the reference's ids exactly), every
/// listed file by size and hash, nothing unlisted in the directory, and a
/// reference whose vectors are real.
pub fn verify(bundle: &Path) -> Result<()> {
    let b = Bundle::open(bundle).map_err(|e| e.message)?;
    Tokenizer::load(&b).map_err(|e| e.message)?;
    let m = &b.manifest;
    let listed: BTreeSet<String> = m.files.iter().map(|f| f.path.clone()).collect();
    for p in &listed {
        b.read_verified(p).map_err(|e| e.message)?;
    }
    for p in walk(bundle, bundle)? {
        if p != "manifest.json" && !listed.contains(&p) {
            return Err(format!("{p} is in the bundle directory but not in files"));
        }
    }

    let bytes = b.read_verified(&m.reference.file).map_err(|e| e.message)?;
    let st = safetensors::File::parse(&m.reference.file, &bytes).map_err(|e| e.message)?;
    let emb = st.get("embeddings", Dtype::F32, 2).map_err(|e| e.message)?;
    let embed = m.embed();
    let (n, dim) = (m.reference.cases.len(), embed.dim as usize);
    if emb.shape != [n as u64, dim as u64] {
        return Err(format!("{}: embeddings are {:?}, not [{n}, {dim}]", m.reference.file, emb.shape));
    }
    let values = emb.f32s();
    for (i, row) in values.chunks_exact(dim).enumerate() {
        let norm = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        if !row.iter().all(|v| v.is_finite()) || norm == 0.0 {
            return Err(format!("{}: case {i} has no real vector", m.reference.file));
        }
        if embed.normalize == Normalize::L2 && (norm - 1.0).abs() > 1e-4 {
            return Err(format!("{}: case {i} has norm {norm}, and the bundle says NORMALIZE_L2", m.reference.file));
        }
    }
    Ok(())
}

fn walk(root: &Path, dir: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for e in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let path = e.map_err(|e| e.to_string())?.path();
        let meta = fs::symlink_metadata(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if meta.is_dir() {
            out.extend(walk(root, &path)?);
        } else {
            let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            out.push(rel);
        }
    }
    Ok(out)
}
