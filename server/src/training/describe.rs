//! What a static model in the training directory is, read from its sealed
//! manifest, with every file it lists read again and checked against the
//! size and SHA-256 recorded there.

use std::fs;
use std::path::Path;

use turbo::bundle::Bundle;

use super::{STAGING, serving};
use crate::proto::training::{ArtifactFile, StaticModelDescriptor, StaticModelQuality, TeacherProvenance};
use crate::state::State;

/// The static model sealed in `dir`, named `name`. A directory that holds
/// no bundle, or a bundle that is not a static model, is an error; a file
/// that does not verify is the descriptor's `problem`.
pub(crate) fn describe(dir: &Path, name: &str, served: bool) -> Result<StaticModelDescriptor, String> {
    let b = Bundle::open(dir).map_err(|e| e.message)?;
    let m = &b.manifest;
    let st = m.static_embedding.as_ref().ok_or("not a static model: the manifest has no static_embedding")?;
    let args = m.artifacts.first().and_then(|a| a.produced_by.as_ref()).map(|p| p.args.as_slice()).unwrap_or(&[]);
    let arg = |key: &str| args.iter().find_map(|a| a.strip_prefix(key)).unwrap_or_default().to_owned();
    // Every listed file is read whole and checked, the first that fails
    // named; the manifest's own digest was checked by the open.
    let problem = m.files.iter().find_map(|f| b.read_verified(&f.path).err()).map(|e| e.message).unwrap_or_default();
    Ok(StaticModelDescriptor {
        name: name.to_owned(),
        model_id: m.model.id.clone(),
        revision: m.model.revision.clone(),
        manifest_sha256: b.manifest_sha256.clone(),
        dims: m.embed.as_ref().map_or(0, |e| e.dim),
        vocab_size: st.vocab_size,
        learned_terms: arg("learned_terms=").parse().unwrap_or(0),
        dtype: arg("dtype="),
        teacher: st.distilled_from.as_ref().map(|d| TeacherProvenance {
            model_id: d.model_id.clone(),
            revision: d.revision.clone(),
            manifest_sha256: d.manifest_sha256.clone(),
        }),
        quality: st.quality.as_ref().map(|q| StaticModelQuality {
            base_top1: q.base_top1,
            static_top1: q.static_top1,
            similarity_spearman: q.similarity_spearman,
        }),
        files: m
            .files
            .iter()
            .map(|f| ArtifactFile { path: f.path.clone(), size: f.size, sha256: f.sha256.clone() })
            .collect(),
        verified: problem.is_empty(),
        problem,
        served,
    })
}

/// Every model in the training directory, in name order. A child that
/// does not describe is listed unverified, with why.
pub(crate) fn list(state: &State, dir: &Path) -> Result<Vec<StaticModelDescriptor>, String> {
    let mut names = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let Ok(name) = entry.file_name().into_string() else { continue };
        // Children only: a link is not followed out of the directory.
        let dir = entry.file_type().is_ok_and(|t| t.is_dir());
        if dir && !name.starts_with(STAGING) && !name.starts_with('.') {
            names.push(name);
        }
    }
    names.sort();
    Ok(names
        .into_iter()
        .map(|name| {
            let path = dir.join(&name);
            let served = serving(state, &name, &path).is_some();
            describe(&path, &name, served).unwrap_or_else(|problem| StaticModelDescriptor {
                name,
                problem,
                served,
                ..Default::default()
            })
        })
        .collect())
}
