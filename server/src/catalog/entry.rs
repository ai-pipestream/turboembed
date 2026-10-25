//! An index artifact as a catalog entry, and whether the cache holds it.

use std::fs;
use std::path::Path;

use tonic::Status;
use turbo_zoo::{Artifact, Policy, UserId};

use super::{Catalog, terms, zoo_status};
use crate::proto::training::{CatalogEntry, CatalogFile, InstalledModel, TrainingSource};

/// The index's artifacts, in its order, of `kind` and in `language` when
/// those are given. With a user, each entry's missing terms are the ones
/// that user has not accepted.
pub(super) fn list(
    catalog: &Catalog,
    user: Option<&UserId>,
    kind: &str,
    language: &str,
) -> Result<Vec<CatalogEntry>, Status> {
    let acceptances = match user {
        Some(_) => Some(catalog.acceptances().map_err(zoo_status)?),
        None => None,
    };
    let cache = catalog.cache();
    let mut out = Vec::new();
    for a in catalog.index.list() {
        if !kind.is_empty() && a.kind.to_string() != kind {
            continue;
        }
        if !language.is_empty() && a.language.as_ref().is_none_or(|l| l.as_str() != language) {
            continue;
        }
        let needed = a.licences_to_accept(Policy::default());
        let missing = match (user, &acceptances) {
            (Some(u), Some(acc)) => needed.iter().filter(|r| !acc.has(u, r)).map(terms).collect(),
            _ => Vec::new(),
        };
        out.push(CatalogEntry {
            id: a.id.to_string(),
            kind: a.kind.to_string(),
            component: a.component.map(|c| c.to_string()).unwrap_or_default(),
            language: a.language.as_ref().map(|l| l.to_string()).unwrap_or_default(),
            format: a.format.to_string(),
            version: a.version.clone(),
            licence: a.license.to_string(),
            licence_terms: a.license_terms.to_string(),
            licence_class: a.class.to_string(),
            attribution: a.attribution.clone().unwrap_or_default(),
            trained_on: a
                .trained_on
                .iter()
                .map(|t| TrainingSource {
                    name: t.name.clone(),
                    licence: t.license.to_string(),
                    licence_terms: t.license_terms.to_string(),
                })
                .collect(),
            files: a.files.iter().map(file).collect(),
            total_bytes: a.files.iter().map(|f| f.size).sum(),
            terms: needed.iter().map(terms).collect(),
            missing_terms: missing,
            installed: installed(&cache, a).is_some(),
        });
    }
    Ok(out)
}

fn file(f: &turbo_zoo::File) -> CatalogFile {
    let d = f.download.as_ref();
    CatalogFile {
        path: f.path.as_str().to_owned(),
        url: f.url.to_string(),
        size: f.size,
        sha256: f.sha256.to_string(),
        sha512: f.sha512.to_string(),
        compression: d.map(|d| compression(d.compression).to_owned()).unwrap_or_default(),
        download_bytes: d.map_or(0, |d| d.download_bytes),
        download_sha256: d.map(|d| d.download_sha256.to_string()).unwrap_or_default(),
    }
}

fn compression(c: turbo_zoo::Compression) -> &'static str {
    match c {
        turbo_zoo::Compression::Gzip => "gzip",
        turbo_zoo::Compression::Zstd => "zstd",
    }
}

/// The artifact as the cache holds it, when every one of its files is a
/// plain file there of the size the index records. The digests are not
/// read here: an install, and every fetch, checks them.
pub(super) fn installed(cache: &Path, a: &Artifact) -> Option<InstalledModel> {
    let dir = cache.join(a.id.as_str());
    let mut files = Vec::with_capacity(a.files.len());
    for f in &a.files {
        let p = dir.join(f.path.as_str());
        let m = fs::symlink_metadata(&p).ok()?;
        if !m.is_file() || m.len() != f.size {
            return None;
        }
        files.push(p.to_string_lossy().into_owned());
    }
    Some(InstalledModel {
        id: a.id.to_string(),
        version: a.version.clone(),
        dir: dir.to_string_lossy().into_owned(),
        files,
        total_bytes: a.files.iter().map(|f| f.size).sum(),
    })
}
