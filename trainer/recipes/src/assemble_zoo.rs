//! Building a verified [`turbo_zoo::Artifact`] entry from an
//! [`AssembledBundle`] (#1726).

use std::fs;

use turbo_zoo::{
    Artifact, ArtifactId, Blobs, Component, File, Format, Index, Kind, Language, Licence, LicenceClass, Sha256, Sha512,
    TrainingData, Url,
};

use crate::assemble::AssembledBundle;
use crate::{Error, Result};

/// Metadata for generating a [`turbo_zoo::Artifact`] from an [`AssembledBundle`].
#[derive(Debug, Clone)]
pub struct ZooEntryOptions {
    pub id: String,
    pub kind: Kind,
    pub component: Option<Component>,
    pub language: Option<String>,
    pub version: String,
    pub license_terms: String,
    pub url_prefix: String,
    pub origin: Option<String>,
}

/// Builds and validates a [`turbo_zoo::Artifact`] from `assembled` and `opts`.
///
/// The entry's `trained_on` list and `attribution` are taken from the verified
/// [`CorpusRecord`](crate::corpus::CorpusRecord) and held-out evaluation of
/// every head in `assembled.runs`, and `class` is computed by
/// [`Artifact::computed_class`].
pub fn zoo_artifact(assembled: &AssembledBundle, opts: &ZooEntryOptions) -> Result<Artifact> {
    turbo::bundle::Bundle::open(&assembled.dir)
        .map_err(|e| Error(format!("{}: {}", assembled.dir.display(), e.message)))?;
    let id: ArtifactId = opts.id.parse().map_err(Error)?;
    let language: Option<Language> = match &opts.language {
        Some(l) => Some(l.parse().map_err(Error)?),
        None => None,
    };
    let license: Licence = assembled
        .manifest
        .model
        .license
        .parse()
        .map_err(|e| Error(format!("model.license {:?}: {e}", assembled.manifest.model.license)))?;
    let license_terms: Url = opts.license_terms.parse().map_err(Error)?;
    if let Some(expected_terms) = license.legal_url()
        && license_terms.as_str() != expected_terms
    {
        return Err(Error(format!(
            "license_terms {} does not match model.license {} ({expected_terms})",
            license_terms, license
        )));
    }
    let prefix = opts.url_prefix.trim().trim_end_matches('/');
    if prefix.is_empty() || prefix.contains("://example.") || prefix.contains(".example.") {
        return Err(Error(format!("url_prefix {:?} must be a non-placeholder download URL prefix", opts.url_prefix)));
    }

    let mut rel_paths = vec!["manifest.json".to_owned()];
    for f in &assembled.manifest.files {
        rel_paths.push(f.path.clone());
    }
    let mut files = Vec::with_capacity(rel_paths.len());
    for rel in rel_paths {
        let disk_path = assembled.dir.join(&rel);
        let bytes = fs::read(&disk_path).map_err(|e| Error(format!("{}: {e}", disk_path.display())))?;
        let url: Url = format!("{prefix}/{}/{rel}", id.as_str()).parse().map_err(Error)?;
        files.push(File {
            path: rel.parse().map_err(Error)?,
            url,
            size: bytes.len() as u64,
            sha256: Sha256::of(&bytes),
            sha512: Sha512::of(&bytes),
            download: None,
        });
    }

    let mut trained_on = Vec::with_capacity(assembled.runs.len());
    let mut attributions: Vec<String> = Vec::new();
    for run in &assembled.runs {
        let c = &run.receipt.corpus;
        let parsed_lic: Licence = c.spdx.parse().map_err(Error)?;
        let data_lic = if c.recipe_only && parsed_lic.class() != LicenceClass::LocalOnly {
            Licence::RestrictedPublic
        } else {
            parsed_lic
        };
        let terms: Url = c.terms.parse().map_err(Error)?;
        trained_on.push(TrainingData { name: c.name.clone(), license: data_lic, license_terms: terms });
        let ev = run.receipt.eval.model.as_ref().expect("checked in verify_run");
        let summary = format!(
            "{} ({}): {} {} {:.4} (baseline {:.4}, {})",
            c.attribution.trim(),
            run.spec.name,
            run.receipt.eval.split,
            ev.metric_name(),
            ev.score(),
            run.receipt.eval.baseline.score(),
            c.id
        );
        if !attributions.contains(&summary) {
            attributions.push(summary);
        }
    }

    let mut artifact = Artifact {
        id: id.clone(),
        kind: opts.kind,
        component: opts.component,
        language,
        format: Format::Safetensors,
        version: opts.version.clone(),
        license,
        license_terms,
        trained_on,
        class: LicenceClass::Permissive,
        attribution: (!attributions.is_empty()).then(|| attributions.join(" | ")),
        origin: None,
        files,
    };
    artifact.class = artifact.computed_class();
    if artifact.class == LicenceClass::LocalOnly {
        let orig_str = opts.origin.clone().unwrap_or_else(|| format!("{prefix}/{}", id.as_str()));
        artifact.origin = Some(orig_str.parse().map_err(Error)?);
    }

    let json = Index::to_canonical_json(std::slice::from_ref(&artifact));
    Index::parse(json.as_bytes()).map_err(|e| Error(e.to_string()))?;
    Ok(artifact)
}

/// Copies every file of `assembled` into the content-addressed [`Blobs`] store.
pub fn seed_blobs(assembled: &AssembledBundle, blobs: &Blobs) -> Result<()> {
    let manifest_bytes = fs::read(assembled.dir.join("manifest.json"))
        .map_err(|e| Error(format!("{}: {e}", assembled.dir.join("manifest.json").display())))?;
    blobs.put(&manifest_bytes).map_err(|e| Error(e.to_string()))?;
    for f in &assembled.manifest.files {
        let p = assembled.dir.join(&f.path);
        let bytes = fs::read(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        blobs.put(&bytes).map_err(|e| Error(e.to_string()))?;
    }
    Ok(())
}
