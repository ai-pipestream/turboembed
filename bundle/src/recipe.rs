//! The recipe a bundle is made from.

use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::{Result, check_rel};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// The manifest, without `files`: the tool adds it.
    pub manifest: Value,
    /// Files fetched from `model.source` at its commit.
    pub upstream: Vec<Upstream>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Upstream {
    /// The path in the upstream repository.
    pub path: String,
    /// Where the bundle carries it. Absent: the file is read only by the
    /// reference pipeline (the upstream model's configuration).
    #[serde(default)]
    pub to: Option<String>,
    /// When present, the fetched bytes must have this SHA-256.
    #[serde(default)]
    pub sha256: Option<String>,
}

impl Recipe {
    pub fn load(path: &Path) -> Result<Recipe> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let r: Recipe = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        if r.manifest.get("files").is_some() {
            return Err(format!("{}: manifest.files is written by the tool, not the recipe", path.display()));
        }
        let (_, commit) = r.source()?;
        if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
            return Err(format!("model.source.commit {commit:?} is not a 40-hex commit"));
        }
        // Every path the manifest names, and every conversion, before
        // anything is fetched.
        crate::seal::named_paths(&r.manifest)?;
        crate::convert::conversions(&r)?;
        for u in &r.upstream {
            check_rel(&u.path)?;
            if let Some(to) = &u.to {
                check_rel(to)?;
            }
            if let Some(h) = &u.sha256
                && (h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()))
            {
                return Err(format!("upstream {}: sha256 {h:?} is not 64 lowercase hex", u.path));
            }
        }
        Ok(r)
    }

    /// `model.source.repository` and `.commit`.
    pub fn source(&self) -> Result<(&str, &str)> {
        let s = &self.manifest["model"]["source"];
        match (s["repository"].as_str(), s["commit"].as_str()) {
            (Some(r), Some(c)) => Ok((r, c)),
            _ => Err("manifest.model.source needs repository and commit".into()),
        }
    }

    pub fn str_at<'a>(&'a self, pointer: &str) -> Result<&'a str> {
        self.manifest.pointer(pointer).and_then(Value::as_str).ok_or_else(|| format!("manifest{pointer}: missing"))
    }
}
