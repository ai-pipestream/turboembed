//! Built-in alias catalog reader (availability only).
//!
//! The harness does not resolve artifacts. It only needs to know whether
//! `config/catalog.toml` has a `[models.<alias>.<arch>]` table so it can
//! soft-skip with `NotAvailableOnArch` instead of hard-failing.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::target::Target;

const BUILTIN_CATALOG: &str = include_str!("../../../config/catalog.toml");

#[derive(Debug, Deserialize)]
struct CatalogFile {
    #[serde(default)]
    models: BTreeMap<String, AliasArches>,
}

/// Presence of a per-arch table is enough; field contents are ignored.
#[derive(Debug, Deserialize)]
struct AliasArches {
    #[serde(default)]
    nvidia: Option<toml::Value>,
    #[serde(default)]
    intel: Option<toml::Value>,
    #[serde(default)]
    apple: Option<toml::Value>,
}

impl AliasArches {
    fn available_arches(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.nvidia.is_some() {
            out.push("nvidia");
        }
        if self.intel.is_some() {
            out.push("intel");
        }
        if self.apple.is_some() {
            out.push("apple");
        }
        out
    }
}

/// Alias → arches that have a real catalog resolution.
#[derive(Debug, Clone)]
pub struct CatalogIndex {
    models: BTreeMap<String, Vec<&'static str>>,
}

impl CatalogIndex {
    pub fn builtin() -> Result<Self, String> {
        Self::from_toml(BUILTIN_CATALOG)
    }

    pub fn from_toml(text: &str) -> Result<Self, String> {
        let file: CatalogFile = toml::from_str(text).map_err(|e| e.to_string())?;
        let models = file
            .models
            .into_iter()
            .map(|(alias, arches)| {
                let available = arches.available_arches();
                // Keep the per-target probe cheap: store available names, and
                // reconstruct `has` from that list (mock is always true).
                (alias, available)
            })
            .collect();
        Ok(Self { models })
    }

    pub fn aliases(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }

    /// Whether the catalog has a real resolution for `alias` on `target`.
    ///
    /// `Target::Mock` is always available (the mock server registers whatever
    /// names the test config asks for).
    pub fn available_on(&self, alias: &str, target: Target) -> bool {
        if target.is_mock() {
            return true;
        }
        self.models
            .get(alias)
            .map(|arches| arches.iter().any(|a| *a == target.as_str()))
            .unwrap_or(false)
    }

    /// Arches that *do* resolve this alias, for skip messages.
    pub fn available_arches(&self, alias: &str) -> Vec<&str> {
        self.models
            .get(alias)
            .map(|v| v.to_vec())
            .unwrap_or_default()
    }

    pub fn known(&self, alias: &str) -> bool {
        self.models.contains_key(alias)
    }
}

/// Why an alias should not be exercised on this target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    NotAvailableOnArch {
        alias: String,
        arch: &'static str,
        available: String,
    },
    NotServed {
        alias: String,
    },
    NotReady {
        alias: String,
    },
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailableOnArch {
                alias,
                arch,
                available,
            } => write!(
                f,
                "NotAvailableOnArch: {alias} has no catalog resolution on {arch} (available on: {available})"
            ),
            Self::NotServed { alias } => {
                write!(f, "not served: {alias} is not in ListModels on this host")
            }
            Self::NotReady { alias } => {
                write!(f, "not ready: {alias} is listed but ModelReady is false")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_matches_documented_matrix() {
        let idx = CatalogIndex::builtin().expect("catalog parses");
        assert!(idx.available_on("minilm", Target::Nvidia));
        assert!(idx.available_on("minilm", Target::Intel));
        assert!(idx.available_on("minilm", Target::Apple));
        assert!(idx.available_on("mpnet", Target::Nvidia));
        assert!(idx.available_on("mpnet", Target::Intel));
        assert!(!idx.available_on("mpnet", Target::Apple));
        assert!(idx.available_on("nomic-embed-text", Target::Nvidia));
        assert!(!idx.available_on("nomic-embed-text", Target::Apple));
        assert!(idx.available_on("qwen-0.5b", Target::Intel));
        assert!(idx.available_on("default-llm", Target::Apple));
        assert!(idx.available_on("minilm", Target::Mock));
    }
}
