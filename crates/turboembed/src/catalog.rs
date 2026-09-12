//! Thin reader for `config/catalog.toml` (same file the server compiles in).
//!
//! turboembed does not depend on `inferstream-server` — that crate pulls tonic.
//! Field meanings match `[models.<alias>.<arch>]` in the catalog.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

const BUILTIN_CATALOG: &str = include_str!("../../../config/catalog.toml");

const LLM_ALIASES: &[&str] = &["default-llm", "qwen-0.5b", "qwen-7b"];

/// Host architecture selecting the per-arch catalog table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Nvidia,
    Intel,
    Apple,
}

impl Arch {
    #[allow(dead_code)]
    pub fn parse(s: &str) -> Result<Self, CatalogError> {
        match s.to_ascii_lowercase().as_str() {
            "nvidia" => Ok(Self::Nvidia),
            "intel" => Ok(Self::Intel),
            "apple" => Ok(Self::Apple),
            other => Err(CatalogError::UnknownArch {
                arch: other.to_string(),
            }),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nvidia => "nvidia",
            Self::Intel => "intel",
            Self::Apple => "apple",
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    #[serde(default)]
    pub models: BTreeMap<String, CatalogEntry>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    #[serde(default)]
    #[allow(dead_code)]
    pub description: Option<String>,
    #[serde(default)]
    pub nvidia: Option<CatalogModelSpec>,
    #[serde(default)]
    pub intel: Option<CatalogModelSpec>,
    #[serde(default)]
    pub apple: Option<CatalogModelSpec>,
}

impl CatalogEntry {
    pub fn for_arch(&self, arch: Arch) -> Option<&CatalogModelSpec> {
        match arch {
            Arch::Nvidia => self.nvidia.as_ref(),
            Arch::Intel => self.intel.as_ref(),
            Arch::Apple => self.apple.as_ref(),
        }
    }

    fn available_arches(&self) -> Vec<&'static str> {
        [
            (Arch::Nvidia, self.nvidia.is_some()),
            (Arch::Intel, self.intel.is_some()),
            (Arch::Apple, self.apple.is_some()),
        ]
        .into_iter()
        .filter(|(_, present)| *present)
        .map(|(arch, _)| arch.as_str())
        .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogModelSpec {
    pub backend: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub endpoint: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub engine_dir: Option<String>,
    #[serde(default)]
    pub tokenizer_dir: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub max_batch_size: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub dtype: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub n_gpu_layers: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub n_ctx: Option<u32>,
    #[serde(default)]
    pub pooling: Option<String>,
    #[serde(default)]
    pub normalize: Option<bool>,
    #[serde(default)]
    pub max_seq_len: Option<u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("failed to read catalog file {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse catalog: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("unknown arch {arch:?}; expected \"nvidia\", \"intel\" or \"apple\"")]
    #[allow(dead_code)]
    UnknownArch { arch: String },
    #[error("unknown model alias {alias:?}; catalog defines: {known}")]
    UnknownAlias { alias: String, known: String },
    #[error(
        "model alias {alias:?} is not available on the {arch} arch \
         (available on: {available})"
    )]
    NotAvailableOnArch {
        alias: String,
        arch: &'static str,
        available: String,
    },
    #[error(
        "catalog alias {alias:?} has backend={backend:?}; turboembed refuses \
         mock (and any non-engine backend) for catalog names. Rebuild with \
         the real provider feature for this arch"
    )]
    MockForbidden { alias: String, backend: String },
    #[error("catalog alias {alias:?} is a generative LLM, not an embedder")]
    NotAnEmbedder { alias: String },
}

impl Catalog {
    pub fn builtin() -> Self {
        Self::from_toml(BUILTIN_CATALOG)
            .expect("built-in config/catalog.toml must parse")
    }

    pub fn from_toml(text: &str) -> Result<Self, CatalogError> {
        Ok(toml::from_str(text)?)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, CatalogError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| CatalogError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml(&text)
    }

    pub fn aliases(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }

    /// Resolve an embedding alias. Mock backends and LLM aliases are errors.
    pub fn resolve_embed(&self, alias: &str, arch: Arch) -> Result<&CatalogModelSpec, CatalogError> {
        if LLM_ALIASES.contains(&alias) {
            return Err(CatalogError::NotAnEmbedder {
                alias: alias.to_string(),
            });
        }
        let entry = self
            .models
            .get(alias)
            .ok_or_else(|| CatalogError::UnknownAlias {
                alias: alias.to_string(),
                known: self.aliases().collect::<Vec<_>>().join(", "),
            })?;
        let spec = entry
            .for_arch(arch)
            .ok_or_else(|| CatalogError::NotAvailableOnArch {
                alias: alias.to_string(),
                arch: arch.as_str(),
                available: entry.available_arches().join(", "),
            })?;
        if spec.backend.eq_ignore_ascii_case("mock") {
            return Err(CatalogError::MockForbidden {
                alias: alias.to_string(),
                backend: spec.backend.clone(),
            });
        }
        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_minilm_nvidia_is_ort_cuda() {
        let catalog = Catalog::builtin();
        let spec = catalog.resolve_embed("minilm", Arch::Nvidia).unwrap();
        assert_eq!(spec.backend, "ort");
        assert_eq!(spec.device.as_deref(), Some("cuda"));
        assert_eq!(spec.pooling.as_deref(), Some("mean"));
        assert_eq!(spec.normalize, Some(true));
        assert!(spec.path.as_deref().unwrap().ends_with("model.onnx"));
    }

    #[test]
    fn builtin_catalog_never_resolves_mock() {
        let catalog = Catalog::builtin();
        for alias in catalog.aliases() {
            if LLM_ALIASES.contains(&alias) {
                continue;
            }
            if let Ok(spec) = catalog.resolve_embed(alias, Arch::Nvidia) {
                assert_ne!(
                    spec.backend, "mock",
                    "{alias} nvidia must not be mock"
                );
            }
        }
    }

    #[test]
    fn llm_alias_is_not_an_embedder() {
        assert!(matches!(
            Catalog::builtin().resolve_embed("default-llm", Arch::Nvidia),
            Err(CatalogError::NotAnEmbedder { .. })
        ));
    }

    #[test]
    fn mock_backend_in_catalog_is_rejected() {
        let catalog = Catalog::from_toml(
            r#"
            [models.minilm.nvidia]
            backend = "mock"
            "#,
        )
        .unwrap();
        assert!(matches!(
            catalog.resolve_embed("minilm", Arch::Nvidia),
            Err(CatalogError::MockForbidden { .. })
        ));
    }
}
