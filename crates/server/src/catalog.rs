//! Model alias catalog: logical model names resolved per architecture.
//!
//! Clients address models by a logical name (`minilm`, `default-llm`); the
//! catalog maps that alias to the optimized artifact for the host arch —
//! ORT-CUDA on NVIDIA, an OVMS pipeline on Intel, MLX on Apple — so clients
//! never learn engine paths. Arch configs opt in with
//! `serve = ["minilm", ...]`; [`crate::config::Config::expand_serve`] turns
//! each alias into a regular model entry at startup, so the registry,
//! `ListModels`, and `ModelMetadata` all expose the logical name.
//!
//! The catalog shipped in `config/catalog.toml` is compiled into every
//! binary as [`Catalog::builtin`]. A config can point `catalog = "path"` at
//! a file to replace it per host without rebuilding.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::config::{BackendKind, ModelConfig};

/// The built-in catalog source, compiled from `config/catalog.toml`.
const BUILTIN_CATALOG: &str = include_str!("../../../config/catalog.toml");

/// Host architecture an inferstream binary is built for. Selects which
/// per-arch resolution a catalog alias expands to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Nvidia,
    Intel,
    Apple,
}

impl Arch {
    /// The lowercase name used as the per-arch table key in catalog files.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Nvidia => "nvidia",
            Self::Intel => "intel",
            Self::Apple => "apple",
        }
    }
}

/// Alias → per-arch resolutions. `BTreeMap` keeps listings deterministic.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    #[serde(default)]
    pub models: BTreeMap<String, CatalogEntry>,
}

/// One logical model: what it is, and how each arch serves it.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    /// Human-readable summary surfaced in errors and docs.
    #[serde(default)]
    pub description: Option<String>,

    #[serde(default)]
    pub nvidia: Option<CatalogModelSpec>,

    #[serde(default)]
    pub intel: Option<CatalogModelSpec>,

    #[serde(default)]
    pub apple: Option<CatalogModelSpec>,
}

impl CatalogEntry {
    /// The resolution for `arch`, if this alias is available there.
    pub fn for_arch(&self, arch: Arch) -> Option<&CatalogModelSpec> {
        match arch {
            Arch::Nvidia => self.nvidia.as_ref(),
            Arch::Intel => self.intel.as_ref(),
            Arch::Apple => self.apple.as_ref(),
        }
    }

    /// Arches this alias resolves on, for error messages.
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

/// A [`ModelConfig`] minus `name`: the alias itself becomes the registered
/// model name when the spec is expanded. Field meanings are identical to the
/// `[[models]]` entry documented in [`crate::config::ModelConfig`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogModelSpec {
    pub backend: BackendKind,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub upstream_model: Option<String>,
    #[serde(default)]
    pub engine_dir: Option<String>,
    #[serde(default)]
    pub tokenizer_dir: Option<String>,
    #[serde(default)]
    pub max_batch_size: Option<u32>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub n_gpu_layers: Option<u32>,
    #[serde(default)]
    pub n_ctx: Option<u32>,
    #[serde(default)]
    pub pooling: Option<String>,
    #[serde(default)]
    pub normalize: Option<bool>,
    #[serde(default)]
    pub max_seq_len: Option<u32>,
}

impl CatalogModelSpec {
    /// Expand into a routable model entry registered under `alias`.
    pub fn into_model_config(self, alias: &str) -> ModelConfig {
        ModelConfig {
            name: alias.to_string(),
            backend: self.backend,
            path: self.path,
            device: self.device,
            endpoint: self.endpoint,
            upstream_model: self.upstream_model,
            engine_dir: self.engine_dir,
            tokenizer_dir: self.tokenizer_dir,
            max_batch_size: self.max_batch_size,
            dtype: self.dtype,
            n_gpu_layers: self.n_gpu_layers,
            n_ctx: self.n_ctx,
            pooling: self.pooling,
            normalize: self.normalize,
            max_seq_len: self.max_seq_len,
        }
    }
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
    #[error("unknown model alias {alias:?}; catalog defines: {known}")]
    UnknownAlias { alias: String, known: String },
    #[error(
        "model alias {alias:?} is not available on the {arch} arch \
         (available on: {available}); add a [models.{alias}.{arch}] entry \
         to the catalog or serve it from a host that has it"
    )]
    NotAvailableOnArch {
        alias: String,
        arch: &'static str,
        available: String,
    },
}

impl Catalog {
    /// The catalog compiled into the binary from `config/catalog.toml`.
    pub fn builtin() -> Self {
        Self::from_toml(BUILTIN_CATALOG)
            .expect("built-in config/catalog.toml must parse; covered by unit test")
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

    /// Aliases the catalog defines, in deterministic order.
    pub fn aliases(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }

    /// Resolve `alias` for `arch` into a routable model entry named `alias`.
    ///
    /// Errors distinguish "no such alias" (with the known alias list) from
    /// "alias exists but has no resolution on this arch" (with the arches
    /// that do have one), so startup failures are actionable.
    pub fn resolve(&self, alias: &str, arch: Arch) -> Result<ModelConfig, CatalogError> {
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
        Ok(spec.clone().into_model_config(alias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalog_parses_and_defines_starters() {
        let catalog = Catalog::builtin();
        let aliases: Vec<&str> = catalog.aliases().collect();
        assert!(aliases.contains(&"minilm"), "aliases: {aliases:?}");
        assert!(aliases.contains(&"mpnet"), "aliases: {aliases:?}");
        assert!(aliases.contains(&"default-llm"), "aliases: {aliases:?}");
    }

    #[test]
    fn minilm_resolves_per_arch_to_the_optimized_backend() {
        let catalog = Catalog::builtin();

        let nvidia = catalog.resolve("minilm", Arch::Nvidia).unwrap();
        assert_eq!(nvidia.name, "minilm");
        assert_eq!(nvidia.backend, BackendKind::Ort);
        assert_eq!(nvidia.device.as_deref(), Some("cuda"));
        assert!(nvidia.path.as_deref().unwrap().ends_with("model.onnx"));
        assert_eq!(nvidia.pooling.as_deref(), Some("mean"));

        let intel = catalog.resolve("minilm", Arch::Intel).unwrap();
        assert_eq!(intel.name, "minilm");
        assert_eq!(intel.backend, BackendKind::Ovms);
        assert_eq!(intel.upstream_model.as_deref(), Some("minilm_pipeline"));
        assert!(intel.endpoint.is_some());

        let apple = catalog.resolve("minilm", Arch::Apple).unwrap();
        assert_eq!(apple.name, "minilm");
        assert_eq!(apple.backend, BackendKind::Mlx);
        assert_eq!(
            apple.path.as_deref(),
            Some("mlx-community/all-MiniLM-L6-v2-4bit")
        );
    }

    #[test]
    fn default_llm_resolves_on_all_three_arches() {
        let catalog = Catalog::builtin();
        assert_eq!(
            catalog
                .resolve("default-llm", Arch::Nvidia)
                .unwrap()
                .backend,
            BackendKind::LlamaCpp
        );
        let intel = catalog.resolve("default-llm", Arch::Intel).unwrap();
        assert_eq!(intel.backend, BackendKind::LlamaCpp);
        assert!(intel.endpoint.is_some(), "intel forwards to llama-server");
        assert_eq!(
            catalog.resolve("default-llm", Arch::Apple).unwrap().backend,
            BackendKind::Mlx
        );
    }

    #[test]
    fn unknown_alias_lists_known_aliases() {
        let error = Catalog::builtin()
            .resolve("does-not-exist", Arch::Nvidia)
            .unwrap_err();
        match error {
            CatalogError::UnknownAlias { alias, known } => {
                assert_eq!(alias, "does-not-exist");
                assert!(known.contains("minilm"), "known: {known}");
            }
            other => panic!("expected UnknownAlias, got {other:?}"),
        }
    }

    #[test]
    fn alias_missing_on_arch_names_available_arches() {
        // mpnet only has an intel resolution in the built-in catalog.
        let error = Catalog::builtin()
            .resolve("mpnet", Arch::Nvidia)
            .unwrap_err();
        match error {
            CatalogError::NotAvailableOnArch {
                alias,
                arch,
                available,
            } => {
                assert_eq!(alias, "mpnet");
                assert_eq!(arch, "nvidia");
                assert_eq!(available, "intel");
            }
            other => panic!("expected NotAvailableOnArch, got {other:?}"),
        }
    }

    #[test]
    fn custom_catalog_parses_and_rejects_unknown_fields() {
        let catalog = Catalog::from_toml(
            r#"
            [models.custom-embed]
            description = "test entry"

            [models.custom-embed.apple]
            backend = "mlx"
            path = "org/custom-model"
            normalize = false
            "#,
        )
        .unwrap();
        let resolved = catalog.resolve("custom-embed", Arch::Apple).unwrap();
        assert_eq!(resolved.name, "custom-embed");
        assert_eq!(resolved.normalize, Some(false));

        let result = Catalog::from_toml(
            r#"
            [models.bad.nvidia]
            backend = "ort"
            not_a_field = 1
            "#,
        );
        assert!(matches!(result, Err(CatalogError::Parse(_))));
    }

    #[test]
    fn missing_catalog_file_is_io_error() {
        assert!(matches!(
            Catalog::from_file("/nonexistent/catalog.toml"),
            Err(CatalogError::Io { .. })
        ));
    }
}
