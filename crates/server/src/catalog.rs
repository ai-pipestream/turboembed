//! Model alias catalog: logical model names resolved per architecture.
//!
//! Clients address models by a logical name (`minilm`, `default-llm`); the
//! catalog maps that alias to the optimized artifact for the host arch —
//! ORT-CUDA on NVIDIA, in-process OpenVINO GenAI on Intel, MLX on Apple —
//! so clients never learn engine paths. Arch configs opt in with
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

    /// Support matrix for the built-in catalog: every alias × arch, true
    /// where a real backend path exists. Keep in sync with
    /// config/catalog.toml — this is the executable version of the README
    /// alias table.
    const BUILTIN_MATRIX: &[(&str, [bool; 3])] = &[
        // alias                (nvidia, intel, apple)
        ("minilm", [true, true, true]),
        ("minilm-l12", [true, true, true]),
        ("mpnet", [true, true, false]),
        ("bge-small", [true, true, true]),
        ("bge-base", [true, true, true]),
        ("bge-large", [true, true, true]),
        ("bge-m3", [true, true, true]),
        ("e5-small", [true, true, true]),
        ("e5-base", [true, true, true]),
        ("e5-large", [true, true, true]),
        ("gte-small", [true, true, true]),
        ("gte-base", [true, true, true]),
        ("nomic-embed-text", [true, true, false]),
        ("default-llm", [true, true, true]),
        ("qwen-0.5b", [true, true, true]),
        ("qwen-7b", [true, true, true]),
    ];

    const LLM_ALIASES: &[&str] = &["default-llm", "qwen-0.5b", "qwen-7b"];

    fn is_llm(alias: &str) -> bool {
        LLM_ALIASES.contains(&alias)
    }

    #[test]
    fn builtin_catalog_defines_exactly_the_documented_aliases() {
        let catalog = Catalog::builtin();
        let expected: Vec<&str> = {
            let mut v: Vec<&str> = BUILTIN_MATRIX.iter().map(|(alias, _)| *alias).collect();
            v.sort_unstable();
            v
        };
        let actual: Vec<&str> = catalog.aliases().collect();
        assert_eq!(actual, expected, "catalog aliases drifted from the matrix");
    }

    /// Every alias × arch: supported combinations resolve to a config named
    /// by the alias; unsupported ones fail with NotAvailableOnArch (never
    /// UnknownAlias, never a fake resolution).
    #[test]
    fn builtin_matrix_resolves_supported_and_rejects_unsupported() {
        let catalog = Catalog::builtin();
        for (alias, supported) in BUILTIN_MATRIX {
            for (arch, expect) in [Arch::Nvidia, Arch::Intel, Arch::Apple]
                .into_iter()
                .zip(supported)
            {
                match catalog.resolve(alias, arch) {
                    Ok(model) if *expect => {
                        assert_eq!(model.name, *alias, "registered under the logical name");
                    }
                    Ok(model) => panic!(
                        "{alias} must NOT resolve on {arch:?} but got backend {:?}",
                        model.backend
                    ),
                    Err(CatalogError::NotAvailableOnArch { available, .. }) if !expect => {
                        assert!(!available.is_empty(), "{alias}: available list empty");
                    }
                    Err(e) => panic!("{alias} on {arch:?}: unexpected error {e:?}"),
                }
            }
        }
    }

    /// Embedding entries must carry the settings the engines need: ORT
    /// entries a path + tokenizer_dir + pooling, OpenVINO GenAI entries a
    /// model dir + GPU device + pooling, MLX entries a model path. Pooling
    /// matches each family's convention.
    #[test]
    fn builtin_embedding_entries_are_engine_complete() {
        let catalog = Catalog::builtin();
        for (alias, [nvidia, intel, apple]) in BUILTIN_MATRIX {
            if is_llm(alias) {
                continue;
            }
            if *nvidia {
                let m = catalog.resolve(alias, Arch::Nvidia).unwrap();
                assert_eq!(m.backend, BackendKind::Ort, "{alias} nvidia");
                assert!(m.path.is_some(), "{alias} nvidia needs a path");
                assert!(m.tokenizer_dir.is_some(), "{alias} nvidia tokenizer");
                let expected_pooling = if alias.starts_with("bge") {
                    "cls"
                } else {
                    "mean"
                };
                assert_eq!(
                    m.pooling.as_deref(),
                    Some(expected_pooling),
                    "{alias} nvidia pooling"
                );
            }
            if *intel {
                let m = catalog.resolve(alias, Arch::Intel).unwrap();
                assert_eq!(m.backend, BackendKind::Openvino, "{alias} intel");
                let path = m
                    .path
                    .as_deref()
                    .unwrap_or_else(|| panic!("{alias} intel needs a GenAI model dir"));
                assert!(
                    path.starts_with("models/ov/"),
                    "{alias} intel path={path} (expected models/ov/<alias>)"
                );
                assert_eq!(m.device.as_deref(), Some("GPU"), "{alias} intel device");
                assert!(m.tokenizer_dir.is_some(), "{alias} intel tokenizer");
                let expected_pooling = if alias.starts_with("bge") {
                    "cls"
                } else {
                    "mean"
                };
                assert_eq!(
                    m.pooling.as_deref(),
                    Some(expected_pooling),
                    "{alias} intel pooling"
                );
                assert!(m.endpoint.is_none(), "{alias} intel must be in-process");
            }
            if *apple {
                let m = catalog.resolve(alias, Arch::Apple).unwrap();
                assert_eq!(m.backend, BackendKind::Mlx, "{alias} apple");
                assert!(m.path.is_some(), "{alias} apple needs an HF repo/path");
                let expected_pooling = if alias.starts_with("bge") {
                    "cls"
                } else {
                    "mean"
                };
                assert_eq!(
                    m.pooling.as_deref(),
                    Some(expected_pooling),
                    "{alias} apple pooling"
                );
            }
        }
    }

    /// LLM entries must carry a real generation path: llama.cpp-CUDA + GGUF
    /// on nvidia, llama.cpp-SYCL in-process + GGUF on intel, native MLX on apple.
    /// Default aliases never use HTTP to a llama-server.
    #[test]
    fn builtin_llm_entries_are_engine_complete() {
        let catalog = Catalog::builtin();
        for alias in LLM_ALIASES {
            let (nvidia, intel, apple) = BUILTIN_MATRIX
                .iter()
                .find(|(name, _)| name == alias)
                .map(|(_, flags)| (flags[0], flags[1], flags[2]))
                .expect("LLM alias listed in BUILTIN_MATRIX");

            if nvidia {
                let m = catalog.resolve(alias, Arch::Nvidia).unwrap();
                assert_eq!(m.backend, BackendKind::LlamaCpp, "{alias} nvidia");
                assert_eq!(m.device.as_deref(), Some("cuda"), "{alias} nvidia device");
                let path = m
                    .path
                    .as_deref()
                    .unwrap_or_else(|| panic!("{alias} nvidia needs a GGUF path"));
                assert!(path.ends_with(".gguf"), "{alias} nvidia path={path}");
                assert_eq!(m.n_ctx, Some(4096), "{alias} nvidia n_ctx");
                assert!(m.endpoint.is_none(), "{alias} nvidia must be in-process");
            }
            if intel {
                let m = catalog.resolve(alias, Arch::Intel).unwrap();
                assert_eq!(m.backend, BackendKind::LlamaCpp, "{alias} intel");
                assert_eq!(m.device.as_deref(), Some("sycl"), "{alias} intel device");
                let path = m
                    .path
                    .as_deref()
                    .unwrap_or_else(|| panic!("{alias} intel needs a GGUF path"));
                assert!(path.ends_with(".gguf"), "{alias} intel path={path}");
                assert_eq!(m.n_ctx, Some(4096), "{alias} intel n_ctx");
                assert!(
                    m.endpoint.is_none(),
                    "{alias} intel must be in-process SYCL (no llama-server HTTP)"
                );
            }
            if apple {
                let m = catalog.resolve(alias, Arch::Apple).unwrap();
                assert_eq!(m.backend, BackendKind::Mlx, "{alias} apple");
                let path = m
                    .path
                    .as_deref()
                    .unwrap_or_else(|| panic!("{alias} apple needs an MLX repo"));
                assert!(path.starts_with("models/mlx/"), "{alias} apple path={path}");
                assert!(
                    m.tokenizer_dir.is_some(),
                    "{alias} apple Tokenize needs tokenizer_dir"
                );
            }
        }
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
        assert_eq!(intel.backend, BackendKind::Openvino);
        assert_eq!(intel.device.as_deref(), Some("GPU"));
        assert_eq!(intel.path.as_deref(), Some("models/ov/minilm"));
        assert_eq!(intel.pooling.as_deref(), Some("mean"));
        assert!(intel.endpoint.is_none());

        let apple = catalog.resolve("minilm", Arch::Apple).unwrap();
        assert_eq!(apple.name, "minilm");
        assert_eq!(apple.backend, BackendKind::Mlx);
        assert_eq!(apple.path.as_deref(), Some("models/mlx/minilm"));
        assert_eq!(apple.pooling.as_deref(), Some("mean"));
        assert_eq!(apple.max_seq_len, Some(256));
    }

    #[test]
    fn default_llm_resolves_per_arch_to_the_generation_backend() {
        let catalog = Catalog::builtin();

        let nvidia = catalog.resolve("default-llm", Arch::Nvidia).unwrap();
        assert_eq!(nvidia.name, "default-llm");
        assert_eq!(nvidia.backend, BackendKind::LlamaCpp);
        assert_eq!(nvidia.device.as_deref(), Some("cuda"));
        assert!(nvidia
            .path
            .as_deref()
            .unwrap()
            .ends_with("qwen2.5-0.5b-instruct-q8_0.gguf"));

        let intel = catalog.resolve("default-llm", Arch::Intel).unwrap();
        assert_eq!(intel.name, "default-llm");
        assert_eq!(intel.backend, BackendKind::LlamaCpp);
        assert_eq!(intel.device.as_deref(), Some("sycl"));
        assert!(intel
            .path
            .as_deref()
            .unwrap()
            .ends_with("qwen2.5-0.5b-instruct-q8_0.gguf"));
        assert!(intel.endpoint.is_none());

        let apple = catalog.resolve("default-llm", Arch::Apple).unwrap();
        assert_eq!(apple.name, "default-llm");
        assert_eq!(apple.backend, BackendKind::Mlx);
        assert_eq!(apple.path.as_deref(), Some("models/mlx/qwen-0.5b"));
        assert_eq!(
            apple.tokenizer_dir.as_deref(),
            Some("models/gguf/qwen-0.5b")
        );
    }

    #[test]
    fn qwen_05b_resolves_on_intel_to_the_05b_gguf() {
        let intel = Catalog::builtin()
            .resolve("qwen-0.5b", Arch::Intel)
            .unwrap();
        assert_eq!(intel.device.as_deref(), Some("sycl"));
        assert!(intel
            .path
            .as_deref()
            .unwrap()
            .ends_with("qwen2.5-0.5b-instruct-q8_0.gguf"));
        assert!(!intel.path.as_deref().unwrap().contains("7b"));
        assert!(intel.endpoint.is_none());
    }

    #[test]
    fn qwen_7b_resolves_on_every_arch() {
        let catalog = Catalog::builtin();
        let nvidia = catalog.resolve("qwen-7b", Arch::Nvidia).unwrap();
        assert!(nvidia
            .path
            .as_deref()
            .unwrap()
            .contains("qwen2.5-7b-instruct-q5_k_m"));
        let intel = catalog.resolve("qwen-7b", Arch::Intel).unwrap();
        assert!(intel
            .path
            .as_deref()
            .unwrap()
            .contains("qwen2.5-7b-instruct-q5_k_m"));
        assert!(intel.endpoint.is_none());
        let apple = catalog.resolve("qwen-7b", Arch::Apple).unwrap();
        assert_eq!(apple.path.as_deref(), Some("models/mlx/qwen-7b"));
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
        // mpnet has no apple resolution (mlx-embeddings lacks MPNet).
        let error = Catalog::builtin()
            .resolve("mpnet", Arch::Apple)
            .unwrap_err();
        match error {
            CatalogError::NotAvailableOnArch {
                alias,
                arch,
                available,
            } => {
                assert_eq!(alias, "mpnet");
                assert_eq!(arch, "apple");
                assert_eq!(available, "nvidia, intel");
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
