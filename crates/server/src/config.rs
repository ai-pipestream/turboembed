//! Server configuration: listen address, auth, and model → backend routing.
//!
//! Loaded from TOML (see `config/example.toml`). Every model entry names the
//! backend kind that serves it; the registry is built once at startup.

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;

use crate::catalog::{Arch, Catalog, CatalogError};

/// Top-level server configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Socket address the gRPC server binds, e.g. `"127.0.0.1:8461"`.
    #[serde(default = "default_listen")]
    pub listen: String,

    #[serde(default)]
    pub auth: AuthConfig,

    /// Logical model aliases to serve from the catalog (`"minilm"`,
    /// `"default-llm"`, ...). Each alias is resolved for the binary's arch
    /// at startup ([`Config::expand_serve`]) into a regular model entry, so
    /// clients address the alias directly in `model_name`.
    #[serde(default)]
    pub serve: Vec<String>,

    /// Path to a catalog file overriding the built-in
    /// [`Catalog::builtin`] (compiled from `config/catalog.toml`). Only
    /// consulted when `serve` is non-empty.
    #[serde(default)]
    pub catalog: Option<String>,

    /// Model routing table. Explicit entries; `serve` aliases are appended
    /// here after catalog expansion.
    #[serde(default)]
    pub models: Vec<ModelConfig>,
}

fn default_listen() -> String {
    "127.0.0.1:8461".to_string()
}

/// Authentication configuration.
///
/// v0.1 supports static bearer tokens (API keys). Tokens can be listed
/// inline and/or supplied via the `INFERSTREAM_API_KEYS` environment
/// variable (comma-separated), which is the recommended way to keep secrets
/// out of config files. mTLS termination is a deliberate later hook: tonic's
/// `ServerTlsConfig::client_ca_root` slots into `serve()` without touching
/// the service layer.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    /// Auth mode: `"none"` (default) or `"bearer"`.
    #[serde(default)]
    pub mode: AuthMode,

    /// Static bearer tokens accepted when `mode = "bearer"`.
    #[serde(default)]
    pub bearer_tokens: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMode {
    #[default]
    None,
    Bearer,
}

impl AuthConfig {
    /// Effective token set: config tokens plus `INFERSTREAM_API_KEYS`
    /// (comma-separated) from the environment.
    pub fn effective_tokens(&self) -> HashSet<String> {
        let mut tokens: HashSet<String> = self
            .bearer_tokens
            .iter()
            .filter(|t| !t.is_empty())
            .cloned()
            .collect();
        if let Ok(env_keys) = std::env::var("INFERSTREAM_API_KEYS") {
            tokens.extend(
                env_keys
                    .split(',')
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .map(String::from),
            );
        }
        tokens
    }
}

/// One routed model.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Model name clients address in `model_name`.
    pub name: String,

    /// Backend kind serving this model.
    pub backend: BackendKind,

    /// Filesystem path to model artifacts (GGUF file, `.onnx`, IR `.xml`,
    /// MLX model directory; unused by `mock` and `trt-llm`).
    #[serde(default)]
    pub path: Option<String>,

    /// Backend-specific device hint. OpenVINO: `"CPU"` / `"GPU"` / `"NPU"` /
    /// `"AUTO"`. llama.cpp: `"cuda"` / `"sycl"` / `"metal"` / `"vulkan"` /
    /// `"cpu"` (must match how the llama.cpp library was built).
    #[serde(default)]
    pub device: Option<String>,

    /// Upstream endpoint for client backends. `backend = "llama-cpp"`: a
    /// running llama-server's HTTP base URL, e.g. `"http://127.0.0.1:8085"`
    /// (server-client mode; models without `path` fall back to
    /// `INFERSTREAM_LLAMACPP_ENDPOINT`).
    #[serde(default)]
    pub endpoint: Option<String>,

    /// TensorRT-LLM: directory containing the compiled engine
    /// (`rank0.engine` + `config.json`). Required for `backend = "trt-llm"`.
    #[serde(default)]
    pub engine_dir: Option<String>,

    /// Tokenizer artifacts directory (TRT-LLM and other engines that
    /// tokenize server-side when clients send `text` instead of ids).
    #[serde(default)]
    pub tokenizer_dir: Option<String>,

    /// Maximum concurrent batch size the engine schedules (TRT-LLM
    /// `max_batch_size`; llama.cpp `n_parallel`).
    #[serde(default)]
    pub max_batch_size: Option<u32>,

    /// Engine compute dtype hint, e.g. `"fp16"`, `"bf16"`, `"fp8"`, `"int8"`.
    #[serde(default)]
    pub dtype: Option<String>,

    /// llama.cpp: layers to offload to the accelerator (`n_gpu_layers`);
    /// omit for full offload.
    #[serde(default)]
    pub n_gpu_layers: Option<u32>,

    /// llama.cpp: context window (`n_ctx`); omit for 4096 capped to the
    /// model's training context.
    #[serde(default)]
    pub n_ctx: Option<u32>,

    /// Embedding backends (ort / openvino): pooling strategy, `"mean"`
    /// (default), `"cls"`, or `"last"` (OpenVINO GenAI LAST_TOKEN).
    #[serde(default)]
    pub pooling: Option<String>,

    /// Embedding backends (ort): L2-normalize outputs (default true).
    #[serde(default)]
    pub normalize: Option<bool>,

    /// Embedding backends (ort): tokenizer truncation length (default 512).
    #[serde(default)]
    pub max_seq_len: Option<u32>,
}

/// Backend kinds a model can route to.
///
/// Which kinds are actually constructible depends on the binary: each arch
/// binary (`inferstream-nvidia` / `inferstream-intel` / `inferstream-apple`)
/// registers a factory for the engines it compiled in. Routing to a kind the
/// binary does not support fails at startup with a clear error, never at
/// request time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    /// Deterministic mock; available in every binary.
    Mock,
    /// TensorRT-LLM in-process Executor (NVIDIA peak path).
    TrtLlm,
    /// llama.cpp / GGUF; device chosen by `device` + how the lib was built.
    LlamaCpp,
    /// ONNX Runtime.
    Ort,
    /// OpenVINO GenAI in-process (Intel CPU / GPU / NPU).
    Openvino,
    /// Apple MLX (native macOS host only).
    Mlx,
    /// TurboRerank C ABI (catalog MiniLM-L6 cross-encoder).
    #[serde(rename = "turborerank")]
    TurboRerank,
}

impl BackendKind {
    /// The kebab-case name used in config files, for error messages.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::TrtLlm => "trt-llm",
            Self::LlamaCpp => "llama-cpp",
            Self::Ort => "ort",
            Self::Openvino => "openvino",
            Self::Mlx => "mlx",
            Self::TurboRerank => "turborerank",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse config: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("duplicate model name {0:?} in routing table")]
    DuplicateModel(String),
    #[error("auth mode is \"bearer\" but no tokens are configured (set [auth] bearer_tokens or INFERSTREAM_API_KEYS)")]
    NoBearerTokens,
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error(
        "config lists serve = [...] catalog aliases, but this binary has no \
         arch catalog (the dev `inferstream` binary serves only explicit \
         [[models]] entries); use inferstream-nvidia / inferstream-intel / \
         inferstream-apple, or list models explicitly"
    )]
    ServeNeedsArch,
}

impl Config {
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let config: Config = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml(&text)
    }

    /// Expand `serve` catalog aliases into regular model entries for `arch`.
    ///
    /// Loads the `catalog` file when set (otherwise the built-in catalog),
    /// resolves every alias for `arch`, and appends the results to `models`
    /// under the alias name — so the registry, `ListModels`, and
    /// `ModelMetadata` expose the logical name clients use. Call once at
    /// startup, before building the registry. A config with a non-empty
    /// `serve` list but no arch (`None`, the dev binary) is an error.
    pub fn expand_serve(&mut self, arch: Option<Arch>) -> Result<(), ConfigError> {
        if self.serve.is_empty() {
            return Ok(());
        }
        let arch = arch.ok_or(ConfigError::ServeNeedsArch)?;
        let catalog = match &self.catalog {
            Some(path) => Catalog::from_file(path)?,
            None => Catalog::builtin(),
        };
        // Drain the alias list: each alias becomes a concrete model entry.
        for alias in std::mem::take(&mut self.serve) {
            self.models.push(catalog.resolve(&alias, arch)?);
        }
        // Re-check invariants: an alias may collide with an explicit
        // [[models]] entry or a repeated serve item.
        self.validate()
    }

    fn validate(&self) -> Result<(), ConfigError> {
        let mut seen = HashSet::new();
        for model in &self.models {
            if !seen.insert(model.name.as_str()) {
                return Err(ConfigError::DuplicateModel(model.name.clone()));
            }
        }
        // serve aliases must not collide with explicit model names even
        // before expansion, so the error appears at parse time too.
        for alias in &self.serve {
            if !seen.insert(alias.as_str()) {
                return Err(ConfigError::DuplicateModel(alias.clone()));
            }
        }
        if self.auth.mode == AuthMode::Bearer && self.auth.effective_tokens().is_empty() {
            return Err(ConfigError::NoBearerTokens);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_example_shape() {
        let config = Config::from_toml(
            r#"
            listen = "127.0.0.1:9000"

            [auth]
            mode = "bearer"
            bearer_tokens = ["secret-1"]

            [[models]]
            name = "mock-embed"
            backend = "mock"

            [[models]]
            name = "llama"
            backend = "llama-cpp"
            path = "/models/llama.gguf"
            device = "cuda"
            n_gpu_layers = 99

            [[models]]
            name = "llama-70b"
            backend = "trt-llm"
            engine_dir = "/engines/llama-70b-fp8"
            tokenizer_dir = "/engines/llama-70b-fp8/tokenizer"
            max_batch_size = 64
            dtype = "fp8"
            "#,
        )
        .unwrap();
        assert_eq!(config.listen, "127.0.0.1:9000");
        assert_eq!(config.models.len(), 3);
        assert_eq!(config.models[1].backend, BackendKind::LlamaCpp);
        assert_eq!(config.models[1].device.as_deref(), Some("cuda"));
        assert_eq!(config.models[2].backend, BackendKind::TrtLlm);
        assert_eq!(
            config.models[2].engine_dir.as_deref(),
            Some("/engines/llama-70b-fp8")
        );
        assert_eq!(config.models[2].max_batch_size, Some(64));
        assert!(config.auth.effective_tokens().contains("secret-1"));
    }

    #[test]
    fn rejects_removed_ovms_backend() {
        let result = Config::from_toml(
            r#"
            [[models]]
            name = "minilm_pipeline"
            backend = "ovms"
            endpoint = "http://172.22.0.2:8000"
            "#,
        );
        assert!(
            matches!(result, Err(ConfigError::Parse(_))),
            "backend = \"ovms\" was removed; Intel embeds use backend = \"openvino\""
        );
    }

    #[test]
    fn parses_nvidia_ort_embedding_model() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "minilm-l6-v2"
            backend = "ort"
            device = "cuda"
            path = "/models/minilm/onnx/model.onnx"
            pooling = "mean"
            normalize = true
            max_seq_len = 256
            tokenizer_dir = "/models/minilm"
            "#,
        )
        .unwrap();
        let model = &config.models[0];
        assert_eq!(model.backend, BackendKind::Ort);
        assert_eq!(model.device.as_deref(), Some("cuda"));
        assert_eq!(model.pooling.as_deref(), Some("mean"));
        assert_eq!(model.normalize, Some(true));
        assert_eq!(model.max_seq_len, Some(256));
        assert_eq!(model.tokenizer_dir.as_deref(), Some("/models/minilm"));
    }

    #[test]
    fn parses_turborerank_ce_model() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "ms-marco-minilm-l6"
            backend = "turborerank"
            device = "cuda"
            path = "models/rerank/ms-marco-minilm-l6"
            max_batch_size = 32
            "#,
        )
        .unwrap();
        assert_eq!(config.models[0].backend, BackendKind::TurboRerank);
        assert_eq!(config.models[0].backend.as_str(), "turborerank");
        assert_eq!(config.models[0].device.as_deref(), Some("cuda"));
        assert_eq!(config.models[0].max_batch_size, Some(32));
    }

    #[test]
    fn parses_apple_mlx_model() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "mlx-embed"
            backend = "mlx"
            path = "/models/mlx-embed"
            "#,
        )
        .unwrap();
        assert_eq!(config.models[0].backend, BackendKind::Mlx);
        assert_eq!(config.models[0].backend.as_str(), "mlx");
    }

    #[test]
    fn rejects_unknown_model_fields() {
        let result = Config::from_toml(
            r#"
            [[models]]
            name = "m"
            backend = "mock"
            not_a_field = true
            "#,
        );
        assert!(matches!(result, Err(ConfigError::Parse(_))));
    }

    #[test]
    fn rejects_unknown_backend_kind() {
        let result = Config::from_toml(
            r#"
            [[models]]
            name = "m"
            backend = "not-a-backend"
            "#,
        );
        assert!(matches!(result, Err(ConfigError::Parse(_))));
    }

    #[test]
    fn default_listen_and_empty_models_parse() {
        let config = Config::from_toml("").unwrap();
        assert_eq!(config.listen, "127.0.0.1:8461");
        assert!(config.models.is_empty());
        assert_eq!(config.auth.mode, AuthMode::None);
    }

    #[test]
    fn env_api_keys_merge_with_config_tokens() {
        in_auth_environment(
            "env_api_keys_merge_with_config_tokens",
            Some("env-a, env-b,,"),
            || {
                let auth = AuthConfig {
                    mode: AuthMode::Bearer,
                    bearer_tokens: vec!["file-key".into(), String::new()],
                };
                let tokens = auth.effective_tokens();
                assert!(tokens.contains("file-key"));
                assert!(tokens.contains("env-a"));
                assert!(tokens.contains("env-b"));
                assert_eq!(tokens.len(), 3, "empty entries are dropped");
            },
        );
    }

    #[test]
    fn rejects_duplicate_models() {
        let result = Config::from_toml(
            r#"
            [[models]]
            name = "m"
            backend = "mock"
            [[models]]
            name = "m"
            backend = "mock"
            "#,
        );
        assert!(matches!(result, Err(ConfigError::DuplicateModel(_))));
    }

    #[test]
    fn expand_serve_resolves_aliases_for_the_arch() {
        let mut config = Config::from_toml(
            r#"
            serve = ["minilm", "bge-small", "e5-small"]

            [[models]]
            name = "mock-embed"
            backend = "mock"
            "#,
        )
        .unwrap();
        config.expand_serve(Some(Arch::Nvidia)).unwrap();

        assert!(config.serve.is_empty(), "aliases drained into models");
        let names: Vec<&str> = config.models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["mock-embed", "minilm", "bge-small", "e5-small"]);

        let minilm = &config.models[1];
        assert_eq!(minilm.backend, BackendKind::Ort);
        assert_eq!(minilm.device.as_deref(), Some("cuda"));
        assert!(minilm.tokenizer_dir.is_some(), "tokenizer rides along");

        // Family pooling conventions survive expansion.
        assert_eq!(config.models[2].pooling.as_deref(), Some("cls"));
        assert_eq!(config.models[3].pooling.as_deref(), Some("mean"));
    }

    #[test]
    fn expand_serve_same_alias_resolves_differently_per_arch() {
        let toml = r#"serve = ["minilm"]"#;

        let mut intel = Config::from_toml(toml).unwrap();
        intel.expand_serve(Some(Arch::Intel)).unwrap();
        assert_eq!(intel.models[0].backend, BackendKind::Openvino);
        assert_eq!(intel.models[0].device.as_deref(), Some("GPU"));
        assert_eq!(intel.models[0].path.as_deref(), Some("models/ov/minilm"));

        let mut apple = Config::from_toml(toml).unwrap();
        apple.expand_serve(Some(Arch::Apple)).unwrap();
        assert_eq!(apple.models[0].backend, BackendKind::Mlx);
        assert_eq!(apple.models[0].name, "minilm", "clients use the alias");
    }

    #[test]
    fn expand_serve_unknown_alias_fails_with_catalog_error() {
        let mut config = Config::from_toml(r#"serve = ["not-in-catalog"]"#).unwrap();
        let error = config.expand_serve(Some(Arch::Nvidia)).unwrap_err();
        assert!(
            matches!(
                error,
                ConfigError::Catalog(CatalogError::UnknownAlias { .. })
            ),
            "{error:?}"
        );
    }

    #[test]
    fn expand_serve_alias_missing_on_arch_fails() {
        // mpnet resolves on nvidia and intel, but not apple (mlx-embeddings
        // has no MPNet forward pass).
        let mut config = Config::from_toml(r#"serve = ["mpnet"]"#).unwrap();
        assert!(config.expand_serve(Some(Arch::Intel)).is_ok());

        let mut config = Config::from_toml(r#"serve = ["mpnet"]"#).unwrap();
        let error = config.expand_serve(Some(Arch::Apple)).unwrap_err();
        assert!(
            matches!(
                error,
                ConfigError::Catalog(CatalogError::NotAvailableOnArch { .. })
            ),
            "{error:?}"
        );
    }

    #[test]
    fn expand_serve_without_arch_is_rejected() {
        let mut config = Config::from_toml(r#"serve = ["minilm"]"#).unwrap();
        assert!(matches!(
            config.expand_serve(None),
            Err(ConfigError::ServeNeedsArch)
        ));

        // No serve list: the dev binary path is unaffected.
        let mut config = Config::from_toml("").unwrap();
        assert!(config.expand_serve(None).is_ok());
    }

    #[test]
    fn serve_alias_colliding_with_explicit_model_is_rejected_at_parse() {
        let result = Config::from_toml(
            r#"
            serve = ["minilm"]

            [[models]]
            name = "minilm"
            backend = "mock"
            "#,
        );
        assert!(matches!(result, Err(ConfigError::DuplicateModel(_))));

        let result = Config::from_toml(r#"serve = ["minilm", "minilm"]"#);
        assert!(matches!(result, Err(ConfigError::DuplicateModel(_))));
    }

    #[test]
    fn catalog_file_override_replaces_builtin() {
        let path = std::env::temp_dir().join(format!(
            "inferstream-catalog-test-{}.toml",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"
            [models.minilm.nvidia]
            backend = "ort"
            device = "cpu"
            path = "/custom/minilm.onnx"
            "#,
        )
        .unwrap();
        let mut config = Config::from_toml(&format!(
            "serve = [\"minilm\"]\ncatalog = {:?}\n",
            path.to_str().unwrap()
        ))
        .unwrap();
        config.expand_serve(Some(Arch::Nvidia)).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(config.models[0].device.as_deref(), Some("cpu"));
        assert_eq!(
            config.models[0].path.as_deref(),
            Some("/custom/minilm.onnx")
        );
    }

    /// The example configs shipped in config/ must parse and expand for
    /// their arch, so `model_name: "minilm"` works on all three.
    #[test]
    fn shipped_arch_configs_serve_minilm() {
        let config_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config");
        for (file, arch) in [
            ("nvidia.toml", Arch::Nvidia),
            ("intel.toml", Arch::Intel),
            ("apple.toml", Arch::Apple),
        ] {
            let mut config = Config::from_file(format!("{config_dir}/{file}"))
                .unwrap_or_else(|e| panic!("{file} must parse: {e}"));
            config
                .expand_serve(Some(arch))
                .unwrap_or_else(|e| panic!("{file} must expand for {arch:?}: {e}"));
            assert!(
                config.models.iter().any(|m| m.name == "minilm"),
                "{file} must serve the minilm alias"
            );
        }
        // Multi-alias defaults: intel serves in-process GenAI embeds plus
        // in-process SYCL LLM aliases; apple serves the small-download
        // embedding trio plus default-llm + qwen-0.5b (same MLX 4-bit);
        // nvidia serves default-llm + qwen-0.5b + qwen-7b (fetch first).
        let mut nvidia = Config::from_file(format!("{config_dir}/nvidia.toml")).unwrap();
        nvidia.expand_serve(Some(Arch::Nvidia)).unwrap();
        for alias in ["default-llm", "qwen-0.5b", "qwen-7b"] {
            assert!(
                nvidia.models.iter().any(|m| m.name == alias),
                "nvidia.toml must serve {alias}"
            );
        }
        let mut intel = Config::from_file(format!("{config_dir}/intel.toml")).unwrap();
        intel.expand_serve(Some(Arch::Intel)).unwrap();
        assert!(intel.models.iter().any(|m| m.name == "mpnet"));
        for alias in ["default-llm", "qwen-0.5b", "qwen-7b"] {
            assert!(
                intel.models.iter().any(|m| m.name == alias),
                "intel.toml must serve {alias}"
            );
        }
        let mut apple = Config::from_file(format!("{config_dir}/apple.toml")).unwrap();
        apple.expand_serve(Some(Arch::Apple)).unwrap();
        for alias in ["minilm", "default-llm", "qwen-0.5b"] {
            assert!(
                apple.models.iter().any(|m| m.name == alias),
                "apple.toml must serve {alias}"
            );
        }
        // The dev config has no serve list and stays arch-neutral.
        let mut example = Config::from_file(format!("{config_dir}/example.toml")).unwrap();
        assert!(example.serve.is_empty());
        example.expand_serve(None).unwrap();
    }

    #[test]
    fn bearer_mode_requires_tokens() {
        in_auth_environment("bearer_mode_requires_tokens", None, || {
            let result = Config::from_toml(
                r#"
                [auth]
                mode = "bearer"
                "#,
            );
            assert!(matches!(result, Err(ConfigError::NoBearerTokens)));
        });
    }

    fn in_auth_environment(name: &str, keys: Option<&str>, test: impl FnOnce()) {
        const CHILD: &str = "INFERSTREAM_AUTH_CONFIG_TEST_CHILD";
        if std::env::var(CHILD).as_deref() == Ok(name) {
            test();
            return;
        }
        // Other tests also read auth configuration. A test-local mutex cannot
        // isolate a process-wide environment write from those readers.
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", &format!("config::tests::{name}"), "--nocapture"])
            .env(CHILD, name)
            .env_remove("INFERSTREAM_API_KEYS");
        if let Some(keys) = keys {
            command.env("INFERSTREAM_API_KEYS", keys);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{name}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
