//! Server configuration: listen address, auth, and model → backend routing.
//!
//! Loaded from TOML (see `config/example.toml`). Every model entry names the
//! backend kind that serves it; the registry is built once at startup.

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;

/// Top-level server configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Socket address the gRPC server binds, e.g. `"127.0.0.1:8461"`.
    #[serde(default = "default_listen")]
    pub listen: String,

    #[serde(default)]
    pub auth: AuthConfig,

    /// Model routing table.
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

    /// Upstream endpoint for client backends. `backend = "ovms"`: the gRPC
    /// listener, e.g. `"http://172.22.0.2:8000"` (OVMS `--port`, not
    /// `--rest_port`; the model `name` must match a model or pipeline the
    /// upstream serves; falls back to `INFERSTREAM_OVMS_ENDPOINT`).
    /// `backend = "llama-cpp"`: a running llama-server's HTTP base URL, e.g.
    /// `"http://127.0.0.1:8085"` (server-client mode; models without `path`
    /// fall back to `INFERSTREAM_LLAMACPP_ENDPOINT`).
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

    /// Embedding backends (ort): pooling strategy, `"mean"` (default) or
    /// `"cls"`.
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
    /// OpenVINO (Intel CPU / GPU / NPU).
    Openvino,
    /// OpenVINO Model Server (or any KServe V2 gRPC server) reached over the
    /// network; inferstream forwards requests instead of executing in-process.
    Ovms,
    /// Apple MLX (native macOS host only).
    Mlx,
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
            Self::Ovms => "ovms",
            Self::Mlx => "mlx",
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

    fn validate(&self) -> Result<(), ConfigError> {
        let mut seen = HashSet::new();
        for model in &self.models {
            if !seen.insert(model.name.as_str()) {
                return Err(ConfigError::DuplicateModel(model.name.clone()));
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
    fn parses_ovms_client_backend() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "minilm_pipeline"
            backend = "ovms"
            endpoint = "http://172.22.0.2:8000"
            "#,
        )
        .unwrap();
        assert_eq!(config.models[0].backend, BackendKind::Ovms);
        assert_eq!(config.models[0].backend.as_str(), "ovms");
        assert_eq!(
            config.models[0].endpoint.as_deref(),
            Some("http://172.22.0.2:8000")
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
        // Serialize env mutation: cargo runs tests in parallel threads.
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("INFERSTREAM_API_KEYS", "env-a, env-b,,");
        let auth = AuthConfig {
            mode: AuthMode::Bearer,
            bearer_tokens: vec!["file-key".into(), String::new()],
        };
        let tokens = auth.effective_tokens();
        std::env::remove_var("INFERSTREAM_API_KEYS");
        assert!(tokens.contains("file-key"));
        assert!(tokens.contains("env-a"));
        assert!(tokens.contains("env-b"));
        assert_eq!(tokens.len(), 3, "empty entries are dropped");
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
    fn bearer_mode_requires_tokens() {
        let result = Config::from_toml(
            r#"
            [auth]
            mode = "bearer"
            "#,
        );
        assert!(matches!(result, Err(ConfigError::NoBearerTokens)));
    }
}
