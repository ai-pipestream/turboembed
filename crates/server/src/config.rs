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

    /// Filesystem path to model artifacts (unused by `mock`).
    #[serde(default)]
    pub path: Option<String>,

    /// Backend-specific device hint (e.g. OpenVINO `"CPU"` / `"GPU"` / `"NPU"`).
    #[serde(default)]
    pub device: Option<String>,
}

/// Backend kinds a model can route to. Engine backends are compile-gated;
/// routing to one that was not compiled in fails at startup with a clear
/// error instead of at request time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    Mock,
    LlamaCpp,
    Ort,
    Openvino,
    Apple,
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
            "#,
        )
        .unwrap();
        assert_eq!(config.listen, "127.0.0.1:9000");
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.models[1].backend, BackendKind::LlamaCpp);
        assert!(config.auth.effective_tokens().contains("secret-1"));
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
