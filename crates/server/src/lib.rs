//! inferstream shared server library.
//!
//! This crate is the arch-neutral core every inferstream binary shares:
//! the OIP V2 gRPC service, bearer auth, config parsing, and the model →
//! backend registry. It knows nothing about GPU engines. Each arch binary
//! (`inferstream-nvidia`, `inferstream-intel`, `inferstream-apple`, and the
//! mock-only `inferstream` dev binary) supplies a [`BackendFactory`] that
//! constructs the engines it compiled in; everything else is shared.

// tonic::Status is 176 bytes and the interceptor contract requires
// Result<Request<T>, Status>; boxing is not an option here.
#![allow(clippy::result_large_err)]

pub mod auth;
pub mod config;
pub mod service;

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tracing::info;

use inferstream_backend::{Backend, Registry};
use inferstream_backend_mock::MockBackend;
use inferstream_protocol::inference::grpc_inference_service_server::GrpcInferenceServiceServer;

use auth::BearerAuth;
use config::{AuthMode, BackendKind, Config, ModelConfig};
use service::InferenceService;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error(transparent)]
    Config(#[from] config::ConfigError),
    #[error(
        "model {model:?} routes to backend {backend:?}, which this binary does not support: {hint}"
    )]
    UnsupportedBackend {
        model: String,
        backend: &'static str,
        hint: String,
    },
    #[error("model {model:?} ({backend}): {message}")]
    InvalidModelConfig {
        model: String,
        backend: &'static str,
        message: String,
    },
    #[error("failed to bind {addr}: {source}")]
    Bind {
        addr: String,
        source: std::io::Error,
    },
    #[error("invalid listen address {0:?}")]
    InvalidAddr(String),
    #[error("gRPC server error: {0}")]
    Transport(#[from] tonic::transport::Error),
}

/// Constructs the backend serving one configured model.
///
/// Each arch binary implements this for the engines it compiled in. Return
/// [`ServerError::UnsupportedBackend`] (via [`unsupported`]) for kinds the
/// binary does not ship so startup fails with an actionable message.
pub trait BackendFactory: Send + Sync {
    fn create(&self, model: &ModelConfig) -> Result<Arc<dyn Backend>, ServerError>;
}

impl<F> BackendFactory for F
where
    F: Fn(&ModelConfig) -> Result<Arc<dyn Backend>, ServerError> + Send + Sync,
{
    fn create(&self, model: &ModelConfig) -> Result<Arc<dyn Backend>, ServerError> {
        self(model)
    }
}

/// Helper for factories: a well-formed "wrong binary / missing feature" error.
pub fn unsupported(model: &ModelConfig, hint: impl Into<String>) -> ServerError {
    ServerError::UnsupportedBackend {
        model: model.name.clone(),
        backend: model.backend.as_str(),
        hint: hint.into(),
    }
}

/// Factory for the mock backend only — what the arch-neutral `inferstream`
/// dev binary uses, and a building block for arch factories (every arch
/// binary also serves `mock` so GPU workers can smoke-test the wire path
/// before engines are loaded).
pub fn mock_factory() -> impl BackendFactory {
    let mock: Arc<MockBackend> = Arc::new(MockBackend::default());
    move |model: &ModelConfig| match model.backend {
        BackendKind::Mock => Ok(mock.clone() as Arc<dyn Backend>),
        _ => Err(unsupported(
            model,
            "this binary only serves the mock backend; run the arch binary that ships this \
             engine (inferstream-nvidia / inferstream-intel / inferstream-apple)",
        )),
    }
}

/// Build the model → backend registry from config using an arch factory.
pub fn build_registry(
    config: &Config,
    factory: &dyn BackendFactory,
) -> Result<Registry, ServerError> {
    let mut registry = Registry::new();
    for model in &config.models {
        registry.register(model.name.clone(), factory.create(model)?);
    }
    Ok(registry)
}

/// Bind the configured address and serve `registry` until `shutdown`
/// resolves.
///
/// Returns the bound local address via `bound_tx` before serving, so callers
/// (tests, health tooling) can bind port 0 and discover the real port.
pub async fn serve(
    config: Config,
    registry: Registry,
    bound_tx: tokio::sync::oneshot::Sender<SocketAddr>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    let service = InferenceService::new(Arc::new(registry));

    let addr: SocketAddr = config
        .listen
        .parse()
        .map_err(|_| ServerError::InvalidAddr(config.listen.clone()))?;
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|source| ServerError::Bind {
            addr: config.listen.clone(),
            source,
        })?;
    let local_addr = listener.local_addr().map_err(|source| ServerError::Bind {
        addr: config.listen.clone(),
        source,
    })?;
    let _ = bound_tx.send(local_addr);

    let auth = match config.auth.mode {
        AuthMode::Bearer => Some(BearerAuth::new(config.auth.effective_tokens())),
        AuthMode::None => None,
    };
    info!(
        addr = %local_addr,
        auth = ?config.auth.mode,
        models = config.models.len(),
        "inferstream listening"
    );

    let interceptor = move |request: tonic::Request<()>| match &auth {
        Some(bearer) => bearer.check(request),
        None => Ok(request),
    };
    let server = GrpcInferenceServiceServer::with_interceptor(service, interceptor);

    Server::builder()
        .add_service(server)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
        .await?;
    Ok(())
}

/// Convenience for tests and simple callers: build the registry from config
/// with `factory`, then [`serve`].
pub async fn serve_with_factory(
    config: Config,
    factory: &dyn BackendFactory,
    bound_tx: tokio::sync::oneshot::Sender<SocketAddr>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    let registry = build_registry(&config, factory)?;
    serve(config, registry, bound_tx, shutdown).await
}

/// Shared CLI entry point for every inferstream binary.
///
/// Parses `--config` / `--listen`, initializes tracing, builds the registry
/// through the binary's factory, and serves until Ctrl-C.
pub async fn run_cli(
    binary_name: &'static str,
    factory: &dyn BackendFactory,
) -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;

    /// Multi-backend KServe OIP V2 gRPC streaming inference façade.
    #[derive(Debug, Parser)]
    #[command(version)]
    struct Args {
        /// Path to the TOML config file.
        #[arg(
            long,
            short,
            env = "INFERSTREAM_CONFIG",
            default_value = "config/example.toml"
        )]
        config: String,

        /// Override the listen address from the config file.
        #[arg(long, env = "INFERSTREAM_LISTEN")]
        listen: Option<String>,
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let mut config = Config::from_file(&args.config)?;
    if let Some(listen) = args.listen {
        config.listen = listen;
    }
    info!(binary = binary_name, config = %args.config, "starting");

    let (bound_tx, _bound_rx) = tokio::sync::oneshot::channel();
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received");
    };
    serve_with_factory(config, factory, bound_tx, shutdown).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_factory_rejects_engine_kinds() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "m"
            backend = "trt-llm"
            engine_dir = "/engines/x"
            "#,
        )
        .unwrap();
        let result = build_registry(&config, &mock_factory());
        assert!(matches!(
            result,
            Err(ServerError::UnsupportedBackend {
                backend: "trt-llm",
                ..
            })
        ));
    }

    #[test]
    fn mock_factory_builds_mock_models() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "m"
            backend = "mock"
            "#,
        )
        .unwrap();
        let registry = build_registry(&config, &mock_factory()).unwrap();
        assert!(registry.lookup("m").is_some());
    }
}
