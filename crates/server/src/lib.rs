//! inferstream server library: config loading, registry construction, auth,
//! and the tonic server assembly. The binary in `main.rs` is a thin CLI over
//! [`serve`]; integration tests drive the same entry points.

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

use inferstream_backend::Registry;
use inferstream_backend_mock::MockBackend;
use inferstream_protocol::inference::grpc_inference_service_server::GrpcInferenceServiceServer;

use auth::BearerAuth;
use config::{AuthMode, BackendKind, Config};
use service::InferenceService;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error(transparent)]
    Config(#[from] config::ConfigError),
    #[error("model {model:?} routes to backend {backend:?}, which was not compiled in (rebuild with --features {feature})")]
    BackendNotCompiled {
        model: String,
        backend: &'static str,
        feature: &'static str,
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

/// Build the model → backend registry from config.
///
/// Engine backends are compile-gated: routing a model to a backend that was
/// not built in fails here, at startup, with the feature flag to use.
pub fn build_registry(config: &Config) -> Result<Registry, ServerError> {
    let mut registry = Registry::new();
    // One shared instance per backend kind is enough in v0.1; engine
    // backends will likely become instance-per-model once they load weights.
    let mock: Arc<MockBackend> = Arc::new(MockBackend::default());

    for model in &config.models {
        let backend: Arc<dyn inferstream_backend::Backend> = match model.backend {
            BackendKind::Mock => mock.clone(),
            BackendKind::LlamaCpp => {
                #[cfg(feature = "llamacpp")]
                {
                    Arc::new(inferstream_backend_llamacpp::LlamaCppBackend::new(
                        model.path.clone(),
                    ))
                }
                #[cfg(not(feature = "llamacpp"))]
                {
                    return Err(ServerError::BackendNotCompiled {
                        model: model.name.clone(),
                        backend: "llama-cpp",
                        feature: "llamacpp",
                    });
                }
            }
            BackendKind::Ort => {
                #[cfg(feature = "ort")]
                {
                    Arc::new(inferstream_backend_ort::OrtBackend::new(model.path.clone()))
                }
                #[cfg(not(feature = "ort"))]
                {
                    return Err(ServerError::BackendNotCompiled {
                        model: model.name.clone(),
                        backend: "ort",
                        feature: "ort",
                    });
                }
            }
            BackendKind::Openvino => {
                #[cfg(feature = "openvino")]
                {
                    Arc::new(inferstream_backend_openvino::OpenVinoBackend::new(
                        model.path.clone(),
                        model.device.clone(),
                    ))
                }
                #[cfg(not(feature = "openvino"))]
                {
                    return Err(ServerError::BackendNotCompiled {
                        model: model.name.clone(),
                        backend: "openvino",
                        feature: "openvino",
                    });
                }
            }
            BackendKind::Apple => {
                #[cfg(feature = "apple")]
                {
                    Arc::new(inferstream_backend_apple::AppleBackend::new(
                        model.path.clone(),
                    ))
                }
                #[cfg(not(feature = "apple"))]
                {
                    return Err(ServerError::BackendNotCompiled {
                        model: model.name.clone(),
                        backend: "apple",
                        feature: "apple",
                    });
                }
            }
        };
        registry.register(model.name.clone(), backend);
    }
    Ok(registry)
}

/// Bind the configured address and serve until `shutdown` resolves.
///
/// Returns the bound local address via `bound_tx` before serving, so callers
/// (tests, health tooling) can bind port 0 and discover the real port.
pub async fn serve(
    config: Config,
    bound_tx: tokio::sync::oneshot::Sender<SocketAddr>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    let registry = Arc::new(build_registry(&config)?);
    let service = InferenceService::new(registry);

    let addr: SocketAddr = config
        .listen
        .parse()
        .map_err(|_| ServerError::InvalidAddr(config.listen.clone()))?;
    let listener = TcpListener::bind(addr).await.map_err(|source| ServerError::Bind {
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
