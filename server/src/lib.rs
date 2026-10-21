//! The Open Inference Protocol (KServe v2) and the analysis result
//! layer over gRPC, served from the library's C interface as
//! docs/kserve.md and docs/grpc-analysis.md map them.

pub mod analysis;
pub mod api;
pub mod capabilities;
pub mod config;
mod infer;
mod kserve;
pub mod results;
pub mod state;
pub mod status;
pub mod stream;

pub use crate::state::Held;

/// The messages and services of the two protos.
pub mod proto {
    #![allow(clippy::all)]
    tonic::include_proto!("inference");
    /// The analysis result layer (docs/grpc-analysis.md).
    pub mod result {
        tonic::include_proto!("turbo.result");
    }
}

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::Code;
use tonic::codegen::http;
use tonic::transport::server::TcpIncoming;

use crate::config::ModelConfig;
use crate::proto::grpc_inference_service_server::GrpcInferenceServiceServer;
use crate::proto::result::analysis_service_server::AnalysisServiceServer;
use crate::state::State;

/// tonic refuses a request message over the decoding limit with
/// OUT_OF_RANGE, read from the frame's length prefix before the message.
/// The services' own refusals all carry turbo-code, so an OUT_OF_RANGE
/// without it is that one, and is answered RESOURCE_EXHAUSTED as other gRPC
/// servers answer it: OUT_OF_RANGE is TURBO_E_CAPACITY's (docs/kserve.md).
fn too_large<B>(mut r: http::Response<B>) -> http::Response<B> {
    let h = r.headers_mut();
    let out_of_range = (Code::OutOfRange as i32).to_string();
    if h.get("grpc-status").is_some_and(|v| v.as_bytes() == out_of_range.as_bytes()) && !h.contains_key("turbo-code") {
        h.insert("grpc-status", (Code::ResourceExhausted as i32).into());
    }
    r
}

/// A server answering gRPC on its listener: the Open Inference Protocol
/// and the analysis result layer on one port. Models are loaded by
/// `load`, until which ServerLive is true and ServerReady false; the
/// analysis service adds and removes bundles while serving.
pub struct Server {
    state: Arc<State>,
    addr: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<std::result::Result<(), tonic::transport::Error>>,
}

/// Whether the server builder serves the admin calls, LoadBundle and
/// UnloadBundle: the explicit opt-in this listener needs before any
/// client may add or remove bundles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Serve {
    pub admin: bool,
}

impl Server {
    /// Binds `listen` and answers gRPC, with nothing loaded yet. A bundle
    /// path that names no bundle, or two models of one name, is refused. A
    /// request message over `max_message_bytes` is refused by gRPC with
    /// RESOURCE_EXHAUSTED before it is read. `serve.admin` opts in to the
    /// admin calls LoadBundle and UnloadBundle; off, they refuse
    /// UNIMPLEMENTED.
    pub async fn start(
        listen: SocketAddr,
        models: Vec<ModelConfig>,
        max_message_bytes: usize,
        serve: Serve,
    ) -> std::result::Result<Server, String> {
        let names = config::names(&models)?;
        let listener = TcpListener::bind(listen).await.map_err(|e| format!("listen on {listen}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| format!("listen on {listen}: {e}"))?;
        let state = Arc::new(State::new(names, models));
        let kserve = GrpcInferenceServiceServer::new(kserve::Inference { state: state.clone() })
            .max_decoding_message_size(max_message_bytes);
        let analysis = AnalysisServiceServer::new(analysis::Analysis { state: state.clone(), admin: serve.admin })
            .max_decoding_message_size(max_message_bytes);
        let (stop, stopped) = oneshot::channel::<()>();
        // TCP_NODELAY on every accepted socket. The builder's tcp_nodelay
        // does not reach a listener served with serve_with_incoming, and
        // without it a response written in two segments waits for the
        // client's delayed ACK (about 40 ms a ModelInfer).
        let incoming = TcpIncoming::from(listener).with_nodelay(Some(true));
        let task = tokio::spawn(
            tonic::transport::Server::builder()
                .layer(tower::util::MapResponseLayer::new(too_large))
                .add_service(kserve)
                .add_service(analysis)
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = stopped.await;
                }),
        );
        Ok(Server { state, addr, stop: Some(stop), task })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Loads every model and makes its sessions. The error names the model,
    /// the call, its status, the field and the library's message.
    pub async fn load(&self) -> std::result::Result<(), String> {
        let state = self.state.clone();
        tokio::task::spawn_blocking(move || state.load()).await.map_err(|e| format!("loading failed: {e}"))?
    }

    /// Takes an idle session of `model` from its pool as a ModelInfer does,
    /// holding it until the returned value is dropped: TURBO_E_BUSY when
    /// every session is held.
    pub fn hold(&self, model: &str) -> api::Result<Held> {
        let (_, loaded) = self.state.ready(model, "")?;
        loaded.take()
    }

    /// The streaming bulk analysis call's gauges: model executions
    /// since the server started, and the documents and bytes queued
    /// across every stream right now, never over the advertised caps.
    pub fn stream_stats(&self) -> stream::Stats {
        self.state.stream.stats()
    }

    /// Stops answering and waits for the server to finish.
    pub async fn stop(mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
        let _ = (&mut self.task).await;
    }
}
