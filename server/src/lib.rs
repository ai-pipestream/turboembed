//! The Open Inference Protocol (KServe v2), the analysis result layer
//! and the model catalog over gRPC, served from the library's C interface as
//! docs/kserve.md and docs/grpc-analysis.md map them.

pub mod analysis;
pub mod api;
pub mod capabilities;
pub mod catalog;
pub mod config;
pub mod deadline;
pub mod format;
mod health;
mod infer;
pub mod inspect;
mod kserve;
pub mod metrics;
pub mod offsets;
pub mod probe;
pub mod progressive;
pub mod results;
pub mod search;
pub mod sink;
pub mod state;
pub mod status;
pub mod stream;
pub mod tasks;
pub mod tls;
mod training;
pub mod vocabulary;

pub use crate::state::Held;
pub use crate::tls::TlsConfig;

/// The messages and services of the protos.
pub mod proto {
    #![allow(clippy::all)]
    tonic::include_proto!("inference");
    /// The analysis result layer (docs/grpc-analysis.md).
    pub mod result {
        pub mod v1 {
            tonic::include_proto!("turbo.result.v1");
        }
        pub use v1::*;
    }
    /// Hybrid search, index lifecycle, aliases, and collection drift (issue #1221).
    pub mod search {
        pub mod v1 {
            tonic::include_proto!("turbo.search.v1");
        }
        pub use v1::*;
    }
    /// The gRPC health checking protocol (`grpc.health.v1.Health`).
    pub mod health {
        tonic::include_proto!("grpc.health.v1");
    }
    /// The downstream document sink (docs/grpc-sink.md).
    pub mod sink {
        pub mod v1 {
            tonic::include_proto!("turbo.sink.v1");
        }
        pub use v1::*;
    }
    /// The vocabulary store's service (docs/grpc-vocabulary.md).
    pub mod vocabulary {
        pub mod v1 {
            tonic::include_proto!("turbo.vocabulary.v1");
        }
        pub use v1::*;
    }
    /// The model catalog and static model training (docs/grpc-catalog.md,
    /// docs/grpc-training.md).
    pub mod training {
        pub mod v1 {
            tonic::include_proto!("turbo.training.v1");
        }
        pub use v1::*;
    }
}

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::Code;
use tonic::codegen::http;
use tonic::transport::server::TcpIncoming;

use crate::config::ModelConfig;
use crate::proto::grpc_inference_service_server::GrpcInferenceServiceServer;
use crate::proto::health::health_server::HealthServer;
use crate::proto::result::analysis_service_server::AnalysisServiceServer;
use crate::proto::search::turbo_search_service_server::TurboSearchServiceServer;
use crate::proto::sink::turbo_sink_service_server::TurboSinkServiceServer;
use crate::proto::training::turbo_catalog_service_server::TurboCatalogServiceServer;
use crate::proto::training::turbo_training_service_server::TurboTrainingServiceServer;
use crate::proto::vocabulary::turbo_vocabulary_service_server::TurboVocabularyServiceServer;
use crate::sink::{SinkConfig, SinkForwarder, SinkReceiver, SinkStats};
use crate::state::State;

/// tonic refuses a request message over the decoding limit with
/// OUT_OF_RANGE, read from the frame's length prefix before the message.
/// The services' own refusals all carry turbo-code, so an OUT_OF_RANGE
/// without it is that one, and is answered RESOURCE_EXHAUSTED as other gRPC
/// servers answer it: OUT_OF_RANGE is TURBO_E_CAPACITY's (docs/kserve.md).
pub(crate) fn too_large<B>(mut r: http::Response<B>) -> http::Response<B> {
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
    search_registry: Arc<search::SearchRegistry>,
    addr: SocketAddr,
    metrics_addr: Option<SocketAddr>,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<std::result::Result<(), tonic::transport::Error>>,
    metrics_stop: Option<oneshot::Sender<()>>,
    metrics_task: Option<JoinHandle<()>>,
    sink: Option<Arc<SinkForwarder>>,
}

/// What the server serves beyond inference and analysis. `admin` is the
/// explicit opt-in this listener needs before any client may add or remove
/// bundles or vocabulary artifacts, record a licence acceptance or install
/// a model. `vocabulary` is the vocabulary store's directory; without one
/// the store holds nothing and refuses writes. `catalog` is the model
/// catalog, which refuses every call when absent. `training` is the
/// directory static models are trained into; without one only
/// ListTeachers is served. `sink` forwards every
/// document an analysis stream completes to a downstream TurboSinkService;
/// `sink_receiver` serves that service here, handing it each accepted item.
#[derive(Debug, Clone, Default)]
pub struct Serve {
    pub admin: bool,
    pub vocabulary: Option<PathBuf>,
    pub catalog: Option<catalog::Catalog>,
    pub training: Option<PathBuf>,
    pub sink: Option<SinkConfig>,
    pub sink_receiver: Option<SinkReceiver>,
    pub tls: Option<TlsConfig>,
    pub metrics_listen: Option<SocketAddr>,
}

impl Server {
    /// Binds `listen` and answers gRPC, with nothing loaded yet. A bundle
    /// path that names no bundle, or two models of one name, is refused. A
    /// request message over `max_message_bytes` is refused by gRPC with
    /// RESOURCE_EXHAUSTED before it is read. `serve.admin` opts in to the
    /// admin calls LoadBundle, UnloadBundle, RecordLicenceAcceptance,
    /// InstallModel, TrainStaticModel and DeleteStaticModel; off, they
    /// refuse UNIMPLEMENTED.
    pub async fn start(
        listen: SocketAddr,
        models: Vec<ModelConfig>,
        max_message_bytes: usize,
        serve: Serve,
    ) -> std::result::Result<Server, String> {
        let names = config::names_with(&models, serve.admin)?;
        let builder = tls::configure_server(tonic::transport::Server::builder(), serve.tls.as_ref())?;
        let listener = TcpListener::bind(listen).await.map_err(|e| format!("listen on {listen}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| format!("listen on {listen}: {e}"))?;
        let state = Arc::new(State::new(names, models));
        let kserve = GrpcInferenceServiceServer::new(kserve::Inference { state: state.clone() })
            .max_decoding_message_size(max_message_bytes);
        let search_registry = Arc::new(search::SearchRegistry::new());
        let search = TurboSearchServiceServer::new(search::SearchService {
            state: state.clone(),
            registry: search_registry.clone(),
            admin: serve.admin,
        })
        .max_decoding_message_size(max_message_bytes);
        let health = HealthServer::new(health::HealthService { state: state.clone() })
            .max_decoding_message_size(max_message_bytes);
        let store = match &serve.vocabulary {
            Some(dir) => Some(Arc::new(vocabulary::Store::open(dir).map_err(|e| format!("vocabulary store: {e}"))?)),
            None => None,
        };
        let training = match &serve.training {
            Some(dir) => Some(training::open(dir).map_err(|e| format!("training directory: {e}"))?),
            None => None,
        };
        let (metrics_addr, metrics_stop, metrics_task) = match serve.metrics_listen {
            Some(m_listen) => {
                let (m_addr, m_stop, m_task) = metrics::start_http_listener(m_listen, state.clone()).await?;
                (Some(m_addr), Some(m_stop), Some(m_task))
            }
            None => (None, None, None),
        };
        // The forwarder spawns its drain, so it starts after every step that
        // can refuse the start: a refused start leaves no task behind.
        let sink = serve.sink.as_ref().map(SinkForwarder::start).transpose()?;
        let analysis = AnalysisServiceServer::new(analysis::Analysis {
            state: state.clone(),
            admin: serve.admin,
            sink: sink.clone(),
        })
        .max_decoding_message_size(max_message_bytes);
        let receiver =
            serve.sink_receiver.map(|r| TurboSinkServiceServer::new(r).max_decoding_message_size(max_message_bytes));
        let training = TurboTrainingServiceServer::new(training::TrainingService::new(
            state.clone(),
            store.clone(),
            training,
            serve.admin,
        ))
        .max_decoding_message_size(max_message_bytes);
        let vocabulary = TurboVocabularyServiceServer::new(vocabulary::Vocabulary { store, admin: serve.admin })
            .max_decoding_message_size(max_message_bytes);
        let catalog = TurboCatalogServiceServer::new(catalog::CatalogService::new(serve.catalog, serve.admin));
        let (stop, stopped) = oneshot::channel::<()>();
        // TCP_NODELAY on every accepted socket. The builder's tcp_nodelay
        // does not reach a listener served with serve_with_incoming, and
        // without it a response written in two segments waits for the
        // client's delayed ACK (about 40 ms a ModelInfer).
        let incoming = TcpIncoming::from(listener).with_nodelay(Some(true));
        let task = tokio::spawn(
            builder
                .layer(metrics::MetricsLayer::new(state.metrics.clone()))
                .add_service(kserve)
                .add_service(analysis)
                .add_service(search)
                .add_service(health)
                .add_service(vocabulary)
                .add_service(catalog)
                .add_service(training)
                .add_optional_service(receiver)
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = stopped.await;
                }),
        );
        Ok(Server {
            state,
            search_registry,
            addr,
            metrics_addr,
            stop: Some(stop),
            task,
            metrics_stop,
            metrics_task,
            sink,
        })
    }

    /// Configures the directory within which `PersistIndex` and
    /// `CreateIndex.load_from_path` read and write plain snapshot file names.
    pub fn set_search_snapshot_dir(&self, dir: impl Into<std::path::PathBuf>) {
        self.search_registry.set_snapshot_dir(dir.into());
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Returns the bound Prometheus `/metrics` HTTP address when `--metrics-listen`
    /// is configured.
    pub fn metrics_addr(&self) -> Option<SocketAddr> {
        self.metrics_addr
    }

    /// Renders the server's Prometheus metrics in text exposition format (`0.0.4`).
    pub fn metrics(&self) -> String {
        self.state.metrics.render(&self.state)
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
    /// since the server started, and the documents and bytes admitted
    /// and not yet answered across every stream right now, never over
    /// the advertised caps.
    pub fn stream_stats(&self) -> stream::Stats {
        self.state.stream.stats()
    }

    /// Marks the server as draining: `ServerReady` and `ModelReady` turn
    /// false and `grpc.health.v1.Health` reports `NOT_SERVING` on `Check`
    /// and `Watch` while requests in flight finish before `stop`.
    pub fn drain(&self) {
        self.state.drain();
    }

    /// What the forwarder to the downstream sink did so far; `None`
    /// without a sink.
    pub fn sink_stats(&self) -> Option<SinkStats> {
        self.sink.as_ref().map(|s| s.stats())
    }

    /// Stops forwarding to the sink: no answer is offered after this, and
    /// what the queue holds is sent, waiting up to `within` for it and the
    /// sink's summary. Answers with the forwarder's final counts.
    pub async fn close_sink(&self, within: std::time::Duration) -> Option<SinkStats> {
        let sink = self.sink.as_ref()?;
        sink.close(within).await;
        Some(sink.stats())
    }

    /// Stops answering and waits for the server to finish, then for the
    /// documents queued for the sink, up to 10 seconds.
    pub async fn stop(mut self) {
        self.state.shutdown();
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
        if let Some(s) = self.metrics_stop.take() {
            let _ = s.send(());
        }
        let _ = (&mut self.task).await;
        if let Some(m) = self.metrics_task.take() {
            let _ = m.await;
        }
        self.close_sink(std::time::Duration::from_secs(10)).await;
    }
}
