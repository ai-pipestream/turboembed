//! Tokio runtime-scoped pooled gRPC client for `turbo-kserve` (`#2338`, `#2346`).
//!
//! Binds channels to `(tokio::runtime::Id, EndpointKey)` with automatic
//! eviction when a Tokio runtime shuts down, propagates constant-time bearer
//! credentials, `x-turbo-priority` (`interactive` vs `bulk`), and `grpc-timeout`
//! deadlines, and verifies remote [`BundleScoringFingerprint`] contracts via
//! `ModelMetadata`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tonic::Request;
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, Endpoint};

use crate::error::{ClientError, Result};
use crate::fingerprint::{BundleScoringFingerprint, BundleScoringSpec, IndexGenerationHandshake};

/// Generated protobuf client messages and service stubs matching `turbo-kserve`.
pub mod proto {
    #![allow(clippy::all)]
    tonic::include_proto!("inference");
    /// Analysis result and search projection/query/rerank RPCs.
    pub mod result {
        pub mod v1 {
            tonic::include_proto!("turbo.result.v1");
        }
        pub use v1::*;
    }
    /// Hybrid search and collection management messages.
    pub mod search {
        pub mod v1 {
            tonic::include_proto!("turbo.search.v1");
        }
        pub use v1::*;
    }
    /// Standard gRPC health check protocol.
    pub mod health {
        tonic::include_proto!("grpc.health.v1");
    }
    /// Downstream document sink protocol.
    pub mod sink {
        pub mod v1 {
            tonic::include_proto!("turbo.sink.v1");
        }
        pub use v1::*;
    }
    /// Acknowledged PII release audit protocol.
    pub mod audit {
        pub mod v1 {
            tonic::include_proto!("turbo.audit.v1");
        }
        pub use v1::*;
    }
    /// Corpus DF and Voronoi centroid vocabulary store service.
    pub mod vocabulary {
        pub mod v1 {
            tonic::include_proto!("turbo.vocabulary.v1");
        }
        pub use v1::*;
    }
    /// Model catalog and static training services.
    pub mod training {
        pub mod v1 {
            tonic::include_proto!("turbo.training.v1");
        }
        pub use v1::*;
    }
}

use proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use proto::infer_parameter::ParameterChoice;
use proto::model_infer_request::InferInputTensor;
use proto::result::analysis_service_client::AnalysisServiceClient;
use proto::result::{
    AnalyzeQueryRequest, AnalyzeQueryResponse, ProjectSearchDocumentRequest, RerankAndHighlightRequest,
    RerankAndHighlightResponse, SearchDocumentProjection,
};
use proto::vocabulary::turbo_vocabulary_service_client::TurboVocabularyServiceClient;
use proto::{InferParameter, InferTensorContents, ModelInferRequest, ModelMetadataRequest};

const PRIORITY_METADATA_KEY: &str = "x-turbo-priority";
const AUTH_METADATA_KEY: &str = "authorization";
const MAX_DECODE_BYTES: usize = 64 * 1024 * 1024;

/// Device scheduling lane requested on `turbo-kserve` (`#2346`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum RemotePriority {
    /// Foreground query / rerank lane (`x-turbo-priority: interactive`).
    #[default]
    Interactive,
    /// Background bulk indexing lane (`x-turbo-priority: bulk`).
    Bulk,
}

impl RemotePriority {
    const fn header_value(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Bulk => "bulk",
        }
    }
}

/// Optional TLS / mutual-TLS material for `https://` `turbo-kserve` endpoints.
#[cfg(feature = "tls")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RemoteTlsConfig {
    /// PEM-encoded CA certificate bundle verifying the server certificate.
    pub server_ca_pem: Vec<u8>,
    /// Optional `(client_cert_pem, client_key_pem)` pair for mutual TLS.
    pub client_identity_pem: Option<(Vec<u8>, Vec<u8>)>,
    /// Optional SNI / certificate domain name override.
    pub domain_name: Option<String>,
}

/// Connection, authentication, priority, and deadline policy for a remote `turbo-kserve` endpoint.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RemoteEndpointConfig {
    /// HTTP/2 gRPC URI (`http://host:port` or `https://host:port`).
    pub endpoint: String,
    /// Optional bearer token sent as `authorization: Bearer <token>`.
    pub bearer_token: Option<String>,
    /// Default priority lane when a call does not override it.
    pub default_priority: RemotePriority,
    /// Optional default per-RPC timeout translated into `grpc-timeout`.
    pub default_timeout: Option<Duration>,
    /// Optional TLS / mTLS configuration (`feature = "tls"`).
    #[cfg(feature = "tls")]
    pub tls: Option<RemoteTlsConfig>,
}

impl RemoteEndpointConfig {
    /// Creates an endpoint configuration with interactive priority and no auth/TLS.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            bearer_token: None,
            default_priority: RemotePriority::Interactive,
            default_timeout: None,
            #[cfg(feature = "tls")]
            tls: None,
        }
    }

    /// Attaches a bearer token (`authorization: Bearer <token>`).
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Sets the default [`RemotePriority`] lane.
    pub fn with_priority(mut self, priority: RemotePriority) -> Self {
        self.default_priority = priority;
        self
    }

    /// Sets the default per-RPC deadline (`grpc-timeout`).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = Some(timeout);
        self
    }

    /// Configures TLS or mutual TLS (`feature = "tls"`).
    #[cfg(feature = "tls")]
    pub fn with_tls(mut self, tls: RemoteTlsConfig) -> Self {
        self.tls = Some(tls);
        self
    }
}

#[derive(Default)]
struct PoolInner {
    channels: HashMap<(tokio::runtime::Id, RemoteEndpointConfig), Channel>,
    watched_runtimes: HashSet<tokio::runtime::Id>,
}

/// Process-safe gRPC channel pool keyed by `(tokio::runtime::Id, RemoteEndpointConfig)`.
///
/// Channels created under one Tokio runtime are never reused on another runtime
/// and are automatically evicted as soon as their owning runtime drops.
#[derive(Clone, Default)]
pub struct RemoteChannelPool {
    inner: Arc<Mutex<PoolInner>>,
}

struct RuntimeEvictGuard {
    rt_id: tokio::runtime::Id,
    pool: Arc<Mutex<PoolInner>>,
}

impl Drop for RuntimeEvictGuard {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.pool.lock() {
            guard.channels.retain(|(id, _), _| *id != self.rt_id);
            guard.watched_runtimes.remove(&self.rt_id);
        }
    }
}

impl RemoteChannelPool {
    /// Creates an empty runtime-scoped channel pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of currently cached runtime-bound channels (for diagnostics/tests).
    pub fn active_channel_count(&self) -> usize {
        self.inner.lock().map_or(0, |g| g.channels.len())
    }

    /// Returns a lazy HTTP/2 [`Channel`] bound to `tokio::runtime::Handle::current()`.
    pub fn channel(&self, cfg: &RemoteEndpointConfig) -> Result<Channel> {
        let handle = tokio::runtime::Handle::try_current()
            .map_err(|_| ClientError::FailedPrecondition("remote gRPC call requires an active Tokio runtime".into()))?;
        let rt_id = handle.id();
        {
            let mut guard = self.inner.lock().map_err(|_| ClientError::Internal("channel pool poisoned".into()))?;
            if let Some(ch) = guard.channels.get(&(rt_id, cfg.clone())) {
                return Ok(ch.clone());
            }
            if guard.watched_runtimes.insert(rt_id) {
                let evict = RuntimeEvictGuard { rt_id, pool: Arc::clone(&self.inner) };
                handle.spawn(async move {
                    let _keep = evict;
                    std::future::pending::<()>().await;
                });
            }
        }
        let endpoint = build_endpoint(cfg)?;
        let ch = endpoint.connect_lazy();
        let mut guard = self.inner.lock().map_err(|_| ClientError::Internal("channel pool poisoned".into()))?;
        Ok(guard.channels.entry((rt_id, cfg.clone())).or_insert(ch).clone())
    }
}

fn build_endpoint(cfg: &RemoteEndpointConfig) -> Result<Endpoint> {
    #[allow(unused_mut)]
    let mut ep = Endpoint::from_shared(cfg.endpoint.clone())
        .map_err(|e| ClientError::InvalidArgument(format!("invalid gRPC endpoint `{}`: {e}", cfg.endpoint)))?
        .tcp_nodelay(true);
    #[cfg(feature = "tls")]
    if let Some(tls) = &cfg.tls {
        use tonic::transport::{Certificate, ClientTlsConfig, Identity};
        let mut tls_cfg = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(tls.server_ca_pem.clone()));
        if let Some(domain) = &tls.domain_name {
            tls_cfg = tls_cfg.domain_name(domain.clone());
        }
        if let Some((cert, key)) = &tls.client_identity_pem {
            tls_cfg = tls_cfg.identity(Identity::from_pem(cert.clone(), key.clone()));
        }
        ep = ep
            .tls_config(tls_cfg)
            .map_err(|e| ClientError::InvalidArgument(format!("invalid TLS configuration: {e}")))?;
    }
    Ok(ep)
}

/// Pooled remote `turbo-kserve` client safe to share (`Clone + Send + Sync`) across runtimes.
#[derive(Clone)]
pub struct RemoteClient {
    config: RemoteEndpointConfig,
    pool: RemoteChannelPool,
}

impl RemoteClient {
    /// Creates a pooled [`RemoteClient`] with a fresh [`RemoteChannelPool`].
    pub fn new(config: RemoteEndpointConfig) -> Self {
        Self { config, pool: RemoteChannelPool::new() }
    }

    /// Creates a pooled [`RemoteClient`] sharing an existing [`RemoteChannelPool`].
    pub fn with_pool(config: RemoteEndpointConfig, pool: RemoteChannelPool) -> Self {
        Self { config, pool }
    }

    /// Active endpoint configuration.
    pub fn config(&self) -> &RemoteEndpointConfig {
        &self.config
    }

    /// Underlying runtime-scoped channel pool.
    pub fn pool(&self) -> &RemoteChannelPool {
        &self.pool
    }

    /// Decorates `payload` with bearer auth, priority metadata, and timeout.
    pub fn prepare_request<T>(
        &self,
        payload: T,
        priority: Option<RemotePriority>,
        timeout: Option<Duration>,
    ) -> Result<Request<T>> {
        let mut req = Request::new(payload);
        let prio = priority.unwrap_or(self.config.default_priority);
        req.metadata_mut().insert(PRIORITY_METADATA_KEY, MetadataValue::from_static(prio.header_value()));
        if let Some(token) = &self.config.bearer_token {
            let header = format!("Bearer {}", token.trim());
            let val = MetadataValue::try_from(header)
                .map_err(|e| ClientError::InvalidArgument(format!("invalid bearer token header: {e}")))?;
            req.metadata_mut().insert(AUTH_METADATA_KEY, val);
        }
        if let Some(dur) = timeout.or(self.config.default_timeout) {
            req.set_timeout(dur);
        }
        Ok(req)
    }

    /// Queries `ModelMetadata` on `turbo-kserve`, computes the remote model's
    /// [`BundleScoringFingerprint`], and verifies it against `handshake`.
    pub async fn verify_model_handshake(
        &self,
        model_name: &str,
        model_version: &str,
        handshake: &IndexGenerationHandshake,
    ) -> Result<(BundleScoringSpec, BundleScoringFingerprint)> {
        let ch = self.pool.channel(&self.config)?;
        let mut client = GrpcInferenceServiceClient::new(ch).max_decoding_message_size(MAX_DECODE_BYTES);
        let req = self.prepare_request(
            ModelMetadataRequest { name: model_name.to_owned(), version: model_version.to_owned() },
            Some(RemotePriority::Interactive),
            None,
        )?;
        let meta = client.model_metadata(req).await?.into_inner();
        let props = &meta.properties;
        let get_str = |k: &str| props.get(k).cloned().unwrap_or_default();
        let dim = get_str("model_info.dim").parse::<u32>().unwrap_or(0);
        let spec = BundleScoringSpec {
            model_id: get_str("model_info.model_id"),
            revision: get_str("model_info.revision"),
            manifest_sha256: get_str("model_info.manifest_sha256"),
            task: get_str("model_info.task"),
            dim,
            pooling: get_str("model_info.pooling"),
            normalize: get_str("model_info.normalize"),
        };
        let fp = spec.fingerprint();
        handshake.verify_bundle(&fp)?;
        Ok((spec, fp))
    }

    /// Runs batched `ModelInfer` over UTF-8 `texts`, returning `(flat_f32_rows, dim)`.
    pub async fn embed_texts(
        &self,
        model_name: &str,
        texts: &[&str],
        role: Option<&str>,
        priority: Option<RemotePriority>,
        timeout: Option<Duration>,
    ) -> Result<(Vec<f32>, usize)> {
        if texts.is_empty() {
            return Ok((Vec::new(), 0));
        }
        let mut parameters = HashMap::new();
        if let Some(r) = role {
            let wire_role = match r.trim() {
                "query" | "QUERY" => "PROMPT_QUERY",
                "document" | "DOCUMENT" => "PROMPT_DOCUMENT",
                "none" | "NONE" => "PROMPT_NONE",
                other => other,
            };
            parameters.insert(
                "prompt_role".to_owned(),
                InferParameter { parameter_choice: Some(ParameterChoice::StringParam(wire_role.to_owned())) },
            );
        }
        let payload = ModelInferRequest {
            model_name: model_name.to_owned(),
            model_version: String::new(),
            id: String::new(),
            parameters,
            inputs: vec![InferInputTensor {
                name: "texts".into(),
                datatype: "BYTES".into(),
                shape: vec![texts.len() as i64],
                parameters: HashMap::new(),
                contents: Some(InferTensorContents {
                    bytes_contents: texts.iter().map(|s| s.as_bytes().to_vec()).collect(),
                    ..Default::default()
                }),
            }],
            outputs: Vec::new(),
            raw_input_contents: Vec::new(),
        };
        let ch = self.pool.channel(&self.config)?;
        let mut client = GrpcInferenceServiceClient::new(ch).max_decoding_message_size(MAX_DECODE_BYTES);
        let resp = client.model_infer(self.prepare_request(payload, priority, timeout)?).await?.into_inner();
        let out = resp
            .outputs
            .first()
            .ok_or_else(|| ClientError::Internal("ModelInferResponse contained no output tensor".into()))?;
        let dim = out.shape.get(1).copied().unwrap_or(0).max(0) as usize;
        let data = if let Some(raw) = resp.raw_output_contents.first()
            && !raw.is_empty()
        {
            if !raw.len().is_multiple_of(4) {
                return Err(ClientError::Internal(format!(
                    "FP32 raw_output_contents byte length {} not /4",
                    raw.len()
                )));
            }
            raw.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect()
        } else if let Some(c) = &out.contents {
            c.fp32_contents.clone()
        } else {
            Vec::new()
        };
        Ok((data, dim))
    }

    /// Invokes one-pass `AnalysisService::AnalyzeQuery` (`#2339`).
    pub async fn analyze_query(
        &self,
        request: AnalyzeQueryRequest,
        timeout: Option<Duration>,
    ) -> Result<AnalyzeQueryResponse> {
        let ch = self.pool.channel(&self.config)?;
        let mut client = AnalysisServiceClient::new(ch).max_decoding_message_size(MAX_DECODE_BYTES);
        Ok(client
            .analyze_query(self.prepare_request(request, Some(RemotePriority::Interactive), timeout)?)
            .await?
            .into_inner())
    }

    /// Invokes Stage-2 `AnalysisService::RerankAndHighlight` (`#2342`).
    pub async fn rerank_and_highlight(
        &self,
        request: RerankAndHighlightRequest,
        timeout: Option<Duration>,
    ) -> Result<RerankAndHighlightResponse> {
        let ch = self.pool.channel(&self.config)?;
        let mut client = AnalysisServiceClient::new(ch).max_decoding_message_size(MAX_DECODE_BYTES);
        Ok(client
            .rerank_and_highlight(self.prepare_request(request, Some(RemotePriority::Interactive), timeout)?)
            .await?
            .into_inner())
    }

    /// Invokes `AnalysisService::ProjectSearchDocument` (`#2345`).
    pub async fn project_search_document(
        &self,
        request: ProjectSearchDocumentRequest,
        priority: Option<RemotePriority>,
        timeout: Option<Duration>,
    ) -> Result<SearchDocumentProjection> {
        let ch = self.pool.channel(&self.config)?;
        let mut client = AnalysisServiceClient::new(ch).max_decoding_message_size(MAX_DECODE_BYTES);
        Ok(client.project_search_document(self.prepare_request(request, priority, timeout)?).await?.into_inner())
    }

    /// Returns a runtime-bound `TurboVocabularyServiceClient` (`#2341`, `#2346`).
    pub fn vocabulary_client(&self) -> Result<TurboVocabularyServiceClient<Channel>> {
        let ch = self.pool.channel(&self.config)?;
        Ok(TurboVocabularyServiceClient::new(ch).max_decoding_message_size(MAX_DECODE_BYTES))
    }
}
