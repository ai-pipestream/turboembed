//! OVMS/KServe gRPC client backend for inferstream.
//!
//! Forwards OIP V2 requests to a running [OpenVINO Model Server] (or any
//! KServe V2 gRPC server) instead of executing graphs in-process. This is the
//! primary Intel path on hosts that already run OVMS in Docker: inferstream
//! keeps the façade role (auth, routing, streaming extension) while OVMS owns
//! model execution on CPU/GPU/NPU — no host OpenVINO runtime install needed.
//!
//! OVMS speaks the same `inference.GRPCInferenceService` this façade serves
//! (both vendor the KServe OIP V2 proto), so requests are forwarded as typed
//! protobuf messages on an existing tonic channel — never re-encoded through
//! JSON or an intermediate representation. `ModelInfer`, `ModelReady`, and
//! `ModelMetadata` map 1:1. OVMS does not implement the Triton-shaped
//! `ModelStreamInfer` extension, so streaming requests are served through the
//! default unary adaptation in [`Backend::infer_stream`] (one chunk per
//! request).
//!
//! **Model naming:** by default the config `name` is forwarded as-is, so it
//! must match a model or DAG pipeline name the upstream server actually
//! serves (check `GET <rest_port>/v1/config` on the OVMS host). With
//! [`OvmsBackend::with_upstream_model`] (config `upstream_model`), requests
//! are forwarded under the upstream's real name while clients keep using
//! the logical name — how catalog aliases like `minilm` front pipelines
//! like `minilm_pipeline`.
//!
//! **Endpoint discovery:** OVMS publishes gRPC on `--port` (REST is the
//! separate `--rest_port`). In Docker, the gRPC port may only be reachable on
//! the container's bridge IP — `docker inspect -f
//! '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container>`.
//!
//! **Embed-convention adaptation:** the façade's `inferstream.v1.Embed` RPC
//! wraps texts as a single BYTES tensor named `"text"` and unwraps an FP32
//! output named `"embedding"`. OVMS embedding DAG pipelines instead declare
//! `"strings"` in and `"sentence_embedding"` out. When a forwarded request
//! matches the façade convention exactly (one BYTES input named `"text"`),
//! this backend renames the input tensor to the pipeline's name and renames
//! the matching FP32 output back to `"embedding"` — a pure field rename on
//! the typed protobuf messages, never a re-encode. Raw OIP proxy calls that
//! already use the upstream tensor names pass through untouched.

use async_trait::async_trait;
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Request, Status};

use inferstream_backend::{Backend, BackendError, ModelMetadata};
use inferstream_protocol::inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inferstream_protocol::inference::{
    ModelInferRequest, ModelInferResponse, ModelMetadataRequest, ModelReadyRequest,
};

/// Backend that proxies OIP V2 RPCs to an upstream KServe V2 gRPC server
/// (OpenVINO Model Server, Triton, another inferstream, ...).
///
/// The tonic channel is created lazily and multiplexes all requests, so the
/// backend is cheap to clone and safe to share; connection failures surface
/// per-request as [`BackendError::Unavailable`], never at startup.
#[derive(Debug, Clone)]
pub struct OvmsBackend {
    endpoint: String,
    client: GrpcInferenceServiceClient<Channel>,
    /// Upstream input tensor name substituted for the façade's `"text"`.
    embed_input_name: String,
    /// Upstream output tensor name renamed back to `"embedding"`.
    embed_output_name: String,
    /// Upstream model/pipeline name forwarded in place of the logical name
    /// clients use (alias support: `minilm` fronting `minilm_pipeline`).
    /// Responses report the name the client asked for.
    upstream_model: Option<String>,
}

/// Input tensor name the façade `Embed` RPC produces.
const FACADE_EMBED_INPUT: &str = "text";
/// Output tensor name the façade `Embed` RPC consumes.
const FACADE_EMBED_OUTPUT: &str = "embedding";
/// Default input name of OVMS embedding DAG pipelines (`config-gpu.json`).
const OVMS_EMBED_INPUT: &str = "strings";
/// Default output name of OVMS embedding DAG pipelines.
const OVMS_EMBED_OUTPUT: &str = "sentence_embedding";

impl OvmsBackend {
    /// Create a backend that forwards to `endpoint`
    /// (e.g. `"http://172.22.0.2:8000"`), adapting façade `Embed` requests to
    /// the standard OVMS pipeline tensor names
    /// (`"strings"` / `"sentence_embedding"`).
    ///
    /// The connection is established lazily on first use, so this succeeds
    /// even while the upstream server is down.
    pub fn new(endpoint: impl Into<String>) -> Result<Self, BackendError> {
        Self::with_embed_names(endpoint, OVMS_EMBED_INPUT, OVMS_EMBED_OUTPUT)
    }

    /// Like [`OvmsBackend::new`] but with explicit upstream tensor names for
    /// the embed-convention adaptation, for KServe upstreams whose embedding
    /// graphs use different input/output names.
    pub fn with_embed_names(
        endpoint: impl Into<String>,
        embed_input_name: impl Into<String>,
        embed_output_name: impl Into<String>,
    ) -> Result<Self, BackendError> {
        let endpoint = endpoint.into();
        let channel = Endpoint::from_shared(endpoint.clone())
            .map_err(|e| {
                BackendError::InvalidRequest(format!("invalid OVMS endpoint {endpoint:?}: {e}"))
            })?
            .connect_lazy();
        Ok(Self {
            endpoint,
            client: GrpcInferenceServiceClient::new(channel),
            embed_input_name: embed_input_name.into(),
            embed_output_name: embed_output_name.into(),
            upstream_model: None,
        })
    }

    /// Forward requests under `upstream_model` instead of the name clients
    /// use, so a logical alias (`"minilm"`) can front an upstream pipeline
    /// with a different name (`"minilm_pipeline"`). Responses keep reporting
    /// the client-facing name.
    pub fn with_upstream_model(mut self, upstream_model: impl Into<String>) -> Self {
        self.upstream_model = Some(upstream_model.into());
        self
    }

    /// The upstream endpoint this backend forwards to.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The name forwarded upstream for the model clients call `model_name`.
    fn upstream_name<'a>(&'a self, model_name: &'a str) -> &'a str {
        self.upstream_model.as_deref().unwrap_or(model_name)
    }

    fn map_status(&self, status: Status) -> BackendError {
        match status.code() {
            Code::NotFound => BackendError::ModelNotFound(status.message().to_string()),
            Code::InvalidArgument => BackendError::InvalidRequest(status.message().to_string()),
            Code::Unavailable | Code::DeadlineExceeded => BackendError::Unavailable(format!(
                "upstream {} unreachable: {}",
                self.endpoint,
                status.message()
            )),
            _ => BackendError::Internal(format!(
                "upstream {} returned {}: {}",
                self.endpoint,
                status.code(),
                status.message()
            )),
        }
    }
}

#[async_trait]
impl Backend for OvmsBackend {
    fn id(&self) -> &str {
        "ovms"
    }

    async fn model_ready(&self, model_name: &str, model_version: &str) -> bool {
        let request = Request::new(ModelReadyRequest {
            name: self.upstream_name(model_name).to_string(),
            version: model_version.to_string(),
        });
        match self.client.clone().model_ready(request).await {
            Ok(response) => response.into_inner().ready,
            Err(status) => {
                tracing::debug!(
                    endpoint = %self.endpoint,
                    model = model_name,
                    error = %status,
                    "upstream ModelReady failed"
                );
                false
            }
        }
    }

    async fn model_metadata(
        &self,
        model_name: &str,
        model_version: &str,
    ) -> Result<ModelMetadata, BackendError> {
        let request = Request::new(ModelMetadataRequest {
            name: self.upstream_name(model_name).to_string(),
            version: model_version.to_string(),
        });
        let response = self
            .client
            .clone()
            .model_metadata(request)
            .await
            .map_err(|status| self.map_status(status))?
            .into_inner();
        // Clients addressed the logical name; report it back, and surface
        // the resolved upstream artifact in properties.
        let mut properties: std::collections::HashMap<String, String> =
            [("upstream_endpoint".to_string(), self.endpoint.clone())].into();
        if let Some(upstream) = &self.upstream_model {
            properties.insert("upstream_model".to_string(), upstream.clone());
        }
        Ok(ModelMetadata {
            name: model_name.to_string(),
            versions: response.versions,
            platform: response.platform,
            inputs: response.inputs,
            outputs: response.outputs,
            properties,
        })
    }

    async fn infer(
        &self,
        mut request: ModelInferRequest,
    ) -> Result<ModelInferResponse, BackendError> {
        // Alias support: forward under the upstream pipeline's real name,
        // then restore the name the client addressed in the response.
        let requested_name = request.model_name.clone();
        request.model_name = self.upstream_name(&requested_name).to_string();
        // Adapt the façade Embed convention (one BYTES tensor named "text")
        // to the upstream pipeline's declared tensor names. Anything else is
        // a raw OIP proxy call and passes through untouched.
        let adapted = matches!(
            request.inputs.as_slice(),
            [input] if input.name == FACADE_EMBED_INPUT && input.datatype == "BYTES"
        );
        if adapted {
            request.inputs[0].name = self.embed_input_name.clone();
        }
        let mut response = self
            .client
            .clone()
            .model_infer(Request::new(request))
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| self.map_status(status))?;
        response.model_name = requested_name;
        if adapted {
            for output in &mut response.outputs {
                if output.name == self.embed_output_name {
                    output.name = FACADE_EMBED_OUTPUT.to_string();
                }
            }
        }
        Ok(response)
    }

    // infer_stream: default unary adaptation. OVMS implements only the
    // upstream OIP surface (no ModelStreamInfer), so each streamed request is
    // forwarded as one unary ModelInfer producing one response chunk.
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::net::SocketAddr;

    use inferstream_protocol::inference::grpc_inference_service_server::{
        GrpcInferenceService, GrpcInferenceServiceServer,
    };
    use inferstream_protocol::inference::{
        ModelMetadataResponse, ModelReadyResponse, ModelStreamInferResponse, ServerLiveRequest,
        ServerLiveResponse, ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest,
        ServerReadyResponse,
    };
    use inferstream_protocol::tensor::{pack_bytes, pack_fp32};
    use tonic::{Response, Streaming};

    /// Minimal upstream standing in for OVMS: serves one model
    /// ("upstream-embed") and echoes a fixed FP32 embedding.
    struct FakeOvms;

    #[tonic::async_trait]
    impl GrpcInferenceService for FakeOvms {
        async fn server_live(
            &self,
            _request: Request<ServerLiveRequest>,
        ) -> Result<Response<ServerLiveResponse>, Status> {
            Ok(Response::new(ServerLiveResponse { live: true }))
        }

        async fn server_ready(
            &self,
            _request: Request<ServerReadyRequest>,
        ) -> Result<Response<ServerReadyResponse>, Status> {
            Ok(Response::new(ServerReadyResponse { ready: true }))
        }

        async fn model_ready(
            &self,
            request: Request<ModelReadyRequest>,
        ) -> Result<Response<ModelReadyResponse>, Status> {
            Ok(Response::new(ModelReadyResponse {
                ready: request.into_inner().name == "upstream-embed",
            }))
        }

        async fn server_metadata(
            &self,
            _request: Request<ServerMetadataRequest>,
        ) -> Result<Response<ServerMetadataResponse>, Status> {
            Ok(Response::new(ServerMetadataResponse {
                name: "fake-ovms".into(),
                version: "0".into(),
                extensions: vec![],
            }))
        }

        async fn model_metadata(
            &self,
            request: Request<ModelMetadataRequest>,
        ) -> Result<Response<ModelMetadataResponse>, Status> {
            let name = request.into_inner().name;
            if name != "upstream-embed" {
                return Err(Status::not_found(format!("model {name:?} not served")));
            }
            Ok(Response::new(ModelMetadataResponse {
                name,
                versions: vec!["1".into()],
                platform: "OpenVINO".into(),
                ..Default::default()
            }))
        }

        async fn model_infer(
            &self,
            request: Request<ModelInferRequest>,
        ) -> Result<Response<ModelInferResponse>, Status> {
            use inferstream_protocol::inference::model_infer_response::InferOutputTensor;
            let request = request.into_inner();
            if request.model_name != "upstream-embed" {
                return Err(Status::not_found(format!(
                    "model {:?} not served",
                    request.model_name
                )));
            }
            // Like a real OVMS DAG pipeline: the input tensor name must
            // match the pipeline's declared input exactly.
            match request.inputs.as_slice() {
                [input] if input.name == "strings" => {}
                _ => {
                    return Err(Status::invalid_argument(
                        "Missing input with specific name - Required input: strings",
                    ));
                }
            }
            Ok(Response::new(ModelInferResponse {
                model_name: request.model_name,
                id: request.id,
                outputs: vec![InferOutputTensor {
                    name: "sentence_embedding".into(),
                    datatype: "FP32".into(),
                    shape: vec![1, 3],
                    ..Default::default()
                }],
                raw_output_contents: vec![pack_fp32(&[0.25, -0.5, 1.0])],
                ..Default::default()
            }))
        }

        type ModelStreamInferStream = std::pin::Pin<
            Box<
                dyn futures::Stream<Item = Result<ModelStreamInferResponse, Status>>
                    + Send
                    + 'static,
            >,
        >;

        async fn model_stream_infer(
            &self,
            _request: Request<Streaming<ModelInferRequest>>,
        ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
            Err(Status::unimplemented(
                "upstream OIP has no ModelStreamInfer",
            ))
        }
    }

    async fn spawn_fake_ovms() -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(GrpcInferenceServiceServer::new(FakeOvms))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        addr
    }

    fn embed_request(model: &str, id: &str) -> ModelInferRequest {
        use inferstream_protocol::inference::model_infer_request::InferInputTensor;
        ModelInferRequest {
            model_name: model.to_string(),
            id: id.to_string(),
            inputs: vec![InferInputTensor {
                name: "strings".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                ..Default::default()
            }],
            raw_input_contents: vec![pack_bytes(&[b"hello world".as_slice()])],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn forwards_unary_infer_and_readiness() {
        let addr = spawn_fake_ovms().await;
        let backend = OvmsBackend::new(format!("http://{addr}")).unwrap();

        assert!(backend.model_ready("upstream-embed", "").await);
        assert!(!backend.model_ready("missing", "").await);

        let response = backend
            .infer(embed_request("upstream-embed", "req-1"))
            .await
            .unwrap();
        assert_eq!(response.id, "req-1");
        assert_eq!(response.model_name, "upstream-embed");
        assert_eq!(response.raw_output_contents.len(), 1);

        let metadata = backend.model_metadata("upstream-embed", "").await.unwrap();
        assert_eq!(metadata.platform, "OpenVINO");
        assert_eq!(
            metadata.properties.get("upstream_endpoint").unwrap(),
            &format!("http://{addr}")
        );
    }

    /// Requests using the façade Embed convention (one BYTES tensor named
    /// "text") are renamed to the pipeline's input, and the pipeline's
    /// output is renamed back to "embedding".
    #[tokio::test]
    async fn adapts_facade_embed_convention_to_pipeline_names() {
        use inferstream_protocol::inference::model_infer_request::InferInputTensor;
        let addr = spawn_fake_ovms().await;
        let backend = OvmsBackend::new(format!("http://{addr}")).unwrap();

        let request = ModelInferRequest {
            model_name: "upstream-embed".to_string(),
            id: "embed-1".to_string(),
            inputs: vec![InferInputTensor {
                name: "text".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                ..Default::default()
            }],
            raw_input_contents: vec![pack_bytes(&[b"hello world".as_slice()])],
            ..Default::default()
        };
        let response = backend.infer(request).await.unwrap();
        assert_eq!(response.outputs.len(), 1);
        assert_eq!(response.outputs[0].name, "embedding");
        assert_eq!(response.outputs[0].datatype, "FP32");
        assert_eq!(response.raw_output_contents.len(), 1);
    }

    /// Raw OIP proxy calls that already use the upstream tensor names pass
    /// through with no renaming in either direction.
    #[tokio::test]
    async fn raw_proxy_calls_pass_through_unadapted() {
        let addr = spawn_fake_ovms().await;
        let backend = OvmsBackend::new(format!("http://{addr}")).unwrap();
        let response = backend
            .infer(embed_request("upstream-embed", "raw-1"))
            .await
            .unwrap();
        assert_eq!(response.outputs[0].name, "sentence_embedding");
    }

    /// With `upstream_model` set, clients address the logical alias while
    /// the wire carries the upstream pipeline name; responses and metadata
    /// report the alias back.
    #[tokio::test]
    async fn upstream_model_remaps_alias_to_pipeline_name() {
        let addr = spawn_fake_ovms().await;
        let backend = OvmsBackend::new(format!("http://{addr}"))
            .unwrap()
            .with_upstream_model("upstream-embed");

        // FakeOvms only serves "upstream-embed"; readiness under the alias
        // proves the remap happened.
        assert!(backend.model_ready("minilm", "").await);

        let response = backend
            .infer(embed_request("minilm", "alias-1"))
            .await
            .unwrap();
        assert_eq!(response.model_name, "minilm", "client-facing name restored");
        assert_eq!(response.raw_output_contents.len(), 1);

        let metadata = backend.model_metadata("minilm", "").await.unwrap();
        assert_eq!(metadata.name, "minilm");
        assert_eq!(
            metadata.properties.get("upstream_model").unwrap(),
            "upstream-embed"
        );
    }

    #[tokio::test]
    async fn maps_not_found_from_upstream() {
        let addr = spawn_fake_ovms().await;
        let backend = OvmsBackend::new(format!("http://{addr}")).unwrap();
        let error = backend
            .infer(embed_request("missing", "req-2"))
            .await
            .unwrap_err();
        assert!(matches!(error, BackendError::ModelNotFound(_)));
    }

    #[tokio::test]
    async fn unreachable_upstream_is_unavailable_not_panic() {
        // Reserved port with nothing listening; connect_lazy defers failure
        // to the RPC, which must surface as Unavailable.
        let backend = OvmsBackend::new("http://127.0.0.1:1").unwrap();
        let error = backend
            .infer(embed_request("upstream-embed", "req-3"))
            .await
            .unwrap_err();
        assert!(matches!(error, BackendError::Unavailable(_)), "{error:?}");
        assert!(!backend.model_ready("upstream-embed", "").await);
    }

    #[tokio::test]
    async fn default_infer_stream_forwards_as_unary() {
        use futures::StreamExt;
        let addr = spawn_fake_ovms().await;
        let backend = OvmsBackend::new(format!("http://{addr}")).unwrap();
        let mut stream = backend
            .infer_stream(embed_request("upstream-embed", "req-4"))
            .await
            .unwrap();
        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk.id, "req-4");
        assert!(stream.next().await.is_none());
    }

    #[test]
    fn rejects_malformed_endpoint() {
        assert!(matches!(
            OvmsBackend::new("not a uri"),
            Err(BackendError::InvalidRequest(_))
        ));
    }
}
