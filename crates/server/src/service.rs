//! The OIP V2 `GRPCInferenceService` implementation: routes every RPC to the
//! backend registered for the requested model.

use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, warn};

use inferstream_backend::{BackendError, Registry};
use inferstream_protocol::inference::grpc_inference_service_server::GrpcInferenceService;
use inferstream_protocol::inference::{
    ModelInferRequest, ModelInferResponse, ModelMetadataRequest, ModelMetadataResponse,
    ModelReadyRequest, ModelReadyResponse, ModelStreamInferResponse, ServerLiveRequest,
    ServerLiveResponse, ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest,
    ServerReadyResponse,
};

/// Server name reported by `ServerMetadata`.
pub const SERVER_NAME: &str = "inferstream";
/// Server version reported by `ServerMetadata`.
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Buffered response chunks per open bidi stream before backpressure applies.
const STREAM_CHANNEL_CAPACITY: usize = 64;

/// The façade service. Holds the immutable model → backend routing table.
pub struct InferenceService {
    registry: Arc<Registry>,
}

impl InferenceService {
    pub fn new(registry: Arc<Registry>) -> Self {
        Self { registry }
    }

    fn backend_for(
        &self,
        model_name: &str,
    ) -> Result<Arc<dyn inferstream_backend::Backend>, Status> {
        self.registry
            .lookup(model_name)
            .ok_or_else(|| Status::not_found(format!("model {model_name:?} is not configured")))
    }
}

fn status_from(error: BackendError) -> Status {
    match error {
        BackendError::ModelNotFound(_) => Status::not_found(error.to_string()),
        BackendError::InvalidRequest(_) => Status::invalid_argument(error.to_string()),
        BackendError::Unavailable(_) => Status::unavailable(error.to_string()),
        BackendError::Internal(_) => Status::internal(error.to_string()),
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for InferenceService {
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
        // Ready once the routing table is loaded; per-model readiness is
        // reported by ModelReady.
        Ok(Response::new(ServerReadyResponse {
            ready: !self.registry.is_empty(),
        }))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let req = request.into_inner();
        let ready = match self.registry.lookup(&req.name) {
            Some(backend) => backend.model_ready(&req.name, &req.version).await,
            None => false,
        };
        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn server_metadata(
        &self,
        _request: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        Ok(Response::new(ServerMetadataResponse {
            name: SERVER_NAME.to_string(),
            version: SERVER_VERSION.to_string(),
            extensions: vec![
                "model_stream_infer".to_string(),
                // Second service on this endpoint: inferstream.v1.InferstreamService
                // (Tokenize / Detokenize / Embed / ListModels / Rerank).
                "inferstream.v1".to_string(),
            ],
        }))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let req = request.into_inner();
        let backend = self.backend_for(&req.name)?;
        let metadata = backend
            .model_metadata(&req.name, &req.version)
            .await
            .map_err(status_from)?;
        Ok(Response::new(ModelMetadataResponse {
            name: metadata.name,
            versions: metadata.versions,
            platform: metadata.platform,
            inputs: metadata.inputs,
            outputs: metadata.outputs,
            properties: metadata.properties.into_iter().collect(),
        }))
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let req = request.into_inner();
        let backend = self.backend_for(&req.model_name)?;
        debug!(model = %req.model_name, id = %req.id, backend = backend.id(), "unary infer");
        let response = backend.infer(req).await.map_err(status_from)?;
        Ok(Response::new(response))
    }

    type ModelStreamInferStream =
        Pin<Box<dyn Stream<Item = Result<ModelStreamInferResponse, Status>> + Send>>;

    async fn model_stream_infer(
        &self,
        request: Request<Streaming<ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let mut incoming = request.into_inner();
        let registry = Arc::clone(&self.registry);
        let (tx, rx) =
            mpsc::channel::<Result<ModelStreamInferResponse, Status>>(STREAM_CHANNEL_CAPACITY);

        // Driver task: pull requests off the stream and fan each one out to
        // its backend. Responses for concurrent requests are multiplexed onto
        // one channel; clients correlate chunks via the echoed request id.
        tokio::spawn(async move {
            while let Some(next) = incoming.next().await {
                let request = match next {
                    Ok(request) => request,
                    Err(status) => {
                        warn!(%status, "inbound stream error, closing");
                        let _ = tx.send(Err(status)).await;
                        return;
                    }
                };
                let request_id = request.id.clone();
                let model_name = request.model_name.clone();
                debug!(model = %model_name, id = %request_id, "stream infer request");

                let backend = match registry.lookup(&model_name) {
                    Some(backend) => backend,
                    None => {
                        let error = per_request_error(
                            &request_id,
                            &model_name,
                            format!("model {model_name:?} is not configured"),
                        );
                        if tx.send(Ok(error)).await.is_err() {
                            return;
                        }
                        continue;
                    }
                };

                let tx = tx.clone();
                tokio::spawn(async move {
                    match backend.infer_stream(request).await {
                        Ok(mut chunks) => {
                            while let Some(chunk) = chunks.next().await {
                                let message = match chunk {
                                    Ok(response) => ModelStreamInferResponse {
                                        error_message: String::new(),
                                        infer_response: Some(response),
                                    },
                                    Err(error) => per_request_error(
                                        &request_id,
                                        &model_name,
                                        error.to_string(),
                                    ),
                                };
                                if tx.send(Ok(message)).await.is_err() {
                                    return; // client went away
                                }
                            }
                        }
                        Err(error) => {
                            let _ = tx
                                .send(Ok(per_request_error(
                                    &request_id,
                                    &model_name,
                                    error.to_string(),
                                )))
                                .await;
                        }
                    }
                });
            }
            // All senders dropping closes the outbound stream once in-flight
            // per-request tasks finish.
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// Build a per-request error chunk that keeps the stream alive. The request
/// id is echoed inside a skeleton `infer_response` so clients can correlate
/// the failure.
fn per_request_error(
    request_id: &str,
    model_name: &str,
    message: String,
) -> ModelStreamInferResponse {
    ModelStreamInferResponse {
        error_message: message,
        infer_response: Some(ModelInferResponse {
            model_name: model_name.to_string(),
            id: request_id.to_string(),
            ..Default::default()
        }),
    }
}
