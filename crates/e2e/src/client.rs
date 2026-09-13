//! Thin tonic clients for both gRPC services, with optional bearer auth.

// tonic::Status is 176 bytes; the interceptor / RPC signatures need it unboxed.
#![allow(clippy::result_large_err)]

use std::collections::HashMap;
use std::time::Duration;

use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::Channel;
use tonic::{Request, Status};

use inferstream_protocol::extension::inferstream_service_client::InferstreamServiceClient;
use inferstream_protocol::extension::{
    DetokenizeRequest, EmbedRequest, ListModelsRequest, ListModelsResponse, ModelInfo,
    RerankRequest, RerankResponse, TokenIds, TokenizeRequest,
};
use inferstream_protocol::inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inferstream_protocol::inference::infer_parameter::ParameterChoice;
use inferstream_protocol::inference::model_infer_request::InferInputTensor;
use inferstream_protocol::inference::{
    InferParameter, ModelInferRequest, ServerLiveRequest, ServerLiveResponse,
};
use inferstream_protocol::tensor::{pack_bytes, unpack_bytes, DataType};

use crate::HarnessError;

const CONNECT_SECS: u64 = 15;
const RPC_SECS: u64 = 180;

#[derive(Clone)]
pub(crate) struct Bearer {
    token: Option<String>,
}

impl Interceptor for Bearer {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        if let Some(ref token) = self.token {
            let value = MetadataValue::try_from(format!("Bearer {token}"))
                .map_err(|e| Status::unauthenticated(e.to_string()))?;
            request.metadata_mut().insert("authorization", value);
        }
        Ok(request)
    }
}

pub type ExtClient =
    InferstreamServiceClient<tonic::service::interceptor::InterceptedService<Channel, Bearer>>;
pub type OipClient =
    GrpcInferenceServiceClient<tonic::service::interceptor::InterceptedService<Channel, Bearer>>;

pub struct Clients {
    pub ext: ExtClient,
    pub oip: OipClient,
}

/// Accept `host:port` or a full `http(s)://` URL.
pub fn endpoint(addr: &str) -> String {
    let trimmed = addr.trim();
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

pub async fn connect(addr: &str, token: Option<&str>) -> Result<Clients, HarnessError> {
    let endpoint = endpoint(addr);
    let channel = Channel::from_shared(endpoint.clone())
        .map_err(|e| HarnessError::Addr(format!("{endpoint}: {e}")))?
        .connect_timeout(Duration::from_secs(CONNECT_SECS))
        .connect()
        .await
        .map_err(|source| HarnessError::Connect {
            endpoint: endpoint.clone(),
            source,
        })?;
    let bearer = Bearer {
        token: token
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(ToOwned::to_owned),
    };
    Ok(Clients {
        ext: InferstreamServiceClient::with_interceptor(channel.clone(), bearer.clone()),
        oip: GrpcInferenceServiceClient::with_interceptor(channel, bearer),
    })
}

fn timed<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.set_timeout(Duration::from_secs(RPC_SECS));
    request
}

pub async fn server_live(oip: &mut OipClient) -> Result<ServerLiveResponse, HarnessError> {
    oip.server_live(timed(ServerLiveRequest {}))
        .await
        .map(|r| r.into_inner())
        .map_err(|e| HarnessError::Rpc(format!("ServerLive: {e}")))
}

pub async fn list_models(ext: &mut ExtClient) -> Result<ListModelsResponse, HarnessError> {
    ext.list_models(timed(ListModelsRequest {}))
        .await
        .map(|r| r.into_inner())
        .map_err(|e| HarnessError::Rpc(format!("ListModels: {e}")))
}

pub async fn tokenize(
    ext: &mut ExtClient,
    model: &str,
    texts: Vec<String>,
) -> Result<inferstream_protocol::extension::TokenizeResponse, Status> {
    ext.tokenize(timed(TokenizeRequest {
        model_name: model.to_string(),
        texts,
        ..Default::default()
    }))
    .await
    .map(|r| r.into_inner())
}

pub async fn detokenize(
    ext: &mut ExtClient,
    model: &str,
    sequences: Vec<Vec<u32>>,
    skip_special_tokens: bool,
) -> Result<inferstream_protocol::extension::DetokenizeResponse, Status> {
    ext.detokenize(timed(DetokenizeRequest {
        model_name: model.to_string(),
        sequences: sequences.into_iter().map(|ids| TokenIds { ids }).collect(),
        skip_special_tokens,
    }))
    .await
    .map(|r| r.into_inner())
}

pub async fn rerank(
    ext: &mut ExtClient,
    model: &str,
    query: &str,
    documents: Vec<String>,
    top_n: u32,
) -> Result<RerankResponse, Status> {
    ext.rerank(timed(RerankRequest {
        model_name: model.to_string(),
        query: query.to_string(),
        documents,
        top_n,
        return_documents: false,
        raw_scores: false,
    }))
    .await
    .map(|r| r.into_inner())
}

pub async fn embed(
    ext: &mut ExtClient,
    model: &str,
    texts: Vec<String>,
    normalize: bool,
) -> Result<inferstream_protocol::extension::EmbedResponse, Status> {
    embed_with(ext, model, texts, normalize, None).await
}

/// Embed with an optional pooling override (`mean` / `cls`) so every arch
/// receives the same family convention the catalog documents.
pub async fn embed_with(
    ext: &mut ExtClient,
    model: &str,
    texts: Vec<String>,
    normalize: bool,
    pooling: Option<&str>,
) -> Result<inferstream_protocol::extension::EmbedResponse, Status> {
    ext.embed(timed(EmbedRequest {
        model_name: model.to_string(),
        texts,
        pooling: pooling.unwrap_or("").to_string(),
        normalize: Some(normalize),
        ..Default::default()
    }))
    .await
    .map(|r| r.into_inner())
}

pub struct StreamResult {
    pub chunks: usize,
    pub tokens: Vec<String>,
    pub saw_final: bool,
}

pub async fn stream_infer(
    oip: &mut OipClient,
    model: &str,
    prompt: &str,
    max_tokens: i64,
) -> Result<StreamResult, Status> {
    let request = ModelInferRequest {
        model_name: model.to_string(),
        id: format!("e2e-{model}"),
        inputs: vec![InferInputTensor {
            name: "text".to_string(),
            datatype: DataType::Bytes.as_oip().to_string(),
            shape: vec![1],
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_input_contents: vec![pack_bytes(&[prompt.as_bytes()])],
        parameters: HashMap::from([(
            "max_tokens".to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::Int64Param(max_tokens)),
            },
        )]),
        ..Default::default()
    };
    let mut stream = oip
        .model_stream_infer(timed(tokio_stream::once(request)))
        .await?
        .into_inner();

    let mut chunks = 0usize;
    let mut tokens = Vec::new();
    let mut saw_final = false;
    while let Some(message) = stream.message().await? {
        if !message.error_message.is_empty() {
            return Err(Status::internal(message.error_message));
        }
        let Some(response) = message.infer_response else {
            continue;
        };
        let is_final = matches!(
            response
                .parameters
                .get("final")
                .and_then(|p| p.parameter_choice.as_ref()),
            Some(ParameterChoice::BoolParam(true))
        );
        if is_final {
            saw_final = true;
        }
        if let Some(raw) = response.raw_output_contents.first() {
            if !raw.is_empty() {
                if let Ok(parts) = unpack_bytes(raw) {
                    for part in parts {
                        if !part.is_empty() {
                            chunks += 1;
                            tokens.push(String::from_utf8_lossy(&part).into_owned());
                        }
                    }
                }
            }
        }
    }
    Ok(StreamResult {
        chunks,
        tokens,
        saw_final,
    })
}

pub fn model_map(listing: &ListModelsResponse) -> HashMap<String, ModelInfo> {
    listing
        .models
        .iter()
        .cloned()
        .map(|m| (m.name.clone(), m))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_adds_http() {
        assert_eq!(endpoint("gpu-lab:8461"), "http://gpu-lab:8461");
        assert_eq!(endpoint("http://127.0.0.1:8461"), "http://127.0.0.1:8461");
    }
}
