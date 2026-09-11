//! The `inferstream.v1.InferstreamService` implementation — the clearly
//! marked extension surface next to the interoperable OIP service:
//! Tokenize / Detokenize / Embed / ListModels / Rerank.

use std::collections::HashMap;
use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::debug;

use inferstream_backend::{Backend, BackendError, Registry, TokenizeOptions};
use inferstream_protocol::extension::inferstream_service_server::InferstreamService;
use inferstream_protocol::extension::{
    DetokenizeRequest, DetokenizeResponse, EmbedRequest, EmbedResponse, Embedding,
    ListModelsRequest, ListModelsResponse, ModelInfo, RerankRequest, RerankResponse, RerankResult,
    TokenizeRequest, TokenizeResponse,
};
use inferstream_protocol::inference::{
    infer_parameter::ParameterChoice, model_infer_request::InferInputTensor, InferParameter,
    ModelInferRequest,
};
use inferstream_protocol::tensor::{pack_bytes, unpack_fp32, DataType};

use crate::tokenizer::TokenizerMap;

/// The extension service. Shares the OIP service's routing table and adds a
/// local tokenizer map for models with a configured `tokenizer.json`.
pub struct ExtensionService {
    registry: Arc<Registry>,
    tokenizers: Arc<TokenizerMap>,
}

impl ExtensionService {
    pub fn new(registry: Arc<Registry>, tokenizers: Arc<TokenizerMap>) -> Self {
        Self {
            registry,
            tokenizers,
        }
    }

    fn backend_for(&self, model_name: &str) -> Result<Arc<dyn Backend>, Status> {
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

fn string_param(value: &str) -> InferParameter {
    InferParameter {
        parameter_choice: Some(ParameterChoice::StringParam(value.to_string())),
    }
}

fn bool_param(value: bool) -> InferParameter {
    InferParameter {
        parameter_choice: Some(ParameterChoice::BoolParam(value)),
    }
}

fn int_param(value: i64) -> InferParameter {
    InferParameter {
        parameter_choice: Some(ParameterChoice::Int64Param(value)),
    }
}

#[tonic::async_trait]
impl InferstreamService for ExtensionService {
    async fn tokenize(
        &self,
        request: Request<TokenizeRequest>,
    ) -> Result<Response<TokenizeResponse>, Status> {
        let req = request.into_inner();
        if req.texts.is_empty() {
            return Err(Status::invalid_argument("texts must not be empty"));
        }
        // Reject unknown models even when a local tokenizer probe would fail
        // first — keeps NotFound semantics identical to the OIP surface.
        let backend = self.backend_for(&req.model_name)?;
        let options = TokenizeOptions {
            add_special_tokens: !req.no_special_tokens,
            with_offsets: req.with_offsets,
            truncate_to: (req.truncate_to > 0).then_some(req.truncate_to as usize),
            pad_to_longest: req.pad_to_longest,
        };
        debug!(model = %req.model_name, batch = req.texts.len(), "tokenize");
        let encodings = match self.tokenizers.get(&req.model_name) {
            Some(local) => local.tokenize(&req.texts, &options).map_err(status_from)?,
            None => backend
                .tokenize(&req.model_name, &req.texts, &options)
                .await
                .map_err(status_from)?,
        };
        Ok(Response::new(TokenizeResponse { encodings }))
    }

    async fn detokenize(
        &self,
        request: Request<DetokenizeRequest>,
    ) -> Result<Response<DetokenizeResponse>, Status> {
        let req = request.into_inner();
        if req.sequences.is_empty() {
            return Err(Status::invalid_argument("sequences must not be empty"));
        }
        let backend = self.backend_for(&req.model_name)?;
        let sequences: Vec<Vec<u32>> = req.sequences.into_iter().map(|s| s.ids).collect();
        debug!(model = %req.model_name, batch = sequences.len(), "detokenize");
        let texts = match self.tokenizers.get(&req.model_name) {
            Some(local) => local
                .detokenize(&sequences, req.skip_special_tokens)
                .map_err(status_from)?,
            None => backend
                .detokenize(&req.model_name, &sequences, req.skip_special_tokens)
                .await
                .map_err(status_from)?,
        };
        Ok(Response::new(DetokenizeResponse { texts }))
    }

    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let req = request.into_inner();
        if req.texts.is_empty() {
            return Err(Status::invalid_argument("texts must not be empty"));
        }
        let backend = self.backend_for(&req.model_name)?;

        // Wrap into the OIP ModelInfer convention: BYTES "text" tensor plus
        // pooling / normalize / truncate as InferParameters.
        let mut parameters = HashMap::new();
        if !req.pooling.is_empty() {
            parameters.insert("pooling".to_string(), string_param(&req.pooling));
        }
        if let Some(normalize) = req.normalize {
            parameters.insert("normalize".to_string(), bool_param(normalize));
        }
        if req.truncate_to > 0 {
            parameters.insert(
                "truncate".to_string(),
                int_param(i64::from(req.truncate_to)),
            );
        }
        let text_bytes: Vec<&[u8]> = req.texts.iter().map(|t| t.as_bytes()).collect();
        let infer_request = ModelInferRequest {
            model_name: req.model_name.clone(),
            parameters,
            inputs: vec![InferInputTensor {
                name: "text".to_string(),
                datatype: DataType::Bytes.as_oip().to_string(),
                shape: vec![req.texts.len() as i64],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_input_contents: vec![pack_bytes(&text_bytes)],
            ..Default::default()
        };
        debug!(model = %req.model_name, batch = req.texts.len(), "embed");
        let response = backend.infer(infer_request).await.map_err(status_from)?;

        // Unpack the FP32 "embedding" output: [d] for one text, [n, d] batch.
        let (index, output) = response
            .outputs
            .iter()
            .enumerate()
            .find(|(_, o)| o.name == "embedding")
            .ok_or_else(|| {
                Status::internal("backend returned no output tensor named \"embedding\"")
            })?;
        if output.datatype != DataType::Fp32.as_oip() {
            return Err(Status::internal(format!(
                "output \"embedding\" must be FP32, backend returned {:?}",
                output.datatype
            )));
        }
        let raw = response
            .raw_output_contents
            .get(index)
            .ok_or_else(|| Status::internal("backend returned no raw content for \"embedding\""))?;
        let values =
            unpack_fp32(raw).map_err(|e| Status::internal(format!("malformed FP32 blob: {e}")))?;
        let dim = output
            .shape
            .last()
            .copied()
            .filter(|&d| d > 0)
            .ok_or_else(|| Status::internal("embedding output reported an empty shape"))?
            as usize;
        if values.len() % dim != 0 {
            return Err(Status::internal(format!(
                "embedding blob length {} is not a multiple of dim {dim}",
                values.len()
            )));
        }
        let rows: Vec<Embedding> = values
            .chunks_exact(dim)
            .map(|chunk| Embedding {
                values: chunk.to_vec(),
            })
            .collect();
        if rows.len() != req.texts.len() {
            return Err(Status::internal(format!(
                "backend returned {} embeddings for {} texts",
                rows.len(),
                req.texts.len()
            )));
        }
        Ok(Response::new(EmbedResponse {
            dim: dim as u32,
            embeddings: rows,
            model_name: response.model_name,
            model_version: response.model_version,
        }))
    }

    async fn list_models(
        &self,
        _request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let mut names: Vec<&str> = self.registry.model_names().collect();
        names.sort_unstable();
        let mut models = Vec::with_capacity(names.len());
        for name in names {
            let backend = match self.registry.lookup(name) {
                Some(backend) => backend,
                None => continue,
            };
            let ready = backend.model_ready(name, "").await;
            let (platform, versions, embedding_dim) = match backend.model_metadata(name, "").await {
                Ok(metadata) => {
                    let dim = metadata
                        .outputs
                        .iter()
                        .find(|o| o.name == "embedding" && o.datatype == DataType::Fp32.as_oip())
                        .and_then(|o| o.shape.last().copied())
                        .filter(|&d| d > 0)
                        .unwrap_or(0);
                    (metadata.platform, metadata.versions, dim)
                }
                // Engines that cannot report metadata (e.g. stub builds) still
                // appear in the listing with what the registry knows.
                Err(_) => (String::new(), Vec::new(), 0),
            };
            // A model is tokenizable when the server has a local tokenizer or
            // the backend implements the tokenize surface (probed with an
            // empty batch, which is side-effect free).
            let has_tokenizer = self.tokenizers.contains_key(name)
                || backend
                    .tokenize(name, &[], &TokenizeOptions::default())
                    .await
                    .is_ok();
            models.push(ModelInfo {
                name: name.to_string(),
                backend: backend.id().to_string(),
                ready,
                platform,
                versions,
                embedding_dim,
                has_tokenizer,
            });
        }
        Ok(Response::new(ListModelsResponse { models }))
    }

    async fn rerank(
        &self,
        request: Request<RerankRequest>,
    ) -> Result<Response<RerankResponse>, Status> {
        let req = request.into_inner();
        if req.documents.is_empty() {
            return Err(Status::invalid_argument("documents must not be empty"));
        }
        let backend = self.backend_for(&req.model_name)?;
        debug!(model = %req.model_name, docs = req.documents.len(), "rerank");
        let scores = backend
            .rerank(&req.model_name, &req.query, &req.documents)
            .await
            .map_err(status_from)?;
        if scores.len() != req.documents.len() {
            return Err(Status::internal(format!(
                "backend returned {} scores for {} documents",
                scores.len(),
                req.documents.len()
            )));
        }
        let mut results: Vec<RerankResult> = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| RerankResult {
                index: index as u32,
                score,
            })
            .collect();
        // Descending score; ties keep input order (stable sort).
        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        if req.top_n > 0 {
            results.truncate(req.top_n as usize);
        }
        Ok(Response::new(RerankResponse { results }))
    }
}
