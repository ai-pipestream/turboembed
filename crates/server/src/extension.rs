//! The `inferstream.v1.InferstreamService` implementation — the clearly
//! marked extension surface next to the interoperable OIP service:
//! Tokenize / Detokenize / Embed / EmbedStream / ListModels / Rerank.
//! Embed maps the TurboEmbed C ABI (typed float[] or packed LE FP32).

use std::pin::Pin;
use std::sync::Arc;

use tokio_stream::Stream;
use tonic::{Request, Response, Status};
use tracing::debug;

use inferstream_backend::{Backend, BackendError, PackedEmbed, Registry, TokenizeOptions};
use inferstream_protocol::extension::inferstream_service_server::InferstreamService;
use inferstream_protocol::extension::{
    DetokenizeRequest, DetokenizeResponse, EmbedChunk, EmbedOutputFormat, EmbedRequest,
    EmbedResponse, Embedding, ListModelsRequest, ListModelsResponse, ModelInfo, RerankRequest,
    RerankResponse, RerankResult, TokenizeRequest, TokenizeResponse,
};
use inferstream_protocol::output_scratch;
use inferstream_protocol::tensor::{unpack_fp32, DataType};
use inferstream_protocol::Bytes;

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

    /// Shared Embed / EmbedStream path: texts → rented LE FP32 slab.
    async fn embed_blob(&self, req: &EmbedRequest) -> Result<(PackedEmbed, Vec<u8>), Status> {
        if req.texts.is_empty() {
            return Err(Status::invalid_argument("texts must not be empty"));
        }
        let backend = self.backend_for(&req.model_name)?;
        let estimate = req.texts.len().saturating_mul(4).saturating_mul(64);
        let mut dest = output_scratch::rent_bytes(estimate);
        debug!(model = %req.model_name, batch = req.texts.len(), "embed");
        let meta = match backend
            .embed_packed_into(
                &req.model_name,
                &req.texts,
                &req.pooling,
                req.normalize,
                req.truncate_to,
                &mut dest,
            )
            .await
        {
            Ok(meta) => meta,
            Err(err) => {
                output_scratch::recycle_bytes(dest);
                return Err(status_from(err));
            }
        };
        if meta.count as usize != req.texts.len() {
            output_scratch::recycle_bytes(dest);
            return Err(Status::internal(format!(
                "backend returned {} embeddings for {} texts",
                meta.count,
                req.texts.len()
            )));
        }
        let want = (meta.count as usize)
            .saturating_mul(meta.dim as usize)
            .saturating_mul(4);
        if dest.len() != want {
            let got = dest.len();
            output_scratch::recycle_bytes(dest);
            return Err(Status::internal(format!(
                "embedding blob length {got} != count {} * dim {} * 4",
                meta.count, meta.dim
            )));
        }
        Ok((meta, dest))
    }
}

fn embed_output_format(req: &EmbedRequest) -> EmbedOutputFormat {
    EmbedOutputFormat::try_from(req.output_format).unwrap_or(EmbedOutputFormat::Typed)
}

fn encode_embed_response(
    req: &EmbedRequest,
    meta: PackedEmbed,
    dest: Vec<u8>,
) -> Result<EmbedResponse, Status> {
    let dim = meta.dim as usize;
    match embed_output_format(req) {
        EmbedOutputFormat::PackedBytes => Ok(EmbedResponse {
            dim: meta.dim,
            embeddings: Vec::new(),
            model_name: meta.model_name,
            model_version: meta.model_version,
            packed_embeddings: output_scratch::adopt_bytes(dest),
        }),
        EmbedOutputFormat::Typed => {
            let values = match unpack_fp32(&dest) {
                Ok(v) => v,
                Err(e) => {
                    output_scratch::recycle_bytes(dest);
                    return Err(Status::internal(format!("malformed FP32 blob: {e}")));
                }
            };
            output_scratch::recycle_bytes(dest);
            Ok(EmbedResponse {
                dim: meta.dim,
                embeddings: values
                    .chunks_exact(dim)
                    .map(|chunk| Embedding {
                        values: chunk.to_vec(),
                    })
                    .collect(),
                model_name: meta.model_name,
                model_version: meta.model_version,
                packed_embeddings: Bytes::new(),
            })
        }
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
        let (meta, dest) = self.embed_blob(&req).await?;
        Ok(Response::new(encode_embed_response(&req, meta, dest)?))
    }

    type EmbedStreamStream =
        Pin<Box<dyn Stream<Item = Result<EmbedChunk, Status>> + Send + 'static>>;

    async fn embed_stream(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<Self::EmbedStreamStream>, Status> {
        let req = request.into_inner();
        let packed = embed_output_format(&req) == EmbedOutputFormat::PackedBytes;
        let (meta, dest) = self.embed_blob(&req).await?;
        let dim = meta.dim as usize;
        let n = meta.count as usize;
        let row_bytes = dim.saturating_mul(4);
        if packed {
            let blob = output_scratch::adopt_bytes(dest);
            let mut chunks = Vec::with_capacity(n);
            for i in 0..n {
                let start = i * row_bytes;
                chunks.push(Ok(EmbedChunk {
                    index: i as u32,
                    embedding: None,
                    packed_row: blob.slice(start..start + row_bytes),
                    r#final: i + 1 == n,
                }));
            }
            return Ok(Response::new(Box::pin(tokio_stream::iter(chunks))));
        }
        let values = match unpack_fp32(&dest) {
            Ok(v) => v,
            Err(e) => {
                output_scratch::recycle_bytes(dest);
                return Err(Status::internal(format!("malformed FP32 blob: {e}")));
            }
        };
        output_scratch::recycle_bytes(dest);
        let mut chunks = Vec::with_capacity(n);
        for i in 0..n {
            let row = &values[i * dim..(i + 1) * dim];
            chunks.push(Ok(EmbedChunk {
                index: i as u32,
                embedding: Some(Embedding {
                    values: row.to_vec(),
                }),
                packed_row: Bytes::new(),
                r#final: i + 1 == n,
            }));
        }
        Ok(Response::new(Box::pin(tokio_stream::iter(chunks))))
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
        // TEI `--max-client-batch-size` default. Catalog CE aliases also
        // enforce model.max_batch_size inside TurboRerankBackend.
        const MAX_RERANK_DOCUMENTS: usize = 32;
        if req.documents.len() > MAX_RERANK_DOCUMENTS {
            return Err(Status::invalid_argument(format!(
                "documents length {} exceeds max_client_batch_size {MAX_RERANK_DOCUMENTS}",
                req.documents.len()
            )));
        }
        let backend = self.backend_for(&req.model_name)?;
        debug!(model = %req.model_name, docs = req.documents.len(), "rerank");
        let mut scores = output_scratch::rent_f32(req.documents.len());
        if let Err(err) = backend
            .rerank_into(
                &req.model_name,
                &req.query,
                &req.documents,
                req.raw_scores,
                &mut scores,
            )
            .await
        {
            output_scratch::recycle_f32(scores);
            return Err(status_from(err));
        }
        if scores.len() != req.documents.len() {
            let n = scores.len();
            output_scratch::recycle_f32(scores);
            return Err(Status::internal(format!(
                "backend returned {} scores for {} documents",
                n,
                req.documents.len()
            )));
        }
        let mut results: Vec<RerankResult> = scores
            .iter()
            .enumerate()
            .map(|(index, score)| RerankResult {
                index: index as u32,
                score: *score,
                document: if req.return_documents {
                    req.documents[index].clone()
                } else {
                    String::new()
                },
            })
            .collect();
        output_scratch::recycle_f32(scores);
        // Descending score; ties keep input order (stable sort).
        // Library / TurboRerank scores stay in input order; sort lives here.
        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        if req.top_n > 0 {
            results.truncate(req.top_n as usize);
        }
        Ok(Response::new(RerankResponse { results }))
    }
}
