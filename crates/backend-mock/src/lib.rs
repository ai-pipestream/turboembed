//! Deterministic mock backend for inferstream.
//!
//! Serves any model name it is registered under. Behavior:
//!
//! * **Unary infer** — expects one `BYTES` input tensor named `text` and
//!   returns one `FP32` output tensor named `embedding` of dimension
//!   [`MockBackend::embedding_dim`], packed into `raw_output_contents`. The
//!   embedding is a deterministic FNV-1a hash expansion of the input bytes,
//!   so tests can assert exact values.
//! * **Streaming infer** — emits [`MockBackend::stream_chunks`] chunks per
//!   request. Each chunk carries a `BYTES` output tensor named `token` with
//!   content `"tok-{i}"` plus a final-flag parameter on the last chunk. Every
//!   chunk echoes the request `id`.

use std::collections::HashMap;

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata, ResponseStream};
use inferstream_protocol::inference::{
    infer_parameter::ParameterChoice, model_infer_response::InferOutputTensor,
    model_metadata_response::TensorMetadata, InferParameter, ModelInferRequest, ModelInferResponse,
};
use inferstream_protocol::tensor::{pack_bytes, pack_fp32, unpack_bytes, DataType};

/// Deterministic mock backend.
#[derive(Debug, Clone)]
pub struct MockBackend {
    embedding_dim: usize,
    stream_chunks: usize,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self {
            embedding_dim: 8,
            stream_chunks: 4,
        }
    }
}

impl MockBackend {
    pub fn new(embedding_dim: usize, stream_chunks: usize) -> Self {
        Self {
            embedding_dim,
            stream_chunks,
        }
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn stream_chunks(&self) -> usize {
        self.stream_chunks
    }

    /// Deterministic pseudo-embedding: FNV-1a over the input, re-hashed per
    /// dimension, mapped into [-1, 1).
    pub fn embed(&self, input: &[u8]) -> Vec<f32> {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut hash = FNV_OFFSET;
        for &byte in input {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        (0..self.embedding_dim)
            .map(|dim| {
                let mut h = hash ^ (dim as u64).wrapping_mul(FNV_PRIME);
                h ^= h >> 33;
                h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
                h ^= h >> 33;
                // Map the top 24 bits to [-1, 1).
                ((h >> 40) as f32) / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn text_input(request: &ModelInferRequest) -> Result<Vec<u8>, BackendError> {
        let (index, tensor) = request
            .inputs
            .iter()
            .enumerate()
            .find(|(_, t)| t.name == "text")
            .ok_or_else(|| {
                BackendError::InvalidRequest("expected an input tensor named \"text\"".into())
            })?;
        if tensor.datatype != DataType::Bytes.as_oip() {
            return Err(BackendError::InvalidRequest(format!(
                "input \"text\" must be BYTES, got {:?}",
                tensor.datatype
            )));
        }
        let raw = request.raw_input_contents.get(index).ok_or_else(|| {
            BackendError::InvalidRequest(
                "input \"text\" must be sent via raw_input_contents".into(),
            )
        })?;
        let elements = unpack_bytes(raw)
            .map_err(|e| BackendError::InvalidRequest(format!("malformed BYTES payload: {e}")))?;
        elements.into_iter().next().ok_or_else(|| {
            BackendError::InvalidRequest("input \"text\" contained no elements".into())
        })
    }
}

#[async_trait]
impl Backend for MockBackend {
    fn id(&self) -> &str {
        "mock"
    }

    async fn model_ready(&self, _model_name: &str, _model_version: &str) -> bool {
        true
    }

    async fn model_metadata(
        &self,
        model_name: &str,
        _model_version: &str,
    ) -> Result<ModelMetadata, BackendError> {
        Ok(ModelMetadata {
            name: model_name.to_string(),
            versions: vec!["1".to_string()],
            platform: "mock".to_string(),
            inputs: vec![TensorMetadata {
                name: "text".to_string(),
                datatype: DataType::Bytes.as_oip().to_string(),
                shape: vec![1],
            }],
            outputs: vec![TensorMetadata {
                name: "embedding".to_string(),
                datatype: DataType::Fp32.as_oip().to_string(),
                shape: vec![self.embedding_dim as i64],
            }],
            properties: HashMap::from([(
                "description".to_string(),
                "deterministic mock backend".to_string(),
            )]),
        })
    }

    async fn infer(&self, request: ModelInferRequest) -> Result<ModelInferResponse, BackendError> {
        let text = Self::text_input(&request)?;
        let embedding = self.embed(&text);
        Ok(ModelInferResponse {
            model_name: request.model_name,
            model_version: request.model_version,
            id: request.id,
            parameters: HashMap::new(),
            outputs: vec![InferOutputTensor {
                name: "embedding".to_string(),
                datatype: DataType::Fp32.as_oip().to_string(),
                shape: vec![self.embedding_dim as i64],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_output_contents: vec![pack_fp32(&embedding)],
        })
    }

    async fn infer_stream(
        &self,
        request: ModelInferRequest,
    ) -> Result<ResponseStream, BackendError> {
        // Validate input up front so malformed requests fail as a
        // per-request error rather than a broken stream.
        Self::text_input(&request)?;
        let chunks = self.stream_chunks.max(1);
        let responses: Vec<Result<ModelInferResponse, BackendError>> = (0..chunks)
            .map(|i| {
                let token = format!("tok-{i}");
                let is_final = i + 1 == chunks;
                Ok(ModelInferResponse {
                    model_name: request.model_name.clone(),
                    model_version: request.model_version.clone(),
                    id: request.id.clone(),
                    parameters: HashMap::from([(
                        "final".to_string(),
                        InferParameter {
                            parameter_choice: Some(ParameterChoice::BoolParam(is_final)),
                        },
                    )]),
                    outputs: vec![InferOutputTensor {
                        name: "token".to_string(),
                        datatype: DataType::Bytes.as_oip().to_string(),
                        shape: vec![1],
                        parameters: HashMap::new(),
                        contents: None,
                    }],
                    raw_output_contents: vec![pack_bytes(&[token.as_bytes()])],
                })
            })
            .collect();
        Ok(Box::pin(futures::stream::iter(responses)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use inferstream_protocol::inference::model_infer_request::InferInputTensor;
    use inferstream_protocol::tensor::unpack_fp32;

    fn text_request(id: &str, text: &str) -> ModelInferRequest {
        ModelInferRequest {
            model_name: "mock-embed".into(),
            id: id.into(),
            inputs: vec![InferInputTensor {
                name: "text".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_input_contents: vec![pack_bytes(&[text.as_bytes()])],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn unary_infer_is_deterministic() {
        let backend = MockBackend::default();
        let a = backend.infer(text_request("r1", "hello")).await.unwrap();
        let b = backend.infer(text_request("r2", "hello")).await.unwrap();
        assert_eq!(a.id, "r1");
        assert_eq!(a.raw_output_contents, b.raw_output_contents);
        let embedding = unpack_fp32(&a.raw_output_contents[0]).unwrap();
        assert_eq!(embedding.len(), backend.embedding_dim());
        assert!(embedding.iter().all(|v| (-1.0..1.0).contains(v)));
        let c = backend
            .infer(text_request("r3", "different"))
            .await
            .unwrap();
        assert_ne!(a.raw_output_contents, c.raw_output_contents);
    }

    #[tokio::test]
    async fn stream_infer_emits_chunks_with_request_id() {
        let backend = MockBackend::new(8, 3);
        let stream = backend
            .infer_stream(text_request("stream-1", "hi"))
            .await
            .unwrap();
        let chunks: Vec<_> = stream.map(|c| c.unwrap()).collect().await;
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.id, "stream-1");
        }
        let tokens: Vec<String> = chunks
            .iter()
            .map(|c| {
                let elements = unpack_bytes(&c.raw_output_contents[0]).unwrap();
                String::from_utf8(elements[0].clone()).unwrap()
            })
            .collect();
        assert_eq!(tokens, ["tok-0", "tok-1", "tok-2"]);
        let finals: Vec<bool> = chunks
            .iter()
            .map(|c| {
                matches!(
                    c.parameters
                        .get("final")
                        .and_then(|p| p.parameter_choice.as_ref()),
                    Some(ParameterChoice::BoolParam(true))
                )
            })
            .collect();
        assert_eq!(finals, [false, false, true]);
    }

    #[tokio::test]
    async fn missing_text_input_is_invalid() {
        let backend = MockBackend::default();
        let request = ModelInferRequest {
            model_name: "mock-embed".into(),
            ..Default::default()
        };
        assert!(matches!(
            backend.infer(request).await,
            Err(BackendError::InvalidRequest(_))
        ));
    }
}
