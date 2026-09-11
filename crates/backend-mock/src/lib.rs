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
use inferstream_backend::{Backend, BackendError, ModelMetadata, ResponseStream, TokenizeOptions};
use inferstream_protocol::extension::{Encoding, Offset};
use inferstream_protocol::inference::{
    infer_parameter::ParameterChoice, model_infer_response::InferOutputTensor,
    model_metadata_response::TensorMetadata, InferParameter, ModelInferRequest, ModelInferResponse,
};
use inferstream_protocol::tensor::{pack_bytes, pack_fp32, unpack_bytes, DataType};

/// Mock tokenizer special token ids (byte `b` maps to `b + 3`).
pub const MOCK_PAD_ID: u32 = 0;
pub const MOCK_BOS_ID: u32 = 1;
pub const MOCK_EOS_ID: u32 = 2;

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

    /// All elements of the `text` input tensor (batch support).
    fn text_batch(request: &ModelInferRequest) -> Result<Vec<Vec<u8>>, BackendError> {
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
        if elements.is_empty() {
            return Err(BackendError::InvalidRequest(
                "input \"text\" contained no elements".into(),
            ));
        }
        Ok(elements)
    }

    fn text_input(request: &ModelInferRequest) -> Result<Vec<u8>, BackendError> {
        Ok(Self::text_batch(request)?.remove(0))
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
        let texts = Self::text_batch(&request)?;
        let batch = texts.len();
        let mut values = Vec::with_capacity(batch * self.embedding_dim);
        for text in &texts {
            values.extend(self.embed(text));
        }
        // Contract: [d] for a single text (backwards compatible), [n, d] for
        // a batch — the same shape convention as the real ORT engine.
        let shape = if batch == 1 {
            vec![self.embedding_dim as i64]
        } else {
            vec![batch as i64, self.embedding_dim as i64]
        };
        Ok(ModelInferResponse {
            model_name: request.model_name,
            model_version: request.model_version,
            id: request.id,
            parameters: HashMap::new(),
            outputs: vec![InferOutputTensor {
                name: "embedding".to_string(),
                datatype: DataType::Fp32.as_oip().to_string(),
                shape,
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_output_contents: vec![pack_fp32(&values)],
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

    /// Deterministic byte-level tokenizer: id 0 = `<pad>`, 1 = `<s>`,
    /// 2 = `</s>`, byte `b` = `b + 3`. Round-trips losslessly, so CI can
    /// exercise the Tokenize/Detokenize wire path with no tokenizer.json.
    async fn tokenize(
        &self,
        _model_name: &str,
        texts: &[String],
        options: &TokenizeOptions,
    ) -> Result<Vec<Encoding>, BackendError> {
        let special = usize::from(options.add_special_tokens) * 2;
        let mut encodings: Vec<Encoding> = texts
            .iter()
            .map(|text| {
                let mut content: Vec<(u32, String, (u32, u32))> = text
                    .bytes()
                    .enumerate()
                    .map(|(i, b)| {
                        let display = if b.is_ascii_graphic() || b == b' ' {
                            (b as char).to_string()
                        } else {
                            format!("<0x{b:02X}>")
                        };
                        (u32::from(b) + 3, display, (i as u32, i as u32 + 1))
                    })
                    .collect();
                if let Some(limit) = options.truncate_to {
                    content.truncate(limit.saturating_sub(special));
                }
                let mut encoding = Encoding::default();
                if options.add_special_tokens {
                    encoding.input_ids.push(MOCK_BOS_ID);
                    encoding.tokens.push("<s>".to_string());
                    encoding.offsets.push(Offset { start: 0, end: 0 });
                }
                for (id, token, (start, end)) in content {
                    encoding.input_ids.push(id);
                    encoding.tokens.push(token);
                    encoding.offsets.push(Offset { start, end });
                }
                if options.add_special_tokens {
                    encoding.input_ids.push(MOCK_EOS_ID);
                    encoding.tokens.push("</s>".to_string());
                    encoding.offsets.push(Offset { start: 0, end: 0 });
                }
                encoding.attention_mask = vec![1; encoding.input_ids.len()];
                if !options.with_offsets {
                    encoding.offsets.clear();
                }
                encoding
            })
            .collect();

        if options.pad_to_longest {
            let longest = encodings
                .iter()
                .map(|e| e.input_ids.len())
                .max()
                .unwrap_or(0);
            for encoding in &mut encodings {
                while encoding.input_ids.len() < longest {
                    encoding.input_ids.push(MOCK_PAD_ID);
                    encoding.attention_mask.push(0);
                    encoding.tokens.push("<pad>".to_string());
                    if options.with_offsets {
                        encoding.offsets.push(Offset { start: 0, end: 0 });
                    }
                }
            }
        }
        Ok(encodings)
    }

    async fn detokenize(
        &self,
        _model_name: &str,
        sequences: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>, BackendError> {
        sequences
            .iter()
            .map(|ids| {
                let mut bytes = Vec::with_capacity(ids.len());
                for &id in ids {
                    match id {
                        MOCK_PAD_ID | MOCK_BOS_ID | MOCK_EOS_ID => {
                            if !skip_special_tokens {
                                let token = match id {
                                    MOCK_PAD_ID => "<pad>",
                                    MOCK_BOS_ID => "<s>",
                                    _ => "</s>",
                                };
                                bytes.extend_from_slice(token.as_bytes());
                            }
                        }
                        3..=258 => bytes.push((id - 3) as u8),
                        other => {
                            return Err(BackendError::InvalidRequest(format!(
                                "token id {other} is out of the mock tokenizer's range (0..=258)"
                            )))
                        }
                    }
                }
                String::from_utf8(bytes).map_err(|e| {
                    BackendError::InvalidRequest(format!("decoded bytes are not UTF-8: {e}"))
                })
            })
            .collect()
    }

    /// Deterministic mock reranker: score = fraction of the query's
    /// lowercase words that appear in the document. Stable across runs, so
    /// tests can assert ordering.
    async fn rerank(
        &self,
        _model_name: &str,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<f32>, BackendError> {
        let query_words: Vec<String> = query
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        Ok(documents
            .iter()
            .map(|doc| {
                if query_words.is_empty() {
                    return 0.0;
                }
                let doc_lower = doc.to_lowercase();
                let hits = query_words
                    .iter()
                    .filter(|w| doc_lower.contains(w.as_str()))
                    .count();
                hits as f32 / query_words.len() as f32
            })
            .collect())
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
    async fn batch_infer_returns_n_by_d() {
        let backend = MockBackend::default();
        let dim = backend.embedding_dim();
        let request = ModelInferRequest {
            model_name: "mock-embed".into(),
            id: "batch-1".into(),
            inputs: vec![InferInputTensor {
                name: "text".into(),
                datatype: "BYTES".into(),
                shape: vec![3],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_input_contents: vec![pack_bytes(&[
                b"alpha".as_slice(),
                b"beta",
                b"alpha", // duplicate of the first: rows must match
            ])],
            ..Default::default()
        };
        let response = backend.infer(request).await.unwrap();
        assert_eq!(response.outputs[0].shape, vec![3, dim as i64]);
        let values = unpack_fp32(&response.raw_output_contents[0]).unwrap();
        assert_eq!(values.len(), 3 * dim);
        assert_eq!(values[..dim], values[2 * dim..], "same text, same row");
        assert_ne!(values[..dim], values[dim..2 * dim]);
    }

    #[tokio::test]
    async fn single_text_keeps_flat_shape() {
        let backend = MockBackend::default();
        let response = backend.infer(text_request("r1", "solo")).await.unwrap();
        assert_eq!(
            response.outputs[0].shape,
            vec![backend.embedding_dim() as i64],
            "single-text shape stays [d] for backwards compatibility"
        );
    }

    #[tokio::test]
    async fn tokenize_detokenize_round_trip() {
        use inferstream_backend::TokenizeOptions;
        let backend = MockBackend::default();
        let texts = vec!["hello world".to_string(), "λ unicode ✓".to_string()];
        let encodings = backend
            .tokenize("m", &texts, &TokenizeOptions::default())
            .await
            .unwrap();
        assert_eq!(encodings.len(), 2);
        // Specials framed around content.
        assert_eq!(encodings[0].input_ids.first(), Some(&MOCK_BOS_ID));
        assert_eq!(encodings[0].input_ids.last(), Some(&MOCK_EOS_ID));
        assert!(encodings[0].attention_mask.iter().all(|&m| m == 1));

        let sequences: Vec<Vec<u32>> = encodings.iter().map(|e| e.input_ids.clone()).collect();
        let decoded = backend.detokenize("m", &sequences, true).await.unwrap();
        assert_eq!(decoded, texts, "skip-specials decode round-trips exactly");

        let with_specials = backend.detokenize("m", &sequences, false).await.unwrap();
        assert_eq!(with_specials[0], "<s>hello world</s>");
    }

    #[tokio::test]
    async fn tokenize_no_special_tokens() {
        use inferstream_backend::TokenizeOptions;
        let backend = MockBackend::default();
        let encodings = backend
            .tokenize(
                "m",
                &["ab".to_string()],
                &TokenizeOptions {
                    add_special_tokens: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            encodings[0].input_ids,
            vec![u32::from(b'a') + 3, u32::from(b'b') + 3]
        );
        assert_eq!(encodings[0].tokens, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn tokenize_truncation_counts_specials() {
        use inferstream_backend::TokenizeOptions;
        let backend = MockBackend::default();
        let encodings = backend
            .tokenize(
                "m",
                &["abcdef".to_string()],
                &TokenizeOptions {
                    truncate_to: Some(4),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        // 4 total = BOS + 2 content + EOS.
        assert_eq!(encodings[0].input_ids.len(), 4);
        assert_eq!(encodings[0].tokens, vec!["<s>", "a", "b", "</s>"]);
    }

    #[tokio::test]
    async fn tokenize_pads_batch_to_longest() {
        use inferstream_backend::TokenizeOptions;
        let backend = MockBackend::default();
        let encodings = backend
            .tokenize(
                "m",
                &["hi".to_string(), "longer text".to_string()],
                &TokenizeOptions {
                    pad_to_longest: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let longest = encodings[1].input_ids.len();
        assert_eq!(encodings[0].input_ids.len(), longest);
        assert_eq!(encodings[0].input_ids.last(), Some(&MOCK_PAD_ID));
        // Mask marks padding with 0, real tokens with 1.
        let real: u32 = encodings[0].attention_mask.iter().sum();
        assert_eq!(real, 4, "BOS + 'h' + 'i' + EOS");
        assert!(encodings[0].attention_mask.ends_with(&[0]));
    }

    #[tokio::test]
    async fn tokenize_offsets_map_back_into_text() {
        use inferstream_backend::TokenizeOptions;
        let backend = MockBackend::default();
        let text = "abc".to_string();
        let encodings = backend
            .tokenize(
                "m",
                std::slice::from_ref(&text),
                &TokenizeOptions {
                    with_offsets: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let offsets = &encodings[0].offsets;
        assert_eq!(offsets.len(), encodings[0].input_ids.len());
        // Content offsets slice the original text; specials carry (0, 0).
        assert_eq!((offsets[0].start, offsets[0].end), (0, 0));
        assert_eq!((offsets[1].start, offsets[1].end), (0, 1));
        assert_eq!(&text[offsets[2].start as usize..offsets[2].end as usize], "b");
    }

    #[tokio::test]
    async fn detokenize_rejects_out_of_range_ids() {
        let backend = MockBackend::default();
        let result = backend.detokenize("m", &[vec![9999]], false).await;
        assert!(matches!(result, Err(BackendError::InvalidRequest(_))));
    }

    #[tokio::test]
    async fn rerank_scores_are_deterministic_and_ordered() {
        let backend = MockBackend::default();
        let docs = vec![
            "rust inference server".to_string(),
            "cooking pasta at home".to_string(),
            "fast rust gRPC inference".to_string(),
        ];
        let scores = backend
            .rerank("m", "rust inference", &docs)
            .await
            .unwrap();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0], 1.0, "both query words hit");
        assert_eq!(scores[1], 0.0, "no query words hit");
        assert_eq!(scores[2], 1.0);
        let again = backend
            .rerank("m", "rust inference", &docs)
            .await
            .unwrap();
        assert_eq!(scores, again);
    }

    #[tokio::test]
    async fn rerank_empty_query_scores_zero() {
        let backend = MockBackend::default();
        let scores = backend
            .rerank("m", "", &["anything".to_string()])
            .await
            .unwrap();
        assert_eq!(scores, vec![0.0]);
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
