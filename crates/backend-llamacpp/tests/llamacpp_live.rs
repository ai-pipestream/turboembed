//! Live tests against a running llama-server (server-client mode).
//!
//! Ignored by default; they need a reachable llama.cpp HTTP server:
//!
//! ```bash
//! INFERSTREAM_LLAMACPP_ENDPOINT=http://127.0.0.1:8085 \
//!   cargo test -p inferstream-backend-llamacpp --test llamacpp_live -- --ignored
//! ```
//!
//! On krick-1 the endpoint is the `vlm-server` container
//! (`ghcr.io/ggml-org/llama.cpp:server-intel`, a GGML_SYCL build on the
//! Battlemage GPU) publishing port 8085.

use std::collections::HashMap;

use futures::StreamExt;
use inferstream_backend::{Backend, BackendError, TokenizeOptions};
use inferstream_backend_llamacpp::{LlamaCppBackend, LlamaCppConfig, LlamaDevice};
use inferstream_protocol::inference::{
    infer_parameter::ParameterChoice, model_infer_request::InferInputTensor, InferParameter,
    ModelInferRequest,
};
use inferstream_protocol::tensor::{pack_bytes, unpack_bytes};

fn live_backend() -> Option<LlamaCppBackend> {
    let endpoint = std::env::var("INFERSTREAM_LLAMACPP_ENDPOINT").ok()?;
    Some(
        LlamaCppBackend::new(LlamaCppConfig {
            endpoint: Some(endpoint),
            device: LlamaDevice::Sycl,
            ..Default::default()
        })
        .expect("endpoint mode constructs"),
    )
}

fn prompt_request(id: &str, prompt: &str, max_tokens: i64) -> ModelInferRequest {
    ModelInferRequest {
        model_name: "qwen2.5-vl-7b-sycl".into(),
        id: id.into(),
        parameters: HashMap::from([
            (
                "max_tokens".to_string(),
                InferParameter {
                    parameter_choice: Some(ParameterChoice::Int64Param(max_tokens)),
                },
            ),
            (
                "temperature".to_string(),
                InferParameter {
                    parameter_choice: Some(ParameterChoice::DoubleParam(0.0)),
                },
            ),
        ]),
        inputs: vec![InferInputTensor {
            name: "text".into(),
            datatype: "BYTES".into(),
            shape: vec![1],
            ..Default::default()
        }],
        raw_input_contents: vec![pack_bytes(&[prompt.as_bytes()])],
        ..Default::default()
    }
}

#[tokio::test]
#[ignore = "needs INFERSTREAM_LLAMACPP_ENDPOINT and a running llama-server"]
async fn health_and_metadata() {
    let backend = live_backend().expect("INFERSTREAM_LLAMACPP_ENDPOINT must be set");
    assert!(
        backend.model_ready("any", "").await,
        "server /health not ok"
    );
    let metadata = backend
        .model_metadata("qwen2.5-vl-7b-sycl", "")
        .await
        .unwrap();
    assert_eq!(metadata.platform, "llama_cpp");
    assert!(metadata.properties.contains_key("endpoint"));
}

#[tokio::test]
#[ignore = "needs INFERSTREAM_LLAMACPP_ENDPOINT and a running llama-server"]
async fn unary_completion_returns_text() {
    let backend = live_backend().expect("INFERSTREAM_LLAMACPP_ENDPOINT must be set");
    let response = backend
        .infer(prompt_request("unary-1", "The capital of France is", 8))
        .await
        .unwrap();
    assert_eq!(response.id, "unary-1");
    assert_eq!(response.outputs[0].name, "text");
    let text =
        String::from_utf8(unpack_bytes(&response.raw_output_contents[0]).unwrap()[0].clone())
            .unwrap();
    assert!(
        text.to_lowercase().contains("paris"),
        "greedy completion should mention Paris, got {text:?}"
    );
    assert!(matches!(
        response
            .parameters
            .get("tokens_predicted")
            .and_then(|p| p.parameter_choice.as_ref()),
        Some(ParameterChoice::Int64Param(n)) if *n > 0
    ));
}

#[tokio::test]
#[ignore = "needs INFERSTREAM_LLAMACPP_ENDPOINT and a running llama-server"]
async fn streaming_emits_token_chunks_with_final_flag() {
    let backend = live_backend().expect("INFERSTREAM_LLAMACPP_ENDPOINT must be set");
    let stream = backend
        .infer_stream(prompt_request("stream-1", "Count from 1 to 10:", 24))
        .await
        .unwrap();
    let chunks: Vec<_> = stream
        .map(|c| c.expect("stream chunk ok"))
        .collect::<Vec<_>>()
        .await;
    assert!(
        chunks.len() > 2,
        "real token streaming should yield many chunks, got {}",
        chunks.len()
    );
    for chunk in &chunks {
        assert_eq!(chunk.id, "stream-1");
        assert_eq!(chunk.outputs[0].name, "token");
    }
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
    assert_eq!(finals.iter().filter(|&&f| f).count(), 1);
    assert!(
        finals.last().copied().unwrap(),
        "last chunk carries final=true"
    );
    let text: String = chunks
        .iter()
        .map(|c| {
            String::from_utf8(unpack_bytes(&c.raw_output_contents[0]).unwrap()[0].clone()).unwrap()
        })
        .collect();
    assert!(!text.trim().is_empty(), "streamed text is non-empty");
}

#[tokio::test]
#[ignore = "needs INFERSTREAM_LLAMACPP_ENDPOINT and a running llama-server"]
async fn tokenize_round_trips_through_detokenize() {
    let backend = live_backend().expect("INFERSTREAM_LLAMACPP_ENDPOINT must be set");
    let texts = vec!["Hello, inferstream!".to_string()];
    let encodings = backend
        .tokenize(
            "qwen2.5-vl-7b-sycl",
            &texts,
            &TokenizeOptions {
                add_special_tokens: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(!encodings[0].input_ids.is_empty());
    assert_eq!(encodings[0].input_ids.len(), encodings[0].tokens.len());
    assert!(encodings[0].attention_mask.iter().all(|&m| m == 1));

    let sequences: Vec<Vec<u32>> = encodings.iter().map(|e| e.input_ids.clone()).collect();
    let decoded = backend
        .detokenize("qwen2.5-vl-7b-sycl", &sequences, true)
        .await
        .unwrap();
    assert_eq!(decoded[0].trim(), texts[0]);
}

#[tokio::test]
#[ignore = "needs INFERSTREAM_LLAMACPP_ENDPOINT and a running llama-server"]
async fn bad_endpoint_reports_unavailable() {
    let backend = LlamaCppBackend::new(LlamaCppConfig {
        endpoint: Some("http://127.0.0.1:1".into()),
        ..Default::default()
    })
    .unwrap();
    assert!(!backend.model_ready("m", "").await);
    assert!(matches!(
        backend.infer(prompt_request("x", "hi", 1)).await,
        Err(BackendError::Unavailable(_))
    ));
}
