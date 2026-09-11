//! GPU/host-model integration tests for the real llama.cpp engine.
//!
//! These are `#[ignore]`d and only compile with the `runtime` feature: they
//! need a GGUF model on disk (see `/work/models/gguf/README.md` on krick) and,
//! for the CUDA path, an NVIDIA GPU. Run explicitly:
//!
//! ```text
//! cargo test -p inferstream-backend-llamacpp --features cuda -- --ignored llama
//! ```
//!
//! Environment:
//! * `INFERSTREAM_GGUF` — model path (default: the krick Qwen2.5-0.5B Q8_0)
//! * `INFERSTREAM_LLAMA_DEVICE` — `cuda` / `sycl` / `cpu` (default: cuda
//!   when built with `cuda`, sycl when built with `sycl`, else cpu)

#![cfg(feature = "runtime")]

use std::collections::HashMap;

use futures::StreamExt;
use inferstream_backend::Backend;
use inferstream_backend_llamacpp::{LlamaCppBackend, LlamaCppConfig, LlamaDevice};
use inferstream_protocol::inference::{
    infer_parameter::ParameterChoice, model_infer_request::InferInputTensor, InferParameter,
    ModelInferRequest,
};
use inferstream_protocol::tensor::{pack_bytes, unpack_bytes};

const DEFAULT_GGUF: &str = "/work/models/gguf/qwen2.5-0.5b-instruct-q8_0.gguf";

fn model_path() -> Option<String> {
    let path = std::env::var("INFERSTREAM_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.into());
    if std::path::Path::new(&path).is_file() {
        Some(path)
    } else {
        eprintln!("skipping: no GGUF at {path} (set INFERSTREAM_GGUF)");
        None
    }
}

fn device() -> LlamaDevice {
    match std::env::var("INFERSTREAM_LLAMA_DEVICE") {
        Ok(value) => LlamaDevice::from_config(&value).expect("valid device"),
        Err(_) if cfg!(feature = "cuda") => LlamaDevice::Cuda,
        Err(_) if cfg!(feature = "sycl") => LlamaDevice::Sycl,
        Err(_) => LlamaDevice::Cpu,
    }
}

fn backend() -> Option<LlamaCppBackend> {
    let model_path = model_path()?;
    Some(
        LlamaCppBackend::new(LlamaCppConfig {
            model_path,
            device: device(),
            n_ctx: Some(2048),
            ..Default::default()
        })
        .expect("gguf model loads"),
    )
}

fn generate_request(id: &str, prompt: &str, max_tokens: i64) -> ModelInferRequest {
    ModelInferRequest {
        model_name: "qwen2.5-0.5b-instruct".into(),
        id: id.into(),
        parameters: HashMap::from([(
            "max_tokens".to_string(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::Int64Param(max_tokens)),
            },
        )]),
        inputs: vec![InferInputTensor {
            name: "text".into(),
            datatype: "BYTES".into(),
            shape: vec![1],
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_input_contents: vec![pack_bytes(&[prompt.as_bytes()])],
        ..Default::default()
    }
}

#[ignore = "needs a GGUF model on disk (and a GPU for the cuda device); see module docs"]
#[tokio::test]
async fn llama_stream_emits_tokens_with_final_flag() {
    let Some(backend) = backend() else { return };
    assert!(backend.model_ready("qwen2.5-0.5b-instruct", "").await);

    let stream = backend
        .infer_stream(generate_request("stream-1", "The capital of France is", 16))
        .await
        .expect("stream starts");
    let chunks: Vec<_> = stream.map(|c| c.expect("chunk ok")).collect().await;
    assert!(!chunks.is_empty());
    let mut text = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.id, "stream-1", "request id echoes on every chunk");
        assert_eq!(chunk.outputs[0].name, "token");
        let is_final = matches!(
            chunk
                .parameters
                .get("final")
                .and_then(|p| p.parameter_choice.as_ref()),
            Some(ParameterChoice::BoolParam(true))
        );
        assert_eq!(
            is_final,
            i + 1 == chunks.len(),
            "final flag only on the last chunk"
        );
        text.extend(
            unpack_bytes(&chunk.raw_output_contents[0])
                .unwrap()
                .remove(0),
        );
    }
    let text = String::from_utf8_lossy(&text);
    eprintln!("streamed completion: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "greedy completion should mention Paris, got {text:?}"
    );
}

#[ignore = "needs a GGUF model on disk (and a GPU for the cuda device); see module docs"]
#[tokio::test]
async fn llama_unary_infer_returns_full_text() {
    let Some(backend) = backend() else { return };
    let response = backend
        .infer(generate_request("unary-1", "The capital of France is", 16))
        .await
        .expect("unary generation");
    assert_eq!(response.id, "unary-1");
    assert_eq!(response.outputs[0].name, "text");
    let text = unpack_bytes(&response.raw_output_contents[0])
        .unwrap()
        .remove(0);
    let text = String::from_utf8_lossy(&text);
    assert!(text.to_lowercase().contains("paris"), "got {text:?}");
}

#[ignore = "needs a GGUF model on disk (and a GPU for the cuda device); see module docs"]
#[tokio::test]
async fn llama_tokenize_detokenize_round_trip() {
    use inferstream_backend::TokenizeOptions;
    let Some(backend) = backend() else { return };
    let texts = vec!["Hello world".to_string()];
    let encodings = backend
        .tokenize("m", &texts, &TokenizeOptions::default())
        .await
        .expect("tokenize");
    assert!(!encodings[0].input_ids.is_empty());
    assert_eq!(encodings[0].input_ids.len(), encodings[0].tokens.len());

    let decoded = backend
        .detokenize("m", &[encodings[0].input_ids.clone()], true)
        .await
        .expect("detokenize");
    assert!(
        decoded[0].contains("Hello world"),
        "round trip preserves content, got {decoded:?}"
    );
}
