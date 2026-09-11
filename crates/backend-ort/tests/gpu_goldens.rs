//! GPU reference-embedding (golden) tests for the real ONNX Runtime engine.
//!
//! These are `#[ignore]`d and only compile with the `runtime` feature, so
//! default CI never needs ONNX Runtime or a GPU. Run them on a host with the
//! model available (krick for CUDA, anywhere for the CPU EP):
//!
//! ```bash
//! INFERSTREAM_ORT_MODEL=/path/to/model.onnx \
//! INFERSTREAM_ORT_GOLDEN=testdata/reference_embeddings/ort_cuda_minilm_short.json \
//! INFERSTREAM_ORT_DEVICE=cuda \
//! cargo test -p inferstream-backend-ort --features cuda -- --ignored gpu_golden
//! ```
//!
//! Golden schema and regeneration instructions:
//! `testdata/reference_embeddings/README.md`.

#![cfg(feature = "runtime")]

use std::collections::HashMap;

use inferstream_backend::Backend;
use inferstream_backend_ort::{OrtBackend, OrtConfig, OrtDevice, Pooling};
use inferstream_protocol::inference::model_infer_request::InferInputTensor;
use inferstream_protocol::inference::ModelInferRequest;
use inferstream_protocol::tensor::{pack_bytes, unpack_fp32};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Compare the live engine against a stored golden. Skips (with a clear
/// panic message) when the required environment variables are missing —
/// the test is `#[ignore]`d, so it only runs when explicitly requested.
#[tokio::test]
#[ignore = "needs a real ONNX model + golden file; see testdata/reference_embeddings/README.md"]
async fn gpu_golden_cosine_at_least_0_999() {
    let model_path = std::env::var("INFERSTREAM_ORT_MODEL")
        .expect("set INFERSTREAM_ORT_MODEL to the .onnx model path");
    let golden_path = std::env::var("INFERSTREAM_ORT_GOLDEN")
        .expect("set INFERSTREAM_ORT_GOLDEN to the golden JSON path");
    let device = match std::env::var("INFERSTREAM_ORT_DEVICE").as_deref() {
        Ok("cuda") => OrtDevice::Cuda,
        Ok("tensorrt") => OrtDevice::TensorRt,
        _ => OrtDevice::Cpu,
    };

    let raw = std::fs::read_to_string(&golden_path).expect("golden file readable");
    let golden: serde_json::Value = serde_json::from_str(&raw).expect("golden parses");
    let text = golden["text"].as_str().expect("golden.text").to_string();
    let pooling = match golden["pooling"].as_str() {
        Some("cls") => Pooling::Cls,
        _ => Pooling::Mean,
    };
    let normalize = golden["normalize"].as_bool().unwrap_or(true);
    // Truncation length the golden was generated with (e.g. the serving
    // config's max_seq_len). Absent = engine default.
    let max_seq_len = golden["max_seq_len"].as_u64().map(|v| v as usize);
    let expected: Vec<f32> = golden["vector"]
        .as_array()
        .expect("golden.vector")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let backend = OrtBackend::new(OrtConfig {
        model_path,
        tokenizer_path: std::env::var("INFERSTREAM_ORT_TOKENIZER").ok(),
        device,
        max_seq_len,
        pooling,
        normalize: Some(normalize),
    })
    .expect("engine loads");

    let request = ModelInferRequest {
        model_name: golden["model"].as_str().unwrap_or("golden").to_string(),
        id: "golden-1".into(),
        inputs: vec![InferInputTensor {
            name: "text".into(),
            datatype: "BYTES".into(),
            shape: vec![1],
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_input_contents: vec![pack_bytes(&[text.as_bytes()])],
        ..Default::default()
    };
    let response = backend.infer(request).await.expect("inference succeeds");
    let vector = unpack_fp32(&response.raw_output_contents[0]).expect("FP32 output");

    assert_eq!(vector.len(), expected.len(), "embedding dim");
    let similarity = cosine(&vector, &expected);
    assert!(
        similarity >= 0.999,
        "cosine similarity {similarity} < 0.999 against {golden_path}"
    );
}
