//! Live GPU (or CPU plugin) goldens for OpenVINO GenAI TextEmbeddingPipeline.
//!
//! `#[ignore]`d and compiled only with `--features genai`, so default CI
//! never needs OpenVINO or an Intel GPU. Run on krick-1 (Battlemage):
//!
//! ```bash
//! # after scripts/build-intel.sh (openvino-genai) and make fetch-ov-genai ALIASES=minilm
//! INFERSTREAM_OV_MODEL=models/ov/minilm \
//! INFERSTREAM_OV_DEVICE=GPU \
//! INFERSTREAM_OV_GOLDEN=testdata/reference_embeddings/ov_genai_minilm_short.json \
//! cargo test -p inferstream-backend-openvino --features genai -- --ignored gpu_golden
//! ```
//!
//! This cloud VM does **not** run these. See docs/intel-genai-embed.md.

#![cfg(feature = "genai")]

use std::collections::HashMap;

use inferstream_backend::Backend;
use inferstream_backend_openvino::{OpenVinoBackend, OpenVinoConfig, OvDevice, Pooling};
use inferstream_protocol::inference::model_infer_request::InferInputTensor;
use inferstream_protocol::inference::ModelInferRequest;
use inferstream_protocol::tensor::{pack_bytes, unpack_fp32};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Compare the live GenAI pipeline against a stored golden.
#[tokio::test]
#[ignore = "needs OpenVINO GenAI + an OV-format model dir + golden; see docs/intel-genai-embed.md"]
async fn gpu_golden_cosine_at_least_0_999() {
    let models_path = std::env::var("INFERSTREAM_OV_MODEL")
        .expect("set INFERSTREAM_OV_MODEL to the GenAI model directory");
    let golden_path = std::env::var("INFERSTREAM_OV_GOLDEN")
        .expect("set INFERSTREAM_OV_GOLDEN to the golden JSON path");
    let device = match std::env::var("INFERSTREAM_OV_DEVICE")
        .unwrap_or_else(|_| "GPU".into())
        .to_ascii_uppercase()
        .as_str()
    {
        "CPU" => OvDevice::Cpu,
        "NPU" => OvDevice::Npu,
        "AUTO" => OvDevice::Auto,
        _ => OvDevice::Gpu,
    };

    let raw = std::fs::read_to_string(&golden_path).expect("golden file readable");
    let golden: serde_json::Value = serde_json::from_str(&raw).expect("golden parses");
    let text = golden["text"].as_str().expect("golden.text").to_string();
    let pooling = match golden["pooling"].as_str() {
        Some("cls") => Pooling::Cls,
        Some("last") | Some("last_token") => Pooling::Last,
        _ => Pooling::Mean,
    };
    let normalize = golden["normalize"].as_bool().unwrap_or(true);
    let max_seq_len = golden["max_seq_len"].as_u64().map(|v| v as usize);
    let expected: Vec<f32> = golden["vector"]
        .as_array()
        .expect("golden.vector")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let backend = OpenVinoBackend::new(OpenVinoConfig {
        models_path,
        device: Some(device),
        pooling,
        normalize: Some(normalize),
        max_seq_len,
    })
    .expect("GenAI pipeline loads");

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
