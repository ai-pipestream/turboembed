//! Live MLX tests — require a macOS host with the venv from
//! `scripts/setup-mlx.sh` (Metal + mlx + mlx-embeddings). Kept out of the
//! default test run twice over: behind the `mlx-live` feature AND
//! `#[ignore]`, same policy as the ORT GPU goldens.
//!
//! Run from the repo root (the bridge paths are repo-relative):
//!
//! ```bash
//! cargo test -p inferstream-backend-apple --features mlx-live -- --ignored
//! ```
#![cfg(feature = "mlx-live")]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use inferstream_backend::Backend;
use inferstream_backend_apple::{MlxBackend, MlxConfig, MlxWorker, MlxWorkerConfig};
use inferstream_protocol::inference::model_infer_request::InferInputTensor;
use inferstream_protocol::inference::ModelInferRequest;
use inferstream_protocol::tensor::{pack_bytes, unpack_fp32, DataType};

const MODEL: &str = "mlx-community/all-MiniLM-L6-v2-4bit";

fn repo_root() -> PathBuf {
    // crates/backend-apple -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root resolves")
}

fn live_worker() -> Arc<MlxWorker> {
    let root = repo_root();
    Arc::new(MlxWorker::new(MlxWorkerConfig {
        python: root.join(".venv/bin/python"),
        script: root.join("python/mlx_bridge.py"),
    }))
}

fn live_backend(worker: Arc<MlxWorker>) -> MlxBackend {
    MlxBackend::new(
        MlxConfig {
            model: MODEL.to_string(),
            ..Default::default()
        },
        worker,
    )
    .expect("valid config")
}

fn text_request(texts: &[&str]) -> ModelInferRequest {
    let bytes: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();
    ModelInferRequest {
        model_name: "minilm-l6-v2".into(),
        id: "live-1".into(),
        inputs: vec![InferInputTensor {
            name: "text".into(),
            datatype: DataType::Bytes.as_oip().into(),
            shape: vec![texts.len() as i64],
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_input_contents: vec![pack_bytes(&bytes)],
        ..Default::default()
    }
}

#[tokio::test]
#[ignore = "needs macOS + .venv from scripts/setup-mlx.sh (Metal)"]
async fn ping_reports_metal_device() {
    let worker = live_worker();
    let result = worker
        .call(serde_json::json!({"op": "ping"}))
        .await
        .expect("bridge pings");
    assert!(result["matmul_ok"].as_bool().unwrap_or(false));
    let device = result["device"].as_str().unwrap_or_default();
    assert!(
        device.contains("gpu"),
        "expected Metal GPU device, got {device:?}"
    );
}

#[tokio::test]
#[ignore = "needs macOS + .venv from scripts/setup-mlx.sh (Metal); downloads MiniLM on first run"]
async fn embed_minilm_batch_shapes_and_normalization() {
    let worker = live_worker();
    let backend = live_backend(worker);
    assert!(backend.model_ready("minilm-l6-v2", "").await);

    let response = backend
        .infer(text_request(&["hello world", "grpc inference on metal"]))
        .await
        .expect("live embed");
    let output = &response.outputs[0];
    assert_eq!(output.name, "embedding");
    assert_eq!(output.shape, vec![2, 384], "MiniLM-L6 is 384-d");
    let values = unpack_fp32(&response.raw_output_contents[0]).unwrap();
    assert_eq!(values.len(), 2 * 384);

    // normalize defaults to true — every row should be unit-length.
    for row in values.chunks_exact(384) {
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "row norm {norm} not ~1.0");
    }

    // Determinism: the same text embeds to the same vector (hot cache path).
    let a = backend.infer(text_request(&["hello world"])).await.unwrap();
    let b = backend.infer(text_request(&["hello world"])).await.unwrap();
    assert_eq!(a.raw_output_contents, b.raw_output_contents);

    // Metadata now reports the observed dimension.
    let metadata = backend.model_metadata("minilm-l6-v2", "").await.unwrap();
    assert_eq!(metadata.outputs[0].shape, vec![384]);
}

#[tokio::test]
#[ignore = "needs macOS + .venv; also needs mlx-lm and downloads Qwen2.5-0.5B (~280 MB)"]
async fn stream_generate_small_lm() {
    use futures::StreamExt;
    use inferstream_protocol::inference::infer_parameter::ParameterChoice;
    use inferstream_protocol::inference::InferParameter;

    let worker = live_worker();
    let backend = MlxBackend::new(
        MlxConfig {
            model: "mlx-community/Qwen2.5-0.5B-Instruct-4bit".to_string(),
            ..Default::default()
        },
        worker,
    )
    .unwrap();

    let mut request = text_request(&["Reply with one short sentence: what is MLX?"]);
    request.parameters.insert(
        "max_tokens".into(),
        InferParameter {
            parameter_choice: Some(ParameterChoice::Int64Param(24)),
        },
    );

    let mut stream = backend.infer_stream(request).await.expect("stream opens");
    let mut tokens = 0usize;
    let mut saw_final = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("chunk ok");
        assert_eq!(chunk.id, "live-1");
        let is_final = matches!(
            chunk
                .parameters
                .get("final")
                .and_then(|p| p.parameter_choice.as_ref()),
            Some(ParameterChoice::BoolParam(true))
        );
        if is_final {
            saw_final = true;
        } else {
            tokens += 1;
        }
    }
    assert!(saw_final, "stream must end with a final chunk");
    assert!(tokens > 0, "expected at least one generated token");
}
