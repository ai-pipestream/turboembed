//! Live native-MLX tests — macOS host + weights from `cargo xtask fetch --mlx`.
//!
//! ```bash
//! cargo xtask fetch --mlx minilm qwen-0.5b
//! cargo test -p inferstream-backend-apple --features mlx-live -- --ignored
//! ```
#![cfg(feature = "mlx-live")]

use std::collections::HashMap;
use std::sync::Arc;

use inferstream_backend::Backend;
use inferstream_backend_apple::{MlxBackend, MlxConfig, MlxEngine};
use inferstream_protocol::inference::model_infer_request::InferInputTensor;
use inferstream_protocol::inference::ModelInferRequest;
use inferstream_protocol::tensor::{pack_bytes, unpack_fp32, DataType};

fn live_backend(model: &str) -> MlxBackend {
    MlxBackend::new(
        MlxConfig {
            model: model.to_string(),
            ..Default::default()
        },
        Arc::new(MlxEngine::new()),
    )
    .expect("valid config")
}

fn text_request(texts: &[&str]) -> ModelInferRequest {
    let bytes: Vec<&[u8]> = texts.iter().map(|t| t.as_bytes()).collect();
    ModelInferRequest {
        model_name: "minilm".into(),
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
#[ignore = "needs macOS Metal + models/mlx/minilm from cargo xtask fetch --mlx"]
async fn ping_reports_metal_device() {
    let engine = MlxEngine::new();
    let result = tokio::task::spawn_blocking(move || engine.ping())
        .await
        .unwrap()
        .expect("native ping");
    eprintln!(
        "native ping: device={} metal={} mem={} peak={}",
        result.device, result.metal_available, result.active_memory, result.peak_memory
    );
    assert!(result.matmul_ok);
    assert!(
        result.device.contains("gpu") || result.metal_available,
        "expected Metal GPU device, got {:?}",
        result.device
    );
}

#[tokio::test]
#[ignore = "needs macOS Metal + models/mlx/minilm"]
async fn embed_minilm_batch_shapes_and_normalization() {
    let backend = live_backend("models/mlx/minilm");
    assert!(backend.model_ready("minilm", "").await);

    let response = backend
        .infer(text_request(&["hello world", "grpc inference on metal"]))
        .await
        .expect("live embed");
    let output = &response.outputs[0];
    assert_eq!(output.name, "embedding");
    assert_eq!(output.shape, vec![2, 384], "MiniLM-L6 is 384-d");
    let values = unpack_fp32(&response.raw_output_contents[0]).unwrap();
    assert_eq!(values.len(), 2 * 384);
    for row in values.chunks_exact(384) {
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "row norm {norm} not ~1.0");
    }
}

#[tokio::test]
#[ignore = "needs macOS Metal + models/mlx/qwen-0.5b"]
async fn stream_generate_small_lm() {
    use futures::StreamExt;
    use inferstream_protocol::inference::infer_parameter::ParameterChoice;
    use inferstream_protocol::inference::InferParameter;

    let backend = live_backend("models/mlx/qwen-0.5b");
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
    let mut tps = 0.0;
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
            if let Some(ParameterChoice::DoubleParam(v)) = chunk
                .parameters
                .get("decode_tokens_per_second")
                .and_then(|p| p.parameter_choice.as_ref())
            {
                tps = *v;
            }
        } else {
            tokens += 1;
        }
    }
    assert!(saw_final, "stream must end with a final chunk");
    assert!(tokens > 0, "expected at least one generated token");
    eprintln!("engine-side decode tok/s = {tps:.1} ({tokens} tokens)");
    assert!(tps > 10.0, "expected competitive Metal decode, got {tps} t/s");
}
