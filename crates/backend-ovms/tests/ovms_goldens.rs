//! GPU reference-embedding (golden) tests for the live OVMS path (krick-1).
//!
//! `#[ignore]`d so default CI never needs a running OVMS. Run on the OVMS
//! host against the server's gRPC endpoint (the Docker bridge IP when only
//! REST is published to the host — see `config/intel.toml`):
//!
//! ```bash
//! INFERSTREAM_OVMS_ENDPOINT=http://172.22.0.2:8000 \
//! cargo test -p inferstream-backend-ovms -- --ignored ovms_golden
//! ```
//!
//! Every `testdata/reference_embeddings/ovms_*.json` golden is embedded
//! through [`OvmsBackend`] using the façade Embed convention (BYTES tensor
//! `"text"`), exercising the tensor-name adaptation to the pipeline's
//! `"strings"` / `"sentence_embedding"` names. Cosine must stay ≥ 0.999 —
//! exact equality is deliberately not required because Intel GPU execution
//! may differ in the low-order bits across driver/OVMS versions.
//!
//! Golden schema and regeneration instructions:
//! `testdata/reference_embeddings/README.md`.

use std::collections::HashMap;
use std::path::PathBuf;

use inferstream_backend::Backend;
use inferstream_backend_ovms::OvmsBackend;
use inferstream_protocol::inference::model_infer_request::InferInputTensor;
use inferstream_protocol::inference::ModelInferRequest;
use inferstream_protocol::tensor::{pack_bytes, unpack_fp32};

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/reference_embeddings")
}

struct Golden {
    name: String,
    model: String,
    text: String,
    dim: usize,
    l2: f32,
    vector: Vec<f32>,
}

fn load_ovms_goldens() -> Vec<Golden> {
    let mut goldens = Vec::new();
    for entry in std::fs::read_dir(goldens_dir()).expect("testdata/reference_embeddings exists") {
        let path = entry.unwrap().path();
        let file_name = path.file_name().unwrap().to_string_lossy().to_string();
        if !file_name.starts_with("ovms_") || path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let raw = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(value["backend"], "ovms", "{file_name}");
        goldens.push(Golden {
            name: file_name,
            model: value["model"].as_str().unwrap().to_string(),
            text: value["text"].as_str().unwrap().to_string(),
            dim: value["dim"].as_u64().unwrap() as usize,
            l2: value["l2"].as_f64().unwrap() as f32,
            vector: value["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect(),
        });
    }
    assert!(
        goldens.len() >= 10,
        "expected the full OVMS golden set (5 prompts x 2 pipelines), found {}",
        goldens.len()
    );
    goldens
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Compare the live OVMS pipelines against every stored `ovms_*` golden.
#[tokio::test]
#[ignore = "needs a running OVMS; set INFERSTREAM_OVMS_ENDPOINT — see testdata/reference_embeddings/README.md"]
async fn ovms_golden_cosine_at_least_0_999() {
    let endpoint = std::env::var("INFERSTREAM_OVMS_ENDPOINT")
        .expect("set INFERSTREAM_OVMS_ENDPOINT to the OVMS gRPC endpoint (bridge IP:port)");
    let backend = OvmsBackend::new(endpoint).expect("valid endpoint");

    for golden in load_ovms_goldens() {
        // Façade Embed convention on purpose: the same shape the Embed RPC
        // forwards, so the pipeline-name adaptation is on the tested path.
        let request = ModelInferRequest {
            model_name: golden.model.clone(),
            id: format!("golden-{}", golden.name),
            inputs: vec![InferInputTensor {
                name: "text".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                parameters: HashMap::new(),
                contents: None,
            }],
            raw_input_contents: vec![pack_bytes(&[golden.text.as_bytes()])],
            ..Default::default()
        };
        let response = backend.infer(request).await.expect("inference succeeds");

        let (index, output) = response
            .outputs
            .iter()
            .enumerate()
            .find(|(_, o)| o.name == "embedding")
            .unwrap_or_else(|| panic!("{}: no adapted \"embedding\" output", golden.name));
        assert_eq!(output.datatype, "FP32", "{}", golden.name);
        let vector = unpack_fp32(&response.raw_output_contents[index]).expect("FP32 output");

        assert_eq!(vector.len(), golden.dim, "{}: dim", golden.name);
        let similarity = cosine(&vector, &golden.vector);
        assert!(
            similarity >= 0.999,
            "{}: cosine similarity {similarity} < 0.999",
            golden.name
        );
        let l2 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (l2 - golden.l2).abs() <= 1e-3 * golden.l2.max(1.0),
            "{}: L2 {l2} deviates from golden {}",
            golden.name,
            golden.l2
        );
    }
}
