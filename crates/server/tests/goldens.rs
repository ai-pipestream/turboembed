//! Reference-embedding (golden) tests for the deterministic mock backend.
//!
//! Always on in CI: the mock is platform-independent, so any drift in its
//! embedding algorithm — or in the Embed RPC path wrapping it — fails these
//! tests. GPU goldens live in `crates/backend-ort/tests/gpu_goldens.rs`
//! (feature-gated + `#[ignore]`); see `testdata/reference_embeddings/README.md`.

use std::path::PathBuf;

use inferstream_backend_mock::MockBackend;
use inferstream_protocol::extension::inferstream_service_client::InferstreamServiceClient;
use inferstream_protocol::extension::EmbedRequest;
use inferstream_server::config::Config;

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/reference_embeddings")
}

struct Golden {
    name: String,
    text: String,
    dim: usize,
    l2: f32,
    vector: Vec<f32>,
}

fn load_mock_goldens() -> Vec<Golden> {
    let mut goldens = Vec::new();
    for entry in std::fs::read_dir(goldens_dir()).expect("testdata/reference_embeddings exists") {
        let path = entry.unwrap().path();
        let file_name = path.file_name().unwrap().to_string_lossy().to_string();
        if !file_name.starts_with("mock_") || path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let raw = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(value["backend"], "mock", "{file_name}");
        goldens.push(Golden {
            name: file_name,
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
        goldens.len() >= 5,
        "expected the full golden prompt set, found {}",
        goldens.len()
    );
    goldens
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        // Both zero vectors count as identical; anything else as orthogonal.
        return if na == nb { 1.0 } else { 0.0 };
    }
    dot / (na * nb)
}

#[test]
fn mock_backend_matches_goldens() {
    let backend = MockBackend::default();
    for golden in load_mock_goldens() {
        let vector = backend.embed(golden.text.as_bytes());
        assert_eq!(vector.len(), golden.dim, "{}: dim", golden.name);

        let similarity = cosine(&vector, &golden.vector);
        assert!(
            similarity >= 0.999,
            "{}: cosine similarity {similarity} < 0.999",
            golden.name
        );

        let l2 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (l2 - golden.l2).abs() <= 1e-4 * golden.l2.max(1.0),
            "{}: L2 {l2} deviates from golden {}",
            golden.name,
            golden.l2
        );

        // The mock is bit-deterministic, so exact equality must also hold —
        // if this fires while cosine passes, the algorithm changed subtly
        // and the goldens must be regenerated deliberately.
        assert_eq!(vector, golden.vector, "{}: exact values", golden.name);
    }
}

/// The same goldens exercised end-to-end through the `inferstream.v1.Embed`
/// RPC, proving the wrap/unwrap path adds no numeric drift.
#[tokio::test]
async fn embed_rpc_matches_goldens() {
    let config = Config::from_toml(
        r#"
        listen = "127.0.0.1:0"

        [[models]]
        name = "mock-embed"
        backend = "mock"
        "#,
    )
    .unwrap();
    let registry =
        inferstream_server::build_registry(&config, &inferstream_server::mock_factory()).unwrap();
    let (bound_tx, bound_rx) = tokio::sync::oneshot::channel();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let handle = tokio::spawn(inferstream_server::serve(
        config,
        registry,
        bound_tx,
        async move {
            let _ = shutdown_rx.await;
        },
    ));
    let addr = bound_rx.await.unwrap();
    let mut client = InferstreamServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();

    for golden in load_mock_goldens() {
        let response = client
            .embed(EmbedRequest {
                model_name: "mock-embed".into(),
                texts: vec![golden.text.clone()],
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.dim as usize, golden.dim, "{}", golden.name);
        let vector = &response.embeddings[0].values;
        let similarity = cosine(vector, &golden.vector);
        assert!(
            similarity >= 0.999,
            "{}: cosine over gRPC {similarity} < 0.999",
            golden.name
        );
        assert_eq!(vector, &golden.vector, "{}: exact over gRPC", golden.name);
    }

    let _ = shutdown_tx.send(());
    let _ = handle.await;
}
