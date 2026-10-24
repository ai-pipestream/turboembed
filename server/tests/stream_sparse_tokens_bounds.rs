//! Bounds, option-validation, and mid-document cancellation regression tests for
//! `ANALYZE_OUTPUT_SPARSE_TERMS` and `ANALYZE_OUTPUT_TOKEN_VECTORS` (`#917`).

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::*;
use serde_json::{Value, json};
use turbo::{TURBO_NORMALIZE_NONE, TURBO_POOLING_CLS, TURBO_TRUNCATE_LEFT};
use turbo_analysis::{Ix, Layer};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{
    AnalyzeDocument, AnalyzeOptions, AnalyzeOutput, AnalyzeStreamRequest, OffsetMode as ProtoOffsetMode,
};
use turbo_kserve::results::Wire;
use turbo_kserve::results::tokens::TokenVectors;
use turbo_kserve::{Serve, Server};

struct TempBundle(PathBuf);

impl Drop for TempBundle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn matryoshka_bundle(tag: &str) -> TempBundle {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("turbo-kserve-917-bounds-{}-{tag}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let src_dir = tiny();
    for f in ["tokenizer.json", "weights/model.safetensors", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(src_dir.join(f), dir.join(f)).unwrap();
    }
    let mut m: Value = serde_json::from_slice(&fs::read(src_dir.join("manifest.json")).unwrap()).unwrap();
    m["embed"]["output_dims"] = json!([16, 32]);
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    TempBundle(dir)
}

async fn serve_models(models: Vec<(&str, &Path)>) -> (Server, Analysis) {
    let cpu_idx = cpu();
    let configs = models
        .into_iter()
        .map(|(name, path)| ModelConfig {
            bundle: path.to_str().unwrap().into(),
            name: Some(name.into()),
            device: Device::Index(cpu_idx),
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            sessions: 1,
        })
        .collect();
    let server = Server::start("127.0.0.1:0".parse().unwrap(), configs, MAX, Serve::default()).await.unwrap();
    server.load().await.unwrap();
    let client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);
    (server, client)
}

#[allow(clippy::too_many_arguments)]
async fn send_typed(
    stream: &mut Stream,
    id: &str,
    model: &str,
    text: &[u8],
    output: AnalyzeOutput,
    offsets: ProtoOffsetMode,
    options: Option<AnalyzeOptions>,
) {
    stream
        .tx
        .send(AnalyzeStreamRequest {
            kind: Some(Kind::Document(AnalyzeDocument {
                id: id.into(),
                model: model.into(),
                text: text.to_vec(),
                options,
                output: output as i32,
                offsets: offsets as i32,
            })),
        })
        .await
        .unwrap();
    stream.sent.push(id.into());
}

#[tokio::test]
async fn stream_token_vector_and_sparse_options_are_validated_and_matryoshka_dim_is_honored() {
    let mat = matryoshka_bundle("mat");
    let tiny_path = tiny();
    let (_s, mut c) = serve_models(vec![("base", &tiny_path), ("mat", &mat.0)]).await;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;

    // 1. output_dim > model.dim -> TURBO_E_INVALID_ARGUMENT (256), field 6.
    send_typed(
        &mut stream,
        "dim-over",
        "base",
        b"hello world",
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        Some(AnalyzeOptions { output_dim: 999, ..Default::default() }),
    )
    .await;
    // 2. output_dim not in output_dims -> TURBO_E_UNSUPPORTED_OPTION (513), field 6.
    send_typed(
        &mut stream,
        "dim-unsup",
        "base",
        b"hello world",
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        Some(AnalyzeOptions { output_dim: 16, ..Default::default() }),
    )
    .await;
    // 3. pooling != MEAN -> TURBO_E_UNSUPPORTED_OPTION (513), field 5.
    send_typed(
        &mut stream,
        "pool-cls",
        "base",
        b"hello world",
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        Some(AnalyzeOptions { pooling: TURBO_POOLING_CLS, ..Default::default() }),
    )
    .await;
    // 4. truncate != NONE -> TURBO_E_UNSUPPORTED_OPTION (513), field 1.
    send_typed(
        &mut stream,
        "trunc-left",
        "base",
        b"hello world",
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        Some(AnalyzeOptions { truncate: TURBO_TRUNCATE_LEFT, ..Default::default() }),
    )
    .await;
    // 5. Matryoshka output_dim = 16 honored on bundle declaring output_dims = [16, 32].
    send_typed(
        &mut stream,
        "mat-16",
        "mat",
        b"hello world",
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        Some(AnalyzeOptions { output_dim: 16, ..Default::default() }),
    )
    .await;
    // 6. normalize = TURBO_NORMALIZE_NONE honored on TokenVectors (unnormalized).
    send_typed(
        &mut stream,
        "raw-norm",
        "mat",
        b"hello world",
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        Some(AnalyzeOptions { output_dim: 16, normalize: TURBO_NORMALIZE_NONE, ..Default::default() }),
    )
    .await;

    let got = by_id(stream.half_close()).await;
    for (k, want_code, want_field) in
        [("dim-over", 256, 6), ("dim-unsup", 513, 6), ("pool-cls", 513, 5), ("trunc-left", 513, 1)]
    {
        match got.get(k).unwrap() {
            Outcome::Failure(f) => {
                assert_eq!(f.turbo_code, want_code, "{k}: {f:?}");
                assert_eq!(f.field, want_field, "{k}: {f:?}");
            }
            other => panic!("{k}: {other:?}"),
        }
    }

    let mat_msg = match got.get("mat-16").unwrap() {
        Outcome::TokenVectors(m) => m,
        other => panic!("mat-16: {other:?}"),
    };
    let mat_vecs = TokenVectors::from_msg(mat_msg).unwrap().vectors().unwrap();
    assert_eq!(mat_vecs.dim, 16);
    assert_eq!(mat_vecs.len(), 2);
    let norm0: f32 = mat_vecs.row(Ix::new(0)).iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm0 - 1.0).abs() <= 1e-5, "L2-normalized 16-dim token vector has unit norm, got {norm0}");

    let raw_msg = match got.get("raw-norm").unwrap() {
        Outcome::TokenVectors(m) => m,
        other => panic!("raw-norm: {other:?}"),
    };
    let raw_vecs = TokenVectors::from_msg(raw_msg).unwrap().vectors().unwrap();
    assert_eq!(raw_vecs.dim, 16);
    assert_ne!(raw_vecs.data, mat_vecs.data, "unnormalized token vectors differ from L2-normalized vectors");
}

#[tokio::test]
async fn stream_token_vectors_refuse_over_max_bytes_before_running_encoder() {
    let tiny_path = tiny();
    let (s, mut c) = serve_models(vec![("base", &tiny_path)]).await;
    let before = s.stream_stats().runs;

    // 262,145 tokens * 32 dims * 4 bytes = 33,554,560 bytes > MAX_TOKEN_VECTOR_BYTES (32 MiB),
    // while document text is ~1.57 MB (well within DEFAULT_MAX_MESSAGE_BYTES = 16 MiB).
    let oversized = "hello ".repeat(262_145);
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    send_typed(
        &mut stream,
        "too-many-bytes",
        "base",
        oversized.as_bytes(),
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        None,
    )
    .await;
    let got = by_id(stream.half_close()).await;
    match got.get("too-many-bytes").unwrap() {
        Outcome::Failure(f) => {
            assert_eq!(f.turbo_code, 771, "expected TURBO_E_CAPACITY (771), got {f:?}");
            assert!(f.message.contains("max_token_vector_bytes"), "message: {}", f.message);
        }
        other => panic!("too-many-bytes: {other:?}"),
    }
    assert_eq!(s.stream_stats().runs, before, "oversized token-vector request must not execute any encoder runs");
}

#[tokio::test]
async fn stream_cancellation_aborts_multi_window_token_vector_execution_promptly() {
    let tiny_path = tiny();
    let (s, mut c) = serve_models(vec![("base", &tiny_path)]).await;

    // 80,000 tokens with max_tokens=6 (window width = 4 tokens) plans 20,000 windows (~312 slot flushes).
    let doc_text = "hello ".repeat(80_000);
    let before = s.stream_stats().runs;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    send_typed(
        &mut stream,
        "cancel-me",
        "base",
        doc_text.as_bytes(),
        AnalyzeOutput::TokenVectors,
        ProtoOffsetMode::Utf8,
        Some(AnalyzeOptions { max_tokens: 6, ..Default::default() }),
    )
    .await;

    // Wait until the worker begins executing windows, then drop the client stream.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while s.stream_stats().runs == before && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    drop(stream);

    // Wait briefly for the worker to observe `dead` and release the slot.
    tokio::time::sleep(Duration::from_millis(80)).await;
    let after_cancel = s.stream_stats().runs;
    tokio::time::sleep(Duration::from_millis(80)).await;
    let settled = s.stream_stats().runs;
    assert_eq!(settled, after_cancel, "encoder runs must stop growing once the stream is cancelled");
    assert!(
        settled - before < 100,
        "cancelled 312-flush document should abort well before finishing all windows (ran {})",
        settled - before
    );
}
