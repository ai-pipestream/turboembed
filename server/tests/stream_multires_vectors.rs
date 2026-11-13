//! Integration tests for `#2344`: `ANALYZE_OUTPUT_MULTIRES_VECTORS` and
//! `AnalysisService::RoundTripMultiResVectors` across UTF-8, UTF-16, and
//! Unicode code-point offset modes.

mod common;

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use serde_json::{Value, json};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{
    AnalyzeDocument, AnalyzeOptions, AnalyzeOutput, AnalyzeStreamRequest, OffsetMode as ProtoOffsetMode,
};
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
    let dir = std::env::temp_dir().join(format!("turbo-kserve-2344-{}-{tag}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let src_dir = tiny();
    for f in ["tokenizer.json", "reference/reference.safetensors", "weights/model.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(src_dir.join(f), dir.join(f)).unwrap();
    }
    let mut m: Value = serde_json::from_slice(&fs::read(src_dir.join("manifest.json")).unwrap()).unwrap();
    m["embed"]["output_dims"] = json!([8, 16, 32]);
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    TempBundle(dir)
}

#[tokio::test]
async fn stream_multires_vectors_single_pass_and_round_trip() {
    let bundle = matryoshka_bundle("stream");
    let models = vec![ModelConfig {
        name: Some("mrl".into()),
        bundle: bundle.0.to_str().unwrap().into(),
        device: Device::Index(cpu()),
        sessions: 1,
        precision: 0,
        max_batch: 8,
        max_seq: 64,
    }];
    let server = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve::default()).await.unwrap();
    server.load().await.unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);

    let opts = AnalyzeOptions {
        matryoshka_dims: vec![16, 8],
        emit_sentence_vectors: true,
        emit_row_sha256: true,
        turboquant_bits: 4,
        turboquant_calibrate: false,
        ..Default::default()
    };

    let text = "Caf\u{e9} search\u{1f680} works fast. Hybrid HNSW prunes shards.";
    let mut stream = open_stream(&mut client, 0, 8, 50).await;
    for (id, mode) in [
        ("utf8", ProtoOffsetMode::Utf8 as i32),
        ("utf16", ProtoOffsetMode::Utf16 as i32),
        ("cp", ProtoOffsetMode::CodePoint as i32),
    ] {
        stream
            .tx
            .send(AnalyzeStreamRequest {
                kind: Some(Kind::Document(AnalyzeDocument {
                    id: id.into(),
                    model: "mrl".into(),
                    text: text.as_bytes().to_vec(),
                    options: Some(opts.clone()),
                    output: AnalyzeOutput::MultiresVectors as i32,
                    offsets: mode,
                    binding: None,
                    vocabulary: None,
                })),
            })
            .await
            .unwrap();
        stream.sent.push(id.into());
    }
    let answers = by_id(stream.half_close()).await;
    assert_eq!(answers.len(), 3);

    for mode_id in ["utf8", "utf16", "cp"] {
        let Outcome::MultiresVectors(res) = answers[mode_id].clone() else {
            panic!("expected MultiresVectors for {mode_id}: {:?}", answers[mode_id]);
        };
        let full = res.full.as_ref().unwrap();
        assert_eq!((full.dim, full.rows), (32, 1));
        assert_eq!(full.data.len(), 32 * 4);
        assert_eq!(full.inv_norm.len(), 1);
        assert!((full.inv_norm[0] - 1.0).abs() < 1e-5);
        assert_eq!(full.content_sha256.len(), 32);
        assert_eq!(full.row_sha256.len(), 32);
        assert_eq!(full.turboquant_bits, 4);
        assert!(!full.turboquant_compact.is_empty());
        assert_eq!(full.turboquant_rows, vec![0]);

        assert_eq!(res.prefixes.len(), 2);
        assert_eq!(res.prefixes[0].dim, 8);
        assert_eq!(res.prefixes[1].dim, 16);

        let sents = res.sentences.as_ref().unwrap();
        assert_eq!(sents.dim, 32);
        assert_eq!(sents.rows, 2);
        assert_eq!(res.sentence_span_start.len(), 2);
        assert_eq!(res.sentence_span_end.len(), 2);

        // RoundTripMultiResVectors validates every invariant and echoes unchanged.
        let echoed = client.round_trip_multi_res_vectors(res.clone()).await.unwrap().into_inner();
        assert_eq!(echoed, res);

        // Tampering with content_sha256 or inv_norm is refused.
        let mut bad_sha = res.clone();
        let mut bytes = bad_sha.full.as_ref().unwrap().content_sha256.clone();
        bytes[0] ^= 0xff;
        bad_sha.full.as_mut().unwrap().content_sha256 = bytes;
        assert!(client.round_trip_multi_res_vectors(bad_sha).await.is_err());

        let mut bad_norm = res.clone();
        bad_norm.full.as_mut().unwrap().inv_norm[0] = 2.0;
        assert!(client.round_trip_multi_res_vectors(bad_norm).await.is_err());
    }

    // Undeclared Matryoshka width is refused per-document without breaking the stream.
    let mut stream2 = open_stream(&mut client, 0, 4, 2).await;
    stream2
        .tx
        .send(AnalyzeStreamRequest {
            kind: Some(Kind::Document(AnalyzeDocument {
                id: "bad-mrl".into(),
                model: "mrl".into(),
                text: b"Hello world.".to_vec(),
                options: Some(AnalyzeOptions { matryoshka_dims: vec![12], ..Default::default() }),
                output: AnalyzeOutput::MultiresVectors as i32,
                offsets: 0,
                binding: None,
                vocabulary: None,
            })),
        })
        .await
        .unwrap();
    stream2.sent.push("bad-mrl".into());
    let answers2 = by_id(stream2.half_close()).await;
    assert!(matches!(answers2["bad-mrl"], Outcome::Failure(_)));

    server.stop().await;
}
