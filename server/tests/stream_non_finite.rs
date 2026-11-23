//! An embedding row is answered only when every value in it is finite,
//! and a row of no tokens says so (issue #2517, F12). The embed paths
//! copied the result bytes to the answer unchecked, so a NaN row passed
//! as a vector, and a static model's text of no tokens answered the zero
//! vector with nothing marking it.
//!
//! The poisoned bundle is the tiny static bundle with one token's table
//! row set to NaN, so any text holding that token pools to NaN.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use serde_json::Value;
use tokio_stream::StreamExt;
use turbo_kserve::config::{Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{
    AnalyzeDocument, AnalyzeOptions, AnalyzeOutput, AnalyzePipelineRequest, AnalyzeQueryRequest, AnalyzeStreamRequest,
    OffsetMode as ProtoOffsetMode, QueryDenseTargetProto,
};
use turbo_kserve::{Serve, Server};

/// The tokenizer's id of "zebra", whose row is poisoned.
const ZEBRA: usize = 29145;
const DIM: usize = 16;
const MODEL: &str = "poisoned-static";
static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A copy of the tiny static bundle whose "zebra" row is NaN, with the
/// manifest's digest of the table brought up to date. Removed on drop.
struct Poisoned(PathBuf);

impl Drop for Poisoned {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let target = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &target);
        } else {
            std::fs::copy(e.path(), target).unwrap();
        }
    }
}

fn poisoned() -> Poisoned {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-static-bundle");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/server-non-finite-tests").join(format!(
        "{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    copy_dir(&src, &dir);
    let table = dir.join("weights/static.safetensors");
    let mut bytes = std::fs::read(&table).unwrap();
    let header = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let row = 8 + header + ZEBRA * DIM * 2;
    for d in 0..DIM {
        // F16 quiet NaN.
        bytes[row + 2 * d..row + 2 * d + 2].copy_from_slice(&0x7e00u16.to_le_bytes());
    }
    std::fs::write(&table, &bytes).unwrap();
    let manifest = dir.join("manifest.json");
    let mut m: Value = serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    // A prefix tier, so multires vectors and AnalyzeQuery have one to ask for.
    m["embed"]["output_dims"] = serde_json::json!([8, 16]);
    for f in m["files"].as_array_mut().unwrap() {
        if f["path"] == "weights/static.safetensors" {
            f["sha256"] = Value::String(turbo_sha256::sha256(&bytes).to_hex());
        }
    }
    std::fs::write(&manifest, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    Poisoned(dir)
}

async fn serve(bundle: &Path) -> (Server, Analysis, Kserve) {
    let model = ModelConfig {
        bundle: bundle.to_str().unwrap().to_string(),
        name: Some(MODEL.into()),
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let s = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![model],
        turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES,
        Serve::default(),
    )
    .await
    .unwrap();
    s.load().await.unwrap();
    let a = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let k = kserve(&s).await;
    (s, a, k)
}

#[tokio::test]
async fn a_non_finite_row_fails_its_document_alone() {
    let bundle = poisoned();
    let (s, mut a, _) = serve(&bundle.0).await;
    let mut stream = open_stream(&mut a, 0, 0, 0).await;
    stream.send("hello", MODEL, "hello there", None).await;
    stream.send("zebra", MODEL, "a zebra", None).await;
    let got = by_id(stream.half_close()).await;
    let Outcome::Vectors(v) = &got["hello"] else { panic!("hello: {:?}", got["hello"]) };
    assert_eq!(v.data.len(), DIM * 4);
    let Outcome::Failure(f) = &got["zebra"] else { panic!("a NaN row answered as a vector: {:?}", got["zebra"]) };
    assert_eq!(f.turbo_code, turbo_kserve::api::TURBO_E_INTERNAL, "{f:?}");
    assert!(f.message.contains("not finite"), "{f:?}");
    s.stop().await;
}

#[tokio::test]
async fn model_infer_refuses_a_non_finite_row() {
    let bundle = poisoned();
    let (s, _, mut k) = serve(&bundle.0).await;
    assert_eq!(embed_shape(&mut k, MODEL, &[b"hello there"]).await, [1, DIM as i64]);
    let e = k.model_infer(texts_req(MODEL, &[b"hello there", b"a zebra"])).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::Internal, "{e:?}");
    assert!(e.message().contains("row 1") && e.message().contains("not finite"), "{e:?}");
    s.stop().await;
}

#[tokio::test]
async fn a_static_text_of_no_tokens_says_it_kept_none() {
    let bundle = poisoned();
    let (s, mut a, _) = serve(&bundle.0).await;
    let mut stream = open_stream(&mut a, 0, 0, 0).await;
    stream.send("empty", MODEL, "", None).await;
    stream.send("hello", MODEL, "hello there", None).await;
    let got = by_id(stream.half_close()).await;
    let Outcome::Vectors(v) = &got["empty"] else { panic!("empty: {:?}", got["empty"]) };
    let t = v.tokens.expect("the row says what was tokenized");
    assert_eq!((t.kept, t.seen, t.truncated), (0, Some(0), false), "{t:?}");
    assert!(v.data.iter().all(|&b| b == 0), "a row of no tokens is the zero vector");
    let Outcome::Vectors(v) = &got["hello"] else { panic!("hello: {:?}", got["hello"]) };
    assert!(v.tokens.unwrap().kept > 0);
    s.stop().await;
}

/// Every float embedding output fails the poisoned document with
/// TURBO_E_INTERNAL, not only pooled vectors: a broken model is the
/// server's fault, never the request's (TURBO_E_INVALID_ARGUMENT).
#[tokio::test]
async fn token_and_multires_vectors_fail_a_non_finite_document() {
    let bundle = poisoned();
    let (s, mut a, _) = serve(&bundle.0).await;
    for output in [AnalyzeOutput::TokenVectors, AnalyzeOutput::MultiresVectors] {
        let mut stream = open_stream(&mut a, 0, 0, 0).await;
        // Multires asks for a prefix tier too; token vectors take no options.
        let opts = (output == AnalyzeOutput::MultiresVectors)
            .then(|| AnalyzeOptions { matryoshka_dims: vec![8], ..Default::default() });
        for (id, text) in [("hello", "hello there"), ("zebra", "a zebra")] {
            let doc = AnalyzeDocument {
                id: id.into(),
                model: MODEL.into(),
                text: text.as_bytes().to_vec(),
                options: opts.clone(),
                output: output as i32,
                offsets: ProtoOffsetMode::Utf8 as i32,
                binding: None,
                vocabulary: None,
            };
            stream.tx.send(AnalyzeStreamRequest { kind: Some(Kind::Document(doc)) }).await.unwrap();
            stream.sent.push(id.into());
        }
        let got = by_id(stream.half_close()).await;
        assert!(!matches!(got["hello"], Outcome::Failure(_)), "{output:?} hello: {:?}", got["hello"]);
        let Outcome::Failure(f) = &got["zebra"] else { panic!("{output:?}: a NaN row answered: {:?}", got["zebra"]) };
        assert_eq!(f.turbo_code, turbo_kserve::api::TURBO_E_INTERNAL, "{output:?}: {f:?}");
        assert!(f.message.contains("not finite"), "{output:?}: {f:?}");
    }
    s.stop().await;
}

#[tokio::test]
async fn analyze_query_refuses_a_non_finite_dense_vector() {
    let bundle = poisoned();
    let (s, mut a, _) = serve(&bundle.0).await;
    let req = |query: &str| AnalyzeQueryRequest {
        query: query.into(),
        default_language: "en".into(),
        dense_targets: vec![QueryDenseTargetProto {
            target_id: "dense".into(),
            model: MODEL.into(),
            matryoshka_dims: vec![8],
            ..Default::default()
        }],
        ..Default::default()
    };
    let ok = a.analyze_query(req("hello there")).await.unwrap().into_inner();
    assert!(!ok.dense_results[0].full.as_ref().unwrap().data.is_empty());
    let e = a.analyze_query(req("a zebra")).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::Internal, "{e:?}");
    assert!(e.message().contains("not finite"), "{e:?}");
    s.stop().await;
}

/// The progressive path reports a broken chunk vector as its step's
/// failure with TURBO_E_INTERNAL; the graph it completes holds no such
/// vector layer.
#[tokio::test]
async fn a_progressive_document_with_a_non_finite_chunk_vector_fails_its_step() {
    let bundle = poisoned();
    let (s, mut a, _) = serve(&bundle.0).await;
    let req = AnalyzePipelineRequest {
        doc_id: "zebra".into(),
        text: "a zebra".into(),
        offsets: ProtoOffsetMode::Utf8 as i32,
        profile: "default-v1".into(),
        ..Default::default()
    };
    let mut events = a.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut failures = Vec::new();
    while let Some(item) = events.next().await {
        if let Some(Event::StepFailed(f)) = item.unwrap().event {
            failures.push(f);
        }
    }
    let f = failures.iter().find(|f| f.step_id == "turbo:chunk-vectors").expect("a NaN chunk vector passed");
    assert_eq!(f.failure.as_ref().unwrap().turbo_code, turbo_kserve::api::TURBO_E_INTERNAL, "{f:?}");
    s.stop().await;
}
