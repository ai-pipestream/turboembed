//! What the analysis result-layer tests share: the served tiny bundle, a
//! real client, the sample in-process results, and the request builders.
//! Each test binary uses its own subset.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;
use tonic::{Code, Request, Status};
use turbo_analysis::{Ix, Key, Layer, ModelRef, Span, View, ViewId};
use turbo_analysis_embed::{Answers, Reranked};
use turbo_analysis_layers::{Chunks, SparseTerms, Subwords, TokenVectorWindows, Vectors};
use turbo_kserve::api::Runtime;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use turbo_kserve::proto::model_infer_request::InferInputTensor;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::*;
use turbo_kserve::proto::{InferTensorContents, ModelInferRequest};
use turbo_kserve::{Serve, Server};

pub type Analysis = AnalysisServiceClient<Channel>;
pub type Kserve = GrpcInferenceServiceClient<Channel>;

/// The inference protocol's client, connected where the server answers.
pub async fn kserve(s: &Server) -> Kserve {
    GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap()
}

/// One ModelInfer of byte texts.
pub fn texts_req(model: &str, texts: &[&[u8]]) -> ModelInferRequest {
    ModelInferRequest {
        model_name: model.into(),
        model_version: String::new(),
        id: String::new(),
        parameters: HashMap::new(),
        inputs: vec![InferInputTensor {
            name: "texts".into(),
            datatype: "BYTES".into(),
            shape: vec![texts.len() as i64],
            parameters: HashMap::new(),
            contents: Some(InferTensorContents {
                bytes_contents: texts.iter().map(|t| t.to_vec()).collect(),
                ..Default::default()
            }),
        }],
        outputs: vec![],
        raw_input_contents: vec![],
    }
}

/// The vectors tensor's shape of one ModelInfer.
pub async fn embed_shape(k: &mut Kserve, model: &str, texts: &[&[u8]]) -> Vec<i64> {
    let out = k.model_infer(texts_req(model, texts)).await.unwrap().into_inner();
    out.outputs[0].shape.clone()
}

pub const NAME: &str = "tiny-bert-bundle";

pub fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle")
}

/// The runtime index of the CPU.
pub fn cpu() -> u32 {
    let rt = Runtime::create().unwrap();
    (0..16).find(|&i| rt.device_info(i).map(|d| d.kind == turbo::TURBO_DEVICE_CPU).unwrap_or(false)).expect("a CPU")
}

fn model(bundle: &Path, name: Option<&str>, sessions: u32) -> ModelConfig {
    ModelConfig {
        bundle: bundle.to_str().unwrap().to_string(),
        name: name.map(String::from),
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions,
    }
}

/// A server whose admin calls are served, for the load and unload tests.
pub async fn serve() -> (Server, Analysis) {
    let s = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![model(&tiny(), None, 2)],
        MAX,
        Serve { admin: true, ..Serve::default() },
    )
    .await
    .unwrap();
    s.load().await.unwrap();
    // The client decodes answers up to the same message budget the server
    // decodes requests with: a refusal can carry an id bounded by the
    // stream's byte cap, which a default 4 MiB decode limit would refuse.
    let c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);
    (s, c)
}

/// The default server: admin calls refused, for the opt-in test.
pub async fn serve_no_admin() -> (Server, Analysis) {
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model(&tiny(), None, 2)], MAX, Serve::default())
        .await
        .unwrap();
    s.load().await.unwrap();
    let c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);
    (s, c)
}

pub fn model_ref() -> ModelRef {
    ModelRef { id: "sentence-transformers/all-MiniLM-L6-v2".into(), revision: "1".into(), sha256: String::new() }
}

/// The gRPC code and turbo-code of a refusal.
pub fn refusal(s: &Status) -> (Code, i32) {
    let md = s.metadata().get("turbo-code").unwrap_or_else(|| panic!("no turbo-code in {s:?}")).to_str().unwrap();
    (s.code(), md.parse().unwrap())
}

// ---- The four sample result kinds ------------------------------------------

pub fn sparse_terms() -> SparseTerms<View> {
    let mut t = SparseTerms::new(model_ref());
    t.push(3, 0.5, Some(Span::new(0, 4)));
    t.push_occurrences(&[Span::new(0, 4)]);
    t.push(9, 1.25, None);
    t.push_occurrences(&[]);
    t.unit();
    t.push(1, 0.75, Some(Span::new(5, 9)));
    t.push_occurrences(&[Span::new(5, 7), Span::new(7, 9)]);
    t.unit();
    t.check().unwrap();
    t
}

pub fn token_vectors() -> (Vectors<Subwords>, TokenVectorWindows) {
    let data: Vec<f32> = (0..12).map(|i| if i / 3 == 1 { 0.0 } else { i as f32 * 0.25 }).collect();
    let v = Vectors::new(3, data, vec![Ix::new(1)], model_ref());
    v.check().unwrap();
    let mut spans = turbo_analysis::Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 10));
    spans.push(Span::new(10, 20));
    let w = TokenVectorWindows { spans };
    w.check().unwrap();
    (v, w)
}

pub fn reranked() -> Reranked<Chunks> {
    let r = Reranked {
        source: Key::named("candidates"),
        items: vec![Ix::new(7), Ix::new(2)],
        scores: vec![9.5, 1.25],
        model: model_ref(),
    };
    r.check().unwrap();
    r
}

pub fn answers() -> Answers {
    let mut spans = turbo_analysis::Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 4));
    spans.push(Span::new(16, 24));
    let out = Answers {
        spans,
        scores: vec![8.0, 4.5],
        probabilities: vec![0.5, 0.3],
        null_score: -2.0,
        null_probability: 0.2,
    };
    out.check().unwrap();
    out
}

// ---- Capability and request helpers -----------------------------------------

/// The bundle's manifest, for what the listing must match.
pub fn manifest() -> Value {
    serde_json::from_slice(&std::fs::read(tiny().join("manifest.json")).unwrap()).unwrap()
}

pub fn the_one_bundle(list: &CapabilitiesResponse) -> &BundleCapabilities {
    assert_eq!(list.bundles.len(), 1, "one bundle is loaded");
    &list.bundles[0]
}

pub fn load_req(name: &str, bundle: &str, sessions: u32, precision: u32) -> LoadBundleRequest {
    LoadBundleRequest {
        name: name.into(),
        bundle: bundle.into(),
        device: cpu(),
        sessions,
        precision,
        max_batch: 0,
        max_seq: 0,
    }
}

/// The reference cases' texts of the tiny harness bundle: the empty
/// document, ASCII, accents, CJK, emoji and tabs, and two long ones.
/// What the stream must answer byte for byte like the single-document
/// call does.
pub fn harness_texts() -> Vec<String> {
    manifest()["reference"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["text"].as_str().unwrap().to_string())
        .collect()
}

// ---- The streaming bulk analysis call's test client --------------------------

use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{AnalyzeDocument, AnalyzeOptions, AnalyzeStreamHeader, AnalyzeStreamRequest};

/// One stream message: the header, opening the call.
pub fn stream_header(order: i32, max_inflight: u32, batch_linger_ms: u32) -> AnalyzeStreamRequest {
    AnalyzeStreamRequest { kind: Some(Kind::Header(AnalyzeStreamHeader { order, max_inflight, batch_linger_ms })) }
}

/// One stream message: a document.
pub fn stream_doc(id: &str, model: &str, text: &str, options: Option<AnalyzeOptions>) -> AnalyzeStreamRequest {
    AnalyzeStreamRequest {
        kind: Some(Kind::Document(AnalyzeDocument {
            id: id.into(),
            model: model.into(),
            text: text.into(),
            options,
            output: 0,
            offsets: 0,
        })),
    }
}

/// One open AnalyzeStream: the sender half, the response half, the ids
/// in send order.
pub struct Stream {
    pub tx: tokio::sync::mpsc::Sender<AnalyzeStreamRequest>,
    pub responses: tonic::Streaming<turbo_kserve::proto::result::AnalyzeStreamResponse>,
    pub sent: Vec<String>,
}

pub async fn open_stream(c: &mut Analysis, order: i32, max_inflight: u32, batch_linger_ms: u32) -> Stream {
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tx.send(stream_header(order, max_inflight, batch_linger_ms)).await.unwrap();
    let responses = c.analyze_stream(Request::new(ReceiverStream::new(rx))).await.unwrap().into_inner();
    Stream { tx, responses, sent: Vec::new() }
}

impl Stream {
    pub async fn send(&mut self, id: &str, model: &str, text: &str, options: Option<AnalyzeOptions>) {
        self.tx.send(stream_doc(id, model, text, options)).await.unwrap();
        self.sent.push(id.into());
    }

    /// The client is done sending; the answers keep flowing.
    pub fn half_close(self) -> Self {
        drop(self.tx);
        Self { tx: tokio::sync::mpsc::channel(1).0, ..self }
    }

    /// Every document's answer, in arrival order.
    pub async fn collect(mut self) -> Vec<(String, Outcome)> {
        let mut out = Vec::new();
        while let Some(r) = self.responses.message().await.unwrap() {
            out.push((r.id.clone(), r.outcome.unwrap()));
        }
        assert_eq!(out.len(), self.sent.len(), "every document answers exactly once");
        out
    }
}

/// The collected answers keyed by id.
pub async fn by_id(stream: Stream) -> std::collections::HashMap<String, Outcome> {
    stream.collect().await.into_iter().collect()
}

// ---- Rerank and extractive QA test bundle fixtures --------------------------

static BUNDLE_SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub struct BundleDir(pub PathBuf);
impl Drop for BundleDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch_dir(tag: &str) -> BundleDir {
    let id =
        format!("{tag}-{}-{}", std::process::id(), BUNDLE_SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let dir = root.join("target/server-rerank-qa-tests").join(id);
    std::fs::create_dir_all(dir.join("weights")).unwrap();
    std::fs::create_dir_all(dir.join("reference")).unwrap();
    BundleDir(dir)
}

fn roberta_pair_reference_safetensors() -> Vec<u8> {
    let mut header = serde_json::to_vec(&serde_json::json!({
        "ids": {"dtype": "I32", "shape": [1, 4], "data_offsets": [0, 16]},
        "types": {"dtype": "I32", "shape": [1, 4], "data_offsets": [16, 32]},
        "lengths": {"dtype": "I32", "shape": [1], "data_offsets": [32, 36]},
        "scores": {"dtype": "F32", "shape": [1, 1], "data_offsets": [36, 40]},
    }))
    .unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    for v in [101i32, 102, 102, 102, 0, 0, 0, 0, 4] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&0.0f32.to_le_bytes());
    out
}

fn copy_bundle_files(d: &Path, tiny: &Path, reference: &Path, ref_override: Option<Vec<u8>>) -> Vec<Value> {
    let mut files = Vec::new();
    for (src, dest) in [
        (tiny.join("tokenizer.json"), "tokenizer.json"),
        (tiny.join("weights/model.safetensors"), "weights/model.safetensors"),
        (reference.join("head.safetensors"), "weights/head.safetensors"),
    ] {
        let b = std::fs::read(src).unwrap();
        std::fs::write(d.join(dest), &b).unwrap();
        files.push(serde_json::json!({"path": dest, "size": b.len(), "sha256": turbo::bundle::sha256_hex(&b)}));
    }
    let ref_bytes = ref_override.unwrap_or_else(|| std::fs::read(reference.join("reference.safetensors")).unwrap());
    let dest = "reference/reference.safetensors";
    std::fs::write(d.join(dest), &ref_bytes).unwrap();
    files.push(
        serde_json::json!({"path": dest, "size": ref_bytes.len(), "sha256": turbo::bundle::sha256_hex(&ref_bytes)}),
    );
    files
}

pub fn make_rerank_bundle(pair_template: &str) -> BundleDir {
    let d = scratch_dir("rerank");
    let r = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let (tiny, reference) = (r.join("testdata/tiny-bert-bundle"), r.join("testdata/tiny-rerank-reference"));
    let mut m: Value = serde_json::from_slice(&std::fs::read(tiny.join("manifest.json")).unwrap()).unwrap();
    m.as_object_mut().unwrap().remove("embed");
    m["task"] = "TASK_RERANK".into();
    let ref_override = if pair_template == "PAIR_ROBERTA" {
        m["model"]["id"] = "BAAI/bge-reranker-base".into();
        m["architecture"]["family"] = "FAMILY_ROBERTA".into();
        m["architecture"]["position_offset"] = 1.into();
        m["reference"]["cases"] = serde_json::json!([{"text": "", "passage": "", "prompt_role": "PROMPT_NONE"}]);
        Some(roberta_pair_reference_safetensors())
    } else {
        m["model"]["id"] = "cross-encoder/ms-marco-MiniLM-L6-v2".into();
        m["reference"]["cases"] =
            serde_json::from_slice(&std::fs::read(reference.join("cases.json")).unwrap()).unwrap();
        None
    };
    m["rerank"] = serde_json::json!({"max_seq": 32, "max_batch": 4, "query_tokens": 8, "pair_template": pair_template});
    m["classifier"] = serde_json::json!({"kind": "CLASSIFIER_SEQUENCE", "labels": ["LABEL_0"]});
    m["artifacts"][0]["files"] = serde_json::json!(["weights/model.safetensors", "weights/head.safetensors"]);
    let names = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    for (k, v) in [
        ("pooler_weight", "bert.pooler.dense.weight"),
        ("pooler_bias", "bert.pooler.dense.bias"),
        ("classifier_weight", "classifier.weight"),
        ("classifier_bias", "classifier.bias"),
    ] {
        names.insert(k.into(), v.into());
    }
    m["files"] = copy_bundle_files(&d.0, &tiny, &reference, ref_override).into();
    std::fs::write(d.0.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    d
}

pub fn make_qa_bundle() -> BundleDir {
    let d = scratch_dir("qa");
    let r = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let (tiny, reference) = (r.join("testdata/tiny-bert-bundle"), r.join("testdata/tiny-qa-reference"));
    let spec: Value = serde_json::from_slice(&std::fs::read(reference.join("cases.json")).unwrap()).unwrap();
    let mut m: Value = serde_json::from_slice(&std::fs::read(tiny.join("manifest.json")).unwrap()).unwrap();
    m.as_object_mut().unwrap().remove("embed");
    m["model"]["id"] = "turbo/test-tiny-qa".into();
    m["task"] = "TASK_EXTRACTIVE_QA".into();
    m["qa"] = spec["qa"].clone();
    m["classifier"] = serde_json::json!({"kind": "CLASSIFIER_TOKEN", "labels": ["start", "end"]});
    m["artifacts"][0]["files"] = serde_json::json!(["weights/model.safetensors", "weights/head.safetensors"]);
    m["artifacts"][0]["tensor_names"]["classifier_weight"] = "qa_outputs.weight".into();
    m["artifacts"][0]["tensor_names"]["classifier_bias"] = "qa_outputs.bias".into();
    let mut cases = spec["cases"].clone();
    for c in cases.as_array_mut().unwrap() {
        c["prompt_role"] = "PROMPT_NONE".into();
    }
    m["reference"]["cases"] = cases;
    m["reference"]["produced_by"]["tool"] = "transformers".into();
    m["files"] = copy_bundle_files(&d.0, &tiny, &reference, None).into();
    std::fs::write(d.0.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    d
}
