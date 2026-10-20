//! The analysis result layer over gRPC with a real client against the
//! server in-process on the CPU: every result kind crosses the wire and
//! back unchanged from its in-process layer, the capability listing
//! follows LoadBundle and UnloadBundle, and an over-limit request is
//! refused with a typed status naming the limit. Nothing is mocked.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tonic::transport::Channel;
use tonic::{Code, Status};
use turbo_analysis::{Ix, Key, Layer, ModelRef, Span, View, ViewId};
use turbo_analysis_embed::{Answers, Reranked};
use turbo_analysis_layers::{Chunks, SparseTerms, Subwords, TokenVectorWindows, Vectors};
use turbo_kserve::Server;
use turbo_kserve::api::Runtime;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use turbo_kserve::proto::infer_parameter::ParameterChoice;
use turbo_kserve::proto::model_infer_request::InferInputTensor;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::*;
use turbo_kserve::proto::{InferTensorContents, ModelInferRequest};
use turbo_kserve::results::Wire;
use turbo_kserve::results::{extraction::Extraction, rerank::RerankScores, tokens::TokenVectors};

type Analysis = AnalysisServiceClient<Channel>;

const NAME: &str = "tiny-bert-bundle";

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle")
}

/// The runtime index of the CPU.
fn cpu() -> u32 {
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

async fn serve() -> (Server, Analysis) {
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model(&tiny(), None, 2)], MAX).await.unwrap();
    s.load().await.unwrap();
    let c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

fn model_ref() -> ModelRef {
    ModelRef { id: "sentence-transformers/all-MiniLM-L6-v2".into(), revision: "1".into(), sha256: String::new() }
}

/// The gRPC code and turbo-code of a refusal.
fn refusal(s: &Status) -> (Code, i32) {
    let md = s.metadata().get("turbo-code").unwrap_or_else(|| panic!("no turbo-code in {s:?}")).to_str().unwrap();
    (s.code(), md.parse().unwrap())
}

// ---- The four result kinds over the wire ----------------------------------

fn sparse_terms() -> SparseTerms<View> {
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

fn token_vectors() -> (Vectors<Subwords>, TokenVectorWindows) {
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

fn reranked() -> Reranked<Chunks> {
    let r = Reranked {
        source: Key::named("candidates"),
        items: vec![Ix::new(7), Ix::new(2)],
        scores: vec![9.5, 1.25],
        model: model_ref(),
    };
    r.check().unwrap();
    r
}

fn answers() -> Answers {
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

#[tokio::test]
async fn sparse_terms_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let t = sparse_terms();
    let msg = <SparseTerms<View> as Wire>::to_msg(&t);
    let back = c.round_trip_sparse_terms(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg, "the answer is the request");
    let decoded = SparseTerms::from_msg(&back).unwrap();
    assert_eq!(decoded.model, t.model);
    assert_eq!(decoded.term, t.term);
    assert_eq!(decoded.weight, t.weight);
    assert_eq!(decoded.span, t.span);
    assert_eq!(decoded.occ, t.occ);
    assert_eq!(decoded.occ_at, t.occ_at);
}

#[tokio::test]
async fn token_vectors_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let (v, w) = token_vectors();
    let t = TokenVectors::new(&v, &w);
    let msg = t.to_msg();
    let back = c.round_trip_token_vectors(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = TokenVectors::from_msg(&back).unwrap();
    assert_eq!(decoded.dim, 3);
    assert_eq!(decoded.data, t.data);
    assert_eq!(decoded.empty, [1]);
    assert_eq!(decoded.windows, t.windows);
    assert_eq!(decoded.vectors().unwrap().data, v.data);
    assert_eq!(decoded.windows_layer(4).unwrap().spans.columns(), w.spans.columns());
}

#[tokio::test]
async fn rerank_scores_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let r = reranked();
    let m = RerankScores::new(&r);
    let msg = m.to_msg();
    let back = c.round_trip_rerank_scores(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = RerankScores::from_msg(&back).unwrap();
    assert_eq!(decoded.model, r.model);
    assert_eq!(decoded.source, "turbo:chunks");
    assert_eq!(decoded.items, r.items.iter().map(|i| i.get() as u32).collect::<Vec<_>>());
    assert_eq!(decoded.scores, r.scores);
}

#[tokio::test]
async fn extraction_answers_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let a = answers();
    let x = Extraction::new(&a, model_ref());
    let msg = x.to_msg();
    let back = c.round_trip_extraction(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = Extraction::from_msg(&back).unwrap();
    assert_eq!(decoded.model, x.model);
    assert_eq!(decoded.spans, x.spans);
    assert_eq!(decoded.scores, x.scores);
    assert_eq!(decoded.probabilities, x.probabilities);
    assert_eq!(decoded.answers().unwrap().spans.columns(), a.spans.columns());
}

// ---- Capabilities ----------------------------------------------------------

/// The bundle's manifest, for what the listing must match.
fn manifest() -> Value {
    serde_json::from_slice(&std::fs::read(tiny().join("manifest.json")).unwrap()).unwrap()
}

fn the_one_bundle(list: &CapabilitiesResponse) -> &BundleCapabilities {
    assert_eq!(list.bundles.len(), 1, "one bundle is loaded");
    &list.bundles[0]
}

fn load_req(name: &str, bundle: &str, sessions: u32, precision: u32) -> LoadBundleRequest {
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

#[tokio::test]
async fn the_capability_listing_matches_the_loaded_bundle() {
    let (_s, mut c) = serve().await;
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let b = the_one_bundle(&list);
    let m = manifest();
    assert_eq!(b.name, NAME);
    assert_eq!(b.model.as_ref().unwrap().id, m["model"]["id"]);
    assert_eq!(b.model.as_ref().unwrap().revision, m["model"]["revision"]);
    assert!(!b.model.as_ref().unwrap().sha256.is_empty());
    assert_eq!(b.task, "TASK_EMBED");
    assert_eq!(b.roles, ["PROMPT_QUERY"], "the bundle carries a query prefix only");
    assert_eq!(b.widths, [m["embed"]["dim"].as_u64().unwrap() as u32]);
    assert_eq!(b.context_limit, m["embed"]["max_seq"].as_u64().unwrap() as u32);
    assert_eq!(b.batch_limit, m["embed"]["max_batch"].as_u64().unwrap() as u32);
    assert!(b.schema.is_none(), "no merged task reads an extraction schema yet");
    let limits = list.limits.as_ref().unwrap();
    assert!(limits.max_sparse_entries > 0 && limits.max_sparse_occurrences >= limits.max_sparse_entries);
    assert!(limits.max_token_vector_bytes > 0 && limits.max_rerank_items > 0 && limits.max_answers > 0);
}

#[tokio::test]
async fn the_listing_follows_load_and_unload() {
    let (s, mut c) = serve().await;
    // The same bundle under a second name, as one bundle at two tiers would be.
    let bundle = tiny().to_str().unwrap().to_string();
    let req = load_req("second", &bundle, 1, 0);
    let loaded = c.load_bundle(req).await.unwrap().into_inner();
    let b = loaded.capabilities.as_ref().unwrap();
    assert_eq!(b.name, "second");
    assert_eq!(b.widths, [32]);
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(list.bundles.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), [NAME, "second"]);

    // The new name serves inference too.
    let mut k = GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let texts = ModelInferRequest {
        model_name: "second".into(),
        model_version: String::new(),
        id: String::new(),
        parameters: HashMap::new(),
        inputs: vec![InferInputTensor {
            name: "texts".into(),
            datatype: "BYTES".into(),
            shape: vec![1],
            parameters: HashMap::new(),
            contents: Some(InferTensorContents { bytes_contents: vec![b"hello".to_vec()], ..Default::default() }),
        }],
        outputs: vec![],
        raw_input_contents: vec![],
    };
    let out = k.model_infer(texts.clone()).await.unwrap().into_inner();
    assert_eq!(out.outputs[0].shape, [1, 32]);

    // Unload: the listing shrinks and the name is refused as never served.
    c.unload_bundle(UnloadBundleRequest { name: "second".into() }).await.unwrap();
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(list.bundles.len(), 1);
    let gone = k.model_infer(texts).await.expect_err("gone");
    assert_eq!(refusal(&gone), (Code::NotFound, 1280));
    let gone = c.unload_bundle(UnloadBundleRequest { name: "second".into() }).await.expect_err("not served");
    assert_eq!(refusal(&gone), (Code::NotFound, 1280));

    // The freed name serves again.
    c.load_bundle(load_req("second", &bundle, 1, 0)).await.unwrap();
    c.unload_bundle(UnloadBundleRequest { name: "second".into() }).await.unwrap();

    // A name that is taken, and requests that cannot mean anything, refuse
    // by name; a bundle that cannot load leaves nothing served.
    let e = c.load_bundle(load_req(NAME, &bundle, 1, 0)).await.expect_err("the name is taken");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));
    for bad in [
        load_req("x", "", 1, 0),
        load_req("x", &bundle, 0, 0),
        load_req("x", &bundle, 1, 9),
        load_req("x", "/no/such/bundle", 1, 0),
    ] {
        c.load_bundle(bad).await.expect_err("refused");
    }
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(list.bundles.len(), 1);
}

// ---- Over a listed limit ----------------------------------------------------

/// Token rows [batch, seq] sent as ints, the direct way past truncation:
/// ids 0 (the pad row) under a full mask.
fn token_rows(batch: i64, seq: i64) -> ModelInferRequest {
    let n = (batch * seq) as usize;
    let int_contents = |v: Vec<i32>| Some(InferTensorContents { int_contents: v, ..Default::default() });
    ModelInferRequest {
        model_name: NAME.into(),
        model_version: String::new(),
        id: String::new(),
        parameters: HashMap::new(),
        inputs: vec![
            InferInputTensor {
                name: "ids".into(),
                datatype: "INT32".into(),
                shape: vec![batch, seq],
                parameters: HashMap::new(),
                contents: int_contents(vec![0; n]),
            },
            InferInputTensor {
                name: "mask".into(),
                datatype: "INT32".into(),
                shape: vec![batch, seq],
                parameters: HashMap::new(),
                contents: int_contents(vec![1; n]),
            },
        ],
        outputs: vec![],
        raw_input_contents: vec![],
    }
}

#[tokio::test]
async fn an_over_limit_request_names_each_listed_limit() {
    let (s, mut c) = serve().await;
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let b = the_one_bundle(&list);
    let limits = list.limits.unwrap();

    // The bundle's limits, exactly as listed, refused the way docs/kserve.md
    // maps TURBO_E_CAPACITY, and nothing is truncated.
    let mut k = GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let over_seq = k.model_infer(token_rows(1, b.context_limit as i64 + 1)).await.expect_err("over context_limit");
    assert_eq!(refusal(&over_seq), (Code::OutOfRange, 771));
    assert!(over_seq.message().contains("max_seq"), "{}", over_seq.message());
    let over_batch = k.model_infer(token_rows(b.batch_limit as i64 + 1, 1)).await.expect_err("over batch_limit");
    assert_eq!(refusal(&over_batch), (Code::OutOfRange, 771));
    assert!(over_batch.message().contains("max_batch"), "{}", over_batch.message());
    // At the limit itself the request is answered.
    k.model_infer(token_rows(1, b.context_limit as i64)).await.unwrap();

    // The result-message limits, one refusal per listed limit.
    let sparse = SparseTermsResult {
        row: vec![0, limits.max_sparse_entries as u32 + 1],
        term: vec![1; limits.max_sparse_entries as usize + 1],
        weight: vec![1.0; limits.max_sparse_entries as usize + 1],
        ..Default::default()
    };
    let e = c.round_trip_sparse_terms(sparse).await.expect_err("over max_sparse_entries");
    assert_eq!(refusal(&e), (Code::OutOfRange, 771));
    assert!(e.message().contains("max_sparse_entries"), "{}", e.message());

    let vectors = TokenVectorsResult {
        dim: 1,
        data: vec![0u8; limits.max_token_vector_bytes as usize + 4].into(),
        ..Default::default()
    };
    let e = c.round_trip_token_vectors(vectors).await.expect_err("over max_token_vector_bytes");
    assert_eq!(refusal(&e), (Code::OutOfRange, 771));
    assert!(e.message().contains("max_token_vector_bytes"), "{}", e.message());

    let rerank = RerankScoresResult {
        item: vec![1; limits.max_rerank_items as usize + 1],
        score: vec![1.0; limits.max_rerank_items as usize + 1],
        ..Default::default()
    };
    let e = c.round_trip_rerank_scores(rerank).await.expect_err("over max_rerank_items");
    assert_eq!(refusal(&e), (Code::OutOfRange, 771));
    assert!(e.message().contains("max_rerank_items"), "{}", e.message());

    let extraction = ExtractionResult {
        null_probability: 1.0,
        span_start: vec![0; limits.max_answers as usize + 1],
        span_end: vec![1; limits.max_answers as usize + 1],
        ..Default::default()
    };
    let e = c.round_trip_extraction(extraction).await.expect_err("over max_answers");
    assert_eq!(refusal(&e), (Code::OutOfRange, 771));
    assert!(e.message().contains("max_answers"), "{}", e.message());
}

#[tokio::test]
async fn a_malformed_result_message_is_a_typed_refusal_not_a_panic() {
    let (s, mut c) = serve().await;
    // A short weight column.
    let mut msg = <SparseTerms<View> as Wire>::to_msg(&sparse_terms());
    msg.weight.pop();
    let e = c.round_trip_sparse_terms(msg).await.expect_err("garbage");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));
    assert!(e.message().contains("weight"), "{}", e.message());
    // A window that ends before it starts.
    let (v, w) = token_vectors();
    let mut msg = TokenVectors::new(&v, &w).to_msg();
    msg.window_start[0] = 30;
    let e = c.round_trip_token_vectors(msg).await.expect_err("garbage");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));
    // Nothing at all.
    let e = c.round_trip_extraction(ExtractionResult::default()).await.expect_err("garbage");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));
    // The listing's roles answer what the inference service honors: a text
    // request names the role the bundle lists.
    let mut k = GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let mut req = ModelInferRequest {
        model_name: NAME.into(),
        model_version: String::new(),
        id: String::new(),
        parameters: HashMap::new(),
        inputs: vec![InferInputTensor {
            name: "texts".into(),
            datatype: "BYTES".into(),
            shape: vec![1],
            parameters: HashMap::new(),
            contents: Some(InferTensorContents { bytes_contents: vec![b"hello".to_vec()], ..Default::default() }),
        }],
        outputs: vec![],
        raw_input_contents: vec![],
    };
    req.parameters.insert(
        "prompt_role".into(),
        turbo_kserve::proto::InferParameter {
            parameter_choice: Some(ParameterChoice::StringParam("PROMPT_QUERY".into())),
        },
    );
    k.model_infer(req).await.unwrap();
}
