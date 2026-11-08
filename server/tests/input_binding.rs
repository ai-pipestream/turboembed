//! The caller's InputBinding (#1939) on AnalyzeStream, on
//! AnalyzeDocumentProgressive and on the sink: copied as given onto the
//! answer, the receipt, the sink item and its typed projection; absent when
//! the caller sent none, never invented; a parse_artifact_digest without a
//! parse_artifact_id refused; and a sink item whose projection or answer
//! carries another binding than its own rejected.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{open_stream, tiny};
use tokio_stream::StreamExt;
use turbo_analysis::export::export;
use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::Sentences;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{
    AnalyzeDocument, AnalyzePipelineRequest, AnalyzeStreamRequest, AnalyzeStreamResponse, InputBinding, OffsetMode,
    PiiOutputOptions, ProcessingReceipt, TypedDocumentResult,
};
use turbo_kserve::proto::sink::DocumentSinkItem;
use turbo_kserve::proto::sink::turbo_sink_service_client::TurboSinkServiceClient;
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::sink::{SinkConfig, SinkReceiver, SinkTyped};
use turbo_kserve::{Serve, Server};

const TEXT: &str = "Größe zählt. Zweiter Satz, café.";
const PII_TEXT: &str = "Write to alice.planted@example.com before noon.";

fn model() -> ModelConfig {
    ModelConfig {
        bundle: tiny().to_str().unwrap().into(),
        name: None,
        device: Device::Index(common::cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

fn binding() -> InputBinding {
    InputBinding {
        source_revision: "rev-7".into(),
        parse_artifact_id: "parse/doc-1".into(),
        parse_artifact_digest: vec![0xab; 32],
    }
}

/// A digest naming no artifact: refused everywhere.
fn orphan_digest() -> InputBinding {
    InputBinding { parse_artifact_id: String::new(), ..binding() }
}

fn graph() -> Vec<u8> {
    let mut doc = Document::new(TEXT);
    let mut spans = Spans::<Sentences>::new(ViewId::SOURCE);
    spans.push(Span::new(0, TEXT.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    export(&doc, turbo_kserve::results::document::codecs()).unwrap().to_bytes().unwrap()
}

type Got = Arc<Mutex<Vec<DocumentSinkItem>>>;

async fn sink_server() -> (Server, Got) {
    let got: Got = Arc::default();
    let keep = got.clone();
    let receiver = SinkReceiver::new(move |item| keep.lock().unwrap().push(item));
    let serve = Serve { sink_receiver: Some(receiver), ..Serve::default() };
    (Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap(), got)
}

async fn sender(sink: &Server, typed: Option<SinkTyped>, pii_output: Disclosure) -> Server {
    let config = SinkConfig {
        endpoint: format!("http://{}", sink.local_addr()),
        max_queue_items: 16,
        max_queue_bytes: 64 << 20,
        documents: true,
        typed,
        pii_output,
    };
    let serve = Serve { sink: Some(config), ..Serve::default() };
    Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap()
}

const TYPED: Option<SinkTyped> = Some(SinkTyped { view: 0, offsets: OffsetMode::Utf8 });

/// Closes the sender's sink and answers the items the sink accepted, all
/// `want` of them, and none rejected.
async fn drained(s: &Server, got: &Got, want: u64) -> Vec<DocumentSinkItem> {
    let start = std::time::Instant::now();
    while s.sink_stats().unwrap().offered < want && start.elapsed() < Duration::from_secs(10) {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    let summary = stats.last_summary.unwrap();
    assert_eq!((summary.accepted_count, summary.rejected_count), (want, 0), "{:?}", summary.errors);
    got.lock().unwrap().clone()
}

/// The typed projection's binding and its receipt's, which must agree.
fn typed_binding(t: &TypedDocumentResult) -> Option<InputBinding> {
    let input = t.input.as_ref().expect("every projection carries its input identity");
    let receipt = t.receipt.as_ref().and_then(|r| r.input.as_ref()).expect("and its receipt's");
    assert_eq!(input, receipt, "the receipt and the typed result agree");
    input.binding.clone()
}

fn doc_msg(id: &str, text: &str, binding: Option<InputBinding>) -> AnalyzeStreamRequest {
    let d =
        AnalyzeDocument { id: id.into(), model: common::NAME.into(), text: text.into(), binding, ..Default::default() };
    AnalyzeStreamRequest { kind: Some(Kind::Document(d)) }
}

#[tokio::test]
async fn analyze_stream_echoes_the_binding_on_the_answer_and_the_sink_item() {
    let (sink, got) = sink_server().await;
    let s = sender(&sink, TYPED, Disclosure::FULL).await;
    s.load().await.unwrap();
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let stream = open_stream(&mut c, 0, 0, 0).await;
    stream.tx.send(doc_msg("bound", TEXT, Some(binding()))).await.unwrap();
    stream.tx.send(doc_msg("plain", TEXT, None)).await.unwrap();
    stream.tx.send(doc_msg("orphan", TEXT, Some(orphan_digest()))).await.unwrap();
    stream.tx.send(doc_msg("", TEXT, Some(binding()))).await.unwrap();
    let mut stream = stream.half_close();
    let mut answers = Vec::new();
    while let Some(r) = stream.responses.message().await.unwrap() {
        answers.push(r);
    }
    let by = |id: &str| answers.iter().find(|a| a.id == id).unwrap().clone();
    assert_eq!(answers.len(), 4);
    assert!(matches!(by("bound").outcome, Some(Outcome::Vectors(_))));
    assert_eq!(by("bound").binding, Some(binding()), "copied as given");
    assert_eq!(by("plain").binding, None, "absent when not sent, never invented");
    let Some(Outcome::Failure(f)) = by("orphan").outcome else { panic!("the orphan digest fails its document") };
    assert!(f.message.contains("parse_artifact_digest and no parse_artifact_id"), "{f:?}");
    assert_eq!(by("orphan").binding, Some(orphan_digest()), "a failure echoes the binding it was sent");
    assert!(matches!(by("").outcome, Some(Outcome::Failure(_))));
    assert_eq!(by("").binding, Some(binding()), "a refusal before the model runs echoes it too");
    // Failures are not offered: the two answers that completed reach the sink.
    let mut items = drained(&s, &got, 2).await;
    items.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
    assert_eq!(items[0].doc_id, "bound");
    assert_eq!(items[0].binding, Some(binding()));
    assert_eq!(items[0].analyzed, Some(by("bound")));
    assert_eq!((items[1].doc_id.as_str(), items[1].binding.as_ref()), ("plain", None));
    assert!(items.iter().all(|i| i.typed.is_none()), "answers carry no projection");
    s.stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn a_forwarded_graph_carries_the_binding_on_the_item_and_its_projection() {
    let (sink, got) = sink_server().await;
    let s = sender(&sink, TYPED, Disclosure::FULL).await;
    assert!(s.forward_bound_document("bound", graph(), Some(binding())).await.unwrap());
    assert!(s.forward_document("plain", graph()).await.unwrap());
    let e = s.forward_bound_document("orphan", graph(), Some(orphan_digest())).await.unwrap_err();
    assert!(e.message.contains("parse_artifact_digest and no parse_artifact_id"), "{e:?}");
    let items = drained(&s, &got, 2).await;
    assert_eq!(s.sink_stats().unwrap().refused, 1, "the orphan digest is refused, not offered");
    let (bound, plain) = (&items[0], &items[1]);
    assert_eq!((bound.doc_id.as_str(), bound.binding.as_ref()), ("bound", Some(&binding())));
    assert_eq!(typed_binding(bound.typed.as_ref().unwrap()), Some(binding()));
    assert_eq!((plain.doc_id.as_str(), plain.binding.as_ref()), ("plain", None));
    assert_eq!(typed_binding(plain.typed.as_ref().unwrap()), None, "never invented");
    s.stop().await;
    sink.stop().await;
}

/// Runs one progressive call and answers its receipt.
async fn progressive(s: &Server, req: AnalyzePipelineRequest) -> ProcessingReceipt {
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let mut stream = c.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut receipt = None;
    while let Some(ev) = stream.next().await {
        if let Some(Event::Receipt(r)) = ev.unwrap().event {
            receipt = Some(r);
        }
    }
    receipt.expect("the call ends with its receipt")
}

fn pipeline(doc_id: &str, binding: Option<InputBinding>) -> AnalyzePipelineRequest {
    AnalyzePipelineRequest {
        doc_id: doc_id.into(),
        text: TEXT.into(),
        offsets: OffsetMode::Utf8 as i32,
        profile: "default".into(),
        binding,
        ..Default::default()
    }
}

#[tokio::test]
async fn a_progressive_call_copies_the_binding_to_its_receipt_and_sink_item() {
    let (sink, got) = sink_server().await;
    let s = sender(&sink, TYPED, Disclosure::FULL).await;
    let receipt = progressive(&s, pipeline("bound", Some(binding()))).await;
    assert_eq!(receipt.input.unwrap().binding, Some(binding()));
    let receipt = progressive(&s, pipeline("plain", None)).await;
    assert_eq!(receipt.input.unwrap().binding, None);
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let e = c.analyze_document_progressive(pipeline("orphan", Some(orphan_digest()))).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::InvalidArgument, "{e:?}");
    assert!(e.message().contains("parse_artifact_digest and no parse_artifact_id"), "{e:?}");
    let mut items = drained(&s, &got, 2).await;
    items.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
    assert_eq!((items[0].doc_id.as_str(), items[0].binding.as_ref()), ("bound", Some(&binding())));
    assert_eq!(typed_binding(items[0].typed.as_ref().unwrap()), Some(binding()));
    assert_eq!((items[1].doc_id.as_str(), items[1].binding.as_ref()), ("plain", None));
    assert_eq!(typed_binding(items[1].typed.as_ref().unwrap()), None);
    s.stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn a_withheld_graph_carries_the_binding_on_its_findings_projection() {
    let (sink, got) = sink_server().await;
    // No configured projection: the findings stand in for the PII graph.
    let s = sender(&sink, None, Disclosure::FINDINGS).await;
    let req = AnalyzePipelineRequest {
        text: PII_TEXT.into(),
        steps: vec!["turbo-pii:pii".into()],
        pii_output: Some(PiiOutputOptions::default()),
        ..pipeline("pii", Some(binding()))
    };
    progressive(&s, req).await;
    let items = drained(&s, &got, 1).await;
    assert!(items[0].graph.is_empty(), "the graph is withheld for PII");
    assert_eq!(items[0].binding, Some(binding()));
    assert_eq!(typed_binding(items[0].typed.as_ref().expect("the findings projection")), Some(binding()));
    s.stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn the_sink_rejects_a_malformed_or_disagreeing_binding() {
    let (sink, got) = sink_server().await;
    let answer = |binding| AnalyzeStreamResponse {
        id: "a".into(),
        outcome: Some(Outcome::Vectors(Default::default())),
        binding,
    };
    let item = |sequence, binding, analyzed| DocumentSinkItem {
        sequence,
        doc_id: "a".into(),
        analyzed: Some(analyzed),
        binding,
        ..Default::default()
    };
    let other = InputBinding { source_revision: "rev-8".into(), ..binding() };
    let items = vec![
        item(1, Some(binding()), answer(Some(binding()))),
        item(2, Some(orphan_digest()), answer(Some(orphan_digest()))),
        item(3, Some(binding()), answer(Some(other))),
        item(4, None, answer(Some(binding()))),
        item(5, None, answer(None)),
    ];
    let mut c = TurboSinkServiceClient::connect(format!("http://{}", sink.local_addr())).await.unwrap();
    let summary = c.stream_documents(tokio_stream::iter(items)).await.unwrap().into_inner();
    assert_eq!((summary.accepted_count, summary.rejected_count), (2, 3), "{:?}", summary.errors);
    assert!(summary.errors[0].starts_with("item 2:") && summary.errors[0].contains("no parse_artifact_id"));
    assert!(summary.errors[1].starts_with("item 3: the answer carries another binding"), "{:?}", summary.errors);
    assert!(summary.errors[2].starts_with("item 4: the answer carries another binding"), "{:?}", summary.errors);
    let kept: Vec<u64> = got.lock().unwrap().iter().map(|i| i.sequence).collect();
    assert_eq!(kept, vec![1, 5]);
    sink.stop().await;
}

#[tokio::test]
async fn the_sink_rejects_a_projection_bound_to_another_input() {
    let (sink, got) = sink_server().await;
    let s = sender(&sink, TYPED, Disclosure::FULL).await;
    assert!(s.forward_bound_document("bound", graph(), Some(binding())).await.unwrap());
    let good = drained(&s, &got, 1).await.pop().unwrap();
    let (sink2, got2) = sink_server().await;
    let unbound = DocumentSinkItem { sequence: 1, binding: None, ..good.clone() };
    let resent = DocumentSinkItem { sequence: 2, ..good };
    let mut c = TurboSinkServiceClient::connect(format!("http://{}", sink2.local_addr())).await.unwrap();
    let summary = c.stream_documents(tokio_stream::iter(vec![unbound, resent])).await.unwrap().into_inner();
    assert_eq!((summary.accepted_count, summary.rejected_count), (1, 1), "{:?}", summary.errors);
    assert!(summary.errors[0].starts_with("item 1: the typed projection carries another binding"));
    assert_eq!(got2.lock().unwrap()[0].binding, Some(binding()));
    s.stop().await;
    sink.stop().await;
    sink2.stop().await;
}
