//! The typed document projection on DocumentSinkItem: a server configured
//! for it forwards each document graph with the projection
//! InspectDocumentGraph answers for it, one without it forwards the item as
//! before, and the sink refuses a projection RoundTripTypedDocument would.
//! With `documents` set, every AnalyzeDocumentProgressive call forwards its
//! complete graph the same way.

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
use turbo_kserve::proto::result::{
    AnalyzePipelineRequest, AnalyzeStreamResponse, DocumentGraphRequest, DocumentGraphResponse,
    InspectDocumentGraphRequest, OffsetMode, TypedDocumentResult,
};
use turbo_kserve::proto::sink::DocumentSinkItem;
use turbo_kserve::proto::sink::turbo_sink_service_client::TurboSinkServiceClient;
use turbo_kserve::sink::{SinkConfig, SinkReceiver, SinkTyped};
use turbo_kserve::{Serve, Server};

/// Two sentences whose UTF-8, UTF-16 and code point offsets all differ.
const TEXT: &str = "Größe zählt. 😀 Zweiter Satz, café.";

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

fn graph() -> Vec<u8> {
    let mut doc = Document::new(TEXT);
    let first_end = TEXT.find('.').unwrap() + '.'.len_utf8();
    let second_start = TEXT.find('😀').unwrap();
    let mut spans = Spans::<Sentences>::new(ViewId::SOURCE);
    spans.push(Span::new(0, first_end as u32));
    spans.push(Span::new(second_start as u32, TEXT.len() as u32));
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

async fn sender(sink: &Server, typed: Option<SinkTyped>) -> Server {
    sender_with(sink, false, typed).await
}

async fn sender_with(sink: &Server, documents: bool, typed: Option<SinkTyped>) -> Server {
    let endpoint = format!("http://{}", sink.local_addr());
    let config = SinkConfig { endpoint, max_queue_items: 16, max_queue_bytes: 64 << 20, documents, typed };
    let serve = Serve { sink: Some(config), ..Serve::default() };
    Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap()
}

/// Forwards one graph through a sender configured with `typed` and answers
/// the item the sink accepted, with the sender still serving.
async fn forward(typed: Option<SinkTyped>) -> (Server, DocumentSinkItem) {
    let (sink, got) = sink_server().await;
    let s = sender(&sink, typed).await;
    assert!(s.forward_document("doc-1", graph()).await.unwrap());
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    let summary = stats.last_summary.unwrap();
    assert_eq!((summary.accepted_count, summary.rejected_count), (1, 0), "{:?}", summary.errors);
    sink.stop().await;
    let item = got.lock().unwrap().pop().unwrap();
    (s, item)
}

async fn inspect(s: &Server, graph: Vec<u8>, offsets: OffsetMode) -> TypedDocumentResult {
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let req = InspectDocumentGraphRequest { doc_id: "doc-1".into(), graph, view: 0, offsets: offsets as i32 };
    c.inspect_document_graph(req).await.unwrap().into_inner()
}

#[tokio::test]
async fn the_forwarded_projection_is_what_inspect_document_graph_answers() {
    let mut projections = Vec::new();
    for offsets in [OffsetMode::Utf8, OffsetMode::Utf16] {
        let (s, item) = forward(Some(SinkTyped { view: 0, offsets })).await;
        let typed = item.typed.clone().expect("the item carries the typed projection");
        assert_eq!(typed, inspect(&s, item.graph.clone(), offsets).await, "{offsets:?}");
        assert_eq!(typed, inspect(&s, graph(), offsets).await, "the projection of the graph as given, {offsets:?}");
        assert_eq!((typed.doc_id.as_str(), typed.raw_text.as_str(), typed.offsets), ("doc-1", TEXT, offsets as i32));
        assert!(typed.layers.iter().any(|l| l.annotations.len() == 2), "{typed:?}");
        s.stop().await;
        projections.push(typed);
    }
    let ends = |t: &TypedDocumentResult| -> Vec<u32> {
        t.layers.iter().flat_map(|l| &l.annotations).flat_map(|a| &a.spans).map(|f| f.end).collect()
    };
    let utf16_len = TEXT.encode_utf16().count() as u32;
    assert_eq!(ends(&projections[0]).last(), Some(&(TEXT.len() as u32)));
    assert_eq!(ends(&projections[1]).last(), Some(&utf16_len));
    assert_ne!(ends(&projections[0]), ends(&projections[1]));
}

#[tokio::test]
async fn without_the_option_the_item_carries_no_projection() {
    let (s, item) = forward(None).await;
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let listed =
        c.round_trip_document_graph(DocumentGraphRequest { graph: graph().into() }).await.unwrap().into_inner();
    let want = DocumentSinkItem {
        sequence: 1,
        doc_id: "doc-1".into(),
        graph: listed.graph.to_vec(),
        layers: listed.layers,
        views: listed.views,
        typed: None,
        analyzed: None,
    };
    assert_eq!(item, want);
    s.stop().await;
}

#[tokio::test]
async fn analyze_stream_answers_carry_no_projection_with_the_option_set() {
    let (sink, got) = sink_server().await;
    let s = sender(&sink, Some(SinkTyped { view: 0, offsets: OffsetMode::Utf16 })).await;
    s.load().await.unwrap();
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    stream.send("a", common::NAME, TEXT, None).await;
    let answers = stream.half_close().collect().await;
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    assert_eq!(stats.last_summary.unwrap().accepted_count, 1);
    let item = got.lock().unwrap().pop().unwrap();
    let (id, outcome) = answers.into_iter().next().unwrap();
    let analyzed = Some(AnalyzeStreamResponse { id: id.clone(), outcome: Some(outcome) });
    let want = DocumentSinkItem { sequence: 1, doc_id: id, analyzed, ..Default::default() };
    assert_eq!(item, want);
    s.stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn a_graph_the_projection_cannot_take_is_refused_and_not_offered() {
    let (sink, _got) = sink_server().await;
    let s = sender(&sink, Some(SinkTyped { view: 7, offsets: OffsetMode::Utf8 })).await;
    let e = s.forward_document("d", graph()).await.unwrap_err();
    assert!(e.message.contains("target view 7"), "{e:?}");
    assert!(s.forward_document("d", vec![1, 2, 3]).await.is_err());
    assert_eq!(s.sink_stats().unwrap().offered, 0);
    s.stop().await;
    let plain = Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, Serve::default()).await.unwrap();
    assert!(plain.forward_document("d", graph()).await.is_err(), "a server without a sink");
    plain.stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn the_sink_refuses_a_projection_round_trip_typed_document_refuses() {
    let (sink, got) = sink_server().await;
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, Serve::default()).await.unwrap();
    let typed = inspect(&s, graph(), OffsetMode::Utf16).await;
    let item = |sequence: u64, graph: Vec<u8>, typed: TypedDocumentResult| DocumentSinkItem {
        sequence,
        doc_id: "doc-1".into(),
        graph,
        typed: Some(typed),
        ..Default::default()
    };
    // Spans past the end of a shortened text, and an unknown offset mode.
    let short = TypedDocumentResult { raw_text: "Größe".into(), ..typed.clone() };
    let unknown_mode = TypedDocumentResult { offsets: 99, ..typed.clone() };
    let answer = AnalyzeStreamResponse { id: "doc-1".into(), outcome: None };
    let without_graph = DocumentSinkItem { analyzed: Some(answer), ..item(4, Vec::new(), typed.clone()) };
    let items =
        vec![item(1, graph(), typed.clone()), item(2, graph(), short), item(3, graph(), unknown_mode), without_graph];
    let mut c = TurboSinkServiceClient::connect(format!("http://{}", sink.local_addr())).await.unwrap();
    let summary = c.stream_documents(tokio_stream::iter(items)).await.unwrap().into_inner();
    assert_eq!((summary.received_count, summary.accepted_count, summary.rejected_count), (4, 1, 3));
    assert!(summary.errors[0].starts_with("item 2: typed document:"), "{:?}", summary.errors);
    assert!(summary.errors[1].starts_with("item 3: typed document:"), "{:?}", summary.errors);
    assert!(
        summary.errors[2].starts_with("item 4: carries a typed projection without a graph"),
        "{:?}",
        summary.errors
    );
    let kept = got.lock().unwrap().clone();
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].typed.as_ref(), Some(&typed));
    s.stop().await;
    sink.stop().await;
}

/// Runs one AnalyzeDocumentProgressive call of the default profile on `s`
/// and answers its Complete event.
async fn progressive(s: &Server, doc_id: &str) -> DocumentGraphResponse {
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let req = AnalyzePipelineRequest {
        doc_id: doc_id.into(),
        text: TEXT.into(),
        offsets: OffsetMode::Utf8 as i32,
        profile: "default".into(),
        ..Default::default()
    };
    let mut stream = c.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut complete = None;
    while let Some(ev) = stream.next().await {
        if let Some(Event::Complete(done)) = ev.unwrap().event {
            complete = Some(done);
        }
    }
    complete.expect("the call ends with its complete graph")
}

/// Waits, up to two seconds, for the sender to offer or refuse a graph:
/// it does once the call's reply has ended.
async fn settle(s: &Server) {
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(2) {
        let st = s.sink_stats().unwrap();
        if st.offered + st.refused > 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn a_progressive_call_forwards_its_graph_with_the_typed_projection() {
    let (sink, got) = sink_server().await;
    let s = sender_with(&sink, true, Some(SinkTyped { view: 0, offsets: OffsetMode::Utf16 })).await;
    let complete = progressive(&s, "doc-p").await;
    settle(&s).await;
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    assert_eq!((stats.offered, stats.forwarded, stats.dropped, stats.refused), (1, 1, 0, 0), "{stats:?}");
    let summary = stats.last_summary.unwrap();
    assert_eq!((summary.accepted_count, summary.rejected_count), (1, 0), "{:?}", summary.errors);
    let item = got.lock().unwrap().pop().unwrap();
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let listed = c.round_trip_document_graph(DocumentGraphRequest { graph: complete.graph.clone() }).await;
    let listed = listed.unwrap().into_inner();
    assert_eq!(item.doc_id, "doc-p");
    assert_eq!(item.graph, listed.graph.to_vec());
    assert_eq!((&item.layers, &item.views), (&listed.layers, &listed.views));
    assert_eq!(item.layers, complete.layers, "the layers the call's Complete event listed");
    assert!(item.analyzed.is_none());
    let typed = item.typed.clone().expect("the item carries the typed projection");
    let want =
        TypedDocumentResult { doc_id: "doc-p".into(), ..inspect(&s, complete.graph.to_vec(), OffsetMode::Utf16).await };
    assert_eq!(typed, want);
    assert_eq!(typed.raw_text, TEXT);
    assert!(!typed.layers.is_empty());
    s.stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn without_documents_a_progressive_call_forwards_nothing() {
    let (sink, got) = sink_server().await;
    let s = sender_with(&sink, false, Some(SinkTyped { view: 0, offsets: OffsetMode::Utf8 })).await;
    progressive(&s, "doc-p").await;
    settle(&s).await;
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    assert_eq!((stats.offered, stats.forwarded), (0, 0), "{stats:?}");
    assert!(got.lock().unwrap().is_empty());
    s.stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn a_progressive_graph_without_the_typed_view_is_refused_and_counted() {
    let (sink, got) = sink_server().await;
    let s = sender_with(&sink, true, Some(SinkTyped { view: 9, offsets: OffsetMode::Utf8 })).await;
    // The call itself completes as it would without a sink.
    progressive(&s, "doc-p").await;
    settle(&s).await;
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    assert_eq!((stats.offered, stats.refused), (0, 1), "{stats:?}");
    assert!(stats.last_error.as_deref().is_some_and(|e| e.contains("target view 9")), "{stats:?}");
    assert!(got.lock().unwrap().is_empty());
    s.stop().await;
    sink.stop().await;
}
