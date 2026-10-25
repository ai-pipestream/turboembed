//! TurboSinkService end to end: one server tees what its analysis
//! streams complete to another serving the sink, and the sink checks each
//! item and answers with its counts.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{open_stream, tiny};
use turbo_analysis::export::export;
use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::Sentences;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{AnalyzeStreamResponse, AnalyzeVectors};
use turbo_kserve::proto::sink::DocumentSinkItem;
use turbo_kserve::proto::sink::turbo_sink_service_client::TurboSinkServiceClient;
use turbo_kserve::sink::{SinkConfig, SinkReceiver};
use turbo_kserve::{Serve, Server};

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

type Got = Arc<Mutex<Vec<DocumentSinkItem>>>;

/// A server serving the sink, keeping what it accepts.
async fn sink_server() -> (Server, Got) {
    let got: Got = Arc::default();
    let keep = got.clone();
    let receiver = SinkReceiver::new(move |item| keep.lock().unwrap().push(item));
    let serve = Serve { sink_receiver: Some(receiver), ..Serve::default() };
    (Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap(), got)
}

#[tokio::test]
async fn completed_stream_documents_reach_the_sink_in_order() {
    let (sink, got) = sink_server().await;
    let config =
        SinkConfig { endpoint: format!("http://{}", sink.local_addr()), max_queue_items: 16, max_queue_bytes: 1 << 20 };
    let serve = Serve { sink: Some(config), ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap();
    s.load().await.unwrap();
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    for (id, m) in [("a", common::NAME), ("b", common::NAME), ("nope", "no-such-model"), ("c", common::NAME)] {
        stream.send(id, m, "a short text to embed", None).await;
    }
    let answers = stream.half_close().collect().await;
    assert_eq!(answers.len(), 4);
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    assert_eq!((stats.offered, stats.forwarded, stats.dropped, stats.failed_calls), (3, 3, 0, 0), "{stats:?}");
    let summary = stats.last_summary.unwrap();
    assert_eq!((summary.received_count, summary.accepted_count, summary.rejected_count), (3, 3, 0));
    let items = got.lock().unwrap().clone();
    assert_eq!(items.iter().map(|i| i.sequence).collect::<Vec<_>>(), [1, 2, 3]);
    let mut ids: Vec<&str> = items.iter().map(|i| i.doc_id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, ["a", "b", "c"], "the failed document is not a completed one");
    for item in &items {
        let a = item.analyzed.as_ref().unwrap();
        assert_eq!(a.id, item.doc_id);
        assert!(matches!(&a.outcome, Some(Outcome::Vectors(v)) if v.dim > 0 && !v.data.is_empty()));
    }
    s.stop().await;
    sink.stop().await;
}

fn answer(id: &str) -> AnalyzeStreamResponse {
    let vectors = AnalyzeVectors { model: "m".into(), version: "1".into(), dim: 1, data: vec![0, 0, 128, 63] };
    AnalyzeStreamResponse { id: id.into(), outcome: Some(Outcome::Vectors(vectors)) }
}

fn graph() -> Vec<u8> {
    let mut doc = Document::new("One sentence.");
    let mut spans = Spans::<Sentences>::new(ViewId::SOURCE);
    spans.push(Span::new(0, 13));
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    export(&doc, turbo_kserve::results::document::codecs()).unwrap().to_bytes().unwrap()
}

#[tokio::test]
async fn the_sink_checks_every_item_and_counts_what_it_refuses() {
    let (sink, got) = sink_server().await;
    let mut c = TurboSinkServiceClient::connect(format!("http://{}", sink.local_addr())).await.unwrap();
    let item = |sequence: u64, graph: Vec<u8>, analyzed: Option<AnalyzeStreamResponse>| DocumentSinkItem {
        sequence,
        doc_id: format!("d{sequence}"),
        graph,
        analyzed,
        ..Default::default()
    };
    let items = vec![
        item(1, Vec::new(), Some(answer("d1"))),
        item(1, Vec::new(), Some(answer("again"))),
        item(3, Vec::new(), None),
        item(4, vec![1, 2, 3], None),
        item(5, graph(), None),
    ];
    let summary = c.stream_documents(tokio_stream::iter(items)).await.unwrap().into_inner();
    assert_eq!((summary.received_count, summary.accepted_count, summary.rejected_count), (5, 2, 3));
    assert_eq!(summary.errors.len(), 3);
    assert!(summary.errors[0].starts_with("item 1: sequence 1 is not over the last"), "{:?}", summary.errors);
    assert!(summary.errors[1].starts_with("item 3: carries neither"), "{:?}", summary.errors);
    assert!(summary.errors[2].starts_with("item 4: document graph"), "{:?}", summary.errors);
    let kept: Vec<u64> = got.lock().unwrap().iter().map(|i| i.sequence).collect();
    assert_eq!(kept, [1, 5]);
    sink.stop().await;
}

#[tokio::test]
async fn a_server_without_a_receiver_does_not_serve_the_sink() {
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, Serve::default()).await.unwrap();
    let mut c = TurboSinkServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let e = c.stream_documents(tokio_stream::iter(Vec::<DocumentSinkItem>::new())).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::Unimplemented);
    assert!(s.sink_stats().is_none());
    s.stop().await;
}
