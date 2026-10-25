//! TurboSinkService end to end: one server tees what its analysis
//! streams complete to another serving the sink, and the sink checks each
//! item and answers with its counts.

mod common;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
use turbo_kserve::sink::{SinkConfig, SinkReceiver, SinkStats};
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
    let vectors = AnalyzeVectors { model: "m".into(), version: "1".into(), dim: 1, data: vec![0, 0, 128, 63].into() };
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

/// A server teeing to `endpoint`, its model loaded.
async fn teeing(endpoint: String, max_queue_items: usize) -> Server {
    let config = SinkConfig { endpoint, max_queue_items, max_queue_bytes: 64 << 20 };
    let serve = Serve { sink: Some(config), ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap();
    s.load().await.unwrap();
    s
}

/// Streams `n` documents through one AnalyzeStream call and waits for
/// every answer. Their ids are `tag` and a number.
async fn analyze(s: &Server, tag: &str, n: usize) {
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    for i in 0..n {
        stream.send(&format!("{tag}{i}"), common::NAME, "a short text to embed", None).await;
    }
    assert_eq!(stream.half_close().collect().await.len(), n);
}

/// Polls the stats until `done` holds, for at most two seconds.
async fn until(s: &Server, done: impl Fn(&SinkStats) -> bool) -> SinkStats {
    let start = Instant::now();
    loop {
        let stats = s.sink_stats().unwrap();
        if done(&stats) || start.elapsed() > Duration::from_secs(2) {
            return stats;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_streams_reach_the_sink_numbered_in_the_order_sent() {
    let (sink, got) = sink_server().await;
    let s = Arc::new(teeing(format!("http://{}", sink.local_addr()), 1024).await);
    // Ids of very different sizes make the answers take very different
    // times to copy into the queue.
    let clients: Vec<_> = (0..16)
        .map(|k| {
            let s = s.clone();
            let tag = if k % 2 == 0 { format!("c{k}-{}-", "x".repeat(256 << 10)) } else { format!("c{k}-") };
            tokio::spawn(async move { analyze(&s, &tag, 4).await })
        })
        .collect();
    for c in clients {
        c.await.unwrap();
    }
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    assert_eq!((stats.offered, stats.forwarded, stats.dropped), (64, 64, 0), "{stats:?}");
    let summary = stats.last_summary.unwrap();
    assert_eq!((summary.accepted_count, summary.rejected_count), (64, 0), "{:?}", summary.errors);
    let sequences: Vec<u64> = got.lock().unwrap().iter().map(|i| i.sequence).collect();
    assert_eq!(sequences, (1..=64).collect::<Vec<u64>>());
    Arc::try_unwrap(s).ok().unwrap().stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn items_a_sink_cannot_take_wait_in_the_queue_and_are_counted_there() {
    // An address nothing listens on.
    let gone = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
    let s = teeing(format!("http://{gone}"), 2).await;
    analyze(&s, "d", 5).await;
    let stats = until(&s, |st| st.failed_calls > 0).await;
    assert_eq!((stats.offered, stats.forwarded, stats.queued, stats.dropped), (5, 0, 2, 3), "{stats:?}");
    assert!(stats.queued_bytes > 0, "{stats:?}");
    s.stop().await;
}

/// Relays TCP to `to` until cut: the tasks are kept so the test can end
/// every connection at once.
struct Relay {
    addr: SocketAddr,
    tasks: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

impl Relay {
    async fn start(to: SocketAddr) -> Relay {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let tasks: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>> = Arc::default();
        let keep = tasks.clone();
        let accept = tokio::spawn(async move {
            while let Ok((mut inbound, _)) = listener.accept().await {
                let conn = tokio::spawn(async move {
                    let mut outbound = tokio::net::TcpStream::connect(to).await.unwrap();
                    let _ = tokio::io::copy_bidirectional(&mut inbound, &mut outbound).await;
                });
                keep.lock().unwrap().push(conn);
            }
        });
        tasks.lock().unwrap().push(accept);
        Relay { addr, tasks }
    }

    fn cut(&self) {
        for t in self.tasks.lock().unwrap().drain(..) {
            t.abort();
        }
    }
}

#[tokio::test]
async fn a_call_cut_while_idle_is_counted_failed_without_waiting_for_the_next_item() {
    let (sink, _got) = sink_server().await;
    let relay = Relay::start(sink.local_addr()).await;
    let s = teeing(format!("http://{}", relay.addr), 16).await;
    analyze(&s, "d", 1).await;
    let stats = until(&s, |st| st.forwarded == 1).await;
    assert_eq!((stats.forwarded, stats.failed_calls), (1, 0), "{stats:?}");
    relay.cut();
    let stats = until(&s, |st| st.failed_calls > 0).await;
    assert_eq!(stats.failed_calls, 1, "no item was offered after the cut: {stats:?}");
    assert!(stats.last_error.is_some(), "{stats:?}");
    s.stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn items_offered_before_the_sink_starts_reach_it_once_it_does() {
    let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = free.local_addr().unwrap();
    drop(free);
    let s = teeing(format!("http://{addr}"), 16).await;
    analyze(&s, "early", 3).await;
    let stats = until(&s, |st| st.failed_calls > 0).await;
    assert_eq!((stats.forwarded, stats.queued, stats.dropped), (0, 3, 0), "{stats:?}");
    let got: Got = Arc::default();
    let keep = got.clone();
    let receiver = SinkReceiver::new(move |item| keep.lock().unwrap().push(item));
    let serve = Serve { sink_receiver: Some(receiver), ..Serve::default() };
    let sink = Server::start(addr, vec![model()], MAX, serve).await.unwrap();
    let stats = until(&s, |st| st.forwarded == 3).await;
    assert_eq!((stats.forwarded, stats.queued, stats.dropped), (3, 0, 0), "{stats:?}");
    let stats = s.close_sink(Duration::from_secs(10)).await.unwrap();
    let summary = stats.last_summary.unwrap();
    assert_eq!((summary.accepted_count, summary.rejected_count), (3, 0), "{:?}", summary.errors);
    assert_eq!(got.lock().unwrap().len(), 3);
    s.stop().await;
    sink.stop().await;
}

#[tokio::test]
async fn closing_with_the_sink_down_stops_the_drain_at_the_deadline() {
    let gone = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
    let s = teeing(format!("http://{gone}"), 4).await;
    analyze(&s, "d", 2).await;
    let start = Instant::now();
    let stats = s.close_sink(Duration::from_millis(300)).await.unwrap();
    assert!(start.elapsed() < Duration::from_secs(2), "{:?}", start.elapsed());
    let failed = stats.failed_calls;
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert_eq!(s.sink_stats().unwrap().failed_calls, failed, "the drain kept retrying after close");
    s.stop().await;
}
