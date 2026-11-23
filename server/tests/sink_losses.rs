//! What the sink leg loses is told (issue #2513): documents dropped at a
//! full queue, left in it at a stop, or sent on a call that then failed
//! are Prometheus series and part of what a stop answers with, so a
//! client answered OK is never the only record of a document the sink
//! did not get.

mod common;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{open_stream, tiny};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
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

/// A server teeing to `endpoint`, its model loaded.
async fn teeing(endpoint: String, max_queue_items: usize) -> Server {
    let config = SinkConfig {
        endpoint,
        max_queue_items,
        max_queue_bytes: 64 << 20,
        documents: false,
        typed: None,
        pii_output: Default::default(),
    };
    let serve = Serve { sink: Some(config), ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap();
    s.load().await.unwrap();
    s
}

/// An address that refuses every connection while the socket is held.
fn refusing() -> (tokio::net::TcpSocket, SocketAddr) {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = socket.local_addr().unwrap();
    (socket, addr)
}

/// Streams `n` documents through one AnalyzeStream call and waits for
/// every answer: the client is answered OK whatever the sink does.
async fn analyze(s: &Server, tag: &str, n: usize) {
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    for i in 0..n {
        stream.send(&format!("{tag}{i}"), common::NAME, "a short text to embed", None).await;
    }
    assert_eq!(stream.half_close().collect().await.len(), n);
}

/// Polls the stats until `done` holds, for at most two seconds of running time.
async fn until(s: &Server, done: impl Fn(&SinkStats) -> bool) -> SinkStats {
    let mut budget = common::Budget::new(Duration::from_secs(2));
    loop {
        let stats = s.sink_stats().unwrap();
        if done(&stats) || !budget.has_left() {
            return stats;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The value of one unlabelled series in a scrape.
fn series(rendered: &str, name: &str) -> Option<u64> {
    rendered.lines().find_map(|l| l.strip_prefix(name)?.strip_prefix(' ')?.parse().ok())
}

#[tokio::test]
async fn a_scrape_shows_what_a_full_queue_dropped() {
    let (_held, gone) = refusing();
    let s = teeing(format!("http://{gone}"), 2).await;
    analyze(&s, "d", 5).await;
    until(&s, |st| st.failed_calls > 0).await;
    let scrape = s.metrics();
    let read = |name: &str| series(&scrape, name).unwrap_or_else(|| panic!("no {name} in the scrape:\n{scrape}"));
    assert_eq!(read("turbo_kserve_sink_offered_total"), 5);
    assert_eq!(read("turbo_kserve_sink_dropped_total"), 3);
    assert_eq!(read("turbo_kserve_sink_forwarded_total"), 0);
    assert_eq!(read("turbo_kserve_sink_queued_items"), 2);
    assert!(read("turbo_kserve_sink_queued_bytes") > 0);
    assert!(read("turbo_kserve_sink_failed_calls_total") >= 1);
    for zero in ["refused", "abandoned", "unconfirmed", "accepted", "rejected"] {
        assert_eq!(read(&format!("turbo_kserve_sink_{zero}_total")), 0, "{zero}");
    }
    // Closed first with a short wait: the stop's own close waits ten
    // seconds for a sink that is down.
    s.close_sink(Duration::from_millis(300)).await.unwrap();
    s.stop().await;
}

#[tokio::test]
async fn a_server_without_a_sink_exports_no_sink_series() {
    let (s, _) = common::serve().await;
    assert!(!s.metrics().contains("turbo_kserve_sink_"));
    s.stop().await;
}

/// Relays TCP to `to` until cut.
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

/// A call that fails never answered with a summary, so what it had
/// taken is not counted forwarded any more: it is unconfirmed.
#[tokio::test]
async fn items_on_a_call_that_fails_are_unconfirmed_and_no_longer_counted_forwarded() {
    let receiver = SinkReceiver::new(|_| {});
    let serve = Serve { sink_receiver: Some(receiver), ..Serve::default() };
    let sink = Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap();
    let relay = Relay::start(sink.local_addr()).await;
    let s = teeing(format!("http://{}", relay.addr), 16).await;
    analyze(&s, "d", 2).await;
    let stats = until(&s, |st| st.forwarded == 2).await;
    assert_eq!((stats.forwarded, stats.failed_calls), (2, 0), "{stats:?}");
    relay.cut();
    until(&s, |st| st.failed_calls > 0).await;
    let scrape = s.metrics();
    let read = |name: &str| series(&scrape, name).unwrap_or_else(|| panic!("no {name} in the scrape:\n{scrape}"));
    assert_eq!(read("turbo_kserve_sink_unconfirmed_total"), 2);
    assert_eq!(read("turbo_kserve_sink_forwarded_total"), 0);
    assert_eq!(read("turbo_kserve_sink_accepted_total"), 0);
    let stats = s.sink_stats().unwrap();
    assert_eq!((stats.offered, stats.forwarded, stats.unconfirmed, stats.accepted), (2, 0, 2, 0), "{stats:?}");
    assert_eq!(stats.undelivered(), 2, "{stats:?}");
    // The relay is gone: closed first with a short wait, as above.
    s.close_sink(Duration::from_millis(300)).await.unwrap();
    s.stop().await;
    sink.stop().await;
}

/// The stop answers with the forwarder's final counts, which the binary
/// prints: here a sink that is down, so the queue is abandoned.
#[tokio::test]
async fn a_stop_reports_what_the_sink_never_got() {
    let (_held, gone) = refusing();
    let s = teeing(format!("http://{gone}"), 2).await;
    analyze(&s, "d", 5).await;
    until(&s, |st| st.failed_calls > 0).await;
    // The stop's own close waits ten seconds for a sink that is down;
    // closing first with a short wait leaves it nothing to wait for.
    s.close_sink(Duration::from_millis(300)).await.unwrap();
    let stopped = s.stop_reporting(turbo_kserve::STOP_GRACE).await;
    assert!(stopped.clean);
    let stats = stopped.sink.expect("a server with a sink reports it");
    assert_eq!((stats.offered, stats.dropped, stats.abandoned, stats.queued), (5, 3, 2, 0), "{stats:?}");
    assert_eq!(stats.undelivered(), 5, "{stats:?}");
    let line = stats.to_string();
    assert!(line.contains("dropped 3") && line.contains("abandoned 2") && line.contains("last error"), "{line}");
}

/// A sink that takes everything: the summary's counts are summed, and
/// nothing is undelivered.
#[tokio::test]
async fn a_stop_reports_what_the_sink_acknowledged() {
    let receiver = SinkReceiver::new(|_| {});
    let serve = Serve { sink_receiver: Some(receiver), ..Serve::default() };
    let sink = Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap();
    let s = teeing(format!("http://{}", sink.local_addr()), 16).await;
    analyze(&s, "d", 3).await;
    let stats = s.stop_reporting(turbo_kserve::STOP_GRACE).await.sink.unwrap();
    let got = (stats.offered, stats.forwarded, stats.accepted, stats.rejected, stats.unconfirmed);
    assert_eq!(got, (3, 3, 3, 0, 0), "{stats:?}");
    assert_eq!(stats.undelivered(), 0, "{stats:?}");
    sink.stop().await;
}

#[tokio::test]
async fn a_server_without_a_sink_reports_none() {
    let (s, _) = common::serve().await;
    assert_eq!(s.stop_reporting(turbo_kserve::STOP_GRACE).await.sink, None);
}

/// A sink that takes items and never answers: the call is still open
/// when the stop's wait runs out, so nothing acknowledged what it sent.
/// The stop settles that call too (review of 6cc9ed0a).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stop_against_a_sink_that_never_answers_counts_its_items_unconfirmed() {
    let receiver = SinkReceiver::new(|_| std::thread::sleep(Duration::from_secs(4)));
    let serve = Serve { sink_receiver: Some(receiver), ..Serve::default() };
    let sink = Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap();
    let s = teeing(format!("http://{}", sink.local_addr()), 16).await;
    analyze(&s, "hung", 3).await;
    until(&s, |st| st.forwarded == 3).await;
    let stats = s.close_sink(Duration::from_millis(500)).await.unwrap();
    assert_eq!((stats.forwarded, stats.unconfirmed, stats.abandoned), (0, 3, 0), "{stats:?}");
    assert_eq!(stats.undelivered(), 3, "{stats:?}");
    s.stop().await;
    sink.stop().await;
}
