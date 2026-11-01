//! Forwarding analyzed documents to a downstream TurboSinkService
//! (proto/turbo_sink.proto, docs/grpc-sink.md). Every answer an analysis
//! stream completes is offered to a bounded queue without waiting: when
//! the queue is full, by items or by bytes, the item is dropped and
//! counted, so a slow sink never holds back the stream that produced it.
//! One task drains the queue into a long-lived StreamDocuments call,
//! opening a new one after a failure. The call takes each item from the
//! queue itself, as it sends it, so what waits is the queue's and counted
//! there, and the sequence follows the order the items are sent in.

mod document;
mod receiver;

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use prost::Message;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::Stream;
use tonic::transport::Endpoint;
use turbo_analysis::Document;

pub use document::SinkTyped;
pub(crate) use document::no_sink;
pub use receiver::{MAX_SUMMARY_ERRORS, SinkReceiver};

use crate::api::Failure;
use crate::proto::result::analyze_stream_response::Outcome;
use crate::proto::result::{AnalyzeStreamResponse, DocumentGraphResponse};
use crate::proto::sink::turbo_sink_service_client::TurboSinkServiceClient;
use crate::proto::sink::{DocumentSinkItem, DocumentSinkSummary};

/// `max_sink_queue_items` when not given.
pub const DEFAULT_SINK_QUEUE_ITEMS: usize = 1024;
/// `max_sink_queue_bytes` when not given: 64 MiB.
pub const DEFAULT_SINK_QUEUE_BYTES: u64 = 64 << 20;

/// The wait before a new call after one fails.
const RETRY: Duration = Duration::from_millis(500);

/// Where completed documents go, and how much may wait for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinkConfig {
    /// The sink's gRPC address, such as `http://sink:8081`.
    pub endpoint: String,
    pub max_queue_items: usize,
    pub max_queue_bytes: u64,
    /// Whether every AnalyzeDocumentProgressive call that ends with its
    /// complete graph forwards that graph too.
    pub documents: bool,
    /// The typed projection every item carrying a graph also carries;
    /// `None` sends none, and items are as they were without it.
    pub typed: Option<SinkTyped>,
}

/// What the forwarder did so far.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SinkStats {
    /// Completed answers offered to the queue.
    pub offered: u64,
    /// Taken by a StreamDocuments call to send. Items a call held when it
    /// failed are lost with it; the sink's summary says what it accepted.
    pub forwarded: u64,
    /// Refused by a full queue.
    pub dropped: u64,
    /// Document graphs refused before the queue: a graph that does not
    /// import, or one without the typed projection's view. The latest
    /// reason is `last_error`.
    pub refused: u64,
    /// Left in the queue when close stopped a drain at its deadline.
    pub abandoned: u64,
    /// Items in the queue now, and their encoded bytes.
    pub queued: u64,
    pub queued_bytes: u64,
    /// Calls that ended in an error, and the latest error.
    pub failed_calls: u64,
    pub last_error: Option<String>,
    /// The latest summary a sink answered with.
    pub last_summary: Option<DocumentSinkSummary>,
}

#[derive(Default)]
struct Counters {
    sequence: AtomicU64,
    offered: AtomicU64,
    forwarded: AtomicU64,
    dropped: AtomicU64,
    refused: AtomicU64,
    abandoned: AtomicU64,
    queued: AtomicU64,
    queued_bytes: AtomicU64,
    failed_calls: AtomicU64,
    last: Mutex<(Option<String>, Option<DocumentSinkSummary>)>,
}

/// A queued item and the bytes it is counted at.
type Queued = (DocumentSinkItem, u64);

/// The tee every analysis stream offers its answers to.
pub(crate) struct SinkForwarder {
    tx: Mutex<Option<mpsc::Sender<Queued>>>,
    rx: Arc<Mutex<mpsc::Receiver<Queued>>>,
    max_bytes: u64,
    documents: bool,
    typed: Option<SinkTyped>,
    counters: Arc<Counters>,
    task: Mutex<Option<JoinHandle<()>>>,
}

impl SinkForwarder {
    /// Starts the draining task. An endpoint that is not a URI is refused;
    /// one that does not answer is retried, and counted, while it serves.
    pub(crate) fn start(config: &SinkConfig) -> Result<Arc<SinkForwarder>, String> {
        let endpoint = Endpoint::from_shared(config.endpoint.clone())
            .map_err(|e| format!("sink endpoint {}: {e}", config.endpoint))?;
        if config.max_queue_items == 0 || config.max_queue_bytes == 0 {
            return Err("the sink queue's item and byte bounds must be over 0".into());
        }
        if let Some(t) = &config.typed {
            t.check()?;
        }
        let (tx, rx) = mpsc::channel(config.max_queue_items);
        let rx = Arc::new(Mutex::new(rx));
        let counters = Arc::new(Counters::default());
        let task = tokio::spawn(drain(endpoint, rx.clone(), counters.clone()));
        Ok(Arc::new(SinkForwarder {
            tx: Mutex::new(Some(tx)),
            rx,
            max_bytes: config.max_queue_bytes,
            documents: config.documents,
            typed: config.typed,
            counters,
            task: Mutex::new(Some(task)),
        }))
    }

    /// Offers one answer without waiting. A failure is not a completed
    /// document and is not offered.
    pub(crate) fn offer(&self, answer: &AnalyzeStreamResponse) {
        if matches!(answer.outcome, None | Some(Outcome::Failure(_))) {
            return;
        }
        let item = DocumentSinkItem { doc_id: answer.id.clone(), analyzed: Some(answer.clone()), ..Default::default() };
        self.enqueue(item);
    }

    /// Offers one document graph without waiting, with the typed
    /// projection when configured. A graph that does not import, or a
    /// projection the graph cannot give, is refused and not offered.
    /// Answers whether the queue took it. Imports the graph: call it off
    /// the async runtime.
    pub(crate) fn offer_graph(&self, doc_id: &str, graph: Vec<u8>) -> Result<bool, Failure> {
        self.offer_item(doc_id, document::graph_item(doc_id, graph, self.typed))
    }

    /// Offers the graph this server exported from `doc` as `exported`,
    /// as `offer_graph` does, without importing it again. Builds the typed
    /// projection when configured: call it off the async runtime.
    pub(crate) fn offer_exported(
        &self,
        doc_id: &str,
        exported: DocumentGraphResponse,
        doc: &Document,
    ) -> Result<bool, Failure> {
        self.offer_item(doc_id, document::exported_item(doc_id, exported, doc, self.typed))
    }

    /// Queues `item`, or counts the refusal that built it.
    fn offer_item(&self, doc_id: &str, item: Result<DocumentSinkItem, Failure>) -> Result<bool, Failure> {
        match item {
            Ok(item) => Ok(self.enqueue(item)),
            Err(f) => {
                let c = &self.counters;
                c.refused.fetch_add(1, Ordering::Relaxed);
                c.last.lock().unwrap_or_else(|p| p.into_inner()).0 = Some(format!("document {doc_id}: {}", f.message));
                Err(f)
            }
        }
    }

    /// Whether AnalyzeDocumentProgressive graphs are forwarded.
    pub(crate) fn documents(&self) -> bool {
        self.documents
    }

    /// Queues `item`, or drops and counts it when the queue is full.
    fn enqueue(&self, mut item: DocumentSinkItem) -> bool {
        let c = &self.counters;
        // The sequence is set as the item is sent; measured at its widest
        // here, so the queue's byte count is never under what it sends.
        item.sequence = u64::MAX;
        let bytes = item.encoded_len() as u64;
        let fits = reserve(&c.queued_bytes, bytes, self.max_bytes);
        // Hold the sender across the offered count and the send so close
        // cannot take it between them: settle-then-close would otherwise
        // see offered and drop an item that never reached the queue.
        let guard = self.tx.lock().unwrap_or_else(|p| p.into_inner());
        c.offered.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        enqueue_gap();
        let sent = fits && guard.as_ref().is_some_and(|tx| tx.try_send((item, bytes)).is_ok());
        if sent {
            c.queued.fetch_add(1, Ordering::Relaxed);
        } else {
            if fits {
                c.queued_bytes.fetch_sub(bytes, Ordering::AcqRel);
            }
            c.dropped.fetch_add(1, Ordering::Relaxed);
        }
        sent
    }

    pub(crate) fn stats(&self) -> SinkStats {
        let c = &self.counters;
        let (last_error, last_summary) = c.last.lock().unwrap_or_else(|p| p.into_inner()).clone();
        SinkStats {
            offered: c.offered.load(Ordering::Relaxed),
            forwarded: c.forwarded.load(Ordering::Relaxed),
            dropped: c.dropped.load(Ordering::Relaxed),
            refused: c.refused.load(Ordering::Relaxed),
            abandoned: c.abandoned.load(Ordering::Relaxed),
            queued: c.queued.load(Ordering::Relaxed),
            queued_bytes: c.queued_bytes.load(Ordering::Relaxed),
            failed_calls: c.failed_calls.load(Ordering::Relaxed),
            last_error,
            last_summary,
        }
    }

    /// Closes the queue and waits up to `within` for what it holds to be
    /// sent and the sink's summary to arrive. A drain still running then,
    /// with a sink that is down, is stopped: what the queue still holds
    /// is taken out of it and counted abandoned.
    pub(crate) async fn close(&self, within: Duration) {
        self.tx.lock().unwrap_or_else(|p| p.into_inner()).take();
        let task = self.task.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(mut t) = task
            && tokio::time::timeout(within, &mut t).await.is_err()
        {
            t.abort();
            let _ = t.await;
            let c = &self.counters;
            let mut rx = self.rx.lock().unwrap_or_else(|p| p.into_inner());
            rx.close();
            while let Ok((_, bytes)) = rx.try_recv() {
                c.queued.fetch_sub(1, Ordering::Relaxed);
                c.queued_bytes.fetch_sub(bytes, Ordering::AcqRel);
                c.abandoned.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Adds `bytes` to `queued` if the sum stays within `max`.
fn reserve(queued: &AtomicU64, bytes: u64, max: u64) -> bool {
    let mut q = queued.load(Ordering::Acquire);
    loop {
        if q + bytes > max {
            return false;
        }
        match queued.compare_exchange_weak(q, q + bytes, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(now) => q = now,
        }
    }
}

/// One call's request stream: each item taken from the queue as the call
/// asks for it, uncounted from the queue, numbered and counted forwarded.
/// The queue's end ends the stream and is recorded in `closed`.
struct Feed {
    rx: Arc<Mutex<mpsc::Receiver<Queued>>>,
    c: Arc<Counters>,
    closed: Arc<AtomicBool>,
}

impl Stream for Feed {
    type Item = DocumentSinkItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<DocumentSinkItem>> {
        let polled = self.rx.lock().unwrap_or_else(|p| p.into_inner()).poll_recv(cx);
        match polled {
            Poll::Ready(Some((mut item, bytes))) => {
                let c = &self.c;
                c.queued.fetch_sub(1, Ordering::Relaxed);
                c.queued_bytes.fetch_sub(bytes, Ordering::AcqRel);
                item.sequence = c.sequence.fetch_add(1, Ordering::Relaxed) + 1;
                c.forwarded.fetch_add(1, Ordering::Relaxed);
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                self.closed.store(true, Ordering::Release);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Drains the queue into StreamDocuments calls until it is closed and
/// empty, opening a new call after one ends in an error. A call that
/// fails ends as it fails, with or without items waiting.
async fn drain(endpoint: Endpoint, rx: Arc<Mutex<mpsc::Receiver<Queued>>>, c: Arc<Counters>) {
    let channel = endpoint.connect_lazy();
    let closed = Arc::new(AtomicBool::new(false));
    loop {
        let mut client = TurboSinkServiceClient::new(channel.clone());
        let feed = Feed { rx: rx.clone(), c: c.clone(), closed: closed.clone() };
        let ended = client.stream_documents(feed).await;
        {
            let mut last = c.last.lock().unwrap_or_else(|p| p.into_inner());
            match ended {
                Ok(summary) => last.1 = Some(summary.into_inner()),
                Err(e) => {
                    c.failed_calls.fetch_add(1, Ordering::Relaxed);
                    last.0 = Some(e.to_string());
                }
            }
        }
        if closed.load(Ordering::Acquire) {
            return;
        }
        tokio::time::sleep(RETRY).await;
    }
}

#[cfg(test)]
fn enqueue_gap() {
    if let Some(hook) = tests::ENQUEUE_GAP.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
        hook();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    use crate::proto::result::{AnalyzeFailure, AnalyzeVectors};

    pub(super) static ENQUEUE_GAP: Mutex<Option<Box<dyn Fn() + Send>>> = Mutex::new(None);

    struct GapGuard;
    impl Drop for GapGuard {
        fn drop(&mut self) {
            *ENQUEUE_GAP.lock().unwrap_or_else(|p| p.into_inner()) = None;
        }
    }

    fn answer(id: &str, failed: bool) -> AnalyzeStreamResponse {
        let outcome = match failed {
            true => Outcome::Failure(AnalyzeFailure::default()),
            false => Outcome::Vectors(AnalyzeVectors { dim: 1, data: vec![0; 4].into(), ..Default::default() }),
        };
        AnalyzeStreamResponse { id: id.into(), outcome: Some(outcome) }
    }

    fn config(items: usize, bytes: u64) -> SinkConfig {
        SinkConfig {
            endpoint: "http://127.0.0.1:1".into(),
            max_queue_items: items,
            max_queue_bytes: bytes,
            documents: false,
            typed: None,
        }
    }

    // On one thread the draining task cannot run between offers, so the
    // queue holds what its bounds let in and the rest is dropped.
    #[tokio::test(flavor = "current_thread")]
    async fn a_full_queue_drops_and_counts_without_waiting() {
        let f = SinkForwarder::start(&config(2, 1 << 20)).unwrap();
        for i in 0..5 {
            f.offer(&answer(&i.to_string(), false));
        }
        f.offer(&answer("failed", true));
        let s = f.stats();
        assert_eq!((s.offered, s.dropped, s.queued), (5, 3, 2), "{s:?}");
        assert!(s.queued_bytes > 0);
        let f = SinkForwarder::start(&config(16, 8)).unwrap();
        f.offer(&answer("too big", false));
        let s = f.stats();
        assert_eq!((s.offered, s.dropped, s.queued, s.queued_bytes), (1, 1, 0, 0), "{s:?}");
        assert!(SinkForwarder::start(&config(0, 8)).is_err());
        assert!(SinkForwarder::start(&SinkConfig { endpoint: "not a uri".into(), ..config(1, 1) }).is_err());
        let typed = SinkTyped { view: 1 << 16, offsets: crate::proto::result::OffsetMode::Utf8 };
        assert!(SinkForwarder::start(&SinkConfig { typed: Some(typed), ..config(1, 1) }).is_err());
    }

    /// close takes the sender under the same lock as enqueue's offered count
    /// and send. Pausing an offer after it has counted offered, then starting
    /// close, must still let that offer into the queue: close blocks on the
    /// lock, the send finishes, and the item is not counted dropped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn close_waits_out_an_in_flight_offer_instead_of_dropping_it() {
        let _gap = GapGuard;
        let f = SinkForwarder::start(&config(8, 1 << 20)).unwrap();
        let entered = Arc::new(Barrier::new(2));
        let finish = Arc::new(Barrier::new(2));
        {
            let entered = entered.clone();
            let finish = finish.clone();
            *ENQUEUE_GAP.lock().unwrap_or_else(|p| p.into_inner()) = Some(Box::new(move || {
                entered.wait();
                finish.wait();
            }));
        }
        let offer = {
            let f = f.clone();
            std::thread::spawn(move || f.offer(&answer("in-flight", false)))
        };
        entered.wait();
        assert_eq!(f.stats().offered, 1, "the offer counted before the send");
        let close = {
            let f = f.clone();
            tokio::spawn(async move { f.close(Duration::from_millis(400)).await })
        };
        // close is blocked on the sender lock held by the offer; release the
        // offer so it can send, then close takes the sender and drains.
        finish.wait();
        offer.join().unwrap();
        close.await.unwrap();
        let s = f.stats();
        assert_eq!(s.offered, 1, "{s:?}");
        assert_eq!(s.dropped, 0, "an in-flight offer must not be dropped by close: {s:?}");
        assert_eq!(s.forwarded + s.abandoned + s.queued, 1, "{s:?}");
    }

    /// Offers that finished before close took the sender stay in the queue
    /// for close to drain or abandon; they are not counted dropped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn offers_finished_before_close_are_not_dropped() {
        let f = SinkForwarder::start(&config(64, 1 << 20)).unwrap();
        let start = Arc::new(Barrier::new(8));
        let mut offers = Vec::new();
        for i in 0..8 {
            let f = f.clone();
            let start = start.clone();
            offers.push(std::thread::spawn(move || {
                start.wait();
                f.offer(&answer(&i.to_string(), false));
            }));
        }
        for o in offers {
            o.join().unwrap();
        }
        assert_eq!(f.stats().queued, 8);
        f.close(Duration::from_millis(400)).await;
        let s = f.stats();
        assert_eq!((s.offered, s.dropped, s.abandoned, s.queued), (8, 0, 8, 0), "{s:?}");
    }
}
