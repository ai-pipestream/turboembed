//! Forwarding analyzed documents to a downstream TurboSinkService
//! (proto/turbo_sink.proto, docs/grpc-sink.md). Every answer an analysis
//! stream completes is offered to a bounded queue without waiting: when
//! the queue is full, by items or by bytes, the item is dropped and
//! counted, so a slow sink never holds back the stream that produced it.
//! One task drains the queue into a long-lived StreamDocuments call,
//! opening a new one after a failure. The call takes each item from the
//! queue itself, as it sends it, so what waits is the queue's and counted
//! there, and the sequence follows the order the items are sent in.

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

pub use receiver::{MAX_SUMMARY_ERRORS, SinkReceiver};

use crate::proto::result::AnalyzeStreamResponse;
use crate::proto::result::analyze_stream_response::Outcome;
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
        let (tx, rx) = mpsc::channel(config.max_queue_items);
        let rx = Arc::new(Mutex::new(rx));
        let counters = Arc::new(Counters::default());
        let task = tokio::spawn(drain(endpoint, rx.clone(), counters.clone()));
        Ok(Arc::new(SinkForwarder {
            tx: Mutex::new(Some(tx)),
            rx,
            max_bytes: config.max_queue_bytes,
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
        let c = &self.counters;
        c.offered.fetch_add(1, Ordering::Relaxed);
        // The sequence is set as the item is sent; measured at its widest
        // here, so the queue's byte count is never under what it sends.
        let item = DocumentSinkItem {
            sequence: u64::MAX,
            doc_id: answer.id.clone(),
            analyzed: Some(answer.clone()),
            ..Default::default()
        };
        let bytes = item.encoded_len() as u64;
        let fits = reserve(&c.queued_bytes, bytes, self.max_bytes);
        let sent = fits
            && self
                .tx
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
                .is_some_and(|tx| tx.try_send((item, bytes)).is_ok());
        if sent {
            c.queued.fetch_add(1, Ordering::Relaxed);
        } else {
            if fits {
                c.queued_bytes.fetch_sub(bytes, Ordering::AcqRel);
            }
            c.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn stats(&self) -> SinkStats {
        let c = &self.counters;
        let (last_error, last_summary) = c.last.lock().unwrap_or_else(|p| p.into_inner()).clone();
        SinkStats {
            offered: c.offered.load(Ordering::Relaxed),
            forwarded: c.forwarded.load(Ordering::Relaxed),
            dropped: c.dropped.load(Ordering::Relaxed),
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
mod tests {
    use super::*;
    use crate::proto::result::{AnalyzeFailure, AnalyzeVectors};

    fn answer(id: &str, failed: bool) -> AnalyzeStreamResponse {
        let outcome = match failed {
            true => Outcome::Failure(AnalyzeFailure::default()),
            false => Outcome::Vectors(AnalyzeVectors { dim: 1, data: vec![0; 4].into(), ..Default::default() }),
        };
        AnalyzeStreamResponse { id: id.into(), outcome: Some(outcome) }
    }

    // On one thread the draining task cannot run between offers, so the
    // queue holds what its bounds let in and the rest is dropped.
    #[tokio::test(flavor = "current_thread")]
    async fn a_full_queue_drops_and_counts_without_waiting() {
        let config = |items, bytes| SinkConfig {
            endpoint: "http://127.0.0.1:1".into(),
            max_queue_items: items,
            max_queue_bytes: bytes,
        };
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
    }
}
