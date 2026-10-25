//! Forwarding analyzed documents to a downstream TurboSinkService
//! (proto/turbo_sink.proto, docs/grpc-sink.md). Every answer an analysis
//! stream completes is offered to a bounded queue without waiting: when
//! the queue is full, by items or by bytes, the item is dropped and
//! counted, so a slow sink never holds back the stream that produced it.
//! One task drains the queue into a long-lived StreamDocuments call,
//! opening a new one after a failure.

mod receiver;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use prost::Message;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
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
    /// Handed to a StreamDocuments call. The sink's summary says what it
    /// accepted.
    pub forwarded: u64,
    /// Refused by a full queue.
    pub dropped: u64,
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
    queued: AtomicU64,
    queued_bytes: AtomicU64,
    failed_calls: AtomicU64,
    last: Mutex<(Option<String>, Option<DocumentSinkSummary>)>,
}

/// The tee every analysis stream offers its answers to.
pub(crate) struct SinkForwarder {
    tx: Mutex<Option<mpsc::Sender<(DocumentSinkItem, u64)>>>,
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
        let counters = Arc::new(Counters::default());
        let task = tokio::spawn(drain(endpoint, rx, counters.clone()));
        Ok(Arc::new(SinkForwarder {
            tx: Mutex::new(Some(tx)),
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
        let item = DocumentSinkItem {
            sequence: c.sequence.fetch_add(1, Ordering::Relaxed) + 1,
            doc_id: answer.id.clone(),
            analyzed: Some(answer.clone()),
            ..Default::default()
        };
        let bytes = item.encoded_len() as u64;
        let fits = c
            .queued_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |q| (q + bytes <= self.max_bytes).then_some(q + bytes))
            .is_ok();
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
            queued: c.queued.load(Ordering::Relaxed),
            queued_bytes: c.queued_bytes.load(Ordering::Relaxed),
            failed_calls: c.failed_calls.load(Ordering::Relaxed),
            last_error,
            last_summary,
        }
    }

    /// Closes the queue and waits up to `within` for what it holds to be
    /// sent and the sink's summary to arrive.
    pub(crate) async fn close(&self, within: Duration) {
        self.tx.lock().unwrap_or_else(|p| p.into_inner()).take();
        let task = self.task.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(t) = task {
            let _ = tokio::time::timeout(within, t).await;
        }
    }
}

/// Drains the queue into StreamDocuments calls until it is closed and
/// empty, opening a new call after one ends in an error.
async fn drain(endpoint: Endpoint, mut rx: mpsc::Receiver<(DocumentSinkItem, u64)>, c: Arc<Counters>) {
    let channel = endpoint.connect_lazy();
    loop {
        let (call_tx, call_rx) = mpsc::channel::<DocumentSinkItem>(64);
        let mut client = TurboSinkServiceClient::new(channel.clone());
        let call = tokio::spawn(async move { client.stream_documents(ReceiverStream::new(call_rx)).await });
        let mut closed = true;
        while let Some((item, bytes)) = rx.recv().await {
            c.queued.fetch_sub(1, Ordering::Relaxed);
            c.queued_bytes.fetch_sub(bytes, Ordering::AcqRel);
            if call_tx.send(item).await.is_err() {
                // The call ended under it: this item is lost with it.
                c.dropped.fetch_add(1, Ordering::Relaxed);
                closed = false;
                break;
            }
            c.forwarded.fetch_add(1, Ordering::Relaxed);
        }
        drop(call_tx);
        let ended = call.await.map_err(|e| e.to_string()).and_then(|r| r.map_err(|s| s.to_string()));
        {
            let mut last = c.last.lock().unwrap_or_else(|p| p.into_inner());
            match ended {
                Ok(summary) => last.1 = Some(summary.into_inner()),
                Err(e) => {
                    c.failed_calls.fetch_add(1, Ordering::Relaxed);
                    last.0 = Some(e);
                }
            }
        }
        if closed {
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
            false => Outcome::Vectors(AnalyzeVectors { dim: 1, data: vec![0; 4], ..Default::default() }),
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
