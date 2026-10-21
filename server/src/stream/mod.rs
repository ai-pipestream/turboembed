//! The streaming bulk document analysis call (issue #722,
//! proto/turbo_analysis.proto): reader, scheduler and batch executor of
//! one AnalyzeStream. The reader validates documents and admits them
//! through the advertised queue caps; the scheduler gathers documents
//! waiting on the same model into batches up to the bundle's batch
//! limit and answers in the header's order; the executor runs one
//! batch on one held session. Cancelling the stream drops what was
//! queued, never what a session is already running.

mod engine;
pub(crate) mod handler;
mod reader;
pub(crate) mod run;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::Status;

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::StreamLimits;
use crate::status::to_status;
use crate::stream::run::Doc;

/// The batching key: the load's identity, then the options. Documents
/// batch together only under one key.
pub(crate) type BatchKey = (usize, u32, u32, u32, u32, u32, u32);

/// What the reader hands the engine for one stream message: an
/// admitted document holding its queue room, a document that failed
/// validation or admission holding the room it was admitted with (the
/// room frees when its answer is sent), or the stream-ending refusal
/// for a misplaced header or a reset.
pub(crate) enum Envelope {
    Doc(Doc),
    Failed { id: String, failure: Failure, room: Option<Room> },
    Abort(Status),
}

/// Documents one stream may have queued, not yet executing; the
/// capability listing shows it. A send past it waits for the queue to
/// drain, never a mid-stream error.
pub const MAX_QUEUED_ITEMS_PER_STREAM: usize = 256;

/// Document bytes one stream may have queued.
pub const MAX_QUEUED_BYTES_PER_STREAM: usize = 4 << 20;

/// Documents all streams may have queued in all.
pub const MAX_QUEUED_ITEMS_SERVER: usize = 4096;

/// Document bytes all streams may have queued in all.
pub const MAX_QUEUED_BYTES_SERVER: usize = 64 << 20;

/// Documents one stream may have executing or awaiting their answer at
/// once; the most a header's max_inflight may ask for.
pub const MAX_INFLIGHT_ITEMS_PER_STREAM: usize = 64;

/// The wait that gathers one model's documents into a batch, when the
/// header does not say.
pub const DEFAULT_BATCH_LINGER: Duration = Duration::from_millis(2);

/// The most a header's batch_linger_ms may ask for.
pub const MAX_BATCH_LINGER: Duration = Duration::from_secs(60);

/// What the streaming call may queue, as the capability listing shows
/// it: the same constants the flow control enforces.
pub(crate) fn stream_limits() -> StreamLimits {
    StreamLimits {
        max_queued_items_per_stream: MAX_QUEUED_ITEMS_PER_STREAM as u64,
        max_queued_bytes_per_stream: MAX_QUEUED_BYTES_PER_STREAM as u64,
        max_queued_items_server: MAX_QUEUED_ITEMS_SERVER as u64,
        max_queued_bytes_server: MAX_QUEUED_BYTES_SERVER as u64,
        max_inflight_items_per_stream: MAX_INFLIGHT_ITEMS_PER_STREAM as u64,
        default_batch_linger_ms: DEFAULT_BATCH_LINGER.as_millis() as u64,
        max_batch_linger_ms: MAX_BATCH_LINGER.as_millis() as u64,
    }
}

/// A header knob over what ServerLimits.stream advertises, refusing with
/// the limit named, before anything is allocated.
pub(crate) fn capped(name: &str, asked: u32, max: u64) -> Result<u64, Status> {
    let asked = asked as u64;
    if asked > max {
        return Err(to_status(
            &Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("{name} is {asked}, over the {max} limit")),
            &[],
        ));
    }
    Ok(asked)
}

/// The server-wide part of the streaming call's state, shared by every
/// stream of one server: the queue caps as semaphores and the gauges
/// the operator and the tests read.
pub(crate) struct Shared {
    /// Documents all streams may have queued in all.
    pub(crate) queued_items: Arc<Semaphore>,

    /// Document bytes all streams may have queued in all.
    pub(crate) queued_bytes: Arc<Semaphore>,

    /// Model executions since the server started, the count the
    /// cancellation and batching proofs read.
    pub(crate) runs: AtomicU64,

    /// Documents admitted and not yet answered across every stream,
    /// right now: queued, executing and buffered for ordering.
    pub(crate) queued_item_count: AtomicUsize,

    /// Their bytes, right now.
    pub(crate) queued_byte_count: AtomicUsize,
}

impl Shared {
    pub(crate) fn new() -> Arc<Shared> {
        Arc::new(Shared {
            queued_items: Arc::new(Semaphore::new(MAX_QUEUED_ITEMS_SERVER)),
            queued_bytes: Arc::new(Semaphore::new(MAX_QUEUED_BYTES_SERVER)),
            runs: AtomicU64::new(0),
            queued_item_count: AtomicUsize::new(0),
            queued_byte_count: AtomicUsize::new(0),
        })
    }
}

/// The gauges `Server::stream_stats` answers: what is admitted and not
/// yet answered, and how many model executions have run, at the moment
/// of the read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Model executions since the server started.
    pub runs: u64,

    /// Documents admitted and not yet answered across every stream:
    /// queued, executing, and buffered for ordering.
    pub queued_items: usize,

    /// Their bytes.
    pub queued_bytes: usize,
}

impl Shared {
    pub(crate) fn stats(&self) -> Stats {
        Stats {
            runs: self.runs.load(Ordering::Relaxed),
            queued_items: self.queued_item_count.load(Ordering::Relaxed),
            queued_bytes: self.queued_byte_count.load(Ordering::Relaxed),
        }
    }
}

/// The queue room one admitted document holds: the per-stream and
/// server-wide item and byte permits. Dropping it hands the room back,
/// so a cancelled stream's work is freed with its documents, and a
/// answered document's room is freed for the next one.
pub(crate) struct Room {
    _server_items: OwnedSemaphorePermit,
    _server_bytes: OwnedSemaphorePermit,
    _stream_items: OwnedSemaphorePermit,
    _stream_bytes: OwnedSemaphorePermit,
    shared: Arc<Shared>,
    bytes: usize,
}

impl Drop for Room {
    fn drop(&mut self) {
        self.shared.queued_item_count.fetch_sub(1, Ordering::Relaxed);
        self.shared.queued_byte_count.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// A document past a byte cap refuses with the limit named, before any
/// room is taken for what is not admitted.
pub(crate) fn over_bytes(bytes: usize, limit: &'static str) -> Failure {
    crate::results::over(limit, "document bytes", bytes.min(u64::MAX as usize) as u64)
}

/// The byte cap a document of `bytes` length is over, if any.
pub(crate) fn over_byte_cap(bytes: usize) -> Option<(&'static str, usize)> {
    if bytes > MAX_QUEUED_BYTES_PER_STREAM {
        Some(("max_queued_bytes_per_stream", MAX_QUEUED_BYTES_PER_STREAM))
    } else if bytes > MAX_QUEUED_BYTES_SERVER {
        Some(("max_queued_bytes_server", MAX_QUEUED_BYTES_SERVER))
    } else {
        None
    }
}

/// Admits one document of `bytes` length: the server-wide caps first,
/// then the stream's, in one order everywhere, so two admits never
/// deadlock. The byte semaphores reserve the document's own byte count,
/// so byte pressure blocks admission even while item permits remain;
/// a document over a byte cap refuses before it is admitted. The
/// permits come back with the returned room's drop.
pub(crate) async fn admit(
    shared: Arc<Shared>,
    stream_items: Arc<Semaphore>,
    stream_bytes: Arc<Semaphore>,
    bytes: usize,
) -> Room {
    let server_items = shared.queued_items.clone().acquire_owned().await.expect("queue semaphore open");
    let server_bytes =
        shared.queued_bytes.clone().acquire_many_owned(bytes.max(1) as u32).await.expect("queue semaphore open");
    let stream_items = stream_items.acquire_owned().await.expect("queue semaphore open");
    let stream_bytes = stream_bytes.acquire_many_owned(bytes.max(1) as u32).await.expect("queue semaphore open");
    shared.queued_item_count.fetch_add(1, Ordering::Relaxed);
    shared.queued_byte_count.fetch_add(bytes, Ordering::Relaxed);
    Room {
        _server_items: server_items,
        _server_bytes: server_bytes,
        _stream_items: stream_items,
        _stream_bytes: stream_bytes,
        shared,
        bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_listing_shows_the_same_caps_the_flow_control_enforces() {
        let l = stream_limits();
        assert_eq!(l.max_queued_items_per_stream as usize, MAX_QUEUED_ITEMS_PER_STREAM);
        assert_eq!(l.max_queued_bytes_per_stream as usize, MAX_QUEUED_BYTES_PER_STREAM);
        assert_eq!(l.max_queued_items_server as usize, MAX_QUEUED_ITEMS_SERVER);
        assert_eq!(l.max_queued_bytes_server as usize, MAX_QUEUED_BYTES_SERVER);
        assert_eq!(l.max_inflight_items_per_stream as usize, MAX_INFLIGHT_ITEMS_PER_STREAM);
        assert_eq!(l.default_batch_linger_ms, DEFAULT_BATCH_LINGER.as_millis() as u64);
        assert_eq!(l.max_batch_linger_ms, MAX_BATCH_LINGER.as_millis() as u64);
    }

    #[test]
    fn a_header_knob_over_the_listed_cap_names_it() {
        let e = capped("max_inflight", MAX_INFLIGHT_ITEMS_PER_STREAM as u32 + 1, MAX_INFLIGHT_ITEMS_PER_STREAM as u64)
            .unwrap_err();
        assert!(e.message().contains("max_inflight"), "{e:?}");
        assert_eq!(capped("max_inflight", 0, 64).unwrap(), 0);
        assert_eq!(capped("max_inflight", 64, 64).unwrap(), 64);
    }
}
