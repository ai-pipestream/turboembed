//! Two-lane priority session scheduler (`PRIORITY_INTERACTIVE` vs
//! `PRIORITY_BULK`, `#2346`) with bounded queue capacity, bounded starvation
//! protection, and pre-execution deadline cancellation.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::Notify;
use tonic::Status;
use tonic::metadata::MetadataMap;

use crate::api::{self, Failure, TURBO_E_BUSY, TURBO_E_INVALID_STATE};
use crate::deadline::Deadline;
use crate::metrics::Metrics;
use crate::state::{Loaded, Slot};

/// gRPC metadata header selecting interactive or bulk device scheduling.
pub const PRIORITY_HEADER: &str = "x-turbo-priority";

/// Maximum consecutive interactive grants while bulk work is waiting before
/// yielding one session slot to bulk (`#2346` starvation bound).
pub const MAX_INTERACTIVE_BURST: u32 = 8;

/// Maximum combined queued interactive and bulk waiters per model pool.
/// Excess callers are refused immediately with `TURBO_E_BUSY` so bursts cannot
/// grow unboundedly or starve Tokio's blocking thread pool.
pub const MAX_QUEUE_DEPTH: usize = 128;

/// Poll quantum while parked in `SessionPool` so external cancellations wake
/// promptly even without a wall-clock timeout.
const WAIT_SLICE: Duration = Duration::from_millis(5);

/// Window during which a deferred non-blocking bulk stream (`try_take_bulk`)
/// counts as waiting bulk work for starvation protection.
const BULK_DEFER_LEASE: Duration = Duration::from_millis(12);

/// Scheduling priority lane in front of device sessions (`#2346`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Priority {
    /// Latency-sensitive query embedding, `AnalyzeQuery`, and `RerankAndHighlight`.
    Interactive,
    /// Throughput-oriented background ingest (`AnalyzeStream`, bulk batches).
    Bulk,
}

impl Priority {
    /// Parses `x-turbo-priority` (`interactive` or `bulk`) from gRPC metadata.
    pub fn from_metadata(metadata: &MetadataMap) -> Result<Option<Priority>, Status> {
        let Some(raw) = metadata.get(PRIORITY_HEADER) else {
            return Ok(None);
        };
        let s = raw.to_str().map_err(|_| Status::invalid_argument("x-turbo-priority must be valid ASCII"))?;
        match s.trim() {
            "interactive" | "PRIORITY_INTERACTIVE" => Ok(Some(Priority::Interactive)),
            "bulk" | "PRIORITY_BULK" => Ok(Some(Priority::Bulk)),
            _ => Err(Status::invalid_argument("x-turbo-priority must be `interactive` or `bulk`")),
        }
    }
}

struct Waiter {
    ticket: u64,
    notify: Arc<Notify>,
}

struct PoolInner {
    idle: Vec<usize>,
    interactive_q: VecDeque<Waiter>,
    bulk_q: VecDeque<Waiter>,
    next_ticket: u64,
    interactive_streak: u32,
    bulk_deferred_until: Option<Instant>,
}

impl PoolInner {
    fn has_bulk_waiting(&self, now: Instant) -> bool {
        !self.bulk_q.is_empty() || self.bulk_deferred_until.is_some_and(|until| now < until)
    }

    fn total_queued(&self) -> usize {
        self.interactive_q.len() + self.bulk_q.len()
    }

    fn remove_ticket(&mut self, priority: Priority, ticket: u64) {
        match priority {
            Priority::Interactive => self.interactive_q.retain(|w| w.ticket != ticket),
            Priority::Bulk => self.bulk_q.retain(|w| w.ticket != ticket),
        }
    }

    fn wake_next(&self, now: Instant) {
        if self.idle.is_empty() {
            return;
        }
        if let Some(w) = self.interactive_q.front()
            && !(self.has_bulk_waiting(now) && self.interactive_streak >= MAX_INTERACTIVE_BURST)
        {
            w.notify.notify_one();
            return;
        }
        if let Some(w) = self.bulk_q.front() {
            w.notify.notify_one();
        }
    }
}

struct TicketGuard<'a> {
    loaded: &'a Loaded,
    priority: Priority,
    ticket: u64,
    active: bool,
}

impl TicketGuard<'_> {
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for TicketGuard<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut g = self.loaded.pool.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.remove_ticket(self.priority, self.ticket);
        g.wake_next(Instant::now());
        drop(g);
        self.loaded.metrics.record_cancelled_before_exec();
    }
}

/// Two-lane session pool backing a [`Loaded`] bundle (`#2346`).
pub(crate) struct SessionPool {
    inner: Mutex<PoolInner>,
}

impl SessionPool {
    pub(crate) fn new(slot_count: usize) -> Self {
        Self {
            inner: Mutex::new(PoolInner {
                idle: (0..slot_count).rev().collect(),
                interactive_q: VecDeque::new(),
                bulk_q: VecDeque::new(),
                next_ticket: 0,
                interactive_streak: 0,
                bulk_deferred_until: None,
            }),
        }
    }

    pub(crate) fn idle_len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).idle.len()
    }

    pub(crate) fn queue_depths(&self) -> (usize, usize) {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        (g.interactive_q.len(), g.bulk_q.len())
    }

    fn release(&self, index: usize) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.idle.push(index);
        g.wake_next(Instant::now());
    }
}

/// Checks `deadline` right before device execution or session acquisition; when
/// expired, increments `cancelled_before_exec` and aborts without touching the
/// device.
pub(crate) fn check_before_exec(deadline: &Deadline, metrics: &Metrics) -> api::Result<()> {
    if deadline.is_expired() {
        metrics.record_cancelled_before_exec();
        return Err(Failure::new(TURBO_E_INVALID_STATE, 0, "gRPC deadline exceeded"));
    }
    Ok(())
}

/// A session taken from its model's pool, returned when this is dropped.
pub struct Held {
    pub(crate) loaded: Arc<Loaded>,
    pub(crate) index: usize,
}

impl Held {
    pub(crate) fn slot(&self) -> &Slot {
        &self.loaded.slots[self.index]
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.loaded.pool.release(self.index);
    }
}

impl Loaded {
    /// Number of currently idle sessions in this model's pool.
    pub(crate) fn idle_len(&self) -> usize {
        self.pool.idle_len()
    }

    /// Current `(interactive_queued, bulk_queued)` waiter counts.
    pub(crate) fn queue_depths(&self) -> (usize, usize) {
        self.pool.queue_depths()
    }

    /// Immediate non-blocking session take (`Server::hold` and plain `ModelInfer`
    /// with no `x-turbo-priority` metadata): returns `TURBO_E_BUSY` at once when
    /// every session is held or priority waiters are already queued.
    pub(crate) fn take(self: &Arc<Loaded>) -> api::Result<Held> {
        let mut g = self.pool.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !g.interactive_q.is_empty() || !g.bulk_q.is_empty() {
            return Err(Failure::new(TURBO_E_BUSY, 0, "priority waiters are queued for this model"));
        }
        let index = g.idle.pop().ok_or_else(|| {
            Failure::new(TURBO_E_BUSY, 0, format!("all {} sessions of the model are held", self.slots.len()))
        })?;
        Ok(Held { loaded: self.clone(), index })
    }

    /// Non-blocking bulk acquisition for `AnalyzeStream`: yields to queued
    /// `Priority::Interactive` callers up to [`MAX_INTERACTIVE_BURST`] grants,
    /// then claims one slot so background ingest never starves.
    pub(crate) fn try_take_bulk(self: &Arc<Loaded>) -> api::Result<Held> {
        let mut g = self.pool.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.idle.is_empty() {
            return Err(Failure::new(
                TURBO_E_BUSY,
                0,
                format!("all {} sessions of the model are held", self.slots.len()),
            ));
        }
        if !g.interactive_q.is_empty() && g.interactive_streak < MAX_INTERACTIVE_BURST {
            g.bulk_deferred_until = Some(Instant::now() + BULK_DEFER_LEASE);
            g.wake_next(Instant::now());
            return Err(Failure::new(TURBO_E_BUSY, 0, "yielding session to interactive priority lane"));
        }
        let index = g.idle.pop().expect("checked non-empty");
        g.interactive_streak = 0;
        g.bulk_deferred_until = None;
        Ok(Held { loaded: self.clone(), index })
    }

    /// Acquires a session immediately when `priority` is `None`, or through the
    /// bounded two-lane async priority queue when `Some(priority)` is given.
    pub(crate) async fn take_or_acquire_async(
        self: &Arc<Loaded>,
        priority: Option<Priority>,
        deadline: &Deadline,
    ) -> api::Result<Held> {
        match priority {
            Some(p) => self.acquire_async(p, deadline).await,
            None => {
                check_before_exec(deadline, &self.metrics)?;
                self.take()
            }
        }
    }

    /// Synchronous wrapper around [`Loaded::acquire_async`] for blocking worker
    /// tasks (`AnalyzeQuery`, `RerankAndHighlight`).
    pub(crate) fn acquire(self: &Arc<Loaded>, priority: Priority, deadline: &Deadline) -> api::Result<Held> {
        tokio::runtime::Handle::current().block_on(self.acquire_async(priority, deadline))
    }

    /// Queues in `priority` until a session is granted or `deadline` expires,
    /// without holding a Tokio blocking thread while waiting. Refuses with
    /// `TURBO_E_BUSY` immediately when [`MAX_QUEUE_DEPTH`] is reached.
    pub(crate) async fn acquire_async(
        self: &Arc<Loaded>,
        priority: Priority,
        deadline: &Deadline,
    ) -> api::Result<Held> {
        check_before_exec(deadline, &self.metrics)?;
        let notify = Arc::new(Notify::new());
        let (ticket, mut guard) = {
            let mut g = self.pool.inner.lock().unwrap_or_else(|e| e.into_inner());
            let now = Instant::now();
            if !g.idle.is_empty() {
                let can_grant_now = match priority {
                    Priority::Interactive => {
                        g.interactive_q.is_empty()
                            && !(g.has_bulk_waiting(now) && g.interactive_streak >= MAX_INTERACTIVE_BURST)
                    }
                    Priority::Bulk => {
                        g.bulk_q.is_empty()
                            && (g.interactive_q.is_empty() || g.interactive_streak >= MAX_INTERACTIVE_BURST)
                    }
                };
                if can_grant_now {
                    let index = g.idle.pop().expect("checked non-empty");
                    match priority {
                        Priority::Interactive => g.interactive_streak = g.interactive_streak.saturating_add(1),
                        Priority::Bulk => {
                            g.interactive_streak = 0;
                            g.bulk_deferred_until = None;
                        }
                    }
                    g.wake_next(now);
                    return Ok(Held { loaded: self.clone(), index });
                }
            }
            if g.total_queued() >= MAX_QUEUE_DEPTH {
                return Err(Failure::new(
                    TURBO_E_BUSY,
                    0,
                    format!("priority queue depth reached limit {MAX_QUEUE_DEPTH}"),
                ));
            }
            let ticket = g.next_ticket;
            g.next_ticket = g.next_ticket.wrapping_add(1);
            let waiter = Waiter { ticket, notify: Arc::clone(&notify) };
            match priority {
                Priority::Interactive => g.interactive_q.push_back(waiter),
                Priority::Bulk => g.bulk_q.push_back(waiter),
            }
            (ticket, TicketGuard { loaded: self, priority, ticket, active: true })
        };

        loop {
            let sleep_dur = {
                let mut g = self.pool.inner.lock().unwrap_or_else(|e| e.into_inner());
                if deadline.is_expired() {
                    g.remove_ticket(priority, ticket);
                    guard.disarm();
                    g.wake_next(Instant::now());
                    drop(g);
                    self.metrics.record_cancelled_before_exec();
                    return Err(Failure::new(TURBO_E_INVALID_STATE, 0, "gRPC deadline exceeded"));
                }

                let now = Instant::now();
                if !g.idle.is_empty() {
                    let can_grant = match priority {
                        Priority::Interactive => {
                            g.interactive_q.front().is_some_and(|w| w.ticket == ticket)
                                && !(g.has_bulk_waiting(now) && g.interactive_streak >= MAX_INTERACTIVE_BURST)
                        }
                        Priority::Bulk => {
                            g.bulk_q.front().is_some_and(|w| w.ticket == ticket)
                                && (g.interactive_q.is_empty() || g.interactive_streak >= MAX_INTERACTIVE_BURST)
                        }
                    };
                    if can_grant {
                        g.remove_ticket(priority, ticket);
                        guard.disarm();
                        let index = g.idle.pop().expect("checked non-empty");
                        match priority {
                            Priority::Interactive => g.interactive_streak = g.interactive_streak.saturating_add(1),
                            Priority::Bulk => {
                                g.interactive_streak = 0;
                                g.bulk_deferred_until = None;
                            }
                        }
                        g.wake_next(now);
                        return Ok(Held { loaded: self.clone(), index });
                    }
                }

                let mut dur = match deadline.stop_at() {
                    Some(stop) => {
                        let tokio_now = tokio::time::Instant::now();
                        if tokio_now >= stop { Duration::ZERO } else { (stop - tokio_now).min(WAIT_SLICE) }
                    }
                    None => WAIT_SLICE,
                };
                if priority == Priority::Interactive
                    && g.interactive_streak >= MAX_INTERACTIVE_BURST
                    && let Some(until) = g.bulk_deferred_until
                    && until > now
                {
                    dur = dur.min(until - now);
                }
                dur
            };

            if sleep_dur.is_zero() {
                tokio::task::yield_now().await;
                continue;
            }
            tokio::select! {
                _ = notify.notified() => {}
                _ = tokio::time::sleep(sleep_dur) => {}
            }
        }
    }
}
