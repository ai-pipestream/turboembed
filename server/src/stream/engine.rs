//! One stream's scheduler (issue #722): admits the reader's documents,
//! gathers documents waiting on the same load into batches up to the
//! bundle's batch limit, runs them on held sessions, and answers in the
//! header's order, input order or completion order. Every admitted
//! document, a failure included, holds its queue room and window permit
//! until its answer is sent, or the stream dies and drops it all.
//! Waiting for a session never blocks the loop: a batch whose model's
//! sessions are all held waits in `stalled` and retries on a timer
//! while the loop keeps servicing completions, new work and
//! cancellation.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::Instant;
use tonic::Status;

use crate::api::Failure;
use crate::proto::result::{AnalyzeOrder, AnalyzeStreamResponse};
use crate::stream::run::{self, Doc, Outcome, Whole};
use crate::stream::{BatchKey, Dead, Envelope, Room, Shared};

/// The header as the scheduler holds it.
#[derive(Debug)]
pub(crate) struct Config {
    pub(crate) order: AnalyzeOrder,
    pub(crate) window: usize,
    pub(crate) linger: Duration,
}

/// How often a stalled batch retries its model's session pool.
const SESSION_RETRY: Duration = Duration::from_millis(2);

/// A queued document, the queue room it holds, and when it arrived.
type Queued = (Doc, Instant);

/// One answered document still holding what it was admitted with: the
/// response to send, the queue room, and the window permit a dispatched
/// document holds. Both free where the answer frees.
struct Pending {
    response: AnalyzeStreamResponse,
    room: Option<Room>,
    window: Option<OwnedSemaphorePermit>,
}

/// A batch that has its window permits but no session yet.
type Stalled = (Vec<Doc>, Vec<OwnedSemaphorePermit>);

pub(crate) struct Engine {
    shared: Arc<Shared>,
    config: Config,
    window_sem: Arc<Semaphore>,
    /// Documents admitted, not yet dispatched.
    pending: VecDeque<Queued>,
    /// Batches running on held sessions.
    inflight: JoinSet<Outcome>,
    /// The next sequence number to assign, and the next one an
    /// input-ordered stream may answer with.
    next_seq: u64,
    next_send: u64,
    /// Input-order answers waiting on an earlier one.
    buffer: BTreeMap<u64, Pending>,
    /// Batches waiting for a session, retried on the timer.
    stalled: VecDeque<Stalled>,
    /// When the head of `stalled` started waiting: the retry deadline
    /// stays put across loop iterations, or a ready branch that never
    /// yields would starve its timer.
    stalled_at: Option<Instant>,
}

impl Engine {
    pub(crate) fn new(shared: Arc<Shared>, config: Config) -> Self {
        let window = config.window;
        Self {
            shared,
            config,
            window_sem: Arc::new(Semaphore::new(window)),
            pending: VecDeque::new(),
            inflight: JoinSet::new(),
            next_seq: 0,
            next_send: 0,
            buffer: BTreeMap::new(),
            stalled: VecDeque::new(),
            stalled_at: None,
        }
    }

    /// The schedule loop. Returns when the input closed and every
    /// document is answered, or when the stream dies: `dead` tells the
    /// reader to stop admitting, and dropping the queue, the buffer,
    /// the stalled batches and the JoinSet frees every room and window
    /// permit with them.
    pub(crate) async fn run(
        mut self,
        mut work: mpsc::Receiver<Envelope>,
        answers: mpsc::Sender<std::result::Result<AnalyzeStreamResponse, Status>>,
        dead: Arc<Dead>,
    ) {
        let mut input_ended = false;
        // The two timers live across the loop and are reset each turn;
        // a branch whose deadline is None is disabled, so a far reset
        // never fires.
        let far = || Instant::now() + Duration::from_secs(86400);
        let mut sleeper = std::pin::pin!(tokio::time::sleep_until(far()));
        let mut stalled_retry = std::pin::pin!(tokio::time::sleep_until(far()));
        loop {
            let deadline = self.deadline(input_ended);
            sleeper.as_mut().reset(deadline.unwrap_or_else(far));
            let stalled_deadline = self.stalled_at.map(|at| at + SESSION_RETRY);
            stalled_retry.as_mut().reset(stalled_deadline.unwrap_or_else(far));
            tokio::select! {
                env = work.recv() => match env {
                    Some(Envelope::Doc(mut doc)) => {
                        doc.seq = self.next_seq;
                        self.next_seq += 1;
                        self.pending.push_back((doc, Instant::now()));
                    }
                    Some(Envelope::Failed { id, failure, room }) => {
                        let seq = self.next_seq;
                        self.next_seq += 1;
                        let pending = Pending { response: failure_response(id, failure), room, window: None };
                        if self.enqueue(seq, pending, &answers).await.is_err() {
                            break;
                        }
                    }
                    Some(Envelope::Abort(status)) => {
                        let _ = answers.send(Err(status)).await;
                        break;
                    }
                    None => input_ended = true,
                },
                outcome = self.inflight.join_next(), if !self.inflight.is_empty() => match outcome {
                    Some(Ok(outcome)) => {
                        if self.deliver(outcome, &answers).await.is_err() {
                            break;
                        }
                    }
                    Some(Err(e)) => {
                        let _ = answers.send(Err(Status::internal(format!("a batch's task failed: {e}")))).await;
                        break;
                    }
                    None => {}
                },
                _ = sleeper.as_mut(), if deadline.is_some() => self.dispatch(true, &dead).await,
                // A batch whose sessions are all held waits here and
                // retries without ever blocking this loop.
                _ = stalled_retry.as_mut(), if stalled_deadline.is_some() => {
                    self.stalled_at = None;
                    self.retry_stalled(&dead).await;
                },
                // The client went away: drop the queue and the batches.
                _ = answers.closed() => break,
            }
            self.dispatch(input_ended, &dead).await;
            // Let the runtime breathe: with an instantly ready branch
            // this loop would otherwise spin without ever polling the
            // timer that drives the linger and the stalled retries.
            tokio::task::yield_now().await;
            if input_ended
                && self.pending.is_empty()
                && self.inflight.is_empty()
                && self.buffer.is_empty()
                && self.stalled.is_empty()
            {
                break;
            }
        }
        dead.set();
    }

    /// When the next dispatch is due: the linger after the oldest
    /// queued document, while more may still arrive. No deadline runs
    /// while no window permit is free: nothing could dispatch, and the
    /// next completion or send re-enters the loop and dispatches anyway.
    fn deadline(&self, input_ended: bool) -> Option<Instant> {
        if input_ended || self.pending.is_empty() || self.window_sem.available_permits() == 0 {
            return None;
        }
        self.pending.front().map(|(_, at)| *at + self.config.linger)
    }

    /// Runs every dispatchable batch: pending documents grouped by load
    /// and options, each batch up to the bundle's batch limit and the
    /// window permits actually free. A dispatched document holds its
    /// permit until its answer is sent, so under input order an answered
    /// document's permit stays parked in the ordered buffer while an
    /// earlier one is still pending: sizing by the configured window
    /// alone would ask for permits that park there and deadlock the
    /// stream. `force` dispatches past the linger, for the
    /// linger's own deadline and the input's half-close. Batches whose
    /// sessions are all held go to `stalled`, never to a wait here.
    async fn dispatch(&mut self, force: bool, dead: &Arc<Dead>) {
        if self.pending.is_empty() || self.window_sem.available_permits() == 0 {
            return;
        }
        if !force && Instant::now() < self.deadline(false).unwrap() {
            return;
        }
        let mut groups: Vec<(BatchKey, Vec<Queued>)> = {
            let mut by_key: BTreeMap<BatchKey, Vec<Queued>> = BTreeMap::new();
            for queued in self.pending.drain(..) {
                by_key.entry(run::key(&queued.0)).or_default().push(queued);
            }
            by_key.into_iter().collect()
        };
        // Dispatch in the order documents arrived: the group holding
        // the earliest pending sequence first. Under input order a
        // buffered response holds its window permit until every earlier
        // sequence is answered, so a later sequence must never take the
        // last permit while an earlier one waits for it.
        groups.sort_by_key(|(_, docs)| docs[0].0.seq);
        // Each group takes at most a share of the window this round, so
        // one load's backlog never stands between another load's
        // documents and the window.
        let share = (self.config.window / groups.len()).max(1);
        let mut leftover: Vec<Queued> = Vec::new();
        let mut iter = groups.into_iter();
        while let Some((_, mut docs)) = iter.next() {
            let n = docs.len().min(docs[0].0.batch_limit as usize).min(share).min(self.window_sem.available_permits());
            if n == 0 {
                leftover.append(&mut docs);
                continue;
            }
            let batch: Vec<Queued> = docs.drain(..n).collect();
            match self.spawn(batch, dead).await {
                Ok(()) => leftover.extend(docs),
                Err(batch) => {
                    // The window shut mid-dispatch: the batch, the rest
                    // of its group, and every group still unvisited go
                    // back to the queue, nothing dropped.
                    leftover.extend(batch);
                    leftover.extend(docs);
                    leftover.extend(iter.flat_map(|(_, rest)| rest));
                    break;
                }
            }
        }
        self.pending = leftover.into();
    }

    /// Retries every stalled batch once, in turn.
    async fn retry_stalled(&mut self, dead: &Arc<Dead>) {
        for _ in 0..self.stalled.len() {
            let Some((docs, permits)) = self.stalled.pop_front() else { return };
            self.respawn(docs, permits, dead).await;
        }
    }

    /// Starts one batch: window permits first, then a session. On a shut
    /// window the documents come back to stay queued; on a busy pool they
    /// wait on `stalled` holding their permits.
    async fn spawn(&mut self, batch: Vec<Queued>, dead: &Arc<Dead>) -> std::result::Result<(), Vec<Queued>> {
        let n = batch.len();
        let mut permits = Vec::with_capacity(n);
        for _ in 0..n {
            match self.window_sem.clone().try_acquire_owned() {
                Ok(p) => permits.push(p),
                Err(_) => {
                    drop(permits);
                    return Err(batch);
                }
            }
        }
        let (docs, _ats): (Vec<Doc>, Vec<Instant>) = batch.into_iter().unzip();
        self.respawn(docs, permits, dead).await;
        Ok(())
    }

    /// Runs one batch that already holds its window permits and its
    /// count. A busy session pool requeues the batch on `stalled`.
    async fn respawn(&mut self, docs: Vec<Doc>, permits: Vec<OwnedSemaphorePermit>, dead: &Arc<Dead>) {
        if dead.is_set() {
            self.stalled.push_back((docs, permits));
            return;
        }
        let Some(loaded) = docs.first().map(|d| d.loaded.clone()) else { return };
        match loaded.take() {
            Ok(held) => {
                let opts = docs[0].opts;
                let shared = self.shared.clone();
                self.inflight.spawn_blocking(move || run::run(&shared, held, docs, &opts, permits));
            }
            Err(_) => self.stash(docs, permits),
        }
    }

    /// A batch waits for a session, retried from the timer.
    fn stash(&mut self, docs: Vec<Doc>, permits: Vec<OwnedSemaphorePermit>) {
        if self.stalled_at.is_none() {
            self.stalled_at = Some(Instant::now());
        }
        self.stalled.push_back((docs, permits));
    }

    /// Answers one batch. A batch that failed whole is retried one
    /// document at a time, so a failing document answers with its own
    /// error and the others keep their own results.
    async fn deliver(
        &mut self,
        outcome: Outcome,
        answers: &mpsc::Sender<std::result::Result<AnalyzeStreamResponse, Status>>,
    ) -> std::result::Result<(), ()> {
        let Outcome { model, version, dim, rows } = outcome;
        match rows {
            Ok(rows) => {
                for row in rows {
                    let vectors = crate::proto::result::AnalyzeVectors {
                        model: model.clone(),
                        version: version.clone(),
                        dim,
                        data: row.data,
                    };
                    let pending = Pending {
                        response: ok_response(row.id, vectors),
                        room: Some(row.room),
                        window: Some(row.window),
                    };
                    if self.enqueue(row.seq, pending, answers).await.is_err() {
                        return Err(());
                    }
                }
                Ok(())
            }
            Err(Whole { failure, docs, mut permits }) if docs.len() == 1 => {
                let doc = docs.into_iter().next().expect("one document");
                let pending = Pending {
                    response: failure_response(doc.id, failure),
                    room: Some(doc.room),
                    window: permits.pop(),
                };
                self.enqueue(doc.seq, pending, answers).await
            }
            Err(Whole { failure: _, docs, permits }) => {
                for (doc, permit) in docs.into_iter().zip(permits) {
                    match doc.loaded.take() {
                        Ok(held) => {
                            let opts = doc.opts;
                            let shared = self.shared.clone();
                            self.inflight
                                .spawn_blocking(move || run::run(&shared, held, vec![doc], &opts, vec![permit]));
                        }
                        Err(_) => self.stash(vec![doc], vec![permit]),
                    }
                }
                Ok(())
            }
        }
    }

    /// Queues one document's answer: input order buffers it, room and
    /// window permit attached, until the documents before it are
    /// answered; completion order answers at once. Either way the
    /// response's permits free only once it is sent.
    async fn enqueue(
        &mut self,
        seq: u64,
        pending: Pending,
        answers: &mpsc::Sender<std::result::Result<AnalyzeStreamResponse, Status>>,
    ) -> std::result::Result<(), ()> {
        match self.config.order {
            AnalyzeOrder::Completion => self.send(pending, answers).await,
            _ => {
                self.buffer.insert(seq, pending);
                while let Some(p) = self.buffer.remove(&self.next_send) {
                    self.next_send += 1;
                    self.send(p, answers).await?;
                }
                Ok(())
            }
        }
    }

    /// Sends one answer; its room and window permit free with the send.
    async fn send(
        &self,
        pending: Pending,
        answers: &mpsc::Sender<std::result::Result<AnalyzeStreamResponse, Status>>,
    ) -> std::result::Result<(), ()> {
        answers.send(Ok(pending.response)).await.map_err(|_| ())?;
        drop(pending.room);
        drop(pending.window);
        Ok(())
    }
}

/// The response for one document's vectors.
fn ok_response(id: String, vectors: crate::proto::result::AnalyzeVectors) -> AnalyzeStreamResponse {
    AnalyzeStreamResponse {
        id,
        outcome: Some(crate::proto::result::analyze_stream_response::Outcome::Vectors(vectors)),
    }
}

/// The response for one document's failure: the turbo status, field and
/// refusal the single-document call would carry.
fn failure_response(id: String, failure: Failure) -> AnalyzeStreamResponse {
    AnalyzeStreamResponse {
        id,
        outcome: Some(crate::proto::result::analyze_stream_response::Outcome::Failure(
            crate::proto::result::AnalyzeFailure {
                turbo_code: failure.code,
                field: failure.field,
                message: crate::status::describe(&failure, &crate::status::EMBED_OPTIONS),
            },
        )),
    }
}

/// A running engine, started by the handler.
pub(crate) fn start(
    shared: Arc<Shared>,
    config: Config,
    work: mpsc::Receiver<Envelope>,
    answers: mpsc::Sender<std::result::Result<AnalyzeStreamResponse, Status>>,
    dead: Arc<Dead>,
) -> JoinHandle<()> {
    tokio::spawn(Engine::new(shared, config).run(work, answers, dead))
}
