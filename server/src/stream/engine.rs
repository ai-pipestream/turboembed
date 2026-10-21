//! One stream's scheduler (issue #722): admits the reader's documents,
//! gathers documents waiting on the same model into batches up to the
//! bundle's batch limit, runs them on held sessions, and answers in the
//! header's order, input order or completion order. Queued documents
//! hold their queue room until a batch takes them; cancelling the
//! stream drops the queue with its rooms, and a batch already running
//! on a session finishes but its answer goes nowhere.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{Semaphore, mpsc};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::Instant;
use tonic::Status;

use crate::api::Failure;
use crate::proto::result::{AnalyzeOrder, AnalyzeStreamResponse};
use crate::stream::run::{self, Doc, Outcome};
use crate::stream::{Envelope, Room, Shared};

/// The batching key: the same model, the same options.
pub(crate) type BatchKey = (String, u32, u32, u32, u32, u32, u32);

/// The header as the scheduler holds it.
#[derive(Debug)]
pub(crate) struct Config {
    pub(crate) order: AnalyzeOrder,
    pub(crate) window: usize,
    pub(crate) linger: Duration,
}

/// A queued document, the queue room it holds, and when it arrived.
type Queued = (Doc, Room, Instant);

pub(crate) struct Engine {
    shared: Arc<Shared>,
    config: Config,
    window_sem: Arc<Semaphore>,
    /// Documents admitted, not yet dispatched.
    pending: VecDeque<Queued>,
    /// Batches running on held sessions.
    inflight: JoinSet<Outcome>,
    /// Documents executing or awaiting their answer.
    inflight_items: usize,
    /// The next sequence number to assign, and the next one an
    /// input-ordered stream may answer with.
    next_seq: u64,
    next_send: u64,
    /// Input-order answers waiting on an earlier one.
    buffer: BTreeMap<u64, AnalyzeStreamResponse>,
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
            inflight_items: 0,
            next_seq: 0,
            next_send: 0,
            buffer: BTreeMap::new(),
        }
    }

    /// The schedule loop. Returns when the input closed and every
    /// document is answered, or when the stream dies: `dead` tells the
    /// reader to stop admitting, and dropping the queue, the JoinSet
    /// and the channels frees the rooms and the window permits.
    pub(crate) async fn run(
        mut self,
        mut work: mpsc::Receiver<Envelope>,
        answers: mpsc::Sender<std::result::Result<AnalyzeStreamResponse, Status>>,
        dead: Arc<AtomicBool>,
    ) {
        let mut input_ended = false;
        loop {
            let deadline = self.deadline(input_ended);
            let mut sleeper = Box::pin(tokio::time::sleep_until(
                deadline.unwrap_or_else(|| Instant::now() + Duration::from_secs(86400)),
            ));
            tokio::select! {
                env = work.recv() => match env {
                    Some(Envelope::Doc(mut doc, room)) => {
                        doc.seq = self.next_seq;
                        self.next_seq += 1;
                        self.pending.push_back((doc, room, Instant::now()));
                    }
                    Some(Envelope::Failed { id, failure }) => {
                        let seq = self.next_seq;
                        self.next_seq += 1;
                        if self.answer(seq, Err((id, failure)), &answers).await.is_err() {
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
                        self.inflight_items -= outcome_len(&outcome);
                        if self.deliver(outcome, &answers, &dead).await.is_err() {
                            break;
                        }
                    }
                    Some(Err(e)) => {
                        let _ = answers.send(Err(Status::internal(format!("a batch's task failed: {e}")))).await;
                        break;
                    }
                    None => {}
                },
                _ = &mut sleeper, if deadline.is_some() => self.dispatch(true, &dead).await,
                // The client went away: drop the queue and the batches.
                _ = answers.closed() => break,
            }
            self.dispatch(input_ended, &dead).await;
            if input_ended && self.pending.is_empty() && self.inflight.is_empty() && self.buffer.is_empty() {
                break;
            }
        }
        dead.store(true, Ordering::SeqCst);
    }

    /// When the next dispatch is due: the linger after the oldest
    /// queued document, while more may still arrive.
    fn deadline(&self, input_ended: bool) -> Option<Instant> {
        if input_ended || self.pending.is_empty() || self.inflight_items >= self.config.window {
            return None;
        }
        self.pending.front().map(|(_, _, at)| *at + self.config.linger)
    }

    /// Runs every dispatchable batch: pending documents grouped by
    /// model and options, each batch up to the bundle's batch limit and
    /// the window's room. `force` dispatches past the linger, for the
    /// linger's own deadline and the input's half-close.
    async fn dispatch(&mut self, force: bool, dead: &Arc<AtomicBool>) {
        if self.pending.is_empty() || self.inflight_items >= self.config.window {
            return;
        }
        if !force && Instant::now() < self.deadline(false).unwrap() {
            return;
        }
        let mut groups: BTreeMap<BatchKey, Vec<Queued>> = BTreeMap::new();
        for queued in self.pending.drain(..) {
            groups.entry(run::key(&queued.0)).or_default().push(queued);
        }
        // Each group takes at most a share of the window this round, so
        // one model's backlog never stands between another model's
        // documents and the window.
        let share = (self.config.window / groups.len()).max(1);
        let mut leftover: Vec<Queued> = Vec::new();
        for (_, mut docs) in groups {
            let n =
                docs.len().min(docs[0].0.batch_limit as usize).min(share).min(self.config.window - self.inflight_items);
            if n == 0 {
                leftover.extend(docs);
                continue;
            }
            let batch: Vec<Queued> = docs.drain(..n).collect();
            match self.spawn(batch, dead).await {
                Ok(()) => leftover.extend(docs),
                Err(batch) => {
                    leftover.extend(batch);
                    leftover.extend(docs);
                    break;
                }
            }
        }
        self.pending = leftover.into();
    }

    /// Takes a session and starts one batch. On `Err` the documents
    /// come back with their rooms, to stay queued.
    async fn spawn(&mut self, batch: Vec<Queued>, dead: &Arc<AtomicBool>) -> std::result::Result<(), Vec<Queued>> {
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
        let held = loop {
            if dead.load(Ordering::SeqCst) {
                return Err(batch);
            }
            match batch[0].0.loaded.take() {
                Ok(h) => break h,
                Err(_) => tokio::time::sleep(Duration::from_millis(1)).await,
            }
        };
        let opts = batch[0].0.opts;
        let (docs, rooms): (Vec<Doc>, Vec<Room>) = batch.into_iter().map(|(d, r, _)| (d, r)).unzip();
        drop(rooms);
        let shared = self.shared.clone();
        self.inflight_items += n;
        self.inflight.spawn_blocking(move || run::run(&shared, held, docs, &opts, permits));
        Ok(())
    }

    /// Answers one batch. A batch that failed whole is retried one
    /// document at a time, so a failing document answers with its own
    /// error and the others keep their own results.
    async fn deliver(
        &mut self,
        outcome: Outcome,
        answers: &mpsc::Sender<std::result::Result<AnalyzeStreamResponse, Status>>,
        dead: &Arc<AtomicBool>,
    ) -> std::result::Result<(), ()> {
        let Outcome { model, version, dim, rows, window } = outcome;
        match rows {
            Ok(rows) => {
                for (seq, id, data) in rows {
                    let vectors = crate::proto::result::AnalyzeVectors {
                        model: model.clone(),
                        version: version.clone(),
                        dim,
                        data,
                    };
                    if self.answer(seq, Ok((id, vectors)), answers).await.is_err() {
                        return Err(());
                    }
                }
                Ok(())
            }
            Err((failure, docs)) if docs.len() == 1 => {
                let doc = &docs[0];
                self.answer(doc.seq, Err((doc.id.clone(), failure)), answers).await
            }
            Err((failure, docs)) => {
                let mut permits = window.into_iter();
                for doc in docs {
                    let Some(permit) = permits.next() else {
                        let seq = doc.seq;
                        let id = doc.id.clone();
                        if self.answer(seq, Err((id, failure.clone())), answers).await.is_err() {
                            return Err(());
                        }
                        continue;
                    };
                    let held = loop {
                        if dead.load(Ordering::SeqCst) {
                            return Err(());
                        }
                        match doc.loaded.take() {
                            Ok(h) => break h,
                            Err(_) => tokio::time::sleep(Duration::from_millis(1)).await,
                        }
                    };
                    let opts = doc.opts;
                    let shared = self.shared.clone();
                    self.inflight_items += 1;
                    self.inflight.spawn_blocking(move || run::run(&shared, held, vec![doc], &opts, vec![permit]));
                }
                Ok(())
            }
        }
    }

    /// Queues one document's answer: input order buffers until the
    /// documents before it are answered; completion order answers at
    /// once.
    async fn answer(
        &mut self,
        seq: u64,
        outcome: std::result::Result<(String, crate::proto::result::AnalyzeVectors), (String, Failure)>,
        answers: &mpsc::Sender<std::result::Result<AnalyzeStreamResponse, Status>>,
    ) -> std::result::Result<(), ()> {
        let (id, kind) = match outcome {
            Ok((id, vectors)) => (id, crate::proto::result::analyze_stream_response::Outcome::Vectors(vectors)),
            Err((id, failure)) => (
                id,
                crate::proto::result::analyze_stream_response::Outcome::Failure(crate::proto::result::AnalyzeFailure {
                    turbo_code: failure.code,
                    field: failure.field,
                    message: crate::status::describe(&failure, &crate::status::EMBED_OPTIONS),
                }),
            ),
        };
        let response = AnalyzeStreamResponse { id, outcome: Some(kind) };
        match self.config.order {
            AnalyzeOrder::Completion => {
                answers.send(Ok(response)).await.map_err(|_| ())?;
            }
            _ => {
                self.buffer.insert(seq, response);
                while let Some(r) = self.buffer.remove(&self.next_send) {
                    self.next_send += 1;
                    answers.send(Ok(r)).await.map_err(|_| ())?;
                }
            }
        }
        Ok(())
    }
}

/// How many documents one outcome carries, for the window accounting.
fn outcome_len(outcome: &Outcome) -> usize {
    match &outcome.rows {
        Ok(rows) => rows.len(),
        Err((_, docs)) => docs.len(),
    }
}

/// A running engine, started by the handler.
pub(crate) fn start(
    shared: Arc<Shared>,
    config: Config,
    work: mpsc::Receiver<Envelope>,
    answers: mpsc::Sender<std::result::Result<AnalyzeStreamResponse, Status>>,
    dead: Arc<AtomicBool>,
) -> JoinHandle<()> {
    tokio::spawn(Engine::new(shared, config).run(work, answers, dead))
}
