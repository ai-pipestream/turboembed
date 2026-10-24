//! The analyzer: a validated graph of steps, run over batches or streamed
//! progressively as individual DAG branches complete.

use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use crate::annotate::{AnchorUse, Annotator, Bound, Step};
use crate::doc::{Document, Status, View};
use crate::error::Error;
use crate::layer::{Key, KeyId, Layer, LayerName};

/// Timing and output inventory for a single completed [`Step`] in
/// [`Analyzer::run_progressive`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepTiming {
    /// Static step identifier (`Annotator::ID`).
    pub step_id: &'static str,
    /// Topological stage index (`0..stages.len()`).
    pub stage: usize,
    /// Wall-clock execution time of `Step::compute` in microseconds.
    pub duration_us: u64,
    /// Layer keys written (or marked skipped/failed) by this step.
    pub written: Vec<KeyId>,
}

/// Incremental callback payload emitted by [`Analyzer::run_progressive`] as each
/// step in the DAG completes.
pub struct StepEvent<'a> {
    /// Step identity, stage, execution duration, and written layer keys.
    pub timing: StepTiming,
    /// Current document slice after storing this step's outputs (and any
    /// immediately cascaded `Status::Skipped` dependent outputs).
    pub docs: &'a [Document],
    /// First error recorded by this step across the batch, if the step failed.
    pub failure: Option<Error>,
    /// Downstream step IDs whose required inputs were failed or skipped on all
    /// documents as a result of this step's failure.
    pub skipped_dependents: Vec<&'static str>,
}

impl std::fmt::Debug for StepEvent<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepEvent")
            .field("timing", &self.timing)
            .field("docs_len", &self.docs.len())
            .field("failure", &self.failure)
            .field("skipped_dependents", &self.skipped_dependents)
            .finish()
    }
}

/// A pipeline. Steps are ordered by what they read and write, not by the
/// order they were added; steps that do not depend on each other form one
/// stage and compute side by side. A stage runs over the whole batch
/// before the next starts, so every step sees all of the batch's documents
/// at once.
pub struct Analyzer {
    steps: Vec<Box<dyn Step>>,
    stages: Vec<Vec<usize>>,
}

pub struct Builder {
    given: Vec<KeyId>,
    steps: Vec<Box<dyn Step>>,
}

impl Analyzer {
    pub fn builder() -> Builder {
        Builder { given: vec![Key::<View>::DEFAULT.id()], steps: Vec::new() }
    }

    /// The steps in run order, a stage per line: for logs and the docs.
    pub fn plan(&self) -> Vec<Vec<&'static str>> {
        self.stages.iter().map(|s| s.iter().map(|&i| self.steps[i].id()).collect()).collect()
    }

    /// Planned step identifiers flattened in topological stage order.
    pub fn step_ids(&self) -> Vec<&'static str> {
        self.stages.iter().flat_map(|s| s.iter().map(|&i| self.steps[i].id())).collect()
    }

    /// Runs every stage over the batch. A step that fails on a document,
    /// by an error or a panic, fails only its own layers there: the steps
    /// that need them are skipped and say why, the others still run, and
    /// no other document is touched. Each document's result is its first
    /// failure.
    pub fn run(&self, docs: &mut [Document]) -> Vec<Result<(), Error>> {
        let mut status: Vec<Result<(), Error>> = (0..docs.len()).map(|_| Ok(())).collect();
        for stage in &self.stages {
            let outputs: Vec<_> = if stage.len() == 1 {
                vec![self.steps[stage[0]].compute(docs)]
            } else {
                let docs = &*docs;
                std::thread::scope(|s| {
                    let running: Vec<_> = stage.iter().map(|&i| s.spawn(move || self.steps[i].compute(docs))).collect();
                    running.into_iter().map(|h| h.join().expect("an annotator's panics are caught")).collect()
                })
            };
            for (&i, out) in stage.iter().zip(outputs) {
                self.steps[i].store(out, docs, &mut status);
            }
        }
        status
    }

    /// Executes the pipeline DAG progressively over `docs`, spawning ready
    /// steps in parallel on `std::thread::scope` and invoking `on_step` on the
    /// caller thread as soon as each individual step finishes (without waiting
    /// for slower sibling steps in the same stage).
    ///
    /// When a step completes:
    /// 1. Its outputs are stored onto `docs` immediately.
    /// 2. If the step failed on all documents in `docs`, any downstream steps
    ///    whose required inputs are now failed or skipped on all documents have
    ///    their outputs marked `Status::Skipped` immediately and their IDs
    ///    reported in [`StepEvent::skipped_dependents`].
    /// 3. `on_step` is invoked with the [`StepEvent`].
    /// 4. If `cancel` is not set, any downstream steps whose `reads()` are now
    ///    satisfied are spawned immediately.
    ///
    /// Setting `cancel` to `true` stops any unstarted steps from being spawned.
    /// When all steps complete without cancellation, each document's layer
    /// entries are placed in canonical stage order so the final [`Document`]s
    /// match [`Analyzer::run`] identically.
    pub fn run_progressive<F>(
        &self,
        docs: &mut [Document],
        cancel: &AtomicBool,
        mut on_step: F,
    ) -> Vec<Result<(), Error>>
    where
        F: FnMut(StepEvent<'_>),
    {
        let mut status: Vec<Result<(), Error>> = (0..docs.len()).map(|_| Ok(())).collect();
        let n = self.steps.len();
        if n == 0 || cancel.load(Ordering::Relaxed) {
            return status;
        }

        let mut stage_of = vec![0usize; n];
        let mut topo_steps = Vec::with_capacity(n);
        let mut canonical_keys = Vec::new();
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            for &step_ix in stage {
                stage_of[step_ix] = stage_idx;
                topo_steps.push(step_ix);
                canonical_keys.extend(self.steps[step_ix].writes());
            }
        }

        let mut producer_of: HashMap<KeyId, usize> = HashMap::new();
        for (i, step) in self.steps.iter().enumerate() {
            for w in step.writes() {
                producer_of.insert(w, i);
            }
        }

        // Steps that write a `View` must not run concurrently with other steps
        // that read or write views because `Document::add` assigns `ViewId`s
        // sequentially by insertion order.
        let view_ty = TypeId::of::<View>();
        let writes_view: Vec<bool> = self.steps.iter().map(|st| st.writes().iter().any(|k| k.ty == view_ty)).collect();

        // 0 = unstarted, 1 = running, 2 = done
        let mut step_state = vec![0u8; n];

        std::thread::scope(|s| {
            let (tx, rx) = mpsc::channel::<(usize, u64, Box<dyn Any + Send>)>();
            let mut inflight = 0usize;
            let mut view_step_inflight = false;

            let spawn_ready = |docs: &[Document],
                               step_state: &mut [u8],
                               inflight: &mut usize,
                               view_step_inflight: &mut bool| {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let mut to_spawn = Vec::new();
                for &i in &topo_steps {
                    if step_state[i] != 0 {
                        continue;
                    }
                    let deps_ready =
                        self.steps[i].reads().iter().all(|r| producer_of.get(r).is_none_or(|&p| step_state[p] == 2));
                    if !deps_ready {
                        continue;
                    }
                    if writes_view[i] && (*inflight > 0 || !to_spawn.is_empty()) {
                        break;
                    }
                    if *view_step_inflight {
                        break;
                    }
                    to_spawn.push(i);
                    if writes_view[i] {
                        *view_step_inflight = true;
                        break;
                    }
                }
                if to_spawn.is_empty() {
                    return;
                }
                let snapshot: Arc<[Document]> = Arc::from(docs.to_vec());
                for i in to_spawn {
                    step_state[i] = 1;
                    *inflight += 1;
                    let tx = tx.clone();
                    let snap = Arc::clone(&snapshot);
                    let step = &*self.steps[i];
                    s.spawn(move || {
                        let start = Instant::now();
                        let out = step.compute(&snap);
                        let duration_us = start.elapsed().as_micros() as u64;
                        let _ = tx.send((i, duration_us, out));
                    });
                }
            };

            spawn_ready(docs, &mut step_state, &mut inflight, &mut view_step_inflight);

            while inflight > 0 {
                let (step_ix, duration_us, out) = rx.recv().expect("sender held while inflight > 0");
                inflight -= 1;
                if writes_view[step_ix] {
                    view_step_inflight = false;
                }
                step_state[step_ix] = 2;

                let written = self.steps[step_ix].writes();
                self.steps[step_ix].store(out, docs, &mut status);

                let failure: Option<Error> = written.iter().find_map(|&k| {
                    docs.iter().find_map(|d| match d.status(k) {
                        Some(Status::Failed(err)) => Some(err.clone()),
                        _ => None,
                    })
                });

                let mut skipped_dependents = Vec::new();
                if failure.is_some() && !docs.is_empty() {
                    for &dep_ix in &topo_steps {
                        if step_state[dep_ix] != 0 {
                            continue;
                        }
                        let req_reads: Vec<KeyId> = self.steps[dep_ix]
                            .reads()
                            .into_iter()
                            .zip(self.steps[dep_ix].required())
                            .filter(|&(_, req)| req)
                            .map(|(k, _)| k)
                            .collect();
                        let all_docs_blocked = docs.iter().all(|d| {
                            req_reads
                                .iter()
                                .any(|&rk| matches!(d.status(rk), Some(Status::Failed(_) | Status::Skipped(_))))
                        });
                        if all_docs_blocked {
                            step_state[dep_ix] = 2;
                            let dep_out = self.steps[dep_ix].compute(docs);
                            self.steps[dep_ix].store(dep_out, docs, &mut status);
                            skipped_dependents.push(self.steps[dep_ix].id());
                        }
                    }
                }

                let event = StepEvent {
                    timing: StepTiming {
                        step_id: self.steps[step_ix].id(),
                        stage: stage_of[step_ix],
                        duration_us,
                        written,
                    },
                    docs,
                    failure,
                    skipped_dependents,
                };
                on_step(event);

                spawn_ready(docs, &mut step_state, &mut inflight, &mut view_step_inflight);
            }
        });

        if !cancel.load(Ordering::Relaxed) {
            let order_map: HashMap<KeyId, usize> =
                canonical_keys.iter().copied().enumerate().map(|(pos, k)| (k, pos)).collect();
            for doc in docs.iter_mut() {
                doc.entries.sort_by_key(|e| order_map.get(&e.key).copied().map_or((0, 0), |pos| (1, pos)));
            }
        }

        status
    }
}

impl Builder {
    /// A layer the documents arrive with, such as document structure from
    /// the parser upstream.
    pub fn given<L: Layer>(mut self, key: Key<L>) -> Builder {
        self.given.push(key.id());
        self
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add<A: Annotator>(self, annotator: A) -> Builder {
        self.step(Bound::new(annotator))
    }

    pub fn step(mut self, step: impl Step + 'static) -> Builder {
        self.steps.push(Box::new(step));
        self
    }

    /// A step made at run time, by a provider.
    pub fn boxed(mut self, step: Box<dyn Step>) -> Builder {
        self.steps.push(step);
        self
    }

    /// Checks the graph and orders it: one layer type per id, every read
    /// with exactly one producer (a step or a given layer), every aligned
    /// output's anchor one of its step's reads or an output the step writes
    /// before it, and no cycle.
    pub fn build(self) -> Result<Analyzer, Error> {
        let mut types: HashMap<&LayerName, TypeId> = HashMap::new();
        let keys = self.steps.iter().flat_map(|s| s.reads().into_iter().chain(s.writes()));
        for k in self.given.iter().copied().chain(keys) {
            if *types.entry(k.layer).or_insert(k.ty) != k.ty {
                return Err(Error::Pipeline(format!("two layer types are both {}", k.layer)));
            }
        }
        let mut producer: HashMap<KeyId, Option<usize>> = self.given.iter().map(|&k| (k, None)).collect();
        for (i, s) in self.steps.iter().enumerate() {
            for w in s.writes() {
                if let Some(prev) = producer.insert(w, Some(i)) {
                    let by = prev.map_or("the input".to_string(), |p| self.steps[p].id().to_string());
                    return Err(Error::Pipeline(format!("{} and {} both write {w}", by, s.id())));
                }
            }
        }
        for s in &self.steps {
            for (r, required) in s.reads().into_iter().zip(s.required()) {
                if required && !producer.contains_key(&r) {
                    return Err(Error::Pipeline(format!("{} reads {r}, which nothing writes", s.id())));
                }
            }
            for (w, a) in s.writes().into_iter().zip(s.anchors()) {
                match a {
                    AnchorUse::Unread(base) => {
                        return Err(Error::Pipeline(format!(
                            "{} writes {w}, aligned to {base}, without reading {base}",
                            s.id()
                        )));
                    }
                    AnchorUse::Later(base) => {
                        return Err(Error::Pipeline(format!(
                            "{} writes {w}, aligned to {base}, before {base}: the base must come first in its writes",
                            s.id()
                        )));
                    }
                    AnchorUse::None | AnchorUse::To(_) => {}
                }
            }
        }
        // Levels: a step runs once everything it reads that anything
        // writes exists; an optional input nothing writes is just absent.
        let mut level = vec![usize::MAX; self.steps.len()];
        let mut done: HashSet<KeyId> = self.given.iter().copied().collect();
        let mut stages = Vec::new();
        while level.contains(&usize::MAX) {
            let ready: Vec<usize> = (0..self.steps.len())
                .filter(|&i| {
                    level[i] == usize::MAX
                        && self.steps[i].reads().iter().all(|r| done.contains(r) || !producer.contains_key(r))
                })
                .collect();
            if ready.is_empty() {
                let stuck: Vec<_> =
                    (0..self.steps.len()).filter(|&i| level[i] == usize::MAX).map(|i| self.steps[i].id()).collect();
                return Err(Error::Pipeline(format!("a cycle among {stuck:?}")));
            }
            for &i in &ready {
                level[i] = stages.len();
                done.extend(self.steps[i].writes());
            }
            stages.push(ready);
        }
        Ok(Analyzer { steps: self.steps, stages })
    }
}
