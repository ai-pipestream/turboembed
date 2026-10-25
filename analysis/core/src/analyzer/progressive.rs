use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use super::Analyzer;
use crate::doc::{Document, Status, View};
use crate::error::Error;
use crate::layer::KeyId;

/// Timing and output inventory for a single completed [`Step`](crate::annotate::Step) in
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
    /// cascaded `Status::Skipped` dependent outputs whose producers have
    /// settled, so their provenance references are present).
    pub docs: &'a [Document],
    /// First error recorded by this step across the batch, if the step failed.
    pub failure: Option<Error>,
    /// Downstream step IDs whose required inputs were failed or skipped on all
    /// documents as a result of this step's failure. A reported step can still
    /// await another input producer before storing its skipped outputs.
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

impl Analyzer {
    /// Executes the pipeline DAG progressively over `docs`, spawning ready
    /// steps in parallel on `std::thread::scope` and invoking `on_step` on the
    /// caller thread as soon as each individual step finishes (without waiting
    /// for slower sibling steps in the same stage).
    ///
    /// When a step completes:
    /// 1. Its outputs are stored onto `docs` immediately.
    /// 2. If the step failed on all documents in `docs`, any downstream steps
    ///    whose required inputs are now failed or skipped on all documents have
    ///    their IDs reported in [`StepEvent::skipped_dependents`]. Outputs are
    ///    marked `Status::Skipped` immediately when their dependencies are
    ///    ready; otherwise the step waits for normal readiness so graph
    ///    snapshots never name a producer's still-absent outputs.
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
        let mut deferred_skips = vec![false; n];
        let deps_ready = |i: usize, state: &[u8]| {
            self.steps[i]
                .reads()
                .iter()
                .all(|r| producer_of.get(r).is_none_or(|&p| stage_of[p] >= stage_of[i] || state[p] == 2))
        };

        std::thread::scope(|s| {
            let (tx, rx) = mpsc::channel::<(usize, u64, Box<dyn Any + Send>)>();
            let mut inflight = 0usize;
            let mut view_step_inflight = false;

            let spawn_ready =
                |docs: &[Document], step_state: &mut [u8], inflight: &mut usize, view_step_inflight: &mut bool| {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let mut to_spawn = Vec::new();
                    for (pos, &i) in topo_steps.iter().enumerate() {
                        if step_state[i] != 0 {
                            continue;
                        }
                        if !deps_ready(i, step_state) {
                            continue;
                        }
                        if writes_view[i] {
                            let earlier_views_done =
                                topo_steps[..pos].iter().all(|&prev| !writes_view[prev] || step_state[prev] == 2);
                            if !earlier_views_done || *inflight > 0 || !to_spawn.is_empty() {
                                break;
                            }
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
                            req_reads.iter().any(|&rk| {
                                matches!(d.status(rk), Some(Status::Failed(_) | Status::Skipped(_)))
                                    || producer_of.get(&rk).is_some_and(|&p| deferred_skips[p])
                            })
                        });
                        if all_docs_blocked {
                            if !deferred_skips[dep_ix] {
                                skipped_dependents.push(self.steps[dep_ix].id());
                            }
                            if deps_ready(dep_ix, &step_state) {
                                step_state[dep_ix] = 2;
                                let dep_out = self.steps[dep_ix].compute(docs);
                                self.steps[dep_ix].store(dep_out, docs, &mut status);
                            } else {
                                // Report the failure cascade now, but let normal
                                // readiness store this skip after its other inputs.
                                deferred_skips[dep_ix] = true;
                            }
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
            debug_assert!(step_state.iter().all(|&s| s == 2), "all steps must complete when not cancelled");
            let order_map: HashMap<KeyId, usize> =
                canonical_keys.iter().copied().enumerate().map(|(pos, k)| (k, pos)).collect();
            for doc in docs.iter_mut() {
                doc.entries.sort_by_key(|e| order_map.get(&e.key).copied().map_or((0, 0), |pos| (1, pos)));
            }
        }

        status
    }
}
