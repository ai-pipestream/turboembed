//! Critical-path DAG execution plan and zero-barrier scheduler over [`WorkerPool`].

use std::any::{Any, TypeId};
use std::collections::{BinaryHeap, HashMap};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};

use crate::annotate::Step;
use crate::doc::{Document, View};
use crate::error::Error;
use crate::layer::{KeyId, Provenance};

use super::Analyzer;
use super::pool::{PriorityKey, WorkerPool};

pub(super) fn build_metadata(stages: &[Vec<usize>], steps: &[Arc<dyn Step>]) -> (Vec<usize>, Vec<bool>) {
    let mut topo_rank = vec![0usize; steps.len()];
    for (rank, &i) in stages.iter().flatten().enumerate() {
        topo_rank[i] = rank;
    }
    let view_ty = TypeId::of::<View>();
    let writes_view = steps.iter().map(|s| s.writes().iter().any(|w| w.ty == view_ty)).collect();
    (topo_rank, writes_view)
}

/// Per-step node in an [`Analyzer`]'s dependency DAG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DagStep {
    /// Step index in insertion order (`0..steps.len()`).
    pub index: usize,
    /// Stable step/annotator identifier (`Step::id()`).
    pub id: &'static str,
    /// Topological stage depth (`0` for root steps with no upstream producers).
    pub stage: usize,
    /// Exact upstream step indices whose outputs must be stored before this step
    /// becomes ready.
    pub predecessors: Vec<usize>,
    /// Exact downstream step indices that depend directly on this step, ordered
    /// by descending critical-path priority.
    pub dependents: Vec<usize>,
    /// Relative per-step execution cost hint (`Step::cost_hint()`).
    pub cost_hint: u32,
    /// Total critical-path weight from this step through its longest downstream
    /// dependency chain (`cost_hint + max_{d in dependents} critical_path_cost[d]`).
    pub critical_path_cost: u64,
    /// Total number of transitive downstream steps reachable from this step.
    pub transitive_dependents: usize,
}

/// Alias for [`DagStep`].
pub type DagNode = DagStep;

pub(super) fn build_dag_steps(
    steps: &[Arc<dyn Step>],
    backedge_reads: &[Vec<KeyId>],
    producer: &HashMap<KeyId, Option<usize>>,
    stage_of: &[usize],
) -> Vec<DagStep> {
    let n = steps.len();
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for r in steps[i].reads() {
            if backedge_reads[i].contains(&r) {
                continue;
            }
            if let Some(Some(u)) = producer.get(&r)
                && stage_of[*u] < stage_of[i]
                && !predecessors[i].contains(u)
            {
                predecessors[i].push(*u);
            }
        }
    }

    // If step `reader` has an optional backedge read produced by step `writer`,
    // `reader` must complete before `writer` stores that layer so `reader`
    // sees the pre-writer document state deterministically without stage barriers.
    for (reader, backedges) in backedge_reads.iter().enumerate() {
        for &r in backedges {
            if let Some(Some(writer)) = producer.get(&r)
                && *writer != reader
                && !predecessors[*writer].contains(&reader)
            {
                predecessors[*writer].push(reader);
            }
        }
    }

    for (i, preds) in predecessors.iter_mut().enumerate() {
        preds.sort_unstable();
        for &u in preds.iter() {
            dependents[u].push(i);
        }
    }

    let cost_hints: Vec<u32> = steps.iter().map(|s| s.cost_hint().max(1)).collect();

    // Process steps in reverse stage order so all dependents of `u` are computed
    // before `u`.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| stage_of[b].cmp(&stage_of[a]).then_with(|| b.cmp(&a)));

    let mut critical_path_cost = vec![0u64; n];
    let mut reachable: Vec<Vec<bool>> = vec![vec![false; n]; n];
    let mut transitive_dependents = vec![0usize; n];

    for &u in &order {
        let max_child = dependents[u].iter().map(|&d| critical_path_cost[d]).max().unwrap_or(0);
        critical_path_cost[u] = u64::from(cost_hints[u]) + max_child;
        for &d in &dependents[u] {
            reachable[u][d] = true;
            let child_reach = reachable[d].clone();
            for (dst, &src) in reachable[u].iter_mut().zip(&child_reach) {
                if src {
                    *dst = true;
                }
            }
        }
        transitive_dependents[u] = reachable[u].iter().filter(|&&b| b).count();
    }

    for deps in &mut dependents {
        deps.sort_by(|&a, &b| {
            critical_path_cost[b]
                .cmp(&critical_path_cost[a])
                .then_with(|| transitive_dependents[b].cmp(&transitive_dependents[a]))
                .then_with(|| cost_hints[b].cmp(&cost_hints[a]))
                .then_with(|| stage_of[a].cmp(&stage_of[b]))
                .then_with(|| a.cmp(&b))
        });
    }

    (0..n)
        .map(|i| DagStep {
            index: i,
            id: steps[i].id(),
            stage: stage_of[i],
            predecessors: std::mem::take(&mut predecessors[i]),
            dependents: std::mem::take(&mut dependents[i]),
            cost_hint: cost_hints[i],
            critical_path_cost: critical_path_cost[i],
            transitive_dependents: transitive_dependents[i],
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ReadyStep {
    priority: PriorityKey,
    step_idx: usize,
}

impl Ord for ReadyStep {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for ReadyStep {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

type StepResult = Result<Box<dyn Any + Send>, Box<dyn Any + Send>>;

impl Analyzer {
    /// Per-step DAG metadata (`predecessors`, `dependents`, `cost_hint`, and
    /// `critical_path_cost`) indexed by step insertion order.
    pub fn dag_plan(&self) -> &[DagStep] {
        &self.dag
    }

    /// Alias for [`Analyzer::dag_plan`].
    pub fn dag_steps(&self) -> &[DagStep] {
        &self.dag
    }

    /// Upstream step indices that must complete before `step_index` can start.
    pub fn predecessors(&self, step_index: usize) -> &[usize] {
        self.dag.get(step_index).map_or(&[], |d| &d.predecessors)
    }

    /// Downstream step indices directly unlocked when `step_index` completes,
    /// ordered by descending critical-path priority.
    pub fn dependents(&self, step_index: usize) -> &[usize] {
        self.dag.get(step_index).map_or(&[], |d| &d.dependents)
    }

    /// Returns the pipeline's step identifiers ordered as a single-worker
    /// critical-path DAG execution schedules them.
    pub fn critical_path_order(&self) -> Vec<&'static str> {
        let n = self.dag.len();
        if n == 0 {
            return Vec::new();
        }
        let mut rem: Vec<usize> = self.dag.iter().map(|d| d.predecessors.len()).collect();
        let mut heap = BinaryHeap::new();
        for (i, d) in self.dag.iter().enumerate() {
            if rem[i] == 0 {
                heap.push(ReadyStep { priority: self.step_priority(d, i as u64), step_idx: i });
            }
        }
        let mut seq = n as u64;
        let mut out = Vec::with_capacity(n);
        while let Some(next) = heap.pop() {
            let u = next.step_idx;
            out.push(self.dag[u].id);
            for &dep in &self.dag[u].dependents {
                rem[dep] -= 1;
                if rem[dep] == 0 {
                    heap.push(ReadyStep { priority: self.step_priority(&self.dag[dep], seq), step_idx: dep });
                    seq += 1;
                }
            }
        }
        out
    }

    fn step_priority(&self, node: &DagStep, seq: u64) -> PriorityKey {
        PriorityKey {
            critical_path_cost: node.critical_path_cost,
            transitive_dependents: node.transitive_dependents,
            cost_hint: node.cost_hint,
            topo_rank: self.topo_rank[node.index],
            seq,
        }
    }

    /// Executes the pipeline on `docs` using `pool` with zero stage barriers:
    /// each step is dispatched as soon as its specific `predecessors` finish,
    /// prioritized by critical-path cost.
    pub fn run_with_pool(&self, docs: &mut [Document], pool: &WorkerPool) -> Vec<Result<(), Error>> {
        if docs.is_empty() {
            return Vec::new();
        }
        let n = self.steps.len();
        if n == 0 {
            return (0..docs.len()).map(|_| Ok(())).collect();
        }

        // Chain View-writing steps in canonical stage order so ViewId assignment
        // in Document::add is deterministic regardless of worker count.
        let mut extra_dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
        let rem_preds: Vec<AtomicUsize> = self.dag.iter().map(|d| AtomicUsize::new(d.predecessors.len())).collect();
        let mut prev_view_step: Option<usize> = None;
        for &step_idx in self.stages.iter().flatten() {
            if self.writes_view[step_idx] {
                if let Some(prev) = prev_view_step
                    && !self.dag[step_idx].predecessors.contains(&prev)
                {
                    extra_dependents[prev].push(step_idx);
                    rem_preds[step_idx].fetch_add(1, Ordering::Relaxed);
                }
                prev_view_step = Some(step_idx);
            }
        }

        let mut ready = BinaryHeap::new();
        for (i, count) in rem_preds.iter().enumerate() {
            if count.load(Ordering::Relaxed) == 0 {
                ready.push(ReadyStep { priority: self.step_priority(&self.dag[i], pool.next_seq()), step_idx: i });
            }
        }

        let (tx, rx) = mpsc::channel::<(usize, StepResult)>();
        let mut in_flight = 0usize;
        let mut completed = 0usize;
        let mut hint = 0usize;
        let mut per_step_status: Vec<Vec<Result<(), Error>>> =
            (0..n).map(|_| (0..docs.len()).map(|_| Ok(())).collect()).collect();

        while completed < n {
            let shared_docs: Option<Arc<[Document]>> =
                (!ready.is_empty()).then(|| Arc::from(docs.to_vec().into_boxed_slice()));
            while let Some(ReadyStep { priority, step_idx }) = ready.pop() {
                let step = Arc::clone(&self.steps[step_idx]);
                let backedges = self.backedge_reads[step_idx].clone();
                let base_snap = Arc::clone(shared_docs.as_ref().expect("snapshot created"));
                let tx_step = tx.clone();
                let worker_hint = hint;
                hint = hint.wrapping_add(1);
                in_flight += 1;
                pool.spawn(worker_hint, priority, move || {
                    let res = catch_unwind(AssertUnwindSafe(|| {
                        if backedges.is_empty() {
                            step.compute(&base_snap)
                        } else {
                            let mut stripped: Vec<Document> = base_snap.iter().cloned().collect();
                            for d in &mut stripped {
                                d.take_entries(&backedges);
                            }
                            step.compute(&stripped)
                        }
                    }));
                    let _ = tx_step.send((step_idx, res));
                });
            }
            drop(shared_docs);

            if in_flight == 0 {
                break;
            }

            let (done_idx, step_res) = rx.recv().expect("worker pool holds sender until step completes");
            in_flight -= 1;
            completed += 1;

            self.store_completed_step(done_idx, step_res, docs, &mut per_step_status[done_idx]);

            for &dep in self.dag[done_idx].dependents.iter().chain(&extra_dependents[done_idx]) {
                if rem_preds[dep].fetch_sub(1, Ordering::AcqRel) == 1 {
                    ready.push(ReadyStep {
                        priority: self.step_priority(&self.dag[dep], pool.next_seq()),
                        step_idx: dep,
                    });
                }
            }
        }

        for doc in docs.iter_mut() {
            doc.entries.sort_by_key(|e| self.order_map.get(&e.key).copied().map_or((0, 0), |pos| (1, pos)));
        }

        (0..docs.len())
            .map(|doc_idx| {
                for &step_idx in self.stages.iter().flatten() {
                    if let Err(err) = &per_step_status[step_idx][doc_idx] {
                        return Err(err.clone());
                    }
                }
                Ok(())
            })
            .collect()
    }

    fn store_completed_step(
        &self,
        step_idx: usize,
        step_res: StepResult,
        docs: &mut [Document],
        step_status: &mut [Result<(), Error>],
    ) {
        let step = &self.steps[step_idx];
        let writes = step.writes();
        match step_res {
            Ok(out) => {
                let backedges = &self.backedge_reads[step_idx];
                let taken: Vec<_> = if backedges.is_empty() {
                    Vec::new()
                } else {
                    docs.iter_mut().map(|d| d.take_entries(backedges)).collect()
                };
                step.store(out, docs, step_status);
                for doc in docs.iter_mut() {
                    for &w in &writes {
                        doc.mark_pipeline_stored(w);
                    }
                }
                for (doc, entries) in docs.iter_mut().zip(taken) {
                    doc.restore_entries(entries);
                }
            }
            Err(panic) => {
                let err = crate::predicate::panicked(step.id(), panic);
                for (doc_idx, doc) in docs.iter_mut().enumerate() {
                    for &w in &writes {
                        if !doc.has(w) {
                            let _ = doc.fail(w, err.clone(), Provenance::by(step.id()));
                        }
                    }
                    if step_status[doc_idx].is_ok() {
                        step_status[doc_idx] = Err(err.clone());
                    }
                }
            }
        }
    }

    /// Executes the pipeline on a dedicated bounded worker pool of `workers`
    /// threads without stage barriers.
    pub fn run_with_workers(&self, docs: &mut [Document], workers: usize) -> Vec<Result<(), Error>> {
        let pool = WorkerPool::new(workers);
        self.run_with_pool(docs, &pool)
    }

    /// Executes the pipeline using the stage-barrier `std::thread::scope`
    /// schedule, retained as a baseline for head-to-head latency and throughput
    /// comparisons.
    pub fn run_staged(&self, docs: &mut [Document]) -> Vec<Result<(), Error>> {
        let mut status: Vec<Result<(), Error>> = (0..docs.len()).map(|_| Ok(())).collect();
        for stage in &self.stages {
            self.run_steps(stage, docs, &mut status);
        }
        status
    }
}
