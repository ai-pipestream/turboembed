//! Content-addressed incremental DAG execution, build-time DAG validation,
//! and sequential/parallel execution helpers for [`Analyzer`].

use std::any::Any;
use std::collections::HashSet;

use crate::annotate::Step;
use crate::doc::{Document, ProvenanceDigest, Status};
use crate::error::Error;
use crate::layer::KeyId;

use super::{Analyzer, LayerExecution};

/// Cache decision for a pipeline step during [`Analyzer::analyze_incremental`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageCacheStatus {
    /// All produced keys were already present on the document with matching
    /// chained provenance digests and upstream content snapshots, so the step
    /// was skipped in O(1).
    Skipped,
    /// The step executed (missing output, config/model change, or upstream invalidation).
    Executed,
}

/// Per-step execution record emitted by [`Analyzer::analyze_incremental`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageIncrementalReport {
    /// Static step identifier (`Annotator::ID`).
    pub step: &'static str,
    /// Topological stage level (`0..plan().len()`).
    pub level: usize,
    /// Whether this step was skipped via provenance cache hit or executed.
    pub cache_status: StageCacheStatus,
    /// 256-bit SHA-256 chained provenance digest for this step's produced layers on the document.
    pub provenance_digest: ProvenanceDigest,
    /// Output layer records after this step settled.
    pub outputs: Vec<LayerExecution>,
}

/// Summary report returned by [`Analyzer::analyze_incremental`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IncrementalReport {
    /// Number of pipeline steps that executed.
    pub stages_run: usize,
    /// Number of pipeline steps skipped in O(1) via chained provenance digest match.
    pub stages_skipped: usize,
    /// Per-step incremental execution records in topological order.
    pub stages: Vec<StageIncrementalReport>,
}

impl IncrementalReport {
    /// Looks up the incremental execution record for `step_id`.
    pub fn step(&self, step_id: &str) -> Option<&StageIncrementalReport> {
        self.stages.iter().find(|s| s.step == step_id)
    }

    /// Returns `true` if `step_id` was skipped via provenance cache hit.
    pub fn was_skipped(&self, step_id: &str) -> bool {
        self.step(step_id).is_some_and(|s| s.cache_status == StageCacheStatus::Skipped)
    }

    /// Returns `true` if `step_id` executed during this incremental run.
    pub fn was_executed(&self, step_id: &str) -> bool {
        self.step(step_id).is_some_and(|s| s.cache_status == StageCacheStatus::Executed)
    }
}

fn can_skip_step(step: &dyn Step, doc: &Document, recomputed_keys: &HashSet<KeyId>) -> Option<ProvenanceDigest> {
    if step.reads().iter().any(|r| recomputed_keys.contains(r)) {
        return None;
    }
    let writes = step.writes();
    if writes.is_empty() {
        return None;
    }
    if !writes
        .iter()
        .all(|&w| doc.present(w) && matches!(doc.status(w), Some(Status::Complete)) && doc.content_digest(w).is_some())
    {
        return None;
    }
    let expected_prov = step.expected_provenance(doc);
    let expected_digest = doc.compute_provenance_digest(&expected_prov);
    let expected_upstream_content = doc.compute_upstream_content_hash(&expected_prov.reads)?;
    for &w in &writes {
        if doc.provenance_digest(w) != Some(expected_digest) {
            return None;
        }
        if doc.upstream_content_hash(w) != Some(expected_upstream_content) {
            return None;
        }
    }
    Some(expected_digest)
}

impl Analyzer {
    /// Validates the pipeline DAG using the build-time schedule verification:
    /// verifies that every layer name maps to a single Rust type, no two steps
    /// write the same key, every required upstream layer (`Annotator::requires`
    /// / `Reads`) has a producer, every aligned output's anchor is read or
    /// written earlier by the same step, and there are no required cycles.
    pub fn validate_dag(&self) -> Result<(), Error> {
        match &self.dag_error {
            Some(err) => Err(err.clone()),
            None => Ok(()),
        }
    }

    /// Executes the pipeline DAG sequentially (one step at a time in topological
    /// stage order without spawning worker threads), useful as a deterministic
    /// reference against parallel stage execution.
    pub fn run_sequential(&self, docs: &mut [Document]) -> Vec<Result<(), Error>> {
        let mut status: Vec<Result<(), Error>> = (0..docs.len()).map(|_| Ok(())).collect();
        for stage in &self.stages {
            for &i in stage {
                let backedges = &self.backedge_reads[i];
                let taken: Vec<_> = if backedges.is_empty() {
                    Vec::new()
                } else {
                    docs.iter_mut().map(|d| d.take_entries(backedges)).collect()
                };
                let out = self.steps[i].compute(docs);
                self.steps[i].store(out, docs, &mut status);
                let writes = self.steps[i].writes();
                for doc in docs.iter_mut() {
                    for &w in &writes {
                        doc.mark_pipeline_stored(w);
                    }
                }
                for (doc, entries) in docs.iter_mut().zip(taken) {
                    doc.restore_entries(entries);
                }
            }
        }
        status
    }

    /// Runs content-addressed incremental DAG execution over a single [`Document`]:
    ///
    /// - Validates the pipeline DAG (`validate_dag`).
    /// - For each topological stage level, checks in $O(1)$ whether each step's
    ///   `produces()` keys are already present on `doc` with matching 256-bit
    ///   chained provenance digests and upstream content snapshots.
    /// - Skips matching steps in $O(1)$ without re-running `Annotator::annotate`.
    /// - Executes remaining steps in the same topological level concurrently via
    ///   `std::thread::scope`, invalidating and recomputing all transitive
    ///   downstream dependents when any upstream step's `config_hash`, `model`,
    ///   or output changes.
    /// - Restores canonical topological layer ordering on `doc`.
    pub fn analyze_incremental(&self, doc: &mut Document) -> Result<IncrementalReport, Error> {
        self.run_incremental(std::slice::from_mut(doc)).pop().expect("single document")
    }

    /// Runs content-addressed incremental DAG execution across a batch of documents,
    /// coalescing all uncached documents at each stage into a single batched
    /// `Step::compute` (`Annotator::annotate_batch_placed`) invocation per step.
    pub fn run_incremental(&self, docs: &mut [Document]) -> Vec<Result<IncrementalReport, Error>> {
        let n_docs = docs.len();
        if n_docs == 0 {
            return Vec::new();
        }
        if let Err(err) = self.validate_dag() {
            return (0..n_docs).map(|_| Err(err.clone())).collect();
        }

        let mut doc_status: Vec<Result<(), Error>> = (0..n_docs).map(|_| Ok(())).collect();
        let mut recomputed_keys: Vec<HashSet<KeyId>> = (0..n_docs).map(|_| HashSet::new()).collect();
        let mut stages_run = vec![0usize; n_docs];
        let mut stages_skipped = vec![0usize; n_docs];
        let mut stage_reports: Vec<Vec<StageIncrementalReport>> =
            (0..n_docs).map(|_| Vec::with_capacity(self.steps.len())).collect();

        for (level, stage) in self.stages.iter().enumerate() {
            // Per step in `stage`: (step_idx, cached_digest_per_doc, miss_doc_indices)
            let mut step_plans: Vec<(usize, Vec<Option<ProvenanceDigest>>, Vec<usize>)> =
                Vec::with_capacity(stage.len());

            for &step_idx in stage {
                let step = self.steps[step_idx].as_ref();
                let backedges = &self.backedge_reads[step_idx];
                let mut decisions = Vec::with_capacity(n_docs);
                let mut misses = Vec::new();
                for (d_idx, doc) in docs.iter_mut().enumerate() {
                    let cached = if backedges.is_empty() {
                        can_skip_step(step, doc, &recomputed_keys[d_idx])
                    } else {
                        let taken = doc.take_entries(backedges);
                        let res = can_skip_step(step, doc, &recomputed_keys[d_idx]);
                        doc.restore_entries(taken);
                        res
                    };
                    if cached.is_none() {
                        misses.push(d_idx);
                    }
                    decisions.push(cached);
                }
                step_plans.push((step_idx, decisions, misses));
            }

            // Remove stale outputs for all (step_idx, d_idx) misses before computing.
            for &(step_idx, _, ref misses) in &step_plans {
                if !misses.is_empty() {
                    let writes = self.steps[step_idx].writes();
                    for &d_idx in misses {
                        for &w in &writes {
                            docs[d_idx].remove(w);
                        }
                    }
                }
            }

            // Fast path when every active step in this stage misses on all `docs`
            // (including single-document `analyze_incremental`).
            let all_full_batch = step_plans.iter().all(|(_, _, misses)| misses.is_empty() || misses.len() == n_docs);

            if all_full_batch {
                let to_run: Vec<usize> =
                    step_plans.iter().filter(|(_, _, misses)| !misses.is_empty()).map(|&(i, _, _)| i).collect();
                if !to_run.is_empty() {
                    let backedges: Vec<KeyId> =
                        to_run.iter().flat_map(|&i| self.backedge_reads[i].iter().copied()).collect();
                    let taken: Vec<_> = if backedges.is_empty() {
                        Vec::new()
                    } else {
                        docs.iter_mut().map(|d| d.take_entries(&backedges)).collect()
                    };
                    let outputs: Vec<Box<dyn Any + Send>> = if to_run.len() == 1 {
                        vec![self.steps[to_run[0]].compute(docs)]
                    } else {
                        let docs_ref: &[Document] = &*docs;
                        std::thread::scope(|s| {
                            let handles: Vec<_> =
                                to_run.iter().map(|&i| s.spawn(move || self.steps[i].compute(docs_ref))).collect();
                            handles.into_iter().map(|h| h.join().expect("an annotator's panics are caught")).collect()
                        })
                    };
                    for (&step_idx, out) in to_run.iter().zip(outputs) {
                        self.steps[step_idx].store(out, docs, &mut doc_status);
                        let writes = self.steps[step_idx].writes();
                        for (d_idx, doc) in docs.iter_mut().enumerate() {
                            for &w in &writes {
                                doc.mark_pipeline_stored(w);
                                recomputed_keys[d_idx].insert(w);
                            }
                        }
                    }
                    for (doc, entries) in docs.iter_mut().zip(taken) {
                        doc.restore_entries(entries);
                    }
                }
            } else {
                struct StepMissBatch {
                    step_idx: usize,
                    misses: Vec<usize>,
                    sub_docs: Vec<Document>,
                }
                // Partial-batch misses: coalesce each step's uncached documents into a contiguous batch.
                let mut batches: Vec<StepMissBatch> = Vec::new();
                for &(step_idx, _, ref misses) in &step_plans {
                    if misses.is_empty() {
                        continue;
                    }
                    let backedges = &self.backedge_reads[step_idx];
                    let mut sub_docs: Vec<Document> = misses.iter().map(|&d_idx| docs[d_idx].clone()).collect();
                    if !backedges.is_empty() {
                        for d in &mut sub_docs {
                            let _ = d.take_entries(backedges);
                        }
                    }
                    batches.push(StepMissBatch { step_idx, misses: misses.clone(), sub_docs });
                }

                let outputs: Vec<Box<dyn Any + Send>> = if batches.len() == 1 {
                    let b = &batches[0];
                    vec![self.steps[b.step_idx].compute(&b.sub_docs)]
                } else {
                    std::thread::scope(|s| {
                        let handles: Vec<_> = batches
                            .iter()
                            .map(|b| {
                                let step = self.steps[b.step_idx].as_ref();
                                let sub_docs = &b.sub_docs;
                                s.spawn(move || step.compute(sub_docs))
                            })
                            .collect();
                        handles.into_iter().map(|h| h.join().expect("an annotator's panics are caught")).collect()
                    })
                };

                // Store each step's outputs directly into the live `docs[d_idx]` in canonical
                // stage step order so multiple steps in the same stage that add derived `View`s
                // allocate distinct `ViewId`s on shared documents.
                for (b, out) in batches.into_iter().zip(outputs) {
                    let writes = self.steps[b.step_idx].writes();
                    let backedges = &self.backedge_reads[b.step_idx];
                    let mut sub_docs: Vec<Document> =
                        b.misses.iter().map(|&d_idx| std::mem::replace(&mut docs[d_idx], Document::new(""))).collect();
                    let taken: Vec<_> = if backedges.is_empty() {
                        Vec::new()
                    } else {
                        sub_docs.iter_mut().map(|d| d.take_entries(backedges)).collect()
                    };
                    let mut sub_status: Vec<Result<(), Error>> =
                        b.misses.iter().map(|&d_idx| doc_status[d_idx].clone()).collect();
                    self.steps[b.step_idx].store(out, &mut sub_docs, &mut sub_status);
                    for (sub_doc, entries) in sub_docs.iter_mut().zip(taken) {
                        sub_doc.restore_entries(entries);
                    }
                    for ((k_idx, d_idx), mut sub_doc) in b.misses.into_iter().enumerate().zip(sub_docs) {
                        for &w in &writes {
                            sub_doc.mark_pipeline_stored(w);
                            recomputed_keys[d_idx].insert(w);
                        }
                        docs[d_idx] = sub_doc;
                        doc_status[d_idx] = sub_status[k_idx].clone();
                    }
                }
            }

            // Record per-document stage reports.
            for (step_idx, decisions, _) in step_plans {
                let step = self.steps[step_idx].as_ref();
                let writes = step.writes();
                let backedges = &self.backedge_reads[step_idx];
                for (d_idx, cached_digest) in decisions.into_iter().enumerate() {
                    let (cache_status, provenance_digest) = match cached_digest {
                        Some(d) => {
                            stages_skipped[d_idx] += 1;
                            (StageCacheStatus::Skipped, d)
                        }
                        None => {
                            stages_run[d_idx] += 1;
                            let d =
                                writes.first().and_then(|&w| docs[d_idx].provenance_digest(w)).unwrap_or_else(|| {
                                    let taken = docs[d_idx].take_entries(backedges);
                                    let d =
                                        docs[d_idx].compute_provenance_digest(&step.expected_provenance(&docs[d_idx]));
                                    docs[d_idx].restore_entries(taken);
                                    d
                                });
                            (StageCacheStatus::Executed, d)
                        }
                    };
                    let outputs = writes
                        .iter()
                        .copied()
                        .map(|key| LayerExecution { key, status: docs[d_idx].status(key).cloned() })
                        .collect();
                    stage_reports[d_idx].push(StageIncrementalReport {
                        step: step.id(),
                        level,
                        cache_status,
                        provenance_digest,
                        outputs,
                    });
                }
            }
        }

        for doc in docs.iter_mut() {
            doc.entries.sort_by_key(|e| self.order_map.get(&e.key).copied().map_or((0, 0), |pos| (1, pos)));
        }

        doc_status
            .into_iter()
            .zip(stages_run)
            .zip(stages_skipped)
            .zip(stage_reports)
            .map(|(((res, run), skipped), stages)| {
                res?;
                Ok(IncrementalReport { stages_run: run, stages_skipped: skipped, stages })
            })
            .collect()
    }
}
