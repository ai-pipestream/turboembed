//! Content-addressed incremental DAG execution, build-time DAG validation,
//! and sequential/parallel execution helpers for [`Analyzer`].

use std::collections::{HashMap, HashSet};

use crate::annotate::Step;
use crate::doc::{Document, Status};
use crate::error::Error;
use crate::layer::KeyId;

use super::{Analyzer, LayerExecution, validate_and_schedule};

/// Cache decision for a pipeline step during [`Analyzer::analyze_incremental`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageCacheStatus {
    /// All produced keys were already present on the document with matching
    /// chained provenance digests and upstream content hashes, so the step
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
    /// Chained provenance digest computed for this step's produced layers on the document.
    pub provenance_digest: u64,
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

fn can_skip_step(step: &dyn Step, doc: &Document, recomputed_keys: &HashSet<KeyId>) -> Option<u64> {
    if step.reads().iter().any(|r| recomputed_keys.contains(r)) {
        return None;
    }
    let writes = step.writes();
    if writes.is_empty() {
        return None;
    }
    if !writes.iter().all(|&w| doc.present(w) && matches!(doc.status(w), Some(Status::Complete))) {
        return None;
    }
    let expected_prov = step.expected_provenance(doc);
    let expected_digest = doc.compute_provenance_digest(&expected_prov);
    let expected_upstream_content = doc.compute_upstream_content_hash(&expected_prov.reads);
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
    /// Validates the pipeline DAG: verifies that every layer name maps to a single
    /// Rust type, no two steps write the same key, every required upstream layer
    /// (`Annotator::requires` / `Reads`) has a producer, every aligned output's
    /// anchor is read or written earlier by the same step, and there are no cycles.
    pub fn validate_dag(&self) -> Result<(), Error> {
        validate_and_schedule(&self.given, &self.steps).map(|_| ())
    }

    /// Executes the pipeline DAG sequentially (one step at a time in topological
    /// stage order without spawning worker threads), useful as a deterministic
    /// reference against parallel stage execution.
    pub fn run_sequential(&self, docs: &mut [Document]) -> Vec<Result<(), Error>> {
        let mut status: Vec<Result<(), Error>> = (0..docs.len()).map(|_| Ok(())).collect();
        for stage in &self.stages {
            for &i in stage {
                let out = self.steps[i].compute(docs);
                self.steps[i].store(out, docs, &mut status);
            }
        }
        status
    }

    /// Runs content-addressed incremental DAG execution over a single [`Document`]:
    ///
    /// - Validates the pipeline DAG (`validate_dag`).
    /// - For each topological stage level, checks in $O(1)$ whether each step's
    ///   `produces()` keys are already present on `doc` with matching chained
    ///   provenance digests and upstream content hashes.
    /// - Skips matching steps in $O(1)` without re-running `Annotator::annotate`.
    /// - Executes remaining steps in the same topological level concurrently via
    ///   `std::thread::scope`, invalidating and recomputing all transitive
    ///   downstream dependents when any upstream step's `config_hash`, `model`,
    ///   or output changes.
    /// - Restores canonical topological layer ordering on `doc`.
    pub fn analyze_incremental(&self, doc: &mut Document) -> Result<IncrementalReport, Error> {
        self.validate_dag()?;
        let mut status: [Result<(), Error>; 1] = [Ok(())];
        let mut recomputed_keys: HashSet<KeyId> = HashSet::new();
        let mut stages_run = 0usize;
        let mut stages_skipped = 0usize;
        let mut stage_reports = Vec::with_capacity(self.steps.len());

        for (level, stage) in self.stages.iter().enumerate() {
            let mut to_run = Vec::new();
            let mut step_decisions: Vec<(usize, Option<u64>)> = Vec::with_capacity(stage.len());

            for &step_idx in stage {
                let step = self.steps[step_idx].as_ref();
                if let Some(digest) = can_skip_step(step, doc, &recomputed_keys) {
                    step_decisions.push((step_idx, Some(digest)));
                } else {
                    to_run.push(step_idx);
                    step_decisions.push((step_idx, None));
                }
            }

            if !to_run.is_empty() {
                for &step_idx in &to_run {
                    for w in self.steps[step_idx].writes() {
                        doc.remove(w);
                    }
                }

                let outputs: Vec<_> = if to_run.len() == 1 {
                    vec![self.steps[to_run[0]].compute(std::slice::from_ref(doc))]
                } else {
                    let doc_ref: &[Document] = std::slice::from_ref(&*doc);
                    std::thread::scope(|s| {
                        let handles: Vec<_> =
                            to_run.iter().map(|&i| s.spawn(move || self.steps[i].compute(doc_ref))).collect();
                        handles.into_iter().map(|h| h.join().expect("an annotator's panics are caught")).collect()
                    })
                };

                for (&step_idx, out) in to_run.iter().zip(outputs) {
                    self.steps[step_idx].store(out, std::slice::from_mut(doc), &mut status);
                    for w in self.steps[step_idx].writes() {
                        recomputed_keys.insert(w);
                    }
                }
            }

            for (step_idx, cached_digest) in step_decisions {
                let step = self.steps[step_idx].as_ref();
                let writes = step.writes();
                let (cache_status, provenance_digest) = match cached_digest {
                    Some(d) => {
                        stages_skipped += 1;
                        (StageCacheStatus::Skipped, d)
                    }
                    None => {
                        stages_run += 1;
                        let d = writes
                            .first()
                            .and_then(|&w| doc.provenance_digest(w))
                            .unwrap_or_else(|| doc.compute_provenance_digest(&step.expected_provenance(doc)));
                        (StageCacheStatus::Executed, d)
                    }
                };
                let outputs =
                    writes.into_iter().map(|key| LayerExecution { key, status: doc.status(key).cloned() }).collect();
                stage_reports.push(StageIncrementalReport {
                    step: step.id(),
                    level,
                    cache_status,
                    provenance_digest,
                    outputs,
                });
            }
        }

        // Place pipeline-written entries in canonical stage order so incremental
        // updates and parallel branch runs produce byte-identical DocumentGraphs.
        let canonical_keys: Vec<KeyId> = self.stages.iter().flatten().flat_map(|&i| self.steps[i].writes()).collect();
        let order_map: HashMap<KeyId, usize> =
            canonical_keys.into_iter().enumerate().map(|(pos, k)| (k, pos)).collect();
        doc.entries.sort_by_key(|e| order_map.get(&e.key).copied().map_or((0, 0), |pos| (1, pos)));

        let [res] = status;
        res?;
        Ok(IncrementalReport { stages_run, stages_skipped, stages: stage_reports })
    }

    /// Runs [`Analyzer::analyze_incremental`] across a batch of documents.
    pub fn run_incremental(&self, docs: &mut [Document]) -> Vec<Result<IncrementalReport, Error>> {
        docs.iter_mut().map(|doc| self.analyze_incremental(doc)).collect()
    }
}
