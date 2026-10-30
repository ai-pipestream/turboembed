//! The analyzer: a validated graph of steps, run over batches or streamed
//! progressively as individual DAG branches complete.

mod incremental;
mod locality;
mod progressive;
mod splice;

use std::any::TypeId;
use std::collections::{HashMap, HashSet};

pub use incremental::{IncrementalReport, StageCacheStatus, StageIncrementalReport};
pub use progressive::{StepEvent, StepTiming};
pub use splice::{SpliceReport, StepRun};

use crate::annotate::{AnchorUse, Annotator, Bound, Locality, Step};
use crate::doc::{Document, View};
use crate::error::Error;
use crate::layer::{Key, KeyId, Layer, LayerName};

mod report;
pub use report::{ExecutionReport, LayerExecution, StepExecution};

/// A pipeline. Steps are ordered by what they read and write, not by the
/// order they were added; steps that do not depend on each other form one
/// stage and compute side by side. A stage runs over the whole batch
/// before the next starts, so every step sees all of the batch's documents
/// at once.
pub struct Analyzer {
    given: Vec<KeyId>,
    steps: Vec<Box<dyn Step>>,
    stages: Vec<Vec<usize>>,
    pub(super) dag_error: Option<Error>,
    pub(super) backedge_reads: Vec<Vec<KeyId>>,
    pub(super) order_map: HashMap<KeyId, usize>,
    /// Each step's effective locality, as the pipeline bears it out.
    localities: Vec<Locality>,
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

    /// Each step's declared and effective locality, in step order: the
    /// two differ where the pipeline does not bear the declaration out
    /// (the step reads a document-wide layer), and the step then runs as
    /// `Locality::Document`.
    pub fn localities(&self) -> Vec<(&'static str, Locality, Locality)> {
        self.steps.iter().zip(&self.localities).map(|(s, &e)| (s.id(), s.locality(), e)).collect()
    }

    /// Runs every stage over the batch. A step that fails on a document,
    /// by an error or a panic, fails only its own layers there: the steps
    /// that need them are skipped and say why, the others still run, and
    /// no other document is touched. Each document's result is its first
    /// failure.
    pub fn run(&self, docs: &mut [Document]) -> Vec<Result<(), Error>> {
        let mut status: Vec<Result<(), Error>> = (0..docs.len()).map(|_| Ok(())).collect();
        for stage in &self.stages {
            self.run_steps(stage, docs, &mut status);
        }
        status
    }

    /// Computes `steps`, side by side when there are several, then stores
    /// their outputs in order. Layers a step reads from a later stage
    /// (`backedge_reads`) are taken off the documents meanwhile.
    pub(super) fn run_steps(&self, steps: &[usize], docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let backedges: Vec<KeyId> = steps.iter().flat_map(|&i| self.backedge_reads[i].iter().copied()).collect();
        let taken: Vec<_> = if backedges.is_empty() {
            Vec::new()
        } else {
            docs.iter_mut().map(|d| d.take_entries(&backedges)).collect()
        };
        for (&i, out) in steps.iter().zip(self.compute_steps(steps, docs)) {
            self.steps[i].store(out, docs, status);
            let writes = self.steps[i].writes();
            for doc in docs.iter_mut() {
                for &w in &writes {
                    doc.mark_pipeline_stored(w);
                }
            }
        }
        for (doc, entries) in docs.iter_mut().zip(taken) {
            doc.restore_entries(entries);
        }
    }

    /// The outputs of `steps` over the batch, computed side by side when
    /// there are several.
    pub(super) fn compute_steps(&self, steps: &[usize], docs: &[Document]) -> Vec<Box<dyn std::any::Any + Send>> {
        match steps {
            [] => Vec::new(),
            [one] => vec![self.steps[*one].compute(docs)],
            _ => std::thread::scope(|s| {
                let running: Vec<_> = steps.iter().map(|&i| s.spawn(move || self.steps[i].compute(docs))).collect();
                running.into_iter().map(|h| h.join().expect("an annotator's panics are caught")).collect()
            }),
        }
    }
}

fn build_order_map(stages: &[Vec<usize>], steps: &[Box<dyn Step>]) -> HashMap<KeyId, usize> {
    stages.iter().flatten().flat_map(|&i| steps[i].writes()).enumerate().map(|(pos, k)| (k, pos)).collect()
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

    /// Validates the configured steps without consuming the builder.
    pub fn validate_dag(&self) -> Result<(), Error> {
        validate_and_schedule(&self.given, &self.steps).map(|_| ())
    }

    /// Builds an [`Analyzer`] without running [`Analyzer::validate_dag`],
    /// placing each step in its insertion order so callers can invoke
    /// `analyzer.validate_dag()` explicitly.
    pub fn build_unvalidated(self) -> Analyzer {
        let stages: Vec<Vec<usize>> = (0..self.steps.len()).map(|i| vec![i]).collect();
        let (dag_error, backedge_reads) = match validate_and_schedule(&self.given, &self.steps) {
            Ok((_, backedges)) => (None, backedges),
            Err(e) => (Some(e), vec![Vec::new(); self.steps.len()]),
        };
        let order_map = build_order_map(&stages, &self.steps);
        let localities = locality::effective(&self.given, &self.steps, &stages);
        Analyzer { given: self.given, steps: self.steps, stages, dag_error, backedge_reads, order_map, localities }
    }

    /// Checks the graph and orders it: one layer type per id, every read
    /// with exactly one producer (a step or a given layer), every aligned
    /// output's anchor one of its step's reads or an output the step writes
    /// before it, and no cycle.
    pub fn build(self) -> Result<Analyzer, Error> {
        let (stages, backedge_reads) = validate_and_schedule(&self.given, &self.steps)?;
        let order_map = build_order_map(&stages, &self.steps);
        let localities = locality::effective(&self.given, &self.steps, &stages);
        Ok(Analyzer {
            given: self.given,
            steps: self.steps,
            stages,
            dag_error: None,
            backedge_reads,
            order_map,
            localities,
        })
    }
}

pub(super) type ScheduleResult = Result<(Vec<Vec<usize>>, Vec<Vec<KeyId>>), Error>;

pub(super) fn validate_and_schedule(given: &[KeyId], steps: &[Box<dyn Step>]) -> ScheduleResult {
    let mut types: HashMap<&LayerName, TypeId> = HashMap::new();
    let keys = steps.iter().flat_map(|s| s.reads().into_iter().chain(s.writes()));
    for k in given.iter().copied().chain(keys) {
        if *types.entry(k.layer).or_insert(k.ty) != k.ty {
            return Err(Error::Pipeline(format!("two layer types are both {}", k.layer)));
        }
    }
    let mut producer: HashMap<KeyId, Option<usize>> = given.iter().map(|&k| (k, None)).collect();
    for (i, s) in steps.iter().enumerate() {
        for w in s.writes() {
            if let Some(prev) = producer.insert(w, Some(i)) {
                let by = prev.map_or("the input".to_string(), |p| steps[p].id().to_string());
                return Err(Error::Pipeline(format!("{} and {} both write {w}", by, s.id())));
            }
        }
    }
    for s in steps {
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
    let mut level = vec![usize::MAX; steps.len()];
    let mut done: HashSet<KeyId> = given.iter().copied().collect();
    let mut stages = Vec::new();
    while level.contains(&usize::MAX) {
        let mut ready: Vec<usize> = (0..steps.len())
            .filter(|&i| {
                level[i] == usize::MAX && steps[i].reads().iter().all(|r| done.contains(r) || !producer.contains_key(r))
            })
            .collect();
        if ready.is_empty()
            && let Some(cycle_member) = (0..steps.len()).find(|&i| {
                level[i] == usize::MAX
                    && steps[i]
                        .reads()
                        .into_iter()
                        .zip(steps[i].required())
                        .all(|(r, req)| !req || done.contains(&r) || !producer.contains_key(&r))
                    && on_optional_cycle(i, steps, &level, &done, &producer)
            })
        {
            ready.push(cycle_member);
        }
        if ready.is_empty() {
            let stuck: Vec<_> = (0..steps.len()).filter(|&i| level[i] == usize::MAX).map(|i| steps[i].id()).collect();
            return Err(Error::Pipeline(format!("a cycle among {stuck:?}")));
        }
        for &i in &ready {
            level[i] = stages.len();
            done.extend(steps[i].writes());
        }
        stages.push(ready);
    }
    let backedge_reads = (0..steps.len())
        .map(|i| {
            steps[i]
                .reads()
                .into_iter()
                .zip(steps[i].required())
                .filter_map(|(r, req)| {
                    if !req
                        && let Some(Some(p)) = producer.get(&r)
                        && level[*p] >= level[i]
                    {
                        Some(r)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();
    Ok((stages, backedge_reads))
}

fn on_optional_cycle(
    candidate: usize,
    steps: &[Box<dyn Step>],
    level: &[usize],
    done: &HashSet<KeyId>,
    producer: &HashMap<KeyId, Option<usize>>,
) -> bool {
    let mut stack: Vec<usize> = steps[candidate]
        .reads()
        .into_iter()
        .zip(steps[candidate].required())
        .filter_map(|(r, req)| {
            if !req && !done.contains(&r) {
                producer.get(&r).copied().flatten().filter(|&p| level[p] == usize::MAX)
            } else {
                None
            }
        })
        .collect();
    let mut visited = vec![false; steps.len()];
    while let Some(cur) = stack.pop() {
        if cur == candidate {
            return true;
        }
        if visited[cur] {
            continue;
        }
        visited[cur] = true;
        for r in steps[cur].reads() {
            if !done.contains(&r)
                && let Some(Some(p)) = producer.get(&r)
                && level[*p] == usize::MAX
            {
                stack.push(*p);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use crate::{Analyzer, Annotator, Document, Error, Layer, Refs, View};

    struct LayerA(bool);
    impl Layer for LayerA {
        const ID: &'static str = "test:a";
        fn len(&self) -> usize {
            1
        }
        fn check(&self) -> Result<(), String> {
            Ok(())
        }
    }

    struct LayerB;
    impl Layer for LayerB {
        const ID: &'static str = "test:b";
        fn len(&self) -> usize {
            1
        }
        fn check(&self) -> Result<(), String> {
            Ok(())
        }
    }

    struct LayerC(bool);
    impl Layer for LayerC {
        const ID: &'static str = "test:c";
        fn len(&self) -> usize {
            1
        }
        fn check(&self) -> Result<(), String> {
            Ok(())
        }
    }

    struct ProduceA;
    impl Annotator for ProduceA {
        const ID: &'static str = "test:produce-a";
        type Reads = (View, Option<LayerB>);
        type Writes = (LayerA,);
        fn annotate(&self, (_, b): Refs<'_, Self>) -> Result<(LayerA,), Error> {
            Ok((LayerA(b.is_some()),))
        }
    }

    struct ProduceBFromA;
    impl Annotator for ProduceBFromA {
        const ID: &'static str = "test:produce-b";
        type Reads = (LayerA,);
        type Writes = (LayerB,);
        fn annotate(&self, (_a,): Refs<'_, Self>) -> Result<(LayerB,), Error> {
            Ok((LayerB,))
        }
    }

    struct BystanderC;
    impl Annotator for BystanderC {
        const ID: &'static str = "test:bystander-c";
        type Reads = (View, Option<LayerA>);
        type Writes = (LayerC,);
        fn annotate(&self, (_, a): Refs<'_, Self>) -> Result<(LayerC,), Error> {
            Ok((LayerC(a.is_some()),))
        }
    }

    #[test]
    fn optional_cycle_schedules_only_cycle_member_and_waits_for_bystanders() {
        let analyzer = Analyzer::builder().add(BystanderC).add(ProduceBFromA).add(ProduceA).build().unwrap();
        let mut docs = vec![Document::new("hello")];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        assert!(!docs[0].get::<LayerA>().unwrap().0);
        assert!(docs[0].get::<LayerB>().is_some());
        assert!(docs[0].get::<LayerC>().unwrap().0, "bystander must wait for LayerA");
    }

    #[test]
    fn optional_cycle_runs_in_progressive_mode() {
        let analyzer = Analyzer::builder().add(BystanderC).add(ProduceBFromA).add(ProduceA).build().unwrap();
        let mut docs = vec![Document::new("hello")];
        let cancel = AtomicBool::new(false);
        let mut events = Vec::new();
        for r in analyzer.run_progressive(&mut docs, &cancel, |ev| {
            events.push(ev.timing.step_id);
        }) {
            r.unwrap();
        }
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], ProduceA::ID);
        assert!(!docs[0].get::<LayerA>().unwrap().0);
        assert!(docs[0].get::<LayerB>().is_some());
        assert!(docs[0].get::<LayerC>().unwrap().0, "bystander must wait for LayerA");
    }
}
