//! The analyzer: a validated graph of steps, run over batches.

use std::any::TypeId;
use std::collections::{HashMap, HashSet};

use crate::annotate::{AnchorUse, Annotator, Bound, Step};
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
