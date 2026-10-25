//! Output statuses from an actual Analyzer run, in pipeline stage order.

use crate::{Document, Error, KeyId, LayerStatus};

use super::Analyzer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerExecution {
    pub key: KeyId,
    /// None means the step left no record for this output. It differs from
    /// an explicitly skipped output and is never reported as Complete.
    pub status: Option<LayerStatus>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepExecution {
    pub step: &'static str,
    pub outputs: Vec<LayerExecution>,
}

/// Per-document output statuses of this run. Input/given layers are omitted.
/// This report records no invented timings or inferred execution durations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionReport {
    pub steps: Vec<StepExecution>,
}

impl Analyzer {
    /// Runs the pipeline once and reports actual complete, skipped and failed
    /// output records for each document, including skipped conditional steps.
    pub fn run_with_report(&self, docs: &mut [Document]) -> Vec<(Result<(), Error>, ExecutionReport)> {
        let results = self.run(docs);
        results
            .into_iter()
            .zip(docs)
            .map(|(result, doc)| {
                let steps = self
                    .stages
                    .iter()
                    .flatten()
                    .map(|&i| {
                        let step = &self.steps[i];
                        StepExecution {
                            step: step.id(),
                            outputs: step
                                .writes()
                                .into_iter()
                                .map(|key| LayerExecution { key, status: doc.status(key).cloned() })
                                .collect(),
                        }
                    })
                    .collect();
                (result, ExecutionReport { steps })
            })
            .collect()
    }
}
