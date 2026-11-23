//! What each document and each analysis step came to (issue #2517 F11).
//! A stream answers every document with its own outcome inside one OK
//! call, so the request counter alone showed a run in which every
//! document failed as one healthy request. These counters count each
//! answer by outcome and turbo status, and each progressive step by its
//! outcome.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::sync::Mutex;

use crate::proto::result::analyze_progressive_event::Event;
use crate::proto::result::analyze_stream_response::Outcome;
use crate::proto::result::{AnalyzeProgressiveEvent, AnalyzeStreamResponse};

/// What became of one progressive step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum StepOutcome {
    Ok,
    Failed,
    /// Not run, because a step it depends on failed.
    Skipped,
}

impl StepOutcome {
    fn name(self) -> &'static str {
        match self {
            StepOutcome::Ok => "ok",
            StepOutcome::Failed => "failed",
            StepOutcome::Skipped => "skipped",
        }
    }
}

/// The counts, keyed by their labels.
#[derive(Default)]
pub(super) struct Outcomes {
    /// (method, failed, turbo code) to answers.
    documents: Mutex<BTreeMap<(&'static str, bool, i32), u64>>,
    /// (step, outcome, turbo code) to steps.
    steps: Mutex<BTreeMap<(String, StepOutcome, i32), u64>>,
}

/// The turbo status's name, `TURBO_OK` for 0.
fn code_name(code: i32) -> &'static str {
    turbo::status::name(code).to_str().unwrap_or("TURBO_E_UNKNOWN")
}

/// A label value with `\` and `"` escaped, and a line break written as
/// `\n`, as the text exposition format says.
fn escaped(v: &str) -> String {
    v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

impl Outcomes {
    pub(super) fn document(&self, method: &'static str, turbo_code: Option<i32>) {
        let key = (method, turbo_code.is_some(), turbo_code.unwrap_or(0));
        *self.documents.lock().unwrap_or_else(|e| e.into_inner()).entry(key).or_insert(0) += 1;
    }

    pub(super) fn step(&self, step: &str, outcome: StepOutcome, turbo_code: i32) {
        let mut steps = self.steps.lock().unwrap_or_else(|e| e.into_inner());
        *steps.entry((step.to_owned(), outcome, turbo_code)).or_insert(0) += 1;
    }

    pub(super) fn render(&self, out: &mut String) {
        let _ = writeln!(
            out,
            "# HELP turbo_kserve_documents_total Documents answered inside a streaming call, by method, outcome, and turbo status."
        );
        let _ = writeln!(out, "# TYPE turbo_kserve_documents_total counter");
        for (&(method, failed, code), n) in self.documents.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            let outcome = if failed { "failed" } else { "ok" };
            let code = code_name(code);
            let _ = writeln!(
                out,
                "turbo_kserve_documents_total{{method=\"{method}\",outcome=\"{outcome}\",turbo_code=\"{code}\"}} {n}"
            );
        }
        let _ = writeln!(
            out,
            "# HELP turbo_kserve_steps_total Progressive analysis steps by step, outcome, and turbo status; a skipped step carries the status of the failure it depends on."
        );
        let _ = writeln!(out, "# TYPE turbo_kserve_steps_total counter");
        for ((step, outcome, code), n) in self.steps.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            let (step, outcome, code) = (escaped(step), outcome.name(), code_name(*code));
            let _ = writeln!(
                out,
                "turbo_kserve_steps_total{{step=\"{step}\",outcome=\"{outcome}\",turbo_code=\"{code}\"}} {n}"
            );
        }
    }
}

/// A stream answer's failure status, None when it succeeded.
pub(crate) fn failure_code(r: &AnalyzeStreamResponse) -> Option<i32> {
    match &r.outcome {
        Some(Outcome::Failure(f)) => Some(f.turbo_code),
        _ => None,
    }
}

/// What one progressive run has counted, so each step is counted once
/// (#2517 F11). A failure names the steps it leaves unrun; one of them
/// that still waits on another input runs later and stores skipped
/// outputs, and that later event must not count it again, or as ok.
#[derive(Default)]
pub(crate) struct StepTally {
    skipped: HashSet<String>,
}

impl StepTally {
    /// Counts a step's event: a completed step, a step that stored
    /// skipped outputs (`wrote_skipped`), or a failed one and each step
    /// it left unrun.
    pub(crate) fn count(&mut self, metrics: &super::Metrics, msg: &AnalyzeProgressiveEvent, wrote_skipped: bool) {
        match &msg.event {
            Some(Event::LayersReady(b)) if !wrote_skipped => metrics.record_step(&b.completed_step, StepOutcome::Ok, 0),
            // A step that declined on its own carries no failure's status.
            Some(Event::LayersReady(b)) => self.skip(metrics, &b.completed_step, 0),
            Some(Event::StepFailed(f)) => {
                let code = f.failure.as_ref().map_or(crate::api::TURBO_E_INTERNAL, |x| x.turbo_code);
                metrics.record_step(&f.step_id, StepOutcome::Failed, code);
                for s in &f.skipped_dependents {
                    self.skip(metrics, s, code);
                }
            }
            _ => {}
        }
    }

    fn skip(&mut self, metrics: &super::Metrics, step: &str, code: i32) {
        if self.skipped.insert(step.to_owned()) {
            metrics.record_step(step, StepOutcome::Skipped, code);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_outcome_is_its_own_series() {
        let o = Outcomes::default();
        o.document("AnalyzeStream", None);
        o.document("AnalyzeStream", Some(crate::api::TURBO_E_INTERNAL));
        o.document("AnalyzeStream", Some(crate::api::TURBO_E_INTERNAL));
        o.step("turbo:chunk-vectors", StepOutcome::Failed, crate::api::TURBO_E_INTERNAL);
        o.step("a\"b", StepOutcome::Ok, 0);
        let mut out = String::new();
        o.render(&mut out);
        for line in [
            "turbo_kserve_documents_total{method=\"AnalyzeStream\",outcome=\"ok\",turbo_code=\"TURBO_OK\"} 1",
            "turbo_kserve_documents_total{method=\"AnalyzeStream\",outcome=\"failed\",turbo_code=\"TURBO_E_INTERNAL\"} 2",
            "turbo_kserve_steps_total{step=\"turbo:chunk-vectors\",outcome=\"failed\",turbo_code=\"TURBO_E_INTERNAL\"} 1",
            "turbo_kserve_steps_total{step=\"a\\\"b\",outcome=\"ok\",turbo_code=\"TURBO_OK\"} 1",
        ] {
            assert!(out.lines().any(|l| l == line), "no {line} in\n{out}");
        }
    }
}
