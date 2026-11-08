//! Audit reporting for the PII detector.

use turbo_analysis::ViewId;
use turbo_analysis_layers::Pii;

use super::PiiDetector;
use crate::rewrite::{HmacTokenizer, PiiAuditReport};

impl PiiDetector {
    /// Runs detection over `text` for `view` and builds a [`PiiAuditReport`]
    /// from the mentions and the candidate decisions.
    pub fn detect_and_audit(
        &self,
        text: &str,
        view: ViewId,
        tokenizer: &HmacTokenizer,
        samples: usize,
    ) -> (Pii, PiiAuditReport) {
        let (pii, candidates) = self.layer_with_candidates(text, view);
        let report = PiiAuditReport::with_candidates(&pii, &candidates, tokenizer, samples);
        (pii, report)
    }

    /// Runs detection over `text` and returns the [`PiiAuditReport`],
    /// including per-kind counts of mentions suppressed by allow rules.
    pub fn audit(&self, text: &str, tokenizer: &HmacTokenizer, samples: usize) -> PiiAuditReport {
        self.detect_and_audit(text, ViewId::SOURCE, tokenizer, samples).1
    }
}
