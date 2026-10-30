//! The rules of the sentence boundary, token split and span classifier
//! heads (#1516), beside the rules every head meets (heads.rs).

use super::heads::{BOUNDARY_LABELS, HeadKind, HeadSpec, KEEP_TOKEN, MAX_SPAN_WIDTH_DIM, MAX_SPAN_WORDS};
use super::helpers::positive;
use crate::status::{Result, invalid};

impl HeadSpec {
    /// What a head of this kind needs beyond `validate`'s kind, scheme and
    /// decode check: the boundary labels, the split labels, the span
    /// shape. `at` gives the field path an error names.
    pub(super) fn validate_kind_rules(&self, at: &dyn Fn(&str) -> String) -> Result<()> {
        let spans = self.kind == HeadKind::SpanClassifier;
        if spans != self.span.is_some() {
            return Err(invalid(format!(
                "manifest.json: {}: given exactly when the kind is HEAD_SPAN_CLASSIFIER",
                at("span")
            )));
        }
        match self.kind {
            HeadKind::SentenceBoundary if self.labels != BOUNDARY_LABELS => Err(invalid(format!(
                "manifest.json: {}: {:?}; a HEAD_SENTENCE_BOUNDARY's labels are {BOUNDARY_LABELS:?}",
                at("labels"),
                self.labels
            ))),
            HeadKind::TokenSplit => {
                if !self.labels.iter().any(|l| l == KEEP_TOKEN) {
                    return Err(invalid(format!(
                        "manifest.json: {}: a HEAD_TOKEN_SPLIT needs the label {KEEP_TOKEN:?}",
                        at("labels")
                    )));
                }
                for (j, l) in self.labels.iter().enumerate().filter(|(_, l)| *l != KEEP_TOKEN) {
                    crate::head::split::parse(l).map_err(|e| {
                        invalid(format!("manifest.json: {}: {}", at(&format!("labels[{j}]")), e.message))
                    })?;
                }
                Ok(())
            }
            HeadKind::SpanClassifier => {
                let Some(s) = self.span else { return Ok(()) };
                if !self.labels.iter().any(|l| l == "O") {
                    return Err(invalid(format!(
                        "manifest.json: {}: a HEAD_SPAN_CLASSIFIER needs the label \"O\"",
                        at("labels")
                    )));
                }
                positive(&at("span.max_words"), s.max_words)?;
                positive(&at("span.width_dim"), s.width_dim)?;
                if s.max_words > MAX_SPAN_WORDS || s.width_dim > MAX_SPAN_WIDTH_DIM {
                    return Err(invalid(format!(
                        "manifest.json: {}: max_words {} and width_dim {}; at most {MAX_SPAN_WORDS} and \
                         {MAX_SPAN_WIDTH_DIM}",
                        at("span"),
                        s.max_words,
                        s.width_dim
                    )));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
