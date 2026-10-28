//! The sentence boundary, token split, span classifier and coreference
//! ranker heads' own fields and rules, beside the rules every head meets
//! (heads.rs).

use serde::{Deserialize, Serialize};

use super::heads::{HeadKind, HeadSpec};
use super::helpers::positive;
use crate::status::{Result, invalid};

/// The most words one span of a span classifier covers.
pub const MAX_SPAN_WORDS: u32 = 128;
/// The widest span-width embedding.
pub const MAX_SPAN_WIDTH_DIM: u32 = 256;

/// The labels of a HEAD_SENTENCE_BOUNDARY, in this order.
pub const BOUNDARY_LABELS: [&str; 2] = ["inside", "last"];
/// The label of a HEAD_TOKEN_SPLIT that keeps a token one word.
pub const KEEP_TOKEN: &str = "_";

/// The most antecedents one mention is scored against.
pub const MAX_ANTECEDENTS: u32 = 1024;
/// The widest mention vector a coreference ranker scores.
pub const MAX_MENTION_DIM: u32 = 4096;
/// The labels of a HEAD_COREF_RANKER.
pub const COREF_LABELS: [&str; 1] = ["coref"];

/// How a coreference ranker scores a pair: each mention's vector is
/// `dim` wide, and a mention is scored against at most `antecedents`
/// mentions before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorefShape {
    pub dim: u32,
    pub antecedents: u32,
}

/// The spans a span classifier reads: every run of 1 to `max_words`
/// words, each with a learned embedding `width_dim` wide of its width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpanShape {
    pub max_words: u32,
    pub width_dim: u32,
}

impl HeadSpec {
    /// What a head of this kind needs beyond `validate`'s kind, scheme and
    /// decode check: the boundary labels, the split labels, the span
    /// shape. `at` gives the field path an error names.
    pub(super) fn validate_kind_rules(&self, at: &dyn Fn(&str) -> String) -> Result<()> {
        let coref = self.kind == HeadKind::CorefRanker;
        let spans = self.kind == HeadKind::SpanClassifier || coref;
        if spans != self.span.is_some() {
            return Err(invalid(format!(
                "manifest.json: {}: given exactly when the kind is HEAD_SPAN_CLASSIFIER or HEAD_COREF_RANKER",
                at("span")
            )));
        }
        if coref != self.coref.is_some() {
            return Err(invalid(format!(
                "manifest.json: {}: given exactly when the kind is HEAD_COREF_RANKER",
                at("coref")
            )));
        }
        if let Some(s) = self.span {
            check_span(at, s)?;
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
            HeadKind::SpanClassifier if !self.labels.iter().any(|l| l == "O") => {
                Err(invalid(format!("manifest.json: {}: a HEAD_SPAN_CLASSIFIER needs the label \"O\"", at("labels"))))
            }
            HeadKind::CorefRanker => {
                if self.labels != COREF_LABELS || self.hash_embed.is_some() {
                    return Err(invalid(format!(
                        "manifest.json: {}: {:?}; a HEAD_COREF_RANKER's labels are {COREF_LABELS:?} and it takes no \
                         hash_embed",
                        at("labels"),
                        self.labels
                    )));
                }
                let Some(c) = self.coref else { return Ok(()) };
                positive(&at("coref.dim"), c.dim)?;
                positive(&at("coref.antecedents"), c.antecedents)?;
                if c.dim > MAX_MENTION_DIM || c.antecedents > MAX_ANTECEDENTS {
                    return Err(invalid(format!(
                        "manifest.json: {}: dim {} and antecedents {}; at most {MAX_MENTION_DIM} and {MAX_ANTECEDENTS}",
                        at("coref"),
                        c.dim,
                        c.antecedents
                    )));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// A span shape's own bounds.
fn check_span(at: &dyn Fn(&str) -> String, s: SpanShape) -> Result<()> {
    positive(&at("span.max_words"), s.max_words)?;
    positive(&at("span.width_dim"), s.width_dim)?;
    if s.max_words > MAX_SPAN_WORDS || s.width_dim > MAX_SPAN_WIDTH_DIM {
        return Err(invalid(format!(
            "manifest.json: {}: max_words {} and width_dim {}; at most {MAX_SPAN_WORDS} and {MAX_SPAN_WIDTH_DIM}",
            at("span"),
            s.max_words,
            s.width_dim
        )));
    }
    Ok(())
}
