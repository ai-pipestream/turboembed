//! The rules of the sentence boundary, token split, span classifier and
//! open-label span typing heads (#1516, #1735), beside the rules every
//! head meets (heads.rs).

use super::heads::{
    BOUNDARY_LABELS, Decode, HeadKind, HeadSpec, KEEP_TOKEN, MAX_SPAN_WIDTH_DIM, MAX_SPAN_WORDS, Scheme,
};
use super::helpers::positive;
use crate::status::{Result, invalid};

impl HeadSpec {
    pub(super) fn validate_kind_decode(&self, at: &dyn Fn(&str) -> String) -> Result<()> {
        let fits_kind = match self.kind {
            HeadKind::TokenTagger => matches!(self.decode, Decode::Argmax | Decode::Crf),
            HeadKind::BiaffineParser => {
                matches!(self.decode, Decode::Mst | Decode::Eisner) && self.scheme == Scheme::Tags
            }
            HeadKind::EditTreeLemmatizer => self.decode == Decode::EditTree && self.scheme == Scheme::Tags,
            HeadKind::SentenceBoundary => {
                matches!(self.decode, Decode::Argmax | Decode::Crf) && self.scheme == Scheme::Tags
            }
            HeadKind::TokenSplit => self.decode == Decode::EditTree && self.scheme == Scheme::Tags,
            HeadKind::SpanClassifier | HeadKind::OpenSpan => {
                matches!(self.decode, Decode::SpansNested | Decode::SpansFlat) && self.scheme == Scheme::Tags
            }
        };
        let parser = self.kind == HeadKind::BiaffineParser;
        let lemmatizer = self.kind == HeadKind::EditTreeLemmatizer;
        if !fits_kind || parser != self.biaffine.is_some() || (!lemmatizer && self.pos_input.is_some()) {
            return Err(invalid(format!(
                "manifest.json: {}: {:?} with {:?}, {:?}, biaffine {} and pos_input {}: a HEAD_TOKEN_TAGGER takes \
                 DECODE_ARGMAX or DECODE_CRF; a HEAD_BIAFFINE_PARSER takes SCHEME_TAGS, DECODE_MST or DECODE_EISNER and \
                 biaffine; a HEAD_EDIT_TREE_LEMMATIZER takes SCHEME_TAGS, DECODE_EDIT_TREE and an optional pos_input; \
                 a HEAD_SENTENCE_BOUNDARY takes SCHEME_TAGS and DECODE_ARGMAX or DECODE_CRF; a HEAD_TOKEN_SPLIT takes \
                 SCHEME_TAGS and DECODE_EDIT_TREE; a HEAD_SPAN_CLASSIFIER takes SCHEME_TAGS and DECODE_SPANS_NESTED or \
                 DECODE_SPANS_FLAT; a HEAD_OPEN_SPAN takes SCHEME_TAGS and DECODE_SPANS_NESTED or DECODE_SPANS_FLAT",
                at("kind"),
                self.kind,
                self.scheme,
                self.decode,
                if self.biaffine.is_some() { "set" } else { "absent" },
                if self.pos_input.is_some() { "set" } else { "absent" }
            )));
        }
        Ok(())
    }

    /// What a head of this kind needs beyond `validate`'s kind, scheme and
    /// decode check: the boundary labels, the split labels, the span
    /// shape. `at` gives the field path an error names.
    pub(super) fn validate_kind_rules(&self, at: &dyn Fn(&str) -> String) -> Result<()> {
        let spans = matches!(self.kind, HeadKind::SpanClassifier | HeadKind::OpenSpan);
        if spans != self.span.is_some() {
            return Err(invalid(format!(
                "manifest.json: {}: given exactly when the kind is HEAD_SPAN_CLASSIFIER or HEAD_OPEN_SPAN",
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
            HeadKind::SpanClassifier | HeadKind::OpenSpan => {
                let Some(s) = self.span else { return Ok(()) };
                if self.kind == HeadKind::SpanClassifier && !self.labels.iter().any(|l| l == "O") {
                    return Err(invalid(format!(
                        "manifest.json: {}: a HEAD_SPAN_CLASSIFIER needs the label \"O\"",
                        at("labels")
                    )));
                }
                if self.kind == HeadKind::OpenSpan && self.labels.iter().any(|l| l == "O") {
                    return Err(invalid(format!(
                        "manifest.json: {}: a HEAD_OPEN_SPAN lists only entity types, not \"O\"",
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
