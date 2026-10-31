//! The sentence boundary, token split, span classifier, coreference
//! ranker and relation classifier heads' own fields and rules, beside the
//! rules every head meets (heads.rs).

use serde::{Deserialize, Serialize};

use super::heads::{Decode, HeadKind, HeadSpec, MAX_BIAFFINE, MAX_LABELS, Scheme};
use super::helpers::{fits, positive, required};
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

/// The label of a HEAD_RELATION_CLASSIFIER for a pair in no relation.
pub const NO_RELATION: &str = "none";
/// The widest entity type embedding a relation classifier reads.
pub const MAX_TYPE_DIM: u32 = 256;

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

/// What a relation classifier reads of each entity: its mentions' first
/// and last word vectors, pooled, and, when `entity_labels` is not empty,
/// a learned embedding `type_dim` wide of its type, through two one-layer
/// MLPs `mlp` wide, its view as a relation's head and as its tail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationShape {
    pub mlp: u32,
    /// The entity types an entity carries, as the entity head that found
    /// it labels them; empty when the head reads no type.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entity_labels: Vec<String>,
    /// 0 exactly when `entity_labels` is empty.
    #[serde(default)]
    pub type_dim: u32,
}

impl HeadSpec {
    /// Whether the head's scheme and decode are ones its kind takes.
    pub(super) fn decode_fits_kind(&self) -> bool {
        let tags = self.scheme == Scheme::Tags;
        match self.kind {
            HeadKind::TokenTagger => matches!(self.decode, Decode::Argmax | Decode::Crf),
            HeadKind::BiaffineParser => matches!(self.decode, Decode::Mst | Decode::Eisner) && tags,
            HeadKind::EditTreeLemmatizer | HeadKind::TokenSplit => self.decode == Decode::EditTree && tags,
            HeadKind::SentenceBoundary => matches!(self.decode, Decode::Argmax | Decode::Crf) && tags,
            HeadKind::SpanClassifier => matches!(self.decode, Decode::SpansNested | Decode::SpansFlat) && tags,
            HeadKind::CorefRanker => self.decode == Decode::Antecedent && tags,
            HeadKind::RelationClassifier => self.decode == Decode::Relations && tags,
        }
    }

    /// What a head of this kind needs beyond `validate`'s kind, scheme and
    /// decode check: the boundary labels, the split labels, the span,
    /// coref and relation shapes. `at` gives the field path an error names.
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
        if (self.kind == HeadKind::RelationClassifier) != self.relation.is_some() {
            return Err(invalid(format!(
                "manifest.json: {}: given exactly when the kind is HEAD_RELATION_CLASSIFIER",
                at("relation")
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
            HeadKind::RelationClassifier => self.validate_relation(at),
            _ => Ok(()),
        }
    }

    fn validate_relation(&self, at: &dyn Fn(&str) -> String) -> Result<()> {
        let Some(r) = &self.relation else { return Ok(()) };
        if !self.labels.iter().any(|l| l == NO_RELATION) {
            return Err(invalid(format!(
                "manifest.json: {}: a HEAD_RELATION_CLASSIFIER needs the label {NO_RELATION:?}",
                at("labels")
            )));
        }
        positive(&at("relation.mlp"), r.mlp)?;
        let (types, dim) = (r.entity_labels.len(), r.type_dim);
        if r.mlp > MAX_BIAFFINE || types > MAX_LABELS || dim > MAX_TYPE_DIM || (types == 0) != (dim == 0) {
            return Err(invalid(format!(
                "manifest.json: {}: mlp {}, {types} entity labels and type_dim {dim}; mlp at most {MAX_BIAFFINE}, \
                 at most {MAX_LABELS} entity labels, and type_dim 1 to {MAX_TYPE_DIM} with entity labels, 0 without",
                at("relation"),
                r.mlp
            )));
        }
        for (j, l) in r.entity_labels.iter().enumerate() {
            let field = at(&format!("relation.entity_labels[{j}]"));
            required(&field, l)?;
            fits(&field, l, 128)?;
            if r.entity_labels[..j].contains(l) {
                return Err(invalid(format!("manifest.json: {field}: {l:?} is listed twice")));
            }
        }
        Ok(())
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
