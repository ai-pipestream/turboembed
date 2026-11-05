//! The sentence boundary, token split, span classifier, open-label span
//! typing (#1516, #1735), coreference ranker and relation classifier
//! heads' own fields and rules, beside the rules every head meets (heads.rs).

use serde::{Deserialize, Serialize};

use super::heads::{
    Decode, HeadKind, HeadSpec, MAX_ADAPTER_HIDDEN, MAX_ADAPTER_LAYERS, MAX_BIAFFINE, MAX_HASH_DIM, MAX_HASH_ROWS,
    MAX_LABELS, MAX_POS_DIM, Scheme,
};
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
/// The most encoder layers a head's scalar mix combines.
pub const MAX_MIX_LAYERS: u32 = 128;
/// The most entries a token split head's seen lookup holds.
pub const MAX_SEEN_TOKENS: usize = 1 << 16;

/// The languages a word head conditions on, and the embedding width.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageInput {
    pub languages: Vec<String>,
    pub dim: u32,
}

pub(super) fn validate_language(kind: HeadKind, language: &LanguageInput, at: &dyn Fn(&str) -> String) -> Result<()> {
    positive(&at("language.dim"), language.dim)?;
    if language.languages.is_empty() || language.languages.len() > 128 || language.dim > MAX_POS_DIM {
        return Err(invalid(format!(
            "manifest.json: {}: {} languages and dim {}, at most 128 and {MAX_POS_DIM}",
            at("language"),
            language.languages.len(),
            language.dim
        )));
    }
    if kind != HeadKind::TokenTagger {
        return Err(invalid(format!("manifest.json: {}: {kind:?} does not take a language input", at("language"))));
    }
    for (i, name) in language.languages.iter().enumerate() {
        required(&at(&format!("language.languages[{i}]")), name)?;
        fits(&at(&format!("language.languages[{i}]")), name, 64)?;
        if language.languages[..i].contains(name) {
            return Err(invalid(format!("manifest.json: {}: {name:?} is listed twice", at("language.languages"))));
        }
    }
    Ok(())
}

/// One entry of a token split head's seen lookup: a token whose every
/// training occurrence split the same way, and the label of that split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeenToken {
    pub token: String,
    pub label: String,
}

/// A token split head's seen-token lookup (#1803): the tokens whose
/// training split was unambiguous over at least `min_count` occurrences,
/// each with that split's label. A looked-up token takes its label
/// without the scorer being read; the head decides every other token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeenLookup {
    pub min_count: u32,
    pub tokens: Vec<SeenToken>,
}

/// A learned scalar mix (ELMo, Peters et al. 2018) of the encoder's last
/// `layers` layers (`gamma * sum_k softmax(weight)_k * tap_k`, with
/// tensors `mix.weight` `[layers]` and `mix.gamma` `[1]`), or a single
/// chosen 1-based encoder layer `layer` (`architecture.layers - layers + 1`)
/// tapped directly without `mix.weight` or `mix.gamma`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerMix {
    pub layers: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,
}

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
    pub(super) fn validate_kind_decode(&self, at: &dyn Fn(&str) -> String) -> Result<()> {
        let tags = self.scheme == Scheme::Tags;
        let fits_kind = match self.kind {
            HeadKind::TokenTagger => matches!(self.decode, Decode::Argmax | Decode::Crf),
            HeadKind::BiaffineParser => matches!(self.decode, Decode::Mst | Decode::Eisner) && tags,
            HeadKind::EditTreeLemmatizer | HeadKind::TokenSplit => self.decode == Decode::EditTree && tags,
            HeadKind::SentenceBoundary => matches!(self.decode, Decode::Argmax | Decode::Crf) && tags,
            HeadKind::SpanClassifier | HeadKind::OpenSpan => {
                matches!(self.decode, Decode::SpansNested | Decode::SpansFlat) && tags
            }
            HeadKind::CorefRanker => self.decode == Decode::Antecedent && tags,
            HeadKind::RelationClassifier => matches!(self.decode, Decode::Relations | Decode::Relation) && tags,
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
                 DECODE_SPANS_FLAT; a HEAD_OPEN_SPAN takes SCHEME_TAGS and DECODE_SPANS_NESTED or DECODE_SPANS_FLAT; \
                 a HEAD_COREF_RANKER takes SCHEME_TAGS and DECODE_ANTECEDENT; a HEAD_RELATION_CLASSIFIER takes SCHEME_TAGS and \
                 DECODE_RELATIONS or DECODE_RELATION",
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
    /// decode check: the boundary labels, the split labels, the span,
    /// coref and relation shapes. `at` gives the field path an error names.
    pub(super) fn validate_kind_rules(&self, at: &dyn Fn(&str) -> String) -> Result<()> {
        let coref = self.kind == HeadKind::CorefRanker;
        let spans = matches!(self.kind, HeadKind::SpanClassifier | HeadKind::OpenSpan) || coref;
        if spans != self.span.is_some() {
            return Err(invalid(format!(
                "manifest.json: {}: given exactly when the kind is HEAD_SPAN_CLASSIFIER, HEAD_OPEN_SPAN or HEAD_COREF_RANKER",
                at("span")
            )));
        }
        if self.seen.is_some() && self.kind != HeadKind::TokenSplit {
            return Err(invalid(format!("manifest.json: {}: a seen lookup is a HEAD_TOKEN_SPLIT's alone", at("seen"))));
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
        self.validate_seen_lookup(at)?;
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
                if let Some(s) = &self.seen {
                    self.check_seen(at, s)?;
                }
                Ok(())
            }
            HeadKind::SpanClassifier if !self.labels.iter().any(|l| l == "O") => {
                Err(invalid(format!("manifest.json: {}: a HEAD_SPAN_CLASSIFIER needs the label \"O\"", at("labels"))))
            }
            HeadKind::OpenSpan if self.labels.iter().any(|l| l == "O") => Err(invalid(format!(
                "manifest.json: {}: a HEAD_OPEN_SPAN lists only entity types, not \"O\"",
                at("labels")
            ))),
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

    /// A seen lookup's own bounds: a positive `min_count`, at most
    /// MAX_SEEN_TOKENS entries, each token listed once, each label one
    /// of the head's.
    fn check_seen(&self, at: &dyn Fn(&str) -> String, s: &SeenLookup) -> Result<()> {
        positive(&at("seen.min_count"), s.min_count)?;
        if s.tokens.len() > MAX_SEEN_TOKENS {
            return Err(invalid(format!(
                "manifest.json: {}: {} tokens, at most {MAX_SEEN_TOKENS}",
                at("seen.tokens"),
                s.tokens.len()
            )));
        }
        let mut listed = std::collections::HashSet::with_capacity(s.tokens.len());
        for (j, e) in s.tokens.iter().enumerate() {
            let field = at(&format!("seen.tokens[{j}].token"));
            required(&field, &e.token)?;
            fits(&field, &e.token, 256)?;
            if !listed.insert(&e.token) {
                return Err(invalid(format!("manifest.json: {field}: {:?} is listed twice", e.token)));
            }
            if !self.labels.contains(&e.label) {
                return Err(invalid(format!(
                    "manifest.json: {}: {:?} is not one of the head's labels",
                    at(&format!("seen.tokens[{j}].label")),
                    e.label
                )));
            }
        }
        Ok(())
    }

    pub(super) fn validate_word_inputs(&self, at: &dyn Fn(&str) -> String) -> Result<()> {
        if let Some(h) = &self.hash_embed {
            if !self.kind.scores_words() {
                return Err(invalid(format!(
                    "manifest.json: {}: only a head that scores words, as a HEAD_TOKEN_TAGGER does, takes it",
                    at("hash_embed")
                )));
            }
            positive(&at("hash_embed.dim"), h.dim)?;
            if h.dim > MAX_HASH_DIM || h.rows.iter().any(|&r| r == 0 || r > MAX_HASH_ROWS) {
                return Err(invalid(format!(
                    "manifest.json: {}: dim {} and rows {:?}; dim 1 to {MAX_HASH_DIM}, rows 1 to {MAX_HASH_ROWS}",
                    at("hash_embed"),
                    h.dim,
                    h.rows
                )));
            }
        }
        if let Some(a) = &self.adapter {
            positive(&at("adapter.bilstm.hidden"), a.bilstm.hidden)?;
            positive(&at("adapter.bilstm.layers"), a.bilstm.layers)?;
            if a.bilstm.hidden > MAX_ADAPTER_HIDDEN || a.bilstm.layers > MAX_ADAPTER_LAYERS {
                return Err(invalid(format!(
                    "manifest.json: {}: hidden {} and layers {}; at most {MAX_ADAPTER_HIDDEN} and {MAX_ADAPTER_LAYERS}",
                    at("adapter.bilstm"),
                    a.bilstm.hidden,
                    a.bilstm.layers
                )));
            }
        }
        if let Some(m) = &self.layer_mix {
            let field = at("layer_mix.layers");
            positive(&field, m.layers)?;
            if m.layers > MAX_MIX_LAYERS {
                return Err(invalid(format!("manifest.json: {field}: {}, at most {MAX_MIX_LAYERS}", m.layers)));
            }
            if let Some(layer) = m.layer {
                positive(&at("layer_mix.layer"), layer)?;
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
