//! Zero-shot relation extraction: relations between entity mentions for
//! relation labels the caller supplies at request time, as GLiREL frames
//! it. A [`RelationScorer`] scores every ordered mention pair against
//! every label in one pass over the text; this module owns what happens
//! around the model: the candidate pairs, the label semantics (a
//! threshold per label, directed or symmetric, the entity kinds a label
//! allows at each end) and the limits, refused by name when exceeded.
//!
//! The annotator ([`ZeroShotRelationAnnotator`]) writes the existing
//! relation layers, `opennlp:relations` and `Edges<Entities, Entities>`,
//! so its output reads like the dependency-path extractor's, and
//! `Relations` implements `Evidenced`, so an evidence step can record
//! each relation's head and tail spans, score and model.
//!
//! The model sits behind the trait. [`LexicalRelationScorer`] is a
//! model-free baseline: it compares the words between and around a pair
//! with the words of each label.

mod annotate;
mod lexical;

pub use annotate::ZeroShotRelationAnnotator;
pub use lexical::LexicalRelationScorer;

use std::fmt;

use turbo_analysis::{ModelRef, Span};

/// One relation label and how it applies.
#[derive(Clone, Debug, PartialEq)]
pub struct RelationLabel {
    /// The label's name, as the model reads it ("founded by").
    pub name: String,
    /// The least score a pair needs; `None` takes the configuration's.
    pub threshold: Option<f64>,
    /// A symmetric label holds both ways ("sibling of"): the two orders
    /// of a pair are scored, the better score kept, and the relation
    /// written once, its head the mention that comes first. Its domain and
    /// range must then be the same.
    pub symmetric: bool,
    /// The entity kinds the head may have; `None` allows any.
    pub domain: Option<Vec<String>>,
    /// The entity kinds the tail may have; `None` allows any.
    pub range: Option<Vec<String>>,
}

impl RelationLabel {
    /// A directed label with no constraints and the default threshold.
    pub fn new(name: impl Into<String>) -> RelationLabel {
        RelationLabel { name: name.into(), threshold: None, symmetric: false, domain: None, range: None }
    }
}

/// What the annotator scores and keeps.
#[derive(Clone, Debug, PartialEq)]
pub struct ZeroShotRelationConfig {
    pub labels: Vec<RelationLabel>,
    /// The least score a pair needs under a label without its own.
    pub threshold: f64,
    /// Pairs only within one sentence, when the document has sentences.
    pub within_sentence: bool,
    /// The most labels a request may hold.
    pub max_labels: usize,
    /// The most entity mentions a document may hold.
    pub max_mentions: usize,
    /// The most candidate pairs a document may send the scorer, which
    /// bounds its output at this many pairs times the labels.
    pub max_pairs: usize,
}

impl ZeroShotRelationConfig {
    /// The defaults: a threshold of 0.5, pairs within a sentence, at most
    /// 64 labels, 512 mentions and 65,536 candidate pairs.
    pub fn new(labels: Vec<RelationLabel>) -> ZeroShotRelationConfig {
        ZeroShotRelationConfig {
            labels,
            threshold: 0.5,
            within_sentence: true,
            max_labels: 64,
            max_mentions: 512,
            max_pairs: 1 << 16,
        }
    }
}

/// One entity mention as the scorer sees it: its span in the view the
/// text is, and its kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Mention<'a> {
    pub span: Span,
    pub kind: &'a str,
}

/// A relation model: scores ordered mention pairs against labels.
pub trait RelationScorer: Send + Sync {
    /// The model, recorded as the layer's provenance; `None` for a
    /// scorer that runs none.
    fn model_ref(&self) -> Option<ModelRef>;

    /// The most labels the model takes in one pass.
    fn max_labels(&self) -> usize;

    /// The most mentions the model takes in one pass.
    fn max_mentions(&self) -> usize;

    /// Scores pair p, `(head, tail)` indexes into `mentions`, against
    /// label l at `out[p * labels.len() + l]`, a probability in [0, 1].
    /// `out` arrives empty.
    fn score(
        &self,
        text: &str,
        mentions: &[Mention<'_>],
        pairs: &[(u32, u32)],
        labels: &[&str],
        out: &mut Vec<f32>,
    ) -> Result<(), RelationError>;
}

/// Why zero-shot relation extraction refused a request.
#[derive(Clone, Debug, PartialEq)]
pub enum RelationError {
    NoLabels,
    BlankLabel(usize),
    DuplicateLabel(String),
    /// A symmetric label whose domain and range differ: its two ends are
    /// interchangeable, so the head written may not fit the domain.
    AsymmetricKinds(String),
    /// A threshold outside [0, 1].
    BadThreshold {
        label: Option<String>,
        threshold: f64,
    },
    /// A named limit exceeded: `limit` is the field (`max_labels`,
    /// `max_mentions`, `max_pairs`), `max` its value, `got` what the
    /// request held (for pairs, the count when the limit was passed).
    OverLimit {
        limit: &'static str,
        max: usize,
        got: usize,
    },
    /// An input layer that does not hold together or is not in the view
    /// given.
    InvalidInput(String),
    /// The scorer returned the wrong shape or a score outside [0, 1].
    BadScorerOutput(String),
    /// The scorer itself failed.
    Scorer(String),
}

impl fmt::Display for RelationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use RelationError::*;
        match self {
            NoLabels => write!(f, "no relation labels"),
            BlankLabel(i) => write!(f, "relation label {i} is blank"),
            DuplicateLabel(l) => write!(f, "relation label {l:?} appears twice"),
            AsymmetricKinds(l) => write!(f, "symmetric relation label {l:?} has a domain unlike its range"),
            BadThreshold { label: Some(l), threshold } => {
                write!(f, "label {l:?}: threshold {threshold} outside [0, 1]")
            }
            BadThreshold { label: None, threshold } => write!(f, "threshold {threshold} outside [0, 1]"),
            OverLimit { limit, max, got } => write!(f, "{got} exceeds {limit} of {max}"),
            InvalidInput(d) => write!(f, "invalid input: {d}"),
            BadScorerOutput(d) => write!(f, "the scorer returned {d}"),
            Scorer(d) => write!(f, "the scorer failed: {d}"),
        }
    }
}

impl std::error::Error for RelationError {}

/// Checks the labels and thresholds, and the label count against both
/// the configuration's limit and the scorer's.
pub(crate) fn check(config: &ZeroShotRelationConfig, scorer: &dyn RelationScorer) -> Result<(), RelationError> {
    let labels = &config.labels;
    if labels.is_empty() {
        return Err(RelationError::NoLabels);
    }
    let max = config.max_labels.min(scorer.max_labels());
    let limit = if config.max_labels <= scorer.max_labels() { "max_labels" } else { "scorer max_labels" };
    if labels.len() > max {
        return Err(RelationError::OverLimit { limit, max, got: labels.len() });
    }
    if labels.len() > u16::MAX as usize {
        return Err(RelationError::OverLimit { limit: "tag set", max: u16::MAX as usize, got: labels.len() });
    }
    let unit = |t: f64| (0.0..=1.0).contains(&t);
    if !unit(config.threshold) {
        return Err(RelationError::BadThreshold { label: None, threshold: config.threshold });
    }
    for (i, l) in labels.iter().enumerate() {
        if l.name.trim().is_empty() {
            return Err(RelationError::BlankLabel(i));
        }
        if labels[..i].iter().any(|o| o.name == l.name) {
            return Err(RelationError::DuplicateLabel(l.name.clone()));
        }
        if l.symmetric && l.domain != l.range {
            return Err(RelationError::AsymmetricKinds(l.name.clone()));
        }
        if let Some(t) = l.threshold.filter(|&t| !unit(t)) {
            return Err(RelationError::BadThreshold { label: Some(l.name.clone()), threshold: t });
        }
    }
    Ok(())
}
