use serde::{Deserialize, Serialize};

use super::error::Result;
use super::{Measure, ModelGateError, PairedUnit};

/// The task the two models perform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    SentenceDetector,
    Tokenizer,
    PartOfSpeechTagger,
    /// A lemmatizer given the held-out tags, scored on its lemmas as a
    /// tagger is on its tags ([`TagScoring`]).
    Lemmatizer,
    /// Names, scored by `TokenNameFinderEvaluator`.
    NameFinder,
    /// Phrase chunks, scored by `ChunkerEvaluator`.
    Chunker,
    /// One label per text, as `LanguageDetectorEvaluator` scores
    /// `predictLanguage`.
    LanguageDetector,
}

/// What a task's units are scored on, and so which [`Scoring`] its
/// record holds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScoringKind {
    /// Spans without a type: [`Scoring::Spans`].
    Spans,
    /// Spans with a type, of the types the gate scores:
    /// [`Scoring::TypedSpans`].
    TypedSpans,
    /// Tags of tokens: [`Scoring::Tags`].
    Tags,
    /// One label per sample: [`Scoring::Labels`].
    Labels,
}

impl Task {
    /// The measure OpenNLP's evaluator for the task reports.
    pub fn measure(self) -> Measure {
        match self {
            Task::SentenceDetector | Task::Tokenizer | Task::NameFinder | Task::Chunker => Measure::FMeasure,
            Task::PartOfSpeechTagger | Task::Lemmatizer | Task::LanguageDetector => Measure::Accuracy,
        }
    }

    /// What the task's units are scored on.
    pub fn scoring(self) -> ScoringKind {
        match self {
            Task::SentenceDetector | Task::Tokenizer => ScoringKind::Spans,
            Task::NameFinder | Task::Chunker => ScoringKind::TypedSpans,
            Task::PartOfSpeechTagger | Task::Lemmatizer => ScoringKind::Tags,
            Task::LanguageDetector => ScoringKind::Labels,
        }
    }
}

/// A model scored by the gate: its zoo id or file name, and the SHA-256
/// of its file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelId {
    pub id: String,
    pub sha256: String,
}

/// The held-out samples both models were scored on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeldOutSet {
    /// The corpus, its release and the split, as the recipe names them.
    pub name: String,
    /// The SHA-256 of the corpus files the samples were read from, as the
    /// recipe computes it.
    pub corpus_sha256: String,
    /// The SHA-256 of the held-out samples as scored, as the recipe
    /// computes it, so two records of the same set can be seen to score
    /// the same samples.
    pub samples_sha256: String,
    /// The sentences of the held-out documents the recipe's reader
    /// skipped, which no unit scores.
    pub skipped_sentences: u64,
    /// How each unit's counts were made from the two models' output.
    pub scoring: Scoring,
}

/// How a gate's units were scored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Scoring {
    /// Spans, by the task's OpenNLP evaluator, against every reference
    /// span of the unit: an F-measure.
    Spans,
    /// Spans with a type, by the task's OpenNLP evaluator, over the spans
    /// of the types scored: an F-measure.
    TypedSpans(TypedSpanScoring),
    /// Tags of tokens, or a lemmatizer's lemmas: an accuracy.
    Tags(TagScoring),
    /// One label per sample, a unit each: an accuracy.
    Labels(LabelScoring),
}

impl Scoring {
    /// The kind of this scoring.
    pub fn kind(&self) -> ScoringKind {
        match self {
            Scoring::Spans => ScoringKind::Spans,
            Scoring::TypedSpans(_) => ScoringKind::TypedSpans,
            Scoring::Tags(_) => ScoringKind::Tags,
            Scoring::Labels(_) => ScoringKind::Labels,
        }
    }
}

/// How the typed spans of the held-out samples (names, chunks) were
/// scored. Only spans of the `types` count, on both sides: a held-out
/// span of another type is left out of the references, as OpenNLP's
/// `NameSampleTypeFilter` leaves it out of a name sample, and a predicted
/// span of another type is left out of the predictions, so neither model
/// is scored on a type the other does not give. A held-out type that the
/// models name otherwise is renamed first (`renamed`). The evaluator then
/// counts each unit as OpenNLP's `FMeasure.updateScores` does: a
/// prediction is right with the reference's tokens and type. A model
/// that reads tags with its tokens (a chunker reads part-of-speech tags)
/// is given the held-out tags, the reference through `reference_inputs`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedSpanScoring {
    /// The span types scored, as the models name them.
    pub types: Vec<String>,
    /// Held-out span types renamed to the type scored before scoring.
    pub renamed: Vec<TypeMapping>,
    /// The held-out input tags the reference is given another tag for, as
    /// the tag it is given; the trained model reads every held-out tag as
    /// written. Empty when both models read the held-out input as written,
    /// as a name finder does.
    pub reference_inputs: Vec<TagMapping>,
    /// The held-out spans of the types scored, after renaming: the
    /// references every unit counts, summed.
    pub spans: u64,
    /// The held-out spans of other types, left out of both models' scores.
    pub left_out: u64,
}

/// A held-out span type and the type it is scored as.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypeMapping {
    pub held_out: String,
    pub scored: String,
}

/// How a tagger's held-out tokens were scored: the tokens whose held-out
/// tag the reference can give, as written or through `reference_mappings`;
/// the candidate's tag compared with the held-out tag as written, and the
/// reference's also counted right where `reference_mappings` maps the
/// held-out tag to it. A lemmatizer's lemmas are scored the same way, with
/// no mapping: both models' lemmas are compared with the held-out lemma as
/// written, over the tokens the reference can get right or the candidate
/// cannot be taught (a token is left out when its gold lemma class, the
/// edit script from the token to its held-out lemma, gives that lemma and
/// none of the reference's classes does).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TagScoring {
    /// The tokens of the held-out samples.
    pub tokens: u64,
    /// The tokens left out of both models' scores: the reference cannot
    /// give their tag, or for a lemmatizer, cannot give the lemma their
    /// gold lemma class gives.
    pub left_out: u64,
    /// The reference's tags counted right for another held-out tag.
    pub reference_mappings: Vec<TagMapping>,
}

/// A held-out tag and the reference's tag for it: the tag counted right
/// for it ([`TagScoring::reference_mappings`]), or the tag the reference
/// is given in its place ([`TypedSpanScoring::reference_inputs`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TagMapping {
    pub held_out: String,
    pub reference: String,
}

/// How a language detector's held-out samples were scored: one unit per
/// sample, with the counts `Counts::accuracy(1, 1)` when the model's best
/// label is right and `Counts::accuracy(0, 1)` when not. The candidate's
/// label is compared with the held-out label as written, and the
/// reference's also counted right where `reference_mappings` maps the
/// held-out label to it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelScoring {
    /// The held-out samples, each a unit of the gate.
    pub samples: u64,
    /// The reference's labels counted right for another held-out label.
    pub reference_mappings: Vec<LabelMapping>,
}

/// A held-out label and a reference label counted right for it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelMapping {
    pub held_out: String,
    pub reference: String,
}

impl HeldOutSet {
    pub(super) fn check(&self, task: Task, units: &[PairedUnit]) -> Result<()> {
        if self.scoring.kind() != task.scoring() {
            return Err(ModelGateError::ScoringMismatch { task, scoring: self.scoring.kind() });
        }
        let sides: [fn(&PairedUnit) -> u64; 2] = [|u| u.candidate.target, |u| u.reference.target];
        match &self.scoring {
            Scoring::Spans => {}
            Scoring::TypedSpans(t) => {
                if t.types.is_empty() {
                    return Err(ModelGateError::NoSpanTypes);
                }
                if t.types.iter().any(|s| s.is_empty()) {
                    return Err(ModelGateError::Field { field: "set.scoring.typed_spans.types", value: String::new() });
                }
                if let Some(span_type) = first_repeat(t.types.iter()) {
                    return Err(ModelGateError::DuplicateSpanType { span_type: span_type.clone() });
                }
                if t.renamed.iter().any(|m| m.held_out.is_empty() || m.scored.is_empty()) {
                    return Err(ModelGateError::Field {
                        field: "set.scoring.typed_spans.renamed",
                        value: String::new(),
                    });
                }
                if let Some(held_out) = first_repeat(t.renamed.iter().map(|m| &m.held_out)) {
                    return Err(ModelGateError::DuplicateRenaming { held_out: held_out.clone() });
                }
                for m in &t.renamed {
                    if !t.types.contains(&m.scored) {
                        let (held_out, scored) = (m.held_out.clone(), m.scored.clone());
                        return Err(ModelGateError::RenamedToUnscored { held_out, scored });
                    }
                    if t.types.contains(&m.held_out) {
                        return Err(ModelGateError::RenamedScoredType { held_out: m.held_out.clone() });
                    }
                }
                if t.reference_inputs.iter().any(|m| m.held_out.is_empty() || m.reference.is_empty()) {
                    return Err(ModelGateError::Field {
                        field: "set.scoring.typed_spans.reference_inputs",
                        value: String::new(),
                    });
                }
                if let Some(held_out) = first_repeat(t.reference_inputs.iter().map(|m| &m.held_out)) {
                    return Err(ModelGateError::DuplicateReferenceInput { held_out: held_out.clone() });
                }
                if let Some(m) = t.reference_inputs.iter().find(|m| m.held_out == m.reference) {
                    return Err(ModelGateError::UnchangedReferenceInput { tag: m.held_out.clone() });
                }
                for side in sides {
                    let scored: u64 = units.iter().map(side).fold(0u64, u64::saturating_add);
                    if scored != t.spans {
                        return Err(ModelGateError::ScoredSpans { spans: t.spans, scored });
                    }
                }
            }
            Scoring::Tags(t) => {
                for side in sides {
                    let scored: u64 = units.iter().map(side).fold(0u64, u64::saturating_add);
                    if t.left_out > t.tokens || scored != t.tokens - t.left_out {
                        return Err(ModelGateError::ScoredTokens { tokens: t.tokens, left_out: t.left_out, scored });
                    }
                }
                if task == Task::Lemmatizer && !t.reference_mappings.is_empty() {
                    return Err(ModelGateError::LemmatizerMappings { mappings: t.reference_mappings.len() });
                }
                let mut seen = std::collections::HashSet::new();
                for m in &t.reference_mappings {
                    let bad = m.held_out.is_empty() || m.reference.is_empty() || m.held_out == m.reference;
                    if bad || !seen.insert(&m.held_out) {
                        return Err(ModelGateError::TagMapping {
                            held_out: m.held_out.clone(),
                            reference: m.reference.clone(),
                        });
                    }
                }
            }
            Scoring::Labels(l) => {
                if self.skipped_sentences != 0 {
                    return Err(ModelGateError::SkippedSamples { skipped: self.skipped_sentences });
                }
                let mut seen = std::collections::HashSet::new();
                for m in &l.reference_mappings {
                    let bad = m.held_out.is_empty() || m.reference.is_empty() || m.held_out == m.reference;
                    if bad || !seen.insert((&m.held_out, &m.reference)) {
                        return Err(ModelGateError::LabelMapping {
                            held_out: m.held_out.clone(),
                            reference: m.reference.clone(),
                        });
                    }
                }
                if l.samples != units.len() as u64 {
                    return Err(ModelGateError::ScoredSamples { samples: l.samples, units: units.len(), unit: None });
                }
                if let Some(unit) = units.iter().position(|u| u.candidate.target != 1 || u.reference.target != 1) {
                    return Err(ModelGateError::ScoredSamples {
                        samples: l.samples,
                        units: units.len(),
                        unit: Some(unit),
                    });
                }
            }
        }
        Ok(())
    }
}

/// The first item of `items` equal to an item before it.
fn first_repeat<'a>(items: impl Iterator<Item = &'a String>) -> Option<&'a String> {
    let mut seen = std::collections::BTreeSet::new();
    items.into_iter().find(|item| !seen.insert(*item))
}
