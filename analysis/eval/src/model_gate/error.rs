use std::fmt;

use super::{MODEL_GATE_RECORD_VERSION, Measure, RESAMPLES, SEED, ScoringKind, Task};

/// What the gate and its record refuse or fail on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelGateError {
    /// Fewer units than [`super::MIN_UNITS`].
    TooFewUnits {
        units: usize,
        min: usize,
    },
    /// More units than `java.util.Random.nextInt` can index.
    TooManyUnits {
        units: usize,
    },
    NoResamples,
    /// The two models were scored against different numbers of references
    /// on one unit, so they did not score the same reference.
    DifferentReferences {
        unit: usize,
        candidate: u64,
        reference: u64,
    },
    /// Accuracy counts whose predictions are not their tokens, or with
    /// more correct tokens than tokens.
    AccuracyCounts {
        unit: usize,
    },
    /// F-measure counts with more true positives than predictions or
    /// references.
    FMeasureCounts {
        unit: usize,
    },
    /// A record drawn with other resamples or another seed than the
    /// gate's [`RESAMPLES`] and [`SEED`].
    Parameters {
        resamples: u32,
        seed: i64,
    },
    /// A record field that is empty or not a digest.
    Field {
        field: &'static str,
        value: String,
    },
    /// A record of another schema version.
    Version {
        found: u32,
    },
    /// A record whose measure is not its task's.
    MeasureMismatch {
        task: Task,
        measure: Measure,
    },
    /// A record whose result is not what the gate gives on its units.
    ResultMismatch,
    /// A record whose scoring is not its task's ([`Task::scoring`]):
    /// spans for a sentence detector or tokenizer, typed spans for a name
    /// finder or chunker, tags for a tagger or lemmatizer, labels for a
    /// language detector.
    ScoringMismatch {
        task: Task,
        scoring: ScoringKind,
    },
    /// A tag scoring whose tokens, less those left out, are not the
    /// tokens a model's units count (`scored`).
    ScoredTokens {
        tokens: u64,
        left_out: u64,
        scored: u64,
    },
    /// A label scoring whose samples are not the record's units, or a unit
    /// that does not score one sample (`unit`, when one is the cause).
    ScoredSamples {
        samples: u64,
        units: usize,
        unit: Option<usize>,
    },
    /// A label scoring with held-out sentences skipped: every held-out
    /// sample is a unit.
    SkippedSamples {
        skipped: u64,
    },
    /// A label mapping with an empty label, mapping a label to itself, or
    /// listed twice.
    LabelMapping {
        held_out: String,
        reference: String,
    },
    /// A tag mapping with an empty tag, mapping a tag to itself, or
    /// mapping the same held-out tag twice.
    TagMapping {
        held_out: String,
        reference: String,
    },
    /// A typed span scoring whose held-out spans are not the references a
    /// model's units count (`scored`).
    ScoredSpans {
        spans: u64,
        scored: u64,
    },
    /// A typed span scoring with no type scored.
    NoSpanTypes,
    /// A typed span scoring that lists the type `span_type` more than
    /// once.
    DuplicateSpanType {
        span_type: String,
    },
    /// A typed span scoring that renames the held-out type `held_out` more
    /// than once.
    DuplicateRenaming {
        held_out: String,
    },
    /// A typed span scoring that renames the held-out type `held_out` to
    /// `scored`, a type it does not score.
    RenamedToUnscored {
        held_out: String,
        scored: String,
    },
    /// A typed span scoring that renames the held-out type `held_out`,
    /// which it also scores as written.
    RenamedScoredType {
        held_out: String,
    },
    /// A typed span scoring that gives the reference another tag for the
    /// held-out input tag `held_out` more than once.
    DuplicateReferenceInput {
        held_out: String,
    },
    /// A typed span scoring that gives the reference the held-out input
    /// tag `tag` for itself.
    UnchangedReferenceInput {
        tag: String,
    },
    /// A lemmatizer's record with tag mappings: its lemmas are compared as
    /// written.
    LemmatizerMappings {
        mappings: usize,
    },
    /// A record file that could not be read or written.
    Io {
        path: String,
        message: String,
    },
    /// A record file that is not a record.
    Json {
        path: String,
        message: String,
    },
}

impl fmt::Display for ModelGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelGateError::TooFewUnits { units, min } => {
                write!(f, "{units} held-out units, fewer than the {min} the gate needs")
            }
            ModelGateError::TooManyUnits { units } => {
                write!(f, "{units} units, more than java.util.Random can index")
            }
            ModelGateError::NoResamples => f.write_str("no resamples"),
            ModelGateError::DifferentReferences { unit, candidate, reference } => write!(
                f,
                "unit {unit}: the candidate was scored against {candidate} references and the reference model against {reference}"
            ),
            ModelGateError::AccuracyCounts { unit } => write!(
                f,
                "unit {unit}: accuracy counts with predictions that are not the tokens, or more correct than tokens"
            ),
            ModelGateError::FMeasureCounts { unit } => {
                write!(f, "unit {unit}: F-measure counts with more true positives than predictions or references")
            }
            ModelGateError::Parameters { resamples, seed } => write!(
                f,
                "a record of {resamples} resamples with seed {seed}; the gate draws {RESAMPLES} with seed {SEED}"
            ),
            ModelGateError::Field { field, value } => {
                write!(f, "{field} {value:?} is empty or not 64 lower case hex")
            }
            ModelGateError::Version { found } => {
                write!(f, "record version {found}, this build reads {MODEL_GATE_RECORD_VERSION}")
            }
            ModelGateError::MeasureMismatch { task, measure } => {
                write!(f, "a {task:?} record with the measure {measure:?}")
            }
            ModelGateError::ResultMismatch => {
                f.write_str("the recorded result is not what the gate gives on the recorded units")
            }
            ModelGateError::ScoringMismatch { task, scoring } => {
                write!(f, "a {task:?} record with a scoring of {scoring:?}")
            }
            ModelGateError::ScoredTokens { tokens, left_out, scored } => {
                write!(f, "{tokens} held-out tokens, {left_out} of them left out, but the units count {scored} tokens")
            }
            ModelGateError::ScoredSamples { samples, units, unit: None } => {
                write!(f, "{samples} held-out samples scored, but {units} units")
            }
            ModelGateError::ScoredSamples { unit: Some(unit), .. } => {
                write!(f, "unit {unit} does not score one held-out sample")
            }
            ModelGateError::SkippedSamples { skipped } => {
                write!(f, "a label scoring with {skipped} held-out sentences skipped")
            }
            ModelGateError::LabelMapping { held_out, reference } => write!(
                f,
                "the label mapping {held_out:?} to {reference:?} is empty, maps a label to itself, or is listed twice"
            ),
            ModelGateError::TagMapping { held_out, reference } => write!(
                f,
                "the tag mapping {held_out:?} to {reference:?} is empty, maps a tag to itself, or maps {held_out:?} twice"
            ),
            ModelGateError::ScoredSpans { spans, scored } => {
                write!(f, "{spans} held-out spans scored, but the units count {scored} references")
            }
            ModelGateError::NoSpanTypes => f.write_str("a typed span scoring with no type scored"),
            ModelGateError::DuplicateSpanType { span_type } => write!(f, "the span type {span_type:?} is scored twice"),
            ModelGateError::DuplicateRenaming { held_out } => {
                write!(f, "the held-out span type {held_out:?} is renamed twice")
            }
            ModelGateError::RenamedToUnscored { held_out, scored } => {
                write!(f, "the held-out span type {held_out:?} is renamed {scored:?}, a type not scored")
            }
            ModelGateError::RenamedScoredType { held_out } => {
                write!(f, "the held-out span type {held_out:?} is renamed, and also scored as written")
            }
            ModelGateError::DuplicateReferenceInput { held_out } => {
                write!(f, "the reference is given another tag for the held-out tag {held_out:?} twice")
            }
            ModelGateError::UnchangedReferenceInput { tag } => {
                write!(f, "the reference is given the held-out tag {tag:?} for itself")
            }
            ModelGateError::LemmatizerMappings { mappings } => {
                write!(f, "a lemmatizer's record with {mappings} tag mappings; its lemmas are compared as written")
            }
            ModelGateError::Io { path, message } | ModelGateError::Json { path, message } => {
                write!(f, "{path}: {message}")
            }
        }
    }
}

impl std::error::Error for ModelGateError {}

pub(super) type Result<T> = std::result::Result<T, ModelGateError>;
