//! The gate a trained model passes against a reference model: the
//! OpenNLP model for the same task, scored on the same held-out samples.
//!
//! Each held-out unit (a sentence, a sample of sentences, or a text whose
//! language is known) is scored by both models as counts: true positives,
//! predictions and references for an F-measure, or correct and total
//! tokens or texts for an accuracy. The
//! score of a set of units is that of their summed counts, as OpenNLP's
//! `FMeasure` and `Mean` add them up. A paired bootstrap draws the units
//! with replacement, [`RESAMPLES`] times with `java.util.Random` seeded
//! with [`SEED`] (the draws of [`crate::significance::bootstrap`]), and
//! scores both models on each draw, so each resample gives both scores
//! and their difference.
//!
//! The gate is a non-inferiority test. The candidate passes when the low
//! end of the 95 percent interval of the difference, candidate minus
//! reference, is at least -[`MARGIN_POINTS`]: the data rule out, at that
//! confidence, a loss of more than half a point. This is the margin the
//! chunker gate allows a set ([`crate::gate::FLOOR_POINTS`]). A candidate
//! whose interval is wide because the held-out set is small fails, rather
//! than passing for lack of evidence, and a set of fewer than
//! [`MIN_UNITS`] units is refused. Each model's own interval is reported
//! beside the difference.
//!
//! Every figure is in points: 0.01 of the score, so an F-measure of
//! 0.953 is 95.3. Where precision and recall are both 0 the F-measure is
//! 0 here; OpenNLP's `FMeasure` returns -1 there.
//!
//! [`ModelGateRecord`] is the record kept with a model card: the two
//! models, the held-out set, the per unit counts and the result, which
//! [`ModelGateRecord::check`] recomputes from the counts.

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_jvm::random::JavaRandom;

use crate::gate::FLOOR_POINTS;
use crate::record::is_hex;
use crate::significance::{Bootstrap, percentile};
pub use crate::significance::{RESAMPLES, SEED};

/// The schema version of [`ModelGateRecord`]: 3, with the language
/// detector's task and scoring ([`Task::LanguageDetector`],
/// [`Scoring::Labels`]). Version 2 added the held-out sentences skipped
/// and how the units were scored ([`HeldOutSet`]). [`ModelGateRecord::read`]
/// reads the version before the rest, so a record of another version is
/// refused for its version, not for a task or scoring this build does not
/// know.
pub const MODEL_GATE_RECORD_VERSION: u32 = 3;

/// The loss, in points, the candidate may be shown not to exceed: the low
/// end of the difference's interval must be at least the negative of it.
/// The chunker gate's floor, so both gates allow the same loss.
pub const MARGIN_POINTS: f64 = -FLOOR_POINTS;

/// The fewest held-out units the gate scores. Below it the bootstrap
/// interval says little about the set's spread.
pub const MIN_UNITS: usize = 30;

/// What the gate and its record refuse or fail on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelGateError {
    /// Fewer units than [`MIN_UNITS`].
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
    /// A record whose scoring does not fit its measure: spans for an
    /// F-measure, tags for an accuracy.
    ScoringMismatch {
        measure: Measure,
    },
    /// A tag scoring whose tokens, less those left out, are not the
    /// tokens a model's units count (`scored`).
    ScoredTokens {
        tokens: u64,
        left_out: u64,
        scored: u64,
    },
    /// A record whose scoring is not its task's: tags for a tagger,
    /// labels for a language detector.
    TaskScoringMismatch {
        task: Task,
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
            ModelGateError::ScoringMismatch { measure } => {
                write!(f, "a record of the measure {measure:?} with a scoring of another kind")
            }
            ModelGateError::ScoredTokens { tokens, left_out, scored } => {
                write!(f, "{tokens} held-out tokens, {left_out} of them left out, but the units count {scored} tokens")
            }
            ModelGateError::TaskScoringMismatch { task } => {
                write!(f, "a {task:?} record with a scoring of another task")
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
            ModelGateError::Io { path, message } | ModelGateError::Json { path, message } => {
                write!(f, "{path}: {message}")
            }
        }
    }
}

impl std::error::Error for ModelGateError {}

type Result<T> = std::result::Result<T, ModelGateError>;

/// What a unit's counts are, and so how they make a score.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Measure {
    /// `true_positives` are predictions that match a reference,
    /// `selected` the predictions and `target` the references; the score
    /// is the harmonic mean of precision and recall.
    FMeasure,
    /// `true_positives` are the tokens (or texts) given the reference's
    /// value and `target` the tokens; `selected` equals `target`. The
    /// score is their ratio.
    Accuracy,
}

/// The task the two models perform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    SentenceDetector,
    Tokenizer,
    PartOfSpeechTagger,
    /// One label per text, as `LanguageDetectorEvaluator` scores
    /// `predictLanguage`.
    LanguageDetector,
}

impl Task {
    /// The measure OpenNLP's evaluator for the task reports.
    pub fn measure(self) -> Measure {
        match self {
            Task::SentenceDetector | Task::Tokenizer => Measure::FMeasure,
            Task::PartOfSpeechTagger | Task::LanguageDetector => Measure::Accuracy,
        }
    }
}

/// One model's counts on one unit.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub true_positives: u64,
    pub selected: u64,
    pub target: u64,
}

impl Counts {
    /// Counts for an accuracy: `correct` of `total` tokens.
    pub fn accuracy(correct: u64, total: u64) -> Counts {
        Counts { true_positives: correct, selected: total, target: total }
    }

    fn add(&mut self, other: &Counts) {
        self.true_positives += other.true_positives;
        self.selected += other.selected;
        self.target += other.target;
    }

    /// The score of these counts under `measure`, in points.
    pub fn score(&self, measure: Measure) -> f64 {
        let ratio = |n: u64, d: u64| if d > 0 { n as f64 / d as f64 } else { 0.0 };
        let score = match measure {
            Measure::Accuracy => ratio(self.true_positives, self.target),
            Measure::FMeasure => {
                let (p, r) = (ratio(self.true_positives, self.selected), ratio(self.true_positives, self.target));
                if p + r > 0.0 { 2.0 * (p * r) / (p + r) } else { 0.0 }
            }
        };
        100.0 * score
    }
}

/// Both models' counts on one held-out unit.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedUnit {
    pub candidate: Counts,
    pub reference: Counts,
}

/// A model's score on the whole set with the 95 percent interval of its
/// resampled scores, in points.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scored {
    pub score: f64,
    pub low: f64,
    pub high: f64,
}

/// The two models compared on one set.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelGate {
    pub candidate: Scored,
    pub reference: Scored,
    /// Candidate minus reference on the whole set, with the 95 percent
    /// interval of the resampled differences. Its `mean` is the
    /// difference of the two pooled scores (each from the set's summed
    /// counts), not a mean of per unit differences as in the chunker
    /// gate; the interval is of that pooled difference over the draws.
    pub difference: Bootstrap,
    /// The difference's interval's low end is at least -[`MARGIN_POINTS`].
    pub pass: bool,
}

fn total<'a>(counts: impl Iterator<Item = &'a Counts>) -> Counts {
    let mut sum = Counts::default();
    counts.for_each(|c| sum.add(c));
    sum
}

/// Scores the paired units and decides the gate, drawing `resamples`
/// resamples from `java.util.Random` seeded with `seed`: each resample is
/// `units.len()` indices from `nextInt(units.len())`, in order.
///
/// # Errors
///
/// Fewer than [`MIN_UNITS`] units, more than `nextInt` can index, no
/// resamples, a unit whose two sides count different references (they
/// did not score the same reference), or accuracy counts whose
/// predictions are not their references.
pub fn gate_models(measure: Measure, units: &[PairedUnit], resamples: u32, seed: i64) -> Result<ModelGate> {
    if units.len() < MIN_UNITS {
        return Err(ModelGateError::TooFewUnits { units: units.len(), min: MIN_UNITS });
    }
    let n = i32::try_from(units.len()).map_err(|_| ModelGateError::TooManyUnits { units: units.len() })?;
    if resamples == 0 {
        return Err(ModelGateError::NoResamples);
    }
    for (unit, u) in units.iter().enumerate() {
        if u.candidate.target != u.reference.target {
            return Err(ModelGateError::DifferentReferences {
                unit,
                candidate: u.candidate.target,
                reference: u.reference.target,
            });
        }
        let accuracy_ok = |c: &Counts| c.selected == c.target && c.true_positives <= c.target;
        if measure == Measure::Accuracy && !(accuracy_ok(&u.candidate) && accuracy_ok(&u.reference)) {
            return Err(ModelGateError::AccuracyCounts { unit });
        }
    }
    let candidate = total(units.iter().map(|u| &u.candidate)).score(measure);
    let reference = total(units.iter().map(|u| &u.reference)).score(measure);
    let mut random = JavaRandom::new(seed);
    let len = resamples as usize;
    let (mut cand, mut refr, mut diff) = (Vec::with_capacity(len), Vec::with_capacity(len), Vec::with_capacity(len));
    for _ in 0..resamples {
        let (mut c, mut r) = (Counts::default(), Counts::default());
        for _ in 0..n {
            let u = &units[random.next_int_bounded(n) as usize];
            c.add(&u.candidate);
            r.add(&u.reference);
        }
        let (c, r) = (c.score(measure), r.score(measure));
        cand.push(c);
        refr.push(r);
        diff.push(c - r);
    }
    let interval = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (percentile(&v, 0.025), percentile(&v, 0.975))
    };
    let (c_low, c_high) = interval(cand);
    let (r_low, r_high) = interval(refr);
    let (d_low, d_high) = interval(diff);
    let difference = Bootstrap { mean: candidate - reference, low: d_low, high: d_high };
    Ok(ModelGate {
        candidate: Scored { score: candidate, low: c_low, high: c_high },
        reference: Scored { score: reference, low: r_low, high: r_high },
        difference,
        pass: difference.low >= -MARGIN_POINTS,
    })
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
    /// Tags of tokens: an accuracy.
    Tags(TagScoring),
    /// One label per sample, a unit each: an accuracy.
    Labels(LabelScoring),
}

/// How a tagger's held-out tokens were scored: the tokens whose held-out
/// tag the reference can give, as written or through `reference_mappings`;
/// the candidate's tag compared with the held-out tag as written, and the
/// reference's also counted right where `reference_mappings` maps the
/// held-out tag to it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TagScoring {
    /// The tokens of the held-out samples.
    pub tokens: u64,
    /// The tokens left out of both models' scores: the reference cannot
    /// give their tag.
    pub left_out: u64,
    /// The reference's tags counted right for another held-out tag.
    pub reference_mappings: Vec<TagMapping>,
}

/// A held-out tag and the reference's tag counted right for it.
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

/// A record's version, read before the rest of it; its other fields are
/// ignored.
#[derive(Deserialize)]
struct RecordVersion {
    record_version: u32,
}

/// The record of one gate run, kept beside the model card.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelGateRecord {
    pub record_version: u32,
    pub task: Task,
    pub measure: Measure,
    pub set: HeldOutSet,
    pub candidate: ModelId,
    pub reference: ModelId,
    pub resamples: u32,
    pub seed: i64,
    pub result: ModelGate,
    /// Both models' counts on each held-out unit, in the order of the set.
    pub units: Vec<PairedUnit>,
}

impl ModelGateRecord {
    /// Runs the gate on `units` with [`RESAMPLES`] and [`SEED`] and
    /// records it.
    pub fn new(
        task: Task,
        set: HeldOutSet,
        candidate: ModelId,
        reference: ModelId,
        units: Vec<PairedUnit>,
    ) -> Result<ModelGateRecord> {
        let measure = task.measure();
        let result = gate_models(measure, &units, RESAMPLES, SEED)?;
        Ok(ModelGateRecord {
            record_version: MODEL_GATE_RECORD_VERSION,
            task,
            measure,
            set,
            candidate,
            reference,
            resamples: RESAMPLES,
            seed: SEED,
            result,
            units,
        })
    }

    /// Checks the version; that the names are not empty and the digests
    /// are 64 lower case hex; that the measure is the task's; that the
    /// record was drawn with [`RESAMPLES`] and [`SEED`]; that the scoring
    /// fits the measure and the task; for tags, that each model's units
    /// count the held-out tokens less those left out; for labels, that
    /// no sentence was skipped, that there is one unit per held-out sample
    /// and each scores one sample, and that each mapping has two labels
    /// that differ and is listed once; and that the result is what the
    /// gate gives on the recorded units, to the bit.
    pub fn check(&self) -> Result<()> {
        if self.record_version != MODEL_GATE_RECORD_VERSION {
            return Err(ModelGateError::Version { found: self.record_version });
        }
        let names =
            [("set.name", &self.set.name), ("candidate.id", &self.candidate.id), ("reference.id", &self.reference.id)];
        if let Some((field, value)) = names.into_iter().find(|(_, v)| v.is_empty()) {
            return Err(ModelGateError::Field { field, value: value.clone() });
        }
        let digests = [
            ("set.corpus_sha256", &self.set.corpus_sha256),
            ("set.samples_sha256", &self.set.samples_sha256),
            ("candidate.sha256", &self.candidate.sha256),
            ("reference.sha256", &self.reference.sha256),
        ];
        if let Some((field, value)) = digests.into_iter().find(|(_, v)| !is_hex(v, 64)) {
            return Err(ModelGateError::Field { field, value: value.clone() });
        }
        if self.resamples != RESAMPLES || self.seed != SEED {
            return Err(ModelGateError::Parameters { resamples: self.resamples, seed: self.seed });
        }
        if self.measure != self.task.measure() {
            return Err(ModelGateError::MeasureMismatch { task: self.task, measure: self.measure });
        }
        match (&self.set.scoring, self.measure) {
            (Scoring::Spans, Measure::FMeasure) => {}
            (Scoring::Tags(t), Measure::Accuracy) => {
                let sides: [fn(&PairedUnit) -> u64; 2] = [|u| u.candidate.target, |u| u.reference.target];
                for side in sides {
                    let scored: u64 = self.units.iter().map(side).sum();
                    if t.left_out > t.tokens || scored != t.tokens - t.left_out {
                        return Err(ModelGateError::ScoredTokens { tokens: t.tokens, left_out: t.left_out, scored });
                    }
                }
            }
            (Scoring::Labels(l), Measure::Accuracy) => {
                if self.set.skipped_sentences != 0 {
                    return Err(ModelGateError::SkippedSamples { skipped: self.set.skipped_sentences });
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
                if l.samples != self.units.len() as u64 {
                    return Err(ModelGateError::ScoredSamples {
                        samples: l.samples,
                        units: self.units.len(),
                        unit: None,
                    });
                }
                if let Some(unit) = self.units.iter().position(|u| u.candidate.target != 1 || u.reference.target != 1) {
                    return Err(ModelGateError::ScoredSamples {
                        samples: l.samples,
                        units: self.units.len(),
                        unit: Some(unit),
                    });
                }
            }
            (_, measure) => return Err(ModelGateError::ScoringMismatch { measure }),
        }
        let task_fits = match &self.set.scoring {
            Scoring::Spans => true,
            Scoring::Tags(_) => self.task == Task::PartOfSpeechTagger,
            Scoring::Labels(_) => self.task == Task::LanguageDetector,
        };
        if !task_fits {
            return Err(ModelGateError::TaskScoringMismatch { task: self.task });
        }
        let again = gate_models(self.measure, &self.units, self.resamples, self.seed)?;
        if again != self.result {
            return Err(ModelGateError::ResultMismatch);
        }
        Ok(())
    }

    /// The record as written: pretty JSON with a final newline.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a gate record serializes");
        s.push('\n');
        s
    }

    /// Reads and checks a record. The version is read first, alone, so a
    /// record of another version is refused with
    /// [`ModelGateError::Version`] whatever its other fields hold.
    pub fn read(path: &Path) -> Result<ModelGateRecord> {
        let json =
            |e: serde_json::Error| ModelGateError::Json { path: path.display().to_string(), message: e.to_string() };
        let bytes = fs::read(path)
            .map_err(|e| ModelGateError::Io { path: path.display().to_string(), message: e.to_string() })?;
        let version: RecordVersion = serde_json::from_slice(&bytes).map_err(json)?;
        if version.record_version != MODEL_GATE_RECORD_VERSION {
            return Err(ModelGateError::Version { found: version.record_version });
        }
        let record: ModelGateRecord = serde_json::from_slice(&bytes).map_err(json)?;
        record.check()?;
        Ok(record)
    }

    /// Writes the record to `path`, which must not exist yet.
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| ModelGateError::Io { path: path.display().to_string(), message: e.to_string() })?;
        f.write_all(self.to_json().as_bytes())
            .map_err(|e| ModelGateError::Io { path: path.display().to_string(), message: e.to_string() })
    }
}

impl fmt::Display for ModelGateRecord {
    /// The two scores with their intervals, the difference with its
    /// interval, and the verdict.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = &self.result;
        let measure = match self.measure {
            Measure::FMeasure => "F-measure",
            Measure::Accuracy => "accuracy",
        };
        writeln!(f, "{:?} on {}, {} units, {measure} in points:", self.task, self.set.name, self.units.len())?;
        for (who, id, s) in [("candidate", &self.candidate, r.candidate), ("reference", &self.reference, r.reference)] {
            writeln!(f, "  {who} {}: {:.2} (95% interval {:.2} to {:.2})", id.id, s.score, s.low, s.high)?;
        }
        writeln!(
            f,
            "  difference: {:+.2} (95% interval {:+.2} to {:+.2})",
            r.difference.mean, r.difference.low, r.difference.high
        )?;
        if r.pass {
            write!(
                f,
                "PASS: the difference's interval starts at {:+.2}, not below -{MARGIN_POINTS}: a loss of more than {MARGIN_POINTS} points is ruled out",
                r.difference.low
            )
        } else {
            write!(
                f,
                "FAIL: the difference's interval starts at {:+.2}, below -{MARGIN_POINTS}: a loss of more than {MARGIN_POINTS} points is not ruled out",
                r.difference.low
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(tp: u64, selected: u64, target: u64) -> Counts {
        Counts { true_positives: tp, selected, target }
    }

    #[test]
    fn the_margin_is_the_chunker_gate_s_floor() {
        assert_eq!(MARGIN_POINTS, 0.5);
        assert_eq!(-MARGIN_POINTS, FLOOR_POINTS);
    }

    #[test]
    fn scores_are_those_of_the_summed_counts() {
        // 8 of 10 predictions right, of 16 references: p 0.8, r 0.5,
        // F = 2 * 0.4 / 1.3.
        let c = f(8, 10, 16);
        assert!((c.score(Measure::FMeasure) - 100.0 * 0.8 / 1.3).abs() < 1e-12);
        assert_eq!(Counts::accuracy(3, 4).score(Measure::Accuracy), 75.0);
        assert_eq!(f(0, 3, 3).score(Measure::FMeasure), 0.0);
        assert_eq!(f(0, 0, 0).score(Measure::Accuracy), 0.0);
        // The set's score is that of the sum, not the mean of the units':
        // 30 units of one reference each, the candidate right on all; one
        // unit of 70 references more, the candidate right on none. Pooled
        // the candidate has 30 of 100; the mean of per unit scores would
        // be 30/31.
        let mut units: Vec<PairedUnit> =
            (0..30).map(|_| PairedUnit { candidate: f(1, 1, 1), reference: f(1, 1, 1) }).collect();
        units.push(PairedUnit { candidate: f(0, 70, 70), reference: f(70, 70, 70) });
        let g = gate_models(Measure::FMeasure, &units, 100, SEED).unwrap();
        assert!((g.candidate.score - 30.0).abs() < 1e-12, "{g:?}");
        assert_eq!(g.reference.score, 100.0);
        // The pooled difference, not a mean of per unit differences.
        assert!((g.difference.mean + 70.0).abs() < 1e-12, "{g:?}");
        assert!(!g.pass);
    }

    /// `n` sentences of 10 tokens; the reference gets 9 right on each, the
    /// candidate `right(i)` on sentence i.
    fn tagging(n: u64, right: impl Fn(u64) -> u64) -> Vec<PairedUnit> {
        (0..n)
            .map(|i| PairedUnit { candidate: Counts::accuracy(right(i), 10), reference: Counts::accuracy(9, 10) })
            .collect()
    }

    #[test]
    fn a_tie_passes_with_a_point_difference() {
        let units: Vec<PairedUnit> = (0..30)
            .map(|i| {
                let c = f(i % 7, 7, 8);
                PairedUnit { candidate: c, reference: c }
            })
            .collect();
        let g = gate_models(Measure::FMeasure, &units, RESAMPLES, SEED).unwrap();
        assert_eq!(g.difference, Bootstrap { mean: 0.0, low: 0.0, high: 0.0 });
        assert!(g.pass);
        assert_eq!(g.candidate, g.reference);
        assert!(g.candidate.low < g.candidate.score && g.candidate.score < g.candidate.high, "{g:?}");
    }

    #[test]
    fn a_set_smaller_than_min_units_is_refused() {
        let units = tagging(MIN_UNITS as u64 - 1, |_| 10);
        assert_eq!(
            gate_models(Measure::Accuracy, &units, RESAMPLES, SEED),
            Err(ModelGateError::TooFewUnits { units: 29, min: MIN_UNITS })
        );
        assert!(gate_models(Measure::Accuracy, &[], RESAMPLES, SEED).is_err());
        assert!(gate_models(Measure::Accuracy, &tagging(MIN_UNITS as u64, |_| 10), RESAMPLES, SEED).is_ok());
    }

    #[test]
    fn a_one_point_loss_on_thirty_units_fails() {
        // 30 units of 100 tokens, the candidate one token behind on each:
        // -1 point on every draw.
        let units: Vec<PairedUnit> = (0..30)
            .map(|_| PairedUnit { candidate: Counts::accuracy(89, 100), reference: Counts::accuracy(90, 100) })
            .collect();
        let g = gate_models(Measure::Accuracy, &units, RESAMPLES, SEED).unwrap();
        assert!((g.difference.mean + 1.0).abs() < 1e-9, "{g:?}");
        assert!((g.difference.low + 1.0).abs() < 1e-9 && (g.difference.high + 1.0).abs() < 1e-9, "{g:?}");
        assert!(!g.pass);
        assert!(record_text(units).contains("FAIL: "));
        // 30 sentences of 10 tokens, the candidate one token behind on
        // every tenth: also -1 point, now with a spread interval.
        let spread = tagging(30, |i| if i % 10 == 0 { 8 } else { 9 });
        let g = gate_models(Measure::Accuracy, &spread, RESAMPLES, SEED).unwrap();
        assert!((g.difference.mean + 1.0).abs() < 1e-9, "{g:?}");
        assert!(g.difference.low < g.difference.high);
        assert!(!g.pass);
    }

    /// A held-out set scored `scoring`, with nothing skipped.
    fn set(name: &str, scoring: Scoring) -> HeldOutSet {
        HeldOutSet {
            name: name.into(),
            corpus_sha256: "a".repeat(64),
            samples_sha256: "b".repeat(64),
            skipped_sentences: 0,
            scoring,
        }
    }

    /// The tag scoring of `units`, `left_out` tokens left out.
    fn tags_of(units: &[PairedUnit], left_out: u64) -> Scoring {
        let scored: u64 = units.iter().map(|u| u.candidate.target).sum();
        let mapping = TagMapping { held_out: "(".into(), reference: "-LRB-".into() };
        Scoring::Tags(TagScoring { tokens: scored + left_out, left_out, reference_mappings: vec![mapping] })
    }

    fn record_text(units: Vec<PairedUnit>) -> String {
        let set = set("s", tags_of(&units, 0));
        let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
        ModelGateRecord::new(Task::PartOfSpeechTagger, set, model("t"), model("r"), units).unwrap().to_string()
    }

    #[test]
    fn a_small_loss_with_a_wide_interval_fails_and_a_narrow_one_passes() {
        // Two sentences of 40 with 7 right, the others 9: a pooled loss of
        // 0.5 points, but the interval reaches far below -0.5, so the
        // loss beyond the margin is not ruled out.
        let wide =
            gate_models(Measure::Accuracy, &tagging(40, |i| if i % 20 == 0 { 7 } else { 9 }), RESAMPLES, SEED).unwrap();
        assert!(wide.difference.low < -MARGIN_POINTS, "{wide:?}");
        assert!(!wide.pass);
        // 400 sentences of 100 tokens, the candidate one token behind on
        // one sentence in five: a pooled loss of 0.2 points with an
        // interval inside the margin.
        let units: Vec<PairedUnit> = (0..400)
            .map(|i| PairedUnit {
                candidate: Counts::accuracy(if i % 5 == 0 { 89 } else { 90 }, 100),
                reference: Counts::accuracy(90, 100),
            })
            .collect();
        let narrow = gate_models(Measure::Accuracy, &units, RESAMPLES, SEED).unwrap();
        assert!((narrow.difference.mean + 0.2).abs() < 1e-9, "{narrow:?}");
        assert!(narrow.difference.low >= -MARGIN_POINTS, "{narrow:?}");
        assert!(narrow.pass);
        // A better candidate passes.
        let better = gate_models(Measure::Accuracy, &tagging(40, |_| 10), RESAMPLES, SEED).unwrap();
        assert!(better.pass && better.difference.low > 0.0);
    }

    #[test]
    fn the_draws_are_java_random_s() {
        // One resample of 30 units, the odd ones right and the even ones
        // wrong: the candidate's score on that draw, which is the interval
        // at both ends, is the share of odd indices among the first 30
        // nextInt(30) of new Random(SEED).
        let units: Vec<PairedUnit> = (0..30)
            .map(|i| PairedUnit { candidate: Counts::accuracy(i % 2, 1), reference: Counts::accuracy(0, 1) })
            .collect();
        let mut r = JavaRandom::new(SEED);
        let hits = (0..30).filter(|_| r.next_int_bounded(30) % 2 == 1).count();
        let want = 100.0 * hits as f64 / 30.0;
        let g = gate_models(Measure::Accuracy, &units, 1, SEED).unwrap();
        assert_eq!((g.candidate.low, g.candidate.high), (want, want));
        assert_eq!((g.difference.low, g.difference.high), (want, want));
    }

    #[test]
    fn bad_units_are_refused() {
        let ok = PairedUnit { candidate: f(1, 1, 1), reference: f(1, 1, 1) };
        let mut units = vec![ok; 30];
        assert_eq!(gate_models(Measure::Accuracy, &units, 0, SEED), Err(ModelGateError::NoResamples));
        units[3] = PairedUnit { candidate: f(1, 1, 2), reference: f(1, 1, 1) };
        assert_eq!(
            gate_models(Measure::FMeasure, &units, RESAMPLES, SEED),
            Err(ModelGateError::DifferentReferences { unit: 3, candidate: 2, reference: 1 })
        );
        units[3] = PairedUnit { candidate: f(15, 10, 10), reference: f(10, 10, 10) };
        assert_eq!(
            gate_models(Measure::Accuracy, &units, RESAMPLES, SEED),
            Err(ModelGateError::AccuracyCounts { unit: 3 }),
            "more correct tokens than tokens"
        );
        units[3] = PairedUnit { candidate: f(1, 2, 1), reference: f(1, 2, 1) };
        assert_eq!(
            gate_models(Measure::Accuracy, &units, RESAMPLES, SEED),
            Err(ModelGateError::AccuracyCounts { unit: 3 })
        );
        assert!(gate_models(Measure::FMeasure, &units, RESAMPLES, SEED).is_ok());
    }

    fn record() -> ModelGateRecord {
        let set = set("corpus-1.0 held out", Scoring::Spans);
        let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
        ModelGateRecord::new(Task::Tokenizer, set, model("trained"), model("published"), tagging(40, |i| 8 + i % 3))
            .unwrap()
    }

    #[test]
    fn a_record_s_scoring_fits_its_measure_and_counts_the_scored_tokens() {
        let units = tagging(40, |i| 8 + i % 3);
        let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
        let tagger = |scoring| {
            ModelGateRecord::new(Task::PartOfSpeechTagger, set("s", scoring), model("t"), model("r"), units.clone())
                .unwrap()
        };
        let r = tagger(tags_of(&units, 3));
        r.check().unwrap();
        assert_eq!(r.set.skipped_sentences, 0);
        let json = r.to_json();
        assert!(
            json.contains("\"scoring\": {\n      \"tags\": {\n        \"tokens\": 403,\n        \"left_out\": 3,"),
            "{json}"
        );
        assert!(json.contains("\"held_out\": \"(\",\n            \"reference\": \"-LRB-\""), "{json}");
        let back: ModelGateRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);

        // The tokens less those left out must be what each model's units
        // count.
        let mut more = r.clone();
        let Scoring::Tags(t) = &mut more.set.scoring else { unreachable!() };
        t.tokens += 1;
        assert_eq!(more.check(), Err(ModelGateError::ScoredTokens { tokens: 404, left_out: 3, scored: 400 }));
        let mut over = r.clone();
        let Scoring::Tags(t) = &mut over.set.scoring else { unreachable!() };
        t.left_out = t.tokens + 1;
        assert!(matches!(over.check(), Err(ModelGateError::ScoredTokens { .. })));
        // Spans are an F-measure's scoring, tags an accuracy's.
        assert_eq!(tagger(Scoring::Spans).check(), Err(ModelGateError::ScoringMismatch { measure: Measure::Accuracy }));
        let mut spans = record();
        spans.set.scoring = tags_of(&spans.units, 0);
        assert_eq!(spans.check(), Err(ModelGateError::ScoringMismatch { measure: Measure::FMeasure }));
        assert_eq!(serde_json::to_string(&Scoring::Spans).unwrap(), "\"spans\"");
    }

    /// `n` held-out texts, the reference right on all but every tenth and
    /// the candidate on all but every `miss`th.
    fn labels(n: usize, miss: usize) -> Vec<PairedUnit> {
        (0..n)
            .map(|i| PairedUnit {
                candidate: Counts::accuracy(u64::from(i % miss != 0), 1),
                reference: Counts::accuracy(u64::from(i % 10 != 0), 1),
            })
            .collect()
    }

    fn language_record(units: Vec<PairedUnit>, scoring: Scoring) -> ModelGateRecord {
        let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
        ModelGateRecord::new(Task::LanguageDetector, set("s", scoring), model("t"), model("r"), units).unwrap()
    }

    fn labels_of(units: &[PairedUnit]) -> Scoring {
        let mapping = LabelMapping { held_out: "est".into(), reference: "ekk".into() };
        Scoring::Labels(LabelScoring { samples: units.len() as u64, reference_mappings: vec![mapping] })
    }

    #[test]
    fn a_language_detector_is_scored_one_unit_per_text() {
        assert_eq!(Task::LanguageDetector.measure(), Measure::Accuracy);
        // 2,000 texts: the reference wrong on one in ten, the candidate on
        // one in twenty, so the candidate is 5 points ahead.
        let units = labels(2000, 20);
        let r = language_record(units.clone(), labels_of(&units));
        r.check().unwrap();
        assert!((r.result.candidate.score - 95.0).abs() < 1e-9, "{:?}", r.result);
        assert!((r.result.reference.score - 90.0).abs() < 1e-9, "{:?}", r.result);
        assert!(r.result.pass);
        let json = r.to_json();
        assert!(json.contains("\"task\": \"language_detector\""), "{json}");
        assert!(json.contains("\"labels\": {\n        \"samples\": 2000,"), "{json}");
        let back: ModelGateRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);

        // One unit per sample, each of one sample.
        let mut more = r.clone();
        let Scoring::Labels(l) = &mut more.set.scoring else { unreachable!() };
        l.samples += 1;
        assert_eq!(more.check(), Err(ModelGateError::ScoredSamples { samples: 2001, units: 2000, unit: None }));
        let mut two = units.clone();
        two[7] = PairedUnit { candidate: Counts::accuracy(2, 2), reference: Counts::accuracy(1, 2) };
        let two = language_record(two.clone(), labels_of(&two));
        assert_eq!(two.check(), Err(ModelGateError::ScoredSamples { samples: 2000, units: 2000, unit: Some(7) }));
        // Labels are a language detector's scoring, and tags a tagger's.
        let tags = language_record(units.clone(), tags_of(&units, 0));
        assert_eq!(tags.check(), Err(ModelGateError::TaskScoringMismatch { task: Task::LanguageDetector }));
        let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
        let tagger =
            ModelGateRecord::new(Task::PartOfSpeechTagger, set("s", labels_of(&units)), model("t"), model("r"), units)
                .unwrap();
        assert_eq!(tagger.check(), Err(ModelGateError::TaskScoringMismatch { task: Task::PartOfSpeechTagger }));
        let mut spans = record();
        spans.set.scoring = labels_of(&spans.units);
        assert_eq!(spans.check(), Err(ModelGateError::ScoringMismatch { measure: Measure::FMeasure }));
    }

    #[test]
    fn a_label_scoring_skips_nothing_and_maps_each_label_once_to_another() {
        let units = labels(40, 20);
        let r = language_record(units.clone(), labels_of(&units));
        r.check().unwrap();
        let mut skipped = r.clone();
        skipped.set.skipped_sentences = 2;
        assert_eq!(skipped.check(), Err(ModelGateError::SkippedSamples { skipped: 2 }));
        let mapping =
            |held_out: &str, reference: &str| LabelMapping { held_out: held_out.into(), reference: reference.into() };
        let with = |mappings: Vec<LabelMapping>| {
            let mut m = r.clone();
            let Scoring::Labels(l) = &mut m.set.scoring else { unreachable!() };
            l.reference_mappings = mappings;
            m.check()
        };
        assert_eq!(with(vec![]), Ok(()));
        assert_eq!(with(vec![mapping("est", "ekk"), mapping("est", "et"), mapping("lvs", "lav")]), Ok(()));
        let refused = |held_out: &str, reference: &str| {
            Err(ModelGateError::LabelMapping { held_out: held_out.into(), reference: reference.into() })
        };
        assert_eq!(with(vec![mapping("", "ekk")]), refused("", "ekk"));
        assert_eq!(with(vec![mapping("est", "")]), refused("est", ""));
        assert_eq!(with(vec![mapping("lvs", "lav"), mapping("est", "est")]), refused("est", "est"));
        assert_eq!(
            with(vec![mapping("est", "ekk"), mapping("lvs", "lav"), mapping("est", "ekk")]),
            refused("est", "ekk")
        );
    }

    #[test]
    fn a_record_of_another_version_is_refused_for_its_version_before_its_fields_are_read() {
        assert_eq!(MODEL_GATE_RECORD_VERSION, 3);
        let units = labels(40, 20);
        let r = language_record(units.clone(), labels_of(&units));
        let json = r.to_json();
        assert!(json.starts_with("{\n  \"record_version\": 3,\n"), "{json}");
        let dir = std::env::temp_dir().join(format!("turbo-eval-model-gate-version-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gate.json");
        let read = |text: &str| {
            fs::write(&path, text).unwrap();
            ModelGateRecord::read(&path)
        };
        assert_eq!(read(&json).unwrap(), r);
        // A record of version 2 holds no language detector; a later one
        // may hold a task this build does not know. Both are refused for
        // their version, not as JSON this build cannot parse.
        let two = json.replacen("\"record_version\": 3", "\"record_version\": 2", 1);
        assert_eq!(read(&two), Err(ModelGateError::Version { found: 2 }));
        let later = json
            .replacen("\"record_version\": 3", "\"record_version\": 4", 1)
            .replacen("\"language_detector\"", "\"a_later_task\"", 1)
            .replacen("\"labels\": {", "\"a_later_scoring\": {", 1);
        assert_eq!(read(&later), Err(ModelGateError::Version { found: 4 }));
        // At this version, the same values are refused as JSON.
        let unknown = later.replacen("\"record_version\": 4", "\"record_version\": 3", 1);
        let Err(ModelGateError::Json { message, .. }) = read(&unknown) else { panic!("{unknown}") };
        assert!(message.contains("unknown variant `a_later_task`"), "{message}");
        // No version at all is not a record.
        assert!(matches!(read("{\"task\": \"tokenizer\"}"), Err(ModelGateError::Json { .. })));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_record_round_trips_and_is_checked_against_its_units() {
        let r = record();
        assert_eq!(r.measure, Measure::FMeasure);
        r.check().unwrap();
        let dir = std::env::temp_dir().join(format!("turbo-eval-model-gate-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gate.json");
        let _ = fs::remove_file(&path);
        r.write(&path).unwrap();
        assert!(matches!(r.write(&path), Err(ModelGateError::Io { .. })), "a record is never replaced");
        assert_eq!(ModelGateRecord::read(&path).unwrap(), r);
        fs::write(&path, "{}").unwrap();
        assert!(matches!(ModelGateRecord::read(&path), Err(ModelGateError::Json { .. })));
        fs::remove_dir_all(&dir).unwrap();

        let mut changed = r.clone();
        changed.units[0].candidate.true_positives += 1;
        assert_eq!(changed.check(), Err(ModelGateError::ResultMismatch));
        let mut wrong = r.clone();
        wrong.measure = Measure::Accuracy;
        assert!(matches!(wrong.check(), Err(ModelGateError::MeasureMismatch { .. })));
        let mut old = r.clone();
        old.record_version = 0;
        assert_eq!(old.check(), Err(ModelGateError::Version { found: 0 }));
        // A record drawn with fewer resamples or another seed is refused,
        // even though its result is what those parameters give.
        for (resamples, seed) in [(1, SEED), (RESAMPLES, SEED + 1)] {
            let mut other = r.clone();
            other.resamples = resamples;
            other.seed = seed;
            other.result = gate_models(other.measure, &other.units, resamples, seed).unwrap();
            assert_eq!(other.check(), Err(ModelGateError::Parameters { resamples, seed }));
        }
        let mut unnamed = r.clone();
        unnamed.candidate.id.clear();
        assert!(matches!(unnamed.check(), Err(ModelGateError::Field { field: "candidate.id", .. })));
        let mut upper = r.clone();
        upper.set.samples_sha256 = "B".repeat(64);
        assert!(matches!(upper.check(), Err(ModelGateError::Field { field: "set.samples_sha256", .. })));
        let mut short = r.clone();
        short.reference.sha256 = "c".repeat(63);
        assert!(matches!(short.check(), Err(ModelGateError::Field { field: "reference.sha256", .. })));
        let json = r.to_json().replace("\"seed\": 154,", "\"seed\": 154,\n  \"extra\": 1,");
        assert!(serde_json::from_str::<ModelGateRecord>(&json).is_err(), "unknown fields are refused");
        let text = r.to_string();
        assert!(text.contains("candidate trained:"), "{text}");
        assert_eq!(text.contains("PASS: "), r.result.pass, "{text}");
        assert_eq!(text.contains("FAIL: "), !r.result.pass, "{text}");
    }
}
