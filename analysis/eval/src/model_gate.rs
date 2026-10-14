//! The gate a trained model passes against a reference model: the
//! OpenNLP model for the same task, scored on the same held-out samples.
//!
//! Each held-out unit (a sentence, or a sample of sentences) is scored
//! by both models as counts: true positives, predictions and references
//! for an F-measure, or correct and total tokens for an accuracy. The
//! score of a set of units is that of their summed counts, as OpenNLP's
//! `FMeasure` and `Mean` add them up. A paired bootstrap draws the units
//! with replacement, [`RESAMPLES`] times with `java.util.Random` seeded
//! with [`SEED`] (the draws of [`crate::significance::bootstrap`]), and
//! scores both models on each draw, so each resample gives both scores
//! and their difference.
//!
//! The candidate passes when the 95 percent interval of the difference,
//! candidate minus reference, reaches zero or above: its high end is not
//! below zero. A candidate that is worse on every draw fails; one whose
//! interval spans zero is not shown to be worse, and passes. Each model's
//! own interval is reported beside the difference.
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

use crate::Result;
use crate::significance::{Bootstrap, percentile};
pub use crate::significance::{RESAMPLES, SEED};

/// The schema version of [`ModelGateRecord`].
pub const MODEL_GATE_RECORD_VERSION: u32 = 1;

/// What a unit's counts are, and so how they make a score.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Measure {
    /// `true_positives` are predictions that match a reference,
    /// `selected` the predictions and `target` the references; the score
    /// is the harmonic mean of precision and recall.
    FMeasure,
    /// `true_positives` are the tokens given the reference's value and
    /// `target` the tokens; `selected` equals `target`. The score is their
    /// ratio.
    Accuracy,
}

/// The task the two models perform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    SentenceDetector,
    Tokenizer,
    PartOfSpeechTagger,
}

impl Task {
    /// The measure OpenNLP's evaluator for the task reports.
    pub fn measure(self) -> Measure {
        match self {
            Task::SentenceDetector | Task::Tokenizer => Measure::FMeasure,
            Task::PartOfSpeechTagger => Measure::Accuracy,
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
    /// interval of the resampled differences.
    pub difference: Bootstrap,
    /// The difference's interval reaches zero or above.
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
/// No units, more units than `nextInt` can index, no resamples, a unit
/// whose two sides count different references (they did not score the
/// same reference), or accuracy counts whose predictions are not their
/// references.
pub fn gate_models(measure: Measure, units: &[PairedUnit], resamples: u32, seed: i64) -> Result<ModelGate> {
    if units.is_empty() {
        return Err("no held-out units".into());
    }
    let n = i32::try_from(units.len())
        .map_err(|_| format!("{} units, more than java.util.Random can index", units.len()))?;
    if resamples == 0 {
        return Err("no resamples".into());
    }
    for (i, u) in units.iter().enumerate() {
        if u.candidate.target != u.reference.target {
            return Err(format!(
                "unit {i}: the candidate was scored against {} references and the reference model against {}",
                u.candidate.target, u.reference.target
            ));
        }
        if measure == Measure::Accuracy
            && (u.candidate.selected != u.candidate.target || u.reference.selected != u.reference.target)
        {
            return Err(format!("unit {i}: accuracy counts with predictions that are not the tokens"));
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
        pass: difference.high >= 0.0,
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

    /// Checks the version, that the measure is the task's, and that the
    /// result is what the gate gives on the recorded units, to the bit.
    pub fn check(&self) -> Result<()> {
        if self.record_version != MODEL_GATE_RECORD_VERSION {
            return Err(format!(
                "record version {}, this build reads {MODEL_GATE_RECORD_VERSION}",
                self.record_version
            ));
        }
        if self.measure != self.task.measure() {
            return Err(format!("a {:?} record with the measure {:?}", self.task, self.measure));
        }
        let again = gate_models(self.measure, &self.units, self.resamples, self.seed)?;
        if again != self.result {
            return Err("the recorded result is not what the gate gives on the recorded units".into());
        }
        Ok(())
    }

    /// The record as written: pretty JSON with a final newline.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a gate record serializes");
        s.push('\n');
        s
    }

    /// Reads and checks a record.
    pub fn read(path: &Path) -> Result<ModelGateRecord> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let record: ModelGateRecord = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        record.check().map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(record)
    }

    /// Writes the record to `path`, which must not exist yet.
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        f.write_all(self.to_json().as_bytes()).map_err(|e| format!("{}: {e}", path.display()))
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
        let verdict = if r.pass {
            "PASS: the difference's interval reaches zero or above"
        } else {
            "FAIL: the difference's interval is below zero"
        };
        write!(f, "{verdict}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(tp: u64, selected: u64, target: u64) -> Counts {
        Counts { true_positives: tp, selected, target }
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
        // The set's score is that of the sum, not the mean of the units'.
        let units = [
            PairedUnit { candidate: f(1, 1, 1), reference: f(1, 1, 1) },
            PairedUnit { candidate: f(0, 3, 3), reference: f(3, 3, 3) },
        ];
        let g = gate_models(Measure::FMeasure, &units, 100, SEED).unwrap();
        assert_eq!(g.candidate.score, 25.0);
        assert_eq!(g.reference.score, 100.0);
        // A draw of the first unit twice scores both models 100, so with
        // two units the difference's interval reaches zero.
        assert_eq!(g.difference.mean, -75.0);
        assert_eq!(g.difference.high, 0.0);
    }

    #[test]
    fn identical_models_pass_with_a_point_difference() {
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

    /// 40 sentences of 10 tokens; the reference gets 9 right on each, the
    /// candidate `right(i)` on sentence i.
    fn tagging(right: impl Fn(u64) -> u64) -> Vec<PairedUnit> {
        (0..40)
            .map(|i| PairedUnit { candidate: Counts::accuracy(right(i), 10), reference: Counts::accuracy(9, 10) })
            .collect()
    }

    #[test]
    fn a_candidate_worse_on_every_draw_fails() {
        let g = gate_models(Measure::Accuracy, &tagging(|_| 8), RESAMPLES, SEED).unwrap();
        assert_eq!(g.difference, Bootstrap { mean: -10.0, low: -10.0, high: -10.0 });
        assert!(!g.pass);
    }

    #[test]
    fn a_difference_whose_interval_spans_zero_passes_and_one_below_zero_fails() {
        // Two sentences of 40 with 7 right, the others 9: a mean of -0.5
        // points. A draw of 40 misses both in 0.95^40, 13 percent, of the
        // draws, so the interval reaches zero.
        let spans =
            gate_models(Measure::Accuracy, &tagging(|i| if i % 20 == 0 { 7 } else { 9 }), RESAMPLES, SEED).unwrap();
        assert!(spans.difference.mean < 0.0 && spans.difference.high >= 0.0, "{spans:?}");
        assert!(spans.pass);
        let below =
            gate_models(Measure::Accuracy, &tagging(|i| if i % 2 == 0 { 7 } else { 9 }), RESAMPLES, SEED).unwrap();
        assert!(below.difference.high < 0.0, "{below:?}");
        assert!(!below.pass);
        // A better candidate passes.
        let better = gate_models(Measure::Accuracy, &tagging(|_| 10), RESAMPLES, SEED).unwrap();
        assert!(better.pass && better.difference.low > 0.0);
    }

    #[test]
    fn the_draws_are_java_random_s() {
        // One resample of two units: the indices are the first two
        // nextInt(2) of new Random(SEED), so the candidate's score on that
        // draw, which is the interval at both ends, is fixed by them.
        let units = [
            PairedUnit { candidate: Counts::accuracy(0, 1), reference: Counts::accuracy(0, 1) },
            PairedUnit { candidate: Counts::accuracy(1, 1), reference: Counts::accuracy(0, 1) },
        ];
        let mut r = JavaRandom::new(SEED);
        let hits = (0..2).map(|_| r.next_int_bounded(2) as u64).sum::<u64>();
        let want = 100.0 * hits as f64 / 2.0;
        let g = gate_models(Measure::Accuracy, &units, 1, SEED).unwrap();
        assert_eq!((g.candidate.low, g.candidate.high), (want, want));
        assert_eq!((g.difference.low, g.difference.high), (want, want));
    }

    #[test]
    fn bad_units_are_refused() {
        assert!(gate_models(Measure::Accuracy, &[], RESAMPLES, SEED).is_err());
        let ok = PairedUnit { candidate: f(1, 1, 1), reference: f(1, 1, 1) };
        assert!(gate_models(Measure::Accuracy, &[ok], 0, SEED).is_err());
        let other_gold = PairedUnit { candidate: f(1, 1, 2), reference: f(1, 1, 1) };
        assert!(gate_models(Measure::FMeasure, &[other_gold], RESAMPLES, SEED).is_err());
        let not_tokens = PairedUnit { candidate: f(1, 2, 1), reference: f(1, 2, 1) };
        assert!(gate_models(Measure::Accuracy, &[not_tokens], RESAMPLES, SEED).is_err());
        assert!(gate_models(Measure::FMeasure, &[not_tokens], RESAMPLES, SEED).is_ok());
    }

    fn record() -> ModelGateRecord {
        let set = HeldOutSet {
            name: "corpus-1.0 held out".into(),
            corpus_sha256: "a".repeat(64),
            samples_sha256: "b".repeat(64),
        };
        let model = |id: &str| ModelId { id: id.into(), sha256: "c".repeat(64) };
        ModelGateRecord::new(Task::Tokenizer, set, model("trained"), model("published"), tagging(|i| 8 + i % 3))
            .unwrap()
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
        assert!(r.write(&path).is_err(), "a record is never replaced");
        assert_eq!(ModelGateRecord::read(&path).unwrap(), r);
        fs::remove_dir_all(&dir).unwrap();

        let mut changed = r.clone();
        changed.units[0].candidate.true_positives += 1;
        assert!(changed.check().is_err(), "counts that do not give the result");
        let mut wrong = r.clone();
        wrong.measure = Measure::Accuracy;
        assert!(wrong.check().is_err());
        let json = r.to_json().replace("\"seed\": 154,", "\"seed\": 154,\n  \"extra\": 1,");
        assert!(serde_json::from_str::<ModelGateRecord>(&json).is_err(), "unknown fields are refused");
        let text = r.to_string();
        assert!(text.contains("candidate trained:"), "{text}");
        assert_eq!(text.ends_with("reaches zero or above"), r.result.pass, "{text}");
    }
}
