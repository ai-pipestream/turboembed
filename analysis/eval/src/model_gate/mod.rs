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
//! For an F-measure the bootstrap resamples whole units, the sentences
//! or documents a recipe scores, and scores each draw as OpenNLP's
//! `FMeasure` scores a set: it sums the true positives, predictions and
//! references of the units drawn and computes precision, recall and
//! their harmonic mean from those sums. It never averages the units' own
//! F-measures. A mean of per unit values would weigh a sentence of one
//! name as much as a sentence of ten, would not be the score OpenNLP's
//! evaluators report for the whole set, and has no value at all for the
//! many sentences with neither a reference nor a predicted span, where
//! precision and recall are both undefined. The names of a sentence stay
//! together in a draw, so the interval reflects how spans cluster in
//! sentences.
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

mod error;
mod record;
mod scoring;
mod sums;
#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use turbo_jvm::random::JavaRandom;

use crate::gate::FLOOR_POINTS;
use crate::significance::{Bootstrap, percentile};
pub use crate::significance::{RESAMPLES, SEED};

pub use error::ModelGateError;
pub use record::ModelGateRecord;
pub use scoring::{
    HeldOutSet, LabelMapping, LabelScoring, ModelId, Scoring, ScoringKind, TagMapping, TagScoring, Task, TypeMapping,
    TypedSpanScoring,
};

use error::Result;
use sums::{Totals, total};

/// The schema version of [`ModelGateRecord`]: 3, with the name finder,
/// chunker and language detector tasks and their scorings
/// ([`Scoring::TypedSpans`], [`Scoring::Labels`]). Version 2 added the
/// held-out sentences skipped and how the units were scored ([`HeldOutSet`]).
/// [`ModelGateRecord::read`] reads the version before the rest, so a record
/// of another version is refused for its version, not for a task or scoring
/// this build does not know.
pub const MODEL_GATE_RECORD_VERSION: u32 = 3;

/// The loss, in points, the candidate may be shown not to exceed: the low
/// end of the difference's interval must be at least the negative of it.
/// The chunker gate's floor, so both gates allow the same loss.
pub const MARGIN_POINTS: f64 = -FLOOR_POINTS;

/// The fewest held-out units the gate scores. Below it the bootstrap
/// interval says little about the set's spread.
pub const MIN_UNITS: usize = 30;

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

    /// The score of these counts under `measure`, in points.
    pub fn score(&self, measure: Measure) -> f64 {
        let mut t = Totals::default();
        t.add(self);
        t.score(measure)
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

/// Scores the paired units and decides the gate, drawing `resamples`
/// resamples from `java.util.Random` seeded with `seed`: each resample is
/// `units.len()` indices from `nextInt(units.len())`, in order.
///
/// # Errors
///
/// Fewer than [`MIN_UNITS`] units, more than `nextInt` can index, no
/// resamples, a unit whose two sides count different references (they
/// did not score the same reference), accuracy counts whose predictions
/// are not their references or have more correct tokens than tokens, or
/// F-measure counts with more true positives than predictions or
/// references.
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
        let f_ok = |c: &Counts| c.true_positives <= c.selected && c.true_positives <= c.target;
        if measure == Measure::FMeasure && !(f_ok(&u.candidate) && f_ok(&u.reference)) {
            return Err(ModelGateError::FMeasureCounts { unit });
        }
    }
    let candidate = total(units.iter().map(|u| &u.candidate)).score(measure);
    let reference = total(units.iter().map(|u| &u.reference)).score(measure);
    let mut random = JavaRandom::new(seed);
    let [cand, refr, diff] = sums::draws(measure, units, n, resamples, &mut random);
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

#[cfg(test)]
mod unit_tests {
    use super::tests::{f, record_text, tagging};
    use super::*;

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
    fn an_f_measure_draw_is_scored_from_the_summed_counts_of_the_units_drawn() {
        // 40 sentences: every fourth has three names, the candidate finding
        // one of them and predicting two more spans; the others have no
        // name, and the candidate predicts a span in every fifth of those.
        let unit = |i: u64| {
            let candidate = match (i % 4, i % 5) {
                (0, _) => f(1, 3, 3),
                (_, 0) => f(0, 1, 0),
                _ => f(0, 0, 0),
            };
            let target = if i.is_multiple_of(4) { 3 } else { 0 };
            PairedUnit { candidate, reference: f(target, target, target) }
        };
        let units: Vec<PairedUnit> = (0..40).map(unit).collect();
        // One draw, so both ends of the interval are that draw's score:
        // the F-measure of the drawn units' summed counts.
        let mut r = JavaRandom::new(SEED);
        let mut sum = Totals::default();
        let mut unit_scores = Vec::new();
        for _ in 0..units.len() {
            let u = &units[r.next_int_bounded(units.len() as i32) as usize];
            sum.add(&u.candidate);
            // A per unit F-measure exists only where the unit has a
            // reference or a prediction.
            if u.candidate.selected + u.candidate.target > 0 {
                unit_scores.push(u.candidate.score(Measure::FMeasure));
            }
        }
        let want = sum.score(Measure::FMeasure);
        let g = gate_models(Measure::FMeasure, &units, 1, SEED).unwrap();
        assert_eq!((g.candidate.low, g.candidate.high), (want, want));
        // Not the mean of the units' own F-measures, which differs here.
        let mean_of_units = unit_scores.iter().sum::<f64>() / unit_scores.len() as f64;
        assert!((mean_of_units - want).abs() > 1.0, "{mean_of_units} {want}");
        // The whole set: 10 of 30 names right, 30 + 6 predictions.
        let pooled = f(10, 36, 30).score(Measure::FMeasure);
        assert_eq!(g.candidate.score, pooled);
        assert_eq!(g.reference.score, 100.0);
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
        // F-measure counts with more true positives than predictions or references are refused.
        units[3] = PairedUnit { candidate: f(2, 1, 2), reference: f(2, 2, 2) };
        assert_eq!(
            gate_models(Measure::FMeasure, &units, RESAMPLES, SEED),
            Err(ModelGateError::FMeasureCounts { unit: 3 })
        );
        units[3] = PairedUnit { candidate: f(2, 2, 1), reference: f(1, 1, 1) };
        assert_eq!(
            gate_models(Measure::FMeasure, &units, RESAMPLES, SEED),
            Err(ModelGateError::FMeasureCounts { unit: 3 })
        );
        // Large unit counts accumulate in u128 so 50% vs 100% stays 50.0 vs 100.0 (and fails).
        let half = Counts::accuracy(u64::MAX / 2, u64::MAX);
        let full = Counts::accuracy(u64::MAX, u64::MAX);
        let huge_units = vec![PairedUnit { candidate: half, reference: full }; MIN_UNITS];
        for m in [Measure::Accuracy, Measure::FMeasure] {
            let g = gate_models(m, &huge_units, 4, SEED).unwrap();
            assert!((g.candidate.score - 50.0).abs() < 1e-9, "{g:?}");
            assert_eq!(g.reference.score, 100.0);
            assert!(!g.pass, "{g:?}");
        }
    }
}
