//! Scoring spans for the gate: a held-out unit's counts from the
//! `FMeasure` OpenNLP's evaluator keeps for it, both models' counts paired
//! unit by unit, and the types a gate of typed spans (names, chunks)
//! scores.
//!
//! Each unit is scored with an evaluator of its own, so its counts are
//! what `FMeasure.updateScores` adds for that unit alone, and the units'
//! counts summed are the counts the evaluator keeps over the whole set.
//! The gate ([`turbo_eval::model_gate`]) resamples whole units and scores
//! each draw from its summed counts; it never averages the units' own
//! F-measures.

use turbo_eval::model_gate::{Counts, PairedUnit, Scoring, TagMapping, TypeMapping, TypedSpanScoring};
use turbo_formats::evaluate::FMeasure;

/// A unit's counts from the `FMeasure` its evaluator kept for it: the
/// true positives (as `FMeasure.countTruePositives` counts them), the
/// predictions and the references.
pub fn counts(f: &FMeasure) -> Counts {
    Counts { true_positives: f.true_positives(), selected: f.selected(), target: f.target() }
}

/// Both models' counts on the same units, paired in the units' order.
///
/// # Panics
///
/// When the two models were scored on different numbers of units, which
/// no recipe does.
pub fn pair(candidate: Vec<Counts>, reference: Vec<Counts>) -> Vec<PairedUnit> {
    assert_eq!(candidate.len(), reference.len(), "both models are scored on the same units");
    candidate.into_iter().zip(reference).map(|(candidate, reference)| PairedUnit { candidate, reference }).collect()
}

/// The span types a gate of typed spans scores, and the held-out types
/// it renames to them. Only spans of [`SpanTypes::types`] count, on both
/// sides: a held-out span of another type is left out of the references,
/// as OpenNLP's `NameSampleTypeFilter` leaves it out of a name sample, and
/// a predicted span of another type is left out of the predictions, so
/// neither model is scored on a type the other does not give.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanTypes {
    /// The types scored, as the models name them.
    pub types: &'static [&'static str],
    /// Held-out types the models name otherwise: the held-out type, then
    /// the type it is scored as, which is one of `types`.
    pub renamed: &'static [(&'static str, &'static str)],
}

impl SpanTypes {
    /// The type a held-out span of type `kind` is scored as: `kind` when
    /// it is scored, the type it is renamed to, or `None` when it is left
    /// out.
    pub fn held_out(&self, kind: &str) -> Option<&'static str> {
        if let Some(&(_, scored)) = self.renamed.iter().find(|(held_out, _)| *held_out == kind) {
            return Some(scored);
        }
        self.types.iter().find(|t| **t == kind).copied()
    }

    /// Whether a predicted span of type `kind` is scored. Predictions are
    /// never renamed: a model gives the types it was trained on.
    pub fn predicted(&self, kind: &str) -> bool {
        self.types.contains(&kind)
    }

    /// The record's scoring: `spans` held-out spans scored, after
    /// renaming, `left_out` held-out spans of other types, and
    /// `reference_inputs`, the held-out input tags the reference is given
    /// another tag for (empty when both models read the held-out input as
    /// written).
    pub fn scoring(&self, spans: u64, left_out: u64, reference_inputs: Vec<TagMapping>) -> Scoring {
        Scoring::TypedSpans(TypedSpanScoring {
            types: self.types.iter().map(|t| (*t).to_owned()).collect(),
            renamed: self
                .renamed
                .iter()
                .map(|&(held_out, scored)| TypeMapping { held_out: held_out.to_owned(), scored: scored.to_owned() })
                .collect(),
            reference_inputs,
            spans,
            left_out,
        })
    }
}

#[cfg(test)]
mod tests {
    use turbo_eval::model_gate::Measure;

    use super::*;

    #[test]
    fn a_unit_s_counts_are_its_evaluator_s_and_add_up_to_the_set_s() {
        let (mut set, mut a, mut b) = (FMeasure::default(), FMeasure::default(), FMeasure::default());
        a.update(&["x", "y"], &["x", "z", "w"]);
        b.update(&["q"], &["q"]);
        set.update(&["x", "y"], &["x", "z", "w"]);
        set.update(&["q"], &["q"]);
        assert_eq!(counts(&a), Counts { true_positives: 1, selected: 3, target: 2 });
        let units = pair(vec![counts(&a), counts(&b)], vec![counts(&b), counts(&a)]);
        assert_eq!(units[1].reference, counts(&a));
        let mut sum = Counts::default();
        for u in &units {
            sum = Counts {
                true_positives: sum.true_positives + u.candidate.true_positives,
                selected: sum.selected + u.candidate.selected,
                target: sum.target + u.candidate.target,
            };
        }
        assert_eq!(sum, counts(&set));
        assert_eq!(sum.score(Measure::FMeasure), 100.0 * set.f_measure());
    }

    #[test]
    #[should_panic(expected = "both models are scored on the same units")]
    fn units_are_paired_only_when_both_sides_have_them_all() {
        pair(vec![Counts::default()], Vec::new());
    }

    #[test]
    fn held_out_types_are_renamed_or_left_out_and_predictions_are_not_renamed() {
        let types = SpanTypes { types: &["organization", "person"], renamed: &[("org", "organization")] };
        assert_eq!(types.held_out("org"), Some("organization"));
        assert_eq!(types.held_out("organization"), Some("organization"));
        assert_eq!(types.held_out("person"), Some("person"));
        assert_eq!(types.held_out("date"), None);
        assert!(types.predicted("organization") && types.predicted("person"));
        assert!(!types.predicted("org") && !types.predicted("date"));
        let Scoring::TypedSpans(s) = types.scoring(12, 3, Vec::new()) else { panic!("typed spans") };
        assert_eq!(s.types, ["organization", "person"]);
        assert_eq!(s.renamed, [TypeMapping { held_out: "org".into(), scored: "organization".into() }]);
        assert_eq!((s.spans, s.left_out), (12, 3));
        assert!(s.reference_inputs.is_empty());
        let input = TagMapping { held_out: "-".into(), reference: ":".into() };
        let Scoring::TypedSpans(s) = types.scoring(12, 3, vec![input.clone()]) else { panic!("typed spans") };
        assert_eq!(s.reference_inputs, [input]);
    }
}
