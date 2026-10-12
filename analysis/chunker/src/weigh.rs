//! The step that weighs the boundary signals and sums them into the
//! default `turbo:boundaries` instance, the one the chunker reads.

use turbo_analysis::{Annotator, Bound, Error, Refs};
use turbo_analysis_layers::{Boundaries, Cut, Sentences};

/// The instance the structural costs are written to beside other signals.
pub const STRUCTURE: &str = "structure";
/// The instance the drift costs are written to.
pub const DRIFT: &str = "drift";

/// A signal's weight, in thousandths: `Weight(1000)` takes its costs as
/// they are, `Weight(500)` at half, `Weight(0)` leaves the signal out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Weight(pub i32);

impl Weight {
    pub const ONE: Weight = Weight(1000);
}

/// The weighted sum of the boundary costs at each gap, rounded down to a
/// whole unit: `(structure * ws + drift * wd) / 1000`, in i64, so no
/// product or sum wraps, then saturated to the i32 range. The reason is
/// the structure's, unless the structure saw only a sentence boundary and
/// the drift a topic shift.
///
/// The two layers must have the same length; `None` if not.
pub fn weigh(structure: &Boundaries, drift: &Boundaries, (ws, wd): (Weight, Weight)) -> Option<Boundaries> {
    if structure.cost.len() != drift.cost.len() || structure.reason.len() != drift.reason.len() {
        return None;
    }
    let cost = structure
        .cost
        .iter()
        .zip(&drift.cost)
        .map(|(&s, &d)| {
            // Each product is at most 2^62 either way; only their sum can
            // pass the i64 range, and then it saturates.
            let sum = (i64::from(s) * i64::from(ws.0)).saturating_add(i64::from(d) * i64::from(wd.0));
            sum.div_euclid(1000).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
        })
        .collect();
    let reason = structure
        .reason
        .iter()
        .zip(&drift.reason)
        .map(|(&s, &d)| if s == Cut::Sentence && d == Cut::TopicShift && wd.0 != 0 { d } else { s })
        .collect();
    Some(Boundaries { cost, reason })
}

/// Weighs the structural costs (the `structure` instance) against the
/// drift (the `drift` instance) and writes the default instance. `bound`
/// binds it to those instances; the structural step is then bound to
/// write `STRUCTURE` (`Bound::new(StructureBoundaries { .. }).write(0,
/// STRUCTURE)`) and the drift step with `DriftBoundaries::bound`.
///
/// A further signal is one more read and one more weight here.
pub struct WeighBoundaries {
    pub structure: Weight,
    pub drift: Weight,
}

impl Default for WeighBoundaries {
    /// Both signals as they are: untuned until the evaluation harness
    /// measures them.
    fn default() -> WeighBoundaries {
        WeighBoundaries { structure: Weight::ONE, drift: Weight::ONE }
    }
}

impl WeighBoundaries {
    /// The step, reading the `structure` and `drift` instances.
    pub fn bound(self) -> Bound<WeighBoundaries> {
        Bound::new(self).read(1, STRUCTURE).read(2, DRIFT)
    }
}

impl Annotator for WeighBoundaries {
    const ID: &'static str = "turbo:weigh-boundaries";
    // The sentences are read because the boundaries are aligned to them.
    type Reads = (Sentences, Boundaries, Boundaries);
    type Writes = (Boundaries,);

    fn annotate(&self, (sentences, structure, drift): Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        let n = sentences.spans.len();
        if structure.cost.len() != n || drift.cost.len() != n {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} and {} boundaries for {n} sentences", structure.cost.len(), drift.cost.len()),
            });
        }
        let out = weigh(structure, drift, (self.structure, self.drift))
            .ok_or_else(|| Error::Annotator { annotator: Self::ID, detail: "a reason for every boundary".into() })?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        [self.structure.0, self.drift.0]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x as u32 as u64).wrapping_mul(0x0100_0000_01b3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(cost: &[i32], reason: &[Cut]) -> Boundaries {
        Boundaries { cost: cost.to_vec(), reason: reason.to_vec() }
    }

    #[test]
    fn costs_are_weighed_summed_and_rounded_down() {
        use Cut::*;
        let s = b(&[1000, 400, 1000, 0], &[Sentence, Paragraph, Sentence, Section]);
        let d = b(&[-1500, -1500, -3, 0], &[TopicShift, TopicShift, Sentence, Sentence]);
        let w = weigh(&s, &d, (Weight::ONE, Weight(500))).unwrap();
        // -1.5 rounds down to -2, never toward zero.
        assert_eq!(w.cost, [250, -350, 998, 0]);
        // A topic shift names a plain sentence boundary; a stronger
        // structural reason stays.
        assert_eq!(w.reason, [TopicShift, Paragraph, Sentence, Section]);
        // A signal weighed at zero is left out, reason and all.
        let w = weigh(&s, &d, (Weight::ONE, Weight(0))).unwrap();
        assert_eq!((w.cost, w.reason), (s.cost.clone(), s.reason.clone()));
    }

    #[test]
    fn sums_saturate_instead_of_wrapping() {
        let r = [Cut::Sentence; 4];
        let s = b(&[i32::MAX, i32::MIN, i32::MAX, i32::MIN], &r);
        let d = b(&[i32::MAX, i32::MIN, i32::MIN, i32::MAX], &r);
        let w = weigh(&s, &d, (Weight(i32::MAX), Weight(i32::MAX))).unwrap();
        // The extremes cancel to one less than zero, times the weight.
        assert_eq!(w.cost, [i32::MAX, i32::MIN, -2_147_484, -2_147_484]);
        let w = weigh(&s, &d, (Weight(i32::MIN), Weight::ONE)).unwrap();
        assert_eq!(w.cost[..2], [i32::MIN, i32::MAX]);
        // Two products of 2^62 sum past the i64 range, and saturate there.
        let w = weigh(&s, &s, (Weight(i32::MIN), Weight(i32::MIN))).unwrap();
        assert_eq!(w.cost, [i32::MIN, i32::MAX, i32::MIN, i32::MAX]);
    }

    #[test]
    fn layers_of_different_lengths_are_refused() {
        let s = b(&[1000, 1000], &[Cut::Sentence; 2]);
        let d = b(&[0], &[Cut::Sentence]);
        assert!(weigh(&s, &d, (Weight::ONE, Weight::ONE)).is_none());
    }
}
