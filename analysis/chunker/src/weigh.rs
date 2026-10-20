//! The step that weighs the boundary signals and sums them into the
//! default `turbo:boundaries` instance, the one the chunker reads.

use turbo_analysis::{Annotator, Bound, Error, Refs};
use turbo_analysis_layers::{Boundaries, Cut, Sentences};

/// The instance the structural costs are written to beside other signals.
pub const STRUCTURE: &str = "structure";
/// The instance the drift costs are written to.
pub const DRIFT: &str = "drift";
/// The instance the sentiment, aspect, and incivility shift costs are written to.
pub const SENTIMENT: &str = "sentiment";
/// The instance the coreference chain continuity costs are written to.
pub const COREF: &str = "coref";
/// The instance the entity continuity costs are written to.
pub const ENTITY: &str = "entity";

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

/// Weighs structure, embedding drift, and sentiment/aspect shift boundary
/// costs at each gap: `(structure * ws + drift * wd + sentiment * wsent) / 1000`,
/// computed in `i128` so three products of `2^62` cannot wrap, rounded down
/// with `div_euclid(1000)`, and saturated to `i32`.
///
/// The reason is the structure's unless the structure saw only `Cut::Sentence`
/// and an active (`Weight != 0`) drift or sentiment signal marked `Cut::TopicShift`.
/// Returns `None` if the layers do not all have the same length.
pub fn weigh_with_sentiment(
    structure: &Boundaries,
    drift: &Boundaries,
    sentiment: &Boundaries,
    (ws, wd, wsent): (Weight, Weight, Weight),
) -> Option<Boundaries> {
    let n = structure.cost.len();
    if drift.cost.len() != n
        || sentiment.cost.len() != n
        || structure.reason.len() != n
        || drift.reason.len() != n
        || sentiment.reason.len() != n
    {
        return None;
    }
    let cost = (0..n)
        .map(|i| {
            let sum = i128::from(structure.cost[i]) * i128::from(ws.0)
                + i128::from(drift.cost[i]) * i128::from(wd.0)
                + i128::from(sentiment.cost[i]) * i128::from(wsent.0);
            sum.div_euclid(1000).clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
        })
        .collect();
    let reason = (0..n)
        .map(|i| {
            let (s, d, sent) = (structure.reason[i], drift.reason[i], sentiment.reason[i]);
            let shift = (d == Cut::TopicShift && wd.0 != 0) || (sent == Cut::TopicShift && wsent.0 != 0);
            if s == Cut::Sentence && shift { Cut::TopicShift } else { s }
        })
        .collect();
    Some(Boundaries { cost, reason })
}

/// Weighs structure, embedding drift, sentiment/aspect shift, and coreference
/// continuity boundary costs at each gap:
/// `(structure * ws + drift * wd + sentiment * wsent + coref * wc) / 1000`,
/// computed in `i128`, rounded down with `div_euclid(1000)`, and saturated to `i32`.
///
/// The reason is the structure's unless the structure saw only `Cut::Sentence`
/// and an active (`Weight != 0`) drift, sentiment, or coref signal marked
/// `Cut::TopicShift`. Returns `None` if the layers do not all have the same length.
pub fn weigh_with_coref(
    structure: &Boundaries,
    drift: &Boundaries,
    sentiment: &Boundaries,
    coref: &Boundaries,
    (ws, wd, wsent, wc): (Weight, Weight, Weight, Weight),
) -> Option<Boundaries> {
    let n = structure.cost.len();
    if drift.cost.len() != n
        || sentiment.cost.len() != n
        || coref.cost.len() != n
        || structure.reason.len() != n
        || drift.reason.len() != n
        || sentiment.reason.len() != n
        || coref.reason.len() != n
    {
        return None;
    }
    let cost = (0..n)
        .map(|i| {
            let sum = i128::from(structure.cost[i]) * i128::from(ws.0)
                + i128::from(drift.cost[i]) * i128::from(wd.0)
                + i128::from(sentiment.cost[i]) * i128::from(wsent.0)
                + i128::from(coref.cost[i]) * i128::from(wc.0);
            sum.div_euclid(1000).clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
        })
        .collect();
    let reason = (0..n)
        .map(|i| {
            let s = structure.reason[i];
            let d = drift.reason[i];
            let sent = sentiment.reason[i];
            let c = coref.reason[i];
            let shift = (d == Cut::TopicShift && wd.0 != 0)
                || (sent == Cut::TopicShift && wsent.0 != 0)
                || (c == Cut::TopicShift && wc.0 != 0);
            if s == Cut::Sentence && shift { Cut::TopicShift } else { s }
        })
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

/// Weighs structure (`STRUCTURE`), embedding drift (`DRIFT`), and optional
/// sentiment/aspect shift (`SENTIMENT`) into the default `Boundaries` instance.
///
/// By default `sentiment` is `Weight(0)` (off by default) so existing chunking
/// pipelines are unchanged unless a non-zero sentiment weight is configured.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WeighAllBoundaries {
    pub structure: Weight,
    pub drift: Weight,
    pub sentiment: Weight,
}

impl Default for WeighAllBoundaries {
    fn default() -> WeighAllBoundaries {
        WeighAllBoundaries { structure: Weight::ONE, drift: Weight::ONE, sentiment: Weight(0) }
    }
}

impl WeighAllBoundaries {
    /// Binds the step to read `STRUCTURE`, `DRIFT`, and `SENTIMENT`.
    pub fn bound(self) -> Bound<WeighAllBoundaries> {
        Bound::new(self).read(1, STRUCTURE).read(2, DRIFT).read(3, SENTIMENT)
    }
}

impl Annotator for WeighAllBoundaries {
    const ID: &'static str = "turbo:weigh-all-boundaries";
    type Reads = (Sentences, Boundaries, Option<Boundaries>, Option<Boundaries>);
    type Writes = (Boundaries,);

    fn annotate(&self, (sentences, structure, drift, sentiment): Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        let n = sentences.spans.len();
        if structure.cost.len() != n {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} structural boundaries for {n} sentences", structure.cost.len()),
            });
        }
        let zero;
        let drift_ref = match drift {
            Some(d) => {
                if d.cost.len() != n {
                    return Err(Error::Annotator {
                        annotator: Self::ID,
                        detail: format!("{} drift boundaries for {n} sentences", d.cost.len()),
                    });
                }
                d
            }
            None => {
                zero = Boundaries { cost: vec![0; n], reason: vec![Cut::Sentence; n] };
                &zero
            }
        };
        let zero_sent;
        let sent_ref = match sentiment {
            Some(s) => {
                if s.cost.len() != n {
                    return Err(Error::Annotator {
                        annotator: Self::ID,
                        detail: format!("{} sentiment boundaries for {n} sentences", s.cost.len()),
                    });
                }
                s
            }
            None => {
                zero_sent = Boundaries { cost: vec![0; n], reason: vec![Cut::Sentence; n] };
                &zero_sent
            }
        };
        let wd = if drift.is_some() { self.drift } else { Weight(0) };
        let wsent = if sentiment.is_some() { self.sentiment } else { Weight(0) };
        let out = weigh_with_sentiment(structure, drift_ref, sent_ref, (self.structure, wd, wsent))
            .ok_or_else(|| Error::Annotator { annotator: Self::ID, detail: "a reason for every boundary".into() })?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        [self.structure.0, self.drift.0, self.sentiment.0]
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
