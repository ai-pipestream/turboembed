//! The drift term of `turbo:boundaries`: how much the sentence vectors
//! change across each gap, as a cost that is lower where they change more,
//! so the optimizer prefers to cut there.
//!
//! For the gap after sentence g it compares the sum of the unit vectors of
//! the `window` sentences before it with the sum of the `window` after it
//! (adjacent groups, as semantic splitters do, rather than single
//! sentences, so one odd sentence does not look like a topic of its own).
//! The cosine is rounded once to millionths; everything after it is
//! integer arithmetic. The cosines are smoothed with a moving average, and
//! each gap is scored by its TextTiling depth (Hearst 1997): how far the
//! smoothed cosine climbs on each side before it falls again, so a valley
//! between two coherent stretches scores high and a steady slow drift
//! scores nothing.
//!
//! Only gaps with a full window of sentences with a vector on both sides
//! are scored, as NLTK's TextTiling leaves the gaps at the ends unscored.
//! With a window cut short at an end, a single odd first or last sentence
//! is compared alone, its gap is a valley with a wall on one side only,
//! and the optimizer sets it apart as a chunk of its own. The price is no
//! drift within `window - 1` sentences of either end, where a cut would
//! leave a chunk of fewer than `window` sentences anyway.

use std::num::NonZeroU32;

use turbo_analysis::{Annotator, Bound, Error, Layer, Refs};
use turbo_analysis_layers::{Boundaries, Cut, Sentences, Vectors};

use crate::sums::{Rows, checked_rows};
use crate::weigh::DRIFT;

/// A cosine of one, in the integer units the drift is computed in.
pub const MICRO: i64 = 1_000_000;

/// What a gap is scored by, from the smoothed cosines.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Score {
    /// The TextTiling depth: the highest smoothed cosine reached climbing
    /// left from the gap, plus the highest reached climbing right, less
    /// twice the gap's own. A slow drift has no valleys and scores zero.
    Depth,
    /// One less the smoothed cosine: the distance the common semantic
    /// splitters threshold. A slow drift scores the same at every gap,
    /// which lowers every cut alike and so favours more chunks.
    Distance,
}

/// How the drift is computed and what it costs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Drift {
    /// Sentences with a vector on each side of a gap. A gap with fewer
    /// on either side is not scored.
    pub window: NonZeroU32,
    /// The moving average takes this many cosines on each side of a gap.
    pub smoothing: u32,
    pub score: Score,
    /// Scores under this, in millionths, count as zero, so rounding noise
    /// on a steady stretch is not a signal.
    pub floor: u32,
    /// What a score of one takes off the gap, in thousandths of a
    /// `Boundaries::UNIT` (the units costs are written in): a gap costs
    /// `-score * scale / MICRO`, truncated toward zero, never more than
    /// zero.
    pub scale: u32,
    /// A gap at the bottom of a valley that scores at least this, in
    /// millionths, gives `Cut::TopicShift` as its reason.
    pub shift: u32,
}

impl Default for Drift {
    /// Untuned: these wait on the evaluation harness. A scale of 2000
    /// makes a depth of 0.5 take a plain sentence boundary's
    /// `Boundaries::UNIT` (1000) off the gap.
    fn default() -> Drift {
        Drift {
            window: NonZeroU32::new(3).expect("not zero"),
            smoothing: 1,
            score: Score::Depth,
            floor: 10_000,
            scale: 2 * Boundaries::UNIT as u32,
            shift: 100_000,
        }
    }
}

/// The cosine across each gap, in millionths, from the `window` sentences
/// with a vector on each side; `None` where either side has fewer, which
/// includes the gap after the last sentence.
pub fn cosines(rows: &Rows<'_>, window: NonZeroU32) -> Vec<Option<i64>> {
    let w = window.get() as usize;
    let present = rows.present();
    (0..rows.len())
        .map(|g| {
            let at = present.partition_point(|&p| p as usize <= g);
            if at < w || present.len() - at < w {
                return None;
            }
            let c = rows.cosine(&present[at - w..at], &present[at..at + w])?;
            Some((c * MICRO as f64).round() as i64)
        })
        .collect()
}

/// The moving average of `xs` over `s` on each side, rounded to the
/// nearest integer. Near the ends it takes as many on each side as the
/// nearer end allows, so the average stays centred: one cut short on one
/// side only would pull an edge toward its neighbours and move a valley
/// next to an end onto the end.
fn smooth(xs: &[i64], s: usize) -> Vec<i64> {
    let mut prefix = vec![0i64; xs.len() + 1];
    for (i, &x) in xs.iter().enumerate() {
        prefix[i + 1] = prefix[i] + x;
    }
    (0..xs.len())
        .map(|i| {
            let r = s.min(i).min(xs.len() - 1 - i);
            let (a, b) = (i - r, i + r + 1);
            let (sum, n) = (prefix[b] - prefix[a], (b - a) as i64);
            (2 * sum + n).div_euclid(2 * n)
        })
        .collect()
}

/// The TextTiling depth of each point: the highest value reached walking
/// left while the values do not fall, plus the same to the right, less
/// twice its own. Linear: a walk from i continues the walk from its
/// neighbour whenever that neighbour is not lower.
fn depths(xs: &[i64]) -> Vec<i64> {
    let n = xs.len();
    let mut left = xs.to_vec();
    for i in 1..n {
        if xs[i - 1] >= xs[i] {
            left[i] = left[i - 1];
        }
    }
    let mut right = xs.to_vec();
    for i in (0..n.saturating_sub(1)).rev() {
        if xs[i + 1] >= xs[i] {
            right[i] = right[i + 1];
        }
    }
    (0..n).map(|i| left[i] + right[i] - 2 * xs[i]).collect()
}

/// The drift cost and reason of every gap. A gap with no cosine (fewer
/// than `window` sentences with a vector on one side) costs zero and is a
/// plain `Sentence`; the others are smoothed and scored among themselves,
/// in order, skipping it.
pub fn drift(rows: &Rows<'_>, d: &Drift) -> (Vec<i32>, Vec<Cut>) {
    let cos = cosines(rows, d.window);
    score_cosines(&cos, d)
}

/// Checked vector-layer entry point for [`drift`]. Refuses invalid row
/// shape/empty metadata and non-finite coordinates with the same string
/// errors as vector projection. Scores are identical to `drift` for valid
/// finite layers, including zero-norm and explicitly empty rows.
///
/// Validation takes the original layer, before [`Rows::of`] can discard
/// a trailing coordinate or treat a non-finite row as having no direction.
pub fn try_drift(vectors: &Vectors<Sentences>, d: &Drift) -> Result<(Vec<i32>, Vec<Cut>), String> {
    Ok(drift(&checked_rows(vectors)?, d))
}

/// Shared integer TextTiling scoring for embedding and lexical cosines.
pub(crate) fn score_cosines(cos: &[Option<i64>], d: &Drift) -> (Vec<i32>, Vec<Cut>) {
    let (_, _, cost, reason) = score_series(cos, d);
    (cost, reason)
}

/// The full scoring series behind [`score_cosines`], per gap: the
/// smoothed cosine (`None` where the gap has no cosine), the score the
/// cost is taken from before the floor (TextTiling depth, or distance
/// under [`Score::Distance`]; 0 where unscored), then the cost and
/// reason exactly as `score_cosines` returns them.
pub(crate) fn score_series(cos: &[Option<i64>], d: &Drift) -> (Vec<Option<i64>>, Vec<i64>, Vec<i32>, Vec<Cut>) {
    let at: Vec<usize> = (0..cos.len()).filter(|&g| cos[g].is_some()).collect();
    let raw: Vec<i64> = at.iter().filter_map(|&g| cos[g]).collect();
    let sm = smooth(&raw, d.smoothing as usize);
    let score = match d.score {
        Score::Depth => depths(&sm),
        Score::Distance => sm.iter().map(|&c| MICRO - c).collect(),
    };
    let mut smoothed = vec![None; cos.len()];
    let mut scored = vec![0i64; cos.len()];
    let mut cost = vec![0i32; cos.len()];
    let mut reason = vec![Cut::Sentence; cos.len()];
    for (k, &g) in at.iter().enumerate() {
        smoothed[g] = Some(sm[k]);
        scored[g] = score[k];
        let s = if score[k] < i64::from(d.floor) { 0 } else { score[k] };
        let c = -(s * i64::from(d.scale) / MICRO);
        cost[g] = c.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        // The bottom of a valley: lower than the point before, not higher
        // than the one after, so a flat bottom marks its first gap.
        let bottom = (k == 0 || sm[k] < sm[k - 1]) && (k + 1 == sm.len() || sm[k] <= sm[k + 1]);
        if bottom && s > 0 && s >= i64::from(d.shift) {
            reason[g] = Cut::TopicShift;
        }
    }
    (smoothed, scored, cost, reason)
}

/// Writes the drift term as the `drift` instance of the boundaries, for
/// `WeighBoundaries` to weigh against the structure; `bound` binds it so.
pub struct DriftBoundaries {
    pub drift: Drift,
}

impl DriftBoundaries {
    /// The step, writing the `drift` instance.
    pub fn bound(self) -> Bound<DriftBoundaries> {
        Bound::new(self).write(0, DRIFT)
    }
}

impl Annotator for DriftBoundaries {
    const ID: &'static str = "turbo:drift-boundaries";
    type Reads = (Sentences, Vectors<Sentences>);
    type Writes = (Boundaries,);

    fn annotate(&self, (sentences, vectors): Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        if vectors.len() != sentences.spans.len() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} vectors for {} sentences", vectors.len(), sentences.spans.len()),
            });
        }
        let (cost, reason) = drift(&Rows::of(vectors), &self.drift);
        Ok((Boundaries { cost, reason },))
    }

    fn config_hash(&self) -> u64 {
        let d = self.drift;
        let score = match d.score {
            Score::Depth => 1,
            Score::Distance => 2,
        };
        [
            u64::from(d.window.get()),
            u64::from(d.smoothing),
            score,
            u64::from(d.floor),
            u64::from(d.scale),
            u64::from(d.shift),
        ]
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit vectors in the plane at these angles, in degrees.
    fn angles(deg: &[f64]) -> Vec<f32> {
        deg.iter().flat_map(|d| [d.to_radians().cos() as f32, d.to_radians().sin() as f32]).collect()
    }

    fn run(data: &[f32], dim: usize, empty: &[usize], d: &Drift) -> (Vec<i32>, Vec<Cut>) {
        drift(&Rows::new(dim, data, |i| empty.contains(&i)), d)
    }

    /// The gap of least cost, the earliest of equals.
    fn cheapest(cost: &[i32]) -> usize {
        (0..cost.len()).min_by_key(|&g| (cost[g], g)).unwrap()
    }

    #[test]
    fn two_clusters_cut_between_them() {
        // Six sentences near 0 degrees, six near 90, with a little jitter.
        let mut deg: Vec<f64> = (0..6).map(|i| f64::from(i % 3) * 4.0).collect();
        deg.extend((0..6).map(|i| 90.0 - f64::from(i % 2) * 5.0));
        let (cost, reason) = run(&angles(&deg), 2, &[], &Drift::default());
        assert_eq!(cheapest(&cost), 5);
        assert!(cost[5] < -Boundaries::UNIT);
        assert_eq!(reason[5], Cut::TopicShift);
        assert_eq!(reason.iter().filter(|&&r| r == Cut::TopicShift).count(), 1);
        // The end of the text has no right side.
        assert_eq!((cost[11], reason[11]), (0, Cut::Sentence));
    }

    #[test]
    fn a_slow_drift_gives_no_signal_by_depth_but_does_by_distance() {
        // Each sentence turns two degrees further.
        let deg: Vec<f64> = (0..30).map(|i| f64::from(i) * 2.0).collect();
        let (cost, reason) = run(&angles(&deg), 2, &[], &Drift::default());
        assert!(cost.iter().all(|&c| c == 0), "{cost:?}");
        assert!(reason.iter().all(|&r| r == Cut::Sentence));
        // Distance lowers every gap alike: no gap stands out, but every
        // cut is cheaper, which favours more chunks.
        let distance = Drift { score: Score::Distance, floor: 0, ..Drift::default() };
        let (cost, _) = run(&angles(&deg), 2, &[], &distance);
        assert!(cost[3..26].iter().all(|&c| c < 0 && c == cost[10]), "{cost:?}");
        // Only gaps with three sentences on each side are scored.
        assert!(cost[..2].iter().chain(&cost[27..]).all(|&c| c == 0), "{cost:?}");
    }

    #[test]
    fn one_outlier_sentence_is_a_shallow_dip_not_two_deep_cuts() {
        // One sentence at 90 degrees among sentences at 0.
        let mut deg = vec![0.0; 15];
        deg[7] = 90.0;
        let (cost, reason) = run(&angles(&deg), 2, &[], &Drift::default());
        // The cheapest gaps cost under half a unit, so with structure's
        // unit per sentence gap, two cuts to set it apart are never free.
        assert!(cost.iter().all(|&c| c > -Boundaries::UNIT / 2), "{cost:?}");
        // Its dip still passes the default `shift`, so the first gap of
        // the flat bottom is named a topic shift; the cost decides.
        let shifts: Vec<usize> = (0..15).filter(|&g| reason[g] == Cut::TopicShift).collect();
        assert_eq!(shifts, [5]);
        let least = *cost.iter().min().unwrap();
        assert!((-450..=-400).contains(&least), "{least}");
        // Comparing single sentences, as a window of one does, makes both
        // sides of the outlier a full-depth valley.
        let single = Drift { window: NonZeroU32::MIN, smoothing: 0, ..Drift::default() };
        let (cost, reason) = run(&angles(&deg), 2, &[], &single);
        assert!(cost[6] <= -Boundaries::UNIT && cost[7] <= -Boundaries::UNIT, "{cost:?}");
        assert_eq!(reason[6], Cut::TopicShift);
    }

    #[test]
    fn the_optimizer_does_not_set_an_outlier_apart_unless_it_is_cheap() {
        use crate::weigh::{Weight, weigh};
        use crate::{Budget, Lengths, segment};
        let mut deg = vec![0.0; 15];
        deg[7] = 90.0;
        let l = Lengths::of(&[100; 15]).unwrap();
        let b = Budget { max: 600, target: 500, size_weight: 1000, overhead: 0 };
        let structure = Boundaries { cost: vec![Boundaries::UNIT; 15], reason: vec![Cut::Sentence; 15] };
        let alone = |d: &Drift| {
            let (cost, reason) = run(&angles(&deg), 2, &[], d);
            let w = weigh(&structure, &Boundaries { cost, reason }, (Weight::ONE, Weight::ONE)).unwrap();
            segment(&w.cost, |i, j| l.run(i, j), b, crate::no_term).iter().any(|s| (s.start, s.end) == (7, 8))
        };
        assert!(!alone(&Drift::default()));
        // Single-sentence windows make both of its gaps cheap enough.
        assert!(alone(&Drift { window: NonZeroU32::MIN, smoothing: 0, ..Drift::default() }));
    }

    #[test]
    fn an_outlier_at_either_end_is_not_set_apart() {
        use crate::weigh::{Weight, weigh};
        use crate::{Budget, Lengths, segment};
        let l = Lengths::of(&[100; 15]).unwrap();
        let b = Budget { max: 600, target: 500, size_weight: 1000, overhead: 0 };
        let structure = Boundaries { cost: vec![Boundaries::UNIT; 15], reason: vec![Cut::Sentence; 15] };
        for odd in [0, 14] {
            let mut deg = vec![0.0; 15];
            deg[odd] = 90.0;
            let (cost, reason) = run(&angles(&deg), 2, &[], &Drift::default());
            // No deeper than a middle outlier's dip. The first or last
            // scored gap, whose window still holds the outlier, has a
            // shallow dip, which passes the default `shift`.
            assert!(cost.iter().all(|&c| c > -Boundaries::UNIT / 2), "{odd}: {cost:?}");
            assert!(cost[..2].iter().chain(&cost[12..]).all(|&c| c == 0), "{odd}: {cost:?}");
            let least = *cost.iter().min().unwrap();
            assert!((-250..=-180).contains(&least), "{odd}: {least}");
            let w = weigh(&structure, &Boundaries { cost, reason }, (Weight::ONE, Weight::ONE)).unwrap();
            let segments = segment(&w.cost, |i, j| l.run(i, j), b, crate::no_term);
            assert!(segments.iter().all(|s| s.end - s.start > 1), "{odd}: {segments:?}");
        }
    }

    #[test]
    fn empty_sentences_are_skipped_not_compared() {
        // Two clusters with an empty sentence between them, one empty at
        // the start and one at the end; their rows are zeros.
        let mut deg = vec![0.0; 14];
        for d in &mut deg[7..] {
            *d = 90.0;
        }
        let mut data = angles(&deg);
        for i in [0, 6, 13] {
            data[2 * i] = 0.0;
            data[2 * i + 1] = 0.0;
        }
        let (cost, reason) = run(&data, 2, &[0, 6, 13], &Drift::default());
        // The gap before sentence 0's neighbour has no left side, so it
        // costs nothing; so does every gap after the last vector.
        assert_eq!((cost[0], cost[12], cost[13]), (0, 0, 0));
        // Around the empty sentence both gaps see the same windows, so
        // they cost the same, and the earlier one is the topic shift.
        assert_eq!(cost[5], cost[6]);
        assert!(cost[5] < -Boundaries::UNIT);
        assert_eq!((reason[5], reason[6]), (Cut::TopicShift, Cut::Sentence));
        // A document with no vectors at all has no drift.
        let (cost, _) = run(&[0.0; 6], 2, &[0, 1, 2], &Drift::default());
        assert_eq!(cost, [0, 0, 0]);
    }

    #[test]
    fn the_same_vectors_give_the_same_costs() {
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let data: Vec<f32> = (0..64 * 40)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect();
        let a = run(&data, 64, &[3, 17], &Drift::default());
        let b = run(&data.clone(), 64, &[3, 17], &Drift::default());
        assert_eq!(a, b);
        assert!(a.0.iter().all(|&c| c <= 0));
    }

    #[test]
    fn depth_is_the_climb_on_both_sides() {
        assert_eq!(depths(&[5, 3, 1, 4, 4, 2, 6]), [0, 2, 7, 0, 0, 6, 0]);
        assert_eq!(depths(&[]), Vec::<i64>::new());
        assert_eq!(smooth(&[1, 2, 4, 8, 16], 1), [1, 2, 5, 9, 16]);
        assert_eq!(smooth(&[1, 2, 4, 8, 16], 2), [1, 2, 6, 9, 16]);
        assert_eq!(smooth(&[-4, -1, -2], 1), [-4, -2, -2]);
        assert_eq!(smooth(&[], 3), Vec::<i64>::new());
    }
}
