//! The bound a score output is held to, rerank scores and extractive QA
//! logits, and the decisions read from those scores (docs/conformance.md,
//! Scores). A cosine over a run of scores does not see a rescaled run, so
//! every score is bounded on its own, and the ranking or answer the
//! reference's scores give is checked to come out the same.

use std::ops::Range;

use turbo::qa::{Answer, DecodeOptions, WindowLogits, decode};
use turbo::{TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_DTYPE_I8};

/// The largest difference allowed between a score and the reference's,
/// as a fraction of the run's scale, the largest reference score in
/// magnitude. A run rescaled by any factor past the bound fails.
pub fn score_bound(compute_dtype: u32) -> f64 {
    match compute_dtype {
        TURBO_DTYPE_F32 => 1e-4,
        TURBO_DTYPE_F16 => 1e-2,
        TURBO_DTYPE_BF16 => 4e-2,
        TURBO_DTYPE_I8 => 0.15,
        _ => panic!("compute dtype {compute_dtype} has no score bound"),
    }
}

/// The scale of a run of reference scores, which must not all be zero.
pub fn scale(want: &[f32]) -> f64 {
    let s = want.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    assert!(s > 0.0, "every reference score is zero");
    s
}

/// Every score of `got` within the bound of its reference in `want`.
/// Returns the largest difference, as a fraction of the scale.
pub fn scores_within(what: &str, got: &[f32], want: &[f32], compute_dtype: u32) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: scores");
    let (s, most) = (scale(want), score_bound(compute_dtype));
    let mut worst = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (f64::from(g) - f64::from(w)).abs() / s;
        assert!(d <= most, "{what}: score {i} is {g}, the reference {w}: {d:e} of the scale {s} is over {most:e}");
        worst = worst.max(d);
    }
    worst
}

/// Every two scores of `got` in the order of the reference's, wherever
/// the reference's two are further apart than `margin`. Returns how many
/// pairs were compared and how many were too close to say.
pub fn same_order(what: &str, got: &[f32], want: &[f32], margin: f64) -> (usize, usize) {
    let (mut compared, mut close) = (0, 0);
    for i in 0..want.len() {
        for j in i + 1..want.len() {
            let gap = f64::from(want[i]) - f64::from(want[j]);
            if gap.abs() <= margin {
                close += 1;
                continue;
            }
            assert_eq!(
                got[i] > got[j],
                gap > 0.0,
                "{what}: scores {i} and {j} are {} and {}, the reference's {} and {}",
                got[i],
                got[j],
                want[i],
                want[j]
            );
            compared += 1;
        }
    }
    (compared, close)
}

/// What a case's logits answer with the default null threshold: the best
/// span, or None, with the margins of that choice over the second span and
/// over the null score.
pub struct Decision {
    pub best: Option<Range<usize>>,
    pub over_second: f64,
    pub over_null: f64,
}

/// The decision `decode` makes on `windows`; every candidate is kept, so
/// the margins are read from the same ranking.
pub fn decision(context: &str, windows: &[WindowLogits]) -> Decision {
    let o = DecodeOptions { null_threshold: -1e9, top_k: 2, non_overlapping: false, ..Default::default() };
    match decode(context, windows, o).unwrap() {
        Answer::None { .. } => Decision { best: None, over_second: f64::INFINITY, over_null: f64::INFINITY },
        Answer::Spans { answers, null_score, .. } => {
            let first = &answers[0];
            let answers_it = first.score - null_score > DecodeOptions::default().null_threshold;
            Decision {
                best: answers_it.then(|| first.bytes.clone()),
                over_second: answers.get(1).map_or(f64::INFINITY, |a| first.score - a.score),
                over_null: (first.score - null_score).abs(),
            }
        }
    }
}

/// The device's decision on a case against the reference's: the same
/// span, or both None, wherever the reference's margins are wider than
/// `margin`. Returns whether the reference decided it.
pub fn same_answer(what: &str, got: &Decision, want: &Decision, margin: f64) -> bool {
    if want.over_null <= margin || (want.best.is_some() && want.over_second <= margin) {
        return false;
    }
    assert_eq!(got.best, want.best, "{what}: the answer differs from the reference's");
    true
}
