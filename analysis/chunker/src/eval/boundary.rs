//! Boundary measures: how a hypothesis segmentation's cuts compare with a
//! reference's over the same units.
//!
//! - Precision, recall and F1 at a tolerance t: a hypothesis boundary
//!   matches a reference boundary at most t units away, each boundary in
//!   at most one match, and the matching is a largest one. Exact is t = 0.
//!   A side with no boundaries scores 1 on its own measure (no cut is
//!   wrong, no cut is missed), and F1 is 0 when precision and recall are.
//! - Pk (Beeferman, Berger and Lafferty, 1999): the share of probe pairs
//!   (i, i + k), for i in `0..units - k`, that one segmentation puts in
//!   the same segment and the other does not.
//! - WindowDiff (Pevzner and Hearst, 2002): the share of windows `(i, i +
//!   k]`, for the same i, in which the two segmentations hold different
//!   numbers of boundaries. It counts a near miss and a false cut alike.
//!
//! k is half the mean reference segment length, rounded down, and at least
//! one: `units / (2 * segments)`. Both are in [0, 1], and 0 when there is
//! no probe (fewer than k + 1 units).

use turbo_analysis::Error;

use super::{Segmentation, invalid};

/// Precision, recall and F1 at one tolerance, with the counts behind them.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Prf {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    /// Hypothesis boundaries matched to a reference boundary.
    pub matched: usize,
    pub hypothesis: usize,
    pub reference: usize,
}

/// Every boundary measure of one hypothesis against one reference.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoundaryScores {
    pub exact: Prf,
    pub within_1: Prf,
    pub within_2: Prf,
    /// The probe distance Pk and WindowDiff used.
    pub k: usize,
    pub pk: f64,
    pub window_diff: f64,
}

/// The tolerances `BoundaryScores` reports, in units.
pub const TOLERANCES: [usize; 3] = [0, 1, 2];

/// Scores `hypothesis` against `reference`; an error when they segment
/// different numbers of units.
pub fn score(reference: &Segmentation, hypothesis: &Segmentation) -> Result<BoundaryScores, Error> {
    same_units(reference, hypothesis)?;
    let k = probe_distance(reference);
    let (pk, window_diff) = windows(reference, hypothesis, k);
    let [exact, within_1, within_2] = TOLERANCES.map(|t| prf(reference.boundaries(), hypothesis.boundaries(), t));
    Ok(BoundaryScores { exact, within_1, within_2, k, pk, window_diff })
}

fn same_units(a: &Segmentation, b: &Segmentation) -> Result<(), Error> {
    if a.units() != b.units() {
        return Err(invalid(format!("a reference of {} units and a hypothesis of {}", a.units(), b.units())));
    }
    Ok(())
}

/// Half the mean reference segment length, rounded down, at least one.
pub fn probe_distance(reference: &Segmentation) -> usize {
    (reference.units() / (2 * reference.segments().max(1))).max(1)
}

/// Pk at probe distance k; an error when the units differ.
pub fn pk(reference: &Segmentation, hypothesis: &Segmentation, k: usize) -> Result<f64, Error> {
    same_units(reference, hypothesis)?;
    Ok(windows(reference, hypothesis, k.max(1)).0)
}

/// WindowDiff at window k; an error when the units differ.
pub fn window_diff(reference: &Segmentation, hypothesis: &Segmentation, k: usize) -> Result<f64, Error> {
    same_units(reference, hypothesis)?;
    Ok(windows(reference, hypothesis, k.max(1)).1)
}

/// Pk and WindowDiff together, over one pass of the probes. k >= 1.
fn windows(reference: &Segmentation, hypothesis: &Segmentation, k: usize) -> (f64, f64) {
    let n = reference.units();
    if n <= k {
        return (0.0, 0.0);
    }
    let (r, h) = (reference.counts(), hypothesis.counts());
    let (mut pk, mut wd) = (0usize, 0usize);
    for i in 0..n - k {
        let (a, b) = (r[i + k] - r[i], h[i + k] - h[i]);
        pk += usize::from((a == 0) != (b == 0));
        wd += usize::from(a != b);
    }
    let probes = (n - k) as f64;
    (pk as f64 / probes, wd as f64 / probes)
}

/// Precision, recall and F1 of `hypothesis` boundaries against
/// `reference` ones, both sorted, at tolerance `t`.
///
/// The matching takes the hypothesis boundaries in order and gives each
/// the earliest unmatched reference boundary within t. On a line, with
/// both sides sorted, that greedy matching is a largest one: a later
/// hypothesis boundary can reach every reference boundary an earlier one
/// could, past the earliest. Linear time.
pub fn prf(reference: &[usize], hypothesis: &[usize], t: usize) -> Prf {
    let (mut matched, mut r) = (0, 0);
    for &h in hypothesis {
        while r < reference.len() && reference[r] + t < h {
            r += 1;
        }
        if r < reference.len() && reference[r] <= h + t {
            matched += 1;
            r += 1;
        }
    }
    let share = |of: usize| if of == 0 { 1.0 } else { matched as f64 / of as f64 };
    let (precision, recall) = (share(hypothesis.len()), share(reference.len()));
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    Prf { precision, recall, f1, matched, hypothesis: hypothesis.len(), reference: reference.len() }
}
