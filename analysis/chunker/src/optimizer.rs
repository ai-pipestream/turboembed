//! Optional decision costs from the same DP used by `segment`.
use crate::{Budget, Segment, chunk_cost};

/// Cost of the selected next cut versus the next cheapest distinct next
/// cut. Both include the optimal suffix, in integer boundary-cost units.
/// This is a local DP decision gap, not a certificate about any specific
/// alternate partition. A downstream tie does not make this cut a tie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionMargin {
    pub best_cost: i128,
    /// None when the budget allows only one next cut.
    pub second_best_cost: Option<i128>,
}

impl DecisionMargin {
    /// Best minus runner-up: negative for a strict preference, zero for
    /// a tie, None with no alternative. Arithmetic saturates like the DP.
    pub fn margin(self) -> Option<i128> {
        self.second_best_cost.map(|second| self.best_cost.saturating_sub(second))
    }
}

/// One decision per selected segment, in the same order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentTrace {
    pub segments: Vec<Segment>,
    pub decisions: Vec<DecisionMargin>,
}

/// Like [`crate::segment`], recording the runner-up at each selected cut.
/// Units must be in source order, so earliest unit end means earliest
/// byte offset. The caller supplies deterministic token and term functions.
pub fn segment_with_trace(
    cut: &[i32],
    tokens: impl Fn(usize, usize) -> u32,
    b: Budget,
    term: impl Fn(usize, usize) -> i64,
) -> SegmentTrace {
    solve::<true>(cut, tokens, b, term)
}

pub(crate) fn solve<const TRACE: bool>(
    cut: &[i32],
    tokens: impl Fn(usize, usize) -> u32,
    b: Budget,
    term: impl Fn(usize, usize) -> i64,
) -> SegmentTrace {
    let n = cut.len();
    let size = |i: usize, j: usize| u64::from(tokens(i, j)) + u64::from(b.overhead);
    let cost = |i: usize, j: usize| {
        let boundary = if j < n { cut[j - 1] } else { 0 };
        chunk_cost(boundary, size(i, j), b, term(i, j))
    };
    let mut best = vec![0i128; n + 1];
    let mut next = vec![n; n + 1];
    let mut second = if TRACE { vec![None; n] } else { Vec::new() };
    for i in (0..n).rev() {
        for j in i + 1..=n {
            if j - i > 1 && size(i, j) > u64::from(b.max) {
                break;
            }
            let c = cost(i, j).saturating_add(best[j]);
            if j == i + 1 {
                best[i] = c;
                next[i] = j;
            } else if c < best[i] {
                if TRACE {
                    second[i] = Some(best[i]);
                }
                best[i] = c;
                next[i] = j;
            } else if TRACE && second[i].is_none_or(|s| c < s) {
                second[i] = Some(c);
            }
        }
    }
    let mut out = SegmentTrace { segments: Vec::new(), decisions: Vec::new() };
    let mut i = 0;
    while i < n {
        let j = next[i];
        out.segments.push(Segment { start: i, end: j, tokens: size(i, j), cost: cost(i, j) });
        if TRACE {
            out.decisions.push(DecisionMargin { best_cost: best[i], second_best_cost: second[i] });
        }
        i = j;
    }
    out
}
