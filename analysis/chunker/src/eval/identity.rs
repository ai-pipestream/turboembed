//! Comparing flat leaf partitions from two runs of the same document.
use super::invalid;
use turbo_analysis::{Error, Overlap, Spans};
use turbo_analysis_layers::Chunks;

/// A closing cut, including the unassigned gap before the next chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutRecord {
    pub end: u32,
    pub next_start: u32,
    /// Best minus runner-up total cost, in integer boundary-cost units.
    /// None means forced or unavailable, not a zero-cost tie.
    pub margin: Option<i128>,
}

impl CutRecord {
    fn position(self) -> (u32, u32) {
        (self.end, self.next_start)
    }
}

/// Unmatched cuts paired in source order between consecutive shared cuts.
/// An insertion/deletion has only one side. No margin is invented for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutDifference {
    pub a: Option<CutRecord>,
    pub b: Option<CutRecord>,
    /// Both sides exist and have a DP gap whose magnitude is <= epsilon.
    /// This flags a precision-sensitive decision; it does not prove that
    /// the particular alternate partition is within epsilon of the best.
    pub within_epsilon: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CutIdentityReport {
    pub total_chunks: [usize; 2],
    pub matching_cuts: usize,
    /// Number of distinct internal cut positions in the union.
    pub total_cuts: usize,
    /// Intersection / union, or 1 for two partitions with no internal cuts.
    pub agreement_rate: f64,
    pub first_differing_offset: Option<u32>,
    pub epsilon: f64,
    pub differences: Vec<CutDifference>,
}

impl CutIdentityReport {
    pub fn is_identical(&self) -> bool {
        self.differences.is_empty()
    }
    /// Missing margins, insertions and deletions cannot pass this check.
    pub fn all_within_epsilon(&self) -> bool {
        self.differences.iter().all(|d| d.within_epsilon)
    }
}

/// Compare a single document's flat, ordered, non-overlapping leaf spans
/// in the same view and with the same outer extent. Call once per corpus
/// document: equal ViewIds alone do not establish equal document content.
///
/// Margins align with chunks and price each closing cut. The final margin
/// is unused. Epsilon must be finite and nonnegative, in the same units as
/// the DP costs. Integer margins are compared exactly to floor(epsilon),
/// avoiding conversion of large i128 costs to an imprecise float.
/// Different gaps between chunks are differences even if their starts agree.
pub fn compare_chunk_cuts(
    a: &Spans<Chunks>,
    margins_a: &[Option<i128>],
    b: &Spans<Chunks>,
    margins_b: &[Option<i128>],
    epsilon: f64,
) -> Result<CutIdentityReport, Error> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(invalid("epsilon must be finite and nonnegative".into()));
    }
    if a.view() != b.view() {
        return Err(invalid("cut runs use different views".into()));
    }
    let ca = records(a, margins_a)?;
    let cb = records(b, margins_b)?;
    let extent = |s: &Spans<Chunks>| {
        let (starts, ends) = s.columns();
        starts.first().zip(ends.last()).map(|(&start, &end)| (start, end))
    };
    if extent(a) != extent(b) {
        return Err(invalid("cut runs have different outer extents".into()));
    }
    let mut report = CutIdentityReport {
        total_chunks: [a.len(), b.len()],
        matching_cuts: 0,
        total_cuts: 0,
        agreement_rate: 1.0,
        first_differing_offset: None,
        epsilon,
        differences: Vec::new(),
    };
    let (mut i, mut j, mut ai, mut bi) = (0, 0, 0, 0);
    while i < ca.len() && j < cb.len() {
        match ca[i].position().cmp(&cb[j].position()) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                append(&ca[ai..i], &cb[bi..j], &mut report);
                report.matching_cuts += 1;
                i += 1;
                j += 1;
                ai = i;
                bi = j;
            }
        }
    }
    append(&ca[ai..], &cb[bi..], &mut report);
    report.total_cuts = ca.len() + cb.len() - report.matching_cuts;
    if report.total_cuts != 0 {
        report.agreement_rate = report.matching_cuts as f64 / report.total_cuts as f64;
    }
    Ok(report)
}

fn records(spans: &Spans<Chunks>, margins: &[Option<i128>]) -> Result<Vec<CutRecord>, Error> {
    spans.check(Overlap::None).map_err(invalid)?;
    if spans.len() != margins.len() {
        return Err(invalid("one margin entry is required per chunk".into()));
    }
    if margins.iter().flatten().any(|&m| m > 0) {
        return Err(invalid("best-minus-runner-up margins must be <= 0".into()));
    }
    if spans.has_discontinuous() {
        return Err(invalid("cut comparison requires continuous spans".into()));
    }
    let (starts, ends) = spans.columns();
    if starts.iter().zip(ends).any(|(s, e)| s >= e) {
        return Err(invalid("chunk spans must be nonempty".into()));
    }
    Ok((0..spans.len().saturating_sub(1))
        .map(|i| CutRecord { end: ends[i], next_start: starts[i + 1], margin: margins[i] })
        .collect())
}

fn append(a: &[CutRecord], b: &[CutRecord], report: &mut CutIdentityReport) {
    let close = |c: CutRecord| c.margin.is_some_and(|m| m.unsigned_abs() <= report.epsilon.floor() as u128);
    for i in 0..a.len().max(b.len()) {
        let (a, b) = (a.get(i).copied(), b.get(i).copied());
        let at = match (a, b) {
            (Some(x), Some(y)) if x.end == y.end => x.next_start.min(y.next_start),
            _ => a.into_iter().chain(b).map(|c| c.end).min().unwrap(),
        };
        report.first_differing_offset = Some(report.first_differing_offset.map_or(at, |first| first.min(at)));
        report.differences.push(CutDifference { a, b, within_epsilon: a.is_some_and(close) && b.is_some_and(close) });
    }
}
