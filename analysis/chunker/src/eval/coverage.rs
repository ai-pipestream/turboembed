//! Span coverage: how tightly a set of retrieved chunks wraps the gold
//! evidence spans of one query.
//!
//! Spans are half-open ranges of one document in any unit: UTF-8 bytes,
//! or token indexes (`token_range` turns a byte span into the subwords it
//! touches). With G the union of the gold spans and C the union of the
//! chunks:
//!
//! - iou = |G ∩ C| / |G ∪ C|, 1 when both are empty.
//! - contained = the gold spans that lie wholly inside one chunk, over
//!   all gold spans; 1 with no gold span. A span two chunks share between
//!   them is not contained, even when both are retrieved.
//! - redundancy = the chunks' summed lengths over |C|: 1 when no two
//!   overlap, 2 when every unit is retrieved twice; 1 with no chunk.
//!
//! Empty spans are left out. Overlapping spans count once in the unions.
//! Lengths are summed in u128, so no range of u64 offsets overflows them.

use std::ops::Range;

use turbo_analysis::{Span, Spans};
use turbo_analysis_layers::Subwords;

/// The coverage of one query's gold spans by its chunks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Coverage {
    pub iou: f64,
    pub contained: f64,
    pub redundancy: f64,
}

/// The coverage of `gold` by `chunks`. Sorts copies of both, so time is
/// their count times its log.
pub fn coverage(gold: &[Range<u64>], chunks: &[Range<u64>]) -> Coverage {
    let g = union(gold);
    let c = union(chunks);
    let (lg, lc) = (length(&g), length(&c));
    let inter = intersection(&g, &c);
    let all = lg + lc - inter;
    let iou = if all == 0 { 1.0 } else { inter as f64 / all as f64 };
    let golds: Vec<&Range<u64>> = gold.iter().filter(|r| r.start < r.end).collect();
    let contained = if golds.is_empty() {
        1.0
    } else {
        let mut sorted: Vec<&Range<u64>> = chunks.iter().filter(|r| r.start < r.end).collect();
        sorted.sort_unstable_by_key(|r| (r.start, r.end));
        // reach[i]: the furthest end of the chunks up to i, so a gold span
        // lies inside one of the chunks that start at or before it exactly
        // when the furthest of their ends reaches its end.
        let mut reach = Vec::with_capacity(sorted.len());
        let mut far = 0;
        for r in &sorted {
            far = far.max(r.end);
            reach.push(far);
        }
        let inside = golds
            .iter()
            .filter(|s| {
                let n = sorted.partition_point(|r| r.start <= s.start);
                n > 0 && reach[n - 1] >= s.end
            })
            .count();
        inside as f64 / golds.len() as f64
    };
    let summed: u128 = chunks.iter().map(|r| u128::from(r.end.saturating_sub(r.start))).sum();
    let redundancy = if lc == 0 { 1.0 } else { summed as f64 / lc as f64 };
    Coverage { iou, contained, redundancy }
}

/// The ranges merged into disjoint ones, sorted, empty ones dropped.
fn union(ranges: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut sorted: Vec<Range<u64>> = ranges.iter().filter(|r| r.start < r.end).cloned().collect();
    sorted.sort_unstable_by_key(|r| (r.start, r.end));
    let mut out: Vec<Range<u64>> = Vec::with_capacity(sorted.len());
    for r in sorted {
        match out.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}

fn length(disjoint: &[Range<u64>]) -> u128 {
    disjoint.iter().map(|r| u128::from(r.end - r.start)).sum()
}

/// The overlap of two sorted disjoint lists, in one merge pass.
fn intersection(a: &[Range<u64>], b: &[Range<u64>]) -> u128 {
    let (mut i, mut j, mut sum) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        let (lo, hi) = (a[i].start.max(b[j].start), a[i].end.min(b[j].end));
        sum += u128::from(hi.saturating_sub(lo));
        if a[i].end < b[j].end {
            i += 1;
        } else {
            j += 1;
        }
    }
    sum
}

/// The subwords a byte span touches, as a range of their indexes: the
/// first that ends after it starts up to the last that starts before it
/// ends. Two binary searches; the subwords are in text order.
pub fn token_range(subwords: &Spans<Subwords>, bytes: Span) -> Range<u64> {
    let (starts, ends) = subwords.columns();
    let first = ends.partition_point(|&x| x <= bytes.start) as u64;
    let last = starts.partition_point(|&x| x < bytes.end) as u64;
    first..last.max(first)
}
