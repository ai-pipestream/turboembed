//! Non-overlapping match selection using OpenNLP's `Span.intersects` rule.

use turbo_analysis::Span;

/// Whether `a` contains `b` under OpenNLP's `Span.contains(Span)` rule:
/// `a.start <= b.start && b.end <= a.end`.
#[inline]
pub fn span_contains(a: Span, b: Span) -> bool {
    a.start <= b.start && b.end <= a.end
}

/// Whether `a` and `b` intersect under OpenNLP's `Span.intersects(Span)` rule:
/// either one span contains the other (including a zero-width span sitting on
/// either boundary of another span) or one span's start lies strictly inside
/// the other span. Adjacent non-empty spans (`[0, 3)` and `[3, 5)`) do not
/// intersect.
#[inline]
pub fn spans_intersect(a: Span, b: Span) -> bool {
    span_contains(a, b)
        || span_contains(b, a)
        || (a.start <= b.start && b.start < a.end)
        || (b.start <= a.start && a.start < b.end)
}

/// Selects a non-overlapping subset from `sorted_candidates`, which must
/// already be ordered by ascending start offset with higher-priority
/// candidates at the same start offset first.
///
/// Because `sorted_candidates` is ordered by start offset, a candidate only
/// needs to be checked against the last accepted span:
/// - if `last.end > candidate.start`, they overlap;
/// - if `last.end == candidate.start`, they intersect if and only if at least
///   one of them is zero-width (`last.is_empty() || candidate.is_empty()`),
///   which `spans_intersect(last, candidate)` checks directly.
pub fn select<T>(sorted_candidates: Vec<T>, mut span_of: impl FnMut(&T) -> Span) -> Vec<T> {
    if sorted_candidates.len() <= 1 {
        return sorted_candidates;
    }
    let mut accepted = Vec::with_capacity(sorted_candidates.len());
    let mut last_accepted: Option<Span> = None;
    for candidate in sorted_candidates {
        let span = span_of(&candidate);
        if last_accepted.is_none_or(|last| !spans_intersect(last, span)) {
            last_accepted = Some(span);
            accepted.push(candidate);
        }
    }
    accepted
}
