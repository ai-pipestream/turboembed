//! Priority-ordered composite glossary matcher.

use std::sync::Arc;

use turbo_analysis::Span;

use crate::selection::spans_intersect;
use crate::{GlossaryError, GlossaryMatch, GlossaryMatcher};

/// A [`GlossaryMatcher`] that chains multiple [`GlossaryMatcher`] delegates in
/// priority order, suppressing any match from a lower-priority matcher whose
/// span intersects an already accepted match from an earlier matcher (or an
/// earlier match from the same delegate).
pub struct CompositeGlossaryMatcher {
    matchers: Vec<Arc<dyn GlossaryMatcher>>,
}

impl std::fmt::Debug for CompositeGlossaryMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeGlossaryMatcher").field("matchers_len", &self.matchers.len()).finish()
    }
}

impl CompositeGlossaryMatcher {
    /// Creates a composite matcher from `matchers` in decreasing priority
    /// order.
    ///
    /// # Errors
    ///
    /// Returns [`GlossaryError::EmptyMatchers`] if `matchers` is empty.
    pub fn new(matchers: Vec<Arc<dyn GlossaryMatcher>>) -> Result<Self, GlossaryError> {
        if matchers.is_empty() {
            return Err(GlossaryError::EmptyMatchers);
        }
        Ok(Self { matchers })
    }
}

impl GlossaryMatcher for CompositeGlossaryMatcher {
    fn match_text(&self, text: &str) -> Vec<GlossaryMatch> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut accepted: Vec<GlossaryMatch> = Vec::new();
        for matcher in &self.matchers {
            let candidates = matcher.match_text(text);
            if candidates.is_empty() {
                continue;
            }
            let mut additions = Vec::new();
            for candidate in candidates {
                let c_span = candidate.span();
                if !overlaps_sorted(&accepted, c_span) && !overlaps_linear(&additions, c_span) {
                    additions.push(candidate);
                }
            }
            if !additions.is_empty() {
                additions
                    .sort_by(|a, b| a.span().start.cmp(&b.span().start).then_with(|| b.span().end.cmp(&a.span().end)));
                accepted = merge_by_start(accepted, additions);
            }
        }
        accepted
    }
}

/// Checks whether `candidate` intersects any span in `accepted`, which is
/// disjoint and sorted by ascending start offset.
fn overlaps_sorted(accepted: &[GlossaryMatch], candidate: Span) -> bool {
    if accepted.is_empty() {
        return false;
    }
    let idx = accepted.partition_point(|m| m.span().start <= candidate.start);
    if idx > 0 && spans_intersect(accepted[idx - 1].span(), candidate) {
        return true;
    }
    for m in &accepted[idx..] {
        let s = m.span();
        if s.start > candidate.end {
            break;
        }
        if spans_intersect(s, candidate) {
            return true;
        }
    }
    false
}

/// Linearly checks whether `candidate` intersects any match in `matches`.
fn overlaps_linear(matches: &[GlossaryMatch], candidate: Span) -> bool {
    matches.iter().any(|m| spans_intersect(m.span(), candidate))
}

/// Merges two lists already sorted by ascending start offset, placing entries
/// from `accepted` before entries from `additions` when start offsets tie.
fn merge_by_start(accepted: Vec<GlossaryMatch>, additions: Vec<GlossaryMatch>) -> Vec<GlossaryMatch> {
    if accepted.is_empty() {
        return additions;
    }
    let mut merged = Vec::with_capacity(accepted.len() + additions.len());
    let mut a = accepted.into_iter().peekable();
    let mut b = additions.into_iter().peekable();
    loop {
        match (a.peek(), b.peek()) {
            (Some(x), Some(y)) => {
                if x.span().start <= y.span().start {
                    merged.push(a.next().unwrap());
                } else {
                    merged.push(b.next().unwrap());
                }
            }
            (Some(_), None) => {
                merged.extend(a);
                break;
            }
            (None, Some(_)) => {
                merged.extend(b);
                break;
            }
            (None, None) => break,
        }
    }
    merged
}
