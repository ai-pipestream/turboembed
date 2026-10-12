//! The regions a scorer must not look at, as a sorted set of disjoint
//! spans so a query is a binary search.

use turbo_analysis::Span;

/// Non-empty exclusions, sorted, overlapping ones joined.
pub(crate) struct Exclusions {
    spans: Vec<Span>,
}

impl Exclusions {
    /// Builds the set from spans in any order. Empty spans exclude
    /// nothing and are dropped. Spans that only touch are kept apart, so a
    /// query sees exactly the overlaps it would see with the spans given.
    pub(crate) fn new(given: impl IntoIterator<Item = Span>) -> Exclusions {
        let mut spans: Vec<Span> = given.into_iter().filter(|s| !s.is_empty()).collect();
        spans.sort_unstable();
        let mut joined: Vec<Span> = Vec::with_capacity(spans.len());
        for s in spans {
            match joined.last_mut() {
                Some(last) if s.start < last.end => last.end = last.end.max(s.end),
                _ => joined.push(s),
            }
        }
        Exclusions { spans: joined }
    }

    /// Whether the non-empty region `start..end` shares a byte with an
    /// exclusion.
    pub(crate) fn overlaps(&self, start: usize, end: usize) -> bool {
        let i = self.spans.partition_point(|s| s.end as usize <= start);
        self.spans.get(i).is_some_and(|s| (s.start as usize) < end)
    }
}
