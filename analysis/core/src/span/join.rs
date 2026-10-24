//! Sweep-line joins of two span layers in one view: for every outer
//! item, the inner items it contains or overlaps, in one forward pass
//! over both layers and no allocation, where the per-span helpers
//! ([`Spans::within`], [`Spans::overlapping_range`]) would pay a
//! binary search per outer item. The pass is $O(M + N)$ while outer
//! ends do not decrease (the usual case, both layers ordered); a
//! nested outer span whose end drops re-seeks its far bound by
//! binary search, so the worst case is $O(M \log N + N)$, never a
//! rescan.
//!
//! Both joins read bounding envelopes, like those helpers, and carry
//! the same shape of precondition: the outer layer must be in start
//! order and the inner layer must be ordered without overlaps
//! ([`Spans::is_ordered`]). The inner condition is what makes each
//! answer one contiguous [`IxRange`]: with nested inner spans the
//! contained or overlapping set of an outer span need not be
//! contiguous (an inner span reaching past the outer end can sit
//! between two that fit), and such layers keep using the scanning
//! helpers ([`Spans::in_range`], [`Spans::overlapping`]). Outer spans
//! may nest freely: only their starts must not decrease, which keeps
//! the inner cursor moving forward. The preconditions are
//! debug-asserted, not checked; on layers that break them the ranges
//! are unspecified, never a panic beyond the asserts.

use super::Span;
use super::index::{Ix, IxRange};
use super::spans::Spans;

/// One outer item and the inner items it joins with.
type Joined<A, B> = (Ix<A>, Span, IxRange<B>);

fn span(start: &[u32], end: &[u32], i: usize) -> Span {
    Span { start: start[i], end: end[i] }
}

/// The sweep shared by both joins: `lead` is the first inner item the
/// current outer item can still join with, `from` computes that lead
/// for an outer span from the previous one, and `to` the first inner
/// item past the join. Both only move forward as outer starts rise.
struct Sweep<'a> {
    inner_start: &'a [u32],
    inner_end: &'a [u32],
    lead: usize,
}

impl Sweep<'_> {
    /// The first inner index at or after `from` whose start is at
    /// least `x`.
    fn start_at_least(&self, mut from: usize, x: u32) -> usize {
        while from < self.inner_start.len() && self.inner_start[from] < x {
            from += 1;
        }
        from
    }

    /// The first inner index at or after `from` whose end is past
    /// `x` (strictly greater).
    fn end_past(&self, mut from: usize, x: u32) -> usize {
        while from < self.inner_end.len() && self.inner_end[from] <= x {
            from += 1;
        }
        from
    }

    /// The end bound for an outer end `x`, continuing from the last
    /// call's answer when `x` has not decreased (the sweep) and
    /// binary-searching from the lead when it has (a nested outer
    /// span), so nested outers stay correct without rescanning.
    fn end_bound(&self, hint: &mut (u32, usize), x: u32) -> usize {
        let hi = if x >= hint.0 {
            self.end_past(hint.1.max(self.lead), x)
        } else {
            self.lead + self.inner_end[self.lead..].partition_point(|&e| e <= x)
        };
        *hint = (x, hi);
        hi
    }

    /// The start bound for an outer end `x`, with the same
    /// continue-or-search rule as [`Sweep::end_bound`].
    fn start_bound(&self, hint: &mut (u32, usize), x: u32) -> usize {
        let hi = if x >= hint.0 {
            self.start_at_least(hint.1.max(self.lead), x)
        } else {
            self.lead + self.inner_start[self.lead..].partition_point(|&s| s < x)
        };
        *hint = (x, hi);
        hi
    }
}

/// For every item of the outer layer, the inner items it contains
/// (`outer.start <= inner.start` and `inner.end <= outer.end`, the
/// rule of [`Spans::within`]), as one [`IxRange`]. Created by
/// [`Spans::join_contained`].
pub struct ContainedJoin<'a, A, B> {
    outer_start: &'a [u32],
    outer_end: &'a [u32],
    sweep: Sweep<'a>,
    at: usize,
    hi_hint: (u32, usize),
    _layers: std::marker::PhantomData<fn() -> (A, B)>,
}

impl<A, B> Iterator for ContainedJoin<'_, A, B> {
    type Item = Joined<A, B>;

    fn next(&mut self) -> Option<Joined<A, B>> {
        if self.at >= self.outer_start.len() {
            return None;
        }
        let outer = span(self.outer_start, self.outer_end, self.at);
        let ix = Ix::new(self.at as u32);
        self.at += 1;
        self.sweep.lead = self.sweep.start_at_least(self.sweep.lead, outer.start);
        // Inner ends rise with inner starts on an ordered layer, so
        // the first end past the outer end closes the range; every
        // item before it starts at or after the outer start (the
        // lead) and ends by the outer end, so it is contained, and
        // an empty inner item on either boundary counts, as in
        // `within`.
        let hi = self.sweep.end_bound(&mut self.hi_hint, outer.end);
        Some((ix, outer, IxRange::new(self.sweep.lead as u32, hi as u32)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.outer_start.len() - self.at;
        (rem, Some(rem))
    }
}

impl<A, B> ExactSizeIterator for ContainedJoin<'_, A, B> {}

/// For every item of the outer layer, the inner items overlapping it
/// (the rule of [`Spans::overlapping_range`], empty-span cases
/// included: an empty outer span is a point query), as one
/// [`IxRange`]. Created by [`Spans::join_overlapping`].
pub struct OverlappingJoin<'a, A, B> {
    outer_start: &'a [u32],
    outer_end: &'a [u32],
    sweep: Sweep<'a>,
    at: usize,
    hi_hint: (u32, usize),
    _layers: std::marker::PhantomData<fn() -> (A, B)>,
}

impl<A, B> Iterator for OverlappingJoin<'_, A, B> {
    type Item = Joined<A, B>;

    fn next(&mut self) -> Option<Joined<A, B>> {
        if self.at >= self.outer_start.len() {
            return None;
        }
        let outer = span(self.outer_start, self.outer_end, self.at);
        let ix = Ix::new(self.at as u32);
        self.at += 1;
        // The first overlap is the earlier of the first item starting
        // at or after the outer start and the first item ending after
        // it, exactly as `overlapping_range` bounds it; both bounds
        // only rise as outer starts rise, so each can sweep.
        let by_start = self.sweep.start_at_least(self.sweep.lead, outer.start);
        let by_end = self.sweep.end_past(self.sweep.lead, outer.start);
        let lo = by_start.min(by_end);
        self.sweep.lead = lo;
        let hi = if outer.is_empty() {
            // A point query also overlaps the item covering the point
            // and empty items at it: everything from `lo` that starts
            // at or before the point.
            lo + self.sweep.inner_start[lo..].partition_point(|&s| s <= outer.start)
        } else {
            self.sweep.start_bound(&mut self.hi_hint, outer.end).max(lo)
        };
        Some((ix, outer, IxRange::new(lo as u32, hi as u32)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.outer_start.len() - self.at;
        (rem, Some(rem))
    }
}

impl<A, B> ExactSizeIterator for OverlappingJoin<'_, A, B> {}

impl<A> Spans<A> {
    /// Joins every item of this layer against the items of `inner`
    /// it contains, sweeping both layers once: $O(M + N)$ and no
    /// allocation, against a binary search per item for
    /// [`Spans::within`]. This layer must be in start order and
    /// `inner` must be ordered without overlaps (debug-asserted);
    /// both must be spans of one view. Discontinuous items join by
    /// their bounding envelopes. Every outer item is yielded, with
    /// an empty range when it contains nothing.
    pub fn join_contained<'a, B>(&'a self, inner: &'a Spans<B>) -> ContainedJoin<'a, A, B> {
        debug_assert!(self.is_start_ordered(), "join_contained: outer spans in start order");
        debug_assert!(inner.is_ordered(), "join_contained: inner spans ordered without overlaps");
        debug_assert_eq!(self.view(), inner.view(), "join_contained: one view");
        let (outer_start, outer_end) = self.columns();
        let (inner_start, inner_end) = inner.columns();
        ContainedJoin {
            outer_start,
            outer_end,
            sweep: Sweep { inner_start, inner_end, lead: 0 },
            at: 0,
            hi_hint: (0, 0),
            _layers: std::marker::PhantomData,
        }
    }

    /// Joins every item of this layer against the items of `inner`
    /// overlapping it, sweeping both layers once: $O(M + N)$ and no
    /// allocation, against a binary search per item for
    /// [`Spans::overlapping_range`]. The same ordering conditions as
    /// [`Spans::join_contained`] apply (debug-asserted), both layers
    /// must be spans of one view, and discontinuous items join by
    /// their bounding envelopes. Every outer item is yielded, with
    /// an empty range when it overlaps nothing.
    pub fn join_overlapping<'a, B>(&'a self, inner: &'a Spans<B>) -> OverlappingJoin<'a, A, B> {
        debug_assert!(self.is_start_ordered(), "join_overlapping: outer spans in start order");
        debug_assert!(inner.is_ordered(), "join_overlapping: inner spans ordered without overlaps");
        debug_assert_eq!(self.view(), inner.view(), "join_overlapping: one view");
        let (outer_start, outer_end) = self.columns();
        let (inner_start, inner_end) = inner.columns();
        OverlappingJoin {
            outer_start,
            outer_end,
            sweep: Sweep { inner_start, inner_end, lead: 0 },
            at: 0,
            hi_hint: (0, 0),
            _layers: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::Spans;
    use crate::text::ViewId;

    struct A;
    struct B;

    fn layer<L>(items: &[(u32, u32)]) -> Spans<L> {
        let mut s = Spans::new(ViewId::SOURCE);
        for &(a, b) in items {
            s.push(Span::new(a, b));
        }
        s
    }

    fn contained(outer: &Spans<A>, inner: &Spans<B>) -> Vec<Vec<u32>> {
        outer.join_contained(inner).map(|(_, _, r)| r.iter().map(|i| i.get() as u32).collect()).collect()
    }

    fn overlapping(outer: &Spans<A>, inner: &Spans<B>) -> Vec<Vec<u32>> {
        outer.join_overlapping(inner).map(|(_, _, r)| r.iter().map(|i| i.get() as u32).collect()).collect()
    }

    /// Every join answer against the per-span binary-search helpers
    /// and against brute force, on fixed generated layers covering
    /// empty spans, shared boundaries, gaps, and nested outers.
    #[test]
    fn joins_match_the_point_helpers() {
        let inner = layer::<B>(&[(0, 0), (0, 3), (3, 3), (4, 7), (7, 9), (9, 9), (12, 15), (15, 20)]);
        let outer =
            layer::<A>(&[(0, 0), (0, 20), (3, 3), (3, 9), (4, 4), (5, 6), (9, 12), (12, 12), (20, 20), (21, 25)]);
        let joined_c = contained(&outer, &inner);
        let joined_o = overlapping(&outer, &inner);
        for (i, o) in outer.iter() {
            let want_c: Vec<u32> = inner.within(o).iter().map(|x| x.get() as u32).collect();
            let want_o: Vec<u32> = inner.overlapping_range(o).iter().map(|x| x.get() as u32).collect();
            assert_eq!(joined_c[i.get()], want_c, "contained {o:?}");
            assert_eq!(joined_o[i.get()], want_o, "overlapping {o:?}");
            let brute_c: Vec<u32> = inner
                .iter()
                .filter(|(_, s)| o.start <= s.start && s.end <= o.end)
                .map(|(x, _)| x.get() as u32)
                .collect();
            let brute_o: Vec<u32> = inner.iter().filter(|(_, s)| s.overlaps(o)).map(|(x, _)| x.get() as u32).collect();
            assert_eq!(joined_c[i.get()], brute_c, "contained brute {o:?}");
            assert_eq!(joined_o[i.get()], brute_o, "overlapping brute {o:?}");
        }
    }

    #[test]
    fn joins_handle_empty_layers() {
        let empty_a = layer::<A>(&[]);
        let empty_b = layer::<B>(&[]);
        let one = layer::<A>(&[(1, 2)]);
        assert_eq!(contained(&empty_a, &empty_b), Vec::<Vec<u32>>::new());
        assert_eq!(overlapping(&empty_a, &empty_b), Vec::<Vec<u32>>::new());
        assert_eq!(contained(&one, &empty_b), vec![Vec::<u32>::new()]);
        assert_eq!(overlapping(&one, &empty_b), vec![Vec::<u32>::new()]);
    }

    /// Nested outer spans (ends decreasing after a wide span) still
    /// answer each span on its own bounds.
    #[test]
    fn joins_handle_nested_outers() {
        let inner = layer::<B>(&[(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]);
        let outer = layer::<A>(&[(0, 10), (0, 3), (2, 5), (9, 10), (10, 10)]);
        assert_eq!(contained(&outer, &inner), vec![vec![0, 1, 2, 3, 4], vec![0], vec![1], vec![], vec![]]);
        assert_eq!(overlapping(&outer, &inner), vec![vec![0, 1, 2, 3, 4], vec![0, 1], vec![1, 2], vec![4], vec![]]);
    }

    /// Generated ordered layers, compared against brute force only,
    /// shaped by a fixed-seed generator inside the test.
    #[test]
    fn joins_match_brute_force_on_generated_layers() {
        let mut state = 0x243f6a8885a308d3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..40 {
            let make = |next: &mut dyn FnMut() -> u64| {
                let mut items = Vec::new();
                let mut at = 0u32;
                for _ in 0..(next() % 24) {
                    at += (next() % 3) as u32;
                    let len = (next() % 5) as u32;
                    items.push((at, at + len));
                    at += len;
                }
                items
            };
            let inner_items = make(&mut next);
            let inner = layer::<B>(&inner_items);
            // Outer spans need only start order; generate possibly
            // nesting spans over the same span of text.
            let mut outer_items = Vec::new();
            let mut at = 0u32;
            for _ in 0..(next() % 24) {
                at += (next() % 4) as u32;
                outer_items.push((at, at + (next() % 9) as u32));
            }
            let outer = layer::<A>(&outer_items);
            for (i, ((_, o), got)) in outer.iter().zip(contained(&outer, &inner)).enumerate() {
                let want: Vec<u32> = inner
                    .iter()
                    .filter(|(_, s)| o.start <= s.start && s.end <= o.end)
                    .map(|(x, _)| x.get() as u32)
                    .collect();
                assert_eq!(got, want, "contained outer {i} {o:?}");
            }
            for (i, ((_, o), got)) in outer.iter().zip(overlapping(&outer, &inner)).enumerate() {
                let want: Vec<u32> = inner.iter().filter(|(_, s)| s.overlaps(o)).map(|(x, _)| x.get() as u32).collect();
                assert_eq!(got, want, "overlapping outer {i} {o:?}");
            }
        }
    }
}
