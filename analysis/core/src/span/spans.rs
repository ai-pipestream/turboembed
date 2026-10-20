use std::fmt;
use std::marker::PhantomData;
use std::ops::Range;

use crate::text::ViewId;

use super::index::{Ix, IxRange};
use super::text::TextSpans;
use super::{FilterRule, FragmentError, Overlap, Span};

/// Iterator over the ordered fragments of one item in [`Spans<L>`].
#[derive(Clone, Debug)]
pub struct Fragments<'a> {
    start: &'a [u32],
    end: &'a [u32],
    pos: usize,
}

impl Iterator for Fragments<'_> {
    type Item = Span;

    fn next(&mut self) -> Option<Span> {
        if self.pos >= self.start.len() {
            return None;
        }
        let s = Span { start: self.start[self.pos], end: self.end[self.pos] };
        self.pos += 1;
        Some(s)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.start.len() - self.pos;
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for Fragments<'_> {}

/// The spans of a positional layer, stored by column: one `Vec<u32>` of
/// starts and one of ends, never an object per item. `L` is the layer the
/// spans belong to, so they are indexed by `Ix<L>` alone.
///
/// When every item is continuous, `frag_start`, `frag_end` and
/// `frag_item_end` stay empty and allocate nothing. When any item is
/// discontinuous (`push_fragments` with two or more fragments), `start`
/// and `end` hold each item's bounding envelope (`first.start..last.end`)
/// while `frag_start`, `frag_end` and `frag_item_end` store the ordered
/// fragments flat by column.
pub struct Spans<L> {
    view: ViewId,
    start: Vec<u32>,
    end: Vec<u32>,
    frag_start: Vec<u32>,
    frag_end: Vec<u32>,
    frag_item_end: Vec<u32>,
    start_ordered: bool,
    _l: PhantomData<fn() -> L>,
}

impl<L> Spans<L> {
    pub fn new(view: ViewId) -> Spans<L> {
        Spans {
            view,
            start: Vec::new(),
            end: Vec::new(),
            frag_start: Vec::new(),
            frag_end: Vec::new(),
            frag_item_end: Vec::new(),
            start_ordered: true,
            _l: PhantomData,
        }
    }

    pub fn with_capacity(view: ViewId, n: usize) -> Spans<L> {
        Spans {
            view,
            start: Vec::with_capacity(n),
            end: Vec::with_capacity(n),
            frag_start: Vec::new(),
            frag_end: Vec::new(),
            frag_item_end: Vec::new(),
            start_ordered: true,
            _l: PhantomData,
        }
    }

    /// The view the offsets are in.
    pub fn view(&self) -> ViewId {
        self.view
    }

    /// A borrowed view of the span columns and fragment columns, for
    /// [`crate::Layer::text_spans`].
    pub fn text_spans(&self) -> TextSpans<'_> {
        TextSpans {
            view: self.view,
            start: &self.start,
            end: &self.end,
            frag_start: &self.frag_start,
            frag_end: &self.frag_end,
            frag_item_end: &self.frag_item_end,
        }
    }

    pub fn push(&mut self, s: Span) -> Ix<L> {
        if let Some(&prev) = self.start.last()
            && s.start < prev
        {
            self.start_ordered = false;
        }
        let i = self.start.len() as u32;
        self.start.push(s.start);
        self.end.push(s.end);
        if !self.frag_item_end.is_empty() {
            self.frag_start.push(s.start);
            self.frag_end.push(s.end);
            self.frag_item_end.push(self.frag_start.len() as u32);
        }
        Ix::new(i)
    }

    /// Adds an item made of one or more ordered, separated fragments.
    /// Its bounding envelope (`first.start..last.end`) is stored in
    /// `start` and `end`; when `fragments` has two or more spans, the
    /// fragment columns are populated. When `fragments` has two or more
    /// spans, every fragment must be non-empty.
    pub fn try_push_fragments(&mut self, fragments: &[Span]) -> Result<Ix<L>, FragmentError> {
        let Some(first) = fragments.first() else {
            return Err(FragmentError::Empty);
        };
        let multi = fragments.len() > 1;
        for f in fragments {
            if f.start > f.end {
                return Err(FragmentError::EndsBeforeStart { start: f.start, end: f.end });
            }
            if multi && f.start == f.end {
                return Err(FragmentError::EmptyFragment { start: f.start });
            }
        }
        for w in fragments.windows(2) {
            if w[0].end >= w[1].start {
                return Err(FragmentError::NotSeparated { first: w[0], second: w[1] });
            }
        }
        if !multi && self.frag_item_end.is_empty() {
            return Ok(self.push(*first));
        }
        if let Some(&prev) = self.start.last()
            && first.start < prev
        {
            self.start_ordered = false;
        }
        if self.frag_item_end.is_empty() && !self.start.is_empty() {
            let n = self.start.len();
            self.frag_start.reserve(n + fragments.len());
            self.frag_end.reserve(n + fragments.len());
            self.frag_item_end.reserve(n + 1);
            self.frag_start.extend_from_slice(&self.start);
            self.frag_end.extend_from_slice(&self.end);
            self.frag_item_end.extend(1..=n as u32);
        }
        let last = fragments.last().expect("at least one fragment");
        let i = self.start.len() as u32;
        self.start.push(first.start);
        self.end.push(last.end);
        for f in fragments {
            self.frag_start.push(f.start);
            self.frag_end.push(f.end);
        }
        self.frag_item_end.push(self.frag_start.len() as u32);
        Ok(Ix::new(i))
    }

    /// Adds an item made of one or more ordered, separated fragments,
    /// panicking if `fragments` is empty, has an empty fragment in a
    /// multi-fragment item, or is not strictly separated in text order.
    pub fn push_fragments(&mut self, fragments: &[Span]) -> Ix<L> {
        self.try_push_fragments(fragments).expect("ordered, separated fragments")
    }

    pub fn len(&self) -> usize {
        self.start.len()
    }

    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }

    /// The bounding span of item `i` (`first.start..last.end` when
    /// discontinuous).
    pub fn get(&self, i: Ix<L>) -> Span {
        Span { start: self.start[i.get()], end: self.end[i.get()] }
    }

    /// Whether any item in the layer has more than one fragment.
    pub fn has_discontinuous(&self) -> bool {
        !self.frag_item_end.is_empty()
    }

    fn frag_range(&self, idx: usize) -> Range<usize> {
        let start = if idx == 0 { 0 } else { self.frag_item_end[idx - 1] as usize };
        let end = self.frag_item_end[idx] as usize;
        start..end
    }

    /// How many fragments item `i` has (1 when continuous).
    pub fn fragment_count(&self, i: Ix<L>) -> usize {
        if self.frag_item_end.is_empty() {
            assert!(i.get() < self.start.len(), "index out of bounds");
            1
        } else {
            self.frag_range(i.get()).len()
        }
    }

    /// Whether item `i` consists of two or more discontinuous fragments.
    pub fn is_discontinuous(&self, i: Ix<L>) -> bool {
        self.fragment_count(i) > 1
    }

    /// The fragments of item `i`, in text order (one span when continuous).
    pub fn fragments(&self, i: Ix<L>) -> Fragments<'_> {
        let idx = i.get();
        if self.frag_item_end.is_empty() {
            Fragments {
                start: std::slice::from_ref(&self.start[idx]),
                end: std::slice::from_ref(&self.end[idx]),
                pos: 0,
            }
        } else {
            let r = self.frag_range(idx);
            Fragments { start: &self.frag_start[r.clone()], end: &self.frag_end[r], pos: 0 }
        }
    }

    /// The raw fragment columns (`[frag_start]`, `[frag_end]`,
    /// `[frag_item_end]`) when any item is discontinuous, or `None` when
    /// every item is a single continuous span.
    pub fn fragment_columns(&self) -> Option<(&[u32], &[u32], &[u32])> {
        if self.frag_item_end.is_empty() { None } else { Some((&self.frag_start, &self.frag_end, &self.frag_item_end)) }
    }

    /// Iterates over `(Ix<L>, Span)` for each item's bounding envelope
    /// (`first.start..last.end`). For individual fragments of a
    /// discontinuous item, use [`Spans::fragments`].
    pub fn iter(&self) -> impl Iterator<Item = (Ix<L>, Span)> + '_ {
        self.start.iter().zip(&self.end).enumerate().map(|(i, (&s, &e))| (Ix::new(i as u32), Span { start: s, end: e }))
    }

    /// The raw envelope columns, for a kernel or the wire: `[start]` and
    /// `[end]`. When any item is discontinuous, these hold each item's
    /// bounding envelope (`first.start..last.end`); see
    /// [`Spans::fragment_columns`] for the per-fragment columns.
    pub fn columns(&self) -> (&[u32], &[u32]) {
        (&self.start, &self.end)
    }

    /// The items whose spans lie inside `outer`, when the spans are in
    /// text order and do not nest (`Overlap::None`): two binary searches,
    /// no scan. For layers whose spans may nest or overlap, use
    /// [`Spans::in_range`].
    pub fn within(&self, outer: Span) -> IxRange<L> {
        let a = self.start.partition_point(|&s| s < outer.start);
        let b = a + self.end[a..].partition_point(|&e| e <= outer.end);
        IxRange::new(a as u32, b as u32)
    }

    /// Whether the `start` column is in non-decreasing text order ($O(1)$,
    /// maintained as spans are pushed).
    pub fn is_start_ordered(&self) -> bool {
        self.start_ordered
    }

    /// Whether the envelope spans are in text order and do not overlap.
    pub fn is_ordered(&self) -> bool {
        self.start_ordered && (1..self.start.len()).all(|i| self.end[i - 1] <= self.start[i])
    }

    /// Checks the fragment table (when present) and the layer's [`Overlap`]
    /// rule.
    pub fn check(&self, overlap: Overlap) -> Result<(), String> {
        self.text_spans().check(overlap)
    }

    /// Every item that covers byte `offset` (for a discontinuous item, only
    /// when one of its fragments covers `offset`, never the gap between
    /// fragments). Uses binary search on the upper bound when `start` is
    /// ordered ($O(1)$ flag check).
    pub fn covering(&self, offset: u32) -> Vec<Ix<L>> {
        let upper = if self.start_ordered { self.start.partition_point(|&s| s <= offset) } else { self.start.len() };
        let mut out = Vec::new();
        for i in 0..upper {
            let env = Span { start: self.start[i], end: self.end[i] };
            if env.covers(offset) {
                let ix = Ix::new(i as u32);
                if self.frag_item_end.is_empty() || self.fragments(ix).any(|f| f.covers(offset)) {
                    out.push(ix);
                }
            }
        }
        out
    }

    /// Every item that overlaps `span` (for a discontinuous item, only when
    /// one of its fragments overlaps `span`, never the gap between
    /// fragments). Uses binary search on the upper bound when `start` is
    /// ordered ($O(1)$ flag check).
    pub fn overlapping(&self, span: Span) -> Vec<Ix<L>> {
        let upper = if self.start_ordered {
            self.start.partition_point(|&s| if span.is_empty() { s <= span.end } else { s < span.end })
        } else {
            self.start.len()
        };
        let mut out = Vec::new();
        for i in range_to(upper) {
            let env = Span { start: self.start[i], end: self.end[i] };
            if env.overlaps(span) {
                let ix = Ix::new(i as u32);
                if self.frag_item_end.is_empty() || self.fragments(ix).any(|f| f.overlaps(span)) {
                    out.push(ix);
                }
            }
        }
        out
    }

    /// Every item whose bounding span lies inside `outer`, even when spans
    /// nest or overlap. Narrowed by binary search on both ends when `start`
    /// is ordered ($O(1)$ flag check).
    pub fn in_range(&self, outer: Span) -> Vec<Ix<L>> {
        let range = if self.start_ordered {
            let a = self.start.partition_point(|&s| s < outer.start);
            let b = a + self.start[a..].partition_point(|&s| s <= outer.end);
            a..b
        } else {
            0..self.start.len()
        };
        let mut out = Vec::new();
        for i in range {
            let s = Span { start: self.start[i], end: self.end[i] };
            if outer.contains(s) {
                out.push(Ix::new(i as u32));
            }
        }
        out
    }

    /// Resolves overlapping or duplicate items into a subset whose bounding
    /// envelopes do not overlap (`!a.overlaps(b)`), in text order,
    /// according to `rule`.
    pub fn filter(&self, scores: Option<&[f64]>, rule: FilterRule) -> Vec<Ix<L>> {
        filter_spans(self, scores, rule)
    }

    /// The items `kept` names, in order: bounding envelopes, and the
    /// fragment columns rebuilt when any kept item is discontinuous (a
    /// kept continuous item is then one fragment). What a predicate-driven
    /// item filter uses to preserve discontinuous spans; `kept` must be
    /// in ascending order with every index inside this layer.
    pub fn subset(&self, kept: &[usize]) -> Spans<L> {
        let mut out = Spans::with_capacity(self.view, kept.len());
        for &i in kept {
            let ix = Ix::new(i as u32);
            if self.is_discontinuous(ix) {
                let fragments: Vec<Span> = self.fragments(ix).collect();
                out.try_push_fragments(&fragments).expect("fragments a valid item holds");
            } else {
                out.push(self.get(ix));
            }
        }
        out
    }
}

#[inline]
const fn range_to(upper: usize) -> Range<usize> {
    0..upper
}

/// Resolves overlapping or duplicate items in `spans` into a subset whose
/// bounding envelopes do not overlap (`!a.overlaps(b)`), returning the kept
/// indices in text order in $O(n \log n)$ time.
///
/// Conflicts and lengths (`end - start`) are computed on each item's
/// bounding envelope (`first.start..last.end`). When every kept item is
/// continuous, the kept spans satisfy [`Overlap::None`].
pub fn filter_spans<L>(spans: &Spans<L>, scores: Option<&[f64]>, rule: FilterRule) -> Vec<Ix<L>> {
    use std::collections::BTreeMap;
    use std::ops::Bound::{Excluded, Unbounded};

    let n = spans.len();
    if let Some(s) = scores {
        assert_eq!(s.len(), n, "a score for every span");
    }
    if n == 0 {
        return Vec::new();
    }
    let score_of = |i: usize| scores.map_or(0.0, |s| s[i]);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let sa = Span { start: spans.start[a], end: spans.end[a] };
        let sb = Span { start: spans.start[b], end: spans.end[b] };
        let len_cmp = sb.len().cmp(&sa.len());
        let score_cmp = score_of(b).total_cmp(&score_of(a));
        let start_cmp = sa.start.cmp(&sb.start);
        match rule {
            FilterRule::LongestFirst => len_cmp.then(score_cmp).then(start_cmp).then(a.cmp(&b)),
            FilterRule::HighestScore => score_cmp.then(len_cmp).then(start_cmp).then(a.cmp(&b)),
            FilterRule::LeftmostLongest => start_cmp.then(len_cmp).then(score_cmp).then(a.cmp(&b)),
        }
    });
    // Kept envelopes never overlap (`!a.overlaps(b)`), so no two kept
    // envelopes can share a `start` offset and their `end` offsets are
    // non-decreasing with `start`. Checking the predecessor (`<= sc.start`)
    // and successor (`> sc.start`) in a BTreeMap is therefore necessary and
    // sufficient to detect any overlap in O(log k) time.
    let mut kept_by_start: BTreeMap<u32, (u32, usize)> = BTreeMap::new();
    for cand in order {
        let sc = Span { start: spans.start[cand], end: spans.end[cand] };
        let hits_prev = kept_by_start
            .range(..=sc.start)
            .next_back()
            .is_some_and(|(&ks, &(ke, _))| sc.overlaps(Span { start: ks, end: ke }));
        if hits_prev {
            continue;
        }
        let hits_next = kept_by_start
            .range((Excluded(sc.start), Unbounded))
            .next()
            .is_some_and(|(&ks, &(ke, _))| sc.overlaps(Span { start: ks, end: ke }));
        if hits_next {
            continue;
        }
        kept_by_start.insert(sc.start, (sc.end, cand));
    }
    kept_by_start.into_values().map(|(_, idx)| Ix::new(idx as u32)).collect()
}

impl<L> Clone for Spans<L> {
    fn clone(&self) -> Self {
        Spans {
            view: self.view,
            start: self.start.clone(),
            end: self.end.clone(),
            frag_start: self.frag_start.clone(),
            frag_end: self.frag_end.clone(),
            frag_item_end: self.frag_item_end.clone(),
            start_ordered: self.start_ordered,
            _l: PhantomData,
        }
    }
}

impl<L> fmt::Debug for Spans<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter().map(|(_, s)| s.start..s.end)).finish()
    }
}
