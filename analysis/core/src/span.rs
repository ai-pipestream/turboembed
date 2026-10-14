//! Offsets. Every offset in the library is a UTF-8 byte offset into one
//! text view, half-open, as u32: the unit turboembed's tokenizer spans
//! already use. UTF-16 and code point offsets are conversions at the edge
//! (Java, the wire), never a second coordinate system inside.

use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::ops::Range;

use crate::text::ViewId;

/// Bytes `[start, end)` of one text view.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const fn new(start: u32, end: u32) -> Span {
        assert!(start <= end, "a span ends at or after its start");
        Span { start, end }
    }

    pub const fn len(self) -> u32 {
        self.end - self.start
    }

    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    pub const fn contains(self, other: Span) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Whether `self` covers byte `offset` (for an empty span at `offset`,
    /// `self.start == offset`; otherwise `self.start <= offset < self.end`).
    pub const fn covers(self, offset: u32) -> bool {
        if self.is_empty() { self.start == offset } else { self.start <= offset && offset < self.end }
    }

    /// Whether `self` and `other` share at least one byte, or an empty
    /// span lies at the start of or inside the other.
    pub const fn overlaps(self, other: Span) -> bool {
        if self.is_empty() && other.is_empty() {
            self.start == other.start
        } else if self.is_empty() {
            other.start <= self.start && self.start < other.end
        } else if other.is_empty() {
            self.start <= other.start && other.start < self.end
        } else {
            self.start < other.end && other.start < self.end
        }
    }

    pub fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

/// What overlap a positional layer's spans may have.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Overlap {
    /// Continuous spans in text order (`start` non-decreasing) that do not
    /// overlap (`end[i - 1] <= start[i]`).
    #[default]
    None,
    /// Spans may nest (`a.contains(b)` or `b.contains(a)` whenever `a` and
    /// `b` overlap), and never partially cross (`a.start < b.start &&
    /// b.start < a.end && a.end < b.end`).
    Nesting,
    /// Any valid spans in the view, including partially crossing spans.
    Any,
}

/// How overlapping spans are resolved to a non-overlapping subset in text
/// order (`Overlap::None`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FilterRule {
    /// Longer span (`end - start`) wins; ties broken by higher score, then
    /// earlier start, then earlier item index.
    LongestFirst,
    /// Higher score wins; ties broken by longer span (`end - start`), then
    /// earlier start, then earlier item index.
    HighestScore,
    /// Earlier start (`start`) wins; at the same start, longer span
    /// (`end - start`) wins, then higher score, then earlier item index.
    LeftmostLongest,
}

/// A position in layer `L`: item `i` of that layer and of no other. An
/// `Ix<Tokens>` cannot index sentences, and a dependency head cannot point
/// at a sentence by mistake; the compiler refuses it.
///
/// ```compile_fail
/// use turbo_analysis::{Ix, Spans};
/// struct Sentences;
/// struct Tokens;
/// fn first_sentence(s: &Spans<Sentences>, t: Ix<Tokens>) {
///     let _ = s.get(t); // expected Ix<Sentences>, found Ix<Tokens>
/// }
/// ```
pub struct Ix<L>(NonZeroU32, PhantomData<fn() -> L>);

// Stored as i + 1, so an absent index costs nothing: a dependency head that
// may be the root, or a parent that may be none, is four bytes.
const _: () = assert!(std::mem::size_of::<Option<Ix<()>>>() == 4);

impl<L> Ix<L> {
    pub const fn new(i: u32) -> Ix<L> {
        match NonZeroU32::new(i.wrapping_add(1)) {
            Some(n) => Ix(n, PhantomData),
            None => panic!("an index below u32::MAX"),
        }
    }

    pub const fn get(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl<L> Clone for Ix<L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<L> Copy for Ix<L> {}
impl<L> PartialEq for Ix<L> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<L> Eq for Ix<L> {}
impl<L> PartialOrd for Ix<L> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<L> Ord for Ix<L> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
impl<L> std::hash::Hash for Ix<L> {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.0.hash(h)
    }
}
impl<L> fmt::Debug for Ix<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = std::any::type_name::<L>();
        write!(f, "{}#{}", name.rsplit("::").next().unwrap_or(name), self.get())
    }
}

/// Items `[start, end)` of layer `L`, such as the tokens of one sentence.
pub struct IxRange<L> {
    pub start: Ix<L>,
    pub end: Ix<L>,
}

impl<L> IxRange<L> {
    pub const fn new(start: u32, end: u32) -> IxRange<L> {
        IxRange { start: Ix::new(start), end: Ix::new(end) }
    }

    pub const fn len(&self) -> usize {
        self.end.get() - self.start.get()
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Ix<L>> + use<L> {
        (self.start.get() as u32..self.end.get() as u32).map(Ix::new)
    }
}

impl<L> Clone for IxRange<L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<L> Copy for IxRange<L> {}
impl<L> PartialEq for IxRange<L> {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}
impl<L> Eq for IxRange<L> {}
impl<L> fmt::Debug for IxRange<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}..{}", self.start, self.end.get())
    }
}

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
            _l: PhantomData,
        }
    }

    /// The view the offsets are in.
    pub fn view(&self) -> ViewId {
        self.view
    }

    pub fn push(&mut self, s: Span) -> Ix<L> {
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
    /// fragment columns are populated.
    pub fn try_push_fragments(&mut self, fragments: &[Span]) -> Result<Ix<L>, String> {
        let Some(first) = fragments.first() else {
            return Err("a span needs at least one fragment".into());
        };
        for f in fragments {
            if f.start > f.end {
                return Err(format!("fragment {}..{} ends before it starts", f.start, f.end));
            }
        }
        for w in fragments.windows(2) {
            if w[0].end >= w[1].start {
                return Err(format!(
                    "fragments {}..{} and {}..{} are not separated in text order",
                    w[0].start, w[0].end, w[1].start, w[1].end
                ));
            }
        }
        if fragments.len() == 1 && self.frag_item_end.is_empty() {
            return Ok(self.push(*first));
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
    /// panicking if `fragments` is empty or not strictly separated in
    /// text order.
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

    pub fn iter(&self) -> impl Iterator<Item = (Ix<L>, Span)> + '_ {
        self.start.iter().zip(&self.end).enumerate().map(|(i, (&s, &e))| (Ix::new(i as u32), Span { start: s, end: e }))
    }

    /// The raw columns, for a kernel or the wire: `[start]` and `[end]`.
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

    /// Whether the `start` column is in non-decreasing text order.
    pub fn is_start_ordered(&self) -> bool {
        self.start.windows(2).all(|w| w[0] <= w[1])
    }

    /// Whether the spans are in text order and do not overlap.
    pub fn is_ordered(&self) -> bool {
        self.is_start_ordered() && self.iter().zip(self.iter().skip(1)).all(|((_, a), (_, b))| a.end <= b.start)
    }

    /// Checks the fragment table (when present) and the layer's [`Overlap`]
    /// rule.
    pub fn check(&self, overlap: Overlap) -> Result<(), String> {
        if self.start.len() != self.end.len() {
            return Err(format!("{} starts and {} ends", self.start.len(), self.end.len()));
        }
        for i in 0..self.start.len() {
            if self.start[i] > self.end[i] {
                return Err(format!("span {i} ends before it starts: {}..{}", self.start[i], self.end[i]));
            }
        }
        if !self.frag_item_end.is_empty() {
            if self.frag_item_end.len() != self.start.len() {
                return Err(format!("{} fragment item ends for {} spans", self.frag_item_end.len(), self.start.len()));
            }
            if self.frag_start.len() != self.frag_end.len() {
                return Err(format!(
                    "{} fragment starts and {} fragment ends",
                    self.frag_start.len(),
                    self.frag_end.len()
                ));
            }
            let total_frags = self.frag_item_end.last().copied().unwrap_or(0) as usize;
            if total_frags != self.frag_start.len() {
                return Err(format!(
                    "fragment item ends reach {total_frags}, fragment columns have {}",
                    self.frag_start.len()
                ));
            }
            for i in 0..self.start.len() {
                let start_f = if i == 0 { 0 } else { self.frag_item_end[i - 1] as usize };
                let end_f = self.frag_item_end[i] as usize;
                if start_f >= end_f || end_f > self.frag_start.len() {
                    return Err(format!("item {i} has an empty or out-of-bounds fragment range"));
                }
                for f in start_f..end_f {
                    if self.frag_start[f] > self.frag_end[f] {
                        return Err(format!(
                            "item {i} fragment ends before it starts: {}..{}",
                            self.frag_start[f], self.frag_end[f]
                        ));
                    }
                    if f > start_f && self.frag_end[f - 1] >= self.frag_start[f] {
                        return Err(format!(
                            "item {i} fragments {}..{} and {}..{} are not separated in text order",
                            self.frag_start[f - 1],
                            self.frag_end[f - 1],
                            self.frag_start[f],
                            self.frag_end[f]
                        ));
                    }
                }
                if self.start[i] != self.frag_start[start_f] || self.end[i] != self.frag_end[end_f - 1] {
                    return Err(format!(
                        "item {i} envelope {}..{} does not match its fragments {}..{}",
                        self.start[i],
                        self.end[i],
                        self.frag_start[start_f],
                        self.frag_end[end_f - 1]
                    ));
                }
            }
        }
        match overlap {
            Overlap::None => {
                if !self.frag_item_end.is_empty() && (0..self.start.len()).any(|i| self.frag_range(i).len() > 1) {
                    return Err("discontinuous spans are not allowed when overlap is None".into());
                }
                if !self.is_ordered() {
                    return Err("spans in text order, not overlapping".into());
                }
                Ok(())
            }
            Overlap::Nesting => {
                check_no_crossing(&self.start, &self.end, "span")?;
                if !self.frag_item_end.is_empty() {
                    check_no_crossing(&self.frag_start, &self.frag_end, "fragment")?;
                }
                Ok(())
            }
            Overlap::Any => Ok(()),
        }
    }

    /// Every item that covers byte `offset` (for a discontinuous item, only
    /// when one of its fragments covers `offset`, never the gap between
    /// fragments). Uses binary search when `start` is ordered.
    pub fn covering(&self, offset: u32) -> Vec<Ix<L>> {
        let upper =
            if self.is_start_ordered() { self.start.partition_point(|&s| s <= offset) } else { self.start.len() };
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
    /// fragments). Uses binary search when `start` is ordered.
    pub fn overlapping(&self, span: Span) -> Vec<Ix<L>> {
        let upper = if self.is_start_ordered() {
            self.start.partition_point(|&s| if span.is_empty() { s <= span.end } else { s < span.end })
        } else {
            self.start.len()
        };
        let mut out = Vec::new();
        for i in 0..upper {
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
    /// is ordered.
    pub fn in_range(&self, outer: Span) -> Vec<Ix<L>> {
        let range = if self.is_start_ordered() {
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

    /// Resolves overlapping or duplicate items into a non-overlapping
    /// subset in text order (`Overlap::None`) according to `rule`.
    pub fn filter(&self, scores: Option<&[f64]>, rule: FilterRule) -> Vec<Ix<L>> {
        filter_spans(self, scores, rule)
    }
}

/// Checks that no two non-empty intervals `[start[i], end[i])` partially
/// cross (`a.start < b.start && b.start < a.end && a.end < b.end`).
fn check_no_crossing(starts: &[u32], ends: &[u32], kind: &str) -> Result<(), String> {
    let n = starts.len();
    if n <= 1 {
        return Ok(());
    }
    let ordered = (1..n).all(|i| starts[i - 1] < starts[i] || (starts[i - 1] == starts[i] && ends[i - 1] >= ends[i]));
    let mut stack: Vec<(usize, u32, u32)> = Vec::new();
    let mut step = |idx: usize| -> Result<(), String> {
        let s = starts[idx];
        let e = ends[idx];
        if s == e {
            return Ok(());
        }
        while let Some(&(_, _, top_end)) = stack.last() {
            if top_end <= s {
                stack.pop();
            } else {
                break;
            }
        }
        if let Some(&(anc_idx, anc_start, anc_end)) = stack.last()
            && e > anc_end
        {
            return Err(format!("{kind} {idx} ({s}..{e}) partially crosses {kind} {anc_idx} ({anc_start}..{anc_end})"));
        }
        stack.push((idx, s, e));
        Ok(())
    };
    if ordered {
        for i in 0..n {
            step(i)?;
        }
    } else {
        let mut order: Vec<usize> = (0..n).filter(|&i| starts[i] < ends[i]).collect();
        order.sort_by(|&a, &b| starts[a].cmp(&starts[b]).then(ends[b].cmp(&ends[a])));
        for i in order {
            step(i)?;
        }
    }
    Ok(())
}

/// Resolves overlapping or duplicate items in `spans` to a non-overlapping
/// subset (`Overlap::None`), returning the kept indices in text order.
pub fn filter_spans<L>(spans: &Spans<L>, scores: Option<&[f64]>, rule: FilterRule) -> Vec<Ix<L>> {
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
    let conflicts = |a: Span, b: Span| a == b || (a.end > b.start && b.end > a.start);
    let mut kept: Vec<usize> = Vec::with_capacity(n);
    for cand in order {
        let sc = Span { start: spans.start[cand], end: spans.end[cand] };
        if !kept.iter().any(|&k| conflicts(sc, Span { start: spans.start[k], end: spans.end[k] })) {
            kept.push(cand);
        }
    }
    kept.sort_by(|&a, &b| spans.start[a].cmp(&spans.start[b]).then(spans.end[a].cmp(&spans.end[b])).then(a.cmp(&b)));
    kept.into_iter().map(|i| Ix::new(i as u32)).collect()
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
            _l: PhantomData,
        }
    }
}

impl<L> fmt::Debug for Spans<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter().map(|(_, s)| s.start..s.end)).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dummy;

    fn spans_of(list: &[(u32, u32)]) -> Spans<Dummy> {
        let mut s = Spans::with_capacity(ViewId::SOURCE, list.len());
        for &(a, b) in list {
            s.push(Span::new(a, b));
        }
        s
    }

    #[test]
    fn overlap_none_nesting_and_any_check_invariants() {
        let ordered = spans_of(&[(0, 4), (4, 8), (10, 14)]);
        assert_eq!(ordered.check(Overlap::None), Ok(()));
        assert_eq!(ordered.check(Overlap::Nesting), Ok(()));
        assert_eq!(ordered.check(Overlap::Any), Ok(()));

        // Nested spans: "Bank of New York" (0..16) and "New York" (8..16),
        // plus an inner "Bank" (0..4) and identical span (8..16).
        let nested = spans_of(&[(0, 16), (0, 4), (8, 16), (8, 16), (17, 22)]);
        assert!(nested.check(Overlap::None).is_err());
        assert_eq!(nested.check(Overlap::Nesting), Ok(()));
        assert_eq!(nested.check(Overlap::Any), Ok(()));

        // Out-of-order nested and disjoint spans still satisfy Nesting.
        let unsorted_nested = spans_of(&[(8, 16), (17, 22), (0, 16), (0, 4)]);
        assert!(unsorted_nested.check(Overlap::None).is_err());
        assert_eq!(unsorted_nested.check(Overlap::Nesting), Ok(()));

        // Partially crossing spans: 0..10 and 6..14 fail Nesting and pass Any.
        let crossing = spans_of(&[(0, 10), (6, 14)]);
        assert!(crossing.check(Overlap::None).is_err());
        assert!(crossing.check(Overlap::Nesting).unwrap_err().contains("partially crosses"));
        assert_eq!(crossing.check(Overlap::Any), Ok(()));
    }

    #[test]
    fn discontinuous_spans_keep_envelopes_and_fragments_by_column() {
        let mut s = Spans::<Dummy>::new(ViewId::SOURCE);
        let i0 = s.push(Span::new(0, 3));
        assert!(!s.has_discontinuous());
        assert!(s.fragment_columns().is_none());

        // "take ... off" at 4..8 and 14..17, with "this coat" in 9..13.
        let i1 = s.push_fragments(&[Span::new(4, 8), Span::new(14, 17)]);
        let i2 = s.push(Span::new(9, 13));
        assert!(s.has_discontinuous());
        assert!(!s.is_discontinuous(i0));
        assert!(s.is_discontinuous(i1));
        assert!(!s.is_discontinuous(i2));
        assert_eq!(s.fragment_count(i0), 1);
        assert_eq!(s.fragment_count(i1), 2);
        assert_eq!(s.fragment_count(i2), 1);
        assert_eq!(s.get(i1), Span::new(4, 17));
        assert_eq!(s.fragments(i0).collect::<Vec<_>>(), [Span::new(0, 3)]);
        assert_eq!(s.fragments(i1).collect::<Vec<_>>(), [Span::new(4, 8), Span::new(14, 17)]);
        assert_eq!(s.fragments(i2).collect::<Vec<_>>(), [Span::new(9, 13)]);

        // Discontinuous spans fail Overlap::None, and pass Nesting when
        // neither envelopes nor fragments cross (here 9..13 is nested
        // inside the envelope 4..17 and disjoint from its fragments).
        assert!(s.check(Overlap::None).is_err());
        assert_eq!(s.check(Overlap::Nesting), Ok(()));
        assert_eq!(s.check(Overlap::Any), Ok(()));

        // Refuses empty, touching, overlapping or out-of-order fragments.
        assert!(s.try_push_fragments(&[]).is_err());
        assert!(s.try_push_fragments(&[Span::new(20, 24), Span::new(24, 28)]).is_err());
        assert!(s.try_push_fragments(&[Span::new(20, 25), Span::new(24, 28)]).is_err());
        assert!(s.try_push_fragments(&[Span::new(24, 28), Span::new(20, 22)]).is_err());
    }

    #[test]
    fn covering_overlapping_and_in_range_handle_nesting_and_gaps() {
        let mut s = Spans::<Dummy>::new(ViewId::SOURCE);
        let outer = s.push(Span::new(0, 16)); // "Bank of New York"
        let inner = s.push(Span::new(8, 16)); // "New York"
        let disc = s.push_fragments(&[Span::new(20, 24), Span::new(30, 33)]); // "take ... off"
        let gap_word = s.push(Span::new(25, 29)); // inside the gap of `disc`

        assert_eq!(s.covering(2), [outer]);
        assert_eq!(s.covering(10), [outer, inner]);
        assert_eq!(s.covering(16), []);
        // Byte 22 is in the first fragment of `disc`; byte 26 is in the gap
        // between its fragments, so `disc` does not cover 26.
        assert_eq!(s.covering(22), [disc]);
        assert_eq!(s.covering(26), [gap_word]);
        assert_eq!(s.covering(31), [disc]);

        assert_eq!(s.overlapping(Span::new(12, 21)), [outer, inner, disc]);
        assert_eq!(s.overlapping(Span::new(25, 29)), [gap_word]);
        assert_eq!(s.overlapping(Span::new(24, 30)), [gap_word]);

        assert_eq!(s.in_range(Span::new(0, 16)), [outer, inner]);
        assert_eq!(s.in_range(Span::new(6, 18)), [inner]);
        assert_eq!(s.in_range(Span::new(20, 33)), [disc, gap_word]);
    }

    #[test]
    fn filter_spans_resolves_conflicts_by_rule_into_ordered_spans() {
        // 0: 0..16 ("Bank of New York", score 0.6)
        // 1: 0..4  ("Bank", score 0.95)
        // 2: 8..16 ("New York", score 0.9)
        // 3: 12..20 ("York City", score 0.85, crosses 0 and 2)
        // 4: 21..27 ("Boston", score 0.7)
        let spans = spans_of(&[(0, 16), (0, 4), (8, 16), (12, 20), (21, 27)]);
        let scores = [0.6, 0.95, 0.9, 0.85, 0.7];

        // LongestFirst: 0..16 (len 16) wins over 0..4, 8..16, and 12..20;
        // 21..27 stays.
        let longest = filter_spans(&spans, Some(&scores), FilterRule::LongestFirst);
        assert_eq!(longest, [Ix::new(0), Ix::new(4)]);

        // HighestScore: 0..4 (0.95) and 8..16 (0.9) win; 0..16 and 12..20
        // conflict with them and are dropped; 21..27 stays.
        let best = filter_spans(&spans, Some(&scores), FilterRule::HighestScore);
        assert_eq!(best, [Ix::new(1), Ix::new(2), Ix::new(4)]);

        // LeftmostLongest: at start 0, 0..16 beats 0..4 and blocks 8..16
        // and 12..20; 21..27 stays.
        let leftmost = filter_spans(&spans, Some(&scores), FilterRule::LeftmostLongest);
        assert_eq!(leftmost, [Ix::new(0), Ix::new(4)]);

        for rule in [FilterRule::LongestFirst, FilterRule::HighestScore, FilterRule::LeftmostLongest] {
            let kept = spans.filter(Some(&scores), rule);
            let mut resolved = Spans::<Dummy>::new(ViewId::SOURCE);
            for ix in kept {
                resolved.push(spans.get(ix));
            }
            assert_eq!(resolved.check(Overlap::None), Ok(()));
        }
    }
}
