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

/// How overlapping or duplicate spans are resolved into a subset whose
/// bounding envelopes do not overlap (`!a.overlaps(b)`), returned in text
/// order.
///
/// Conflicts and span lengths (`end - start`) are evaluated on each item's
/// bounding envelope (`first.start..last.end`), so for a discontinuous item
/// the gap between fragments is counted in its length and treated as covered
/// when checking conflicts. When every kept item is continuous, the result
/// satisfies [`Overlap::None`].
///
/// Scores are compared with [`f64::total_cmp`] (layers such as `Entities`
/// refuse `NaN` scores in [`crate::Layer::check`]).
///
/// Compared with OpenNLP's `NameFinderME.dropOverlappingSpans`:
/// - [`FilterRule::LeftmostLongest`] matches `dropOverlappingSpans` on all
///   non-empty, distinct spans.
/// - On empty spans, [`Span::overlaps`] treats an empty span at the end of a
///   non-empty span (`[3, 3)` after `[1, 3)`) as non-overlapping, whereas
///   OpenNLP's `Span.intersects` counts containment.
/// - On identical spans of different entity types, `dropOverlappingSpans`
///   keeps the UTF-16-first type (`String.compareTo`), whereas `filter_spans`
///   breaks ties by higher score and then earlier item index.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FilterRule {
    /// Longer envelope (`end - start`) wins; ties broken by higher score,
    /// then earlier start, then earlier item index. Matches spaCy's
    /// `util.filter_spans` when scores are equal.
    LongestFirst,
    /// Higher score wins; ties broken by longer envelope (`end - start`),
    /// then earlier start, then earlier item index.
    HighestScore,
    /// Earlier start (`start`) wins; at the same start, longer envelope
    /// (`end - start`) wins, then higher score, then earlier item index.
    LeftmostLongest,
}

/// Why [`Spans::try_push_fragments`] refused a slice of fragments.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FragmentError {
    /// `fragments` is empty.
    Empty,
    /// A fragment's `start` is greater than its `end`.
    EndsBeforeStart { start: u32, end: u32 },
    /// A discontinuous item (`fragments.len() > 1`) has an empty fragment.
    EmptyFragment { start: u32 },
    /// Two adjacent fragments are not strictly separated (`first.end >= second.start`).
    NotSeparated { first: Span, second: Span },
}

impl fmt::Display for FragmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            FragmentError::Empty => f.write_str("a span needs at least one fragment"),
            FragmentError::EndsBeforeStart { start, end } => {
                write!(f, "fragment {start}..{end} ends before it starts")
            }
            FragmentError::EmptyFragment { start } => {
                write!(f, "discontinuous span has an empty fragment at {start}..{start}")
            }
            FragmentError::NotSeparated { first, second } => write!(
                f,
                "fragments {}..{} and {}..{} are not separated in text order",
                first.start, first.end, second.start, second.end
            ),
        }
    }
}

impl std::error::Error for FragmentError {}

/// A borrowed view of a [`Spans<L>`]'s envelope columns and optional
/// fragment columns, returned by [`Spans::text_spans`] and
/// [`crate::Layer::text_spans`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TextSpans<'a> {
    pub view: ViewId,
    pub start: &'a [u32],
    pub end: &'a [u32],
    pub frag_start: &'a [u32],
    pub frag_end: &'a [u32],
    pub frag_item_end: &'a [u32],
}

impl<'a> TextSpans<'a> {
    /// Whether the `start` column is in non-decreasing text order.
    pub fn is_start_ordered(&self) -> bool {
        self.start.windows(2).all(|w| w[0] <= w[1])
    }

    /// Whether the envelope spans are in text order and do not overlap.
    pub fn is_ordered(&self) -> bool {
        self.is_start_ordered() && (1..self.start.len()).all(|i| self.end[i - 1] <= self.start[i])
    }

    /// The flat fragment index range `start_f..end_f` for item `idx` when
    /// `!self.frag_item_end.is_empty()`.
    pub fn item_frag_range(&self, idx: usize) -> Range<usize> {
        let start = if idx == 0 { 0 } else { self.frag_item_end[idx - 1] as usize };
        let end = self.frag_item_end[idx] as usize;
        start..end
    }

    /// Checks the span columns, the fragment table (when present), and the
    /// layer's [`Overlap`] rule.
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
                let r = self.item_frag_range(i);
                let (start_f, end_f) = (r.start, r.end);
                if start_f >= end_f || end_f > self.frag_start.len() {
                    return Err(format!("item {i} has an empty or out-of-bounds fragment range"));
                }
                let multi = end_f - start_f > 1;
                for f in start_f..end_f {
                    if self.frag_start[f] > self.frag_end[f] {
                        return Err(format!(
                            "item {i} fragment ends before it starts: {}..{}",
                            self.frag_start[f], self.frag_end[f]
                        ));
                    }
                    if multi && self.frag_start[f] == self.frag_end[f] {
                        return Err(format!(
                            "item {i} has an empty fragment in a discontinuous span: {}..{}",
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
                if !self.frag_item_end.is_empty() && (0..self.start.len()).any(|i| self.item_frag_range(i).len() > 1) {
                    return Err("discontinuous spans are not allowed when overlap is None".into());
                }
                if !self.is_ordered() {
                    return Err("spans in text order, not overlapping".into());
                }
                Ok(())
            }
            Overlap::Nesting => {
                if self.frag_item_end.is_empty() {
                    check_no_crossing(self.start, self.end, "span")?;
                } else {
                    check_no_crossing(self.frag_start, self.frag_end, "fragment")?;
                    self.check_fragment_nesting()?;
                }
                Ok(())
            }
            Overlap::Any => Ok(()),
        }
    }

    /// When any item is discontinuous under [`Overlap::Nesting`], verifies
    /// that whenever two items share covered bytes (a fragment of one
    /// overlaps a fragment of the other), one item contains the other
    /// (every fragment of the inner item lies inside a fragment of the
    /// outer item).
    fn check_fragment_nesting(&self) -> Result<(), String> {
        let n = self.start.len();
        let frags_of = |i: usize| -> (&[u32], &[u32]) {
            let r = self.item_frag_range(i);
            (&self.frag_start[r.clone()], &self.frag_end[r])
        };
        let contains_item = |outer: usize, inner: usize| -> bool {
            let (os, oe) = frags_of(outer);
            let (is, ie) = frags_of(inner);
            is.iter().zip(ie).all(|(&cs, &ce)| {
                let child = Span { start: cs, end: ce };
                os.iter().zip(oe).any(|(&ps, &pe)| Span { start: ps, end: pe }.contains(child))
            })
        };
        let shares_bytes = |a: usize, b: usize| -> bool {
            let (as_, ae) = frags_of(a);
            let (bs, be) = frags_of(b);
            as_.iter().zip(ae).any(|(&s1, &e1)| {
                let f1 = Span { start: s1, end: e1 };
                bs.iter().zip(be).any(|(&s2, &e2)| f1.overlaps(Span { start: s2, end: e2 }))
            })
        };
        for a in 0..n {
            let env_a = Span { start: self.start[a], end: self.end[a] };
            for b in (a + 1)..n {
                let env_b = Span { start: self.start[b], end: self.end[b] };
                if env_a.overlaps(env_b) && shares_bytes(a, b) && !contains_item(a, b) && !contains_item(b, a) {
                    return Err(format!(
                        "item {b} ({}..{}) partially crosses item {a} ({}..{}) across fragments",
                        env_b.start, env_b.end, env_a.start, env_a.end
                    ));
                }
            }
        }
        Ok(())
    }
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
}

#[inline]
const fn range_to(upper: usize) -> Range<usize> {
    0..upper
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

        // Out-of-order nested and disjoint spans fail Overlap::None and
        // pass Overlap::Nesting and Overlap::Any (e.g. RegexNameFinder emits
        // entities grouped by type rather than text start order).
        let unsorted_nested = spans_of(&[(8, 16), (17, 22), (0, 16), (0, 4)]);
        assert!(!unsorted_nested.is_start_ordered());
        assert!(unsorted_nested.check(Overlap::None).is_err());
        assert_eq!(unsorted_nested.check(Overlap::Nesting), Ok(()));
        assert_eq!(unsorted_nested.check(Overlap::Any), Ok(()));

        // Two identical spans (0..5, 0..5) and 0..10 containing 0..5 and 5..10
        // satisfy Overlap::Nesting; 0..6 and 4..10 partially cross.
        assert_eq!(spans_of(&[(0, 5), (0, 5)]).check(Overlap::Nesting), Ok(()));
        assert_eq!(spans_of(&[(0, 10), (0, 5), (5, 10)]).check(Overlap::Nesting), Ok(()));
        assert!(spans_of(&[(0, 6), (4, 10)]).check(Overlap::Nesting).unwrap_err().contains("partially crosses"));

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

        // Refuses empty input, backwards fragment, empty fragment in
        // multi-fragment, and touching/overlapping/out-of-order fragments.
        assert_eq!(s.try_push_fragments(&[]), Err(FragmentError::Empty));
        assert_eq!(
            s.try_push_fragments(&[Span { start: 25, end: 20 }]),
            Err(FragmentError::EndsBeforeStart { start: 25, end: 20 }),
        );
        assert_eq!(
            s.try_push_fragments(&[Span::new(20, 20), Span::new(24, 28)]),
            Err(FragmentError::EmptyFragment { start: 20 }),
        );
        assert_eq!(
            s.try_push_fragments(&[Span::new(20, 24), Span::new(24, 28)]),
            Err(FragmentError::NotSeparated { first: Span::new(20, 24), second: Span::new(24, 28) }),
        );
        assert_eq!(
            s.try_push_fragments(&[Span::new(20, 25), Span::new(24, 28)]),
            Err(FragmentError::NotSeparated { first: Span::new(20, 25), second: Span::new(24, 28) }),
        );
        assert_eq!(
            s.try_push_fragments(&[Span::new(24, 28), Span::new(20, 22)]),
            Err(FragmentError::NotSeparated { first: Span::new(24, 28), second: Span::new(20, 22) }),
        );
    }

    #[test]
    fn with_capacity_followed_by_push_fragments_preserves_first_item_offsets() {
        let mut s = Spans::<Dummy>::with_capacity(ViewId::SOURCE, 4);
        let i0 = s.push_fragments(&[Span::new(2, 5), Span::new(10, 14)]);
        let i1 = s.push(Span::new(15, 20));
        assert_eq!(s.get(i0), Span::new(2, 14));
        assert_eq!(s.fragments(i0).collect::<Vec<_>>(), [Span::new(2, 5), Span::new(10, 14)]);
        assert_eq!(s.fragments(i1).collect::<Vec<_>>(), [Span::new(15, 20)]);
        assert_eq!(s.check(Overlap::Nesting), Ok(()));
    }

    #[test]
    fn discontinuous_fragment_nesting_rejects_crossing_fragments_and_gap_splits() {
        // Valid: outer 0..20 contains discontinuous 2..5; 10..15.
        let mut ok = Spans::<Dummy>::new(ViewId::SOURCE);
        ok.push(Span::new(0, 20));
        ok.push_fragments(&[Span::new(2, 5), Span::new(10, 15)]);
        assert_eq!(ok.check(Overlap::Nesting), Ok(()));

        // Valid: two discontinuous items whose fragments are completely
        // disjoint (0..4; 10..13 and 5..8; 14..16), even though their
        // bounding envelopes (0..13 and 5..16) cross.
        let mut disjoint_interleaved = Spans::<Dummy>::new(ViewId::SOURCE);
        disjoint_interleaved.push_fragments(&[Span::new(0, 4), Span::new(10, 13)]);
        disjoint_interleaved.push_fragments(&[Span::new(5, 8), Span::new(14, 16)]);
        assert_eq!(disjoint_interleaved.check(Overlap::Nesting), Ok(()));

        // Invalid: item 0 has fragments 0..6 and 15..20 (envelope 0..20),
        // item 1 is 4..10 (inside envelope 0..20, but partially crosses
        // fragment 0..6 and is not contained by item 0's fragments).
        let mut bad = Spans::<Dummy>::new(ViewId::SOURCE);
        bad.push_fragments(&[Span::new(0, 6), Span::new(15, 20)]);
        bad.push(Span::new(4, 10));
        let err = bad.check(Overlap::Nesting).unwrap_err();
        assert!(err.contains("partially crosses"), "{err}");
        // Under Overlap::Any, the same spans are accepted.
        assert_eq!(bad.check(Overlap::Any), Ok(()));

        // Invalid, and only the item-level check sees it: no fragment
        // crosses another, but item 1 (2..4; 10..12) is neither inside
        // item 0 (0..6; 15..20) nor outside it.
        let mut bad = Spans::<Dummy>::new(ViewId::SOURCE);
        bad.push_fragments(&[Span::new(0, 6), Span::new(15, 20)]);
        bad.push_fragments(&[Span::new(2, 4), Span::new(10, 12)]);
        assert_eq!(check_no_crossing(bad.frag_start.as_slice(), bad.frag_end.as_slice(), "fragment"), Ok(()));
        let err = bad.check(Overlap::Nesting).unwrap_err();
        assert!(err.contains("across fragments"), "{err}");

        // Invalid: item 0 has fragments 0..5 and 10..20, item 1 is 2..15
        // (shares bytes with both fragments of item 0 without either item
        // containing the other).
        let mut bad2 = Spans::<Dummy>::new(ViewId::SOURCE);
        bad2.push_fragments(&[Span::new(0, 5), Span::new(10, 20)]);
        bad2.push(Span::new(2, 15));
        assert!(bad2.check(Overlap::Nesting).is_err());
    }

    #[test]
    fn text_spans_check_detects_corrupted_fragment_columns() {
        let start = [0u32, 10];
        let end = [8u32, 20];
        let frag_start = [0u32, 5, 10];
        let frag_end = [3u32, 8, 20];
        let frag_item_end = [2u32, 3];
        let base = TextSpans {
            view: ViewId::SOURCE,
            start: &start,
            end: &end,
            frag_start: &frag_start,
            frag_end: &frag_end,
            frag_item_end: &frag_item_end,
        };
        assert_eq!(base.check(Overlap::Nesting), Ok(()));

        // 1. Unequal frag_start / frag_end length.
        let bad_frag_len = TextSpans { frag_end: &frag_end[..2], ..base };
        assert!(bad_frag_len.check(Overlap::Any).unwrap_err().contains("fragment starts"));

        // 2. frag_item_end.len() != start.len().
        let bad_item_end_len = TextSpans { frag_item_end: &frag_item_end[..1], ..base };
        assert!(bad_item_end_len.check(Overlap::Any).unwrap_err().contains("fragment item ends"));

        // 3. frag_item_end.last() != Some(frag_start.len()).
        let bad_total = [1u32, 2];
        let bad_last = TextSpans { frag_item_end: &bad_total, ..base };
        assert!(bad_last.check(Overlap::Any).unwrap_err().contains("fragment item ends reach"));

        // 4. 0-fragment item / non-monotonic frag_item_end.
        let zero_frag = [0u32, 3];
        let bad_zero = TextSpans { frag_item_end: &zero_frag, ..base };
        assert!(bad_zero.check(Overlap::Any).unwrap_err().contains("empty or out-of-bounds fragment range"));

        // 5. Envelope mismatch with first/last fragment.
        let wrong_env_start = [1u32, 10];
        let bad_env = TextSpans { start: &wrong_env_start, ..base };
        assert!(bad_env.check(Overlap::Any).unwrap_err().contains("envelope"));
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

        // Empty Spans, out-of-order Spans, zero-width spans at query boundary,
        // and duplicate spans.
        let empty = Spans::<Dummy>::new(ViewId::SOURCE);
        assert_eq!(empty.covering(0), []);
        assert_eq!(empty.overlapping(Span::new(0, 5)), []);
        assert_eq!(empty.in_range(Span::new(0, 5)), []);

        let unsorted = spans_of(&[(10, 15), (5, 5), (0, 8), (0, 8)]);
        assert!(!unsorted.is_start_ordered());
        assert_eq!(unsorted.covering(5), [Ix::new(1), Ix::new(2), Ix::new(3)]);
        assert_eq!(unsorted.overlapping(Span::new(5, 10)), [Ix::new(1), Ix::new(2), Ix::new(3)]);
        assert_eq!(unsorted.in_range(Span::new(0, 10)), [Ix::new(1), Ix::new(2), Ix::new(3)]);

        let boundary = spans_of(&[(0, 5), (5, 5), (5, 10), (10, 10)]);
        assert_eq!(boundary.overlapping(Span::new(5, 10)), [Ix::new(1), Ix::new(2)]);
        assert_eq!(boundary.overlapping(Span::new(5, 5)), [Ix::new(1), Ix::new(2)]);
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

        // Difference between LongestFirst and LeftmostLongest when an earlier
        // shorter span crosses a later longer span: 0..6 (len 6) vs 4..14 (len 10).
        let cross = spans_of(&[(0, 6), (4, 14)]);
        assert_eq!(filter_spans(&cross, None, FilterRule::LongestFirst), [Ix::new(1)]);
        assert_eq!(filter_spans(&cross, None, FilterRule::LeftmostLongest), [Ix::new(0)]);

        for rule in [FilterRule::LongestFirst, FilterRule::HighestScore, FilterRule::LeftmostLongest] {
            let kept = spans.filter(Some(&scores), rule);
            let mut resolved = Spans::<Dummy>::new(ViewId::SOURCE);
            for ix in kept {
                resolved.push(spans.get(ix));
            }
            assert_eq!(resolved.check(Overlap::None), Ok(()));
        }
    }

    /// Parity fixture against spaCy's `spacy.util.filter_spans`: sorts by
    /// `(span.end - span.start)` descending, then `span.start` ascending,
    /// keeping non-overlapping spans and returning them in text order.
    #[test]
    fn filter_spans_longest_first_matches_spacy_util_filter_spans() {
        type Case<'a> = (&'a [(u32, u32)], &'a [u32]);
        let fixtures: &[Case<'_>] = &[
            // Nested and crossing spans from spaCy's test_filter_spans.
            (&[(1, 4), (2, 3), (5, 8), (6, 9), (1, 2)], &[0, 2]),
            // Equal-length overlapping spans: earlier start wins (2..6 over 4..8).
            (&[(4, 8), (2, 6)], &[1]),
            // Identical duplicate spans: earlier input index wins (index 0 over index 1).
            (&[(2, 6), (2, 6), (6, 10)], &[0, 2]),
            // Touching spans do not overlap and are all kept.
            (&[(0, 3), (3, 7), (7, 10)], &[0, 1, 2]),
            // Chain of crossing spans where the middle longest span unblocks both ends.
            (&[(0, 4), (2, 10), (8, 12)], &[1]),
            (&[(0, 3), (2, 5), (4, 10), (9, 12)], &[0, 2]),
        ];
        for &(input, expected) in fixtures {
            let s = spans_of(input);
            let got: Vec<u32> =
                filter_spans(&s, None, FilterRule::LongestFirst).into_iter().map(|ix| ix.get() as u32).collect();
            assert_eq!(got, expected, "input: {input:?}");
        }
    }
}
