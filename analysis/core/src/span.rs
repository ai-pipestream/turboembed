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

    pub fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
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

    pub fn iter(&self) -> impl Iterator<Item = Ix<L>> + use<L> {
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

/// The spans of a positional layer, stored by column: one `Vec<u32>` of
/// starts and one of ends, never an object per item. `L` is the layer the
/// spans belong to, so they are indexed by `Ix<L>` alone.
pub struct Spans<L> {
    view: ViewId,
    start: Vec<u32>,
    end: Vec<u32>,
    _l: PhantomData<fn() -> L>,
}

impl<L> Spans<L> {
    pub fn new(view: ViewId) -> Spans<L> {
        Spans { view, start: Vec::new(), end: Vec::new(), _l: PhantomData }
    }

    pub fn with_capacity(view: ViewId, n: usize) -> Spans<L> {
        Spans { view, start: Vec::with_capacity(n), end: Vec::with_capacity(n), _l: PhantomData }
    }

    /// The view the offsets are in.
    pub fn view(&self) -> ViewId {
        self.view
    }

    pub fn push(&mut self, s: Span) -> Ix<L> {
        let i = self.start.len() as u32;
        self.start.push(s.start);
        self.end.push(s.end);
        Ix::new(i)
    }

    pub fn len(&self) -> usize {
        self.start.len()
    }

    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }

    pub fn get(&self, i: Ix<L>) -> Span {
        Span { start: self.start[i.get()], end: self.end[i.get()] }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Ix<L>, Span)> + '_ {
        self.start.iter().zip(&self.end).enumerate().map(|(i, (&s, &e))| (Ix::new(i as u32), Span { start: s, end: e }))
    }

    /// The raw columns, for a kernel or the wire: `[start]` and `[end]`.
    pub fn columns(&self) -> (&[u32], &[u32]) {
        (&self.start, &self.end)
    }

    /// The items whose spans lie inside `outer`, when the spans are in
    /// text order: two binary searches, no scan.
    pub fn within(&self, outer: Span) -> IxRange<L> {
        let a = self.start.partition_point(|&s| s < outer.start);
        let b = a + self.end[a..].partition_point(|&e| e <= outer.end);
        IxRange::new(a as u32, b as u32)
    }

    /// Whether the spans are in text order and do not overlap.
    pub fn is_ordered(&self) -> bool {
        self.start.windows(2).all(|w| w[0] <= w[1])
            && self.iter().zip(self.iter().skip(1)).all(|((_, a), (_, b))| a.end <= b.start)
    }
}

impl<L> Clone for Spans<L> {
    fn clone(&self) -> Self {
        Spans { view: self.view, start: self.start.clone(), end: self.end.clone(), _l: PhantomData }
    }
}

impl<L> fmt::Debug for Spans<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter().map(|(_, s)| s.start..s.end)).finish()
    }
}
