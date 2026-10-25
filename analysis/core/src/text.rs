//! Text views. The source text never changes. A step that changes text
//! (Unicode normalization, dehyphenation, noise removal) makes a new view
//! with an alignment to its parent, so every span in every view maps back
//! to exact source bytes, and every source span maps forward.

use crate::span::Span;

/// A view of a document's text. View 0 is the source.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ViewId(pub u16);

impl ViewId {
    pub const SOURCE: ViewId = ViewId(0);
}

/// One changed stretch: `parent` bytes became `child` bytes. Outside the
/// edits the two texts are byte for byte the same.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Edit {
    pub parent: Span,
    pub child: Span,
}

mod compose;

use compose::compose_alignments;

/// How a view's bytes line up with its parent's: the edits, in text order
/// on both sides, never overlapping. Identity is no edits. Mapping is a
/// binary search, both ways.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Alignment {
    edits: Vec<Edit>,
}

/// Which side of a span an offset is: a start widens to the start of an
/// edit it falls inside, an end to the edit's end, so a mapped span always
/// covers everything the original covered.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    Start,
    End,
}

impl Alignment {
    pub fn identity() -> Alignment {
        Alignment { edits: Vec::new() }
    }

    pub fn edits(&self) -> &[Edit] {
        &self.edits
    }

    /// An alignment from edits made elsewhere, checked: no span ending
    /// before it starts, in text order on both sides, never overlapping, none empty on both sides, and the
    /// unchanged stretch before each edit as long in the parent as in the
    /// child. The texts themselves are not seen here; `SourceBytes::new`
    /// checks the unchanged stretches byte for byte.
    pub fn from_edits(edits: Vec<Edit>) -> Result<Alignment, String> {
        let (mut parent_end, mut child_end) = (0u32, 0u32);
        for (i, e) in edits.iter().enumerate() {
            if e.parent.start > e.parent.end || e.child.start > e.child.end {
                return Err(format!("edit {i} ends before it starts"));
            }
            if e.parent.is_empty() && e.child.is_empty() {
                return Err(format!("edit {i} changes nothing"));
            }
            if e.parent.start < parent_end || e.child.start < child_end {
                return Err(format!("edit {i} starts before the edit before it ends"));
            }
            if e.parent.start - parent_end != e.child.start - child_end {
                return Err(format!(
                    "the unchanged stretch before edit {i} is {} bytes in the parent and {} in the child",
                    e.parent.start - parent_end,
                    e.child.start - child_end
                ));
            }
            (parent_end, child_end) = (e.parent.end, e.child.end);
        }
        Ok(Alignment { edits })
    }

    /// Edits a builder in this crate recorded, which are in order by
    /// construction.
    pub(crate) fn from_builder(edits: Vec<Edit>) -> Alignment {
        Alignment { edits }
    }

    /// A child offset in parent bytes.
    pub fn to_parent(&self, x: u32, side: Side) -> u32 {
        map(&self.edits, x, side, |e| e.child, |e| e.parent)
    }

    /// A parent offset in child bytes.
    pub fn to_child(&self, x: u32, side: Side) -> u32 {
        map(&self.edits, x, side, |e| e.parent, |e| e.child)
    }

    /// A child span in parent bytes: its start mapped as a `Side::Start`,
    /// its end as a `Side::End`.
    ///
    /// Inserted or deleted bytes belong to neither neighbour, so an empty
    /// span at them stays empty and lands before them. Mapped as two single
    /// offsets, the start of such a span would go after those bytes and its
    /// end before them; the span keeps the smaller of the two offsets
    /// instead. Every other span maps as its two offsets do.
    pub fn span_to_parent(&self, s: Span) -> Span {
        ordered(self.to_parent(s.start, Side::Start), self.to_parent(s.end, Side::End))
    }

    /// A parent span in child bytes, by the same rules as `span_to_parent`:
    /// an empty span at inserted or deleted bytes stays empty and lands
    /// before them.
    pub fn span_to_child(&self, s: Span) -> Span {
        ordered(self.to_child(s.start, Side::Start), self.to_child(s.end, Side::End))
    }

    /// Inverts this alignment in $O(n)$, swapping `parent` and `child` in
    /// every edit so the returned alignment maps from `child` (as parent)
    /// to `parent` (as child).
    pub fn invert(&self) -> Alignment {
        Alignment { edits: self.edits.iter().map(|e| Edit { parent: e.child, child: e.parent }).collect() }
    }

    /// Composes `self` (`A -> B`) with `child` (`B -> C`) into a single flat
    /// net `Alignment` (`A -> C`) represented purely by `edits: Vec<Edit>`.
    ///
    /// Pure insertions in `self` and pure deletions in `child` are split at
    /// adjacent edit boundaries before merging, overlapping edits in `B`
    /// coalesce into net `A -> C` edits, and cancelled `0..0 -> 0..0` edits
    /// are omitted. Because the result is a single flat edit list, it
    /// round-trips losslessly through [`Alignment::from_edits`] and
    /// `DocumentGraph` serialization and applies single-stage `Side::Start` /
    /// `Side::End` and `ordered` rules on the net `A -> C` edits. For
    /// sequential step-by-step projection across intermediate views in a
    /// [`crate::Document`], use [`crate::Document::map_offset`],
    /// [`crate::Document::map_span`], and [`crate::Document::project_spans`].
    pub fn compose(&self, child: &Alignment) -> Alignment {
        compose_alignments(self, child)
    }
}

/// The span from a mapped start to a mapped end. A start can only map past
/// the end when the span was empty and sat on an edit with an empty side;
/// the result is then the empty span at the end, which is the offset before
/// the inserted or deleted bytes.
fn ordered(start: u32, end: u32) -> Span {
    Span::new(start.min(end), end)
}

fn map(edits: &[Edit], x: u32, side: Side, from: impl Fn(&Edit) -> Span, to: impl Fn(&Edit) -> Span) -> u32 {
    let k = match side {
        Side::Start => edits.partition_point(|e| from(e).end <= x),
        Side::End => edits.partition_point(|e| from(e).end < x),
    };
    map_at(edits, k, x, side, from, to)
}

/// [`map`] with its partition point already found: `k` edits end at or
/// before `x` on the `from` side, by the rule of `side`.
fn map_at(
    edits: &[Edit],
    k: usize,
    x: u32,
    side: Side,
    from: impl Fn(&Edit) -> Span,
    to: impl Fn(&Edit) -> Span,
) -> u32 {
    if let Some(e) = edits.get(k) {
        let (f, t) = (from(e), to(e));
        if f.start < x && x <= f.end && !(side == Side::Start && x == f.end) {
            return match side {
                Side::Start => t.start,
                Side::End => t.end,
            };
        }
    }
    match k.checked_sub(1).map(|p| &edits[p]) {
        Some(prev) => to(prev).end + (x - from(prev).end),
        None => x,
    }
}

/// A forward cursor over one [`Alignment`], for mapping many offsets
/// in order: each of the four (direction, side) streams advances its
/// own edit index and never retreats, so a whole layer of monotonic
/// spans maps in $O(\text{edits} + \text{spans})$ with no binary
/// search per offset and no allocation. Offsets in each stream must
/// be non-decreasing (debug-asserted); results are bit-for-bit what
/// [`Alignment::to_parent`], [`Alignment::to_child`],
/// [`Alignment::span_to_parent`] and [`Alignment::span_to_child`]
/// give. Interleaving the streams is fine: mapping a span's start
/// then its end advances two independent streams.
#[derive(Debug)]
pub struct AlignmentCursor<'a> {
    edits: &'a [Edit],
    at: [usize; 4],
    last: [u32; 4],
}

const PARENT_START: usize = 0;
const PARENT_END: usize = 1;
const CHILD_START: usize = 2;
const CHILD_END: usize = 3;

impl<'a> AlignmentCursor<'a> {
    pub fn new(alignment: &'a Alignment) -> AlignmentCursor<'a> {
        AlignmentCursor { edits: &alignment.edits, at: [0; 4], last: [0; 4] }
    }

    fn advance(&mut self, stream: usize, x: u32, side: Side, from: impl Fn(&Edit) -> Span) -> usize {
        debug_assert!(x >= self.last[stream], "alignment cursor offsets must not decrease");
        self.last[stream] = x;
        let mut k = self.at[stream];
        while let Some(e) = self.edits.get(k) {
            let end = from(e).end;
            if side == Side::Start && end <= x || side == Side::End && end < x {
                k += 1;
            } else {
                break;
            }
        }
        self.at[stream] = k;
        k
    }

    /// A child offset in parent bytes, as [`Alignment::to_parent`].
    pub fn to_parent(&mut self, x: u32, side: Side) -> u32 {
        let stream = if side == Side::Start { PARENT_START } else { PARENT_END };
        let k = self.advance(stream, x, side, |e| e.child);
        map_at(self.edits, k, x, side, |e| e.child, |e| e.parent)
    }

    /// A parent offset in child bytes, as [`Alignment::to_child`].
    pub fn to_child(&mut self, x: u32, side: Side) -> u32 {
        let stream = if side == Side::Start { CHILD_START } else { CHILD_END };
        let k = self.advance(stream, x, side, |e| e.parent);
        map_at(self.edits, k, x, side, |e| e.parent, |e| e.child)
    }

    /// A child span in parent bytes, as [`Alignment::span_to_parent`].
    pub fn span_to_parent(&mut self, s: Span) -> Span {
        ordered(self.to_parent(s.start, Side::Start), self.to_parent(s.end, Side::End))
    }

    /// A parent span in child bytes, as [`Alignment::span_to_child`].
    pub fn span_to_child(&mut self, s: Span) -> Span {
        ordered(self.to_child(s.start, Side::Start), self.to_child(s.end, Side::End))
    }
}

impl Alignment {
    /// A forward cursor over this alignment's edits, for mapping a
    /// monotonic run of offsets or spans; see [`AlignmentCursor`].
    pub fn cursor(&self) -> AlignmentCursor<'_> {
        AlignmentCursor::new(self)
    }
}

/// Builds a view from its parent while recording the alignment, so a
/// normalizer cannot produce text without saying where it came from.
pub struct ViewBuilder<'a> {
    parent: &'a str,
    out: String,
    edits: Vec<Edit>,
    at: usize,
}

impl<'a> ViewBuilder<'a> {
    pub fn new(parent: &'a str) -> ViewBuilder<'a> {
        ViewBuilder { parent, out: String::with_capacity(parent.len()), edits: Vec::new(), at: 0 }
    }

    /// Where the builder is in the parent.
    pub fn position(&self) -> usize {
        self.at
    }

    /// Copies the parent unchanged up to `end`.
    pub fn keep(&mut self, end: usize) {
        self.out.push_str(&self.parent[self.at..end]);
        self.at = end;
    }

    /// Writes `with` in place of the parent's bytes up to `end`. An empty
    /// `with` deletes them; `end` at the current position inserts. One call
    /// records one edit; calls are not merged.
    pub fn replace(&mut self, end: usize, with: &str) {
        let parent = Span::new(self.at as u32, end as u32);
        let child = Span::new(self.out.len() as u32, (self.out.len() + with.len()) as u32);
        self.out.push_str(with);
        self.at = end;
        if !(parent.is_empty() && child.is_empty()) {
            self.edits.push(Edit { parent, child });
        }
    }

    pub fn finish(mut self) -> (String, Alignment) {
        let end = self.parent.len();
        self.keep(end);
        (self.out, Alignment { edits: self.edits })
    }
}

#[cfg(test)]
mod tests;
