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

    /// Composes `self` (`A -> B`) with `child` (`B -> C`) into a single
    /// minimal `Alignment` (`A -> C`) in $O(|self| + |child|)$ time.
    ///
    /// Edits whose intermediate `B` spans overlap, or where an empty-side
    /// edit (an insertion or deletion) falls strictly inside a non-empty `B`
    /// span, coalesce into one combined `A -> C` edit. Edits that are
    /// disjoint or only touch at a `B` boundary stay separate in `B` order,
    /// unless a cluster cancels out to an empty edit on both sides (`0..0`),
    /// which is omitted.
    pub fn compose(&self, child: &Alignment) -> Alignment {
        if self.edits.is_empty() {
            return child.clone();
        }
        if child.edits.is_empty() {
            return self.clone();
        }
        let mut out = Vec::with_capacity(self.edits.len() + child.edits.len());
        let (mut i, mut j) = (0usize, 0usize);
        let (mut a_at, mut b_at, mut c_at) = (0u32, 0u32, 0u32);

        while i < self.edits.len() || j < child.edits.len() {
            if i < self.edits.len()
                && (j == child.edits.len() || self.edits[i].child.end <= child.edits[j].parent.start)
            {
                let e1 = self.edits[i];
                let d = e1.child.start - b_at;
                let c_start = c_at + d;
                let c_end = c_start + e1.child.len();
                out.push(Edit { parent: e1.parent, child: Span::new(c_start, c_end) });
                a_at = e1.parent.end;
                b_at = e1.child.end;
                c_at = c_end;
                i += 1;
            } else if j < child.edits.len()
                && (i == self.edits.len() || child.edits[j].parent.end <= self.edits[i].child.start)
            {
                let e2 = child.edits[j];
                let d = e2.parent.start - b_at;
                let a_start = a_at + d;
                let a_end = a_start + e2.parent.len();
                out.push(Edit { parent: Span::new(a_start, a_end), child: e2.child });
                a_at = a_end;
                b_at = e2.parent.end;
                c_at = e2.child.end;
                j += 1;
            } else {
                let e1 = self.edits[i];
                let e2 = child.edits[j];
                let cluster_b_start = e1.child.start.min(e2.parent.start);
                let d = cluster_b_start - b_at;
                let cluster_a_start = a_at + d;
                let cluster_c_start = c_at + d;
                let mut cluster_b_end = e1.child.end.max(e2.parent.end);
                i += 1;
                j += 1;
                loop {
                    if i < self.edits.len()
                        && self.edits[i].child.start < cluster_b_end
                        && cluster_b_start < self.edits[i].child.end
                    {
                        cluster_b_end = cluster_b_end.max(self.edits[i].child.end);
                        i += 1;
                        continue;
                    }
                    if j < child.edits.len()
                        && child.edits[j].parent.start < cluster_b_end
                        && cluster_b_start < child.edits[j].parent.end
                    {
                        cluster_b_end = cluster_b_end.max(child.edits[j].parent.end);
                        j += 1;
                        continue;
                    }
                    break;
                }
                let last_ab = self.edits[i - 1];
                let last_bc = child.edits[j - 1];
                let cluster_a_end = last_ab.parent.end + (cluster_b_end - last_ab.child.end);
                let cluster_c_end = last_bc.child.end + (cluster_b_end - last_bc.parent.end);
                a_at = cluster_a_end;
                b_at = cluster_b_end;
                c_at = cluster_c_end;
                if !(cluster_a_start == cluster_a_end && cluster_c_start == cluster_c_end) {
                    out.push(Edit {
                        parent: Span::new(cluster_a_start, cluster_a_end),
                        child: Span::new(cluster_c_start, cluster_c_end),
                    });
                }
            }
        }
        Alignment { edits: out }
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
    // k is the first edit not wholly before x. For a start, an edit ending
    // at x is before it, so a start at the boundary of two touching edits
    // lands at the beginning of the second edit's output. For an end, an
    // edit ending at x is not before it, so an end at that boundary lands at
    // the end of the first edit's output. An edit whose `from` side is
    // empty (an insertion, read from the parent; a deletion, read from the
    // child) is never strictly inside, so the offset falls through to the
    // identity shift: a start goes after its output, an end before it, and
    // the inserted or deleted bytes belong to neither neighbour. Edits only
    // have to be sorted and non-overlapping: touching edits are fine.
    let k = match side {
        Side::Start => edits.partition_point(|e| from(e).end <= x),
        Side::End => edits.partition_point(|e| from(e).end < x),
    };
    if let Some(e) = edits.get(k) {
        let (f, t) = (from(e), to(e));
        if f.start < x && x <= f.end && !(side == Side::Start && x == f.end) {
            return match side {
                Side::Start => t.start,
                Side::End => t.end,
            };
        }
    }
    // Identity, shifted by the edits before x.
    match k.checked_sub(1).map(|p| &edits[p]) {
        Some(prev) => to(prev).end + (x - from(prev).end),
        None => x,
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
        ViewBuilder { parent: out_cap(parent), out: String::with_capacity(parent.len()), edits: Vec::new(), at: 0 }
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
    /// `with` deletes them; `end` at the current position inserts.
    ///
    /// Each call records exactly one edit, and an edit is the smallest unit
    /// an offset can be mapped to: an offset inside it widens to the whole
    /// edit. Calls are never merged with the edit before them, even when
    /// they touch, so the caller chooses the granularity. Replacing three
    /// characters one call each keeps a one-to-one mapping per character;
    /// replacing them in one call makes one edit over all three. A call that
    /// neither consumes nor writes any bytes changes nothing and records
    /// nothing.
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

#[inline]
const fn out_cap(parent: &str) -> &str {
    parent
}

#[cfg(test)]
mod tests;
