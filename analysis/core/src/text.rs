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

    /// An alignment from edits made elsewhere, checked: in text order on
    /// both sides, never overlapping, none empty on both sides, and the
    /// unchanged stretch before each edit as long in the parent as in the
    /// child. The texts themselves are not seen here; `SourceBytes::new`
    /// checks the unchanged stretches byte for byte.
    pub fn from_edits(edits: Vec<Edit>) -> Result<Alignment, String> {
        let (mut parent_end, mut child_end) = (0u32, 0u32);
        for (i, e) in edits.iter().enumerate() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_map_both_ways_through_edits() {
        // "The ﬁnal  report" -> "The final report": a ligature expanded and
        // a run of spaces folded.
        let src = "The \u{FB01}nal  report";
        let mut b = ViewBuilder::new(src);
        b.keep(4);
        b.replace(7, "fi"); // U+FB01 is 3 bytes
        b.keep(10);
        b.replace(12, " ");
        let (out, a) = b.finish();
        assert_eq!(out, "The final report");
        // "final" in the view is the ligature and "nal" in the source.
        let final_view = Span::new(4, 9);
        assert_eq!(&src[a.span_to_parent(final_view).range()], "\u{FB01}nal");
        // "report" moves back by one byte.
        let report = Span::new(10, 16);
        assert_eq!(&out[report.range()], "report");
        assert_eq!(&src[a.span_to_parent(report).range()], "report");
        // And forward again.
        assert_eq!(a.span_to_child(Span::new(12, 18)), report);
        // An offset inside the ligature widens to the whole of it.
        assert_eq!(a.span_to_parent(Span::new(5, 6)), Span::new(4, 7));
    }

    #[test]
    fn deleted_bytes_belong_to_neither_neighbour() {
        // A soft hyphen (2 bytes) removed: "co\u{AD}operate" -> "cooperate".
        let src = "co\u{AD}operate";
        let mut b = ViewBuilder::new(src);
        b.keep(2);
        b.replace(4, "");
        let (out, a) = b.finish();
        assert_eq!(out, "cooperate");
        assert_eq!(&src[a.span_to_parent(Span::new(0, 2)).range()], "co");
        assert_eq!(&src[a.span_to_parent(Span::new(2, 9)).range()], "operate");
        assert_eq!(&src[a.span_to_parent(Span::new(0, 9)).range()], src);
    }

    #[test]
    fn characters_replaced_one_by_one_keep_their_own_source_bytes() {
        // Three Arabic-Indic digits (2 bytes each) folded to ASCII, one
        // replace call per digit.
        let src = "n=\u{661}\u{662}\u{663};";
        let mut b = ViewBuilder::new(src);
        b.keep(2);
        b.replace(4, "1");
        b.replace(6, "2");
        b.replace(8, "3");
        let (out, a) = b.finish();
        assert_eq!(out, "n=123;");
        assert_eq!(
            a.edits(),
            [
                Edit { parent: Span::new(2, 4), child: Span::new(2, 3) },
                Edit { parent: Span::new(4, 6), child: Span::new(3, 4) },
                Edit { parent: Span::new(6, 8), child: Span::new(4, 5) },
            ]
        );
        // The middle digit maps to its own source digit and back.
        assert_eq!(a.span_to_parent(Span::new(3, 4)), Span::new(4, 6));
        assert_eq!(&src[a.span_to_parent(Span::new(3, 4)).range()], "\u{662}");
        assert_eq!(a.span_to_child(Span::new(4, 6)), Span::new(3, 4));
        // The whole number still maps to the whole run.
        assert_eq!(a.span_to_parent(Span::new(2, 5)), Span::new(2, 8));
        assert_eq!(a.span_to_child(Span::new(2, 8)), Span::new(2, 5));
    }

    #[test]
    fn a_run_replaced_in_one_call_is_one_edit() {
        let src = "n=\u{661}\u{662}\u{663};";
        let mut b = ViewBuilder::new(src);
        b.keep(2);
        b.replace(8, "123");
        let (out, a) = b.finish();
        assert_eq!(out, "n=123;");
        assert_eq!(a.edits(), [Edit { parent: Span::new(2, 8), child: Span::new(2, 5) }]);
        // One edit is one unit: the middle digit widens to the whole run.
        assert_eq!(a.span_to_parent(Span::new(3, 4)), Span::new(2, 8));
    }

    #[test]
    fn a_call_that_changes_nothing_records_nothing() {
        let mut b = ViewBuilder::new("abc");
        b.keep(1);
        b.replace(1, "");
        let (out, a) = b.finish();
        assert_eq!(out, "abc");
        assert!(a.edits().is_empty());
    }

    /// Checks every offset of both texts against a table written out by
    /// hand: `to_child[x]` is where parent offset x lands as (a start, an
    /// end), and `to_parent` the same the other way.
    fn assert_maps(a: &Alignment, to_child: &[(u32, u32)], to_parent: &[(u32, u32)]) {
        for (x, &want) in to_child.iter().enumerate() {
            let got = (a.to_child(x as u32, Side::Start), a.to_child(x as u32, Side::End));
            assert_eq!(got, want, "parent offset {x} to child");
        }
        for (x, &want) in to_parent.iter().enumerate() {
            let got = (a.to_parent(x as u32, Side::Start), a.to_parent(x as u32, Side::End));
            assert_eq!(got, want, "child offset {x} to parent");
        }
    }

    // The tables below follow these rules. Outside every edit an offset
    // shifts by the edits before it. Strictly inside an edit, a start goes
    // to the start of the edit's output and an end to its end. At the
    // boundary between two touching edits, a start goes with the edit that
    // begins there and an end with the edit that ends there. Inserted or
    // deleted bytes (an edit with one empty side) belong to neither
    // neighbour: a start at them goes after them, an end before them.

    #[test]
    fn touching_replacements_map_at_every_boundary() {
        // "aXYb" -> "apqrb": X became "pq", then Y became "r".
        let mut b = ViewBuilder::new("aXYb");
        b.keep(1);
        b.replace(2, "pq");
        b.replace(3, "r");
        let (out, a) = b.finish();
        assert_eq!(out, "apqrb");
        assert_eq!(a.edits().len(), 2);
        assert_maps(
            &a,
            &[(0, 0), (1, 1), (3, 3), (4, 4), (5, 5)],
            // Child 2 is inside "pq", so it widens to X alone; child 3 is
            // the boundary between "pq" and "r", which is parent 2.
            &[(0, 0), (1, 1), (1, 2), (2, 2), (3, 3), (4, 4)],
        );
    }

    #[test]
    fn an_insertion_touching_a_replacement_maps_at_every_boundary() {
        // "aXb" -> "a+pqb": "+" inserted before X, then X became "pq".
        let mut b = ViewBuilder::new("aXb");
        b.keep(1);
        b.replace(1, "+");
        b.replace(2, "pq");
        let (out, a) = b.finish();
        assert_eq!(out, "a+pqb");
        assert_eq!(a.edits().len(), 2);
        assert_maps(
            &a,
            // Parent 1: a start goes after "+", to the start of "pq"; an
            // end goes before "+".
            &[(0, 0), (2, 1), (4, 4), (5, 5)],
            // Child 1 and 2 are both parent 1 (before and after "+"); child
            // 3 is inside "pq", which widens to X.
            &[(0, 0), (1, 1), (1, 1), (1, 2), (2, 2), (3, 3)],
        );
    }

    #[test]
    fn a_replacement_touching_a_deletion_maps_at_every_boundary() {
        // "aXYb" -> "apqb": X became "pq", then Y was deleted.
        let mut b = ViewBuilder::new("aXYb");
        b.keep(1);
        b.replace(2, "pq");
        b.replace(3, "");
        let (out, a) = b.finish();
        assert_eq!(out, "apqb");
        assert_eq!(a.edits().len(), 2);
        assert_maps(
            &a,
            // Parent 2 and 3 are both child 3 (before and after Y).
            &[(0, 0), (1, 1), (3, 3), (3, 3), (4, 4)],
            // Child 3: a start goes after Y, an end before it.
            &[(0, 0), (1, 1), (1, 2), (3, 2), (4, 4)],
        );
        // "pq" maps back to X without Y, "b" to "b" without Y.
        assert_eq!(a.span_to_parent(Span::new(1, 3)), Span::new(1, 2));
        assert_eq!(a.span_to_parent(Span::new(3, 4)), Span::new(3, 4));
    }

    #[test]
    fn a_deletion_touching_an_insertion_maps_at_every_boundary() {
        // "aXb" -> "a+b": X deleted, then "+" inserted after it. Two
        // calls, so two edits, not one replacement of X by "+".
        let mut b = ViewBuilder::new("aXb");
        b.keep(1);
        b.replace(2, "");
        b.replace(2, "+");
        let (out, a) = b.finish();
        assert_eq!(out, "a+b");
        assert_eq!(a.edits().len(), 2);
        assert_maps(
            &a,
            // Parent 2: a start goes after "+", an end before it (and
            // after X, which became nothing).
            &[(0, 0), (1, 1), (2, 1), (3, 3)],
            // Child 1: a start goes after X, an end before it.
            &[(0, 0), (2, 1), (2, 2), (3, 3)],
        );
        // "+" maps to the empty stretch after X, X to the empty stretch
        // before "+".
        assert_eq!(a.span_to_parent(Span::new(1, 2)), Span::new(2, 2));
        assert_eq!(a.span_to_child(Span::new(1, 2)), Span::new(1, 1));
    }
}
