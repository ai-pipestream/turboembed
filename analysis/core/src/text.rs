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

    /// A child offset in parent bytes.
    pub fn to_parent(&self, x: u32, side: Side) -> u32 {
        map(&self.edits, x, side, |e| e.child, |e| e.parent)
    }

    /// A parent offset in child bytes.
    pub fn to_child(&self, x: u32, side: Side) -> u32 {
        map(&self.edits, x, side, |e| e.parent, |e| e.child)
    }

    pub fn span_to_parent(&self, s: Span) -> Span {
        Span::new(self.to_parent(s.start, Side::Start), self.to_parent(s.end, Side::End))
    }

    pub fn span_to_child(&self, s: Span) -> Span {
        Span::new(self.to_child(s.start, Side::Start), self.to_child(s.end, Side::End))
    }
}

fn map(edits: &[Edit], x: u32, side: Side, from: impl Fn(&Edit) -> Span, to: impl Fn(&Edit) -> Span) -> u32 {
    // The first edit not wholly before x (for an end, an edit ending at x
    // counts as before it only when it is empty).
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
    pub fn replace(&mut self, end: usize, with: &str) {
        let parent = Span::new(self.at as u32, end as u32);
        let child = Span::new(self.out.len() as u32, (self.out.len() + with.len()) as u32);
        self.out.push_str(with);
        self.at = end;
        // Touching edits merge, so the edit list stays minimal.
        match self.edits.last_mut() {
            Some(last) if last.parent.end == parent.start && last.child.end == child.start => {
                last.parent.end = parent.end;
                last.child.end = child.end;
            }
            _ => self.edits.push(Edit { parent, child }),
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
}
