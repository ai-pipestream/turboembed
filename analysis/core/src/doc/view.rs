//! Text views (`turbo:view`) and their alignment validation.

use std::sync::{Arc, OnceLock};

use crate::error::Error;
use crate::layer::{Anchor, Layer};
use crate::offsets::{CodePointSpan, OffsetTable, Utf16Span};
use crate::span::Span;
use crate::text::{Alignment, ViewId};

/// A text view as a layer. The source text is the default `View`; a
/// normalizer writes a named one ("nfkc", "dehyphenated"), and annotators
/// that should work on normalized text read that one instead. A layer's
/// spans name the view they are in, and the document maps any of them back
/// to the source.
#[derive(Clone, Debug)]
pub struct View {
    pub(crate) id: ViewId,
    pub(crate) parent: Option<ViewId>,
    pub(crate) text: Arc<str>,
    pub(crate) alignment: Alignment,
    /// UTF-16 and code point checkpoints, built on first use.
    pub(crate) offsets: OnceLock<OffsetTable>,
}

impl View {
    /// A view made from `parent` (alignment from a `ViewBuilder`). Its id is
    /// given when the document takes it.
    pub fn derived(parent: &View, text: String, alignment: Alignment) -> View {
        View { id: ViewId(u16::MAX), parent: Some(parent.id), text: text.into(), alignment, offsets: OnceLock::new() }
    }

    pub fn id(&self) -> ViewId {
        self.id
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn parent(&self) -> Option<ViewId> {
        self.parent
    }

    pub fn alignment(&self) -> &Alignment {
        &self.alignment
    }

    /// The table that converts this view's byte offsets to UTF-16 units
    /// and code points and back, for Java and the wire. It is built the
    /// first time it is asked for and kept.
    pub fn offsets(&self) -> &OffsetTable {
        self.offsets.get_or_init(|| OffsetTable::build(&self.text))
    }

    /// Converts a byte [`Span`] in this view into a [`Utf16Span`].
    pub fn span_to_utf16(&self, s: Span) -> Utf16Span {
        self.offsets().span_to_utf16_typed(&self.text, s)
    }

    /// Converts a [`Utf16Span`] in this view into a byte [`Span`].
    pub fn span_from_utf16(&self, s: Utf16Span) -> Result<Span, Error> {
        self.offsets().span_from_utf16(&self.text, s)
    }

    /// Converts a byte [`Span`] in this view into a [`CodePointSpan`].
    pub fn span_to_code_point(&self, s: Span) -> CodePointSpan {
        self.offsets().span_to_code_point_typed(&self.text, s)
    }

    /// Converts a [`CodePointSpan`] in this view into a byte [`Span`].
    pub fn span_from_code_point(&self, s: CodePointSpan) -> Result<Span, Error> {
        self.offsets().span_from_code_point(&self.text, s)
    }
}

impl Layer for View {
    const ID: &'static str = "turbo:view";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        1
    }

    fn check(&self) -> Result<(), String> {
        if u32::try_from(self.text.len()).is_err() {
            return Err(format!("view text is {} bytes, past u32::MAX", self.text.len()));
        }
        let (mut parent_end, mut child_end) = (0usize, 0usize);
        for (i, e) in self.alignment.edits().iter().enumerate() {
            let (p, c) = (e.parent.range(), e.child.range());
            if p.start > p.end || c.start > c.end {
                return Err(format!("edit {i} ends before it starts"));
            }
            if p.is_empty() && c.is_empty() {
                return Err(format!("edit {i} changes nothing"));
            }
            if p.start < parent_end || c.start < child_end {
                return Err(format!("edit {i} starts before the edit before it ends"));
            }
            if p.start - parent_end != c.start - child_end {
                return Err(format!(
                    "the unchanged stretch before edit {i} is {} bytes in the parent and {} in the child",
                    p.start - parent_end,
                    c.start - child_end
                ));
            }
            if c.end > self.text.len() {
                return Err(format!("edit {i} ends at {}, past {}", c.end, self.text.len()));
            }
            if !self.text.is_char_boundary(c.start) || !self.text.is_char_boundary(c.end) {
                return Err(format!("edit {i} cuts a character of the view"));
            }
            (parent_end, child_end) = (p.end, c.end);
        }
        Ok(())
    }
}

/// Checks a derived view's alignment edits and unchanged stretches against
/// its parent view's text.
pub(crate) fn check_view_alignment(parent_text: &str, child_text: &str, alignment: &Alignment) -> Result<(), String> {
    let (mut parent_end, mut child_end) = (0usize, 0usize);
    for (i, e) in alignment.edits().iter().enumerate() {
        let (p, c) = (e.parent.range(), e.child.range());
        if p.end > parent_text.len() {
            return Err(format!("edit {i} ends at {}, past {}", p.end, parent_text.len()));
        }
        if !parent_text.is_char_boundary(p.start) || !parent_text.is_char_boundary(p.end) {
            return Err(format!("edit {i} cuts a character of the parent view"));
        }
        if parent_text.as_bytes()[parent_end..p.start] != child_text.as_bytes()[child_end..c.start] {
            return Err(format!("the unchanged stretch before edit {i} differs between parent and child"));
        }
        (parent_end, child_end) = (p.end, c.end);
    }
    if parent_text.as_bytes()[parent_end..] != child_text.as_bytes()[child_end..] {
        return Err("the unchanged suffix after the last edit differs between parent and child".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::doc::Document;
    use crate::layer::{Key, Provenance};
    use crate::span::Span;
    use crate::text::{Alignment, Edit};

    use super::View;

    #[test]
    fn document_add_rejects_misaligned_views() {
        let mut doc = Document::new("éab");
        // Edit ending past the child view's text.
        let past_child =
            Alignment::from_edits(vec![Edit { parent: Span::new(0, 2), child: Span::new(0, 10) }]).unwrap();
        let bad_child_len = View::derived(doc.get::<View>().unwrap(), "éab".into(), past_child);
        assert!(doc.add(&Key::<View>::named("bad_child_len"), bad_child_len, Provenance::by("test")).is_err());

        // Edit cutting a multi-byte character in the parent view ("é" is 0..2).
        let mid_parent = Alignment::from_edits(vec![Edit { parent: Span::new(1, 2), child: Span::new(1, 2) }]).unwrap();
        let bad_parent = View::derived(doc.get::<View>().unwrap(), "xab".into(), mid_parent);
        assert!(doc.add(&Key::<View>::named("bad_parent"), bad_parent, Provenance::by("test")).is_err());

        // Edit cutting a multi-byte character in the child view.
        let mid_child = Alignment::from_edits(vec![Edit { parent: Span::new(0, 2), child: Span::new(0, 1) }]).unwrap();
        let bad_child = View::derived(doc.get::<View>().unwrap(), "éab".into(), mid_child);
        assert!(doc.add(&Key::<View>::named("bad_child"), bad_child, Provenance::by("test")).is_err());

        // Edit ending past the parent view's text.
        let past_parent =
            Alignment::from_edits(vec![Edit { parent: Span::new(0, 10), child: Span::new(0, 4) }]).unwrap();
        let bad_past = View::derived(doc.get::<View>().unwrap(), "éab".into(), past_parent);
        assert!(doc.add(&Key::<View>::named("bad_past"), bad_past, Provenance::by("test")).is_err());

        // Identity alignment with extra child bytes or different unchanged bytes.
        let mut doc_a = Document::new("a");
        let extra_child = View::derived(doc_a.get::<View>().unwrap(), "ab".into(), Alignment::identity());
        assert!(doc_a.add(&Key::<View>::named("extra_child"), extra_child, Provenance::by("test")).is_err());

        let mut doc_utf8 = Document::new("é");
        let diff_bytes = View::derived(doc_utf8.get::<View>().unwrap(), "ab".into(), Alignment::identity());
        assert!(doc_utf8.add(&Key::<View>::named("diff_bytes"), diff_bytes, Provenance::by("test")).is_err());

        // Valid edit followed by unrecorded extra child byte or mismatched prefix gap.
        let rep = Alignment::from_edits(vec![Edit { parent: Span::new(0, 2), child: Span::new(0, 1) }]).unwrap();
        let extra_suffix = View::derived(doc.get::<View>().unwrap(), "eabc".into(), rep);
        assert!(doc.add(&Key::<View>::named("extra_suffix"), extra_suffix, Provenance::by("test")).is_err());

        let rep2 = Alignment::from_edits(vec![Edit { parent: Span::new(2, 3), child: Span::new(2, 3) }]).unwrap();
        let bad_gap = View::derived(doc.get::<View>().unwrap(), "xxb".into(), rep2);
        assert!(doc.add(&Key::<View>::named("bad_gap"), bad_gap, Provenance::by("test")).is_err());
    }
}
