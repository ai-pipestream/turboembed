//! Source byte accessors and bi-directional view/source offset conversions for [`Document`].

use crate::bytes::SourceBytes;
use crate::offsets::OffsetTable;
use crate::span::Span;
use crate::text::{Side, ViewId};

use super::Document;

impl Document {
    pub fn source(&self) -> &str {
        &self.views[0].text
    }

    /// The bytes the document was decoded from, with their alignment to
    /// view 0; None for a document made from text.
    pub fn source_bytes(&self) -> Option<&SourceBytes> {
        self.bytes.as_deref()
    }

    /// The bytes of the original file: the decoded bytes, or for a
    /// document made from text, that text's UTF-8.
    pub fn bytes(&self) -> &[u8] {
        match &self.bytes {
            Some(b) => b.bytes(),
            None => self.source().as_bytes(),
        }
    }

    /// A span of any view in bytes of the original file: `to_source`, then
    /// one step more through the decoder's alignment, widening where a
    /// character's bytes are cut.
    pub fn to_source_bytes(&self, v: ViewId, s: Span) -> Span {
        let s = self.to_source(v, s);
        match &self.bytes {
            Some(b) => b.alignment().span_to_parent(s),
            None => s,
        }
    }

    /// A span of the original file's bytes in a view: through the
    /// decoder's alignment to view 0, then down the chain.
    pub fn from_source_bytes(&self, v: ViewId, s: Span) -> Span {
        let s = match &self.bytes {
            Some(b) => b.alignment().span_to_child(s),
            None => s,
        };
        self.from_source(v, s)
    }

    /// An offset of a view in bytes of the original file, for one side of
    /// a span.
    pub fn offset_to_source_bytes(&self, v: ViewId, x: u32, side: Side) -> u32 {
        let x = self.offset_to_source(v, x, side);
        match &self.bytes {
            Some(b) => b.alignment().to_parent(x, side),
            None => x,
        }
    }

    /// An offset of the original file's bytes in a view, for one side of a
    /// span.
    pub fn offset_from_source_bytes(&self, v: ViewId, x: u32, side: Side) -> u32 {
        let x = match &self.bytes {
            Some(b) => b.alignment().to_child(x, side),
            None => x,
        };
        self.offset_from_source(v, x, side)
    }

    /// The text of a view.
    pub fn text(&self, v: ViewId) -> &str {
        &self.views[v.0 as usize].text
    }

    /// How many views the document has: valid `ViewId`s are `0..view_count()`.
    pub fn view_count(&self) -> usize {
        self.views.len()
    }

    /// The offset table of a view: see `View::offsets`.
    pub fn offsets(&self, v: ViewId) -> &OffsetTable {
        self.views[v.0 as usize].offsets()
    }

    /// A span of any view, in source bytes: up the chain of views, widening
    /// where an edit is cut.
    pub fn to_source(&self, v: ViewId, mut s: Span) -> Span {
        let mut at = &self.views[v.0 as usize];
        while let Some(p) = at.parent {
            s = at.alignment.span_to_parent(s);
            at = &self.views[p.0 as usize];
        }
        s
    }

    /// A source span in a view's bytes: down the chain.
    pub fn from_source(&self, v: ViewId, s: Span) -> Span {
        let mut chain = Vec::new();
        let mut at = &self.views[v.0 as usize];
        while let Some(p) = at.parent {
            chain.push(at);
            at = &self.views[p.0 as usize];
        }
        chain.iter().rev().fold(s, |s, view| view.alignment.span_to_child(s))
    }

    /// A source offset in a view, for one side of a span: down the chain of
    /// views, one `Alignment::to_child` per view. At inserted bytes a start
    /// goes after them and an end before them, as for a single view.
    pub fn offset_from_source(&self, v: ViewId, x: u32, side: Side) -> u32 {
        let mut chain = Vec::new();
        let mut at = &self.views[v.0 as usize];
        while let Some(p) = at.parent {
            chain.push(at);
            at = &self.views[p.0 as usize];
        }
        chain.iter().rev().fold(x, |x, view| view.alignment.to_child(x, side))
    }

    /// An offset of a view in source bytes, for one side of a span: up the
    /// chain of views, one `Alignment::to_parent` per view.
    pub fn offset_to_source(&self, v: ViewId, mut x: u32, side: Side) -> u32 {
        let mut at = &self.views[v.0 as usize];
        while let Some(p) = at.parent {
            x = at.alignment.to_parent(x, side);
            at = &self.views[p.0 as usize];
        }
        x
    }
}
