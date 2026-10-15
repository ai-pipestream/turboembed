//! The contract a markup extractor (HTML, XML, a word processor format)
//! hands over, so its text joins the document with exact offsets. This
//! crate parses no markup: the extractor does, and fills a
//! [`MarkupText`] with the text it produced, the source span every run of
//! that text came from, and the spans of the tags it dropped. The result
//! becomes a view whose alignment to its parent is exact, so a token found
//! in the extracted text maps back to the markup it came from (and, below
//! view 0, to the file's bytes).
//!
//! Spans on the source side are byte offsets in the parent view, which is
//! normally view 0, the decoded markup. An extractor that reports offsets
//! in the original file's bytes (as `lol_html`'s `source_location()` does)
//! converts them with `Document::from_source_bytes` first; for a UTF-8 file
//! the two are the same numbers.
//!
//! Lucene's `HTMLStripCharFilter` keeps the same information as a list of
//! offset corrections (`correctOffset`); here it is an [`Alignment`], so it
//! maps both ways and composes with every other view.

use std::fmt;

use crate::doc::View;
use crate::span::Span;
use crate::text::ViewBuilder;

/// How a run of extracted text relates to its source span.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TextRunKind {
    /// The text is the source span's bytes unchanged: character data
    /// between tags.
    Copied,
    /// The text stands for the source span: an entity or character
    /// reference (`&amp;` for `&`), CDATA content without its markers, a
    /// normalized line break. One run is one edit, so an offset inside it
    /// widens to the whole source span.
    Decoded,
    /// Text the extractor adds with no source of its own, a line break for
    /// a block element for example. Its source span is empty and sits
    /// where the text is inserted.
    Inserted,
}

/// One run of the extracted text and where it came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextRun {
    /// Bytes of the extracted text.
    pub text: Span,
    /// Bytes of the parent view.
    pub source: Span,
    pub kind: TextRunKind,
}

/// What kind of markup a dropped span was.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TagKind {
    /// `<p>`
    Start,
    /// `</p>`
    End,
    /// `<br/>`
    Empty,
    /// `<!-- -->`
    Comment,
    /// `<!DOCTYPE html>`
    Declaration,
    /// `<?xml ?>`
    ProcessingInstruction,
    /// Anything else the extractor dropped: a script or style body.
    Other,
}

/// A span of markup the extractor dropped, in the parent view.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TagSpan {
    pub source: Span,
    pub kind: TagKind,
}

/// What an extractor hands over: the text it produced, the runs of that
/// text in order with their source spans, and the tags it dropped.
///
/// The runs tile the text: the first starts at 0, each starts where the one
/// before it ends, and the last ends at the end of the text. Their source
/// spans are in order and do not overlap; source bytes no run covers (tags,
/// comments, ignorable whitespace) are deleted from the view. Tag spans
/// are kept as the extractor reported them, in order of their starts; they
/// may contain runs (an attribute value an extractor chose to emit).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MarkupText {
    pub text: String,
    pub runs: Vec<TextRun>,
    pub tags: Vec<TagSpan>,
}

/// Why a `MarkupText` was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MarkupError {
    /// Run `run` does not start where the one before it ends, or the runs
    /// do not reach the end of the text.
    Gap { run: usize },
    /// Run `run`'s source starts before the previous run's source ends.
    Order { run: usize },
    /// A run or tag outside the parent, off a character boundary of the
    /// parent or the text, or a text longer than a `u32` offset reaches.
    Bounds { what: &'static str, index: usize },
    /// A copied run whose text is not its source's bytes, or an inserted
    /// run with a non-empty source.
    Kind { run: usize },
    /// Tag `tag` starts before the tag before it.
    TagOrder { tag: usize },
}

impl fmt::Display for MarkupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarkupError::Gap { run } => write!(f, "markup run {run} does not continue the text"),
            MarkupError::Order { run } => write!(f, "markup run {run} starts before the run before it ends"),
            MarkupError::Bounds { what, index } => write!(f, "markup {what} {index} is out of bounds"),
            MarkupError::Kind { run } => write!(f, "markup run {run} does not match its kind"),
            MarkupError::TagOrder { tag } => write!(f, "markup tag {tag} starts before the tag before it"),
        }
    }
}

impl std::error::Error for MarkupError {}

impl MarkupText {
    pub fn new() -> MarkupText {
        MarkupText::default()
    }

    /// Appends the parent's bytes `source` unchanged.
    pub fn push_copied(&mut self, parent: &str, source: Span) {
        let start = self.text.len() as u32;
        self.text.push_str(&parent[source.range()]);
        self.runs.push(TextRun { text: Span::new(start, self.text.len() as u32), source, kind: TextRunKind::Copied });
    }

    /// Appends `text` standing for the parent's bytes `source`.
    pub fn push_decoded(&mut self, source: Span, text: &str) {
        let start = self.text.len() as u32;
        self.text.push_str(text);
        self.runs.push(TextRun { text: Span::new(start, self.text.len() as u32), source, kind: TextRunKind::Decoded });
    }

    /// Appends `text` with no source, inserted at parent offset `at`.
    pub fn push_inserted(&mut self, at: u32, text: &str) {
        let start = self.text.len() as u32;
        self.text.push_str(text);
        let source = Span::new(at, at);
        self.runs.push(TextRun { text: Span::new(start, self.text.len() as u32), source, kind: TextRunKind::Inserted });
    }

    /// Records a dropped span of markup.
    pub fn push_tag(&mut self, source: Span, kind: TagKind) {
        self.tags.push(TagSpan { source, kind });
    }

    /// Checks the contract against the parent text (see the type's
    /// documentation).
    pub fn check(&self, parent: &str) -> Result<(), MarkupError> {
        let in_parent = |s: Span| {
            (s.end as usize) <= parent.len()
                && parent.is_char_boundary(s.start as usize)
                && parent.is_char_boundary(s.end as usize)
        };
        if u32::try_from(self.text.len()).is_err() {
            return Err(MarkupError::Bounds { what: "text", index: 0 });
        }
        let (mut text_at, mut source_at) = (0u32, 0u32);
        for (i, r) in self.runs.iter().enumerate() {
            if r.text.start != text_at {
                return Err(MarkupError::Gap { run: i });
            }
            text_at = r.text.end;
            let text_ok = (r.text.end as usize) <= self.text.len()
                && self.text.is_char_boundary(r.text.start as usize)
                && self.text.is_char_boundary(r.text.end as usize);
            if !text_ok || !in_parent(r.source) {
                return Err(MarkupError::Bounds { what: "run", index: i });
            }
            if r.source.start < source_at {
                return Err(MarkupError::Order { run: i });
            }
            source_at = r.source.end;
            let kind_ok = match r.kind {
                TextRunKind::Copied => self.text[r.text.range()] == parent[r.source.range()],
                TextRunKind::Decoded => true,
                TextRunKind::Inserted => r.source.is_empty(),
            };
            if !kind_ok {
                return Err(MarkupError::Kind { run: i });
            }
        }
        if text_at as usize != self.text.len() {
            return Err(MarkupError::Gap { run: self.runs.len() });
        }
        let mut tag_at = 0;
        for (i, t) in self.tags.iter().enumerate() {
            if !in_parent(t.source) {
                return Err(MarkupError::Bounds { what: "tag", index: i });
            }
            if t.source.start < tag_at {
                return Err(MarkupError::TagOrder { tag: i });
            }
            tag_at = t.source.start;
        }
        Ok(())
    }

    /// The extracted text as a view of `parent`, after `check`. Source
    /// bytes between runs are deleted, one edit per stretch; a copied run
    /// is unchanged text, with no edit; a decoded or inserted run is one
    /// edit. Add it to the document under a name of its own
    /// (`Key::<View>::named("markup")`).
    pub fn view(&self, parent: &View) -> Result<View, MarkupError> {
        let text = parent.text();
        self.check(text)?;
        let mut b = ViewBuilder::new(text);
        for r in &self.runs {
            let (start, end) = (r.source.start as usize, r.source.end as usize);
            if b.position() < start {
                b.replace(start, "");
            }
            match r.kind {
                TextRunKind::Copied => b.keep(end),
                TextRunKind::Decoded | TextRunKind::Inserted => b.replace(end, &self.text[r.text.range()]),
            }
        }
        if b.position() < text.len() {
            b.replace(text.len(), "");
        }
        let (out, alignment) = b.finish();
        debug_assert_eq!(out, self.text);
        Ok(View::derived(parent, out, alignment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::Document;
    use crate::layer::{Key, Provenance};
    use crate::text::Side;

    /// What an HTML extractor would report for `<p>Fish &amp; chips</p>`.
    fn fish() -> (&'static str, MarkupText) {
        let html = "<p>Fish &amp; chips</p>";
        let mut m = MarkupText::new();
        m.push_tag(Span::new(0, 3), TagKind::Start);
        m.push_copied(html, Span::new(3, 8));
        m.push_decoded(Span::new(8, 13), "&");
        m.push_copied(html, Span::new(13, 19));
        m.push_tag(Span::new(19, 23), TagKind::End);
        (html, m)
    }

    #[test]
    fn extracted_text_maps_back_to_the_markup() {
        let (html, m) = fish();
        assert_eq!(m.text, "Fish & chips");
        let mut doc = Document::new(html);
        let view = m.view(doc.get::<View>().unwrap()).unwrap();
        doc.add(&Key::<View>::named("markup"), view, Provenance::by("test")).unwrap();
        let v = doc.get_key(&Key::<View>::named("markup")).unwrap().id();
        assert_eq!(doc.text(v), "Fish & chips");
        // Words map to their character data.
        assert_eq!(doc.to_source(v, Span::new(0, 4)), Span::new(3, 7));
        assert_eq!(doc.to_source(v, Span::new(7, 12)), Span::new(14, 19));
        // The ampersand is the whole entity, five bytes.
        assert_eq!(doc.to_source(v, Span::new(5, 6)), Span::new(8, 13));
        // A span inside the entity widens to the ampersand.
        assert_eq!(doc.from_source(v, Span::new(9, 11)), Span::new(5, 6));
        // The tags map to no text.
        assert_eq!(doc.from_source(v, Span::new(0, 3)), Span::new(0, 0));
        assert_eq!(doc.from_source(v, Span::new(19, 23)), Span::new(12, 12));
        assert_eq!(doc.offset_to_source(v, 0, Side::Start), 3);
        assert_eq!(doc.offset_to_source(v, 12, Side::End), 19);
        // With no bytes below view 0, its bytes are its UTF-8.
        assert_eq!(doc.to_source_bytes(v, Span::new(5, 6)), Span::new(8, 13));
    }

    #[test]
    fn inserted_text_sits_between_its_neighbours() {
        let html = "a<br>b";
        let mut m = MarkupText::new();
        m.push_copied(html, Span::new(0, 1));
        m.push_tag(Span::new(1, 5), TagKind::Empty);
        m.push_inserted(5, "\n");
        m.push_copied(html, Span::new(5, 6));
        let parent = Document::new(html);
        let view = m.view(parent.get::<View>().unwrap()).unwrap();
        assert_eq!(view.text(), "a\nb");
        let edits = view.alignment().edits();
        // The tag is deleted and the line break inserted: two edits.
        assert_eq!(edits.len(), 2);
        assert_eq!((edits[0].parent, edits[0].child), (Span::new(1, 5), Span::new(1, 1)));
        assert_eq!((edits[1].parent, edits[1].child), (Span::new(5, 5), Span::new(1, 2)));
    }

    #[test]
    fn a_broken_contract_is_refused() {
        let (html, good) = fish();
        assert_eq!(good.check(html), Ok(()));
        let mut m = good.clone();
        m.runs[1].text = Span::new(4, 6);
        assert_eq!(m.check(html), Err(MarkupError::Gap { run: 1 }));
        let mut m = good.clone();
        m.runs[2].source = Span::new(6, 13);
        assert_eq!(m.check(html), Err(MarkupError::Order { run: 2 }));
        let mut m = good.clone();
        m.runs[2].source = Span::new(13, 30);
        assert_eq!(m.check(html), Err(MarkupError::Bounds { what: "run", index: 2 }));
        let mut m = good.clone();
        m.runs[0].kind = TextRunKind::Inserted;
        assert_eq!(m.check(html), Err(MarkupError::Kind { run: 0 }));
        let mut m = good.clone();
        m.runs[1].kind = TextRunKind::Copied;
        assert_eq!(m.check(html), Err(MarkupError::Kind { run: 1 }));
        let mut m = good.clone();
        m.text.push('!');
        assert_eq!(m.check(html), Err(MarkupError::Gap { run: 3 }));
        let mut m = good.clone();
        m.tags.swap(0, 1);
        assert_eq!(m.check(html), Err(MarkupError::TagOrder { tag: 1 }));
        // Off a character boundary of the parent.
        let mut m = MarkupText::new();
        m.push_decoded(Span::new(1, 2), "x");
        assert_eq!(m.check("é"), Err(MarkupError::Bounds { what: "run", index: 0 }));
        // No runs at all is an empty text, with every byte deleted.
        let parent = Document::new("<p></p>");
        let view = MarkupText::new().view(parent.get::<View>().unwrap()).unwrap();
        assert_eq!(view.text(), "");
        assert_eq!(view.alignment().edits().len(), 1);
    }
}
