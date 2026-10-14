//! The offset invariants every layer with spans must keep, checked as
//! properties over any document: on the fuzz corpus, on the golden
//! inputs, on anything an annotator produced.
//!
//! For each span of a text anchored layer, in the view its spans are in:
//!
//! - it is in bounds and ordered, and both ends are on character
//!   boundaries of the view's text;
//! - UTF-8 to UTF-16 to UTF-8 round trips through the view's
//!   `OffsetTable`, and so do code point offsets; the UTF-16 offsets agree
//!   with `turbo_unicode`'s own conversion, which is written apart from it;
//! - it maps to source bytes exactly through every normalizer between the
//!   view and the source: the source span is on character boundaries,
//!   mapping it forward again covers the span, mapping that back gives the
//!   same source span, and where the span touches no edit of the view's
//!   alignment the source bytes are the view's bytes.
//!
//! `check_document` finds every instance of the shared layer types that
//! have text spans (`known_layers`); `check_spans` takes any span columns.
//! A `Violation` names the layer, the span and what failed, so a test can
//! print the list and stop.

use std::any::TypeId;
use std::fmt;

use turbo_analysis::{Document, Key, Layer, Span, View, ViewId};
use turbo_analysis_layers::{
    Chunks, Entities, Money, Numbers, PhraseChunks, Quantities, Sentences, Structure, Subwords, Tokens,
};
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};

/// One broken invariant: the layer (as the document names it), the view
/// and span it was found on, and what failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub layer: String,
    pub view: ViewId,
    pub index: usize,
    pub span: Span,
    pub what: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} item {} at {}..{} in view {}: {}",
            self.layer, self.index, self.span.start, self.span.end, self.view.0, self.what
        )
    }
}

/// Checks the spans of one layer of `doc`: `starts[i]..ends[i]` in `view`.
pub fn check_spans(doc: &Document, layer: &str, view: ViewId, starts: &[u32], ends: &[u32]) -> Vec<Violation> {
    let mut out = Vec::new();
    let text = doc.text(view);
    let offsets = doc.offsets(view);
    let source = doc.source();
    let edits = view_edits(doc, view);
    if starts.len() != ends.len() {
        out.push(Violation {
            layer: layer.into(),
            view,
            index: 0,
            span: Span::new(0, 0),
            what: format!("{} starts and {} ends", starts.len(), ends.len()),
        });
        return out;
    }
    for (i, (&start, &end)) in starts.iter().zip(ends).enumerate() {
        // Not `Span::new`: a span ending before its start is one of the
        // things reported here.
        let span = Span { start, end };
        let mut fail = |what: String| out.push(Violation { layer: layer.into(), view, index: i, span, what });
        if start > end || end as usize > text.len() {
            fail(format!("not inside the {} bytes of the view, or ends before it starts", text.len()));
            continue;
        }
        if !text.is_char_boundary(start as usize) || !text.is_char_boundary(end as usize) {
            fail("cuts a character".into());
            continue;
        }
        for (what, b) in [("start", start), ("end", end)] {
            let u = offsets.to_utf16(text, b);
            match offsets.from_utf16(text, u) {
                Ok(back) if back == b => {}
                Ok(back) => fail(format!("{what} {b} goes to UTF-16 {} and back to {back}", u.0)),
                Err(e) => fail(format!("{what} {b} goes to UTF-16 {} and does not come back: {e}", u.0)),
            }
            let independent = byte_to_utf16(text, b as usize);
            if independent != Some(u.0 as usize) {
                fail(format!("{what} {b}: UTF-16 {} by the offset table, {independent:?} by turbo_unicode", u.0));
            }
            if utf16_to_byte(text, u.0 as usize) != Some(b as usize) {
                fail(format!("{what} {b}: UTF-16 {} does not come back through turbo_unicode", u.0));
            }
            let cp = offsets.to_code_point(text, b);
            match offsets.from_code_point(text, cp) {
                Ok(back) if back == b => {}
                Ok(back) => fail(format!("{what} {b} goes to code point {} and back to {back}", cp.0)),
                Err(e) => fail(format!("{what} {b} goes to code point {} and does not come back: {e}", cp.0)),
            }
        }
        let (u16s, u16e) = offsets.span_to_utf16(text, span);
        if u16s.0 > u16e.0 || (u16e.0 - u16s.0) as usize != text[span.range()].encode_utf16().count() {
            fail(format!("the UTF-16 span {}..{} is not the text's length in units", u16s.0, u16e.0));
        }
        // Back to the source and forward again.
        let back = doc.to_source(view, span);
        if back.start > back.end || back.end as usize > source.len() {
            fail(format!("maps to {}..{}, outside the source", back.start, back.end));
            continue;
        }
        if !source.is_char_boundary(back.start as usize) || !source.is_char_boundary(back.end as usize) {
            fail(format!("maps to {}..{}, which cuts a source character", back.start, back.end));
            continue;
        }
        let forward = doc.from_source(view, back);
        if forward.start > start || forward.end < end {
            fail(format!(
                "maps to source {}..{}, which maps forward to {}..{}, not covering it",
                back.start, back.end, forward.start, forward.end
            ));
        }
        let again = doc.to_source(view, forward);
        if again != back {
            fail(format!(
                "maps to source {}..{}, forward to {}..{}, back to {}..{}",
                back.start, back.end, forward.start, forward.end, again.start, again.end
            ));
        }
        if view != ViewId::SOURCE && !touches_an_edit(&edits, span) && source[back.range()] != text[span.range()] {
            fail(format!(
                "touches no edit, yet maps to source {}..{} {:?}, not its own text {:?}",
                back.start,
                back.end,
                &source[back.range()],
                &text[span.range()]
            ));
        }
    }
    out
}

/// The edits of the view's own alignment (against its parent), when the
/// view is a layer of the document; a view the document holds only in its
/// chain has none listed, and the text check is skipped for it.
fn view_edits(doc: &Document, view: ViewId) -> Vec<Span> {
    if view == ViewId::SOURCE {
        return Vec::new();
    }
    let target = Key::<View>::DEFAULT.id().ty;
    for k in doc.layers() {
        if k.ty == target
            && let Some(v) = doc.get_key(&Key::<View>::named(k.name))
            && v.id() == view
        {
            return v.alignment().edits().iter().map(|e| e.child).collect();
        }
    }
    Vec::new()
}

/// Whether `span` overlaps an edit's child bytes, or has a deletion (an
/// edit with no child bytes) strictly inside it: either makes its source
/// bytes differ from its own.
fn touches_an_edit(edits: &[Span], span: Span) -> bool {
    edits.iter().any(|e| {
        if e.start == e.end {
            span.start < e.start && e.start < span.end
        } else {
            e.start < span.end && e.end > span.start
        }
    })
}

/// Checks one instance of a layer type with text spans.
pub fn check_layer<L: Layer>(doc: &Document, key: &Key<L>) -> Vec<Violation> {
    let Some(layer) = doc.get_key(key) else { return Vec::new() };
    let Some((view, starts, ends)) = layer.text_spans() else { return Vec::new() };
    check_spans(doc, &key.id().to_string(), view, starts, ends)
}

type Check = fn(&Document, &'static str) -> Vec<Violation>;

fn checker<L: Layer>() -> (TypeId, Check) {
    (Key::<L>::DEFAULT.id().ty, |doc, name| check_layer(doc, &Key::<L>::named(name)))
}

/// The shared layer types with text spans, which `check_document` looks
/// for on a document.
fn known_layers() -> Vec<(TypeId, Check)> {
    vec![
        checker::<Sentences>(),
        checker::<Tokens>(),
        checker::<Subwords>(),
        checker::<Entities>(),
        checker::<PhraseChunks>(),
        checker::<Numbers>(),
        checker::<Money>(),
        checker::<Quantities>(),
        checker::<Structure>(),
        checker::<Chunks>(),
    ]
}

/// Checks every instance, default or named, of every known layer type on
/// `doc`, and returns how many layers were checked with the violations.
pub fn check_document(doc: &Document) -> (usize, Vec<Violation>) {
    let known = known_layers();
    let mut checked = 0;
    let mut out = Vec::new();
    for k in doc.layers() {
        if let Some((_, check)) = known.iter().find(|(ty, _)| *ty == k.ty) {
            checked += 1;
            out.extend(check(doc, k.name));
        }
    }
    (checked, out)
}

/// Panics with every violation listed when there is one.
#[track_caller]
pub fn assert_invariants(doc: &Document, what: &str) -> usize {
    let (checked, violations) = check_document(doc);
    assert!(
        violations.is_empty(),
        "{what}: {} offset invariants broken:\n{}",
        violations.len(),
        violations.iter().map(|v| format!("  {v}")).collect::<Vec<_>>().join("\n")
    );
    checked
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Alignment, Provenance, Spans, ViewBuilder};

    fn tokens(view: ViewId, spans: &[(u32, u32)]) -> Tokens {
        let mut s = Spans::new(view);
        for &(a, b) in spans {
            s.push(Span::new(a, b));
        }
        Tokens { spans: s, sentence: None }
    }

    #[test]
    fn well_formed_spans_pass_in_text_of_every_width() {
        let text = "a é 中 😀 b";
        let mut doc = Document::new(text);
        let spans: Vec<(u32, u32)> = text
            .split(' ')
            .scan(0u32, |at, w| {
                let s = (*at, *at + w.len() as u32);
                *at += w.len() as u32 + 1;
                Some(s)
            })
            .collect();
        doc.add(&Key::DEFAULT, tokens(ViewId::SOURCE, &spans), Provenance::by("test")).unwrap();
        assert_eq!(assert_invariants(&doc, "plain"), 1);
    }

    #[test]
    fn a_span_cutting_a_character_or_leaving_the_text_is_reported() {
        let doc = Document::new("é😀");
        let v = check_spans(&doc, "opennlp:tokens", ViewId::SOURCE, &[1, 0, 3], &[2, 7, 2]);
        assert_eq!(v.len(), 3);
        assert!(v[0].what.contains("cuts a character"), "{}", v[0]);
        assert!(v[1].what.contains("not inside"), "{}", v[1]);
        assert!(v[2].what.contains("ends before"), "{}", v[2]);
        assert_eq!(v[0].to_string(), "opennlp:tokens item 0 at 1..2 in view 0: cuts a character");
    }

    #[test]
    fn spans_in_a_normalized_view_map_back_exactly_outside_the_edits() {
        // "Hello  WORLD" with the double space collapsed and WORLD lowered.
        let source = "Hello  WORLD!";
        let mut doc = Document::new(source);
        let mut b = ViewBuilder::new(source);
        b.keep(5);
        b.replace(7, " ");
        b.replace(12, "world");
        let (text, alignment) = b.finish();
        assert_eq!(text, "Hello world!");
        let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
        doc.add(&Key::<View>::named("norm"), view, Provenance::by("test")).unwrap();
        let id = doc.get_key(&Key::<View>::named("norm")).unwrap().id();
        // "Hello", "world" and "!" as tokens of the view.
        doc.add(&Key::DEFAULT, tokens(id, &[(0, 5), (6, 11), (11, 12)]), Provenance::by("test")).unwrap();
        assert_eq!(assert_invariants(&doc, "normalized"), 1);
        assert_eq!(doc.to_source(id, Span::new(6, 11)), Span::new(7, 12));
        assert_eq!(doc.to_source(id, Span::new(11, 12)), Span::new(12, 13));
    }

    #[test]
    fn a_view_whose_alignment_lies_is_caught() {
        // An alignment claiming the view is the source byte for byte,
        // while the view's text differs: the mapped source bytes are not
        // the span's own.
        let source = "abc";
        let mut doc = Document::new(source);
        let view = View::derived(doc.get::<View>().unwrap(), "xyz".to_string(), Alignment::identity());
        doc.add(&Key::<View>::named("lie"), view, Provenance::by("test")).unwrap();
        let id = doc.get_key(&Key::<View>::named("lie")).unwrap().id();
        doc.add(&Key::DEFAULT, tokens(id, &[(0, 3)]), Provenance::by("test")).unwrap();
        let (checked, v) = check_document(&doc);
        assert_eq!(checked, 1);
        assert_eq!(v.len(), 1);
        assert!(v[0].what.contains("touches no edit"), "{}", v[0]);
    }

    #[test]
    fn named_instances_of_several_types_are_all_checked() {
        let mut doc = Document::new("one two");
        doc.add(&Key::DEFAULT, tokens(ViewId::SOURCE, &[(0, 3), (4, 7)]), Provenance::by("test")).unwrap();
        doc.add(&Key::<Tokens>::named("other"), tokens(ViewId::SOURCE, &[(0, 7)]), Provenance::by("test")).unwrap();
        let mut s = Spans::new(ViewId::SOURCE);
        s.push(Span::new(0, 7));
        doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();
        assert_eq!(check_document(&doc), (3, Vec::new()));
    }
}
