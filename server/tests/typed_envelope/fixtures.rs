//! Documents with chained, sibling and masked views for the envelope
//! tests: combining marks, emoji ZWJ sequences, a ligature expanded
//! across a mention boundary, insertion and deletion anchors and
//! unequal replacements.

#![allow(dead_code)]

use turbo_analysis::{Document, Edit, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis_layers::Tokens;

/// The native edits of view `id` on `doc`.
pub fn view_edits(doc: &Document, id: ViewId) -> Vec<Edit> {
    for key in doc.layers() {
        if key.ty == std::any::TypeId::of::<View>()
            && let Some(v) = doc.get_key(&Key::<View>::named(key.name))
            && v.id() == id
        {
            return v.alignment().edits().to_vec();
        }
    }
    panic!("no view {}", id.0)
}

fn view_of(doc: &Document, id: ViewId) -> View {
    for key in doc.layers() {
        if key.ty == std::any::TypeId::of::<View>()
            && let Some(v) = doc.get_key(&Key::<View>::named(key.name))
            && v.id() == id
        {
            return v.clone();
        }
    }
    panic!("no view {}", id.0)
}

/// Adds a child of `parent` built by `build` as the view `name`.
pub fn derive(doc: &mut Document, parent: ViewId, name: &'static str, build: impl FnOnce(&mut ViewBuilder<'_>)) {
    let parent_view = view_of(doc, parent);
    let mut b = ViewBuilder::new(parent_view.text());
    build(&mut b);
    let (text, alignment) = b.finish();
    doc.add(&Key::<View>::named(name), View::derived(&parent_view, text, alignment), Provenance::by("test:view"))
        .unwrap();
}

/// Replaces every scalar of `parent[start..end]` with one mask byte,
/// as a masked view does: one edit per character.
fn mask_each(b: &mut ViewBuilder<'_>, parent: &str, start: usize, end: usize) {
    b.keep(start);
    for (at, c) in parent[start..end].char_indices() {
        b.replace(start + at + c.len_utf8(), "*");
    }
}

fn tokens_on(doc: &mut Document, view: ViewId, name: &'static str, spans: &[(u32, u32)]) {
    let mut s = Spans::<Tokens>::new(view);
    for &(a, b) in spans {
        s.push(Span::new(a, b));
    }
    doc.add(&Key::<Tokens>::named(name), Tokens { spans: s, sentence: None }, Provenance::by("test:tokens")).unwrap();
}

/// "e" + combining acute, then NFC composes it; a grandchild masks it.
pub fn combining_marks() -> Document {
    let text = "cafe\u{301} au lait";
    let mut doc = Document::new(text);
    derive(&mut doc, ViewId::SOURCE, "nfc", |b| {
        b.keep(3);
        b.replace(3 + "e\u{301}".len(), "é");
    });
    derive(&mut doc, ViewId(1), "masked", |b| mask_each(b, "café au lait", 0, "café".len()));
    tokens_on(&mut doc, ViewId(1), "nfc", &[(0, "café".len() as u32)]);
    doc
}

/// A woman-technologist ZWJ sequence masked one scalar at a time, with
/// unequal widths (4, 3 and 4 bytes to 1), beside an unmasked emoji.
pub fn emoji_zwj_masked() -> Document {
    let text = "hi 👩\u{200d}💻 and 😀!";
    let mut doc = Document::new(text);
    let start = text.find('👩').unwrap();
    let end = start + "👩\u{200d}💻".len();
    derive(&mut doc, ViewId::SOURCE, "masked", |b| mask_each(b, text, start, end));
    tokens_on(&mut doc, ViewId(1), "masked", &[(3, 6), (11, 15)]);
    doc
}

/// The ligature "ﬁ" expanded to "fi", with a mention that starts inside
/// the expansion.
pub fn ligature_expansion() -> Document {
    let text = "the ﬁle ﬂew";
    let mut doc = Document::new(text);
    derive(&mut doc, ViewId::SOURCE, "expanded", |b| {
        let fi = text.find('ﬁ').unwrap();
        b.keep(fi);
        b.replace(fi + 'ﬁ'.len_utf8(), "fi");
        let fl = text.find('ﬂ').unwrap();
        b.keep(fl);
        b.replace(fl + 'ﬂ'.len_utf8(), "fl");
    });
    // "ile" of "file": starts one byte into the expansion.
    tokens_on(&mut doc, ViewId(1), "expanded", &[(5, 8)]);
    doc
}

/// An insertion, a deletion, and a child of that view deleting more.
pub fn insertion_and_deletion() -> Document {
    let text = "ab--cd";
    let mut doc = Document::new(text);
    derive(&mut doc, ViewId::SOURCE, "spaced", |b| {
        b.keep(1);
        b.replace(1, " ");
        b.keep(2);
        b.replace(4, "");
        b.keep(5);
        b.replace(5, "\u{200b}");
    });
    derive(&mut doc, ViewId(1), "tight", |b| {
        let t = "a b";
        b.keep(1);
        b.replace(t.len(), "");
    });
    doc
}

/// Replacements of unequal length both ways in one view.
pub fn unequal_replacements() -> Document {
    let text = "aaaa b cccccc";
    let mut doc = Document::new(text);
    derive(&mut doc, ViewId::SOURCE, "normalized", |b| {
        b.replace(4, "x");
        b.keep(5);
        b.replace(6, "yyyyy");
        b.keep(7);
        b.replace(13, "zz");
    });
    doc
}

/// Two children of the source and a grandchild of the first: mapping
/// between siblings goes through their lowest common ancestor.
pub fn sibling_views() -> Document {
    let text = "Hello, Wörld 😀";
    let mut doc = Document::new(text);
    derive(&mut doc, ViewId::SOURCE, "ascii", |b| {
        let o = text.find('ö').unwrap();
        b.keep(o);
        b.replace(o + 'ö'.len_utf8(), "o");
        let e = text.find('😀').unwrap();
        b.keep(e);
        b.replace(e + '😀'.len_utf8(), ":)");
    });
    derive(&mut doc, ViewId::SOURCE, "masked", |b| mask_each(b, text, 7, 7 + "Wörld".len()));
    derive(&mut doc, ViewId(1), "lower", |b| {
        b.replace(1, "h");
        let w = "Hello, World :)".find('W').unwrap();
        b.keep(w);
        b.replace(w + 1, "w");
    });
    tokens_on(&mut doc, ViewId(2), "masked", &[(7, 13)]);
    doc
}

pub fn fixture_documents() -> Vec<(&'static str, Document)> {
    vec![
        ("combining_marks", combining_marks()),
        ("emoji_zwj_masked", emoji_zwj_masked()),
        ("ligature_expansion", ligature_expansion()),
        ("insertion_and_deletion", insertion_and_deletion()),
        ("unequal_replacements", unequal_replacements()),
        ("sibling_views", sibling_views()),
    ]
}
