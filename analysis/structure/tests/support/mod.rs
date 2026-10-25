//! Hand-labelled documents: each one's whole structure, written out as an
//! outline, one block per line, indented under its parent, with its kind,
//! its depth and the exact text its span covers. Every structure is also
//! added to a document, so the layer's own check (a tree in text order)
//! and the span checks (in bounds, on character boundaries) run on it.

use turbo_analysis::{Document, Key, Provenance, ViewId};
use turbo_analysis_layers::Structure;
use turbo_structure::{Syntax, detect};

pub fn outline(text: &str, syntax: Syntax) -> String {
    let s = detect(text, syntax, ViewId::SOURCE);
    let mut level = vec![0usize; s.spans.len()];
    let mut out = String::new();
    for (i, span) in s.spans.iter() {
        let i = i.get();
        level[i] = s.parent[i].map_or(0, |p| level[p.get()] + 1);
        let pad = "  ".repeat(level[i]);
        out.push_str(&format!("{pad}{:?} {} {:?}\n", s.kind[i], s.depth[i], &text[span.range()]));
    }
    let mut doc = Document::new(text);
    doc.add(&Key::<Structure>::DEFAULT, s, Provenance::by("test")).unwrap();
    out
}

#[track_caller]
pub fn check(text: &str, syntax: Syntax, want: &str) {
    let want: String = want.lines().filter(|l| !l.trim().is_empty()).map(|l| format!("{}\n", &l[4..])).collect();
    assert_eq!(outline(text, syntax), want, "\n{}", outline(text, syntax));
}

pub const MD: Syntax = Syntax::Markdown;
