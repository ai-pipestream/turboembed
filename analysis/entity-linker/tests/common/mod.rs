//! Documents with whitespace tokens, one sentence and named entities, for
//! the consolidation tests.

#![allow(dead_code)]

use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Sentences, Tokens};

pub fn spans_of<L>(view: ViewId, items: impl IntoIterator<Item = Span>) -> Spans<L> {
    let mut s = Spans::new(view);
    for sp in items {
        s.push(sp);
    }
    s
}

/// The whitespace-separated tokens of `text`.
pub fn token_spans(text: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut at = None;
    for (i, ch) in text.char_indices() {
        if !ch.is_whitespace() {
            at.get_or_insert(i);
        } else if let Some(start) = at.take() {
            out.push(Span::new(start as u32, i as u32));
        }
    }
    if let Some(start) = at {
        out.push(Span::new(start as u32, text.len() as u32));
    }
    out
}

/// `text` with its tokens, one sentence, and `names` (surface, kind) found
/// left to right as `Entities`.
pub fn doc_with_names(text: &str, names: &[(&str, &str)]) -> Document {
    let mut doc = Document::new(text);
    let view = ViewId(0);
    let toks = token_spans(text);
    doc.add(
        &Key::<Sentences>::DEFAULT,
        Sentences { spans: spans_of(view, [Span::new(0, text.len() as u32)]) },
        Provenance::by("test:sentences"),
    )
    .unwrap();
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: spans_of(view, toks.clone()), sentence: Some(vec![Ix::new(0); toks.len()]) },
        Provenance::by("test:tokens"),
    )
    .unwrap();
    let mut labels: Vec<&str> = names.iter().map(|(_, k)| *k).collect();
    labels.sort_unstable();
    labels.dedup();
    let kinds = TagSet::new("ner", &labels);
    let (mut spans, mut tokens, mut kind) = (Vec::new(), Vec::new(), Vec::new());
    let mut from = 0usize;
    for &(surface, k) in names {
        let start = from + text[from..].find(surface).expect("surface in text");
        let end = start + surface.len();
        from = end;
        let first = toks.iter().position(|t| t.end as usize > start && (t.start as usize) < end).unwrap();
        let last = toks.iter().rposition(|t| t.end as usize > start && (t.start as usize) < end).unwrap();
        spans.push(Span::new(start as u32, end as u32));
        tokens.push(IxRange::new(first as u32, last as u32 + 1));
        kind.push(kinds.tag(k).unwrap());
    }
    let score = vec![0.9; spans.len()];
    doc.add(
        &Key::<Entities>::DEFAULT,
        Entities { spans: spans_of(view, spans), tokens, kind, kinds, score },
        Provenance::by("test:entities"),
    )
    .unwrap();
    doc
}
