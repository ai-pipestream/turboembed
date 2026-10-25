//! Hand-parsed sentences for the modality tests: tokens split on spaces
//! and a CoNLL-style `head:relation` per token (1-based heads, `0` for
//! the root).

#![allow(dead_code)]

use turbo_analysis::{Document, Ix, Key, Provenance, Span, Spans, TagSet, ViewId};
use turbo_analysis_layers::modality::{DeonticKind, EpistemicStatus, Modality, ModalityTag, Polarity};
use turbo_analysis_layers::{Dependencies, Sentences, Tokens};
use turbo_relation::ModalityAnnotator;

/// A document with sentences and tokens only.
pub fn tokens_only(text: &str) -> Document {
    let mut spans = Spans::new(ViewId(0));
    let mut at = 0;
    for w in text.split(' ') {
        spans.push(Span::new(at as u32, (at + w.len()) as u32));
        at += w.len() + 1;
    }
    let n = spans.len();
    let mut d = Document::new(text);
    let mut sentence = Spans::new(ViewId(0));
    sentence.push(Span::new(0, text.len() as u32));
    d.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sentence }, Provenance::by("test:sentences")).unwrap();
    d.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans, sentence: Some(vec![Ix::new(0); n]) },
        Provenance::by("test:tokens"),
    )
    .unwrap();
    d
}

/// A document with tokens and the given dependencies.
pub fn parsed(text: &str, conll: &str) -> Document {
    let mut d = tokens_only(text);
    let rows: Vec<(Option<u32>, &str)> = conll
        .split(' ')
        .map(|r| {
            let (h, rel) = r.split_once(':').unwrap();
            let h: u32 = h.parse().unwrap();
            ((h > 0).then(|| h - 1), rel)
        })
        .collect();
    assert_eq!(rows.len(), text.split(' ').count(), "{text}");
    let mut labels: Vec<&str> = rows.iter().map(|r| r.1).collect();
    labels.sort_unstable();
    labels.dedup();
    let relations = TagSet::new("ud", &labels);
    let deps = Dependencies {
        head: rows.iter().map(|r| r.0.map(Ix::new)).collect(),
        relation: rows.iter().map(|r| relations.tag(r.1).unwrap()).collect(),
        relations,
    };
    d.add(&Key::<Dependencies>::DEFAULT, deps, Provenance::by("test:deps")).unwrap();
    d
}

/// The bytes of token `t`.
pub fn token(d: &Document, t: u32) -> Span {
    d.require(&Key::<Tokens>::DEFAULT).unwrap().spans.get(Ix::new(t))
}

/// The bytes of tokens `a..b`.
pub fn tokens(d: &Document, a: u32, b: u32) -> Span {
    Span::new(token(d, a).start, token(d, b - 1).end)
}

pub fn modality(d: &Document) -> Modality {
    ModalityAnnotator::new().modality(d).unwrap()
}

/// The tag over token `t`, or the unmarked default.
pub fn at(m: &Modality, d: &Document, t: u32) -> (Polarity, EpistemicStatus, DeonticKind) {
    let tag = m.tag_for_span(token(d, t)).unwrap_or_default();
    (tag.polarity, tag.epistemic, tag.deontic)
}

pub fn tag_at(m: &Modality, d: &Document, t: u32) -> Option<ModalityTag> {
    m.tag_for_span(token(d, t))
}
