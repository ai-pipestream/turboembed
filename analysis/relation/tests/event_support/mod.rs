//! Hand-parsed sentences for the event extractor tests: tokens split on
//! spaces, a CoNLL-style `head:relation` per token (1-based heads, `0`
//! for the root), and mentions of the anchoring layers over token
//! ranges.

#![allow(dead_code)]

use std::sync::Arc;

use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, TagSet, ViewId};
use turbo_analysis_layers::{
    Currency, Dependencies, Entities, Events, Money, MoneyMention, Origin, PERCENT, PosTags, Quantities,
    QuantityMention, Sentences, TemporalMention, TemporalValue, Temporals, Tokens,
};
use turbo_relation::{EventConfig, EventExtractor};

pub fn parsed(text: &str, conll: &str) -> Document {
    let mut spans = Spans::new(ViewId(0));
    let mut at = 0;
    for w in text.split(' ') {
        spans.push(Span::new(at as u32, (at + w.len()) as u32));
        at += w.len() + 1;
    }
    let n = spans.len();
    let rows: Vec<(Option<u32>, &str)> = conll
        .split(' ')
        .map(|r| {
            let (h, rel) = r.split_once(':').unwrap();
            let h: u32 = h.parse().unwrap();
            ((h > 0).then(|| h - 1), rel)
        })
        .collect();
    assert_eq!(rows.len(), n, "{text}");
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

/// The bytes of tokens `a..b`.
pub fn span(d: &Document, (a, b): (u32, u32)) -> Span {
    let t = d.require(&Key::<Tokens>::DEFAULT).unwrap();
    Span::new(t.spans.get(Ix::new(a)).start, t.spans.get(Ix::new(b - 1)).end)
}

pub fn add_pos(d: &mut Document, tags: &str) {
    let tags: Vec<&str> = tags.split(' ').collect();
    let mut labels = tags.clone();
    labels.sort_unstable();
    labels.dedup();
    let tagset = TagSet::new("ptb", &labels);
    let tags = tags.iter().map(|t| tagset.tag(t).unwrap()).collect();
    d.add(&Key::<PosTags>::DEFAULT, PosTags { tagset, tags, prob: None }, Provenance::by("test:pos")).unwrap();
}

pub fn add_temporals(d: &mut Document, rows: &[((u32, u32), TemporalValue)]) {
    let mut t = Temporals::new(ViewId(0));
    for &(r, value) in rows {
        t.push(TemporalMention { span: span(d, r), value, origin: Origin::Absolute });
    }
    d.add(&Key::<Temporals>::DEFAULT, t, Provenance::by("test:temporal")).unwrap();
}

pub fn add_money(d: &mut Document, rows: &[(u32, u32)]) {
    let mut m = Money::new(ViewId(0));
    for &r in rows {
        let currency = Currency::parse("USD").unwrap();
        m.push(MoneyMention { span: span(d, r), negative: false, coefficient: "1", scale: 0, currency });
    }
    d.add(&Key::<Money>::DEFAULT, m, Provenance::by("test:money")).unwrap();
}

pub fn add_quantities(d: &mut Document, rows: &[(u32, u32)]) {
    let mut q = Quantities::new(ViewId(0));
    for &r in rows {
        q.push(QuantityMention { span: span(d, r), negative: false, coefficient: "1", scale: 0, unit: PERCENT });
    }
    d.add(&Key::<Quantities>::DEFAULT, q, Provenance::by("test:quantities")).unwrap();
}

pub fn add_entities(d: &mut Document, rows: &[((u32, u32), &str)]) {
    let mut labels: Vec<&str> = rows.iter().map(|r| r.1).collect();
    labels.sort_unstable();
    labels.dedup();
    let kinds: Arc<TagSet> = TagSet::new("ner", &labels);
    let mut spans = Spans::new(ViewId(0));
    for &(r, _) in rows {
        spans.push(span(d, r));
    }
    let e = Entities {
        spans,
        tokens: rows.iter().map(|&((a, b), _)| IxRange::new(a, b)).collect(),
        kind: rows.iter().map(|r| kinds.tag(r.1).unwrap()).collect(),
        kinds,
        score: vec![1.0; rows.len()],
    };
    d.add(&Key::<Entities>::DEFAULT, e, Provenance::by("test:ner")).unwrap();
}

/// Runs the extractor on `d` and returns its events.
pub fn events(d: &Document, config: EventConfig) -> Events {
    EventExtractor::with_config(config).events(d).unwrap()
}

/// `(trigger, role, argument text, anchored layer)` rows.
pub fn rows(d: &Document, e: &Events) -> Vec<(String, String, String, String)> {
    let text = d.text(ViewId(0));
    let at = |s: Span| text[s.start as usize..s.end as usize].to_string();
    let mut out = Vec::new();
    for f in e.iter() {
        for a in f.arguments {
            out.push((at(f.trigger), format!("{:?}", a.role), at(a.span), format!("{:?}", a.anchored_layer)));
        }
    }
    out
}
