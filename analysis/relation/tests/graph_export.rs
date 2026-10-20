//! The relations and their edges cross the document graph
//! boundary byte-identically, and corrupted payloads are refused.

use std::sync::Arc;

use turbo_analysis::export::{DocumentGraph, Writer, export, import};
use turbo_analysis::{
    Analyzer, Document, Edges, Ix, IxRange, Key, Layer, Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{Dependencies, Entities, Sentences, Tokens};
use turbo_relation::export::codecs;
use turbo_relation::{RelationAnnotator, RelationPattern, Relations};

fn spans<L>(ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(a, b) in ranges {
        out.push(Span::new(a, b));
    }
    out
}

fn document() -> Document {
    let mut doc = Document::new("Acme Corp acquired Bolt in 2024.");
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 32)]) }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Tokens {
            spans: spans(&[(0, 4), (5, 9), (10, 18), (19, 23), (24, 26), (27, 31), (31, 32)]),
            sentence: Some(vec![Ix::new(0); 7]),
        },
        by.clone(),
    )
    .unwrap();
    let labels = ["compound", "nsubj", "root", "obj", "case", "obl", "punct"];
    let tag = |l: &str| Tag(labels.iter().position(|x| *x == l).unwrap() as u16);
    doc.add(
        &Key::DEFAULT,
        Dependencies::<Tokens> {
            head: vec![
                Some(Ix::new(1)),
                Some(Ix::new(2)),
                None,
                Some(Ix::new(2)),
                Some(Ix::new(5)),
                Some(Ix::new(2)),
                Some(Ix::new(2)),
            ],
            relation: vec![
                tag("compound"),
                tag("nsubj"),
                tag("root"),
                tag("obj"),
                tag("case"),
                tag("obl"),
                tag("punct"),
            ],
            relations: TagSet::new("dependency", &labels),
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Entities {
            spans: spans(&[(0, 9), (19, 23), (27, 31)]),
            tokens: vec![IxRange::new(0, 2), IxRange::new(3, 4), IxRange::new(5, 6)],
            kind: vec![Tag(0), Tag(0), Tag(1)],
            kinds: TagSet::new("entity", &["organization", "date"]),
            score: vec![1.0; 3],
        },
        by,
    )
    .unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .add(
            RelationAnnotator::<Tokens>::new(vec![
                RelationPattern::with_trigger("acquisition", "<nsubj >obj", "acquired").unwrap(),
                RelationPattern::with_trigger("acquisition", "<nsubj >obj", "acquired").unwrap(),
            ])
            .unwrap(),
        )
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    docs.pop().unwrap()
}

#[test]
fn relations_and_edges_round_trip() {
    let doc = document();
    let registry = codecs();
    let graph = export(&doc, &registry).unwrap();
    let back = import(&graph, &registry).unwrap();
    let relations = back.get::<Relations>().unwrap();
    // The duplicate registration survives as two mentions.
    assert_eq!(relations.spans.len(), 2);
    let (s, o, kind, _) = relations.mention(Ix::new(0));
    assert_eq!((s.get(), o.get(), kind), (0, 1, "acquisition"));
    assert_eq!(back.get::<Edges<Entities, Entities>>().unwrap().len(), 1);
    let again = export(&back, &registry).unwrap();
    assert_eq!(again.to_bytes(), graph.to_bytes(), "re-export is byte-identical");
}

fn splice(graph: &mut DocumentGraph, body: Vec<u8>, len: u32) {
    let at = graph.layers.iter().position(|l| l.layer == "opennlp:relations").unwrap();
    graph.layers[at].body = Some(body);
    graph.layers[at].len = len;
}

/// A relations body by hand: one span, a one-label tag set, then
/// the given endpoint pair.
fn body(subject: u32, object: u32) -> Vec<u8> {
    let mut w = Writer::new();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.usize(1).unwrap();
    w.span(Span::new(0, 23));
    w.str("relation").unwrap();
    w.usize(1).unwrap();
    w.str("acquisition").unwrap();
    w.usize(1).unwrap();
    w.u32(subject);
    w.u32(object);
    w.u16(0);
    w.f64(1.0);
    w.finish()
}

#[test]
fn corrupted_relations_payloads_are_refused() {
    let registry = codecs();
    let base = export(&document(), &registry).unwrap();
    // The hand-built control imports.
    let mut graph = base.clone();
    splice(&mut graph, body(0, 1), 1);
    assert!(import(&graph, &registry).is_ok(), "the hand-built relations control was refused");
    // An endpoint past the entities.
    let mut graph = base.clone();
    splice(&mut graph, body(0, 99), 1);
    assert!(import(&graph, &registry).is_err(), "an endpoint past the entities imported");
    // Subject equal to object.
    let mut graph = base.clone();
    splice(&mut graph, body(1, 1), 1);
    assert!(import(&graph, &registry).is_err(), "a self relation imported");
    // A truncated payload.
    let mut graph = base;
    let mut truncated = body(0, 1);
    truncated.truncate(6);
    splice(&mut graph, truncated, 1);
    assert!(import(&graph, &registry).is_err(), "a truncated payload imported");
    let _ = Arc::<()>::new(());
}

#[test]
fn out_of_bounds_relation_spans_do_not_survive_import() {
    // A hand-built body whose covering span runs past the text is
    // refused on import (the document insertion behind import
    // validates the exposed text spans).
    let registry = codecs();
    let mut graph = export(&document(), &registry).unwrap();
    let mut w = Writer::new();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.usize(1).unwrap();
    w.span(Span::new(0, 99));
    w.str("relation").unwrap();
    w.usize(1).unwrap();
    w.str("acquisition").unwrap();
    w.usize(1).unwrap();
    w.u32(0);
    w.u32(1);
    w.u16(0);
    w.f64(1.0);
    splice(&mut graph, w.finish(), 1);
    assert!(import(&graph, &registry).is_err(), "an out-of-bounds relation span imported");
}
