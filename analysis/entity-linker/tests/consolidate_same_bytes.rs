//! Two `Entities` rows on the same bytes (two taggers, or two kinds) land in
//! one document entity as two mentions: each row keeps its owner, so
//! `entity_of` finds both and a relation edge from either is lifted.

mod common;

use common::{spans_of, token_spans};
use turbo_analysis::{Document, Edge, Edges, Ix, IxRange, Key, Layer, Provenance, Span, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_entity_linker::EntityConsolidator;

const TEXT: &str = "Obama met Merkel .";

/// `Obama` twice on bytes 0..5, then `Merkel`, all PERSON; one edge from
/// the second `Obama` row to `Merkel`.
fn doc() -> Document {
    let mut doc = Document::new(TEXT);
    let view = ViewId(0);
    let toks = token_spans(TEXT);
    let whole = Sentences { spans: spans_of(view, [Span::new(0, TEXT.len() as u32)]) };
    doc.add(&Key::<Sentences>::DEFAULT, whole, Provenance::by("test:sentences")).unwrap();
    let tokens = Tokens { spans: spans_of(view, toks.clone()), sentence: Some(vec![Ix::new(0); toks.len()]) };
    doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test:tokens")).unwrap();
    let kinds = TagSet::new("ner", &["PERSON"]);
    let person = kinds.tag("PERSON").unwrap();
    let entities = Entities {
        spans: spans_of(view, [Span::new(0, 5), Span::new(0, 5), Span::new(10, 16)]),
        tokens: vec![IxRange::new(0, 1), IxRange::new(0, 1), IxRange::new(2, 3)],
        kind: vec![person; 3],
        kinds,
        score: vec![0.9; 3],
    };
    doc.add(&Key::<Entities>::DEFAULT, entities, Provenance::by("test:entities")).unwrap();
    let rel = TagSet::new("rel", &["met"]);
    let met = rel.tag("met").unwrap();
    let edges: Edges<Entities, Entities> = Edges::new(rel, [Edge::new(Ix::new(1), Ix::new(2), met, 0.7)]);
    doc.add(&Key::<Edges<Entities, Entities>>::DEFAULT, edges, Provenance::by("test:relations")).unwrap();
    doc
}

#[test]
fn two_rows_on_the_same_bytes_both_stay_mentions_with_an_owner() {
    let graph = EntityConsolidator::new().consolidate(&doc()).unwrap();
    graph.check().unwrap();
    assert_eq!(graph.entities.len(), 2);
    let obama = &graph.entities[0];
    assert_eq!(&*obama.canonical_name, "Obama");
    let rows: Vec<Option<u32>> = obama.mentions.iter().map(|m| m.entity.map(|r| r.get() as u32)).collect();
    assert_eq!(rows, [Some(0), Some(1)]);
    assert_eq!(obama.mentions.iter().filter(|m| m.is_representative).count(), 1);
    assert_eq!(graph.entity_of(Ix::new(0)), Some(0));
    assert_eq!(graph.entity_of(Ix::new(1)), Some(0));
    let lifted: Vec<(u32, u32, &str)> = graph.relations.iter().map(|r| (r.subject, r.object, &*r.kind)).collect();
    assert_eq!(lifted, [(0, 1, "met")]);
}
