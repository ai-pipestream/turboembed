//! The coreference layers across the document graph boundary: Mentions
//! (with fragments and entity links), Speakers, and the chains and links
//! over mentions round-trip through bytes, and corrupted payloads are
//! refused.

mod common;

use std::slice;

use common::Fixture;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Analyzer, Document, Edge, Edges, Groups, Ix, IxRange, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Entities, PosTags, Sentences, Tokens};
use turbo_coref::{
    Agreement, Animacy, CorefAnnotator, Gender, MentionKind, Mentions, Number, Origin, Person, Speakers, codecs,
    sieve_tags,
};

fn story() -> Fixture {
    Fixture::new()
        .sentence(&["Mary/NNP", "Jones/NNP", "founded/VBD", "Acme/NNP", "./."])
        .sentence(&["She/PRP", "praised/VBD", "it/PRP", "./."])
        .speaker(1, "narrator")
        .entity("Mary Jones", "person")
        .entity("Acme", "organization")
}

fn annotated() -> Document {
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .add(CorefAnnotator::default())
        .build()
        .unwrap();
    let mut doc = story().build().document();
    analyzer.run(slice::from_mut(&mut doc)).remove(0).unwrap();
    doc
}

fn round_trip(doc: &Document) -> Document {
    let graph = export(doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
    back
}

fn spans(m: &Mentions) -> Vec<Span> {
    m.spans.iter().map(|(_, s)| s).collect()
}

#[test]
fn annotator_output_round_trips() {
    let doc = annotated();
    let back = round_trip(&doc);
    let (a, b) = (doc.get::<Mentions>().unwrap(), back.get::<Mentions>().unwrap());
    assert_eq!(spans(a), spans(b));
    assert_eq!(a.kind, b.kind);
    assert_eq!(a.origin, b.origin);
    assert_eq!(a.agreement, b.agreement);
    assert_eq!(a.entity, b.entity);
    assert_eq!(a.head, b.head);
    let groups =
        |d: &Document| d.get::<Groups<Mentions>>().unwrap().iter().map(|(_, g)| g.to_vec()).collect::<Vec<_>>();
    assert_eq!(groups(&doc), groups(&back));
    let (ea, eb) = (doc.get::<Edges<Mentions, Mentions>>().unwrap(), back.get::<Edges<Mentions, Mentions>>().unwrap());
    assert_eq!((&ea.from, &ea.to, &ea.kind, &ea.score), (&eb.from, &eb.to, &eb.kind, &eb.score));
    assert_eq!(eb.kinds.name, sieve_tags().name);
    let speakers = back.get::<Speakers>().unwrap();
    assert_eq!(speakers.speakers.label(speakers.speaker[0]), "narrator");
}

fn find(text: &str, word: &str) -> Span {
    let at = text.find(word).unwrap();
    Span::new(at as u32, (at + word.len()) as u32)
}

/// A gold mention over "Mary" and "Acme" as one discontinuous mention,
/// and a second one over "it", linked into one chain.
fn gold_document() -> Document {
    let built = story().build();
    let text = built.text.clone();
    let mut doc = built.document();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.try_push_fragments(&[find(&text, "Mary"), find(&text, "Acme")]).unwrap();
    spans.push(find(&text, "it"));
    let agreement = Agreement::new(Number::Singular, Gender::Neutral, Animacy::Inanimate, Person::Third);
    let mentions = Mentions {
        spans,
        sentence: vec![Ix::new(0), Ix::new(1)],
        tokens: vec![IxRange::new(0, 4), IxRange::new(7, 8)],
        head: vec![Ix::new(3), Ix::new(7)],
        entity: None,
        kind: vec![MentionKind::Proper, MentionKind::Pronominal],
        origin: vec![Origin::Gold, Origin::Gold],
        agreement: vec![agreement, agreement],
    };
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, mentions, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, Groups::<Mentions>::new([vec![Ix::new(0), Ix::new(1)]]), by.clone()).unwrap();
    let tags = sieve_tags();
    let edges = Edges::<Mentions, Mentions>::new(
        tags.clone(),
        [Edge::new(Ix::new(1), Ix::new(0), tags.tag("exact").unwrap(), 1.0)],
    );
    doc.add(&Key::DEFAULT, edges, by).unwrap();
    doc
}

#[test]
fn discontinuous_gold_mentions_round_trip() {
    let doc = gold_document();
    let back = round_trip(&doc);
    let m = back.get::<Mentions>().unwrap();
    assert!(m.spans.is_discontinuous(Ix::new(0)));
    let text = back.source();
    let fragments: Vec<&str> = m.spans.fragments(Ix::new(0)).map(|s| &text[s.range()]).collect();
    assert_eq!(fragments, ["Mary", "Acme"]);
    assert_eq!(m.entity, None);
    assert_eq!(m.agreement[1].gender(), Gender::Neutral);
}

/// An edit of a layer payload.
type Edit = fn(&mut Vec<u8>);

fn splice(graph: &mut DocumentGraph, layer: &str, edit: impl FnOnce(&mut Vec<u8>)) {
    let at = graph.layers.iter().position(|l| l.layer == layer).unwrap();
    edit(graph.layers[at].body.as_mut().unwrap());
}

#[test]
fn corrupted_coreference_payloads_are_refused() {
    let base = export(&annotated(), &codecs()).unwrap();
    let cases: [(&str, Edit, &str); 5] = [
        ("opennlp:mentions", |b| b.truncate(1), "a truncated mentions payload"),
        ("opennlp:mentions", |b| *b.last_mut().unwrap() = 0b11, "an agreement byte no agreement packs to"),
        ("opennlp:mentions", |b| b.push(0), "trailing bytes after mentions"),
        ("opennlp:speakers", |b| b.truncate(1), "a truncated speakers payload"),
        ("opennlp:speakers", |b| b.push(0), "trailing bytes after speakers"),
    ];
    for (layer, edit, what) in cases {
        let mut graph = base.clone();
        splice(&mut graph, layer, edit);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }
}
