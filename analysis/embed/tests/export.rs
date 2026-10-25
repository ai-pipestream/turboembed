//! The QA and rerank layers cross the document graph boundary, and a
//! corrupt payload is refused instead of restored.
use turbo_analysis::export::{Codecs, DocumentGraph, ExportError, export, import};
use turbo_analysis::{Document, Ix, Key, Layer, ModelRef, Provenance, Span, Spans, View};
use turbo_analysis_embed::{Answers, Question, Reranked, Shortlist};
use turbo_analysis_layers::Sentences;

fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    turbo_analysis_embed::register_codecs(&mut codecs);
    codecs.register::<Shortlist<Sentences>>().register::<Reranked<Sentences>>();
    codecs
}

fn document() -> Document {
    let mut doc = Document::new("Paris. Rome. Bonn.");
    let view = doc.get::<View>().unwrap().id();
    doc.add(&Key::DEFAULT, Question("Where is Paris?".into()), Provenance::by("test")).unwrap();
    let mut spans = Spans::new(view);
    spans.push(Span::new(0, 5));
    spans.push(Span::new(7, 11));
    let answers = Answers {
        spans,
        scores: vec![2.5, -0.25],
        probabilities: vec![0.75, 0.125],
        null_score: -1.0,
        null_probability: 0.125,
    };
    doc.add(&Key::DEFAULT, answers, Provenance::by("test")).unwrap();
    let mut spans = Spans::new(view);
    for (a, b) in [(0, 6), (7, 12), (13, 18)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    let mut p = Provenance::by("test:shortlist");
    p.reads = vec![Key::<Sentences>::DEFAULT.id()];
    let shortlist =
        Shortlist::<Sentences> { query: "France".into(), source: Key::DEFAULT, items: vec![Ix::new(2), Ix::new(1)] };
    doc.add(&Key::DEFAULT, shortlist, p).unwrap();
    let mut p = Provenance::by("test:rerank");
    p.reads = vec![Key::<Sentences>::DEFAULT.id()];
    let reranked = Reranked::<Sentences> {
        source: Key::DEFAULT,
        items: vec![Ix::new(1), Ix::new(2)],
        scores: vec![0.9, -0.5],
        model: ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() },
    };
    doc.add(&Key::DEFAULT, reranked, p).unwrap();
    doc
}

#[test]
fn qa_and_rerank_layers_round_trip() {
    let codecs = codecs();
    let graph = export(&document(), &codecs).unwrap();
    let graph = DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap();
    let back = import(&graph, &codecs).unwrap();
    assert_eq!(back.get::<Question>().unwrap().0, "Where is Paris?");
    let a = back.get::<Answers>().unwrap();
    assert_eq!(a.spans.columns(), (&[0u32, 7][..], &[5u32, 11][..]));
    assert_eq!((a.scores.clone(), a.probabilities.clone()), (vec![2.5, -0.25], vec![0.75, 0.125]));
    assert_eq!((a.null_score, a.null_probability), (-1.0, 0.125));
    let s = back.get::<Shortlist<Sentences>>().unwrap();
    assert_eq!((s.query.as_str(), s.items.clone()), ("France", vec![Ix::new(2), Ix::new(1)]));
    let r = back.get::<Reranked<Sentences>>().unwrap();
    assert_eq!((r.items.clone(), r.scores.clone()), (vec![Ix::new(1), Ix::new(2)], vec![0.9, -0.5]));
    assert_eq!(r.model.sha256, "s");
    assert_eq!(export(&back, &codecs).unwrap(), graph);
}

/// The record name a layer type is exported under.
fn record(id: &str) -> String {
    let graph = export(&document(), &codecs()).unwrap();
    let names: Vec<_> = graph.layers.iter().map(|r| r.layer.clone()).collect();
    names.iter().find(|n| n.starts_with(id)).unwrap_or_else(|| panic!("{id} in {names:?}")).clone()
}

fn refused(id: &str, edit: impl Fn(&mut Vec<u8>)) -> ExportError {
    let codecs = codecs();
    let layer = record(id);
    let mut graph = export(&document(), &codecs).unwrap();
    let row = graph.layers.iter_mut().find(|r| r.layer == layer).unwrap();
    edit(row.body.as_mut().unwrap());
    import(&graph, &codecs).err().unwrap_or_else(|| panic!("{layer}: corrupt payload imported"))
}

fn replace(body: &mut [u8], from: &[u8], to: &[u8]) {
    let i = body.windows(from.len()).rposition(|w| w == from).unwrap();
    body[i..i + to.len()].copy_from_slice(to);
}

#[test]
fn corrupt_qa_and_rerank_payloads_are_refused() {
    for id in [Question::ID, Answers::ID, "turbo:rerank-shortlist", "turbo:reranked"] {
        refused(id, |b| b.push(0));
        refused(id, |b| {
            b.pop();
        });
    }
    // A probability that no longer sums to one with the null.
    refused(Answers::ID, |b| replace(b, &0.75f64.to_le_bytes(), &0.5f64.to_le_bytes()));
    // A shortlist item past the end of the levels.
    let e = refused("turbo:rerank-shortlist", |b| replace(b, &1u32.to_le_bytes(), &3u32.to_le_bytes()));
    assert!(matches!(e, ExportError::DanglingEndpoint { .. }), "{e:?}");
    // A duplicate shortlist item.
    refused("turbo:rerank-shortlist", |b| replace(b, &1u32.to_le_bytes(), &2u32.to_le_bytes()));
    // A reranked item past the end of the levels.
    let e = refused("turbo:reranked", |b| replace(b, &2u32.to_le_bytes(), &7u32.to_le_bytes()));
    assert!(matches!(e, ExportError::DanglingEndpoint { .. }), "{e:?}");
    // Unsorted and non-finite reranked scores.
    for bad in [2.0f32, f32::NAN] {
        refused("turbo:reranked", |b| replace(b, &(-0.5f32).to_bits().to_le_bytes(), &bad.to_bits().to_le_bytes()));
    }
}
