//! The stopword flags across the document graph boundary: they
//! round-trip through bytes aligned to their tokens, and a flag
//! column that does not match the token count is refused.

use turbo_analysis::export::{Codecs, DocumentGraph, Writer, export, import};
use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_stopword_layers::export::register_codecs;
use turbo_stopword_layers::{StopwordLanguage, Stopwords};

fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    register_codecs(&mut codecs);
    codecs
}

fn tokens(pairs: &[(u32, u32)]) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        spans.push(Span::new(a, b));
    }
    Tokens { spans, sentence: None }
}

fn document() -> Document {
    let mut doc = Document::new("the cat sat");
    doc.add(&Key::DEFAULT, tokens(&[(0, 3), (4, 7), (8, 11)]), Provenance::by("test")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Stopwords { language: StopwordLanguage::English, stop: vec![true, false, true] },
        Provenance::by("test"),
    )
    .unwrap();
    doc
}

fn splice(graph: &mut DocumentGraph, body: Vec<u8>, len: u32) {
    let at = graph.layers.iter().position(|l| l.layer == "turbo:stopwords").unwrap();
    graph.layers[at].body = Some(body);
    graph.layers[at].len = len;
}

fn body(language: u8, stop: &[u8]) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(language);
    w.usize(stop.len()).unwrap();
    for &s in stop {
        w.u8(s);
    }
    w.finish()
}

#[test]
fn stopwords_round_trip() {
    let graph = export(&document(), &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(
        back.get::<Stopwords>().unwrap(),
        &Stopwords { language: StopwordLanguage::English, stop: vec![true, false, true] }
    );
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn stopwords_align_to_a_named_tokens_instance() {
    let mut doc = Document::new("the cat sat");
    doc.add(&Key::DEFAULT, tokens(&[(0, 3), (4, 7), (8, 11)]), Provenance::by("test")).unwrap();
    doc.add(&Key::named("alt"), tokens(&[(0, 7), (8, 11)]), Provenance::by("test")).unwrap();
    let mut provenance = Provenance::by("test");
    provenance.reads = vec![Key::<Tokens>::named("alt").id()];
    doc.add(&Key::DEFAULT, Stopwords { language: StopwordLanguage::German, stop: vec![false, true] }, provenance)
        .unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(back.get::<Stopwords>().unwrap().stop, [false, true]);
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn corrupted_stopwords_are_refused() {
    let base = export(&document(), &codecs()).unwrap();
    for (what, payload, len) in [
        ("unknown language", body(11, &[1, 0, 1]), 3),
        ("a flag tag that is not a boolean", body(3, &[1, 2, 1]), 3),
        ("fewer flags than tokens", body(3, &[1, 0]), 2),
        ("more flags than tokens", body(3, &[1, 0, 1, 0]), 4),
        ("truncated", vec![3], 0),
    ] {
        let mut graph = base.clone();
        splice(&mut graph, payload, len);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }
}
