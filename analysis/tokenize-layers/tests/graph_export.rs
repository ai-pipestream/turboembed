//! Word types and dictionary readings across the document graph
//! boundary: they round-trip through bytes aligned to their tokens,
//! and columns that do not match the token count are refused.

use turbo_analysis::export::{Codecs, DocumentGraph, Writer, export, import};
use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Strs, Tokens};
use turbo_tokenize_layers::export::register_codecs;
use turbo_tokenize_layers::{Readings, WordType, WordTypes};

fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    register_codecs(&mut codecs);
    codecs
}

fn strs(items: &[&str]) -> Strs {
    let mut s = Strs::default();
    for i in items {
        s.push(i);
    }
    s
}

fn document() -> Document {
    let mut doc = Document::new("cat 42 go");
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 3), (4, 6), (7, 9)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    doc.add(
        &Key::DEFAULT,
        WordTypes { kind: vec![WordType::Alphanumeric, WordType::Numeric, WordType::Alphanumeric] },
        Provenance::by("test"),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Readings {
            reading: Some(strs(&["キャット", "", "ゴー"])),
            pronunciation: None,
            lemma: Some(strs(&["cat", "42", "go"])),
        },
        Provenance::by("test"),
    )
    .unwrap();
    doc
}

fn splice(graph: &mut DocumentGraph, layer: &str, body: Vec<u8>, len: u32) {
    let at = graph.layers.iter().position(|l| l.layer == layer).unwrap();
    graph.layers[at].body = Some(body);
    graph.layers[at].len = len;
}

#[test]
fn word_types_and_readings_round_trip() {
    let graph = export(&document(), &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(
        back.get::<WordTypes>().unwrap().kind,
        [WordType::Alphanumeric, WordType::Numeric, WordType::Alphanumeric]
    );
    let readings = back.get::<Readings>().unwrap();
    assert_eq!(readings.reading.as_ref().unwrap().get(0), "キャット");
    assert_eq!(readings.lemma.as_ref().unwrap().get(2), "go");
    assert!(readings.pronunciation.is_none());
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn corrupted_word_types_are_refused() {
    let base = export(&document(), &codecs()).unwrap();
    let kinds = |values: &[u8]| {
        let mut w = Writer::new();
        w.usize(values.len()).unwrap();
        for &v in values {
            w.u8(v);
        }
        w.finish()
    };
    for (what, payload, len) in [
        ("unknown word type", kinds(&[0, 8, 0]), 3),
        ("fewer types than tokens", kinds(&[0, 1]), 2),
        ("more types than tokens", kinds(&[0, 1, 0, 0]), 4),
        ("truncated", Vec::new(), 0),
    ] {
        let mut graph = base.clone();
        splice(&mut graph, "turbo:word-types", payload, len);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }
}

#[test]
fn corrupted_readings_are_refused() {
    let base = export(&document(), &codecs()).unwrap();
    let column = |w: &mut Writer, values: Option<&[&str]>| match values {
        None => w.u8(0),
        Some(v) => {
            w.u8(1);
            w.usize(v.len()).unwrap();
            for s in v {
                w.str(s).unwrap();
            }
        }
    };
    let readings = |reading: Option<&[&str]>, lemma: Option<&[&str]>| {
        let mut w = Writer::new();
        column(&mut w, reading);
        column(&mut w, None);
        column(&mut w, lemma);
        w.finish()
    };
    for (what, payload) in [
        ("no column at all", readings(None, None)),
        ("columns of different lengths", readings(Some(&["a", "b"]), Some(&["x"]))),
        ("columns shorter than the tokens", readings(Some(&["a", "b"]), Some(&["x", "y"]))),
        ("an optional column tag of 2", vec![2]),
    ] {
        let mut graph = base.clone();
        splice(&mut graph, "turbo:readings", payload, 0);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }
}
