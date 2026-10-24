//! Term weights across the document graph boundary: weights over the
//! chunks, the level units and the whole view round-trip through bytes
//! bit for bit, and corrupted payloads are refused without panicking.

use std::marker::PhantomData;

use turbo_analysis::export::{Codecs, DocumentGraph, Writer, export, import};
use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Levels, Sentences, Strs, Structure, Unit};
use turbo_chunker::export::register_codecs;
use turbo_chunker::terms::{Bm25, CollectionStatisticsBuilder, TermStats, Vocabulary, WeightedTerms};

fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    register_codecs(&mut codecs);
    codecs
}

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

/// "good food. bad food." as two chunks, its levels, its terms, and BM25
/// weights at the chunk and view levels; the level weights are set by
/// hand, one per unit.
fn document() -> Document {
    let mut doc = Document::new("good food. bad food.");
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 10), (11, 20)]) }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Chunks {
            spans: spans(&[(0, 10), (11, 20)]),
            level: vec![0, 0],
            parent: vec![None, None],
            sentences: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
            cut: vec![Cut::DocumentStart, Cut::Sentence],
            cost: vec![0, 5],
            model_tokens: vec![3, 3],
            over_budget: vec![false, false],
            prompt_overhead: 0,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Structure { spans: spans(&[]), kind: Vec::new(), parent: Vec::new(), depth: Vec::new() },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Levels {
            spans: spans(&[(0, 20), (0, 10), (11, 20)]),
            unit: vec![Unit::Document, Unit::Chunk(Ix::new(0)), Unit::Chunk(Ix::new(1))],
            parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
        },
        by.clone(),
    )
    .unwrap();
    let mut text = Strs::default();
    for t in ["bad", "food", "good"] {
        text.push(t);
    }
    let vocabulary = Vocabulary { text };
    let chunks = TermStats::<Chunks> {
        row: vec![0, 2, 4],
        term: vec![Ix::new(1), Ix::new(2), Ix::new(0), Ix::new(1)],
        tf: vec![1, 1, 1, 1],
        length: vec![2, 2],
        df: vec![1, 2, 1],
        _of: PhantomData,
    };
    let view = TermStats::<View> {
        row: vec![0, 3],
        term: vec![Ix::new(0), Ix::new(1), Ix::new(2)],
        tf: vec![1, 2, 1],
        length: vec![4],
        df: vec![1, 1, 1],
        _of: PhantomData,
    };
    let mut b = CollectionStatisticsBuilder::new();
    b.add(&chunks, &vocabulary).unwrap();
    let collection = b.build();
    let chunk_weights = WeightedTerms::bm25(&chunks, &vocabulary, &collection, Bm25::LUCENE).unwrap();
    let view_weights = WeightedTerms::bm25(&view, &vocabulary, &collection, Bm25::LUCENE).unwrap();
    let level_weights =
        WeightedTerms::<Levels> { row: vec![0, 1, 2, 3], weight: vec![0.25, -1.5, 1e-300], _of: PhantomData };
    doc.add(&Key::DEFAULT, vocabulary, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, chunks, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, view, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, chunk_weights, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, view_weights, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, level_weights, by).unwrap();
    doc
}

fn bits(w: &[f64]) -> Vec<u64> {
    w.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn weighted_terms_round_trip_bit_for_bit() {
    let doc = document();
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();

    let (a, b) = (doc.get::<WeightedTerms<Chunks>>().unwrap(), back.get::<WeightedTerms<Chunks>>().unwrap());
    assert_eq!((&a.row, bits(&a.weight)), (&b.row, bits(&b.weight)));
    assert_eq!(b.unit(1).len(), 2);
    assert!(b.weight.iter().all(|w| *w > 0.0));
    let (a, b) = (doc.get::<WeightedTerms<View>>().unwrap(), back.get::<WeightedTerms<View>>().unwrap());
    assert_eq!((&a.row, bits(&a.weight)), (&b.row, bits(&b.weight)));
    let b = back.get::<WeightedTerms<Levels>>().unwrap();
    assert_eq!((b.row.as_slice(), bits(&b.weight)), (&[0, 1, 2, 3][..], bits(&[0.25, -1.5, 1e-300])));

    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

fn splice(graph: &mut DocumentGraph, layer: &str, body: Vec<u8>, len: u32) {
    let at = graph.layers.iter().position(|l| l.layer == layer).unwrap();
    graph.layers[at].body = Some(body);
    graph.layers[at].len = len;
}

fn body(row: &[u32], weight: &[f64]) -> Vec<u8> {
    let mut w = Writer::new();
    w.usize(row.len()).unwrap();
    for &r in row {
        w.u32(r);
    }
    w.usize(weight.len()).unwrap();
    for &x in weight {
        w.f64(x);
    }
    w.finish()
}

#[test]
fn corrupted_weighted_terms_are_refused() {
    let base = export(&document(), &codecs()).unwrap();
    let chunks = "turbo:weighted-terms<turbo:chunks>";
    let cases = [
        ("fewer row starts than chunks", body(&[0, 4], &[1.0; 4]), 1),
        ("more row starts than chunks", body(&[0, 1, 2, 4], &[1.0; 4]), 3),
        ("rows that do not start at zero", body(&[1, 2, 4], &[1.0; 4]), 2),
        ("rows that end short of the weights", body(&[0, 2, 3], &[1.0; 4]), 2),
        ("rows out of order", body(&[0, 3, 2], &[1.0; 2]), 2),
        ("a NaN weight", body(&[0, 2, 4], &[1.0, f64::NAN, 1.0, 1.0]), 2),
        ("an infinite weight", body(&[0, 2, 4], &[1.0, 1.0, f64::INFINITY, 1.0]), 2),
    ];
    for (what, payload, len) in cases {
        let mut graph = base.clone();
        splice(&mut graph, chunks, payload, len);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }

    // Trailing bytes, a count the payload cannot back, and an empty body.
    let mut trailing = body(&[0, 2, 4], &[1.0; 4]);
    trailing.push(0);
    let mut huge = Writer::new();
    huge.usize(u32::MAX as usize).unwrap();
    for (what, payload) in [("trailing bytes", trailing), ("a billion rows from four bytes", huge.finish())] {
        let mut graph = base.clone();
        splice(&mut graph, chunks, payload, 2);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }
    let mut graph = base.clone();
    splice(&mut graph, chunks, Vec::new(), 2);
    assert!(import(&graph, &codecs()).is_err(), "an empty payload imported");

    // Level weights must cover the three level units, not the two chunks.
    let mut graph = base.clone();
    splice(&mut graph, "turbo:weighted-terms<turbo:levels>", body(&[0, 1, 2], &[1.0, 1.0]), 2);
    assert!(import(&graph, &codecs()).is_err(), "level weights short of the level units imported");

    // The valid control behind the cases imports.
    let mut graph = base.clone();
    splice(&mut graph, chunks, body(&[0, 2, 4], &[1.0, -2.0, 0.0, 3.5]), 2);
    let back = import(&graph, &codecs()).unwrap();
    assert_eq!(back.get::<WeightedTerms<Chunks>>().unwrap().unit(1), [0.0, 3.5]);
}
