//! The chunker's layers across the document graph boundary: a
//! fully populated document round-trips through bytes, and wire
//! frames with indexes outside the vocabulary or the aspects,
//! unsorted vocabularies, mismatched unit counts, bad scores and
//! invalid enum bytes are refused without panicking.

use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::export::{Codecs, DocumentGraph, Writer, export, import};
use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{
    Aspects, Chunks, Cut, Incivility, IncivilityLabel, IncivilityScores, IncivilityTarget, Opinions, Polarity,
    Sentences, Strs, Tokens,
};
use turbo_chunker::ChunkFacets;
use turbo_chunker::export::register_codecs;
use turbo_chunker::terms::{IndexedTerms, TermStats, Vocabulary};

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

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

fn document() -> Document {
    let mut doc = Document::new("good food. bad food.");
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 10), (11, 20)]) }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Tokens {
            spans: spans(&[(0, 4), (5, 9), (9, 10), (11, 14), (15, 19), (19, 20)]),
            sentence: Some(vec![Ix::new(0), Ix::new(0), Ix::new(0), Ix::new(1), Ix::new(1), Ix::new(1)]),
        },
        by.clone(),
    )
    .unwrap();
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
        },
        by.clone(),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Vocabulary { text: strs(&["bad", "food", "good"]) }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        IndexedTerms { term: vec![Some(Ix::new(2)), Some(Ix::new(1)), None, Some(Ix::new(0)), Some(Ix::new(1)), None] },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        TermStats::<Chunks> {
            row: vec![0, 2, 4],
            term: vec![Ix::new(1), Ix::new(2), Ix::new(0), Ix::new(1)],
            tf: vec![1, 1, 1, 1],
            length: vec![2, 2],
            df: vec![1, 2, 1],
            _of: PhantomData,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        TermStats::<View> {
            row: vec![0, 3],
            term: vec![Ix::new(0), Ix::new(1), Ix::new(2)],
            tf: vec![1, 2, 1],
            length: vec![4],
            df: vec![1, 1, 1],
            _of: PhantomData,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Opinions {
            tokens: vec![IxRange::new(0, 1), IxRange::new(3, 4)],
            polarity: vec![0.8, -0.4],
            intensity: vec![1.0, 2.0],
            negated: vec![false, true],
            tokens_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Aspects {
            tokens: vec![Some(IxRange::new(0, 1)), None],
            sentence: vec![Some(Ix::new(0)), Some(Ix::new(1))],
            category: vec![Some(Tag(0)), None],
            categories: Arc::clone(&TagSet::new("aspect-cats", &["food", "service"])),
            positive: vec![0.5, 0.1],
            negative: vec![0.2, 0.6],
            neutral: vec![0.3, 0.3],
            opinions: vec![Ix::new(0)],
            opinion_end: vec![1, 1],
            tokens_key: Key::DEFAULT,
            sentences_key: Key::DEFAULT,
            opinions_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Incivility {
            target: vec![IncivilityTarget::Sentence(Ix::new(0))],
            insult: vec![0.9],
            profanity: vec![0.0],
            threat: vec![0.1],
            identity_attack: vec![0.0],
            intolerance: vec![0.2],
            severe_toxicity: vec![0.3],
            evidence_label: vec![IncivilityLabel::Insult],
            evidence_tokens: vec![IxRange::new(0, 1)],
            evidence_end: vec![1],
            tokens_key: Key::DEFAULT,
            sentences_key: Key::DEFAULT,
            chunks_key: Key::DEFAULT,
            structure_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        ChunkFacets {
            polarity: vec![Polarity::Positive, Polarity::Negative],
            polarity_score: vec![0.8, -0.6],
            aspect_row: vec![0, 1, 2],
            aspect: vec![Ix::new(0), Ix::new(1)],
            aspect_label: strs(&["food", "service"]),
            aspect_polarity: vec![Polarity::Positive, Polarity::Negative],
            incivility_row: vec![0, 1, 1],
            incivility_label: vec![IncivilityLabel::Insult],
            incivility_max: vec![
                IncivilityScores {
                    insult: 0.9,
                    threat: 0.1,
                    intolerance: 0.2,
                    severe_toxicity: 0.3,
                    ..Default::default()
                },
                IncivilityScores::default(),
            ],
        },
        by,
    )
    .unwrap();
    doc
}

#[test]
fn chunker_layers_round_trip() {
    let graph = export(&document(), &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();

    let vocab = back.get::<Vocabulary>().unwrap();
    assert_eq!(vocab.lookup("food"), Some(Ix::new(1)));
    let indexed = back.get::<IndexedTerms>().unwrap();
    assert_eq!(indexed.term[3].map(|ix| vocab.text.get(ix.get())), Some("bad"));
    let stats = back.get::<TermStats<Chunks>>().unwrap();
    assert_eq!(stats.tf(Ix::new(1), Ix::new(1)), 1);
    assert_eq!(stats.units().count(), 2);
    let doc_stats = back.get::<TermStats<View>>().unwrap();
    assert_eq!(doc_stats.tf(Ix::new(0), Ix::new(1)), 2);
    assert_eq!(doc_stats.length, [4]);

    let facets = back.get::<ChunkFacets>().unwrap();
    assert_eq!(facets.chunk_polarity(Ix::new(0)), Polarity::Positive);
    assert_eq!(facets.chunk_polarity_score(Ix::new(1)), -0.6);
    assert_eq!(facets.aspect_ixs(Ix::new(1)), [Ix::new(1)]);
    assert_eq!(facets.incivility_labels(Ix::new(0)), [IncivilityLabel::Insult]);
    assert_eq!(facets.incivility_scores(Ix::new(0)).insult, 0.9);

    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

fn splice(graph: &mut DocumentGraph, layer: &str, body: Vec<u8>, len: u32) {
    let at = graph.layers.iter().position(|l| l.layer == layer).unwrap();
    graph.layers[at].body = Some(body);
    graph.layers[at].len = len;
}

fn u32_col(w: &mut Writer, values: &[u32]) {
    w.usize(values.len()).unwrap();
    for &v in values {
        w.u32(v);
    }
}

fn facets_body(aspect: u32, score: f64, max: f64) -> Vec<u8> {
    let mut w = Writer::new();
    w.usize(2).unwrap();
    w.u8(0);
    w.u8(1);
    w.usize(2).unwrap();
    w.f64(score);
    w.f64(-0.6);
    u32_col(&mut w, &[0, 1, 1]);
    w.usize(1).unwrap();
    w.u32(aspect);
    w.usize(1).unwrap();
    w.str("food").unwrap();
    w.usize(1).unwrap();
    w.u8(0);
    u32_col(&mut w, &[0, 0, 0]);
    w.usize(0).unwrap();
    w.usize(2).unwrap();
    for v in [max, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] {
        w.f64(v);
    }
    w.finish()
}

#[test]
fn corrupted_chunker_payloads_are_refused() {
    let base = export(&document(), &codecs()).unwrap();

    // A vocabulary out of order is not a vocabulary.
    let mut w = Writer::new();
    w.usize(2).unwrap();
    w.str("food").unwrap();
    w.str("bad").unwrap();
    let mut graph = base.clone();
    splice(&mut graph, "turbo:vocabulary", w.finish(), 2);
    assert!(import(&graph, &codecs()).is_err(), "an unsorted vocabulary imported");

    // An indexed term past the vocabulary, the reserved index, and
    // a count the payload cannot back.
    for (what, first) in [("past the vocabulary", 3u32), ("the reserved index", u32::MAX)] {
        let mut w = Writer::new();
        w.usize(6).unwrap();
        w.u8(1);
        w.u32(first);
        for _ in 0..5 {
            w.u8(0);
        }
        let mut graph = base.clone();
        splice(&mut graph, "turbo:indexed-terms", w.finish(), 6);
        assert!(import(&graph, &codecs()).is_err(), "an indexed term {what} imported");
    }
    let mut w = Writer::new();
    w.usize(u32::MAX as usize).unwrap();
    let mut graph = base.clone();
    splice(&mut graph, "turbo:indexed-terms", w.finish(), 0);
    assert!(import(&graph, &codecs()).is_err(), "a billion indexed terms from four bytes imported");

    // Fewer indexed terms than tokens.
    let mut w = Writer::new();
    w.usize(2).unwrap();
    w.u8(0);
    w.u8(0);
    let mut graph = base.clone();
    splice(&mut graph, "turbo:indexed-terms", w.finish(), 2);
    assert!(import(&graph, &codecs()).is_err(), "indexed terms short of the tokens imported");

    // Term statistics: more unit lengths than chunks, then a tf sum
    // that is not the unit length.
    let stats = |length: &[u32], tf: &[u32]| {
        let mut w = Writer::new();
        u32_col(&mut w, &[0, 1, 1]);
        w.usize(1).unwrap();
        w.u32(1);
        u32_col(&mut w, tf);
        u32_col(&mut w, length);
        u32_col(&mut w, &[0, 1, 0]);
        w.finish()
    };
    let mut graph = base.clone();
    splice(&mut graph, "turbo:term-stats<turbo:chunks>", stats(&[1, 0, 0], &[1]), 3);
    assert!(import(&graph, &codecs()).is_err(), "term stats for three units over two chunks imported");
    let mut graph = base.clone();
    splice(&mut graph, "turbo:term-stats<turbo:chunks>", stats(&[1, 0], &[2]), 2);
    assert!(import(&graph, &codecs()).is_err(), "term stats whose counts do not sum to the length imported");

    // Facets: an aspect past the aspects, a NaN score, a score out
    // of range, an incivility maximum above one, and a polarity
    // byte that is not a polarity.
    for (what, payload) in [
        ("an aspect past the aspects", facets_body(7, 0.5, 0.0)),
        ("the reserved aspect index", facets_body(u32::MAX, 0.5, 0.0)),
        ("a NaN polarity score", facets_body(0, f64::NAN, 0.0)),
        ("a polarity score past one", facets_body(0, 1.5, 0.0)),
        ("an incivility maximum past one", facets_body(0, 0.5, 1.5)),
    ] {
        let mut graph = base.clone();
        splice(&mut graph, "turbo:chunk-facets<turbo:chunks>", payload, 2);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }
    let mut w = Writer::new();
    w.usize(1).unwrap();
    w.u8(3);
    let mut graph = base.clone();
    splice(&mut graph, "turbo:chunk-facets<turbo:chunks>", w.finish(), 1);
    assert!(import(&graph, &codecs()).is_err(), "a polarity byte of 3 imported");

    // The valid control behind the facet cases imports.
    let mut graph = base;
    splice(&mut graph, "turbo:chunk-facets<turbo:chunks>", facets_body(0, 0.5, 0.0), 2);
    assert!(import(&graph, &codecs()).is_ok(), "the hand-built facets control was refused");
}
