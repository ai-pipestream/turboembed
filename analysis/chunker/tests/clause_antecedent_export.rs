//! Clause cuts, chunk antecedents, and `TermStats<Levels>` across
//! the document graph boundary (#1046): populated layers
//! round-trip, and out-of-bounds sentence references are refused
//! on import and caught by document validation.

use std::marker::PhantomData;

use turbo_analysis::export::{Codecs, export, import};
use turbo_analysis::{Document, Ix, IxRange, Key, Layer, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Levels, Sentences, Structure, Unit};
use turbo_chunker::export::register_codecs;
use turbo_chunker::terms::{TermStats, Vocabulary};
use turbo_chunker::{AntecedentLink, ChunkAntecedents, ClauseBreak, ClauseCuts, SentenceClauseCuts};

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

fn document(cuts: ClauseCuts, antecedents: ChunkAntecedents) -> Document {
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
    let mut text = turbo_analysis_layers::Strs::default();
    text.push("food");
    text.push("good");
    doc.add(&Key::DEFAULT, Vocabulary { text }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        TermStats::<Levels> {
            row: vec![0, 2, 3, 4],
            term: vec![Ix::new(0), Ix::new(1), Ix::new(0), Ix::new(0)],
            tf: vec![2, 1, 1, 1],
            length: vec![3, 1, 1],
            df: vec![3, 1],
            _of: PhantomData,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, cuts, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, antecedents, by).unwrap();
    doc
}

fn sample() -> (ClauseCuts, ChunkAntecedents) {
    let cuts = ClauseCuts {
        sentences: vec![
            SentenceClauseCuts {
                tokens: vec![Span::new(0, 4), Span::new(5, 9), Span::new(9, 10)],
                breaks: vec![ClauseBreak::Clause(-7), ClauseBreak::Held],
            },
            SentenceClauseCuts {
                tokens: vec![Span::new(11, 14), Span::new(15, 19)],
                breaks: vec![ClauseBreak::Token(12)],
            },
        ],
    };
    let antecedents = ChunkAntecedents {
        row: vec![0, 0, 1],
        links: vec![AntecedentLink {
            mention_span: Span::new(15, 19),
            mention_sentence: Ix::new(1),
            antecedent_span: Span::new(5, 9),
            antecedent_sentence: Ix::new(0),
            mention_text: "it".into(),
            antecedent_text: "food".into(),
        }],
        headers: vec![String::new(), "food".into()],
    };
    (cuts, antecedents)
}

#[test]
fn clause_cuts_antecedents_and_level_stats_round_trip() {
    let (cuts, antecedents) = sample();
    let doc = document(cuts.clone(), antecedents.clone());
    let graph = export(&doc, &codecs()).unwrap();
    let back =
        import(&turbo_analysis::export::DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs())
            .unwrap();
    assert_eq!(back.get::<ClauseCuts>().unwrap(), &cuts);
    assert_eq!(back.get::<ChunkAntecedents>().unwrap(), &antecedents);
    let stats = back.get::<TermStats<Levels>>().unwrap();
    assert_eq!(stats.row, vec![0, 2, 3, 4]);
    assert_eq!(stats.df, vec![3, 1]);
}

#[test]
fn antecedent_indexes_bound_both_sentence_columns() {
    let (_, antecedents) = sample();
    let idx = antecedents.indexes();
    assert_eq!(idx.len(), 1);
    assert_eq!(idx[0].0, Key::<Sentences>::DEFAULT.id());
    assert_eq!(idx[0].1, 2, "max sentence index 1, plus one");
    assert!(ChunkAntecedents::default().indexes().is_empty());
}

#[test]
fn a_nonzero_first_row_offset_is_refused() {
    // Row [1, 1, 1] is monotonic and ends at links.len(), but the one
    // stored link belongs to no chunk and links_of can never reach it.
    let (_, mut antecedents) = sample();
    antecedents.row = vec![1, 1, 1];
    let err = Layer::check(&antecedents).unwrap_err();
    assert!(err.contains("start at 0"), "{err}");
}

#[test]
fn out_of_bounds_sentence_references_are_refused_when_added() {
    let (cuts, mut antecedents) = sample();
    antecedents.links[0].mention_sentence = Ix::new(9);
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
    let _ = cuts;
    assert!(doc.add(&Key::DEFAULT, antecedents, by).is_err(), "sentence 9 of 2 accepted");
}
