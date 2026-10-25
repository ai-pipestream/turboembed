//! Clause token offsets are checked in the restored sentence instance and view.
use turbo_analysis::export::{DocumentGraph, ExportError, Writer, export, import};
use turbo_analysis::{Annotator, Document, Key, Layer, ModelRef, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences, Subwords};
use turbo_chunker::{Budget, Chunker, ClauseBreak, ClauseCuts, SentenceClauseCuts};

const SENTENCES: Key<Sentences> = Key::named("selected-sentences");
const CUTS: Key<ClauseCuts> = Key::named("selected-cuts");

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        out.push(Span::new(a, b));
    }
    out
}

fn codecs() -> turbo_analysis::export::Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    turbo_chunker::export::register_codecs(&mut codecs);
    codecs
}

fn document(cuts: SentenceClauseCuts, default_parent: bool) -> Document {
    let mut doc = Document::new("é foo. bar.");
    if default_parent {
        doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 12)]) }, Provenance::by("test")).unwrap();
    }
    doc.add(&SENTENCES, Sentences { spans: spans(&[(0, 7), (8, 12)]) }, Provenance::by("test")).unwrap();
    let on_sentences = || Provenance { reads: vec![SENTENCES.id()], ..Provenance::by("test") };
    doc.add(
        &Key::DEFAULT,
        Subwords {
            spans: spans(&[(0, 2), (3, 4), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (11, 12)]),
            ids: vec![1; 9],
            token: None,
            model: ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".repeat(64) },
        },
        Provenance::by("test"),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Boundaries { cost: vec![0, 0], reason: vec![Cut::Sentence; 2] }, on_sentences()).unwrap();
    doc.add(
        &CUTS,
        ClauseCuts { sentences: vec![cuts, SentenceClauseCuts { tokens: vec![Span::new(8, 12)], breaks: vec![] }] },
        on_sentences(),
    )
    .unwrap();
    doc
}

fn valid_cuts() -> SentenceClauseCuts {
    SentenceClauseCuts { tokens: vec![Span::new(0, 2), Span::new(3, 7)], breaks: vec![ClauseBreak::Clause(-7)] }
}

fn from_bytes(graph: &DocumentGraph) -> Result<Document, ExportError> {
    let bytes = graph.to_bytes().unwrap();
    import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs())
}

#[test]
fn byte_mutations_outside_the_sentence_or_utf8_boundaries_are_refused() {
    for default_parent in [false, true] {
        let original = export(&document(valid_cuts(), default_parent), &codecs()).unwrap();
        for tokens in [
            vec![Span::new(0, 1), Span::new(1, 7)],
            vec![Span::new(0, 13), Span::new(13, 14)],
            vec![Span::new(8, 9), Span::new(10, 12)],
            vec![Span::new(8, 8)],
            vec![Span::new(1, 1)],
        ] {
            let mut w = Writer::new();
            w.usize(2).unwrap();
            w.usize(tokens.len()).unwrap();
            for &token in &tokens {
                w.span(token);
            }
            w.usize(tokens.len().saturating_sub(1)).unwrap();
            for _ in 1..tokens.len() {
                w.u8(0);
                w.u32((-7i32) as u32);
            }
            w.usize(1).unwrap();
            w.span(Span::new(8, 12));
            w.usize(0).unwrap();
            let mut graph = original.clone();
            graph.layers.iter_mut().find(|r| r.layer == ClauseCuts::NAME.to_string()).unwrap().body = Some(w.finish());
            let error = from_bytes(&graph).err().expect("invalid contextual token spans accepted");
            assert!(matches!(error, ExportError::InvalidLayer { .. }), "{tokens:?}: {error}");
        }
    }
}

#[test]
fn named_parents_multibyte_and_empty_boundaries_preserve_all_break_tags() {
    let first = SentenceClauseCuts {
        tokens: vec![Span::new(0, 0), Span::new(0, 2), Span::new(3, 4), Span::new(4, 5), Span::new(7, 7)],
        breaks: vec![
            ClauseBreak::Held,
            ClauseBreak::Clause(i32::MIN),
            ClauseBreak::Phrase(-7),
            ClauseBreak::Token(i32::MAX),
        ],
    };
    for default_parent in [false, true] {
        let graph = export(&document(first.clone(), default_parent), &codecs()).unwrap();
        let back = from_bytes(&graph).unwrap();
        assert_eq!(back.get_key(&CUTS).unwrap().sentences[0], first);
        assert_eq!(back.get_key(&SENTENCES).unwrap().spans.len(), 2);
        assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
    }
}

#[test]
fn valid_restored_multibyte_cuts_still_feed_the_real_chunker() {
    for default_parent in [false, true] {
        let graph = export(&document(valid_cuts(), default_parent), &codecs()).unwrap();
        let mut back = from_bytes(&graph).unwrap();
        let chunker = Chunker { budget: Budget { max: 2, target: 2, size_weight: 0, overhead: 0 } };
        let (chunks,) = chunker
            .annotate((
                back.get_key(&SENTENCES).unwrap(),
                back.get::<Subwords>().unwrap(),
                back.get::<Boundaries>().unwrap(),
                None,
                back.get_key(&CUTS),
                None,
            ))
            .unwrap();
        assert!(chunks.spans.len() >= 2);
        let provenance = Provenance { reads: vec![SENTENCES.id()], ..Provenance::by("test") };
        back.add(&Key::<Chunks>::named("restored-output"), chunks, provenance).unwrap();
    }
}
