//! Term statistics and BM25 over hand-made tokens, structure and chunks,
//! with the terms read from a case folded view.

#[path = "terms/bm25.rs"]
mod bm25;
#[path = "terms/counter.rs"]
mod counter;
#[path = "terms/views.rs"]
mod views;

use std::collections::HashMap;

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Chunks, Cut, Sentences, Structure, Tokens};
use turbo_chunker::terms::{Bm25, Collection, LocalCollection, Query, TermCounter, TermIndexer, TermStats, Vocabulary};
use turbo_normalize::{CaseFold, NormalizeStep};

const TEXT: &str = "Cats purr. Cats nap. Dogs nap too. Rust compiles. RUST ships fast.";

#[test]
fn linguistic_terms_and_raw_index_can_coexist() {
    let pipeline = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .add(turbo_stem::TermAnnotator::for_language("en").unwrap())
        .add(TermIndexer)
        .build()
        .unwrap();
    let mut docs = [document()];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    assert!(docs[0].get::<turbo_analysis_layers::Terms>().is_some());
    assert!(docs[0].get::<turbo_chunker::terms::Terms>().is_some());
}

/// The source span from the start of `first` to the end of `last`.
fn span(text: &str, first: &str, last: &str) -> Span {
    let a = text.find(first).unwrap();
    let b = text.find(last).unwrap() + last.len();
    Span::new(a as u32, b as u32)
}

/// Words split at spaces, with a sentence's closing period a token of its
/// own; sentences end at a period.
fn tokens_and_sentences(text: &str) -> (Tokens, Sentences) {
    let (mut tokens, mut sentences) = (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    let (mut at, mut start) = (0u32, 0u32);
    for word in text.split(' ') {
        let w = word.trim_end_matches('.');
        tokens.push(Span::new(at, at + w.len() as u32));
        if w.len() < word.len() {
            tokens.push(Span::new(at + w.len() as u32, at + word.len() as u32));
            sentences.push(Span::new(start, at + word.len() as u32));
            start = at + word.len() as u32 + 1;
        }
        at += word.len() as u32 + 1;
    }
    (Tokens { spans: tokens, sentence: None }, Sentences { spans: sentences })
}

/// Two top-level sections, the second with a section nested in it, and a
/// paragraph in the first; chunks one level up per section and two cut
/// from the first.
fn document() -> Document {
    let mut doc = Document::new(TEXT);
    let (tokens, sentences) = tokens_and_sentences(TEXT);
    let mut st =
        Structure { spans: Spans::new(ViewId::SOURCE), kind: Vec::new(), parent: Vec::new(), depth: Vec::new() };
    for (first, last, kind, parent) in [
        ("Cats purr", "too.", Block::Section, None),
        ("Cats purr", "nap.", Block::Paragraph, Some(0)),
        ("Rust", "fast.", Block::Section, None),
        ("RUST", "fast.", Block::Section, Some(2)),
    ] {
        st.spans.push(span(TEXT, first, last));
        st.kind.push(kind);
        st.parent.push(parent.map(Ix::new));
        st.depth.push(u8::from(parent.is_some()));
    }
    let mut ch = Chunks {
        spans: Spans::new(ViewId::SOURCE),
        level: Vec::new(),
        parent: Vec::new(),
        sentences: Vec::new(),
        cut: Vec::new(),
        cost: Vec::new(),
        over_budget: Vec::new(),
        prompt_overhead: 0,
        model_tokens: Vec::new(),
    };
    for (first, last, level, parent, s) in [
        ("Cats purr", "too.", 1, None, (0, 3)),
        ("Cats purr", "nap.", 0, Some(0), (0, 2)),
        ("Dogs", "too.", 0, Some(0), (2, 3)),
        ("Rust", "fast.", 1, None, (3, 5)),
        ("Rust", "fast.", 0, Some(3), (3, 5)),
    ] {
        ch.spans.push(span(TEXT, first, last));
        ch.level.push(level);
        ch.parent.push(parent.map(Ix::new));
        ch.sentences.push(IxRange::new(s.0, s.1));
        ch.cut.push(Cut::Sentence);
        ch.cost.push(0);
        ch.over_budget.push(false);
        ch.model_tokens.push(0);
    }
    doc.add(&Key::DEFAULT, sentences, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, tokens, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, st, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, ch, Provenance::by("t")).unwrap();
    doc
}

/// Case folds the source into "folded", reads the terms from it, and
/// counts them per chunk, per structure block and for the document.
fn pipeline() -> Analyzer {
    Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .step(Bound::new(NormalizeStep::new(CaseFold::full())).write(0, "folded"))
        .step(Bound::new(TermIndexer).read(0, "folded"))
        .add(TermCounter::<Chunks>::new())
        .add(TermCounter::<Structure>::new())
        .add(TermCounter::<View>::new())
        .build()
        .unwrap()
}

fn run() -> Document {
    let mut docs = vec![document()];
    let status = pipeline().run(&mut docs);
    assert_eq!(status[0], Ok(()));
    docs.pop().unwrap()
}

/// A unit's terms and counts, as text.
fn row<L: Layer>(stats: &TermStats<L>, v: &Vocabulary, i: u32) -> Vec<(String, u32)> {
    let (terms, tf) = stats.unit(Ix::new(i));
    terms.iter().zip(tf).map(|(&t, &c)| (v.get(t).to_string(), c)).collect()
}

fn pairs(xs: &[(&str, u32)]) -> Vec<(String, u32)> {
    xs.iter().map(|&(t, c)| (t.to_string(), c)).collect()
}

#[test]
fn terms_come_from_the_folded_view_and_punctuation_is_no_term() {
    let doc = run();
    let v = doc.get::<Vocabulary>().unwrap();
    let words: Vec<&str> = v.text.iter().collect();
    assert_eq!(words, ["cats", "compiles", "dogs", "fast", "nap", "purr", "rust", "ships", "too"]);
    let terms = doc.get::<turbo_chunker::terms::Terms>().unwrap();
    let (tokens, _) = tokens_and_sentences(TEXT);
    assert_eq!(terms.term.len(), tokens.spans.len());
    // "Cats" "purr" "." ...: the period is a token with no term.
    assert_eq!(terms.term[2], None);
    assert_eq!(terms.term[0].map(|t| v.get(t)), Some("cats"));
    // "RUST" and "Rust" are one term.
    let rust: Vec<usize> = tokens
        .spans
        .iter()
        .filter(|(_, s)| TEXT[s.range()].eq_ignore_ascii_case("rust"))
        .map(|(i, _)| i.get())
        .collect();
    assert_eq!(rust.len(), 2);
    assert_eq!(terms.term[rust[0]], terms.term[rust[1]]);
}

#[test]
fn every_level_gets_its_counts_lengths_and_df() {
    let doc = run();
    let v = doc.get::<Vocabulary>().unwrap();

    let chunks = doc.get::<TermStats<Chunks>>().unwrap();
    assert_eq!(chunks.length, [7, 4, 3, 5, 5]);
    assert_eq!(row(chunks, v, 1), pairs(&[("cats", 2), ("nap", 1), ("purr", 1)]));
    assert_eq!(row(chunks, v, 2), pairs(&[("dogs", 1), ("nap", 1), ("too", 1)]));
    assert_eq!(row(chunks, v, 0), pairs(&[("cats", 2), ("dogs", 1), ("nap", 2), ("purr", 1), ("too", 1)]));
    assert_eq!(row(chunks, v, 4), pairs(&[("compiles", 1), ("fast", 1), ("rust", 2), ("ships", 1)]));
    // "nap" is in the first section chunk and in both of its children.
    assert_eq!(chunks.df[v.lookup("nap").unwrap().get()], 3);

    let blocks = doc.get::<TermStats<Structure>>().unwrap();
    assert_eq!(blocks.length, [7, 4, 5, 3]);
    assert_eq!(row(blocks, v, 3), pairs(&[("fast", 1), ("rust", 1), ("ships", 1)]));
    // The nested section's terms count in it and in the section around it.
    assert_eq!(blocks.tf(Ix::new(2), v.lookup("rust").unwrap()), 2);

    let whole = doc.get::<TermStats<View>>().unwrap();
    assert_eq!(whole.length, [12]);
    assert_eq!(whole.tf(Ix::new(0), v.lookup("cats").unwrap()), 2);
    assert!(whole.df.iter().all(|&d| d == 1));

    let p = doc.provenance(Key::<TermStats<Chunks>>::DEFAULT.id()).unwrap();
    assert_eq!(p.annotator, "turbo:term-counter");
    assert_eq!(Key::<TermStats<Chunks>>::DEFAULT.id().to_string(), "turbo:term-stats<turbo:chunks>");
}
