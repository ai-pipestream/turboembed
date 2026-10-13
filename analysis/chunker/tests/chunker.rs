//! The chunker over hand-made layers: sentences, the embedder's subwords
//! and boundary costs, as the signal annotators upstream will write them.

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Sentences, Structure, Subwords};
use turbo_chunker::{Budget, Chunker};

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

/// A document with sentences (each ends at a period) and subwords (every
/// word and every period is one), as a chunker reads them.
fn sentences_and_subwords(text: &str) -> Document {
    let mut doc = Document::new(text);
    let (mut sentences, mut pieces) = (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if c == '.' {
            sentences.push(Span::new(start, i as u32 + 1));
            start = i as u32 + 2;
        }
    }
    let mut at = 0;
    for word in text.split(' ') {
        let w = word.trim_end_matches('.');
        pieces.push(Span::new(at, at + w.len() as u32));
        if w.len() < word.len() {
            pieces.push(Span::new(at + w.len() as u32, at + word.len() as u32));
        }
        at += word.len() as u32 + 1;
    }
    let ids = vec![0; pieces.len()];
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, Subwords { spans: pieces, ids, model: toy_model(), token: None }, Provenance::by("t"))
        .unwrap();
    doc
}

fn chunker(budget: Budget) -> Analyzer {
    Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .add(Chunker { budget })
        .build()
        .unwrap()
}

#[test]
fn the_chunker_cuts_where_the_topic_shifts() {
    let mut doc = sentences_and_subwords("Cats purr. Cats nap. Cats hunt. Rust compiles. Rust checks. Rust ships.");
    // A plain sentence boundary costs a unit; the topic shift after the third, a tenth.
    let mut reason = vec![Cut::Sentence; 6];
    reason[2] = Cut::TopicShift;
    let cost = vec![1000, 1000, 100, 1000, 1000, 1000];
    doc.add(&Key::DEFAULT, Boundaries { cost, reason }, Provenance::by("t")).unwrap();

    let budget = Budget { max: 12, target: 9, size_weight: 1000, overhead: 0 };
    let mut docs = vec![doc];
    assert!(chunker(budget).run(&mut docs)[0].is_ok());
    let chunks = docs[0].get::<Chunks>().unwrap();
    let texts: Vec<&str> = chunks.spans.iter().map(|(_, s)| &docs[0].source()[s.range()]).collect();
    // Filling to 12 tokens would take four sentences and cut mid-topic.
    assert_eq!(texts, ["Cats purr. Cats nap. Cats hunt.", "Rust compiles. Rust checks. Rust ships."]);
    assert_eq!(chunks.cut, [Cut::DocumentStart, Cut::TopicShift]);
    assert_eq!(chunks.model_tokens, [9, 9]);
}

#[test]
fn sections_are_never_crossed_and_become_the_level_above() {
    let text = "Cats purr. Cats nap. Cats hunt. Rust compiles. Rust ships. Rust runs.";
    let mut plain = sentences_and_subwords(text);
    // Every sentence is 3 tokens and a chunk holds 2. Cutting after "nap."
    // or "ships." is cheaper than elsewhere; nothing in the costs marks the
    // section break after "hunt.".
    let cost = vec![1000, 500, 1000, 1000, 500, 1000];
    plain.add(&Key::DEFAULT, Boundaries { cost, reason: vec![Cut::Sentence; 6] }, Provenance::by("t")).unwrap();
    let mut sectioned = plain.clone();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 31));
    spans.push(Span::new(32, 69));
    let structure = Structure { spans, kind: vec![Block::Section; 2], parent: vec![None; 2], depth: vec![0; 2] };
    sectioned.add(&Key::DEFAULT, structure, Provenance::by("parser")).unwrap();

    let budget = Budget { max: 7, target: 6, size_weight: 1000, overhead: 0 };
    let mut docs = vec![plain, sectioned];
    assert!(chunker(budget).run(&mut docs).iter().all(|s| s.is_ok()));
    let texts = |d: &Document| {
        let c = d.get::<Chunks>().unwrap();
        c.spans.iter().map(|(_, s)| d.source()[s.range()].to_owned()).collect::<Vec<_>>()
    };

    // Without structure the cheapest cut runs a chunk across the sections.
    assert_eq!(texts(&docs[0]), ["Cats purr. Cats nap.", "Cats hunt. Rust compiles.", "Rust ships. Rust runs."]);

    // With it, each section is cut on its own and is the chunks' parent.
    let c = docs[1].get::<Chunks>().unwrap();
    assert_eq!(
        texts(&docs[1]),
        [
            "Cats purr. Cats nap. Cats hunt.",
            "Cats purr. Cats nap.",
            "Cats hunt.",
            "Rust compiles. Rust ships. Rust runs.",
            "Rust compiles. Rust ships.",
            "Rust runs."
        ]
    );
    assert_eq!(c.level, [1, 0, 0, 1, 0, 0]);
    let (s1, s2) = (Some(Ix::new(0)), Some(Ix::new(3)));
    assert_eq!(c.parent, [None, s1, s1, None, s2, s2]);
    assert_eq!(
        c.cut,
        [Cut::DocumentStart, Cut::DocumentStart, Cut::Sentence, Cut::Section, Cut::Section, Cut::Sentence]
    );
    assert_eq!(c.model_tokens, [9, 6, 3, 9, 6, 3]);
    // The provenance says the structure was read.
    let reads = &docs[1].provenance(Key::<Chunks>::DEFAULT.id()).unwrap().reads;
    assert_eq!(reads.last(), Some(&Key::<Structure>::DEFAULT.id()));
    assert_eq!(docs[0].provenance(Key::<Chunks>::DEFAULT.id()).unwrap().reads.len(), 3);
}

#[test]
fn what_lies_between_sentences_counts_inside_a_chunk() {
    // Byte-level BPE makes a token of the paragraph break, which neither
    // sentence holds: the sentences are 3 tokens each, and 7 together.
    let mut doc = Document::new("One two.\n\nThree four.");
    let (mut sentences, mut pieces) = (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    for (a, b) in [(0, 8), (10, 21)] {
        sentences.push(Span::new(a, b));
    }
    for (a, b) in [(0, 3), (4, 7), (7, 8), (8, 10), (10, 15), (16, 20), (20, 21)] {
        pieces.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("t")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Subwords { spans: pieces, ids: vec![0; 7], model: toy_model(), token: None },
        Provenance::by("t"),
    )
    .unwrap();
    let bounds = Boundaries { cost: vec![1000; 2], reason: vec![Cut::Paragraph, Cut::Sentence] };
    doc.add(&Key::DEFAULT, bounds, Provenance::by("t")).unwrap();

    let lengths = |max| {
        let mut docs = vec![doc.clone()];
        assert!(chunker(Budget { max, target: max, size_weight: 1000, overhead: 0 }).run(&mut docs)[0].is_ok());
        docs[0].get::<Chunks>().unwrap().model_tokens.clone()
    };
    assert_eq!(lengths(7), [7]);
    // A sum over the sentences alone would fit them in 6.
    assert_eq!(lengths(6), [3, 3]);
}
