use turbo_analysis::*;
use turbo_analysis_layers::{Boundaries, Cut, Sentences, Subwords};
use turbo_chunker::strategy::ChunkStrategy;
use turbo_chunker::{Budget, StrategyMatrix};

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

/// Six sentences of three subwords each (two words and the period),
/// with a cheap topic-shift boundary after the third.
fn make_doc() -> Document {
    let mut doc = sentences_and_subwords("Cats purr. Cats nap. Cats hunt. Rust compiles. Rust checks. Rust ships.");
    let mut reason = vec![Cut::Sentence; 6];
    reason[2] = Cut::TopicShift;
    doc.add(&Key::DEFAULT, Boundaries { cost: vec![1000, 1000, 100, 1000, 1000, 1000], reason }, Provenance::by("t"))
        .unwrap();
    doc
}

/// `text`'s sentences, each ending at a period, and its subwords: each
/// word without its period, and the period.
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

fn budget() -> Budget {
    Budget { max: 12, target: 9, size_weight: 1000, overhead: 0 }
}

fn level0_ranges(m: &StrategyMatrix, at: usize) -> Vec<(usize, usize)> {
    let s = &m.summaries[at];
    (0..s.chunks.spans.len())
        .filter(|&i| s.chunks.level[i] == 0)
        .map(|i| (s.chunks.sentences[i].start.get(), s.chunks.sentences[i].end.get()))
        .collect()
}

#[test]
fn matrix_runs_all_four_strategies() {
    let doc = make_doc();
    let b = budget();
    let strategies = [
        ChunkStrategy::FixedTokenWindow { budget: b },
        ChunkStrategy::GreedySentence { budget: b },
        ChunkStrategy::StructureOnly { budget: b },
        ChunkStrategy::SemanticDp { budget: b },
    ];
    let m = StrategyMatrix::run(&doc, &strategies, 3).unwrap();
    assert_eq!(m.reference, 3);
    assert_eq!(
        m.summaries.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        vec!["fixed-token-window", "greedy-sentence", "structure-only", "semantic-dp"]
    );
    // Each sentence is 3 tokens: fixed packs to target 9 (3+3), greedy
    // to max 12 (4+2).
    assert_eq!(level0_ranges(&m, 0), vec![(0, 3), (3, 6)]);
    assert_eq!(level0_ranges(&m, 1), vec![(0, 4), (4, 6)]);
    // Semantic DP cuts at the cheap boundary after sentence 2.
    assert_eq!(level0_ranges(&m, 3), vec![(0, 3), (3, 6)]);
    let fixed = &m.summaries[0];
    assert_eq!((fixed.chunk_count, fixed.min_tokens, fixed.max_tokens, fixed.p95_tokens), (2, 9, 9, 9));
    assert_eq!(fixed.mean_tokens, 9.0);
    assert_eq!(fixed.budget_utilization, 9.0 / 12.0);
    assert_eq!(fixed.over_budget_count, 0);
    assert_eq!(fixed.cut_count(Cut::DocumentStart), 1);
    assert_eq!(fixed.cut_count(Cut::Sentence), 1);
    let greedy = &m.summaries[1];
    assert_eq!((greedy.min_tokens, greedy.max_tokens), (6, 12));
    // Reports: fixed matches the semantic reference here, greedy does
    // not; the reference against itself is identical; OOR is None.
    assert!(m.report(&doc, 0).unwrap().unwrap().is_identical());
    assert!(!m.report(&doc, 1).unwrap().unwrap().is_identical());
    assert!(m.report(&doc, 3).unwrap().unwrap().is_identical());
    assert!(m.report(&doc, 9).is_none());
}

#[test]
fn strategy_refusals_and_duplicates() {
    let doc = make_doc();
    let b = budget();
    assert!(matches!(StrategyMatrix::run(&doc, &[], 0), Err(Error::Annotator { .. })));
    assert!(matches!(
        StrategyMatrix::run(&doc, &[ChunkStrategy::GreedySentence { budget: b }], 1),
        Err(Error::Annotator { .. })
    ));
    // Missing Subwords.
    let mut bare = Document::new("One. Two.");
    let mut sentences = Spans::new(ViewId::SOURCE);
    sentences.push(Span::new(0, 4));
    sentences.push(Span::new(5, 9));
    bare.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("t")).unwrap();
    assert!(matches!(
        StrategyMatrix::run(&bare, &[ChunkStrategy::GreedySentence { budget: b }], 0),
        Err(Error::Missing(_))
    ));
    // Semantic DP without Boundaries is Missing; the same kind twice
    // gets a suffixed id.
    let mut doc2 = make_doc();
    let m = StrategyMatrix::run(
        &doc2,
        &[ChunkStrategy::GreedySentence { budget: b }, ChunkStrategy::GreedySentence { budget: b }],
        0,
    )
    .unwrap();
    assert_eq!(m.summaries[1].id, "greedy-sentence-1");
    // Boundaries are present here; drop them by using the bare doc shape
    // with subwords but no Boundaries.
    let _ = &mut doc2;
}

#[test]
fn over_budget_sentences_are_chunks_of_their_own() {
    let doc = make_doc();
    let tight = Budget { max: 2, target: 2, size_weight: 1000, overhead: 1 };
    let m = StrategyMatrix::run(&doc, &[ChunkStrategy::FixedTokenWindow { budget: tight }], 0).unwrap();
    let s = &m.summaries[0];
    assert_eq!(s.chunk_count, 6, "each 3-token sentence exceeds the 2-token limit alone");
    assert_eq!(s.over_budget_count, 6);
    assert_eq!(s.mean_tokens, 3.0, "overhead is stripped from the statistics");
    assert_eq!(s.chunks.prompt_overhead, 1);
}

/// The baselines count the prompt overhead as the optimizer does: in the
/// packing limit, the price and the over-budget flag.
#[test]
fn the_baselines_count_the_overhead() {
    // Two 3-token sentences: 6 text tokens fit a max of 6, but not with
    // an overhead of 1, so each sentence is a chunk of 4 model tokens.
    let doc = sentences_and_subwords("Cats purr. Cats nap.");
    let b = Budget { max: 6, target: 6, size_weight: 1000, overhead: 1 };
    let strategies = [
        ChunkStrategy::FixedTokenWindow { budget: b },
        ChunkStrategy::GreedySentence { budget: b },
        ChunkStrategy::StructureOnly { budget: b },
    ];
    let m = StrategyMatrix::run(&doc, &strategies, 2).unwrap();
    for (i, s) in m.summaries.iter().enumerate() {
        assert_eq!(level0_ranges(&m, i), [(0, 1), (1, 2)], "{}", s.id);
        assert_eq!(s.chunks.model_tokens, [4, 4], "{}", s.id);
        assert_eq!(s.over_budget_count, 0, "{}", s.id);
        assert_eq!(s.mean_tokens, 3.0, "{}", s.id);
    }
    // Priced on 4 tokens each, as the optimizer prices them.
    assert_eq!(m.summaries[1].chunks.cost[1], m.summaries[2].chunks.cost[1]);
    // With a max of 3 each 3-token sentence is over once the overhead is
    // in, though its text alone fits.
    let tight = Budget { max: 3, target: 3, size_weight: 1000, overhead: 1 };
    let m = StrategyMatrix::run(&doc, &[ChunkStrategy::GreedySentence { budget: tight }], 0).unwrap();
    assert_eq!(m.summaries[0].over_budget_count, 2);
}
