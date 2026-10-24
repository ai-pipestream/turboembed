//! Fragment-aware aggregation (#1048): term counting skips
//! excluded gaps of discontinuous units, and chunk facets do
//! not duplicate token-anchored aspects or evidenced
//! incivility across clause chunks of one sentence.

use turbo_analysis::*;
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Chunks, Cut, Incivility, IncivilityItem, IncivilityLabel, IncivilityScores,
    IncivilityTarget, PolarityScores, Sentences, Strs, Tokens,
};
use turbo_chunker::terms::{TermCounter, TermStats, Terms, Vocabulary};
use turbo_chunker::{ChunkFacetBuilder, ChunkFacets};

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

/// Six one-word tokens over "aa bb cc dd ee ff" (3 bytes apart).
fn tokens() -> Tokens {
    Tokens { spans: spans(&[(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17)]), sentence: Some(vec![Ix::new(0); 6]) }
}

#[test]
fn term_counter_skips_the_gap_of_a_discontinuous_chunk() {
    let mut doc = Document::new("aa bb cc dd ee ff");
    let by = Provenance::by("t");
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 17)]) }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, tokens(), by.clone()).unwrap();
    let mut vocab = Strs::default();
    for w in ["aa", "bb", "cc", "dd", "ee", "ff"] {
        vocab.push(w);
    }
    doc.add(&Key::DEFAULT, Vocabulary { text: vocab }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, Terms { term: (0..6).map(|i| Some(Ix::new(i))).collect() }, by.clone()).unwrap();
    // One discontinuous chunk holding fragments [0,5) and [12,17):
    // tokens aa, bb, ee, ff; cc and dd sit in the excluded gap.
    let mut chunk_spans = Spans::new(ViewId::SOURCE);
    chunk_spans.push_fragments(&[Span::new(0, 5), Span::new(12, 17)]);
    doc.add(
        &Key::DEFAULT,
        Chunks {
            spans: chunk_spans,
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![4],
            over_budget: vec![false],
            prompt_overhead: 0,
        },
        by,
    )
    .unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Terms>::DEFAULT)
        .given(Key::<Vocabulary>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .add(TermCounter::<Chunks>::new())
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs).remove(0).unwrap();
    let stats = docs[0].get::<TermStats<Chunks>>().unwrap();
    assert_eq!(stats.length, vec![4]);
    let (terms, _) = stats.unit(Ix::new(0));
    assert_eq!(terms.iter().map(|t| t.get()).collect::<Vec<_>>(), vec![0, 1, 4, 5]);
    assert_eq!(stats.df, vec![1, 1, 0, 0, 1, 1]);
}

#[test]
fn facets_stay_in_their_clause_chunk() {
    let mut doc = Document::new("aa bb cc dd ee ff");
    let by = Provenance::by("t");
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 17)]) }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, tokens(), by.clone()).unwrap();
    // Two clause chunks of the single sentence: [0,8) and [9,17).
    doc.add(
        &Key::DEFAULT,
        Chunks {
            spans: spans(&[(0, 8), (9, 17)]),
            level: vec![0, 0],
            parent: vec![None, None],
            sentences: vec![IxRange::new(0, 1), IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart, Cut::Clause],
            cost: vec![0, 1],
            model_tokens: vec![3, 3],
            over_budget: vec![false, false],
            prompt_overhead: 0,
        },
        by.clone(),
    )
    .unwrap();
    // Aspect on token 1 (bb, first clause), aspect on token 4 (ee, second).
    let mut aspects = Aspects::default();
    for tok in [1u32, 4] {
        aspects.push(Aspect {
            target: AspectTarget::Term { tokens: IxRange::new(tok, tok + 1), sentence: Some(Ix::new(0)) },
            category: None,
            polarity: PolarityScores::new(0.9, 0.05, 0.05),
            opinions: &[],
        });
    }
    doc.add(&Key::DEFAULT, aspects, by.clone()).unwrap();
    // Incivility evidence on token 5 (ff, second clause only).
    let mut inc = Incivility::new();
    inc.push(IncivilityItem {
        target: IncivilityTarget::Sentence(Ix::new(0)),
        scores: IncivilityScores { insult: 0.9, ..IncivilityScores::default() },
        evidence: &[(IncivilityLabel::Insult, IxRange::new(5, 6))],
    });
    doc.add(&Key::DEFAULT, inc, by).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Chunks>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Aspects>::DEFAULT)
        .given(Key::<Incivility>::DEFAULT)
        .add(ChunkFacetBuilder)
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs).remove(0).unwrap();
    let facets = docs[0].get::<ChunkFacets>().unwrap();
    assert_eq!(facets.aspect_row, vec![0, 1, 2], "one aspect per clause chunk");
    assert_eq!(facets.aspect[0], Ix::new(0));
    assert_eq!(facets.aspect[1], Ix::new(1));
    assert_eq!(facets.incivility_row, vec![0, 0, 1], "evidence only in the second clause chunk");
    assert_eq!(facets.incivility_max[0].insult, 0.0);
    assert_eq!(facets.incivility_max[1].insult, 0.9);
}
