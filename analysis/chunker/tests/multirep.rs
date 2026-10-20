//! Multi-representation indexing and parent expansion over a
//! hand-built tree: a document, two sections, four chunks (one
//! sentence each), token counts 10/20 under the first section
//! (30 in all) and 30/40 under the second (70 in all).

use std::marker::PhantomData;

use turbo_analysis::{Ix, IxRange, Layer, Span, Spans, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Level, Levels, Sentences, Strs, Unit};
use turbo_chunker::terms::{TermStats, Vocabulary};
use turbo_chunker::{
    ChunkId, ExpansionPolicy, MultiRepChunkIndex, ParentExpander, RepresentationKind, ScoreAggregation, unit_tokens,
};

const TEXT: &str = "Alpha beta. Gamma delta. Epsilon zeta. Eta theta.";

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

fn sentence_spans() -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut start = 0u32;
    for (i, c) in TEXT.char_indices() {
        if c == '.' {
            out.push((start, i as u32 + 1));
            start = i as u32 + 2;
        }
    }
    out
}

fn sentences() -> Sentences {
    Sentences { spans: spans(&sentence_spans()) }
}

fn chunks() -> Chunks {
    Chunks {
        spans: spans(&sentence_spans()),
        level: vec![0; 4],
        parent: vec![None; 4],
        sentences: (0..4).map(|i| IxRange::new(i, i + 1)).collect(),
        cut: vec![Cut::DocumentStart, Cut::Sentence, Cut::Sentence, Cut::Sentence],
        cost: vec![0; 4],
        model_tokens: vec![10, 20, 30, 40],
        over_budget: vec![false; 4],
    }
}

fn levels() -> Levels {
    let s = sentence_spans();
    let end = TEXT.len() as u32;
    Levels {
        spans: spans(&[(0, end), (s[0].0, s[1].1), s[0], s[1], (s[2].0, s[3].1), s[2], s[3]]),
        unit: vec![
            Unit::Document,
            Unit::Section(Ix::new(0)),
            Unit::Chunk(Ix::new(0)),
            Unit::Chunk(Ix::new(1)),
            Unit::Section(Ix::new(1)),
            Unit::Chunk(Ix::new(2)),
            Unit::Chunk(Ix::new(3)),
        ],
        parent: vec![
            None,
            Some(Ix::new(0)),
            Some(Ix::new(1)),
            Some(Ix::new(1)),
            Some(Ix::new(0)),
            Some(Ix::new(4)),
            Some(Ix::new(4)),
        ],
    }
}

const DOC: u32 = 0;
const SEC0: u32 = 1;
const C0: u32 = 2;
const C1: u32 = 3;
const SEC1: u32 = 4;
const C2: u32 = 5;
const C3: u32 = 6;

fn ix(u: u32) -> Ix<Levels> {
    Ix::new(u)
}

fn tokens() -> Vec<u32> {
    unit_tokens(&levels(), &chunks()).unwrap()
}

#[test]
fn unit_tokens_roll_up_the_tree() {
    assert_eq!(tokens(), [100, 30, 10, 20, 70, 30, 40]);
    let mut bad = chunks();
    bad.model_tokens.pop();
    assert!(unit_tokens(&levels(), &bad).is_err());
}

#[test]
fn surface_index_ranks_and_resolves() {
    let index = MultiRepChunkIndex::surface(TEXT, &levels(), None).unwrap();
    assert_eq!(index.len(), 4);
    assert_eq!(index.of(ix(C1)).count(), 1);
    assert_eq!(index.by_kind(RepresentationKind::Surface).count(), 4);
    let hits = index.rank(&["gamma"]).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk, ix(C1));
    assert_eq!(index.resolve(hits[0].rep), Some((ix(C1), None)));
    assert!(index.rank(&["nonexistent"]).unwrap().is_empty());
    assert!(index.rank(&[]).is_err());
    assert!(index.resolve(99).is_none());
}

#[test]
fn representations_of_one_chunk_deduplicate_in_expansion() {
    let levels = levels();
    let mut index = MultiRepChunkIndex::surface(TEXT, &levels, None).unwrap();
    let id = ChunkId::from_bytes([7; 16]);
    let summary = index
        .add(RepresentationKind::Summary, ix(C0), Some(id), levels.spans.get(ix(C0)), "gamma rays and alpha particles")
        .unwrap();
    assert!(index.add(RepresentationKind::Header, ix(C0), None, levels.spans.get(ix(C0)), "  ").is_err());
    index.add_sentence_spans(TEXT, &levels, &sentences(), None).unwrap();
    assert_eq!(index.by_kind(RepresentationKind::ChildSpan).count(), 4);

    let hits = index.rank(&["gamma"]).unwrap();
    // The summary of chunk 0, the surface of chunk 1 and chunk
    // 1's sentence span all match; expansion must emit each
    // parent once.
    assert!(hits.len() >= 3, "{hits:?}");
    let expander = ParentExpander::default();
    let out = expander.expand_rep_hits(&index, &levels, &tokens(), &hits).unwrap();
    let units: Vec<u32> = out.iter().map(|h| h.unit.get() as u32).collect();
    assert_eq!(units, [SEC0], "every hit resolves into the first section, once");
    assert!(out[0].reps.contains(&summary));
    assert_eq!(out[0].children, [ix(C0), ix(C1)]);
    assert!(out[0].chunk_ids.contains(&Some(id)));
}

#[test]
fn keyphrases_come_from_term_stats() {
    let levels = levels();
    let mut text = Strs::default();
    for term in ["alpha", "beta", "gamma"] {
        text.push(term);
    }
    let vocabulary = Vocabulary { text };
    // Chunk 0 (unit 2): alpha x2, beta x1. Chunk 1 (unit 3):
    // gamma x3. All other units hold nothing.
    let stats = TermStats::<Levels> {
        row: vec![0, 0, 0, 2, 3, 3, 3, 3],
        term: vec![Ix::new(0), Ix::new(1), Ix::new(2)],
        tf: vec![2, 1, 3],
        length: vec![0, 0, 3, 3, 0, 0, 0],
        df: vec![1, 1, 1],
        _of: PhantomData,
    };
    stats.check().unwrap();
    let mut index = MultiRepChunkIndex::new();
    let added = index.add_keyphrases(&levels, &stats, &vocabulary, None, 1).unwrap();
    assert_eq!(added, 2);
    let texts: Vec<&str> = index.entries().iter().map(|e| e.text.as_str()).collect();
    assert_eq!(texts, ["alpha", "gamma"]);
    assert!(index.add_keyphrases(&levels, &stats, &vocabulary, None, 0).is_err());
}

fn policy(aggregation: ScoreAggregation) -> ExpansionPolicy {
    ExpansionPolicy { aggregation, ..ExpansionPolicy::default() }
}

#[test]
fn aggregation_modes_over_one_parent() {
    let levels = levels();
    let hits = [(ix(C0), 0.5), (ix(C1), 0.9)];
    let max = ParentExpander::new(policy(ScoreAggregation::Max)).expand(&levels, &tokens(), &hits).unwrap();
    assert_eq!(max.len(), 1);
    assert_eq!((max[0].unit, max[0].tokens), (ix(SEC0), 30));
    assert_eq!(max[0].children, [ix(C0), ix(C1)]);
    assert!((max[0].score - 0.9).abs() < 1e-12);

    let sum = ParentExpander::new(policy(ScoreAggregation::Sum)).expand(&levels, &tokens(), &hits).unwrap();
    assert!((sum[0].score - 1.4).abs() < 1e-12);
    let mean = ParentExpander::new(policy(ScoreAggregation::Mean)).expand(&levels, &tokens(), &hits).unwrap();
    assert!((mean[0].score - 0.7).abs() < 1e-12);
    let rrf = ParentExpander::new(policy(ScoreAggregation::ReciprocalRankFusion { k: 60 }))
        .expand(&levels, &tokens(), &hits)
        .unwrap();
    assert!((rrf[0].score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
}

#[test]
fn parent_token_budget_keeps_leaves() {
    let levels = levels();
    let p = ExpansionPolicy { max_parent_tokens: 25, ..ExpansionPolicy::default() };
    let out = ParentExpander::new(p).expand(&levels, &tokens(), &[(ix(C0), 0.5), (ix(C1), 0.9)]).unwrap();
    let units: Vec<u32> = out.iter().map(|h| h.unit.get() as u32).collect();
    assert_eq!(units, [C1, C0], "the 30-token section is over budget, leaves stay, best first");

    // With no target level the document itself is the ceiling,
    // still subject to the same budget at every step.
    let mut p = ExpansionPolicy { target_level: None, ..ExpansionPolicy::default() };
    let out = ParentExpander::new(p.clone()).expand(&levels, &tokens(), &[(ix(C0), 1.0)]).unwrap();
    assert_eq!(out[0].unit, ix(DOC));
    p.max_parent_tokens = 50;
    let out = ParentExpander::new(p).expand(&levels, &tokens(), &[(ix(C0), 1.0)]).unwrap();
    assert_eq!(out[0].unit, ix(SEC0), "the 100-token document is over the 50-token parent budget");
}

#[test]
fn ancestor_and_descendant_never_both_emitted() {
    let levels = levels();
    let p = ExpansionPolicy { max_parent_tokens: 0, ..ExpansionPolicy::default() };
    // The child outranks its section: the section is blocked.
    let out = ParentExpander::new(p.clone()).expand(&levels, &tokens(), &[(ix(SEC1), 0.9), (ix(C2), 0.95)]).unwrap();
    assert_eq!(out.iter().map(|h| h.unit.get() as u32).collect::<Vec<_>>(), [C2]);
    // The section outranks its child: the child is blocked.
    let out = ParentExpander::new(p).expand(&levels, &tokens(), &[(ix(SEC1), 0.95), (ix(C2), 0.9)]).unwrap();
    assert_eq!(out.iter().map(|h| h.unit.get() as u32).collect::<Vec<_>>(), [SEC1]);
    assert_eq!(out[0].level, Level::Section);
}

#[test]
fn total_budget_caps_and_skips_to_what_fits() {
    let levels = levels();
    let hits = [(ix(SEC0), 0.9), (ix(SEC1), 0.8)];
    let p = ExpansionPolicy { max_total_tokens: Some(50), ..ExpansionPolicy::default() };
    let out = ParentExpander::new(p).expand(&levels, &tokens(), &hits).unwrap();
    assert_eq!(out.iter().map(|h| h.unit.get() as u32).collect::<Vec<_>>(), [SEC0]);

    // A too-large best candidate is skipped, not fatal: the
    // smaller later one still fits.
    let hits = [(ix(SEC1), 0.9), (ix(SEC0), 0.8)];
    let p = ExpansionPolicy { max_total_tokens: Some(50), ..ExpansionPolicy::default() };
    let out = ParentExpander::new(p).expand(&levels, &tokens(), &hits).unwrap();
    assert_eq!(out.iter().map(|h| h.unit.get() as u32).collect::<Vec<_>>(), [SEC0]);
}

#[test]
fn expansion_refusals() {
    let levels = levels();
    let expander = ParentExpander::default();
    assert!(expander.expand(&levels, &tokens()[..3], &[]).is_err());
    assert!(expander.expand(&levels, &tokens(), &[(ix(99), 1.0)]).is_err());
    assert!(expander.expand(&levels, &tokens(), &[(ix(C0), f64::NAN)]).is_err());
    assert!(expander.expand(&levels, &tokens(), &[]).unwrap().is_empty());

    let index = MultiRepChunkIndex::surface(TEXT, &levels, None).unwrap();
    let mut hits = index.rank(&["gamma"]).unwrap();
    hits[0].chunk = ix(C3);
    assert!(expander.expand_rep_hits(&index, &levels, &tokens(), &hits).is_err(), "a hit naming the wrong chunk");
    hits[0].rep = 99;
    hits[0].chunk = ix(C1);
    assert!(expander.expand_rep_hits(&index, &levels, &tokens(), &hits).is_err());
}
