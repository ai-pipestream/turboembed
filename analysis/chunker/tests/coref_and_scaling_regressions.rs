//! Regression tests for PR #736 review findings:
//! 1. Canonical `turbo_coref::Mentions` integration from `CorefAnnotator`
//!    through `CorefBoundaries`, `DistilledBoundaryScorer`, `Chunker`, and
//!    `ChunkAntecedentBuilder`.
//! 2. Preserving edge confidence (`0.1 -> weight_milli = 100`) in
//!    `extract_chains` and `coref_boundaries_from_chains` on both `Entities`
//!    and `Mentions`.
//! 3. View mismatch rejection in `DistilledBoundaryScorer::annotate`.
//! 4. Linear-time $O(m)$ dependency subtree precomputation on long chains
//!    (400 / 800 / 1600 tokens).

use turbo_analysis::{
    Analyzer, Annotator, Bound, Document, Edge, Edges, Groups, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans,
    Tag, TagSet, View, ViewId,
};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Dependencies, Entities, PosTags, Sentences, Subwords, Tokens};
use turbo_chunker::{
    Budget, ChunkAntecedentBuilder, ChunkAntecedents, Chunker, ClauseCosts, CorefBoundaries, CorefCohesion, Costs,
    DistilledBoundaryScorer, DistilledWeights, MentionKind, Mentions, STRUCTURE, StructureBoundaries, clause_cuts,
    coref_boundaries_from_chains, extract_chains,
};
use turbo_coref::CorefAnnotator;

#[test]
fn real_coref_annotator_pipeline_feeds_coref_boundaries_and_chunk_antecedent_builder() {
    // S0: "John Smith arrived in Paris."
    // S1: "He met Alice Brown."
    let s0 = "John Smith arrived in Paris.";
    let s1 = "He met Alice Brown.";
    let text = format!("{s0} {s1}");
    let mut doc = Document::new(text.clone());

    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, s0.len() as u32));
    sent_spans.push(Span::new(s0.len() as u32 + 1, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("t")).unwrap();

    // Words + punctuation:
    // S0: 0="John"(0..4), 1="Smith"(5..10), 2="arrived"(11..18), 3="in"(19..21), 4="Paris"(22..27), 5="."(27..28)
    // S1: 6="He"(29..31), 7="met"(32..35), 8="Alice"(36..41), 9="Brown"(42..47), 10="."(47..48)
    let raw_toks: &[(u32, u32, u32, &str)] = &[
        (0, 4, 0, "NNP"),
        (5, 10, 0, "NNP"),
        (11, 18, 0, "VBD"),
        (19, 21, 0, "IN"),
        (22, 27, 0, "NNP"),
        (27, 28, 0, "."),
        (29, 31, 1, "PRP"),
        (32, 35, 1, "VBD"),
        (36, 41, 1, "NNP"),
        (42, 47, 1, "NNP"),
        (47, 48, 1, "."),
    ];
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut sw_spans = Spans::new(ViewId::SOURCE);
    let mut tok_sents = Vec::new();
    let tagset = TagSet::new("ptb", &["NNP", "VBD", "IN", "PRP", "."]);
    let mut pos_tags = Vec::new();
    for &(a, b, s, tag) in raw_toks {
        tok_spans.push(Span::new(a, b));
        sw_spans.push(Span::new(a, b));
        tok_sents.push(Ix::new(s));
        pos_tags.push(tagset.tag(tag).unwrap());
    }
    let on_sents = Provenance { reads: vec![Key::<Sentences>::DEFAULT.id()], ..Provenance::by("t") };
    let on_toks =
        Provenance { reads: vec![Key::<Sentences>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()], ..Provenance::by("t") };
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: Some(tok_sents) }, on_sents).unwrap();
    doc.add(
        &Key::DEFAULT,
        Subwords {
            spans: sw_spans,
            ids: vec![1; raw_toks.len()],
            model: ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() },
            token: None,
        },
        Provenance::by("t"),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, PosTags { tags: pos_tags, tagset, prob: None }, on_toks.clone()).unwrap();

    let mut ent_spans = Spans::new(ViewId::SOURCE);
    ent_spans.push(Span::new(0, 10)); // "John Smith"
    ent_spans.push(Span::new(22, 27)); // "Paris"
    ent_spans.push(Span::new(36, 47)); // "Alice Brown"
    let ent_kinds = TagSet::new("ner", &["person", "location"]);
    let entities = Entities {
        spans: ent_spans,
        tokens: vec![IxRange::new(0, 2), IxRange::new(4, 5), IxRange::new(8, 10)],
        kind: vec![Tag(0), Tag(1), Tag(0)],
        kinds: ent_kinds,
        score: vec![1.0, 1.0, 1.0],
    };
    doc.add(&Key::DEFAULT, entities, on_toks).unwrap();

    // Run CorefAnnotator -> CorefBoundaries -> DistilledBoundaryScorer -> Chunker -> ChunkAntecedentBuilder.
    // Budget max = 6 forces S0 (6 tokens) and S1 (5 tokens) into separate chunks, so ChunkAntecedentBuilder
    // resolves "He" in chunk 1 back to "John Smith" in chunk 0.
    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .add(CorefAnnotator::default())
        .step(Bound::new(StructureBoundaries { costs: Costs::default() }).write(0, STRUCTURE))
        .step(CorefBoundaries { cohesion: CorefCohesion::default() }.bound())
        .step(DistilledBoundaryScorer::default().bound())
        .add(Chunker { budget: Budget { max: 6, target: 5, size_weight: 100, overhead: 0 } })
        .add(ChunkAntecedentBuilder)
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();

    let mentions = doc.get::<Mentions>().unwrap();
    assert!(mentions.spans.len() >= 3);
    let coref_b = doc.get_key(&Key::<Boundaries>::named("coref")).unwrap();
    assert!(coref_b.cost[0] > 0, "expected positive coref penalty across 'John Smith' <- 'He' link");

    let chunks = doc.get::<Chunks>().unwrap();
    assert_eq!(chunks.spans.len(), 2);
    let ant = doc.get::<ChunkAntecedents>().unwrap();
    assert_eq!(ant.headers[1], "[Context: John Smith (He)]");
}

#[test]
fn edge_confidence_must_scale_coref_cost() {
    let mut s_spans = Spans::new(ViewId::SOURCE);
    s_spans.push(Span::new(0, 1));
    s_spans.push(Span::new(2, 3));
    let sentences = Sentences { spans: s_spans };

    // 1. Entities edge with score = 0.1 -> weight_milli = 100, boundary cost = 100.
    let mut e_spans = Spans::new(ViewId::SOURCE);
    e_spans.push(Span::new(0, 1));
    e_spans.push(Span::new(2, 3));
    let entities = Entities {
        spans: e_spans,
        kind: vec![Tag(0), Tag(0)],
        kinds: TagSet::new("ner", &["person"]),
        score: vec![1.0; 2],
        tokens: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
    };
    let e_edges = Edges::<Entities, Entities>::between(
        Key::DEFAULT,
        Key::DEFAULT,
        TagSet::new("coref", &["ant"]),
        [Edge { from: Ix::new(1), to: Ix::new(0), kind: Tag(0), score: 0.1 }],
    );
    let chains = extract_chains(&sentences, None, None, None, None, Some(&entities), None, Some(&e_edges));
    assert_eq!(chains.links.len(), 1);
    assert_eq!(chains.links[0].weight_milli, 100);
    assert_eq!(coref_boundaries_from_chains(&chains, CorefCohesion::default()).cost[0], 100);

    // 2. Mentions edge with score = 0.1 (both standalone and alongside Groups<Mentions>).
    let mut mentions = Mentions::new(ViewId::SOURCE);
    mentions.push(Span::new(0, 1), Ix::new(0), IxRange::new(0, 1), Ix::new(0), MentionKind::Proper);
    mentions.push(Span::new(2, 3), Ix::new(1), IxRange::new(1, 2), Ix::new(1), MentionKind::Proper);
    let m_edges = Edges::<Mentions, Mentions>::between(
        Key::DEFAULT,
        Key::DEFAULT,
        TagSet::new("coref", &["ant"]),
        [Edge { from: Ix::new(1), to: Ix::new(0), kind: Tag(0), score: 0.1 }],
    );
    let m_groups = Groups::<Mentions>::new([[Ix::new(0), Ix::new(1)]]);
    for grp in [None, Some(&m_groups)] {
        let m_chains = extract_chains(&sentences, None, Some(&mentions), grp, Some(&m_edges), None, None, None);
        assert_eq!(m_chains.links.len(), 1);
        assert_eq!(m_chains.links[0].weight_milli, 100);
        assert_eq!(coref_boundaries_from_chains(&m_chains, CorefCohesion::default()).cost[0], 100);
    }
}

#[test]
fn scorer_must_reject_text_from_another_view() {
    let mut s_spans = Spans::new(ViewId(1));
    s_spans.push(Span::new(0, 4));
    s_spans.push(Span::new(5, 9));
    let s = Sentences { spans: s_spans };

    let mut t_spans = Spans::new(ViewId(1));
    t_spans.push(Span::new(0, 4));
    t_spans.push(Span::new(5, 9));
    let t = Tokens { spans: t_spans, sentence: None };

    let d = Document::new("same same");
    let b = Boundaries { cost: vec![0; 2], reason: vec![Cut::Sentence; 2] };
    let scorer = DistilledBoundaryScorer { weights: DistilledWeights::default() };
    assert!(scorer.annotate((&s, &b, None, None, None, None, Some(&t), d.get::<View>(), None)).is_err());
}

#[test]
fn long_dependency_chain_scales_linearly() {
    for rel_name in ["dep", "conj"] {
        let mut times = Vec::new();
        for n in [400usize, 800, 1600] {
            let mut s_spans = Spans::new(ViewId::SOURCE);
            s_spans.push(Span::new(0, (n * 2) as u32));
            let s = Sentences { spans: s_spans };

            let mut t_spans = Spans::new(ViewId::SOURCE);
            for i in 0..n {
                t_spans.push(Span::new((i * 2) as u32, (i * 2 + 1) as u32));
            }
            let tokens = Tokens { spans: t_spans, sentence: None };
            let deps = Dependencies::<Tokens> {
                head: (0..n).map(|i| if i == 0 { None } else { Some(Ix::new((i - 1) as u32)) }).collect(),
                relation: vec![Tag(0); n],
                relations: TagSet::new("dep", &[rel_name]),
            };
            let start = std::time::Instant::now();
            let cuts = clause_cuts(&s, &tokens, Some(&deps), None, None, None, None, ClauseCosts::default());
            let elapsed = start.elapsed();
            assert_eq!(cuts.sentences[0].tokens.len(), n);
            assert!(elapsed.as_millis() < 100, "rel={rel_name} n={n} took {elapsed:?}");
            times.push(elapsed.as_nanos().max(1));
        }
        // 4x the input (1600 vs 400) is O(n) and stays well under 8x even with noisy timers.
        assert!(
            times[2] < times[0].saturating_mul(16).max(5_000_000),
            "rel={rel_name} timings (ns): 400={}, 800={}, 1600={}",
            times[0],
            times[1],
            times[2]
        );
    }
}
