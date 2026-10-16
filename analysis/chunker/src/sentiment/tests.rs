use super::*;
use crate::{Budget, Chunker, Costs, STRUCTURE, StructureBoundaries, WeighAllBoundaries, Weight, weigh_with_sentiment};
use turbo_analysis::{Analyzer, Document, IxRange, Key, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Aspect, AspectTarget, Chunks, IncivilityItem, PolarityScores, Subwords};

fn build_doc_with_sentences_and_subwords(sents_text: &[&str], subwords_per_sent: u32) -> Document {
    let full = sents_text.join(" ");
    let mut doc = Document::new(full);
    let mut s_spans = Spans::new(ViewId::SOURCE);
    let mut sw_spans = Spans::new(ViewId::SOURCE);
    let mut sw_ids = Vec::new();
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut tok_sent = Vec::new();

    let mut cursor = 0u32;
    for (s_idx, s_str) in sents_text.iter().enumerate() {
        let len = s_str.len() as u32;
        s_spans.push(Span::new(cursor, cursor + len));
        tok_spans.push(Span::new(cursor, cursor + len));
        tok_sent.push(Ix::new(s_idx as u32));
        for k in 0..subwords_per_sent {
            let a = cursor + (k * len) / subwords_per_sent;
            let b = cursor + ((k + 1) * len) / subwords_per_sent;
            sw_spans.push(Span::new(a, b.max(a + 1).min(cursor + len)));
            sw_ids.push(1);
        }
        cursor += len + 1;
    }

    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: Some(tok_sent) }, Provenance::by("t")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Subwords {
            spans: sw_spans,
            ids: sw_ids,
            model: ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() },
            token: None,
        },
        Provenance::by("t"),
    )
    .unwrap();
    doc
}

#[test]
fn default_weigh_all_boundaries_keeps_sentiment_off() {
    let w = WeighAllBoundaries::default();
    assert_eq!(w.structure, Weight::ONE);
    assert_eq!(w.drift, Weight::ONE);
    assert_eq!(w.sentiment, Weight(0));

    let s = Boundaries { cost: vec![1000, 1000, 0], reason: vec![Cut::Sentence; 3] };
    let d = Boundaries { cost: vec![0, 0, 0], reason: vec![Cut::Sentence; 3] };
    let sent =
        Boundaries { cost: vec![-1000, -1000, 0], reason: vec![Cut::TopicShift, Cut::TopicShift, Cut::Sentence] };

    let off = weigh_with_sentiment(&s, &d, &sent, (w.structure, w.drift, w.sentiment)).unwrap();
    assert_eq!(off.cost, [1000, 1000, 0]);
    assert_eq!(off.reason, [Cut::Sentence, Cut::Sentence, Cut::Sentence]);

    let on = weigh_with_sentiment(&s, &d, &sent, (Weight::ONE, Weight::ONE, Weight::ONE)).unwrap();
    assert_eq!(on.cost, [0, 0, 0]);
    assert_eq!(on.reason, [Cut::TopicShift, Cut::TopicShift, Cut::Sentence]);
}

#[test]
fn sentiment_and_aspect_shift_places_boundary_and_populates_chunk_facets() {
    let sents_text = [
        "The handmade pasta was fresh and delicious.",
        "The risotto had wonderful texture and flavor.",
        "Our dessert was warm and plated with care.",
        "Then the waiter was rude and ignored our table.",
        "The manager yelled insults when we asked for water.",
        "Billing service was a complete disaster.",
    ];
    let mut doc = build_doc_with_sentences_and_subwords(&sents_text, 100);

    let cat_set = TagSet::new("sentiment", &["positive", "negative", "neutral"]);
    let mut cats = Categories::<Sentences>::with_capacity(cat_set, 6);
    for &(best, scores) in &[
        (Tag(0), [0.92, 0.04, 0.04]),
        (Tag(0), [0.90, 0.05, 0.05]),
        (Tag(0), [0.94, 0.03, 0.03]),
        (Tag(1), [0.05, 0.90, 0.05]),
        (Tag(1), [0.03, 0.94, 0.03]),
        (Tag(1), [0.04, 0.92, 0.04]),
    ] {
        cats.push(best, &scores);
    }

    let asp_cats = TagSet::new("aspect", &["food", "service"]);
    let mut aspects = Aspects::new(asp_cats);
    for (s_idx, cat_tag, pol) in [
        (0u32, Tag(0), PolarityScores::new(0.90, 0.05, 0.05)),
        (1u32, Tag(0), PolarityScores::new(0.88, 0.06, 0.06)),
        (2u32, Tag(0), PolarityScores::new(0.92, 0.04, 0.04)),
        (3u32, Tag(1), PolarityScores::new(0.05, 0.90, 0.05)),
        (4u32, Tag(1), PolarityScores::new(0.04, 0.92, 0.04)),
        (5u32, Tag(1), PolarityScores::new(0.03, 0.94, 0.03)),
    ] {
        aspects.push(Aspect {
            target: AspectTarget::Term { tokens: IxRange::new(s_idx, s_idx + 1), sentence: Some(Ix::new(s_idx)) },
            category: Some(cat_tag),
            polarity: pol,
            opinions: &[],
        });
    }

    let mut incivility = Incivility::new();
    incivility.push(IncivilityItem {
        target: IncivilityTarget::Sentence(Ix::new(4)),
        scores: IncivilityScores { insult: 0.85, profanity: 0.10, ..IncivilityScores::default() },
        evidence: &[(IncivilityLabel::Insult, IxRange::new(4, 5))],
    });

    doc.add(&Key::DEFAULT, cats, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, aspects, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, incivility, Provenance::by("t")).unwrap();

    let budget = Budget { max: 400, target: 300, size_weight: 0, overhead: 0 };

    // 1. Default pipeline (sentiment Weight(0), off by default):
    let mut doc_off = doc.clone();
    let analyzer_off = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Categories<Sentences>>::DEFAULT)
        .given(Key::<Aspects>::DEFAULT)
        .given(Key::<Incivility>::DEFAULT)
        .step(Bound::new(StructureBoundaries { costs: Costs::default() }).write(0, STRUCTURE))
        .step(SentimentBoundaries::default().bound())
        .step(WeighAllBoundaries::default().bound())
        .add(Chunker { budget })
        .add(ChunkFacetBuilder)
        .build()
        .unwrap();
    analyzer_off.run(std::slice::from_mut(&mut doc_off))[0].as_ref().unwrap();

    let chunks_off = doc_off.get::<Chunks>().unwrap();
    assert_eq!(chunk_boundary_gaps(chunks_off), vec![1]);

    // 2. Pipeline with sentiment boundary signal enabled (`sentiment: Weight::ONE`):
    let mut doc_on = doc;
    let analyzer_on = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Categories<Sentences>>::DEFAULT)
        .given(Key::<Aspects>::DEFAULT)
        .given(Key::<Incivility>::DEFAULT)
        .step(Bound::new(StructureBoundaries { costs: Costs::default() }).write(0, STRUCTURE))
        .step(SentimentBoundaries::default().bound())
        .step(WeighAllBoundaries { structure: Weight::ONE, drift: Weight::ONE, sentiment: Weight::ONE }.bound())
        .add(Chunker { budget })
        .add(ChunkFacetBuilder)
        .build()
        .unwrap();
    analyzer_on.run(std::slice::from_mut(&mut doc_on))[0].as_ref().unwrap();

    let sent_bounds = doc_on.get_key(&Key::<Boundaries>::named(SENTIMENT)).unwrap();
    assert_eq!(sent_bounds.cost[0], 0);
    assert_eq!(sent_bounds.cost[1], 0);
    assert!(sent_bounds.cost[2] < -800, "expected strong negative cost at gap 2, got {}", sent_bounds.cost[2]);
    assert_eq!(sent_bounds.reason[2], Cut::TopicShift);

    let chunks_on = doc_on.get::<Chunks>().unwrap();
    assert_eq!(chunks_on.len(), 2);
    assert_eq!(chunks_on.sentences[0], IxRange::new(0, 3));
    assert_eq!(chunks_on.sentences[1], IxRange::new(3, 6));
    assert_eq!(chunks_on.cut[1], Cut::TopicShift);

    let gold_gaps = [2u32];
    let b_sim_off = boundary_similarity(&gold_gaps, &chunk_boundary_gaps(chunks_off), 1);
    let b_sim_on = boundary_similarity(&gold_gaps, &chunk_boundary_gaps(chunks_on), 1);
    assert!((b_sim_off - 0.5).abs() < 1e-12, "without signal, cut at 1 vs gold 2 is a near miss (0.5): {b_sim_off}");
    assert!((b_sim_on - 1.0).abs() < 1e-12, "with signal, cut at 2 vs gold 2 is exact (1.0): {b_sim_on}");

    let facets = doc_on.get::<ChunkFacets>().unwrap();
    assert_eq!(facets.len(), 2);
    assert_eq!(facets.chunk_polarity(Ix::new(0)), Polarity::Positive);
    assert!(facets.chunk_polarity_score(Ix::new(0)) > 0.80);
    let c0_aspects: Vec<_> = facets.aspects_of(Ix::new(0)).map(|(_, l, p)| (l, p)).collect();
    assert_eq!(
        c0_aspects,
        vec![("food", Polarity::Positive), ("food", Polarity::Positive), ("food", Polarity::Positive),]
    );
    assert!(facets.incivility_labels(Ix::new(0)).is_empty());

    assert_eq!(facets.chunk_polarity(Ix::new(1)), Polarity::Negative);
    assert!(facets.chunk_polarity_score(Ix::new(1)) < -0.80);
    let c1_aspects: Vec<_> = facets.aspects_of(Ix::new(1)).map(|(_, l, p)| (l, p)).collect();
    assert_eq!(
        c1_aspects,
        vec![("service", Polarity::Negative), ("service", Polarity::Negative), ("service", Polarity::Negative),]
    );
    assert_eq!(facets.incivility_labels(Ix::new(1)), &[IncivilityLabel::Insult]);
    assert!((facets.incivility_scores(Ix::new(1)).insult - 0.85).abs() < 1e-9);
}

#[test]
fn boundary_similarity_scores_exact_near_and_missed_cuts() {
    assert!((boundary_similarity(&[], &[], 1) - 1.0).abs() < 1e-12);
    assert!((boundary_similarity(&[2, 5], &[2, 5], 1) - 1.0).abs() < 1e-12);
    assert!((boundary_similarity(&[2, 5], &[2, 6], 1) - 0.75).abs() < 1e-12);
    assert!((boundary_similarity(&[2, 5], &[2, 8], 1) - (1.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn resolve_aspect_label_handles_non_utf8_boundary_spans_without_panicking() {
    // "café" has 'é' at byte range 3..5. A malformed token span 0..4 ends inside 'é'.
    let doc = Document::new("café service");
    let view = doc.get::<View>().unwrap();
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    tok_spans.push(Span::new(0, 4));
    let toks = Tokens { spans: tok_spans, sentence: Some(vec![Ix::new(0)]) };

    let mut aspects = Aspects::new(TagSet::new("aspect", &[]));
    let a_ix = aspects.push(Aspect {
        target: AspectTarget::Term { tokens: IxRange::new(0, 1), sentence: Some(Ix::new(0)) },
        category: None,
        polarity: PolarityScores::new(0.8, 0.1, 0.1),
        opinions: &[],
    });

    assert_eq!(resolve_aspect_label(a_ix, &aspects, Some(&toks), Some(view)), "");
}

#[test]
fn annotators_reject_mismatched_views_and_layer_lengths() {
    let doc = build_doc_with_sentences_and_subwords(&["First sentence.", "Second sentence."], 4);
    let sents = doc.get::<Sentences>().unwrap();
    let subwords = doc.get::<Subwords>().unwrap();

    // 1. Chunker rejects Boundaries whose length does not match Sentences.
    let short_bounds = Boundaries { cost: vec![0], reason: vec![Cut::Sentence] };
    let chunker = Chunker { budget: Budget { max: 100, target: 50, size_weight: 100, overhead: 0 } };
    let err = chunker.annotate((sents, subwords, &short_bounds, None)).err().unwrap();
    assert!(err.to_string().contains("boundary costs"), "unexpected error: {err}");

    // 2. SentimentBoundaries rejects Tokens in a different view.
    let foreign_toks = Tokens { spans: Spans::new(ViewId(7)), sentence: None };
    let err = SentimentBoundaries::default()
        .annotate((sents, doc.get::<View>(), Some(&foreign_toks), None, None, None))
        .err()
        .unwrap();
    assert!(err.to_string().contains("different views"), "unexpected error: {err}");

    // 3. ChunkFacetBuilder rejects Chunks in a different view and mismatched Categories length.
    let foreign_chunks = Chunks {
        spans: Spans::new(ViewId(7)),
        level: Vec::new(),
        parent: Vec::new(),
        sentences: Vec::new(),
        cut: Vec::new(),
        cost: Vec::new(),
        model_tokens: Vec::new(),
        over_budget: Vec::new(),
    };
    let err =
        ChunkFacetBuilder.annotate((&foreign_chunks, sents, doc.get::<View>(), None, None, None, None)).err().unwrap();
    assert!(err.to_string().contains("different views"), "unexpected error: {err}");

    let valid_bounds = Boundaries { cost: vec![0, 0], reason: vec![Cut::Sentence; 2] };
    let (chunks,) = chunker.annotate((sents, subwords, &valid_bounds, None)).unwrap();
    let short_cats = Categories::<Sentences>::with_capacity(TagSet::new("sentiment", &["pos"]), 1);
    let err = ChunkFacetBuilder
        .annotate((&chunks, sents, doc.get::<View>(), None, Some(&short_cats), None, None))
        .err()
        .unwrap();
    assert!(err.to_string().contains("sentence categories"), "unexpected error: {err}");
}

#[test]
fn multi_level_chunk_facets_preserve_aspect_order_and_merge_chunk_incivility() {
    let doc = build_doc_with_sentences_and_subwords(&["First sentence.", "Second sentence."], 4);
    let sents = doc.get::<Sentences>().unwrap();

    // Parent chunk (level 1, sentences 0..2) + two child chunks (level 0, sentences 0..1 and 1..2).
    let mut c_spans = Spans::new(ViewId::SOURCE);
    c_spans.push(Span::new(0, 31));
    c_spans.push(Span::new(0, 15));
    c_spans.push(Span::new(16, 31));
    let chunks = Chunks {
        spans: c_spans,
        level: vec![1, 0, 0],
        parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
        sentences: vec![IxRange::new(0, 2), IxRange::new(0, 1), IxRange::new(1, 2)],
        cut: vec![Cut::DocumentStart, Cut::DocumentStart, Cut::Sentence],
        cost: vec![0, 0, 0],
        model_tokens: vec![8, 4, 4],
        over_budget: vec![false, false, false],
    };

    // Push aspect in sentence 1 first, then aspect in sentence 0 (out of sentence order).
    let asp_cats = TagSet::new("aspect", &["price", "quality"]);
    let mut aspects = Aspects::new(asp_cats);
    let a0 = aspects.push(Aspect {
        target: AspectTarget::Term { tokens: IxRange::new(1, 2), sentence: Some(Ix::new(1)) },
        category: Some(Tag(0)),
        polarity: PolarityScores::new(0.1, 0.8, 0.1),
        opinions: &[],
    });
    let a1 = aspects.push(Aspect {
        target: AspectTarget::Term { tokens: IxRange::new(0, 1), sentence: Some(Ix::new(0)) },
        category: Some(Tag(1)),
        polarity: PolarityScores::new(0.9, 0.05, 0.05),
        opinions: &[],
    });

    let mut inc = Incivility::new();
    inc.push(IncivilityItem {
        target: IncivilityTarget::Chunk(Ix::new(0)),
        scores: IncivilityScores { threat: 0.75, ..IncivilityScores::default() },
        evidence: &[],
    });
    inc.push(IncivilityItem {
        target: IncivilityTarget::Sentence(Ix::new(1)),
        scores: IncivilityScores { insult: 0.80, ..IncivilityScores::default() },
        evidence: &[],
    });

    let (facets,) = ChunkFacetBuilder
        .annotate((&chunks, sents, doc.get::<View>(), doc.get::<Tokens>(), None, Some(&aspects), Some(&inc)))
        .unwrap();
    facets.check().unwrap();
    // Parent chunk (0..2) must preserve `a_ix` order [a0, a1] and merge both chunk and sentence incivility.
    assert_eq!(facets.aspect_ixs(Ix::new(0)), &[a0, a1]);
    assert_eq!(facets.incivility_labels(Ix::new(0)), &[IncivilityLabel::Insult, IncivilityLabel::Threat]);
    assert_eq!(facets.aspect_ixs(Ix::new(1)), &[a1]);
    assert_eq!(facets.aspect_ixs(Ix::new(2)), &[a0]);
}
