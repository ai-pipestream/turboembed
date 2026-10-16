//! Integration tests for sentiment, aspect, and incivility boundary signals
//! (`SentimentBoundaries`, `WeighAllBoundaries`, `SentimentBoundariesProvider`,
//! `WeighProvider`) and per-chunk facets (`ChunkFacets`, `ChunkFacetBuilder`).

use turbo_analysis::annotate::Step;
use turbo_analysis::registry::{Registry, Spec};
use turbo_analysis::*;
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Boundaries, Categories, Chunks, Cut, Incivility, IncivilityItem, IncivilityLabel,
    IncivilityScores, IncivilityTarget, Polarity, PolarityScores, Sentences, Strs, Subwords, Tokens,
};
use turbo_chunker::{
    Budget, ChunkFacetBuilder, ChunkFacets, Chunker, DriftProvider, SENTIMENT, SentimentBoundariesProvider,
    StructureBoundariesProvider, WeighProvider, boundary_similarity, chunk_boundary_gaps,
};

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

fn registry() -> Registry<dyn Step> {
    let mut r = Registry::default();
    r.register(StructureBoundariesProvider)
        .register(DriftProvider)
        .register(SentimentBoundariesProvider)
        .register(WeighProvider);
    r
}

fn build_review_document(sents: &[&str], words_per_sent: u32) -> Document {
    let text = sents.join(" ");
    let mut doc = Document::new(text);
    let mut s_spans = Spans::new(ViewId::SOURCE);
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut tok_sent = Vec::new();
    let mut sw_spans = Spans::new(ViewId::SOURCE);
    let mut sw_ids = Vec::new();

    let mut cursor = 0u32;
    for (s_idx, s_str) in sents.iter().enumerate() {
        let len = s_str.len() as u32;
        s_spans.push(Span::new(cursor, cursor + len));
        tok_spans.push(Span::new(cursor, cursor + len));
        tok_sent.push(Ix::new(s_idx as u32));
        for k in 0..words_per_sent {
            let a = cursor + (k * len) / words_per_sent;
            let b = cursor + ((k + 1) * len) / words_per_sent;
            sw_spans.push(Span::new(a, b.max(a + 1).min(cursor + len)));
            sw_ids.push(0);
        }
        cursor += len + 1;
    }

    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: Some(tok_sent) }, Provenance::by("t")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Subwords { spans: sw_spans, ids: sw_ids, model: toy_model(), token: None },
        Provenance::by("t"),
    )
    .unwrap();
    doc
}

#[test]
fn sentiment_boundaries_provider_and_weigh_provider_validate_options() {
    let r = registry();
    assert!(r.select("sentiment-boundaries", &Spec::default().with("unknown", "1")).is_err());
    assert!(r.create("sentiment-boundaries", &Spec::default().with("window", "0")).is_err());
    assert!(r.create("sentiment-boundaries", &Spec::default().with("scale", "-1")).is_err());
    assert!(
        r.create(
            "sentiment-boundaries",
            &Spec::default()
                .with("window", "2")
                .with("polarity_weight", "1000")
                .with("aspect_weight", "1000")
                .with("incivility_weight", "500")
                .with("floor", "100000")
                .with("scale", "1500")
                .with("shift", "400000"),
        )
        .is_ok()
    );
    assert!(r.create("weigh-boundaries", &Spec::default().with("sentiment", "1000")).is_ok());
    assert!(r.create("weigh-boundaries", &Spec::default().with("sentiment", "not-an-int")).is_err());
}

#[test]
fn registry_pipeline_improves_boundary_similarity_and_single_chunk_span_containment() {
    // 6-sentence review:
    // Sentences 0..3: positive about battery life (gold span 0: sentences 0..3).
    // Sentences 3..6: negative about customer support with uncivil insult in sentence 4 (gold span 1: sentences 3..6).
    let sents = [
        "The battery lasted two full days on a single charge.",
        "Fast charging reached eighty percent in thirty minutes.",
        "Power efficiency remained steady during heavy travel.",
        "Customer support hung up on me three times yesterday.",
        "The rude agent shouted pathetic insults on the phone.",
        "Warranty replacement tickets were closed without any reply.",
    ];
    let mut base_doc = build_review_document(&sents, 10);

    let cat_set = TagSet::new("sentiment", &["positive", "negative", "neutral"]);
    let mut cats = Categories::<Sentences>::with_capacity(cat_set, 6);
    for &(best, scores) in &[
        (Tag(0), [0.93, 0.04, 0.03]),
        (Tag(0), [0.91, 0.05, 0.04]),
        (Tag(0), [0.90, 0.05, 0.05]),
        (Tag(1), [0.04, 0.92, 0.04]),
        (Tag(1), [0.02, 0.96, 0.02]),
        (Tag(1), [0.05, 0.91, 0.04]),
    ] {
        cats.push(best, &scores);
    }

    let asp_set = TagSet::new("aspect", &["battery", "support"]);
    let mut aspects = Aspects::new(asp_set);
    for (s_idx, cat_tag, pol) in [
        (0u32, Tag(0), PolarityScores::new(0.92, 0.04, 0.04)),
        (1u32, Tag(0), PolarityScores::new(0.90, 0.05, 0.05)),
        (2u32, Tag(0), PolarityScores::new(0.89, 0.06, 0.05)),
        (3u32, Tag(1), PolarityScores::new(0.05, 0.91, 0.04)),
        (4u32, Tag(1), PolarityScores::new(0.03, 0.95, 0.02)),
        (5u32, Tag(1), PolarityScores::new(0.04, 0.92, 0.04)),
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
        scores: IncivilityScores { insult: 0.88, ..IncivilityScores::default() },
        evidence: &[(IncivilityLabel::Insult, IxRange::new(4, 5))],
    });

    base_doc.add(&Key::DEFAULT, cats, Provenance::by("t")).unwrap();
    base_doc.add(&Key::DEFAULT, aspects, Provenance::by("t")).unwrap();
    base_doc.add(&Key::DEFAULT, incivility, Provenance::by("t")).unwrap();

    // Each sentence has 10 subwords; budget max = 40, target = 30, size_weight = 0.
    // Without sentiment (sentiment = 0), earliest equal cut splits at gap 1 ([0..2, 2..6]),
    // which splits the 3-sentence battery span across two chunks!
    // With sentiment = 1000, the cut lands at gap 2 ([0..3, 3..6]), keeping both gold aspect spans intact.
    let budget = Budget { max: 40, target: 30, size_weight: 0, overhead: 0 };

    let run_with_weight = |sentiment_weight: &str| -> Document {
        let steps = [
            ("structure-boundaries", Spec::default().with("instance", "structure")),
            ("sentiment-boundaries", Spec::default().with("window", "1")),
            ("weigh-boundaries", Spec::default().with("structure", "1000").with("sentiment", sentiment_weight)),
        ];
        let analyzer = registry()
            .pipeline(&steps, |b| {
                b.given(Key::<View>::DEFAULT)
                    .given(Key::<Sentences>::DEFAULT)
                    .given(Key::<Tokens>::DEFAULT)
                    .given(Key::<Subwords>::DEFAULT)
                    .given(Key::<Categories<Sentences>>::DEFAULT)
                    .given(Key::<Aspects>::DEFAULT)
                    .given(Key::<Incivility>::DEFAULT)
                    .add(Chunker { budget })
                    .add(ChunkFacetBuilder)
            })
            .unwrap();
        let mut docs = vec![base_doc.clone()];
        analyzer.run(&mut docs).remove(0).unwrap();
        docs.remove(0)
    };

    let doc_off = run_with_weight("0");
    let doc_on = run_with_weight("1000");

    let chunks_off = doc_off.get::<Chunks>().unwrap();
    let chunks_on = doc_on.get::<Chunks>().unwrap();

    let gold_gaps = [2u32];
    let b_off = boundary_similarity(&gold_gaps, &chunk_boundary_gaps(chunks_off), 1);
    let b_on = boundary_similarity(&gold_gaps, &chunk_boundary_gaps(chunks_on), 1);
    assert!((b_off - 0.5).abs() < 1e-12);
    assert!((b_on - 1.0).abs() < 1e-12);

    // Check single-chunk evidence containment for the two gold aspect passages (sentences 0..3 and 3..6):
    let s_spans = &base_doc.get::<Sentences>().unwrap().spans;
    let gold_battery = Span::new(s_spans.get(Ix::new(0)).start, s_spans.get(Ix::new(2)).end);
    let gold_support = Span::new(s_spans.get(Ix::new(3)).start, s_spans.get(Ix::new(5)).end);

    let contained_count = |chunks: &Chunks| -> usize {
        [gold_battery, gold_support]
            .iter()
            .filter(|&&gold| chunks.spans.iter().any(|(_, cs)| cs.start <= gold.start && gold.end <= cs.end))
            .count()
    };
    // With sentiment = 0, gold_battery is split across chunk 0 (0..2) and chunk 1 (2..6), so only 1/2 is contained:
    assert_eq!(contained_count(chunks_off), 1);
    // With sentiment = 1000, both gold_battery (0..3) and gold_support (3..6) are 100% contained in pure chunks:
    assert_eq!(contained_count(chunks_on), 2);
    assert_eq!(chunks_on.cut, [Cut::DocumentStart, Cut::TopicShift]);

    // Verify the sentiment boundary provenance recorded on the default Boundaries layer:
    let prov = doc_on.provenance(Key::<Boundaries>::DEFAULT.id()).unwrap();
    assert!(prov.reads.contains(&Key::<Boundaries>::named(SENTIMENT).id()));

    // Verify facet filtering on `ChunkFacets`:
    let facets = doc_on.get::<ChunkFacets>().unwrap();
    assert_eq!(facets.chunk_polarity(Ix::new(0)), Polarity::Positive);
    assert_eq!(facets.chunk_polarity(Ix::new(1)), Polarity::Negative);
    assert_eq!(facets.incivility_labels(Ix::new(0)), &[]);
    assert_eq!(facets.incivility_labels(Ix::new(1)), &[IncivilityLabel::Insult]);
}

#[test]
fn chunk_facets_check_rejects_invalid_invariants() {
    let valid = ChunkFacets {
        polarity: vec![Polarity::Positive],
        polarity_score: vec![0.8],
        aspect_row: vec![0, 1],
        aspect: vec![Ix::new(0)],
        aspect_label: {
            let mut s = Strs::default();
            s.push("food");
            s
        },
        aspect_polarity: vec![Polarity::Positive],
        incivility_row: vec![0, 2],
        incivility_label: vec![IncivilityLabel::Insult, IncivilityLabel::Threat],
        incivility_max: vec![IncivilityScores::default()],
    };
    assert!(valid.check().is_ok());

    let mut bad_score_len = valid.clone();
    bad_score_len.polarity_score.pop();
    assert!(bad_score_len.check().is_err());

    let mut bad_aspect_row = valid.clone();
    bad_aspect_row.aspect_row = vec![0, 2];
    assert!(bad_aspect_row.check().is_err());

    let mut bad_incivility_order = valid.clone();
    bad_incivility_order.incivility_label = vec![IncivilityLabel::Threat, IncivilityLabel::Insult];
    assert!(bad_incivility_order.check().is_err());

    let mut bad_nan = valid;
    bad_nan.polarity_score[0] = f64::NAN;
    assert!(bad_nan.check().is_err());
}
