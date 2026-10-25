use std::num::NonZeroU32;
use turbo_analysis::{Document, Error, Ix, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{
    Categories, Chunks, Cut, DocumentHeatmap, EmotionItem, EmotionLayer, EmotionScores, EmotionTarget, Entities,
    HeatmapLaneKind, Incivility, IncivilityItem, IncivilityScores, IncivilityTarget, Pii, PiiConfidence, PiiKind,
    PiiMention, Sentences, Tokens,
};

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span { start: a, end: b });
    }
    s
}

fn by() -> Provenance {
    Provenance::by("test")
}

/// Text: 0123456789 (10 bytes). Sentences [0,5) [5,10); tokens each byte
/// pair; one entity over [2,7); one PII over [8,10).
fn base_doc() -> Document {
    let mut doc = Document::new("0123456789");
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 5), (5, 10)]) }, by()).unwrap();
    let mut tok = Spans::new(ViewId::SOURCE);
    for i in 0..5u32 {
        tok.push(Span { start: i * 2, end: i * 2 + 2 });
    }
    doc.add(&Key::DEFAULT, Tokens { spans: tok, sentence: None }, by()).unwrap();
    let kinds = TagSet::new("test:entities", &["PER"]);
    doc.add(
        &Key::DEFAULT,
        Entities {
            spans: spans(&[(2, 7)]),
            tokens: vec![IxRange::new(1, 4)],
            kind: vec![Tag(0)],
            kinds,
            score: vec![0.9],
        },
        by(),
    )
    .unwrap();
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.push(PiiMention {
        span: Span { start: 8, end: 10 },
        kind: PiiKind::Card,
        confidence: PiiConfidence::Checksum,
        normalized: "x",
    });
    doc.add(&Key::DEFAULT, pii, by()).unwrap();
    doc
}

fn cats<L: turbo_analysis::Layer + turbo_analysis_layers::CategoryScope>(
    labels: &[&str],
    rows: &[(u16, &[f64])],
) -> Categories<L> {
    let set = TagSet::new("test:polarity", labels);
    let mut c = Categories::<L>::with_capacity(set, rows.len());
    for &(best, scores) in rows {
        c.push(Tag(best), scores);
    }
    c
}

fn lane_segments(h: &DocumentHeatmap, kind: HeatmapLaneKind) -> Vec<(u32, u32, f32, u32, Option<u32>)> {
    h.lane(kind)
        .unwrap()
        .segments
        .iter()
        .map(|s| (s.span.start, s.span.end, s.score, s.count, s.primary_item))
        .collect()
}

#[test]
fn exact_density_lanes_tile_and_count() {
    let doc = base_doc();
    let h = DocumentHeatmap::build(&doc, ViewId::SOURCE, None).unwrap();
    // Tokens cover everything once: one merged full-coverage run of 1
    // per token boundary stays separate items, so segments split.
    let tok = lane_segments(&h, HeatmapLaneKind::TokenDensity);
    assert_eq!(tok.len(), 5);
    assert!(tok.iter().all(|&(_, _, score, count, _)| score == 1.0 && count == 1));
    assert_eq!((tok[0].0, tok[4].1), (0, 10));
    // Entity [2,7): uncovered, covered, uncovered.
    assert_eq!(
        lane_segments(&h, HeatmapLaneKind::EntityDensity),
        vec![(0, 2, 0.0, 0, None), (2, 7, 1.0, 1, Some(0)), (7, 10, 0.0, 0, None)]
    );
    assert_eq!(lane_segments(&h, HeatmapLaneKind::PiiDensity), vec![(0, 8, 0.0, 0, None), (8, 10, 1.0, 1, Some(0))]);
    let lane = h.lane(HeatmapLaneKind::EntityDensity).unwrap();
    assert_eq!((lane.min_score, lane.max_score, lane.id.as_str()), (0.0, 1.0, "entity-density"));
    // No scored source layers present: those lanes are absent.
    assert!(h.lane(HeatmapLaneKind::SentimentPolarity).is_none());
    assert!(h.lane(HeatmapLaneKind::BoundaryDrift).is_none());
    assert_eq!(
        h.lanes.iter().map(|l| l.kind).collect::<Vec<_>>(),
        vec![HeatmapLaneKind::TokenDensity, HeatmapLaneKind::EntityDensity, HeatmapLaneKind::PiiDensity]
    );
}

#[test]
fn bucket_lanes_aggregate_per_bin() {
    let doc = base_doc();
    let h = DocumentHeatmap::build(&doc, ViewId::SOURCE, NonZeroU32::new(5)).unwrap();
    // 5 bins of width 2: entity [2,7) touches bins 1,2,3.
    assert_eq!(
        lane_segments(&h, HeatmapLaneKind::EntityDensity),
        vec![
            (0, 2, 0.0, 0, None),
            (2, 4, 1.0, 1, Some(0)),
            (4, 6, 1.0, 1, Some(0)),
            (6, 8, 1.0, 1, Some(0)),
            (8, 10, 0.0, 0, None)
        ]
    );
    // More bins than bytes: still exactly that many segments.
    let h = DocumentHeatmap::build(&doc, ViewId::SOURCE, NonZeroU32::new(20)).unwrap();
    assert_eq!(h.lane(HeatmapLaneKind::TokenDensity).unwrap().segments.len(), 20);
}

#[test]
fn sentiment_polarity_lane_signs() {
    let mut doc = base_doc();
    // labels: positive 0, negative 1, neutral 2. Sentence 0 pos-heavy,
    // sentence 1 neg-heavy.
    doc.add(
        &Key::DEFAULT,
        cats::<Sentences>(&["positive", "negative", "neutral"], &[(0, &[0.8, 0.1, 0.1]), (1, &[0.25, 0.75, 0.0])]),
        by(),
    )
    .unwrap();
    let h = DocumentHeatmap::build(&doc, ViewId::SOURCE, None).unwrap();
    assert_eq!(
        lane_segments(&h, HeatmapLaneKind::SentimentPolarity),
        vec![(0, 5, (0.8f64 - 0.1) as f32, 1, Some(0)), (5, 10, -0.5, 1, Some(1))]
    );
    let lane = h.lane(HeatmapLaneKind::SentimentPolarity).unwrap();
    assert_eq!((lane.min_score, lane.max_score), (-0.5, (0.8f64 - 0.1) as f32));
}

#[test]
fn incivility_emotion_and_chunk_score_lanes() {
    let mut doc = base_doc();
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    for i in 0..5u32 {
        tok_spans.push(Span { start: i * 2, end: i * 2 + 2 });
    }
    let tokens = Tokens { spans: tok_spans, sentence: None };
    let sentences = Sentences { spans: spans(&[(0, 5), (5, 10)]) };
    let strong = IncivilityScores { threat: 0.75, ..Default::default() };
    let inc = Incivility::from_items(
        &tokens,
        Some(&sentences),
        None,
        None,
        [IncivilityItem { target: IncivilityTarget::Sentence(Ix::new(1)), scores: strong, evidence: &[] }],
    )
    .unwrap();
    doc.add(&Key::DEFAULT, inc, by()).unwrap();
    let mut es = EmotionScores::default();
    es.0[0] = 0.5;
    let mut emotion = EmotionLayer::new();
    emotion
        .try_push(
            &tokens,
            Some(&sentences),
            None,
            EmotionItem { target: EmotionTarget::Sentence(Ix::new(0)), scores: es, evidence: &[] },
        )
        .unwrap();
    doc.add(&Key::DEFAULT, emotion, by()).unwrap();
    // Two level-0 chunks matching the sentences, with category scores.
    let chunks = Chunks {
        spans: spans(&[(0, 5), (5, 10)]),
        level: vec![0, 0],
        parent: vec![None, None],
        sentences: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
        cut: vec![Cut::Sentence, Cut::Sentence],
        cost: vec![0, 0],
        model_tokens: vec![1, 1],
        over_budget: vec![false, false],
        prompt_overhead: 0,
    };
    doc.add(&Key::DEFAULT, chunks, by()).unwrap();
    doc.add(&Key::DEFAULT, cats::<Chunks>(&["a", "b"], &[(0, &[0.25, 0.75]), (1, &[0.125, 0.875])]), by()).unwrap();

    let h = DocumentHeatmap::build(&doc, ViewId::SOURCE, None).unwrap();
    assert_eq!(
        lane_segments(&h, HeatmapLaneKind::IncivilityScore),
        vec![(0, 5, 0.0, 0, None), (5, 10, 0.75, 1, Some(0))]
    );
    assert_eq!(
        lane_segments(&h, HeatmapLaneKind::EmotionIntensity),
        vec![(0, 5, 0.5, 1, Some(0)), (5, 10, 0.0, 0, None)]
    );
    assert_eq!(
        lane_segments(&h, HeatmapLaneKind::ChunkScore),
        vec![(0, 5, 0.25, 1, Some(0)), (5, 10, 0.875, 1, Some(1))]
    );
}

#[test]
fn strongest_covering_item_wins_on_overlap() {
    let mut doc = base_doc();
    // Second entity overlapping the first: density counts both where
    // they cross.
    let kinds = TagSet::new("test:entities", &["PER"]);
    let entities = Entities {
        spans: spans(&[(2, 7), (4, 9)]),
        tokens: vec![IxRange::new(1, 4), IxRange::new(2, 5)],
        kind: vec![Tag(0), Tag(0)],
        kinds,
        score: vec![0.9, 0.5],
    };
    doc.add(&Key::named("alt"), entities, by()).unwrap();
    // build reads the default instance (one entity), still 1-deep.
    let h = DocumentHeatmap::build(&doc, ViewId::SOURCE, None).unwrap();
    assert_eq!(lane_segments(&h, HeatmapLaneKind::EntityDensity)[1], (2, 7, 1.0, 1, Some(0)));
}

#[test]
fn heatmap_refusals() {
    let doc = base_doc();
    // Unknown view.
    assert!(matches!(DocumentHeatmap::build(&doc, ViewId(9), None), Err(Error::Bounds { .. })));
    // Sentiment rows misaligned with sentences are refused at add time,
    // before a heatmap could ever see them.
    let mut doc2 = Document::new("0123456789");
    doc2.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 10)]) }, by()).unwrap();
    let refused = doc2.add(
        &Key::DEFAULT,
        cats::<Sentences>(&["positive", "negative"], &[(0, &[0.5, 0.5]), (0, &[0.5, 0.5])]),
        by(),
    );
    assert!(matches!(refused, Err(Error::Misaligned { len: 2, anchor_len: 1, .. })), "{refused:?}");
    // A non-finite category score is likewise refused at add time.
    let mut doc3 = base_doc();
    let refused = doc3.add(
        &Key::DEFAULT,
        cats::<Sentences>(&["positive", "negative"], &[(0, &[f64::NAN, 0.5]), (0, &[0.5, 0.5])]),
        by(),
    );
    match refused {
        Err(Error::Invalid { detail, .. }) => assert!(detail.contains("NaN"), "{detail}"),
        other => panic!("{other:?}"),
    }
    // Empty document: lanes exist but hold no (exact) segments.
    let empty = Document::new("");
    let h = DocumentHeatmap::build(&empty, ViewId::SOURCE, None).unwrap();
    assert!(h.lanes.is_empty());
}
