use std::num::NonZeroU32;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{
    Document, Error, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{Categories, Chunks, Cut, DocumentHeatmap, HeatmapLaneKind, Sentences, Tokens, Vectors};

fn spans<L>(view: ViewId, pairs: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(view);
    for &(a, b) in pairs {
        out.push(Span::new(a, b));
    }
    out
}

fn transport(doc: &Document) -> Document {
    let codecs = turbo_analysis_layers::export::codecs();
    import(&DocumentGraph::from_bytes(&export(doc, &codecs).unwrap().to_bytes().unwrap()).unwrap(), &codecs).unwrap()
}

fn cats(best: Tag) -> Categories<Sentences> {
    let mut out = Categories::with_capacity(TagSet::new("review:sentiment", &["positive", "negative"]), 1);
    out.push(best, &[0.9, 0.1]);
    out
}

#[test]
fn scored_rows_follow_actual_named_view_and_fragments() {
    let mut doc = Document::new("abcd");
    let source = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.keep(4);
    let (text, alignment) = b.finish();
    doc.add(&Key::<View>::named("other"), View::derived(&source, text, alignment), Provenance::by("review:view"))
        .unwrap();
    doc.add(
        &Key::<Sentences>::DEFAULT,
        Sentences { spans: spans(ViewId::SOURCE, &[(0, 4)]) },
        Provenance::by("review:default"),
    )
    .unwrap();
    let mut fragments = Spans::new(ViewId(1));
    fragments.push(Span::new(0, 4));
    let gold = Key::<Sentences>::named("gold");
    doc.add(&gold, Sentences { spans: fragments }, Provenance::by("review:gold")).unwrap();
    let key = Key::<Categories<Sentences>>::DEFAULT;
    doc.add(&key, cats(Tag(0)), Provenance { on: vec![gold.id()], ..Provenance::by("review:scorer") }).unwrap();
    assert_eq!(doc.made_on(key.id()), &[gold.id()]);
    for doc in [&doc, &transport(&doc)] {
        for buckets in [None, NonZeroU32::new(4)] {
            let result = DocumentHeatmap::build(doc, ViewId::SOURCE, buckets);
            assert!(
                matches!(result, Err(Error::Bounds { .. })),
                "other-view named rows laid on default envelope: {result:?}"
            );
            let good = DocumentHeatmap::build(doc, ViewId(1), buckets).unwrap();
            let lane = good.lane(HeatmapLaneKind::SentimentPolarity).unwrap();
            assert_eq!(lane.view, ViewId(1));
            assert!(lane.segments.iter().any(|s| s.count > 0));
        }
    }
}

#[test]
fn scored_chunk_rows_follow_named_fragments_with_equal_envelopes() {
    let mut doc = Document::new("abcd");
    doc.add(
        &Key::<Sentences>::DEFAULT,
        Sentences { spans: spans(ViewId::SOURCE, &[(0, 4)]) },
        Provenance::by("review:s"),
    )
    .unwrap();
    for (key, fragments) in [
        (Key::<Chunks>::DEFAULT, vec![Span::new(0, 4)]),
        (Key::<Chunks>::named("gold"), vec![Span::new(0, 1), Span::new(3, 4)]),
    ] {
        let mut items = Spans::new(ViewId::SOURCE);
        items.try_push_fragments(&fragments).unwrap();
        doc.add(
            &key,
            Chunks {
                spans: items,
                level: vec![0],
                parent: vec![None],
                sentences: vec![IxRange::new(0, 1)],
                cut: vec![Cut::Sentence],
                cost: vec![0],
                model_tokens: vec![1],
                over_budget: vec![false],
                prompt_overhead: 0,
            },
            Provenance::by("review:chunks"),
        )
        .unwrap();
    }
    let mut scores = Categories::<Chunks>::with_capacity(TagSet::new("review:score", &["quality"]), 1);
    scores.push(Tag(0), &[0.75]);
    doc.add(
        &Key::<Categories<Chunks>>::DEFAULT,
        scores,
        Provenance { on: vec![Key::<Chunks>::named("gold").id()], ..Provenance::by("review:score") },
    )
    .unwrap();
    for doc in [&doc, &transport(&doc)] {
        for buckets in [None, NonZeroU32::new(4)] {
            let h = DocumentHeatmap::build(doc, ViewId::SOURCE, buckets).unwrap();
            let lane = h.lane(HeatmapLaneKind::ChunkScore).unwrap();
            assert!(
                lane.segments.iter().filter(|s| s.count > 0).all(|s| s.span.end <= 1 || s.span.start >= 3),
                "named fragment gap was scored using default envelope: {lane:?}"
            );
        }
    }
}

fn density_doc(text: &str, pairs: &[(u32, u32)]) -> Document {
    let mut doc = Document::new(text);
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: spans(ViewId::SOURCE, pairs), sentence: None },
        Provenance::by("review:tokens"),
    )
    .unwrap();
    doc
}

fn assert_bucket_oracle(doc: &Document, n: u32) {
    let h = DocumentHeatmap::build(doc, ViewId::SOURCE, NonZeroU32::new(n)).unwrap();
    let lane = h.lane(HeatmapLaneKind::TokenDensity).unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    for seg in &lane.segments {
        let expected =
            tokens.spans.iter().filter(|(_, s)| s.start < seg.span.end && seg.span.start < s.end).count() as u32;
        assert_eq!(seg.count, expected, "bin {:?} does not overlap the counted items; bins={n}", seg.span);
        assert_eq!(seg.score, expected as f32);
    }
}

#[test]
fn unequal_bucket_edges_use_actual_overlap() {
    let doc = density_doc("0123456789", &[(3, 4)]);
    for doc in [&doc, &transport(&doc)] {
        assert_bucket_oracle(doc, 3);
    }
}

#[test]
fn more_buckets_than_bytes_do_not_count_empty_bins() {
    let doc = density_doc("ab", &[(0, 1)]);
    assert_bucket_oracle(&doc, 4);
}

#[test]
fn abstained_sentiment_has_no_score_or_evidence_count() {
    let mut doc = Document::new("abcd");
    doc.add(
        &Key::<Sentences>::DEFAULT,
        Sentences { spans: spans(ViewId::SOURCE, &[(0, 4)]) },
        Provenance::by("review:sentences"),
    )
    .unwrap();
    doc.add(&Key::<Categories<Sentences>>::DEFAULT, cats(Tag::NONE), Provenance::by("review:abstained")).unwrap();
    for buckets in [None, NonZeroU32::new(2)] {
        let h = DocumentHeatmap::build(&doc, ViewId::SOURCE, buckets).unwrap();
        let lane = h.lane(HeatmapLaneKind::SentimentPolarity).unwrap();
        assert!(
            lane.segments.iter().all(|s| s.count == 0 && s.score == 0.0 && s.primary_item.is_none()),
            "abstention became evidence: {lane:?}"
        );
    }
}

#[test]
fn original_same_envelope_other_view_and_numeric_refusals() {
    let mut doc = density_doc("é😀", &[(0, 2), (2, 6)]);
    let source = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.keep(6);
    let (text, alignment) = b.finish();
    doc.add(&Key::<View>::named("same"), View::derived(&source, text, alignment), Provenance::by("review:view"))
        .unwrap();
    for buckets in [None, NonZeroU32::new(3)] {
        assert!(matches!(DocumentHeatmap::build(&doc, ViewId(1), buckets), Err(Error::Bounds { .. })));
        assert!(matches!(DocumentHeatmap::build(&doc, ViewId(99), buckets), Err(Error::Bounds { .. })));
    }
    for score in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -1.0] {
        let mut d = Document::new("a");
        d.add(
            &Key::<Sentences>::DEFAULT,
            Sentences { spans: spans(ViewId::SOURCE, &[(0, 1)]) },
            Provenance::by("review:s"),
        )
        .unwrap();
        let mut c = cats(Tag(0));
        c.scores[0] = score;
        if d.add(&Key::<Categories<Sentences>>::DEFAULT, c, Provenance::by("review:c")).is_ok() {
            for buckets in [None, NonZeroU32::new(2)] {
                assert!(matches!(DocumentHeatmap::build(&d, ViewId::SOURCE, buckets), Err(Error::Invalid { .. })));
            }
        }
    }
}

#[test]
fn vector_partial_rows_and_empty_or_malformed_graphs() {
    let mut doc = density_doc("ab", &[(0, 1), (1, 2)]);
    let model = ModelRef { id: "review:vec".into(), revision: "1".into(), sha256: "none".into() };
    let v = Vectors::<Tokens>::new(2, vec![1.0, 0.0], vec![], model.clone());
    assert!(matches!(
        doc.add(&Key::<Vectors<Tokens>>::DEFAULT, v, Provenance::by("review:partial")),
        Err(Error::Misaligned { .. })
    ));
    let v = Vectors::<Tokens>::new(2, vec![1.0, 0.0, 0.0, 0.0], vec![Ix::new(1)], model);
    doc.add(&Key::<Vectors<Tokens>>::DEFAULT, v, Provenance::by("review:complete")).unwrap();
    for buckets in [None, NonZeroU32::new(2)] {
        let h = DocumentHeatmap::build(&transport(&doc), ViewId::SOURCE, buckets).unwrap();
        assert_eq!(h.lanes.iter().map(|l| l.kind).collect::<Vec<_>>(), [HeatmapLaneKind::TokenDensity]);
        assert!(h.lanes[0].segments.iter().all(|s| s.count == 1 && s.score == 1.0 && s.score.is_finite()));
    }
    for bytes in [vec![], vec![255, 0, 1]] {
        assert!(DocumentGraph::from_bytes(&bytes).is_err());
    }
    for buckets in [None, NonZeroU32::new(3)] {
        assert!(DocumentHeatmap::build(&Document::new(""), ViewId::SOURCE, buckets).unwrap().lanes.is_empty());
    }
}

#[test]
fn bucket_counts_and_primary_match_overlap_oracle() {
    let doc = density_doc("0123456789", &[(0, 1), (1, 4), (3, 8), (8, 10)]);
    let restored = transport(&doc);
    for n in 1..24 {
        let h = DocumentHeatmap::build(&restored, ViewId::SOURCE, NonZeroU32::new(n)).unwrap();
        assert_eq!(h, DocumentHeatmap::build(&restored, ViewId::SOURCE, NonZeroU32::new(n)).unwrap());
        let lane = h.lane(HeatmapLaneKind::TokenDensity).unwrap();
        let tokens = restored.get::<Tokens>().unwrap();
        for s in &lane.segments {
            let covering: Vec<_> = tokens
                .spans
                .iter()
                .filter(|(_, span)| !s.span.is_empty() && span.start < s.span.end && s.span.start < span.end)
                .map(|(ix, _)| ix.get() as u32)
                .collect();
            assert_eq!(s.count as usize, covering.len(), "n={n}, {s:?}");
            assert_eq!(s.score, covering.len() as f32);
            assert_eq!(s.primary_item, covering.first().copied());
        }
        assert_eq!(lane.min_score, lane.segments.iter().map(|s| s.score).fold(f32::INFINITY, f32::min));
        assert_eq!(lane.max_score, lane.segments.iter().map(|s| s.score).fold(f32::NEG_INFINITY, f32::max));
    }
}

#[test]
fn filtered_scored_rows_keep_original_indices_and_ties() {
    let mut doc = Document::new("abcd");
    doc.add(
        &Key::<Sentences>::DEFAULT,
        Sentences { spans: spans(ViewId::SOURCE, &[(0, 4)]) },
        Provenance::by("test:s"),
    )
    .unwrap();
    doc.add(
        &Key::<Chunks>::DEFAULT,
        Chunks {
            spans: spans(ViewId::SOURCE, &[(0, 4), (0, 4), (0, 4)]),
            level: vec![0; 3],
            parent: vec![None; 3],
            sentences: vec![IxRange::new(0, 1); 3],
            cut: vec![Cut::Sentence; 3],
            cost: vec![0; 3],
            model_tokens: vec![1; 3],
            over_budget: vec![false; 3],
            prompt_overhead: 0,
        },
        Provenance::by("test:c"),
    )
    .unwrap();
    let mut scores = Categories::<Chunks>::with_capacity(TagSet::new("test:score", &["quality"]), 3);
    scores.push(Tag::NONE, &[0.9]);
    scores.push(Tag(0), &[0.5]);
    scores.push(Tag(0), &[0.5]);
    doc.add(&Key::<Categories<Chunks>>::DEFAULT, scores, Provenance::by("test:score")).unwrap();
    for buckets in [None, NonZeroU32::new(3)] {
        let h = DocumentHeatmap::build(&transport(&doc), ViewId::SOURCE, buckets).unwrap();
        assert!(
            h.lane(HeatmapLaneKind::ChunkScore)
                .unwrap()
                .segments
                .iter()
                .all(|s| s.count == 2 && s.primary_item == Some(1) && s.score == 0.5)
        );
    }
}

#[test]
fn empty_density_columns_still_validate_their_view() {
    let mut doc = Document::new("a");
    let source = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.keep(1);
    let (text, alignment) = b.finish();
    doc.add(&Key::<View>::named("other"), View::derived(&source, text, alignment), Provenance::by("test:view"))
        .unwrap();
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: Spans::new(ViewId(1)), sentence: None },
        Provenance::by("test:empty"),
    )
    .unwrap();
    for buckets in [None, NonZeroU32::new(3)] {
        assert!(matches!(DocumentHeatmap::build(&doc, ViewId::SOURCE, buckets), Err(Error::Bounds { .. })));
        assert!(DocumentHeatmap::build(&doc, ViewId(1), buckets).is_ok());
    }
}

#[test]
fn named_category_rows_need_not_match_default_anchor_count() {
    let mut doc = Document::new("abcd");
    doc.add(
        &Key::<Sentences>::DEFAULT,
        Sentences { spans: spans(ViewId::SOURCE, &[(0, 4)]) },
        Provenance::by("test:default"),
    )
    .unwrap();
    let gold = Key::<Sentences>::named("gold");
    doc.add(&gold, Sentences { spans: spans(ViewId::SOURCE, &[(0, 2), (2, 4)]) }, Provenance::by("test:gold")).unwrap();
    let mut scores = Categories::<Sentences>::with_capacity(TagSet::new("test:polarity", &["positive", "negative"]), 2);
    scores.push(Tag(0), &[0.9, 0.1]);
    scores.push(Tag(1), &[0.1, 0.9]);
    doc.add(
        &Key::<Categories<Sentences>>::DEFAULT,
        scores,
        Provenance { on: vec![gold.id()], ..Provenance::by("test:score") },
    )
    .unwrap();
    for buckets in [None, NonZeroU32::new(2)] {
        let h = DocumentHeatmap::build(&transport(&doc), ViewId::SOURCE, buckets).unwrap();
        let lane = h.lane(HeatmapLaneKind::SentimentPolarity).unwrap();
        assert_eq!(
            lane.segments.iter().map(|s| (s.span, s.primary_item, s.count)).collect::<Vec<_>>(),
            [(Span::new(0, 2), Some(0), 1), (Span::new(2, 4), Some(1), 1)]
        );
        assert!(lane.segments[0].score > 0.0 && lane.segments[1].score < 0.0);
    }
}
