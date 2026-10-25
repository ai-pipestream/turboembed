use std::num::NonZeroU32;
use turbo_analysis::{Document, Key, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Categories, DocumentHeatmap, HeatmapLaneKind, HeatmapSegment, Sentences, Tokens};

fn assert_empty_bins(segments: &[HeatmapSegment], text_len: u32, n: u32) {
    assert_eq!(segments.len(), n as usize);
    for (i, segment) in segments.iter().enumerate() {
        assert_eq!(
            segment.span,
            Span::new(
                (i as u64 * u64::from(text_len) / u64::from(n)) as u32,
                ((i as u64 + 1) * u64::from(text_len) / u64::from(n)) as u32,
            )
        );
        assert_eq!((segment.score, segment.count, segment.primary_item), (0.0, 0, None));
    }
}

#[test]
fn absent_lanes_do_not_allocate_requested_bucket_storage() {
    // No lane exists, so this public call allocates no bucket arrays.
    for text in ["", "abc"] {
        let heatmap = DocumentHeatmap::build(&Document::new(text), ViewId::SOURCE, NonZeroU32::new(u32::MAX)).unwrap();
        assert!(heatmap.lanes.is_empty());
    }
}

#[test]
fn empty_density_preserves_every_requested_bin() {
    for text in ["", "abc"] {
        let mut doc = Document::new(text);
        doc.add(
            &Key::DEFAULT,
            Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None },
            Provenance::by("test:empty"),
        )
        .unwrap();
        for n in [1, 7, 257] {
            let heatmap = DocumentHeatmap::build(&doc, ViewId::SOURCE, NonZeroU32::new(n)).unwrap();
            let lane = heatmap.lane(HeatmapLaneKind::TokenDensity).unwrap();
            assert_empty_bins(&lane.segments, text.len() as u32, n);
            assert_eq!((lane.min_score, lane.max_score), (0.0, 0.0));
        }
    }
}

#[test]
fn populated_safe_bucket_counts_keep_exact_overlap_and_primary_indices() {
    let mut doc = Document::new("abc");
    let mut spans = Spans::new(ViewId::SOURCE);
    for i in 0..3 {
        spans.push(Span::new(i, i + 1));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test:tokens")).unwrap();
    for n in [1, 7, 257] {
        let heatmap = DocumentHeatmap::build(&doc, ViewId::SOURCE, NonZeroU32::new(n)).unwrap();
        let lane = heatmap.lane(HeatmapLaneKind::TokenDensity).unwrap();
        assert_eq!(lane.segments.len(), n as usize);
        for segment in &lane.segments {
            let covering: Vec<_> = (0..3)
                .filter(|&i| !segment.span.is_empty() && i < segment.span.end && segment.span.start < i + 1)
                .collect();
            assert_eq!(segment.count as usize, covering.len());
            assert_eq!(segment.score, covering.len() as f32);
            assert_eq!(segment.primary_item, covering.first().copied());
        }
    }
}

#[test]
fn abstention_only_scored_lane_preserves_empty_bins() {
    let mut doc = Document::new("abc");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 3));
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    let mut categories =
        Categories::<Sentences>::with_capacity(TagSet::new("test:sentiment", &["positive", "negative"]), 1);
    categories.push(Tag::NONE, &[0.9, 0.1]);
    doc.add(
        &Key::DEFAULT,
        categories,
        Provenance { on: vec![Key::<Sentences>::DEFAULT.id()], ..Provenance::by("test:categories") },
    )
    .unwrap();
    let heatmap = DocumentHeatmap::build(&doc, ViewId::SOURCE, NonZeroU32::new(257)).unwrap();
    let lane = heatmap.lane(HeatmapLaneKind::SentimentPolarity).unwrap();
    assert_empty_bins(&lane.segments, 3, 257);
    assert_eq!((lane.min_score, lane.max_score), (0.0, 0.0));
}

#[test]
fn zero_bucket_option_selects_exact_mode() {
    let mut doc = Document::new("abc");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 3));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test:tokens")).unwrap();
    let zero = NonZeroU32::new(0);
    assert_eq!(zero, None);
    let heatmap = DocumentHeatmap::build(&doc, ViewId::SOURCE, zero).unwrap();
    assert_eq!(heatmap, DocumentHeatmap::build(&doc, ViewId::SOURCE, None).unwrap());
    let lane = heatmap.lane(HeatmapLaneKind::TokenDensity).unwrap();
    assert_eq!(lane.segments, [HeatmapSegment { span: Span::new(0, 3), score: 1.0, count: 1, primary_item: Some(0) }]);
}
