use std::num::NonZeroU32;

use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{DocumentHeatmap, HeatmapLaneKind, Tokens};

fn density_doc(text: &str, populated: bool) -> Document {
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    if populated {
        spans.push(Span::new(0, text.len() as u32));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test:tokens")).unwrap();
    doc
}

fn ranges(doc: &Document, buckets: u32) -> Vec<(u32, u32)> {
    let heatmap = DocumentHeatmap::build(doc, ViewId::SOURCE, NonZeroU32::new(buckets)).unwrap();
    heatmap
        .lane(HeatmapLaneKind::TokenDensity)
        .unwrap()
        .segments
        .iter()
        .map(|segment| (segment.span.start, segment.span.end))
        .collect()
}

#[test]
fn bucket_ranges_are_raw_bytes_while_exact_ranges_are_text_slices() {
    for (text, n, expected) in [
        ("ééé", 4, vec![(0, 1), (1, 3), (3, 4), (4, 6)]),
        ("A😀éB", 3, vec![(0, 2), (2, 5), (5, 8)]),
        ("東京", 4, vec![(0, 1), (1, 3), (3, 4), (4, 6)]),
    ] {
        let doc = density_doc(text, true);
        assert_eq!(ranges(&doc, n), expected);
        let heatmap = DocumentHeatmap::build(&doc, ViewId::SOURCE, NonZeroU32::new(n)).unwrap();
        let lane = heatmap.lane(HeatmapLaneKind::TokenDensity).unwrap();
        assert_eq!(lane.segments.len(), n as usize);
        assert!(lane.segments.iter().all(|s| (s.score, s.count, s.primary_item) == (1.0, 1, Some(0))));
        assert!(lane.segments.iter().any(|s| text.get(s.span.start as usize..s.span.end as usize).is_none()));
        let exact = DocumentHeatmap::build(&doc, ViewId::SOURCE, None).unwrap();
        for segment in &exact.lane(HeatmapLaneKind::TokenDensity).unwrap().segments {
            assert!(text.get(segment.span.start as usize..segment.span.end as usize).is_some());
        }
    }
}

#[test]
fn empty_density_uses_the_same_raw_byte_bins() {
    let text = "ééé";
    let empty = density_doc(text, false);
    let full = density_doc(text, true);
    assert_eq!(ranges(&empty, 4), ranges(&full, 4));
    let heatmap = DocumentHeatmap::build(&empty, ViewId::SOURCE, NonZeroU32::new(4)).unwrap();
    let lane = heatmap.lane(HeatmapLaneKind::TokenDensity).unwrap();
    assert_eq!(lane.segments.len(), 4);
    assert!(lane.segments.iter().all(|s| (s.score, s.count, s.primary_item) == (0.0, 0, None)));
    assert_eq!((lane.min_score, lane.max_score), (0.0, 0.0));
    assert_eq!(text.get(0..1), None);
}

#[test]
fn an_empty_raw_bin_can_be_inside_a_scalar_and_stays_uncovered() {
    let text = "é";
    let doc = density_doc(text, true);
    assert_eq!(ranges(&doc, 4), [(0, 0), (0, 1), (1, 1), (1, 2)]);
    let heatmap = DocumentHeatmap::build(&doc, ViewId::SOURCE, NonZeroU32::new(4)).unwrap();
    let lane = heatmap.lane(HeatmapLaneKind::TokenDensity).unwrap();
    assert_eq!(lane.segments.iter().map(|s| s.count).collect::<Vec<_>>(), [0, 1, 0, 1]);
    assert_eq!(lane.segments[2].span, Span::new(1, 1));
    assert_eq!(text.get(1..1), None);
    for s in &lane.segments {
        assert_eq!(s.primary_item, (s.count > 0).then_some(0));
    }
}
