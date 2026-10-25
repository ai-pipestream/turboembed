//! Retained overlap/operation regressions at the shared implementation.
use super::*;
use crate::{DocumentHeatmap, Tokens};
use turbo_analysis::{Document, Key, Provenance, Spans, ViewId};
fn item(source: u32, score: f32, frags: &[(u32, u32)]) -> Item {
    Item { source, score, view: ViewId::SOURCE, frags: frags.to_vec() }
}
#[test]
fn nested_spans_query_the_aggregate_once_per_output_interval() {
    for n in [64u32, 128] {
        let mut doc = Document::new("x".repeat((2 * n) as usize));
        let mut spans = Spans::new(ViewId::SOURCE);
        for i in 0..n {
            spans.push(Span::new(i, 2 * n - i));
        }
        doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
        QUERIES.with(|q| q.set(0));
        let h = DocumentHeatmap::build(&doc, ViewId::SOURCE, None).unwrap();
        let lane = &h.lanes[0];
        assert_eq!(lane.segments.len(), (2 * n - 1) as usize);
        assert_eq!(lane.segments.iter().map(|s| u64::from(s.count)).sum::<u64>(), u64::from(n) * u64::from(n));
        assert_eq!(QUERIES.with(|q| q.get()), lane.segments.len());
    }
}
fn run(items: &[Item], n: u32, density: bool) -> Vec<HeatmapSegment> {
    let kind = if density { HeatmapLaneKind::TokenDensity } else { HeatmapLaneKind::SentimentPolarity };
    let key = Key::<Tokens>::DEFAULT.id();
    if n == 0 {
        exact(12, key, kind, items).unwrap()
    } else {
        super::super::buckets::bucket_segments(&Document::new("abcdefghijkl"), ViewId::SOURCE, key, 12, n, kind, items)
            .unwrap()
    }
}
#[test]
fn exact_scores_and_ties_match_a_direct_span_oracle() {
    let items =
        [item(5, 0.2, &[(0, 12)]), item(3, -0.9, &[(1, 11)]), item(1, 0.9, &[(2, 10)]), item(7, 0.9, &[(2, 10)])];
    oracle(&items);
}
fn oracle(items: &[Item]) {
    for n in [0, 1, 2, 3, 4, 12, 25] {
        for density in [false, true] {
            for s in run(items, n, density) {
                let active: Vec<_> = items
                    .iter()
                    .filter(|it| {
                        it.frags.iter().any(|&(a, b)| a < s.span.end && b > s.span.start && !s.span.is_empty())
                    })
                    .collect();
                assert_eq!(s.count as usize, active.len(), "n={n}, {s:?}");
                let primary = if density {
                    active.iter().min_by_key(|it| it.source).copied()
                } else {
                    active
                        .iter()
                        .copied()
                        .max_by(|a, b| a.score.abs().total_cmp(&b.score.abs()).then_with(|| b.source.cmp(&a.source)))
                };
                assert_eq!(s.primary_item, primary.map(|it| it.source));
                let score = if density { active.len() as f32 } else { primary.map_or(0.0, |it| it.score) };
                assert_eq!(s.score, score, "n={n}, {s:?}");
            }
        }
    }
}
#[test]
fn bucketed_heatmaps_query_one_aggregate_per_bucket() {
    for n in [64u32, 128] {
        let items: Vec<_> = (0..n).map(|i| item(i, 1.0, &[(0, 12)])).collect();
        QUERIES.with(|q| q.set(0));
        let actual = run(&items, 128, true);
        assert_eq!(QUERIES.with(|q| q.get()), 128);
        assert!(actual.iter().filter(|s| !s.span.is_empty()).all(|s| s.count == n && s.score == n as f32));
        assert!(actual.iter().filter(|s| s.span.is_empty()).all(|s| s.count == 0 && s.score == 0.0));
    }
}
#[test]
fn a_discontinuous_item_counts_once_per_bucket() {
    let items = [item(0, 0.5, &[(0, 2), (10, 12)])];
    for density in [false, true] {
        let bins = run(&items, 1, density);
        assert_eq!(bins[0].count, 1);
        assert_eq!(bins[0].score, if density { 1.0 } else { 0.5 });
    }
}
#[test]
fn discontinuous_items_enter_and_leave_buckets_without_duplicate_scores() {
    oracle(&[item(0, -0.75, &[(0, 2), (3, 5), (9, 11)]), item(1, 0.5, &[(1, 7)])]);
}
#[test]
fn interior_and_eof_points_preserve_exact_segmentation() {
    let items = [item(0, 0.5, &[(0, 4)]), item(1, -0.75, &[(2, 2)]), item(2, 0.25, &[(12, 12)])];
    let spans: Vec<_> = run(&items, 0, false).into_iter().map(|s| (s.span, s.count, s.primary_item)).collect();
    assert_eq!(
        spans,
        [
            (Span::new(0, 2), 1, Some(0)),
            (Span::new(2, 2), 1, Some(1)),
            (Span::new(2, 4), 1, Some(0)),
            (Span::new(4, 12), 0, None),
            (Span::new(12, 12), 1, Some(2))
        ]
    );
}
