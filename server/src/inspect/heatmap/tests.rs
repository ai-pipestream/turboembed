use super::*;
use crate::offsets::OffsetMode;
use turbo_analysis::{Key, Provenance, Spans};

#[test]
fn nested_spans_query_the_aggregate_once_per_output_interval() {
    for n in [64u32, 128] {
        let mut doc = Document::new("x".repeat((2 * n) as usize));
        let mut spans = Spans::new(ViewId::SOURCE);
        for i in 0..n {
            spans.push(Span::new(i, 2 * n - i));
        }
        doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
        let bytes =
            turbo_analysis::export::export(&doc, crate::results::document::codecs()).unwrap().to_bytes().unwrap();
        let graph = turbo_analysis::export::DocumentGraph::from_bytes(&bytes).unwrap();
        let doc = turbo_analysis::export::import(&graph, crate::results::document::codecs()).unwrap();
        let text = doc.source();
        AGGREGATE_QUERIES.with(|count| count.set(0));
        let lanes =
            build_heatmap_lanes(&doc, ViewId::SOURCE, text, &Offsetter::new(text, OffsetMode::Utf8), 0).unwrap();
        let lane = &lanes[0];
        assert_eq!(lane.buckets.len(), (2 * n - 1) as usize);
        assert_eq!(lane.buckets.iter().map(|b| u64::from(b.count)).sum::<u64>(), u64::from(n) * u64::from(n));
        let visits = AGGREGATE_QUERIES.with(|count| count.get());
        eprintln!("{n} nested spans: {visits} aggregate queries for {} output intervals", lane.buckets.len());
        assert_eq!(visits, lane.buckets.len(), "exact heatmaps must not rescan all active items");
    }
}

#[test]
fn exact_scores_and_ties_match_a_direct_span_oracle() {
    let text = "abcdef";
    let offsetter = Offsetter::new(text, OffsetMode::Utf8);
    let items = [
        ScoredSpan { span: Span::new(0, 6), score: 0.2, item: 5 },
        ScoredSpan { span: Span::new(1, 5), score: -0.9, item: 3 },
        ScoredSpan { span: Span::new(2, 4), score: 0.9, item: 1 },
        ScoredSpan { span: Span::new(2, 4), score: 0.9, item: 7 },
    ];
    for (density, buckets) in [false, true].into_iter().flat_map(|density| [0, 1, 3, 16].map(|b| (density, b))) {
        let actual = if buckets == 0 {
            exact_segments(text, &offsetter, &items, density).unwrap()
        } else {
            binned_buckets(text, &offsetter, buckets, &items, density)
        };
        for bucket in actual {
            let active: Vec<_> = items
                .iter()
                .enumerate()
                .filter(|(_, it)| it.span.start < bucket.end && it.span.end > bucket.start)
                .collect();
            assert_eq!(bucket.count as usize, active.len());
            let (score, primary) = if density {
                (active.iter().map(|(_, it)| it.score).sum::<f32>(), active.first().map(|(_, it)| it.item))
            } else if active.is_empty() {
                (0.0, None)
            } else {
                let (_, strongest) = active
                    .iter()
                    .copied()
                    .max_by(|(i, a), (j, b)| a.score.abs().total_cmp(&b.score.abs()).then_with(|| j.cmp(i)))
                    .unwrap();
                (strongest.score, Some(strongest.item))
            };
            assert!((bucket.score - score).abs() < 1e-6);
            assert_eq!(bucket.primary_item, primary);
        }
    }
}

#[test]
fn bucketed_heatmaps_query_one_aggregate_per_bucket() {
    let text = "x".repeat(256);
    let offsetter = Offsetter::new(&text, OffsetMode::Utf8);
    for n in [64u32, 128] {
        let items: Vec<_> = (0..n).map(|item| ScoredSpan { span: Span::new(0, 256), score: 1.0, item }).collect();
        AGGREGATE_QUERIES.with(|count| count.set(0));
        let actual = binned_buckets(&text, &offsetter, 128, &items, true);
        let queries = AGGREGATE_QUERIES.with(|count| count.get());
        assert_eq!(queries, 128);
        assert!(actual.iter().all(|bucket| bucket.count == n && bucket.score == n as f32));
        eprintln!("{n} whole-view spans: {queries} aggregate queries for 128 buckets");
    }
}

#[test]
fn a_discontinuous_item_counts_once_per_bucket() {
    // Entities, Pii and Chunks push one ScoredSpan per fragment, all with the
    // same item; a bucket holding two fragments of one entity holds one item.
    let text = "abcdef";
    let offsetter = Offsetter::new(text, OffsetMode::Utf8);
    let items = [
        ScoredSpan { span: Span::new(0, 2), score: 0.5, item: 0 },
        ScoredSpan { span: Span::new(4, 6), score: 0.5, item: 0 },
    ];
    for density in [false, true] {
        let buckets = binned_buckets(text, &offsetter, 1, &items, density);
        assert_eq!(buckets[0].count, 1, "density={density}");
        if density {
            assert_eq!(buckets[0].score, 0.5);
        }
    }
}
