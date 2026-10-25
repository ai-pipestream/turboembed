//! Independent original combined-segmentation oracle on small varied graphs.
use super::*;
use crate::Tokens;
use turbo_analysis::{Document, Key, TextSpans, ViewId};
#[test]
fn event_summaries_match_combined_segments_with_points_gaps_and_fragments() {
    let doc = Document::new("abcdefghijkl");
    let key = Key::<Tokens>::DEFAULT.id();
    let window = doc.window(ViewId::SOURCE, 0, 12).unwrap();
    for seed in 0..128u32 {
        let items: Vec<_> = (0..8u32)
            .map(|i| {
                let a = (seed + i * 3) % 4;
                let b = a + 1 + (seed + i) % 2;
                let c = 7 + (seed + i * 2) % 3;
                let d = c + 1 + (seed + i * 3) % 2;
                Item {
                    source: 15 - i,
                    score: if i % 2 == 0 { 0.5 } else { -0.5 },
                    view: ViewId::SOURCE,
                    frags: if (seed + i) % 3 == 0 {
                        vec![((seed + i) % 13, (seed + i) % 13)]
                    } else {
                        vec![(a, b), (c, d)]
                    },
                }
            })
            .collect();
        let (mut start, mut end, mut fs, mut fe, mut at) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for it in &items {
            start.push(it.frags[0].0);
            end.push(it.frags.last().unwrap().1);
            for &(s, e) in &it.frags {
                fs.push(s);
                fe.push(e);
            }
            at.push(fs.len() as u32);
        }
        let ts = TextSpans {
            view: ViewId::SOURCE,
            start: &start,
            end: &end,
            frag_start: &fs,
            frag_end: &fe,
            frag_item_end: &at,
        };
        let shape = doc.combined_segments(&window, &[(key, ts)]).unwrap();
        for kind in [HeatmapLaneKind::TokenDensity, HeatmapLaneKind::SentimentPolarity] {
            let actual = exact(12, key, kind, &items).unwrap();
            assert_eq!(actual.len(), shape.segments.len(), "seed={seed}");
            for (a, b) in actual.iter().zip(&shape.segments) {
                assert_eq!(a.span, b.span, "seed={seed}");
                let covered: Vec<_> = shape.members_of(b).iter().map(|m| &items[m.item as usize]).collect();
                assert_eq!(a.count as usize, covered.len());
                let primary = if kind.is_density() {
                    covered.iter().min_by_key(|it| it.source).copied()
                } else {
                    covered
                        .iter()
                        .copied()
                        .max_by(|a, b| a.score.abs().total_cmp(&b.score.abs()).then_with(|| b.source.cmp(&a.source)))
                };
                assert_eq!(a.primary_item, primary.map(|it| it.source));
                assert_eq!(
                    a.score,
                    if kind.is_density() { covered.len() as f32 } else { primary.map_or(0.0, |it| it.score) }
                );
            }
        }
    }
}
