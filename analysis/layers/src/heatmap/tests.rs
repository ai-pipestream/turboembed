//! Bucket mode lays offsets on the view as they are: one past the view's
//! end is refused, not clamped into the last bin.

use super::*;

fn token_items(view: ViewId, frags: &[(u32, u32)]) -> Vec<Item> {
    frags.iter().enumerate().map(|(i, &f)| Item { source: i as u32, score: 0.0, view, frags: vec![f] }).collect()
}

fn bucket(doc: &Document, items: &[Item]) -> Result<Vec<HeatmapSegment>, Error> {
    bucket_segments(doc, ViewId::SOURCE, Key::<Tokens>::DEFAULT.id(), 10, 4, HeatmapLaneKind::TokenDensity, items)
}

#[test]
fn bucket_mode_refuses_offsets_past_the_view() {
    let doc = Document::new("0123456789");
    let segments = bucket(&doc, &token_items(ViewId::SOURCE, &[(8, 10), (10, 10)])).unwrap();
    assert_eq!(segments.iter().map(|s| s.count).collect::<Vec<_>>(), [0, 0, 0, 2]);
    for frag in [(8, 12), (11, 11), (6, 5)] {
        match bucket(&doc, &token_items(ViewId::SOURCE, &[frag])) {
            Err(Error::Bounds { detail, .. }) => assert!(detail.contains("past the end"), "{frag:?}: {detail}"),
            other => panic!("{frag:?}: {other:?}"),
        }
    }
}

#[test]
fn both_modes_refuse_items_from_another_view() {
    let doc = Document::new("0123456789");
    let items = token_items(ViewId(1), &[(0, 2)]);
    let key = Key::<Tokens>::DEFAULT.id();
    let exact = exact_segments(&doc, ViewId::SOURCE, 10, HeatmapLaneKind::TokenDensity, key, &items);
    for refused in [exact, bucket(&doc, &items)] {
        match refused {
            Err(Error::Bounds { detail, .. }) => assert!(detail.contains("in view 1"), "{detail}"),
            other => panic!("{other:?}"),
        }
    }
}
