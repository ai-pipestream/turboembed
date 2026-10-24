#[path = "../../../core/tests/rerank/common.rs"]
mod common;
use std::path::Path;
use turbo::rerank::{RerankOptions, Reranker};
use turbo_analysis::{Annotator, Document, Ix, Key, Layer, Span, Spans, View, ViewId};
use turbo_analysis_embed::{RerankAnnotator, Shortlist};
use turbo_analysis_layers::{Levels, Unit};
use turbo_vector_index::Hit;

fn fixture() -> common::Fixture {
    common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap())
}
fn levels() -> Levels {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 13));
    spans.push(Span::new(0, 5));
    spans.push(Span::new(6, 13));
    Levels {
        spans,
        unit: vec![Unit::Document, Unit::Chunk(Ix::new(0)), Unit::Chunk(Ix::new(1))],
        parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
    }
}
#[test]
fn shortlist_uses_index_metadata_and_scores_only_selected_units() {
    let f = fixture();
    let d = Document::new("Paris Germany");
    let levels = levels();
    let shortlist = Shortlist::from_hits(
        "France".into(),
        Key::DEFAULT,
        &[Hit { row: 1, score: 0.9 }, Hit { row: 0, score: 0.8 }],
        2,
        &[Ix::new(1), Ix::new(2)],
    )
    .unwrap();
    let a = RerankAnnotator::open(&f.dir, 1, RerankOptions::default()).unwrap();
    let (ranked,) = a.annotate((d.get::<View>().unwrap(), &levels, &shortlist)).unwrap();
    ranked.check().unwrap();
    assert_eq!(ranked.items.len(), 2);
    let expected = Reranker::open(&f.dir, 1)
        .unwrap()
        .score(&[("France", "Germany"), ("France", "Paris")], RerankOptions::default())
        .unwrap();
    for (item, score) in ranked.items.iter().zip(&ranked.scores) {
        let i = shortlist.items.iter().position(|x| x == item).unwrap();
        assert_eq!(*score, expected.scores[i]);
    }
    assert_eq!(ranked.model.sha256, expected.model.manifest_sha256);
}
#[test]
fn malformed_shortlists_and_level_spans_fail_before_slicing() {
    let f = fixture();
    let a = RerankAnnotator::open(&f.dir, 1, RerankOptions::default()).unwrap();
    let d = Document::new("éaris Germany");
    let view = d.get::<View>().unwrap();
    let mut l = levels();
    let shortlist = Shortlist { query: "q".into(), source: Key::DEFAULT, items: vec![Ix::new(9)] };
    assert!(a.annotate((view, &l, &shortlist)).is_err());
    let mut shortlist = Shortlist { query: "q".into(), source: Key::DEFAULT, items: vec![Ix::new(1)] };
    l.spans = Spans::new(ViewId::SOURCE);
    l.spans.push(Span::new(0, 14));
    l.spans.push(Span::new(1, 6));
    l.spans.push(Span::new(7, 14));
    assert!(a.annotate((view, &l, &shortlist)).is_err());
    shortlist.source = Key::named("other");
    assert!(a.annotate((view, &levels(), &shortlist)).is_err());
    assert!(
        Shortlist::<Levels>::from_hits("q".into(), Key::DEFAULT, &[Hit { row: 0, score: f32::NAN }], 1, &[Ix::new(1)])
            .is_err()
    );
    assert!(
        Shortlist::<Levels>::from_hits("q".into(), Key::DEFAULT, &[Hit { row: 9, score: 1.0 }], 1, &[Ix::new(1)])
            .is_err()
    );
}

#[test]
fn reranker_batches_documents_and_isolates_invalid_shortlists() {
    let f = fixture();
    let a = RerankAnnotator::open(&f.dir, 1, RerankOptions::default()).unwrap();
    let d = Document::new("Paris Germany");
    let view = d.get::<View>().unwrap();
    let l = levels();
    let s = Shortlist { query: "France".into(), source: Key::DEFAULT, items: vec![Ix::new(1), Ix::new(2)] };
    let t = Shortlist { query: "Europe".into(), source: Key::DEFAULT, items: vec![Ix::new(2)] };
    let bad = Shortlist { query: "bad".into(), source: Key::DEFAULT, items: vec![Ix::new(9)] };
    let expected = [a.annotate((view, &l, &s)).unwrap().0, a.annotate((view, &l, &t)).unwrap().0];
    let mut actual = a.annotate_batch(vec![(view, &l, &s), (view, &l, &bad), (view, &l, &t)]).into_iter();
    for (i, expected) in expected.into_iter().enumerate() {
        if i == 1 {
            assert!(actual.next().unwrap().is_err());
        }
        let got = actual.next().unwrap().unwrap().0;
        assert_eq!(got.items, expected.items);
        assert_eq!(got.scores, expected.scores);
        assert_eq!(got.model, expected.model);
    }
    assert!(a.annotate_batch(Vec::new()).is_empty());
}
