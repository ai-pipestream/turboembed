use turbo_analysis::{Document, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Entities, Tokens};

fn spans<L>(view: ViewId, ranges: &[(u32, u32)]) -> Spans<L> {
    let mut spans = Spans::new(view);
    for &(start, end) in ranges {
        spans.push(Span::new(start, end));
    }
    spans
}

#[test]
fn entities_must_not_keep_indexes_into_an_unrelated_target_token_layer() {
    let mut doc = Document::new("aa bb cc");
    let source = doc.get::<View>().unwrap().clone();
    let (text, alignment) = ViewBuilder::new(source.text()).finish();
    doc.add(&Key::<View>::named("child"), View::derived(&source, text, alignment), Provenance::by("test")).unwrap();
    let child = doc.get_key(&Key::<View>::named("child")).unwrap().id();
    doc.add(&Key::DEFAULT, Tokens { spans: spans(ViewId::SOURCE, &[(0, 8)]), sentence: None }, Provenance::by("test"))
        .unwrap();
    let entities = Entities {
        spans: spans(child, &[(6, 8)]),
        tokens: vec![IxRange::new(2, 3)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("test", &["x"]),
        score: vec![1.0],
    };
    assert!(doc.project_layer(&entities, ViewId::SOURCE).is_err());
}
