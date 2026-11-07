use super::*;
use crate::View;
use crate::layer::{Key, Provenance};
use crate::text::ViewBuilder;

/// "ab cdef gh" with "cdef" replaced by "X" and " gh" deleted.
fn doc() -> (Document, ViewId) {
    let text = "ab cdef gh";
    let mut doc = Document::new(text);
    let src = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(text);
    b.keep(3);
    b.replace(7, "X");
    b.replace(10, "");
    let (t, a) = b.finish();
    assert_eq!(t, "ab X");
    doc.add(&Key::<View>::named("v"), View::derived(&src, t, a), Provenance::by("test")).unwrap();
    let id = doc.get_key(&Key::<View>::named("v")).unwrap().id();
    (doc, id)
}

#[test]
fn outcomes_are_exact_widened_or_deleted() {
    let (doc, v) = doc();
    let p = |s, e| doc.map_span_outcome(ViewId::SOURCE, v, Span::new(s, e)).unwrap();
    assert_eq!(p(0, 2), ProjectedSpan { span: Span::new(0, 2), outcome: SpanOutcome::Exact });
    assert_eq!(p(3, 7), ProjectedSpan { span: Span::new(3, 4), outcome: SpanOutcome::Exact });
    assert_eq!(p(4, 6), ProjectedSpan { span: Span::new(3, 4), outcome: SpanOutcome::Widened });
    assert_eq!(p(8, 10), ProjectedSpan { span: Span::new(4, 4), outcome: SpanOutcome::Deleted });
    assert_eq!(p(1, 1).outcome, SpanOutcome::Exact);
    // Back again, every outcome is containment.
    for (s, e) in [(0, 2), (3, 7), (4, 6), (8, 10)] {
        let back = doc.map_span(v, ViewId::SOURCE, p(s, e).span).unwrap();
        assert!(back.contains(Span::new(s, e)) || p(s, e).outcome == SpanOutcome::Deleted);
    }
}

#[test]
fn coalescing_is_its_own_fact_and_keeps_every_member() {
    let (doc, v) = doc();
    let frags = [Span::new(0, 1), Span::new(4, 5), Span::new(5, 6), Span::new(8, 10)];
    let out = doc.project_fragments(ViewId::SOURCE, v, frags).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(
        out[0],
        ProjectedFragment {
            span: Span::new(0, 1),
            outcome: SpanOutcome::Exact,
            members: vec![(0, SpanOutcome::Exact)]
        }
    );
    assert!(out[1].is_coalesced());
    assert_eq!(out[1].span, Span::new(3, 4));
    assert_eq!(out[1].outcome, SpanOutcome::Widened);
    assert_eq!(out[1].members, vec![(1, SpanOutcome::Widened), (2, SpanOutcome::Widened), (3, SpanOutcome::Deleted)]);
    // Only deleted members: a deleted fragment, not coalesced.
    let only = doc.project_fragments(ViewId::SOURCE, v, [Span::new(8, 9)]).unwrap();
    assert_eq!((only[0].outcome, only[0].is_coalesced()), (SpanOutcome::Deleted, false));
}
