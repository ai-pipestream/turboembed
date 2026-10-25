use super::*;

#[test]
fn ordered_windows_only_visit_matching_items_after_one_validation() {
    let doc = Document::new("x".repeat(10000));
    let mut spans = crate::Spans::<()>::new(ViewId::SOURCE);
    for i in 0..10000 {
        spans.push(Span::new(i, i + 1));
    }
    assert!(spans.is_ordered());
    let prepared = doc.prepare_segments(ViewId::SOURCE, &[(view_key(), spans.text_spans())]).unwrap();
    ITEM_VISITS.with(|c| c.set(0));
    VALIDATED_ENDPOINTS.with(|c| c.set(0));
    for from in [0, 1, 4999, 5000, 9999] {
        let window = doc.window(ViewId::SOURCE, from, 1).unwrap();
        let shape = prepared.combined_segments(&window).unwrap();
        assert_eq!(shape.segments.len(), 1);
        assert_eq!(shape.members[0].item, from);
    }
    let visits = ITEM_VISITS.with(|c| c.get());
    let validations = VALIDATED_ENDPOINTS.with(|c| c.get());
    eprintln!("five one-item windows: {visits} item visits, {validations} endpoint revalidations");
    assert_eq!(visits, 5, "ordered windows must not visit the other 9,999 items");
    assert_eq!(validations, 0, "validation belongs to preparation, not each window");
}

#[test]
fn raw_columns_retain_full_validation_and_the_arbitrary_scan() {
    let doc = Document::new("x".repeat(10000));
    let mut spans = crate::Spans::<()>::new(ViewId::SOURCE);
    for i in 0..10000 {
        spans.push(Span::new(i, i + 1));
    }
    let window = doc.window(ViewId::SOURCE, 5000, 1).unwrap();
    ITEM_VISITS.with(|c| c.set(0));
    VALIDATED_ENDPOINTS.with(|c| c.set(0));
    let shape = doc.combined_segments(&window, &[(view_key(), spans.text_spans())]).unwrap();
    assert_eq!(shape.members[0].item, 5000);
    assert_eq!(ITEM_VISITS.with(|c| c.get()), 10000);
    assert_eq!(VALIDATED_ENDPOINTS.with(|c| c.get()), 20000);
}

#[test]
fn a_discontinuous_candidate_only_visits_fragments_meeting_the_window() {
    let doc = Document::new("x".repeat(20000));
    let fragments: Vec<_> = (0..10000).map(|i| Span::new(2 * i, 2 * i + 1)).collect();
    let mut spans = crate::Spans::<()>::new(ViewId::SOURCE);
    spans.try_push_fragments(&fragments).unwrap();
    let prepared = doc.prepare_segments(ViewId::SOURCE, &[(view_key(), spans.text_spans())]).unwrap();
    let window = doc.window(ViewId::SOURCE, 10000, 1).unwrap();
    FRAGMENT_VISITS.with(|c| c.set(0));
    let shape = prepared.combined_segments(&window).unwrap();
    assert_eq!(shape.members[0].item, 0);
    let visits = FRAGMENT_VISITS.with(|c| c.get());
    eprintln!("one matching fragment out of 10000: {visits} fragment visits");
    assert_eq!(visits, 1, "fragments outside the window need no sweep events");
}
