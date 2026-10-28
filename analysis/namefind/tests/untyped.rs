//! Live rule APIs keep typed labels and reject invalid source spans.
mod native_support;

use turbo_analysis::{Annotator, Document, Span, Spans, View, ViewId};

#[test]
fn rule_kinds_are_explicit_and_shared_across_documents() {
    for kind in ["person", "default", "人物"] {
        let finder = native_support::finder(0, kind);
        let first = native_support::entities(&finder);
        assert!(first.kind.iter().all(|&tag| first.kinds.label(tag) == kind));
        let second = native_support::entities(&finder);
        assert!(std::sync::Arc::ptr_eq(&first.kinds, &second.kinds));
        assert_eq!(first.score, second.score);
    }
}

#[test]
fn rule_annotator_rejects_wrong_views_and_split_unicode_boundaries() {
    let finder = native_support::finder(0, "person");
    let (doc, mut tokens) = native_support::input();
    tokens.spans = Spans::new(ViewId(99));
    assert!(finder.annotate((doc.get::<View>().unwrap(), &tokens)).is_err());
    let doc = Document::new("é");
    tokens.spans = Spans::new(ViewId::SOURCE);
    tokens.spans.push(Span::new(0, 1));
    assert!(finder.annotate((doc.get::<View>().unwrap(), &tokens)).is_err());
}

#[test]
fn rule_hash_tracks_dictionary_and_kind_without_archive_bytes() {
    let original = native_support::finder(0, "person").config_hash();
    assert_eq!(original, native_support::finder(0, "person").config_hash());
    assert_ne!(original, native_support::finder(1, "person").config_hash());
    assert_ne!(original, native_support::finder(0, "place").config_hash());
}
