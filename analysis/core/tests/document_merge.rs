//! `Document::merge_with`: joining two independently built documents
//! of one text, duplicate policies, and atomic refusal.

use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_layers::{Entities, PosTags, Sentences, Tokens};

const TEXT: &str = "a b c d";

fn tokens_on(view: ViewId, n: u32) -> Tokens {
    let mut spans = Spans::new(view);
    for i in 0..n {
        spans.push(Span::new(i * 2, i * 2 + 1));
    }
    Tokens { spans, sentence: None }
}

fn sentences(n: u32) -> Sentences {
    let mut spans = Spans::new(ViewId::SOURCE);
    for i in 0..n {
        spans.push(Span::new(i * 4, i * 4 + 3));
    }
    Sentences { spans }
}

fn keys(doc: &Document) -> Vec<KeyId> {
    doc.layers().collect()
}

#[test]
fn disjoint_independent_documents_merge_under_refuse() {
    let mut a = Document::new(TEXT);
    a.add(&Key::DEFAULT, tokens_on(ViewId::SOURCE, 4), Provenance::by("t:tokens")).unwrap();
    let mut b = Document::new(TEXT);
    b.add(&Key::DEFAULT, sentences(2), Provenance::by("t:sentences")).unwrap();
    b.add(&Key::named("b"), tokens_on(ViewId::SOURCE, 2), Provenance::by("t:tokens-b")).unwrap();

    a.merge_with(&b, DuplicateLayerPolicy::Refuse).unwrap();
    assert_eq!(a.get::<Sentences>().unwrap().spans.len(), 2);
    assert_eq!(a.get_key(&Key::<Tokens>::named("b")).unwrap().spans.len(), 2);
    assert_eq!(a.provenance(Key::<Sentences>::DEFAULT.id()).unwrap().annotator, "t:sentences");
    // The other document is untouched.
    assert!(b.get::<Tokens>().is_none());
    // The branch-only merge still refuses independent documents.
    let mut c = Document::new(TEXT);
    let err = c.merge(&b).unwrap_err();
    assert!(err.to_string().contains("two copies of one document"), "{err}");
}

#[test]
fn different_texts_are_refused_and_change_nothing() {
    let mut a = Document::new(TEXT);
    a.add(&Key::DEFAULT, tokens_on(ViewId::SOURCE, 4), Provenance::by("t:tokens")).unwrap();
    let b = Document::new("a b c e");
    let before = keys(&a);
    let err = a.merge_with(&b, DuplicateLayerPolicy::KeepFirst).unwrap_err();
    assert!(err.to_string().contains("source texts differ"), "{err}");
    assert_eq!(keys(&a), before);
}

#[test]
fn duplicate_layers_follow_the_policy() {
    let build = |n: u32| {
        let mut d = Document::new(TEXT);
        d.add(&Key::DEFAULT, tokens_on(ViewId::SOURCE, n), Provenance::by("t:tokens")).unwrap();
        d
    };

    let mut a = build(2);
    let b = build(4);
    let before = keys(&a);
    let err = a.merge_with(&b, DuplicateLayerPolicy::Refuse).unwrap_err();
    assert!(matches!(err, Error::Duplicate(_)), "{err}");
    assert_eq!(keys(&a), before, "a refused merge is atomic");
    assert_eq!(a.get::<Tokens>().unwrap().spans.len(), 2);

    let err = a.merge_with(&b, DuplicateLayerPolicy::KeepBoth).unwrap_err();
    assert!(err.to_string().contains("keep-both cannot rename"), "{err}");
    assert_eq!(keys(&a), before);

    a.merge_with(&b, DuplicateLayerPolicy::KeepFirst).unwrap();
    assert_eq!(a.get::<Tokens>().unwrap().spans.len(), 2, "the first document's layer wins");
}

#[test]
fn keep_first_drops_layers_made_on_the_losing_layer() {
    let mut a = Document::new(TEXT);
    a.add(&Key::DEFAULT, tokens_on(ViewId::SOURCE, 2), Provenance::by("t:tokens")).unwrap();

    let mut b = Document::new(TEXT);
    b.add(&Key::DEFAULT, tokens_on(ViewId::SOURCE, 4), Provenance::by("t:tokens")).unwrap();
    let set = TagSet::new("pos", &["X"]);
    let tags = PosTags { tagset: set.clone(), tags: vec![set.tag("X").unwrap(); 4], prob: None };
    b.add(&Key::DEFAULT, tags, Provenance::by("t:pos")).unwrap();
    let kinds = TagSet::new("ner", &["PER"]);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 3));
    let entities = Entities {
        spans,
        tokens: vec![IxRange::new(0, 2)],
        kind: vec![kinds.tag("PER").unwrap()],
        kinds: kinds.clone(),
        score: vec![1.0],
    };
    b.add(&Key::DEFAULT, entities, Provenance::by("t:ner")).unwrap();
    b.add(&Key::DEFAULT, sentences(2), Provenance::by("t:sentences")).unwrap();

    a.merge_with(&b, DuplicateLayerPolicy::KeepFirst).unwrap();
    assert_eq!(a.get::<Tokens>().unwrap().spans.len(), 2);
    assert!(a.get::<PosTags>().is_none(), "tags aligned to the losing tokens are dropped");
    assert!(a.get::<Entities>().is_none(), "entities indexing the losing tokens are dropped");
    assert_eq!(a.get::<Sentences>().unwrap().spans.len(), 2, "independent layers still merge");
}

#[test]
fn branch_copies_merge_with_under_refuse() {
    let mut a = Document::new(TEXT);
    a.add(&Key::DEFAULT, tokens_on(ViewId::SOURCE, 4), Provenance::by("t:tokens")).unwrap();
    let mut branch = a.clone();
    branch.add(&Key::DEFAULT, sentences(2), Provenance::by("t:sentences")).unwrap();
    a.merge_with(&branch, DuplicateLayerPolicy::Refuse).unwrap();
    assert_eq!(a.get::<Sentences>().unwrap().spans.len(), 2);
    // And the original branch merge keeps its behavior.
    let mut c = Document::new(TEXT);
    c.add(&Key::DEFAULT, tokens_on(ViewId::SOURCE, 4), Provenance::by("t:tokens")).unwrap();
    let mut branch2 = c.clone();
    branch2.add(&Key::DEFAULT, sentences(1), Provenance::by("t:sentences")).unwrap();
    c.merge(&branch2).unwrap();
    assert_eq!(c.get::<Sentences>().unwrap().spans.len(), 1);
}

#[test]
fn views_are_adopted_and_stay_projectable() {
    let text = "Cats nap";
    let mut a = Document::new(text);
    let mut b = Document::new(text);
    let source = b.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(text);
    builder.replace(1, "c");
    builder.keep(text.len());
    let (folded, alignment) = builder.finish();
    assert_eq!(folded, "cats nap");
    b.add(&Key::named("folded"), View::derived(&source, folded, alignment), Provenance::by("t:fold")).unwrap();
    let folded_id = b.get_key(&Key::<View>::named("folded")).unwrap().id();
    b.add(&Key::named("folded"), tokens_on(folded_id, 2), Provenance::by("t:tokens")).unwrap();

    a.merge_with(&b, DuplicateLayerPolicy::Refuse).unwrap();
    let view = a.get_key(&Key::<View>::named("folded")).unwrap();
    assert_eq!(view.text(), "cats nap");
    assert_eq!(view.id(), folded_id);
    let adopted = a.get_key(&Key::<Tokens>::named("folded")).unwrap();
    let first = adopted.spans.get(Ix::new(0));
    assert_eq!(a.to_source(folded_id, first), Span::new(0, 1));
}

#[test]
fn a_different_view_at_the_same_position_is_refused() {
    let text = "Cats nap";
    let with_view = |replacement: &str, expected: &str| {
        let mut d = Document::new(text);
        let source = d.get::<View>().unwrap().clone();
        let mut builder = ViewBuilder::new(text);
        builder.replace(1, replacement);
        builder.keep(text.len());
        let (out, alignment) = builder.finish();
        assert_eq!(out, expected);
        d.add(&Key::named("v"), View::derived(&source, out, alignment), Provenance::by("t:v")).unwrap();
        d
    };
    let mut a = with_view("c", "cats nap");
    let b = with_view("C", "Cats nap");
    let before = keys(&a);
    let err = a.merge_with(&b, DuplicateLayerPolicy::KeepFirst).unwrap_err();
    assert!(err.to_string().contains("view 1 differs"), "{err}");
    assert_eq!(keys(&a), before);
}

#[test]
fn skipped_view_records_adopt_without_a_view() {
    let mut a = Document::new(TEXT);
    let mut b = Document::new(TEXT);
    b.skip(Key::<View>::named("folded").id(), "declined".into(), Provenance::by("t:fold")).unwrap();
    a.merge_with(&b, DuplicateLayerPolicy::Refuse).unwrap();
    assert!(a.has(Key::<View>::named("folded").id()));
    assert!(a.get_key(&Key::<View>::named("folded")).is_none(), "the record holds no view");
}

#[test]
fn one_view_key_naming_two_positions_is_refused() {
    let text = "Cats nap";
    let view_from = |doc: &Document, replacement: &str| {
        let source = doc.get::<View>().unwrap().clone();
        let mut b = ViewBuilder::new(text);
        b.replace(1, replacement);
        b.keep(text.len());
        let (out, alignment) = b.finish();
        View::derived(&source, out, alignment)
    };
    let mut a = Document::new(text);
    let v = view_from(&a, "c");
    a.add(&Key::named("v"), v, Provenance::by("t:v")).unwrap();

    let mut b = Document::new(text);
    let w = view_from(&b, "c");
    b.add(&Key::named("w"), w, Provenance::by("t:w")).unwrap();
    let parent = b.get_key(&Key::<View>::named("w")).unwrap().clone();
    let mut bb = ViewBuilder::new(parent.text());
    bb.keep(4);
    bb.replace(5, "N");
    bb.keep(parent.text().len());
    let (out, alignment) = bb.finish();
    b.add(&Key::named("v"), View::derived(&parent, out, alignment), Provenance::by("t:v")).unwrap();

    let before = a.layers().count();
    let err = a.merge_with(&b, DuplicateLayerPolicy::KeepFirst).unwrap_err();
    assert!(err.to_string().contains("names view 1 here and view 2"), "{err}");
    assert_eq!(a.layers().count(), before);
}

#[test]
fn decoded_and_text_documents_of_one_text_are_refused() {
    // Latin-1 bytes for the same decoded text: spans in file bytes
    // would mean different things in the two documents.
    let mut a = Document::new("café");
    let bytes: Arc<[u8]> = Arc::from([b'c', b'a', b'f', 0xE9].as_slice());
    let mut builder = ViewBuilder::new("caf\u{0}");
    builder.keep(3);
    builder.replace(4, "é");
    let (_, alignment) = builder.finish();
    let b = Document::decoded(bytes, "café", alignment).expect("the latin-1 fixture decodes");
    let err = a.merge_with(&b, DuplicateLayerPolicy::Refuse).unwrap_err();
    assert!(err.to_string().contains("decoded from bytes"), "{err}");
}
