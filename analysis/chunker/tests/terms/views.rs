//! Terms read through a view: case folding, the source view, views the
//! tokens are not of, unit ends, documents with no terms, and edits at
//! token edges.

use super::*;

#[test]
fn full_case_folding_maps_through_the_view() {
    // "Straße" folds to "strasse", "ẞ" (U+1E9E) to "ss" as well; the term
    // view's bytes differ from the source's, and each token's span is
    // mapped to it.
    let text = "Straße STRASSE STRAẞE.";
    let mut doc = Document::new(text);
    let (tokens, sentences) = tokens_and_sentences(text);
    doc.add(&Key::DEFAULT, sentences, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, tokens, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(NormalizeStep::new(CaseFold::full())).write(0, "folded"))
        .step(Bound::new(TermIndexer).read(0, "folded"))
        .add(TermCounter::<View>::new())
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let v = docs[0].get::<Vocabulary>().unwrap();
    assert_eq!(v.text.iter().collect::<Vec<_>>(), ["strasse"]);
    let whole = docs[0].get::<TermStats<View>>().unwrap();
    assert_eq!((whole.length[0], whole.tf(Ix::new(0), Ix::new(0))), (3, 3));
}

#[test]
fn terms_of_the_source_view_keep_their_case() {
    let mut doc = document();
    let pipeline =
        Analyzer::builder().given(Key::<View>::DEFAULT).given(Key::<Tokens>::DEFAULT).add(TermIndexer).build().unwrap();
    let mut docs = vec![doc.clone()];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let v = docs[0].get::<Vocabulary>().unwrap();
    assert!(v.lookup("RUST").is_some() && v.lookup("Rust").is_some() && v.lookup("rust").is_none());
    // Units in a view other than the tokens' are refused.
    let (text, alignment) = turbo_normalize::normalize(&CaseFold::full(), TEXT);
    let folded = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("folded"), folded, Provenance::by("t")).unwrap();
    let folded_id = doc.get_key(&Key::<View>::named("folded")).unwrap().id();
    let mut other =
        Structure { spans: Spans::new(folded_id), kind: vec![Block::Section], parent: vec![None], depth: vec![0] };
    other.spans.push(Span::new(0, 4));
    doc.add(&Key::named("other"), other, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::named("other"))
        .add(TermIndexer)
        .step(Bound::new(TermCounter::<Structure>::new()).read(3, "other").write(0, "other"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let err = pipeline.run(&mut docs).pop().unwrap().unwrap_err();
    assert!(err.to_string().contains("the units are in view 1, the tokens in view 0"), "{err}");
}

#[test]
fn tokens_of_an_unrelated_view_are_refused() {
    // Terms read from a view two steps from the tokens' view: the step has
    // no document to follow the chain through, so it refuses.
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(NormalizeStep::new(CaseFold::full())).write(0, "folded"))
        .step(Bound::new(NormalizeStep::new(CaseFold::lowercase())).read(0, "folded").write(0, "twice"))
        .step(Bound::new(TermIndexer).read(0, "twice"))
        .build()
        .unwrap();
    let mut docs = vec![document()];
    let err = pipeline.run(&mut docs)[0].clone().unwrap_err();
    assert!(err.to_string().contains("neither the terms' view 2 nor its parent"), "{err}");
    // And the document says the terms failed rather than were empty.
    assert!(matches!(docs[0].status(Key::<Vocabulary>::DEFAULT.id()), Some(Status::Failed(_))));
    assert!(docs[0].get::<Vocabulary>().is_none());
}

#[test]
fn a_token_a_unit_end_cuts_counts_once_in_the_unit_it_starts_in() {
    // Units "ab c" and "d ef" over tokens "ab", "cd", "ef": "cd" starts in
    // the first.
    let text = "ab cd ef";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 2), (3, 5), (6, 8)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
    let mut st = Structure {
        spans: Spans::new(ViewId::SOURCE),
        kind: vec![Block::Section; 2],
        parent: vec![None; 2],
        depth: vec![0; 2],
    };
    st.spans.push(Span::new(0, 4));
    st.spans.push(Span::new(4, 8));
    doc.add(&Key::DEFAULT, st, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(TermIndexer)
        .add(TermCounter::<Structure>::new())
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let blocks = docs[0].get::<TermStats<Structure>>().unwrap();
    assert_eq!(blocks.length, [2, 1]);
}

#[test]
fn a_document_with_no_terms_has_empty_rows_and_scores_nothing() {
    let text = ". , ;";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 1), (2, 3), (4, 5)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(TermIndexer)
        .add(TermCounter::<View>::new())
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let v = docs[0].get::<Vocabulary>().unwrap();
    let whole = docs[0].get::<TermStats<View>>().unwrap();
    assert!(v.is_empty());
    assert_eq!((whole.length.as_slice(), whole.row.as_slice()), (&[0][..], &[0, 0][..]));
    let local = LocalCollection::new(whole, v);
    assert_eq!((local.units(), local.mean_length()), (0, 0.0));
    let q = Query::new(Bm25::LUCENE, &local, v, &["anything"]);
    assert!(q.terms.is_empty());
    assert_eq!(q.score(whole, Ix::new(0)), 0.0);
}

#[test]
fn a_token_edge_inside_an_edit_of_the_terms_view_is_refused() {
    // One edit turns "don't" into "do not" over the tokens "do" and "n't":
    // mapped, each would read the whole "do not".
    let text = "don't stop";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 2), (2, 5), (6, 10)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
    let mut out = ViewBuilder::new(text);
    out.replace(5, "do not");
    let (expanded, alignment) = out.finish();
    let view = View::derived(doc.get::<View>().unwrap(), expanded, alignment);
    doc.add(&Key::named("expanded"), view, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::named("expanded"))
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(TermIndexer).read(0, "expanded"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let err = pipeline.run(&mut docs)[0].clone().unwrap_err();
    assert!(err.to_string().contains("token 0 at 0..2 has an edge inside the edit of 0..5"), "{err}");
    assert!(err.to_string().contains("must not cross a token edge"), "{err}");
    assert!(docs[0].get::<Vocabulary>().is_none());
}

#[test]
fn an_edit_that_is_a_whole_token_or_an_insertion_is_fine() {
    // "can't" is one token and one edit; an insertion before "go" lies at
    // a token edge, not inside a token.
    let text = "can't go";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 5), (6, 8)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
    let mut out = ViewBuilder::new(text);
    out.replace(5, "cannot");
    out.keep(6);
    out.replace(6, "now ");
    let (changed, alignment) = out.finish();
    assert_eq!(changed, "cannot now go");
    let view = View::derived(doc.get::<View>().unwrap(), changed, alignment);
    doc.add(&Key::named("changed"), view, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::named("changed"))
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(TermIndexer).read(0, "changed"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let v = docs[0].get::<Vocabulary>().unwrap();
    assert_eq!(v.text.iter().collect::<Vec<_>>(), ["cannot", "go"]);
}
