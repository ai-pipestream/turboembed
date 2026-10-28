//! Retokenizing term vectors: terms built on a dehyphenated view,
//! stored on source tokens.

use turbo_analysis::*;
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_chunker::terms::{RetokenizingTermIndexer, TermCounter, TermStats, Vocabulary};
use turbo_normalize::{CaseFold, Dehyphenate, NormalizeStep};
use turbo_tokenize::UntypedWordTokenizer;

const TEXT: &str = "The litiga-\ntion ends.";

fn dehyphenated_pipeline() -> Analyzer {
    Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .step(Bound::new(NormalizeStep::new(Dehyphenate::new())).write(0, "dehyphenated"))
        .step(Bound::new(UntypedWordTokenizer::default()).read(0, "dehyphenated").write(0, "norm"))
        .step(Bound::new(RetokenizingTermIndexer).read(0, "dehyphenated").read(1, "norm"))
        .add(TermCounter::<View>::new())
        .build()
        .unwrap()
}

#[test]
fn dehyphenated_terms_are_stored_on_source_tokens() {
    let mut docs = vec![Document::new(TEXT)];
    assert_eq!(dehyphenated_pipeline().run(&mut docs)[0], Ok(()));
    let doc = &docs[0];

    let vocabulary = doc.get::<Vocabulary>().unwrap();
    let litigation = vocabulary.lookup("litigation").expect("the joined word is one term");
    assert_eq!(vocabulary.lookup("litiga-"), None);
    assert_eq!(vocabulary.lookup("tion"), None);

    let source_tokens = doc.get::<Tokens>().unwrap();
    assert_eq!(source_tokens.spans.view(), ViewId::SOURCE);
    let norm_tokens = doc.get_key(&Key::<Tokens>::named("norm")).unwrap();
    let dehyphenated = doc.get_key(&Key::<View>::named("dehyphenated")).unwrap();
    assert_eq!(dehyphenated.text(), "The litigation ends.");

    let terms = doc.get::<turbo_chunker::terms::Terms>().unwrap();
    assert_eq!(terms.term.len(), source_tokens.spans.len());
    for (i, source_span) in source_tokens.spans.iter() {
        let back = doc.from_source(dehyphenated.id(), source_span);
        let norm_span = norm_tokens.spans.get(i);
        assert_eq!(back, norm_span, "token {} does not project back exactly", i.get());
    }
    let at = terms.term.iter().position(|t| *t == Some(litigation)).expect("a token carries the term");
    let span = source_tokens.spans.get(Ix::new(at as u32));
    assert_eq!(&TEXT[span.range()], "litiga-\ntion");

    let whole = doc.get::<TermStats<View>>().unwrap();
    assert_eq!(whole.tf(Ix::new(0), litigation), 1);
}

#[test]
fn tokens_splitting_one_expansion_edit_are_refused() {
    let text = "dont stop";
    let mut doc = Document::new(text);
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(text);
    builder.replace(4, "do not");
    builder.keep(text.len());
    let (expanded_text, alignment) = builder.finish();
    assert_eq!(expanded_text, "do not stop");
    doc.add(&Key::<View>::named("expanded"), View::derived(&source, expanded_text, alignment), Provenance::by("t"))
        .unwrap();
    let expanded_id = doc.get_key(&Key::<View>::named("expanded")).unwrap().id();
    let mut spans = Spans::new(expanded_id);
    for (a, b) in [(0, 2), (3, 6), (7, 11)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::named("norm"), Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();

    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<View>::named("expanded"))
        .given(Key::<Tokens>::named("norm"))
        .step(Bound::new(RetokenizingTermIndexer).read(0, "expanded").read(1, "norm"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let err = pipeline.run(&mut docs)[0].clone().unwrap_err();
    assert!(err.to_string().contains("does not project exactly"), "{err}");
    assert!(docs[0].get::<Vocabulary>().is_none(), "a refused run writes no vocabulary");
}

#[test]
fn a_view_two_hops_from_the_source_is_refused() {
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .step(Bound::new(NormalizeStep::new(CaseFold::full())).write(0, "folded"))
        .step(Bound::new(NormalizeStep::new(CaseFold::lowercase())).read(0, "folded").write(0, "twice"))
        .step(Bound::new(UntypedWordTokenizer::default()).read(0, "twice").write(0, "norm"))
        .step(Bound::new(RetokenizingTermIndexer).read(0, "twice").read(1, "norm"))
        .build()
        .unwrap();
    let mut docs = vec![Document::new("Cats nap.")];
    let err = pipeline.run(&mut docs)[0].clone().unwrap_err();
    assert!(err.to_string().contains("not made directly from the source"), "{err}");
}

#[test]
fn sentence_indexed_normalized_tokens_are_refused() {
    let mut doc = Document::new("Cats nap.");
    let source = doc.get::<View>().unwrap().clone();
    let (text, alignment) = turbo_normalize::normalize(&CaseFold::full(), "Cats nap.");
    doc.add(&Key::named("folded"), View::derived(&source, text, alignment), Provenance::by("t")).unwrap();
    let folded_id = doc.get_key(&Key::<View>::named("folded")).unwrap().id();
    let mut sentence_spans = Spans::new(folded_id);
    sentence_spans.push(Span::new(0, 9));
    doc.add(&Key::DEFAULT, Sentences { spans: sentence_spans }, Provenance::by("t")).unwrap();
    let mut spans = Spans::new(folded_id);
    spans.push(Span::new(0, 4));
    doc.add(&Key::named("norm"), Tokens { spans, sentence: Some(vec![Ix::new(0)]) }, Provenance::by("t")).unwrap();

    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<View>::named("folded"))
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::named("norm"))
        .step(Bound::new(RetokenizingTermIndexer).read(0, "folded").read(1, "norm"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let err = pipeline.run(&mut docs)[0].clone().unwrap_err();
    assert!(err.to_string().contains("sentence indexes"), "{err}");
}
