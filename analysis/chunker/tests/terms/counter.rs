//! `TermCounter` over other layers, normalized terms, and selected columns.

use super::*;

/// Pages with no spans: a layer `TermCounter` cannot place tokens in when
/// it has more than one item.
struct Pages(usize);

impl Layer for Pages {
    const ID: &'static str = "test:pages";
    fn anchor() -> Anchor {
        Anchor::Document
    }
    fn len(&self) -> usize {
        self.0
    }
}

#[test]
fn the_counter_refuses_tokens_out_of_order_and_many_units_with_no_spans() {
    let text = "ab cd";
    let counter = |tokens: &[(u32, u32)], pages: usize| {
        let mut doc = Document::new(text);
        let mut spans = Spans::new(ViewId::SOURCE);
        for &(a, b) in tokens {
            spans.push(Span::new(a, b));
        }
        doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("t")).unwrap();
        doc.add(&Key::DEFAULT, Pages(pages), Provenance::by("t")).unwrap();
        let pipeline = Analyzer::builder()
            .given(Key::<View>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Pages>::DEFAULT)
            .add(TermIndexer)
            .add(TermCounter::<Pages>::new())
            .build()
            .unwrap();
        let mut docs = vec![doc];
        let status = pipeline.run(&mut docs).pop().unwrap();
        (status, docs.pop().unwrap())
    };
    let (status, doc) = counter(&[(0, 2), (3, 5)], 1);
    assert_eq!(status, Ok(()));
    assert_eq!(doc.get::<TermStats<Pages>>().unwrap().length, [2]);
    let err = counter(&[(3, 5), (0, 2)], 1).0.unwrap_err();
    assert!(err.to_string().contains("the tokens are not in text order"), "{err}");
    let err = counter(&[(0, 2), (3, 5)], 2).0.unwrap_err();
    assert!(err.to_string().contains("2 units with no spans; only one can hold every token"), "{err}");
}

#[test]
fn normalized_terms_feed_counts_and_bm25() {
    use turbo_chunker::terms::{IndexedTerms, NormalizedTermIndexer, TermForm};
    let pipeline = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .add(turbo_stem::TermAnnotator::for_language("en").unwrap())
        .add(NormalizedTermIndexer { form: TermForm::Folded })
        .add(TermCounter::<Chunks>::new())
        .build()
        .unwrap();
    let mut docs = [document()];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let doc = &docs[0];
    let forms = doc.get::<turbo_analysis_layers::Terms>().unwrap();
    assert_eq!(forms.surface.get(0), "Cats");
    assert_eq!(forms.folded.get(0), "cats");
    let vocab = doc.get::<Vocabulary>().unwrap();
    let ids = doc.get::<IndexedTerms>().unwrap();
    assert_eq!(ids.term[0].map(|t| vocab.get(t)), Some("cats"));
    assert_eq!(ids.term[2], None);
    let stats = doc.get::<TermStats<Chunks>>().unwrap();
    assert_eq!(stats.tf(Ix::new(1), vocab.lookup("cats").unwrap()), 2);
    let local = LocalCollection::only(stats, vocab, |i| [1, 2, 4].contains(&i.get()));
    let ranked = Query::new(Bm25::LUCENE, &local, vocab, &["cats"]).rank(stats, [Ix::new(1), Ix::new(2), Ix::new(4)]);
    assert_eq!(ranked[0].0, Ix::new(1));
    assert!(ranked[0].1 > 0.0);
    let provenance = doc.provenance(Key::<IndexedTerms>::DEFAULT.id()).unwrap();
    assert!(provenance.reads.contains(&Key::<turbo_analysis_layers::Terms>::DEFAULT.id()));
}

#[test]
fn selected_columns_are_exact_and_missing_columns_fail() {
    use turbo_analysis_layers::{Strs, Terms as Forms};
    use turbo_chunker::terms::{NormalizedTermIndexer, TermForm};
    let strs = |words: &[&str]| {
        let mut s = Strs::default();
        for w in words {
            s.push(w);
        }
        s
    };
    let (tokens, _) = tokens_and_sentences("Cats RUNNING.");
    let forms = Forms {
        surface: strs(&["Cats", ".", "RUNNING"]),
        folded: strs(&["cats", ".", "running"]),
        stem: Some(strs(&["cat", "", "run"])),
        lemma: Some(strs(&["cat", ".", "run"])),
    };
    for (form, expected) in [
        (TermForm::Surface, vec!["Cats", "RUNNING"]),
        (TermForm::Folded, vec!["cats", "running"]),
        (TermForm::Stem, vec!["cat", "run"]),
        (TermForm::Lemma, vec!["cat", "run"]),
    ] {
        let (vocab, ids) = NormalizedTermIndexer { form }.annotate((&forms, &tokens)).unwrap();
        assert_eq!(vocab.text.iter().collect::<Vec<_>>(), expected);
        assert_eq!(ids.term[1], None);
    }
    let missing = Forms { stem: None, lemma: None, ..forms.clone() };
    for form in [TermForm::Stem, TermForm::Lemma] {
        let err = NormalizedTermIndexer { form }.annotate((&missing, &tokens)).err().unwrap();
        assert!(err.to_string().contains("column is missing"));
    }
    let short_stem = Forms { stem: Some(strs(&["cat", "run"])), ..forms };
    let err = NormalizedTermIndexer { form: TermForm::Stem }.annotate((&short_stem, &tokens)).err().unwrap();
    assert!(err.to_string().contains("not aligned to tokens"));
    let hashes: std::collections::HashSet<_> = [TermForm::Surface, TermForm::Folded, TermForm::Stem, TermForm::Lemma]
        .map(|form| NormalizedTermIndexer { form }.config_hash())
        .into_iter()
        .collect();
    assert_eq!(hashes.len(), 4);
}
