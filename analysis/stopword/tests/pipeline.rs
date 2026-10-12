//! The step in a pipeline: it flags the tokens with the list of the given
//! language, or of the detected one, and declines a document it has no
//! list for.

use turbo_analysis::{Analyzer, Bound, Document, Key, Provenance, Span, Spans, Status, ViewId};
use turbo_analysis_layers::{Language, Strs, Tokens};
use turbo_stopword::{ListChoice, StopwordAnnotator, StopwordLanguage, Stopwords};

/// Tokens at the spaces of `text`.
fn tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for piece in text.split(' ') {
        spans.push(Span::new(start as u32, (start + piece.len()) as u32));
        start += piece.len() + 1;
    }
    Tokens { spans, sentence: None }
}

fn document(text: &str, language: Option<&str>) -> Document {
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, tokens(text), Provenance::by("test")).unwrap();
    if let Some(code) = language {
        let mut codes = Strs::default();
        codes.push(code);
        doc.add(&Key::DEFAULT, Language { code: codes, score: vec![0.9] }, Provenance::by("test")).unwrap();
    }
    doc
}

fn run(choice: ListChoice, docs: &mut [Document]) {
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Language>::DEFAULT)
        .step(Bound::new(StopwordAnnotator::new(choice)))
        .build()
        .unwrap();
    let results = analyzer.run(docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
}

fn flags(doc: &Document) -> (StopwordLanguage, Vec<bool>) {
    let s = doc.get::<Stopwords>().unwrap();
    (s.language, s.stop.clone())
}

#[test]
fn a_fixed_list_ignores_the_detected_language() {
    let mut docs = vec![document("The cat and THE dog", Some("deu"))];
    run(ListChoice::Fixed(StopwordLanguage::English), &mut docs);
    assert_eq!(flags(&docs[0]), (StopwordLanguage::English, vec![true, false, true, true, false]));
    let kept: Vec<usize> = docs[0].get::<Stopwords>().unwrap().kept().map(|i| i.get()).collect();
    assert_eq!(kept, [1, 4]);
}

#[test]
fn the_detected_language_picks_the_list() {
    let mut docs = vec![
        document("Der Hund und die Katze", Some("deu")),
        document("Der Hund und die Katze", Some("eng")),
        document("Der Hund", Some("swe")),
        document("the dog", None),
    ];
    run(ListChoice::Detected { fallback: None }, &mut docs);
    assert_eq!(flags(&docs[0]), (StopwordLanguage::German, vec![true, false, true, true, false]));
    assert_eq!(flags(&docs[1]), (StopwordLanguage::English, vec![false; 5]));
    let key = Key::<Stopwords>::DEFAULT.id();
    assert_eq!(docs[2].status(key), Some(&Status::Skipped("no stopword list for language \"swe\"".into())));
    assert_eq!(docs[3].status(key), Some(&Status::Skipped("no language to choose a stopword list by".into())));
}

#[test]
fn the_fallback_covers_missing_and_unsupported_languages() {
    let mut docs = vec![document("och the", Some("swe")), document("the och", None), document("de la", Some("fre"))];
    run(ListChoice::Detected { fallback: Some(StopwordLanguage::English) }, &mut docs);
    assert_eq!(flags(&docs[0]), (StopwordLanguage::English, vec![false, true]));
    assert_eq!(flags(&docs[1]), (StopwordLanguage::English, vec![true, false]));
    assert_eq!(flags(&docs[2]), (StopwordLanguage::French, vec![true, true]));
}

#[test]
fn configurations_hash_apart() {
    let hashes: Vec<u64> = [
        ListChoice::Fixed(StopwordLanguage::Bulgarian),
        ListChoice::Fixed(StopwordLanguage::English),
        ListChoice::Detected { fallback: None },
        ListChoice::Detected { fallback: Some(StopwordLanguage::Bulgarian) },
    ]
    .into_iter()
    .map(|c| {
        let annotator = StopwordAnnotator::new(c);
        assert_eq!(annotator.choice(), c);
        turbo_analysis::Annotator::config_hash(&annotator)
    })
    .collect();
    for (i, a) in hashes.iter().enumerate() {
        assert!(!hashes[i + 1..].contains(a));
    }
}
