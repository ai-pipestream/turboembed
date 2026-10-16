//! The Hunspell stemmer as an analysis step: it writes a stem per token,
//! aligned to the tokens, the token itself when the dictionary does not
//! read it, and refuses tokens of another view.

use std::sync::Arc;

use turbo_analysis::{Alignment, Analyzer, Bound, Document, Key, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_hunspell::{Dictionary, HunspellAnnotator, LoadMode, Stems};

fn dictionary(affix: &str, words: &str) -> Arc<Dictionary> {
    Arc::new(Dictionary::from_bytes(affix.as_bytes(), words.as_bytes(), LoadMode::Strict).unwrap())
}

fn english() -> Arc<Dictionary> {
    dictionary(
        "SET UTF-8\nSFX S Y 2\nSFX S y ies [^aeiou]y\nSFX S 0 s [^y]\nSFX D Y 1\nSFX D 0 ed .\n",
        "3\nwalk/SD\npony/S\ncafé/S\n",
    )
}

/// Tokens at the spaces of `text`.
fn tokens(text: &str, view: ViewId) -> Tokens {
    let mut spans = Spans::new(view);
    let mut start = 0;
    for piece in text.split(' ') {
        let end = start + piece.len();
        spans.push(Span::new(start as u32, end as u32));
        start = end + 1;
    }
    Tokens { spans, sentence: None }
}

fn stems(dictionary: Arc<Dictionary>, text: &str) -> Vec<String> {
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(HunspellAnnotator::new(dictionary)))
        .build()
        .unwrap();
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, tokens(text, ViewId::SOURCE), Provenance::by("test")).unwrap();
    let mut docs = vec![doc];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let stems = docs[0].get::<Stems>().unwrap();
    (0..stems.stem.len()).map(|i| stems.stem.get(i).to_string()).collect()
}

#[test]
fn a_stem_per_token() {
    assert_eq!(
        stems(english(), "the ponies walked Walks cafés ponies"),
        ["the", "pony", "walk", "walk", "café", "pony"]
    );
}

#[test]
fn an_unknown_or_empty_token_is_its_own_stem() {
    assert_eq!(stems(english(), "xyz  walks"), ["xyz", "", "walk"]);
    assert_eq!(stems(english(), ""), [""]);
}

#[test]
fn tokens_after_multibyte_text_keep_their_offsets() {
    assert_eq!(stems(english(), "cafés café ponies"), ["café", "café", "pony"]);
}

#[test]
fn refuses_tokens_of_another_view() {
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<View>::named("lower"))
        .step(Bound::new(HunspellAnnotator::new(english())))
        .build()
        .unwrap();
    let mut doc = Document::new("Walks");
    let lower = View::derived(doc.get::<View>().unwrap(), "walks".into(), Alignment::identity());
    doc.add(&Key::named("lower"), lower, Provenance::by("test")).unwrap();
    let lower_id = doc.get_key(&Key::<View>::named("lower")).unwrap().id();
    doc.add(&Key::DEFAULT, tokens("walks", lower_id), Provenance::by("test")).unwrap();
    let mut docs = vec![doc];
    let error = format!("{:?}", analyzer.run(&mut docs));
    assert!(error.contains("the tokens are in another view"), "{error}");
}

#[test]
fn config_hashes_follow_the_dictionary() {
    use turbo_analysis::Annotator;
    let a = HunspellAnnotator::new(english());
    let b = HunspellAnnotator::new(english());
    let c = HunspellAnnotator::new(dictionary("SFX S Y 1\nSFX S 0 s .\n", "1\nwalk/S\n"));
    assert_eq!(a.config_hash(), b.config_hash());
    assert_ne!(a.config_hash(), c.config_hash());
}
