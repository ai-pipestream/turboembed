//! The stemmer as an analysis step: it writes a stem per token, aligned to
//! the tokens, and refuses tokens of another view.

use turbo_analysis::{Alignment, Analyzer, Bound, Document, Key, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_stem::{Algorithm, LightStemmer, SnowballAlgorithm, SnowballStemmer, StemAnnotator, Stems};

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

fn stems(algorithm: Algorithm, text: &str) -> Vec<String> {
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(StemAnnotator::new(algorithm)))
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
    let english = Algorithm::Snowball(SnowballStemmer::new(SnowballAlgorithm::English));
    assert_eq!(stems(english, "the cats were running and running"), ["the", "cat", "were", "run", "and", "run"]);
    assert_eq!(stems(Algorithm::Porter, "generalizations ponies"), ["gener", "poni"]);
    let minimal = Algorithm::Light(LightStemmer::EnglishMinimal);
    assert_eq!(stems(minimal, "ponies cats"), ["pony", "cat"]);
}

#[test]
fn an_empty_token_has_an_empty_stem() {
    assert_eq!(stems(Algorithm::Porter, ""), [""]);
    assert_eq!(stems(Algorithm::Porter, "cats  cats"), ["cat", "", "cat"]);
}

#[test]
fn refuses_tokens_of_another_view() {
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<View>::named("lower"))
        .step(Bound::new(StemAnnotator::new(Algorithm::Porter)))
        .build()
        .unwrap();
    let mut doc = Document::new("Cats");
    let lower = View::derived(doc.get::<View>().unwrap(), "cats".into(), Alignment::identity());
    doc.add(&Key::named("lower"), lower, Provenance::by("test")).unwrap();
    let lower_id = doc.get_key(&Key::<View>::named("lower")).unwrap().id();
    doc.add(&Key::DEFAULT, tokens("cats", lower_id), Provenance::by("test")).unwrap();
    let mut docs = vec![doc];
    let results = analyzer.run(&mut docs);
    let error = format!("{results:?}");
    assert!(error.contains("the tokens are in another view"), "{error}");
}

#[test]
fn config_hashes_differ() {
    use turbo_stem::NorwegianStandard as N;
    let all = [
        Algorithm::Porter,
        Algorithm::Snowball(SnowballStemmer::new(SnowballAlgorithm::Porter)),
        Algorithm::Snowball(SnowballStemmer::with_repeat(SnowballAlgorithm::Porter, 2.try_into().unwrap())),
        Algorithm::Light(LightStemmer::NorwegianLight(N::Both)),
        Algorithm::Light(LightStemmer::NorwegianMinimal(N::Both)),
        Algorithm::Light(LightStemmer::NorwegianMinimal(N::Bokmaal)),
    ];
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert_ne!(a.config_hash(), b.config_hash(), "{a:?} {b:?}");
        }
    }
}
