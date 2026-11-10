//! Case folding does not change which place names are found (#2115). The
//! place-name step matches text as written, so in a chain that folds case
//! it reads the last view that keeps it, as the rule sentence step does;
//! tokens still read the folded view, and every place is a span of the
//! source.

use turbo_analysis::{Document, Key, View, ViewId};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_kserve::progressive::compile_analyzer;

/// Places with non-ASCII case pairs, and three spellings a full fold turns
/// into a place name the text does not write: the ligatures of `finland`
/// and `florida`, and the long s of `Nis`.
const TEXT: &str = "Zo\u{eb} flew from S\u{e3}o Paulo to \u{130}stanbul, then Z\u{fc}rich, \u{c5}rhus, \u{141}\u{f3}d\u{17a}, \
                    Gie\u{df}en and K\u{d6}LN. Not places: \u{fb01}nland, \u{fb02}orida, Ni\u{17f}.";

const LEXICAL: &[&str] = &["turbo-tokenize:uax29-untyped", "turbo-geo:names"];

fn analyzed(norms: &[&str]) -> Document {
    let mut doc = Document::new(TEXT);
    let steps: Vec<&str> = norms.iter().chain(LEXICAL).copied().collect();
    let analyzer = compile_analyzer(&doc, &steps).unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().for_each(Result::unwrap);
    doc
}

/// Each place as a slice of the source: `get` refuses a span that is not
/// on character boundaries.
fn places(doc: &Document) -> Vec<&'static str> {
    let entities = doc.get::<Entities>().expect("the place-name layer");
    assert_eq!(entities.spans.view(), ViewId::SOURCE);
    let slice = |start: u32, end: u32| TEXT.get(start as usize..end as usize).expect("a span on char boundaries");
    entities.spans.iter().map(|(_, s)| slice(s.start, s.end)).collect()
}

#[test]
fn a_fold_first_finds_the_places_of_the_unnormalized_request() {
    let unnormalized = analyzed(&[]);
    let expected = places(&unnormalized);
    for place in
        ["S\u{e3}o Paulo", "\u{130}stanbul", "Z\u{fc}rich", "\u{c5}rhus", "\u{141}\u{f3}d\u{17a}", "Gie\u{df}en"]
    {
        assert!(expected.contains(&place), "{place:?} in {expected:?}");
    }
    assert!(!expected.iter().any(|p| p.contains('\u{fb01}') || p.contains('\u{17f}')), "{expected:?}");
    for norms in [
        &["turbo:normalize:case-fold"][..],
        &["case_fold", "turbo:normalize:whitespace"][..],
        &["turbo:normalize:case-fold", "turbo:normalize:invisible"][..],
    ] {
        let doc = analyzed(norms);
        assert_eq!(places(&doc), expected, "{norms:?}");
        // The fold still reaches the steps that read the last view.
        let last = doc.require(&Key::<View>::named("normalized")).unwrap();
        assert!(last.text().contains("z\u{fc}rich") && last.text().contains("giessen"), "{}", last.text());
        assert_eq!(doc.get::<Tokens>().unwrap().spans.len(), unnormalized.get::<Tokens>().unwrap().spans.len());
    }
}

#[test]
fn a_fold_after_a_normalizer_finds_the_places_of_the_chain_without_it() {
    for before in [&["turbo:normalize"][..], &["turbo:normalize:nfc"][..]] {
        let expected = analyzed(before);
        let with_fold: Vec<&str> = before.iter().copied().chain(["turbo:normalize:case-fold"]).collect();
        assert_eq!(places(&analyzed(&with_fold)), places(&expected), "{with_fold:?}");
        let then_more: Vec<&str> = with_fold.iter().copied().chain(["turbo:normalize:whitespace"]).collect();
        assert_eq!(places(&analyzed(&then_more)), places(&expected), "{then_more:?}");
    }
}
