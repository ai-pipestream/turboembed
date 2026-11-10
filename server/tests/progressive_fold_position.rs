//! Where a case fold sits among the normalizers does not change what the
//! readers of cased text find (#2141). Rule sentences and place names read
//! the chain with its folds left out, so a normalizer placed after a fold
//! reaches them as one placed before it does; tokens still read the folded
//! view, and every span is a span of the source.

use turbo_analysis::{Document, Key, View, ViewId};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_kserve::progressive::compile_analyzer;

/// `Zürich` with a zero-width space inside, `São Paulo` with a combining
/// tilde (not NFC), runs of spaces, and a sentence break after `Mr.` that
/// only cased text rules out.
const TEXT: &str = "Mr.  Zo\u{eb} flew from Sa\u{303}o Paulo to Z\u{fc}\u{200b}rich.   Then \u{130}stanbul and \
                    K\u{d6}LN. It rained in Gie\u{df}en.";

const LEXICAL: &[&str] = &["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo-geo:names"];
const FOLD: &str = "turbo:normalize:case-fold";

fn analyzed(norms: &[&str]) -> Document {
    let mut doc = Document::new(TEXT);
    let steps: Vec<&str> = norms.iter().chain(LEXICAL).copied().collect();
    let analyzer = compile_analyzer(&doc, &steps).unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().for_each(Result::unwrap);
    doc
}

fn slice(start: u32, end: u32) -> &'static str {
    TEXT.get(start as usize..end as usize).expect("a span on char boundaries")
}

fn places(doc: &Document) -> Vec<&'static str> {
    let entities = doc.get::<Entities>().expect("the place-name layer");
    assert_eq!(entities.spans.view(), ViewId::SOURCE);
    entities.spans.iter().map(|(_, s)| slice(s.start, s.end)).collect()
}

fn sentences(doc: &Document) -> Vec<&'static str> {
    let sentences = doc.get::<Sentences>().expect("the sentence layer");
    assert_eq!(sentences.spans.view(), ViewId::SOURCE);
    sentences.spans.iter().map(|(_, s)| slice(s.start, s.end)).collect()
}

/// `chain` with a fold put in at `at`.
fn with_fold(chain: &[&'static str], at: usize) -> Vec<&'static str> {
    let mut out = chain.to_vec();
    out.insert(at, FOLD);
    out
}

#[test]
fn a_fold_anywhere_in_a_chain_leaves_places_and_sentences_as_the_chain_without_it() {
    let chains: [&[&'static str]; 5] = [
        &["turbo:normalize:invisible"],
        &["turbo:normalize:nfc"],
        &["turbo:normalize"],
        &["turbo:normalize:invisible", "turbo:normalize:nfc"],
        &["turbo:normalize:invisible", "turbo:normalize:nfc", "turbo:normalize:whitespace"],
    ];
    for chain in chains {
        let expected = analyzed(chain);
        for at in 0..=chain.len() {
            let norms = with_fold(chain, at);
            let doc = analyzed(&norms);
            assert_eq!(places(&doc), places(&expected), "{norms:?}");
            assert_eq!(sentences(&doc), sentences(&expected), "{norms:?}");
            // The fold still reaches the last view and the tokens.
            let last = doc.require(&Key::<View>::named("normalized")).unwrap();
            assert!(last.text().contains("giessen") && !last.text().contains("Mr."), "{}", last.text());
            assert_eq!(doc.get::<Tokens>().unwrap().spans.len(), expected.get::<Tokens>().unwrap().spans.len());
        }
    }
}

#[test]
fn a_normalizer_after_the_fold_reaches_the_place_names() {
    // The zero-width space hides `Zürich` from the source; stripping it
    // finds the place, on either side of the fold.
    let hidden = |doc: &Document| places(doc).iter().any(|p| p.contains('\u{200b}'));
    assert!(!hidden(&analyzed(&[])) && !hidden(&analyzed(&[FOLD])));
    for norms in [
        &["turbo:normalize:invisible"][..],
        &["turbo:normalize:invisible", FOLD][..],
        &[FOLD, "turbo:normalize:invisible"][..],
        &[FOLD, "turbo:normalize:invisible", FOLD][..],
    ] {
        assert!(hidden(&analyzed(norms)), "{norms:?}: {:?}", places(&analyzed(norms)));
    }
}

#[test]
fn only_a_normalizer_after_a_fold_with_a_cased_reader_adds_a_view() {
    let cased = |norms: &[&str], lexical: &[&str]| {
        let mut doc = Document::new(TEXT);
        let steps: Vec<&str> = norms.iter().chain(lexical).copied().collect();
        let analyzer = compile_analyzer(&doc, &steps).unwrap();
        analyzer.run(std::slice::from_mut(&mut doc)).into_iter().for_each(Result::unwrap);
        doc.require(&Key::<View>::named("cased_1")).is_ok()
    };
    assert!(cased(&[FOLD, "turbo:normalize:invisible"], LEXICAL));
    // The fold last, no fold, or no reader of cased text: the chain as it was.
    assert!(!cased(&["turbo:normalize:invisible", FOLD], LEXICAL));
    assert!(!cased(&["turbo:normalize:invisible", "turbo:normalize:nfc"], LEXICAL));
    assert!(!cased(&[FOLD, "turbo:normalize:invisible"], &["turbo-tokenize:uax29-untyped"]));
}
