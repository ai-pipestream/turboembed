//! Case folding does not cost sentence boundaries (#2086). The rule
//! sentence detector decides on case, so in a chain that folds it reads
//! the chain with the folds left out; tokens and the other lexical stages
//! still read the folded view.

use turbo_analysis::{Document, Layer};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_kserve::progressive::compile_analyzer;

const TEXT: &str = "# Relatório\n\nZoë Müller moved from São Paulo to Paris on 3 March 2024. She paid €1,200 for \
                    the ﬁrst flat. Straße und İstanbul liegen fern.\n\nThe World Health Organization (WHO) met in \
                    Berlin. It met again in May.";

fn analyzed(steps: &[&str]) -> Document {
    let mut doc = Document::new(TEXT);
    let analyzer = compile_analyzer(&doc, steps).unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().for_each(Result::unwrap);
    doc
}

fn spans<L: Layer>(doc: &Document, of: impl Fn(&L) -> Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    of(doc.get::<L>().unwrap_or_else(|| panic!("{} is in the graph", L::NAME)))
}

fn sentences(doc: &Document) -> Vec<(u32, u32)> {
    spans::<Sentences>(doc, |l| l.spans.iter().map(|(_, s)| (s.start, s.end)).collect())
}

fn texts(doc: &Document) -> Vec<&str> {
    sentences(doc).into_iter().map(|(a, b)| &TEXT[a as usize..b as usize]).collect()
}

const LEXICAL: &[&str] = &["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped"];

fn chain(norms: &[&'static str]) -> Vec<&'static str> {
    norms.iter().chain(LEXICAL).copied().collect()
}

#[test]
fn a_fold_after_normalize_keeps_the_unfolded_sentences() {
    let plain = analyzed(&chain(&["turbo:normalize"]));
    let folded = analyzed(&chain(&["turbo:normalize", "turbo:normalize:case-fold"]));
    let found = texts(&plain);
    assert_eq!(found.len(), 5, "{found:?}");
    assert!(found[1].starts_with("She paid") && found[3].starts_with("The World"), "{found:?}");
    assert_eq!(sentences(&folded), sentences(&plain), "{:?}", texts(&folded));
}

#[test]
fn a_fold_first_reads_sentences_on_the_chain_without_it() {
    // A fold alone leaves the source; a normalizer after it still reaches
    // the sentences, as it does with no fold in the chain (#2141).
    for norms in [
        &["turbo:normalize:case-fold"][..],
        &["turbo:normalize:case-fold", "turbo:normalize:whitespace"][..],
        &["turbo:normalize:nfc", "turbo:normalize:case-fold", "turbo:normalize:invisible"][..],
    ] {
        let unfolded: Vec<&str> = norms.iter().copied().filter(|n| *n != "turbo:normalize:case-fold").collect();
        let (doc, expected) = (analyzed(&chain(norms)), analyzed(&chain(&unfolded)));
        assert_eq!(sentences(&doc), sentences(&expected), "{norms:?}: {:?}", texts(&doc));
    }
}

#[test]
fn the_fold_chain_still_tokenizes_the_source_words() {
    let doc = analyzed(&chain(&["turbo:normalize", "turbo:normalize:case-fold"]));
    let tokens = doc.get::<Tokens>().expect("tokens");
    // Tokens read the folded view; `ß` folds to `ss` and `İ` to `i̇`, and
    // each still projects back onto the whole source word.
    let words: Vec<&str> = tokens.spans.iter().map(|(_, s)| &TEXT[s.start as usize..s.end as usize]).collect();
    for w in ["Relatório", "Straße", "İstanbul", "WHO"] {
        assert!(words.contains(&w), "{w:?} in {words:?}");
    }
    assert_eq!(texts(&doc).len(), 5);
}
