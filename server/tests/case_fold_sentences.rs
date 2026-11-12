//! Case folding does not cost sentence boundaries (#2086, #2141). The rule
//! sentence detector decides on case, so in a chain that folds it reads
//! the cased view: the normalizer chain with the fold steps left out.
//! Tokens and the other lexical stages still read the folded view.

use turbo_analysis::{Document, Key, Layer, View};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_kserve::progressive::compile_analyzer;

const TEXT: &str = "# Relatório\n\nZoë Müller moved from São Paulo to Paris on 3 March 2024. She paid €1,200 for \
                    the ﬁrst flat. Straße und İstanbul liegen fern.\n\nThe World Health Organization (WHO) met in \
                    Berlin. It met again in May.";

/// Text where invisible characters (`J\u{200b}.`), non-NFC initials
/// (`E\u{301}.`), and paragraph whitespace (`# Relatório\n\n`) each change
/// sentence boundaries when their normalizer reaches the detector.
const DIRTY: &str = "# Relat\u{f3}rio\n\nJ\u{200b}. M\u{fc}ller and E\u{301}. Zola met in S\u{61}\u{303}o   Paulo.   \
                     She paid \u{20ac}1,200 for the \u{fb01}rst flat.\n\n\nThe World Health Organization (WHO) met in \
                     Berlin. It met again in May.";

fn analyzed_text(text: &str, steps: &[&str]) -> Document {
    let mut doc = Document::new(text);
    let analyzer = compile_analyzer(&doc, steps).unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().for_each(Result::unwrap);
    doc
}

fn analyzed(steps: &[&str]) -> Document {
    analyzed_text(TEXT, steps)
}

fn spans<L: Layer>(doc: &Document, of: impl Fn(&L) -> Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    of(doc.get::<L>().unwrap_or_else(|| panic!("{} is in the graph", L::NAME)))
}

fn sentences(doc: &Document) -> Vec<(u32, u32)> {
    spans::<Sentences>(doc, |l| l.spans.iter().map(|(_, s)| (s.start, s.end)).collect())
}

fn texts<'a>(text: &'a str, doc: &Document) -> Vec<&'a str> {
    sentences(doc).into_iter().map(|(a, b)| &text[a as usize..b as usize]).collect()
}

const LEXICAL: &[&str] = &["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped"];

fn chain(norms: &[&'static str]) -> Vec<&'static str> {
    norms.iter().chain(LEXICAL).copied().collect()
}

fn without_fold(norms: &[&'static str]) -> Vec<&'static str> {
    norms.iter().copied().filter(|&s| !matches!(s, "turbo:normalize:case-fold" | "case_fold")).collect()
}

#[test]
fn a_fold_after_normalize_keeps_the_unfolded_sentences() {
    let plain = analyzed(&chain(&["turbo:normalize"]));
    let folded = analyzed(&chain(&["turbo:normalize", "turbo:normalize:case-fold"]));
    let found = texts(TEXT, &plain);
    assert_eq!(found.len(), 5, "{found:?}");
    assert!(found[1].starts_with("She paid") && found[3].starts_with("The World"), "{found:?}");
    assert_eq!(sentences(&folded), sentences(&plain), "{:?}", texts(TEXT, &folded));
    assert!(folded.get_key(&Key::<View>::named("cased")).is_none());
}

#[test]
fn a_fold_anywhere_in_the_chain_matches_the_chain_without_the_fold() {
    let raw = sentences(&analyzed_text(DIRTY, LEXICAL));
    // Each non-fold normalizer changes the sentence boundaries on DIRTY.
    for single in
        [&["turbo:normalize:invisible"][..], &["turbo:normalize:nfc"][..], &["turbo:normalize:whitespace"][..]]
    {
        assert_ne!(sentences(&analyzed_text(DIRTY, &chain(single))), raw, "{single:?} should change boundaries");
    }

    for norms in [
        // Fold first.
        &["turbo:normalize:case-fold"][..],
        &["turbo:normalize:case-fold", "turbo:normalize:invisible"][..],
        &["turbo:normalize:case-fold", "turbo:normalize:nfc"][..],
        &["turbo:normalize:case-fold", "turbo:normalize:whitespace"][..],
        &["turbo:normalize:case-fold", "turbo:normalize"][..],
        &["case_fold", "turbo:normalize:invisible", "turbo:normalize:nfc", "turbo:normalize:whitespace"][..],
        // Fold in the middle.
        &["turbo:normalize:nfc", "turbo:normalize:case-fold", "turbo:normalize:invisible"][..],
        &[
            "turbo:normalize:invisible",
            "turbo:normalize:case-fold",
            "turbo:normalize:nfc",
            "turbo:normalize:whitespace",
        ][..],
        &[
            "turbo:normalize:invisible",
            "turbo:normalize:nfc",
            "turbo:normalize:case-fold",
            "turbo:normalize:whitespace",
        ][..],
        // Fold last.
        &["turbo:normalize:invisible", "turbo:normalize:case-fold"][..],
        &["turbo:normalize:nfc", "turbo:normalize:case-fold"][..],
        &["turbo:normalize:whitespace", "turbo:normalize:case-fold"][..],
        &[
            "turbo:normalize:invisible",
            "turbo:normalize:nfc",
            "turbo:normalize:whitespace",
            "turbo:normalize:case-fold",
        ][..],
    ] {
        let unfolded = without_fold(norms);
        let expected = analyzed_text(DIRTY, &chain(&unfolded));
        let got = analyzed_text(DIRTY, &chain(norms));
        assert_eq!(sentences(&got), sentences(&expected), "{norms:?}: {:?}", texts(DIRTY, &got));
        assert!(expected.get_key(&Key::<View>::named("cased")).is_none(), "{unfolded:?}");
        if matches!(norms.last().copied(), Some("turbo:normalize:case-fold" | "case_fold")) {
            assert!(got.get_key(&Key::<View>::named("cased")).is_none(), "{norms:?}");
            assert!(got.get_key(&Key::<View>::named("cased_1")).is_none(), "{norms:?}");
        }
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
    assert_eq!(texts(TEXT, &doc).len(), 5);
}
