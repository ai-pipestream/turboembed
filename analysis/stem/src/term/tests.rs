//! Unit tests for `Dimension`, `Term`, `TermAnalyzer`, and `TermAnalyzerBuilder`.

use std::sync::{Arc, Barrier};

use turbo_analysis::Span;
use turbo_normalize::{CaseFold, ConfusableSkeleton, GermanUmlauts, Whitespace, WhitespaceMode, apply};
use turbo_unicode::script::Script;

use super::*;
use crate::porter::PorterStemmer;
use crate::snowball::{SnowballAlgorithm, SnowballStemmer};

#[test]
fn dimension_exposes_character_defaults_and_none_for_token_dimensions() {
    for dim in [
        Dimension::Nfc,
        Dimension::Nfkc,
        Dimension::Whitespace,
        Dimension::Dash,
        Dimension::CaseFold,
        Dimension::FullCaseFold,
        Dimension::AccentFold,
        Dimension::EmojiFold,
        Dimension::ConfusableFold,
    ] {
        assert!(dim.is_character_level(), "{dim}");
        assert!(dim.default_normalizer().is_some(), "{dim}");
    }
    for dim in [Dimension::Original, Dimension::Stem, Dimension::Lemma] {
        assert!(!dim.is_character_level(), "{dim}");
        assert!(dim.default_normalizer().is_none(), "{dim}");
    }
}

#[test]
fn no_dimensions_leaves_token_unchanged() {
    let analyzer = TermAnalyzer::builder().build();
    let terms = analyzer.analyze("Hello").unwrap();
    let term = &terms[0];
    assert_eq!(term.original(), "Hello");
    assert_eq!(term.normalized(), "Hello");
    assert_eq!(term.peel(), "Hello");
    assert!(analyzer.dimensions().is_empty());
}

#[test]
fn chain_applies_in_pipeline_order_regardless_of_builder_order() {
    let analyzer = TermAnalyzer::builder().accent_fold().case_fold().unwrap().build();
    assert_eq!(analyzer.dimensions(), &[Dimension::CaseFold, Dimension::AccentFold]);
    let terms = analyzer.analyze("CAF\u{00C9}").unwrap();
    let term = &terms[0];
    assert_eq!(term.original(), "CAF\u{00C9}");
    assert_eq!(term.normalized(), "cafe");
    assert_eq!(term.peel(), "caf\u{00E9}");
}

#[test]
fn full_case_fold_orders_before_accent_fold_regardless_of_builder_order() {
    let analyzer = TermAnalyzer::builder().accent_fold().full_case_fold().unwrap().build();
    assert_eq!(analyzer.dimensions(), &[Dimension::FullCaseFold, Dimension::AccentFold]);
    let terms = analyzer.analyze("CAF\u{00C9}").unwrap();
    let term = &terms[0];
    assert_eq!(term.normalized(), "cafe");
    assert_eq!(term.peel(), "caf\u{00E9}");
}

#[test]
fn stem_is_the_top_layer_and_peel_returns_layer_below() {
    let analyzer = TermAnalyzer::builder().case_fold().unwrap().stem(PorterStemmer).build();
    let terms = analyzer.analyze("Running").unwrap();
    let term = &terms[0];
    assert_eq!(term.peel(), "running");
    assert_eq!(term.normalized(), "run");
    assert_eq!(term.at(Dimension::Stem).unwrap(), "run");
}

#[test]
fn unconfigured_char_dimension_is_computed_lazily_and_memoized() {
    let analyzer = TermAnalyzer::builder().build();
    let terms = analyzer.analyze("HELLO").unwrap();
    let term = &terms[0];
    assert_eq!(term.normalized(), "HELLO");
    let first = term.at(Dimension::CaseFold).unwrap();
    let second = term.at(Dimension::CaseFold).unwrap();
    assert_eq!(first, "hello");
    assert!(std::ptr::eq(first, second));
}

#[test]
fn stem_and_lemma_without_engines_fail_loudly() {
    let analyzer = TermAnalyzer::builder().case_fold().unwrap().build();
    let terms = analyzer.analyze("running").unwrap();
    let term = &terms[0];
    assert_eq!(term.at(Dimension::Stem), Err(TermError::MissingStemmer));
    assert_eq!(term.at(Dimension::Lemma), Err(TermError::MissingLemmatizer));
}

#[test]
fn analyze_text_produces_spans_and_analyze_tokens_has_none() {
    let analyzer = TermAnalyzer::builder().case_fold().unwrap().build();
    let terms = analyzer.analyze("The Cats").unwrap();
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0].original(), "The");
    assert_eq!(terms[0].normalized(), "the");
    assert_eq!(terms[0].span(), Some(Span::new(0, 3)));
    assert_eq!(terms[1].original(), "Cats");
    assert_eq!(terms[1].span(), Some(Span::new(4, 8)));

    let pre = analyzer.analyze_tagged(&["Cats"], &["NNS"]).unwrap();
    assert_eq!(pre[0].span(), None);
    assert_eq!(pre[0].normalized(), "cats");
}

#[test]
fn lemma_dimension_with_lemmatizer_and_error_cases() {
    let analyzer = TermAnalyzer::builder()
        .case_fold()
        .unwrap()
        .lemmatize(|tok: &str, tag: &str| if tok == "was" && tag == "VBD" { Some("be".to_owned()) } else { None })
        .build();
    // Fails fast on raw text even if empty.
    assert_eq!(analyzer.analyze(""), Err(TermError::LemmaRequiresPosTagsInAnalyzeText));
    assert_eq!(analyzer.analyze("was"), Err(TermError::LemmaRequiresPosTagsInAnalyzeText));

    let terms = analyzer.analyze_tagged(&["was"], &["VBD"]).unwrap();
    assert_eq!(terms[0].normalized(), "be");

    // Missing POS tag fails loudly.
    assert_eq!(
        analyzer.analyze_tokens(&["was"], &[None]).unwrap_err(),
        TermError::MissingPosTag { token: "was".to_owned() }
    );
    // Lemmatizer returning None fails loudly instead of looping.
    assert_eq!(
        analyzer.analyze_tagged(&["unknown"], &["NN"]).unwrap_err(),
        TermError::EmptyLemma { token: "unknown".to_owned() }
    );
}

#[test]
fn builder_guards_and_options_match_opennlp() {
    assert_eq!(TermAnalyzer::builder().case_fold().unwrap().full_case_fold().err(), Some(TermError::BothCaseFolds));
    assert_eq!(TermAnalyzer::builder().full_case_fold().unwrap().case_fold().err(), Some(TermError::BothCaseFolds));
    assert_eq!(
        TermAnalyzer::builder().full_case_fold().unwrap().case_fold_locale("tr").err(),
        Some(TermError::BothCaseFolds)
    );
    assert_eq!(
        TermAnalyzer::builder().transform(Dimension::Stem, CaseFold::lowercase()).err(),
        Some(TermError::NonCharacterTransform(Dimension::Stem))
    );
    assert_eq!(TermAnalyzer::builder().max_token_length(0).err(), Some(TermError::InvalidMaxTokenLength(0)));
    assert_eq!(
        TermAnalyzer::builder().stem_cached_with_capacity(SnowballStemmer::new(SnowballAlgorithm::English), 0).err(),
        Some(TermError::InvalidCacheCapacity(0))
    );
}

#[test]
fn multilingual_and_character_dimension_behaviors_match_opennlp() {
    // German umlauts fold to digraphs.
    let german = TermAnalyzer::builder()
        .case_fold()
        .unwrap()
        .transform(Dimension::AccentFold, GermanUmlauts::new())
        .unwrap()
        .build();
    assert_eq!(german.analyze("GR\u{00DC}\u{00DF}E").unwrap()[0].normalized(), "gruesse");
    assert_eq!(german.analyze("\u{00DC}ber").unwrap()[0].normalized(), "ueber");

    // Generic accent fold strips Romance diacritics.
    let romance = TermAnalyzer::builder().case_fold().unwrap().accent_fold().build();
    assert_eq!(romance.analyze("\u{00C9}l\u{00E8}ve").unwrap()[0].normalized(), "eleve");
    assert_eq!(romance.analyze("Ni\u{00F1}o").unwrap()[0].normalized(), "nino");
    assert_eq!(romance.analyze("Canci\u{00F3}n").unwrap()[0].normalized(), "cancion");

    // Cyrillic case fold.
    let cyrillic = TermAnalyzer::builder().case_fold().unwrap().build();
    assert_eq!(
        cyrillic.analyze("\u{041C}\u{041E}\u{0421}\u{041A}\u{0412}\u{0410}").unwrap()[0].normalized(),
        "\u{043C}\u{043E}\u{0441}\u{043A}\u{0432}\u{0430}"
    );

    // Fullwidth A under NFKC + caseFold.
    let nfkc = TermAnalyzer::builder().nfkc().case_fold().unwrap().build();
    assert_eq!(nfkc.analyze("\u{FF21}").unwrap()[0].normalized(), "a");

    // Kanji unchanged under NFC.
    let nfc = TermAnalyzer::builder().nfc().build();
    assert_eq!(nfc.analyze("\u{6771}").unwrap()[0].normalized(), "\u{6771}");

    // Confusable fold composes with case fold.
    let confusable = TermAnalyzer::builder().case_fold().unwrap().confusable_fold().build();
    let spoof = "P\u{0430}yp\u{0430}l";
    assert_eq!(confusable.analyze(spoof).unwrap()[0].normalized(), apply(&ConfusableSkeleton, "paypal"));

    // Turkish locale case fold maps I to dotless i (U+0131).
    let tr = TermAnalyzer::builder().case_fold_locale("tr").unwrap().build();
    assert_eq!(tr.analyze("I").unwrap()[0].normalized(), "\u{0131}");

    // Accent fold restricted to Latin without stroke letters.
    let lat_only = TermAnalyzer::builder().accent_fold_scripts(&[Script::Latin], false).build();
    assert_eq!(lat_only.analyze("caf\u{00E9}").unwrap()[0].normalized(), "cafe");

    // Custom whitespace normalizer.
    let ws = TermAnalyzer::builder().whitespace_with(Whitespace::new(WhitespaceMode::KeepLineBreaks)).unwrap().build();
    assert_eq!(ws.analyze_tagged(&["a\n\n\tb"], &["X"]).unwrap()[0].normalized(), "a\nb");

    // Full case fold expands sharp s.
    let full = TermAnalyzer::builder().full_case_fold().unwrap().build();
    let term = &full.analyze("Ma\u{00DF}e").unwrap()[0];
    assert_eq!(term.normalized(), "masse");
    assert_eq!(term.at(Dimension::FullCaseFold).unwrap(), "masse");

    // Emoji fold layers emoticon on token.
    let emoji = TermAnalyzer::builder().emoji_fold().build();
    let grin = &emoji.analyze("\u{1F600}").unwrap()[0];
    assert_eq!(grin.at(Dimension::EmojiFold).unwrap(), ":D");
    assert_eq!(grin.at(Dimension::Original).unwrap(), "\u{1F600}");
    assert_eq!(emoji.analyze("hello").unwrap()[0].at(Dimension::EmojiFold).unwrap(), "hello");

    // Max token length chops tokens in UTF-16 units.
    let chopped = TermAnalyzer::builder().max_token_length(3).unwrap().build();
    let parts = chopped.analyze("abcdefg").unwrap();
    let originals: Vec<&str> = parts.iter().map(Term::original).collect();
    assert_eq!(originals, ["abc", "def", "g"]);
}

#[test]
fn concurrent_lazy_access_returns_same_cached_slice() {
    let analyzer = TermAnalyzer::builder().build();
    let term = Arc::new(analyzer.analyze("HELLO").unwrap().into_iter().next().unwrap());
    let threads = 8;
    let barrier = Arc::new(Barrier::new(threads));
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let t = Arc::clone(&term);
        let b = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            b.wait();
            let s = t.at(Dimension::CaseFold).unwrap();
            (s.to_owned(), s.as_ptr() as usize)
        }));
    }
    let first = handles.remove(0).join().unwrap();
    assert_eq!(first.0, "hello");
    for h in handles {
        let res = h.join().unwrap();
        assert_eq!(res, first);
    }
}
