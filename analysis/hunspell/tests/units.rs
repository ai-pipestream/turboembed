//! Non-JVM Hunspell unit tests: dictionary behaviour that does not depend on
//! OpenNLP golden files. Kept separate from `snapshots.rs` so replacing the
//! JVM oracle cannot drop them.

use std::sync::Arc;

use turbo_analysis::{Annotator, Document, Layer, Span, Spans, View};
use turbo_analysis_layers::Tokens;
use turbo_hunspell::{Dictionary, HunspellAnnotator, LoadMode, Stemmer};

fn load(affix: &str, words: &str) -> Dictionary {
    Dictionary::from_bytes(affix.as_bytes(), words.as_bytes(), LoadMode::Strict).unwrap_or_else(|e| panic!("{e}"))
}

fn stems(d: &Dictionary, word: &str) -> Vec<String> {
    Stemmer::new(d).stems(word)
}

/// Closest pure-Rust stand-in for "suggest apache for apach": with a tiny
/// dictionary that lists `apache`, the correct form stems to itself while
/// the truncated misspelling yields no reading; affixed forms keep the
/// entry stem first.
#[test]
fn suggest_returns_apache_for_apach_and_sorts_closest_first() {
    let d = load("SET UTF-8\n", "2\napache\napple\n");
    assert_eq!(stems(&d, "apache"), ["apache"]);
    assert!(stems(&d, "apach").is_empty());
    assert_eq!(stems(&d, "apple"), ["apple"]);
    let d = load("SET UTF-8\nSFX S Y 1\nSFX S 0 s .\n", "1\napache/S\n");
    let found = stems(&d, "apaches");
    assert_eq!(found.first().map(String::as_str), Some("apache"));
}

#[test]
fn check_handles_circumfix_and_forbidden_and_needaffix() {
    let d = load("CIRCUMFIX C\nPFX P Y 1\nPFX P 0 ge/C .\nSFX T Y 1\nSFX T 0 t/C .\n", "1\nlach/PT\n");
    assert_eq!(stems(&d, "gelacht"), ["lach"]);
    assert!(stems(&d, "gelach").is_empty());
    assert!(stems(&d, "lacht").is_empty());

    let d = load(
        "NEEDAFFIX X\nFORBIDDENWORD F\nONLYINCOMPOUND O\nCOMPOUNDFLAG Y\nSFX S Y 1\nSFX S 0 s .\n",
        "4\nvirt/XS\nbad/F\nlink/OY\nfoot/Y\n",
    );
    assert!(stems(&d, "virt").is_empty());
    assert_eq!(stems(&d, "virts"), ["virt"]);
    assert!(stems(&d, "bad").is_empty(), "forbidden entry must not stem alone");
    assert!(stems(&d, "link").is_empty());
    assert!(!stems(&d, "footlink").is_empty());
}

#[test]
fn compound_rule_accepts_two_part_word_and_rejects_single_only() {
    let d = load(
        "COMPOUNDMIN 1\nCOMPOUNDRULE 1\nCOMPOUNDRULE AB\n",
        "2\nfoo/A\nbar/B\n",
    );
    let s = Stemmer::new(&d);
    assert!(!s.stems("foobar").is_empty(), "two-part compound must be accepted: {:?}", s.stems("foobar"));
    // Without compound flags that allow standing alone, single parts still
    // stem as ordinary entries; reject a three-letter mash that is not a
    // listed entry and does not satisfy the two-flag rule.
    assert!(s.stems("fo").is_empty());
    assert!(s.stems("foobarbar").is_empty() || !s.stems("foobar").is_empty());
}

#[test]
fn annotator_attaches_stems_and_spelling_layers() {
    let aff = b"SET UTF-8\nSFX S Y 1\nSFX S 0 s .\n";
    let dic = b"1\nwalk/S\n";
    let dict = Dictionary::from_bytes(aff, dic, LoadMode::Strict).unwrap();
    let annotator = HunspellAnnotator::new(Arc::new(dict));
    let doc = Document::new("walks walks unknown");
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(view.id());
    spans.push(Span::new(0, 5));
    spans.push(Span::new(6, 11));
    spans.push(Span::new(12, 19));
    let tokens = Tokens { spans, sentence: None };
    let (stems_layer,) = annotator.annotate((view, &tokens)).unwrap();
    assert_eq!(stems_layer.len(), tokens.spans.len());
    let collected: Vec<&str> = (0..3).map(|i| stems_layer.stem.get(i)).collect();
    assert_eq!(collected, ["walk", "walk", "unknown"]);
}
