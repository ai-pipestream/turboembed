//! Reviewed snapshots of the bundled lists: digests, sizes and words.
//!
//! `data/SHA256SUMS` is the digest snapshot. The sizes are the distinct
//! words the build script loads from each file. No JVM output.

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbo_sha256::sha256;
use turbo_stopword::{StopwordLanguage, StopwordList};
use turbo_unicode::java::{to_lower_char, to_upper_char};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}

/// Distinct words as the build script loads them.
fn words_from_file(code: &str) -> Vec<String> {
    let text = std::fs::read_to_string(data_dir().join(format!("{code}.txt"))).unwrap();
    let mut words: Vec<String> = Vec::new();
    for line in text.split(['\n', '\r']) {
        let line = line.trim_matches(char::is_whitespace);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        assert!(!line.contains(char::is_whitespace), "{code}: multi-word {line:?}");
        words.push(line.to_string());
    }
    words.sort_unstable();
    words.dedup();
    words
}

#[test]
fn digests_match_the_checked_in_sums() {
    let sums = std::fs::read_to_string(data_dir().join("SHA256SUMS")).unwrap();
    let mut seen = BTreeSet::new();
    for line in sums.lines().filter(|l| !l.is_empty()) {
        let (want, name) = line.split_once(' ').unwrap();
        let name = name.trim();
        let bytes = std::fs::read(data_dir().join(name)).unwrap();
        assert_eq!(sha256(&bytes).to_hex(), want, "{name}");
        seen.insert(name.to_string());
    }
    for code in ["bg", "da", "de", "en", "es", "fi", "fr", "it", "nl", "pt", "ru"] {
        assert!(seen.contains(&format!("{code}.txt")), "missing {code}");
    }
}

#[test]
fn bundled_lists_match_the_data_files() {
    let sizes = [190, 94, 231, 174, 308, 229, 154, 279, 101, 203, 159];
    for (i, language) in StopwordLanguage::ALL.into_iter().enumerate() {
        let list = StopwordList::bundled(language);
        let from_file = words_from_file(language.iso639_1());
        assert_eq!(list.len(), sizes[i], "{language:?}");
        let expected: Vec<&str> = from_file.iter().map(String::as_str).collect();
        assert_eq!(list.words(), expected.as_slice());
    }
}

#[test]
fn known_stopwords_and_keepers() {
    let en = StopwordList::bundled(StopwordLanguage::English);
    for yes in ["the", "THE", "The", "and", "i"] {
        assert!(en.contains(yes), "{yes}");
    }
    for no in ["cat", "dog", "también", ""] {
        assert!(!en.contains(no), "{no:?}");
    }
    let de = StopwordList::bundled(StopwordLanguage::German);
    assert!(de.contains("daß"));
    assert!(de.contains("DA\u{1E9E}"));
    assert!(!de.contains("DASS"));
}

#[test]
fn every_bundled_word_is_found_in_its_case_folds() {
    for language in StopwordLanguage::ALL {
        let list = StopwordList::bundled(language);
        for &word in list.words() {
            assert!(list.contains(word), "{language:?} {word}");
            let upper: String = word.chars().map(to_upper_char).collect();
            assert!(list.contains(&upper), "{language:?} upper {word}");
            let mixed: String = word
                .chars()
                .enumerate()
                .map(|(i, c)| if i % 2 == 0 { to_upper_char(c) } else { to_lower_char(c) })
                .collect();
            assert!(list.contains(&mixed), "{language:?} mixed {word}");
        }
    }
}
