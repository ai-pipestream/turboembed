//! Property tests: fixture dictionaries stem stably; offsets round-trip.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use turbo_analysis::OffsetTable;
use turbo_hunspell::{Dictionary, LoadMode, Stemmer};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/hunspell")
}

#[test]
fn fixture_affixes_stems_are_stable() {
    let aff = fs::read(fixtures().join("affixes.aff")).unwrap();
    let dic = fs::read(fixtures().join("affixes.dic")).unwrap();
    let dict = Dictionary::from_bytes(&aff, &dic, LoadMode::Strict).unwrap();
    let stemmer = Stemmer::new(&dict);
    for word in ["walk", "walks", "walking", "xxx"] {
        let a = stemmer.stems(word);
        let b = stemmer.stems(word);
        assert_eq!(a, b, "{word}");
        let one = stemmer.stem(word).unwrap_or_else(|| word.to_string());
        assert!(a.contains(&one) || a.is_empty() && one == word, "{word}: {one} in {a:?}");
    }
    let _ = Arc::new(dict);
}

#[test]
fn offset_tables_round_trip_on_stemmed_words() {
    let text = "walks walking café";
    let table = OffsetTable::build(text);
    for (byte, _) in text.char_indices() {
        let u16 = table.to_utf16(text, byte as u32);
        let cp = table.to_code_point(text, byte as u32);
        assert_eq!(table.from_utf16(text, u16).unwrap(), byte as u32);
        assert_eq!(table.from_code_point(text, cp).unwrap(), byte as u32);
    }
}
