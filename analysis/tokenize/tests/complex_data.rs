//! The bundled dictionary files (`data/icu4x-segmenter/*.dict`): their
//! contents, and how they are made from ICU4X's baked segmenter data.
//!
//! `regenerate_from_icu4x` (ignored by default) reads ICU4X's
//! `provider/data/segmenter/data/segmenter_dictionary_extended_v2.rs.data`
//! from `TURBO_ICU4X_SEGMENTER_DATA`, rebuilds the four files from it and
//! compares them with the committed ones; with `TURBO_ICU4X_WRITE=1` it
//! writes them instead. `matches_icu4c_word_lists` (ignored) compares the
//! words with ICU4C's `icu4c/source/data/brkitr/dictionaries/*dict.txt`
//! in `TURBO_ICU4C_DICTIONARIES`. The data directory's README says which
//! versions were used.

use std::collections::BTreeSet;
use std::path::PathBuf;

use turbo_tokenize::complex::dictionary::MAGIC;
use turbo_tokenize::complex::{ComplexScript, Dictionary, bundled_dictionary};

/// The file name of each script's dictionary, and the name ICU4X gives it.
const FILES: [(ComplexScript, &str, &str); 4] = [
    (ComplexScript::Thai, "thai.dict", "thaidict"),
    (ComplexScript::Lao, "lao.dict", "laodict"),
    (ComplexScript::Khmer, "khmer.dict", "khmerdict"),
    (ComplexScript::Myanmar, "burmese.dict", "burmesedict"),
];

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data").join("icu4x-segmenter")
}

fn words(script: ComplexScript) -> BTreeSet<String> {
    let dictionary = Dictionary::from_bytes(bundled_dictionary(script)).unwrap();
    let mut words = BTreeSet::new();
    dictionary.for_each_word(|w| {
        assert!(words.insert(w.to_owned()), "{w} twice");
    });
    words
}

#[test]
fn bundled_dictionaries() {
    let expected = [
        (ComplexScript::Thai, 26383, 76),
        (ComplexScript::Lao, 30550, 93),
        (ComplexScript::Khmer, 81028, 96),
        (ComplexScript::Myanmar, 41120, 82),
    ];
    for (script, count, alphabet) in expected {
        let dictionary = Dictionary::from_bytes(bundled_dictionary(script)).unwrap();
        assert_eq!(dictionary.alphabet_len(), alphabet, "{script:?}");
        let words = words(script);
        assert_eq!(words.len(), count, "{script:?}");
        for word in &words {
            assert!(dictionary.contains(word), "{word}");
            assert!(!word.is_empty());
        }
    }
    let thai = Dictionary::from_bytes(bundled_dictionary(ComplexScript::Thai)).unwrap();
    assert!(thai.contains("ภาษา") && thai.contains("ไทย") && thai.contains("ประเทศไทย"));
    assert!(!thai.contains("ภาษาไทยภาษา"));
}

/// A Rust byte string literal's contents with its escapes resolved.
fn unescape(literal: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut chars = literal.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        match chars.next().expect("an escape") {
            'x' => {
                let hex: String = chars.by_ref().take(2).collect();
                out.push(u8::from_str_radix(&hex, 16).expect("a hexadecimal escape"));
            }
            'n' => out.push(b'\n'),
            'r' => out.push(b'\r'),
            't' => out.push(b'\t'),
            '0' => out.push(0),
            '\\' => out.push(b'\\'),
            '"' => out.push(b'"'),
            '\'' => out.push(b'\''),
            other => panic!("unknown escape \\{other}"),
        }
    }
    out
}

/// The byte string literals passed to `from_bytes_unchecked` in a baked
/// data file, in order.
fn byte_literals(source: &str) -> Vec<Vec<u8>> {
    const OPEN: &str = "from_bytes_unchecked(b\"";
    let mut out = Vec::new();
    let mut rest = source;
    while let Some(at) = rest.find(OPEN) {
        rest = &rest[at + OPEN.len()..];
        let mut end = 0;
        let bytes = rest.as_bytes();
        while bytes[end] != b'"' {
            end += if bytes[end] == b'\\' { 2 } else { 1 };
        }
        out.push(unescape(&rest[..end]));
        rest = &rest[end..];
    }
    out
}

/// The dictionary files rebuilt from ICU4X's baked
/// `segmenter_dictionary_extended_v2.rs.data`: each entry is a
/// `DictionaryBreakData::ZeroTrie` whose alphabet is a code point
/// inversion list of 3-byte little-endian code points and whose trie is
/// the store; the entries come in the order of their names (burmesedict,
/// khmerdict, laodict, thaidict).
fn rebuild(source: &str) -> Vec<(&'static str, Vec<u8>)> {
    // The lookup trie of the data: "und/burmesedict" is entry 0,
    // "und/khmerdict" 1, "und/laodict" 2 and "und/thaidict" 3.
    assert!(
        source.contains(r#"b"und\x1E\xC4bklt\x0B\x14\x1Burmesedict\x80hmerdict\x81aodict\x82haidict\x83""#),
        "the four dictionaries, in the order burmese, khmer, lao, thai"
    );
    assert_eq!(source.matches("DictionaryBreakData::ZeroTrie").count(), 4, "four ZeroTrie dictionaries");
    let literals = byte_literals(source);
    assert_eq!(literals.len(), 8, "an alphabet and a trie for each dictionary");
    let order = ["burmese.dict", "khmer.dict", "lao.dict", "thai.dict"];
    let mut files = Vec::new();
    for (k, name) in order.into_iter().enumerate() {
        let (alphabet, trie) = (&literals[2 * k], &literals[2 * k + 1]);
        assert_eq!(alphabet.len() % 6, 0, "{name}: inversion list of 3-byte code points in pairs");
        let points: Vec<u32> =
            alphabet.chunks(3).map(|p| u32::from(p[0]) | u32::from(p[1]) << 8 | u32::from(p[2]) << 16).collect();
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&((points.len() / 2) as u32).to_le_bytes());
        for p in &points {
            file.extend_from_slice(&p.to_le_bytes());
        }
        file.extend_from_slice(trie);
        files.push((name, file));
    }
    files
}

#[test]
#[ignore = "needs ICU4X's baked segmenter data in TURBO_ICU4X_SEGMENTER_DATA"]
fn regenerate_from_icu4x() {
    let path = std::env::var("TURBO_ICU4X_SEGMENTER_DATA").expect("TURBO_ICU4X_SEGMENTER_DATA");
    let source = std::fs::read_to_string(&path).unwrap();
    let write = std::env::var("TURBO_ICU4X_WRITE").is_ok_and(|v| v == "1");
    for (name, bytes) in rebuild(&source) {
        let target = data_dir().join(name);
        if write {
            std::fs::write(&target, &bytes).unwrap();
        } else {
            assert!(std::fs::read(&target).unwrap() == bytes, "{name} differs from what ICU4X's data gives");
        }
    }
}

#[test]
#[ignore = "needs ICU4C's dictionary sources in TURBO_ICU4C_DICTIONARIES"]
fn matches_icu4c_word_lists() {
    let dir = PathBuf::from(std::env::var("TURBO_ICU4C_DICTIONARIES").expect("TURBO_ICU4C_DICTIONARIES"));
    for (script, _, icu_name) in FILES {
        let text = std::fs::read_to_string(dir.join(format!("{icu_name}.txt"))).unwrap();
        let mut icu = BTreeSet::new();
        for line in text.trim_start_matches('\u{FEFF}').lines() {
            let word = line.split('#').next().unwrap().trim();
            if !word.is_empty() {
                icu.insert(word.split('\t').next().unwrap().to_owned());
            }
        }
        let ours = words(script);
        let missing: Vec<_> = icu.difference(&ours).take(5).collect();
        let extra: Vec<_> = ours.difference(&icu).take(5).collect();
        assert!(missing.is_empty() && extra.is_empty(), "{icu_name}: missing {missing:?}, extra {extra:?}");
    }
}

#[test]
fn the_file_names_match_the_scripts() {
    for (script, file, _) in FILES {
        assert_eq!(std::fs::read(data_dir().join(file)).unwrap(), bundled_dictionary(script));
    }
}
