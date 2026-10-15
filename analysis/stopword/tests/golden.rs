//! Compares the lists, the code resolution and the lookup with what a JVM
//! running OpenNLP returned (the goldens of `StopwordGolden.java` in
//! `analysis/goldens-tool`, under `analysis/goldens/data/stopword`,
//! described in `README.md`).

use std::collections::HashMap;

use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_stopword::{StopwordLanguage, StopwordList};
use turbo_unicode::java::to_lower_char;

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const STR: ColumnType = ColumnType::Str;
const U32: ColumnType = ColumnType::U32;

/// The layers of the stopword goldens, all anchored on the document.
const LAYERS: &[(&str, &[ColumnSpec])] = &[
    (
        "turbo:lists",
        &[col("language", STR), col("words", U32), col("case-sensitive", U32), col("resource-sha256", STR)],
    ),
    ("turbo:list-words", &[col("language", STR), col("word", STR)]),
    ("turbo:codes", &[col("code", STR), col("list", STR)]),
    ("turbo:words", &[col("word", STR)]),
    ("turbo:stopwords", &[col("language", STR), col("word", U32)]),
    ("turbo:filter-cases", &[col("case", U32), col("language", STR)]),
    ("turbo:filter-tokens", &[col("case", U32), col("word", U32), col("filter-kept", U32), col("tokenizer-kept", U32)]),
    ("turbo:letters", &[col("code-point", U32), col("lower", U32)]),
];

/// The golden of a run on its input.
fn read(run: &str, input: &str) -> Golden {
    let shapes: Vec<Shape> =
        LAYERS.iter().map(|&(id, columns)| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns }).collect();
    let path = turbo_goldens::data_dir().join("stopword").join(run).join("stopword").join(format!("{input}.golden"));
    Golden::read_with(&path, &shapes).unwrap_or_else(|e| panic!("{e}"))
}

fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layer(id).unwrap_or_else(|| panic!("no layer {id}"))
}

fn u32s<'l>(l: &'l GoldenLayer, name: &str) -> &'l [u32] {
    l.u32s(name).unwrap_or_else(|| panic!("{}: no u32 column {name}", l.id))
}

fn strs(l: &GoldenLayer, name: &str) -> Vec<String> {
    l.strs(name).unwrap_or_else(|| panic!("{}: no str column {name}", l.id)).iter().map(str::to_string).collect()
}

fn language(code: &str) -> StopwordLanguage {
    StopwordLanguage::from_code(code).unwrap_or_else(|| panic!("no list for {code}"))
}

/// The words the lookup was asked about, by row of `turbo:words`.
fn words() -> Vec<String> {
    let words = strs(layer(&read("words", "words"), "turbo:words"), "word");
    assert!(words.len() > 20_000);
    words
}

#[test]
fn every_list_is_the_one_opennlp_loads() {
    let g = read("lists", "words");
    let summary = layer(&g, "turbo:lists");
    let codes = strs(summary, "language");
    let order: Vec<StopwordLanguage> = codes.iter().map(|c| language(c)).collect();
    assert_eq!(order, StopwordLanguage::ALL);
    assert!(u32s(summary, "case-sensitive").iter().all(|&c| c == 0), "bundled lists are case-insensitive");
    // The crate bundles the jar's resource files byte for byte.
    for (code, sha) in codes.iter().zip(strs(summary, "resource-sha256")) {
        let path = format!("{}/data/{code}.txt", env!("CARGO_MANIFEST_DIR"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!(turbo_goldens::hex::to_hex(&turbo_goldens::sha256(&bytes)), sha, "{path}");
    }
    let l = layer(&g, "turbo:list-words");
    let mut lists: HashMap<String, Vec<String>> = HashMap::new();
    for (code, word) in strs(l, "language").into_iter().zip(strs(l, "word")) {
        lists.entry(code).or_default().push(word);
    }
    for (c, code) in codes.iter().enumerate() {
        let mut words = lists.remove(code).unwrap_or_default();
        assert_eq!(words.len(), u32s(summary, "words")[c] as usize, "{code}");
        // Java sorted by UTF-16 unit; with every word in the BMP that is
        // the byte order, but sort again rather than rely on it.
        words.sort_unstable();
        assert_eq!(StopwordList::bundled(language(code)).words(), words, "{code}");
    }
}

#[test]
fn language_codes_resolve_as_for_language_resolves_them() {
    let g = read("codes", "codes");
    let l = layer(&g, "turbo:codes");
    let mut supported = 0;
    for (code, result) in strs(l, "code").into_iter().zip(strs(l, "list")) {
        let got = StopwordLanguage::from_code(&code).map(|l| l.iso639_1()).unwrap_or("");
        assert_eq!(got, result, "{code:?}");
        supported += usize::from(!result.is_empty());
    }
    assert_eq!(supported, 25);
}

#[test]
fn every_word_is_flagged_as_the_jvm_flags_it() {
    let words = words();
    let g = read("words", "words");
    let l = layer(&g, "turbo:stopwords");
    let mut stops: HashMap<String, Vec<bool>> = HashMap::new();
    for (code, &i) in strs(l, "language").into_iter().zip(u32s(l, "word")) {
        stops.entry(code).or_insert_with(|| vec![false; words.len()])[i as usize] = true;
    }
    for language in StopwordLanguage::ALL {
        let code = language.iso639_1();
        let stop = stops.get(code).unwrap_or_else(|| panic!("{code}: no stopwords"));
        let list = StopwordList::bundled(language);
        for (word, &want) in words.iter().zip(stop) {
            assert_eq!(list.contains(word), want, "{code} {word:?}");
        }
    }
    assert_eq!(stops.len(), 11);
}

#[test]
fn removal_keeps_the_tokens_that_are_not_flagged() {
    let words = words();
    let g = read("filter", "words");
    let cases = layer(&g, "turbo:filter-cases");
    let codes = strs(cases, "language");
    let tokens = layer(&g, "turbo:filter-tokens");
    let (case, word) = (u32s(tokens, "case"), u32s(tokens, "word"));
    let (filter_kept, tokenizer_kept) = (u32s(tokens, "filter-kept"), u32s(tokens, "tokenizer-kept"));
    let mut rows = vec![Vec::new(); cases.len()];
    for r in 0..tokens.len() {
        rows[case[r] as usize].push(r);
    }
    let mut per_language: HashMap<&str, usize> = HashMap::new();
    let mut flags = Vec::new();
    for (c, rows) in rows.iter().enumerate() {
        let sequence: Vec<&str> = rows.iter().map(|&r| words[word[r] as usize].as_str()).collect();
        flags.clear();
        StopwordList::bundled(language(&codes[c])).flag(sequence.iter().copied(), &mut flags);
        let kept: Vec<bool> = flags.iter().map(|&stop| !stop).collect();
        let filter: Vec<bool> = rows.iter().map(|&r| filter_kept[r] == 1).collect();
        let tokenizer: Vec<bool> = rows.iter().map(|&r| tokenizer_kept[r] == 1).collect();
        assert_eq!(kept, filter, "filter: case {c} {sequence:?}");
        assert_eq!(kept, tokenizer, "tokenizePos: case {c} {sequence:?}");
        *per_language.entry(codes[c].as_str()).or_default() += 1;
    }
    assert_eq!(per_language.len(), 11);
    assert!(per_language.values().all(|&n| n == 150));
}

/// Every code point whose Java lowercase is a character of a bundled word
/// lowercases to the same character here, and no other code point
/// lowercases to one: the lookup's case mapping is the JVM's wherever it
/// can change an answer.
#[test]
fn lowercase_mapping_into_list_letters_is_the_jvms() {
    let g = read("letters", "words");
    let l = layer(&g, "turbo:letters");
    let java: HashMap<u32, u32> = u32s(l, "code-point").iter().copied().zip(u32s(l, "lower").iter().copied()).collect();
    let letters: std::collections::HashSet<u32> = java.values().copied().collect();
    for c in (0..=0x10FFFF).filter_map(char::from_u32) {
        let lower = to_lower_char(c) as u32;
        match java.get(&(c as u32)) {
            Some(&want) => assert_eq!(lower, want, "U+{:04X}", c as u32),
            None => assert!(!letters.contains(&lower), "U+{:04X} lowercases to U+{lower:04X}", c as u32),
        }
    }
}
