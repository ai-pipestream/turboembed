//! Compares the lists, the code resolution and the lookup with the files a
//! JVM running OpenNLP wrote (`golden/`, described in `README.md`).

use std::collections::HashMap;

use turbo_stopword::{StopwordLanguage, StopwordList};
use turbo_unicode::java::to_lower_char;

fn expected(name: &str) -> String {
    let path = format!("{}/golden/expected/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The lines after the header, which must name `component`.
fn body<'a>(text: &'a str, component: &str) -> impl Iterator<Item = &'a str> {
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(format!("component: {component}").as_str()));
    assert_eq!(lines.next(), Some("opennlp: cdf33c653d5d61d5af7d37ffee189a83bd8392ea"));
    assert!(lines.next().unwrap().starts_with("jdk: 21"));
    lines
}

/// A string escaped by the generator: `\uXXXX` per UTF-16 unit outside
/// printable ASCII, `\\`, `\t`, `\n` and `\r`.
fn unescape(s: &str) -> String {
    let mut units = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            units.push(c as u16);
            continue;
        }
        match chars.next() {
            Some('\\') => units.push(0x5C),
            Some('t') => units.push(0x09),
            Some('n') => units.push(0x0A),
            Some('r') => units.push(0x0D),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                units.push(u16::from_str_radix(&hex, 16).unwrap());
            }
            other => panic!("bad escape {other:?} in {s}"),
        }
    }
    String::from_utf16(&units).unwrap()
}

fn language(code: &str) -> StopwordLanguage {
    StopwordLanguage::from_code(code).unwrap_or_else(|| panic!("no list for {code}"))
}

fn indices(s: &str) -> Vec<usize> {
    s.split(' ').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()
}

#[test]
fn every_list_is_the_one_opennlp_loads() {
    let text = expected("lists.txt");
    let mut lists: Vec<(StopwordLanguage, Vec<String>)> = Vec::new();
    for line in body(&text, "lists") {
        if let Some(rest) = line.strip_prefix("list\t") {
            let f: Vec<&str> = rest.split('\t').collect();
            assert_eq!(f[2], "false", "bundled lists are case-insensitive");
            lists.push((language(f[0]), Vec::with_capacity(f[1].parse().unwrap())));
        } else {
            lists.last_mut().unwrap().1.push(unescape(line));
        }
    }
    let order: Vec<StopwordLanguage> = lists.iter().map(|l| l.0).collect();
    assert_eq!(order, StopwordLanguage::ALL);
    for (language, mut words) in lists {
        // Java sorted by UTF-16 unit; with every word in the BMP that is
        // the byte order, but sort again rather than rely on it.
        words.sort_unstable();
        assert_eq!(StopwordList::bundled(language).words(), words, "{language:?}");
    }
}

#[test]
fn language_codes_resolve_as_for_language_resolves_them() {
    let text = expected("codes.txt");
    let mut supported = 0;
    for line in body(&text, "codes") {
        let (code, result) = line.split_once('\t').unwrap();
        let code = unescape(code);
        let got = StopwordLanguage::from_code(&code).map(|l| l.iso639_1()).unwrap_or("-");
        assert_eq!(got, result, "{code:?}");
        supported += usize::from(result != "-");
    }
    assert_eq!(supported, 25);
}

#[test]
fn every_word_is_flagged_as_the_jvm_flags_it() {
    let words_text = expected("words.txt");
    let words: Vec<String> = body(&words_text, "words").map(unescape).collect();
    let flags_text = expected("flags.txt");
    let mut checked = 0;
    for line in body(&flags_text, "flags") {
        let (code, stops) = line.split_once('\t').unwrap();
        let list = StopwordList::bundled(language(code));
        let mut stop = vec![false; words.len()];
        for i in indices(stops) {
            stop[i] = true;
        }
        for (word, &want) in words.iter().zip(&stop) {
            assert_eq!(list.contains(word), want, "{code} {word:?}");
        }
        checked += 1;
    }
    assert_eq!(checked, 11);
    assert!(words.len() > 20_000);
}

#[test]
fn removal_keeps_the_tokens_that_are_not_flagged() {
    let words_text = expected("words.txt");
    let words: Vec<String> = body(&words_text, "words").map(unescape).collect();
    let text = expected("filter.txt");
    let mut per_language: HashMap<&str, usize> = HashMap::new();
    let mut flags = Vec::new();
    for line in body(&text, "filter") {
        let f: Vec<&str> = line.split('\t').collect();
        let tokens: Vec<&str> = indices(f[1]).into_iter().map(|i| words[i].as_str()).collect();
        flags.clear();
        StopwordList::bundled(language(f[0])).flag(tokens.iter().copied(), &mut flags);
        let kept: Vec<usize> = flags.iter().enumerate().filter(|&(_, &s)| !s).map(|(i, _)| i).collect();
        assert_eq!(kept, indices(f[2]), "filter: {line}");
        assert_eq!(kept, indices(f[3]), "tokenizePos: {line}");
        *per_language.entry(f[0]).or_default() += 1;
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
    let text = expected("letters.txt");
    let mut java: HashMap<u32, u32> = HashMap::new();
    for line in body(&text, "letters") {
        let (cp, lower) = line.split_once(' ').unwrap();
        java.insert(u32::from_str_radix(cp, 16).unwrap(), u32::from_str_radix(lower, 16).unwrap());
    }
    let letters: std::collections::HashSet<u32> = java.values().copied().collect();
    for c in (0..=0x10FFFF).filter_map(char::from_u32) {
        let lower = to_lower_char(c) as u32;
        match java.get(&(c as u32)) {
            Some(&want) => assert_eq!(lower, want, "U+{:04X}", c as u32),
            None => assert!(!letters.contains(&lower), "U+{:04X} lowercases to U+{lower:04X}", c as u32),
        }
    }
}
