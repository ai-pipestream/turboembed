//! Compares the properties and dictionary readers with what
//! java.util.Properties and OpenNLP read from the same files
//! (fixtures/text, made by fixtures/src/MakeTextFixtures.java).

mod common;

use common::unescape;
use turbo_maxent::{Dictionary, Entries, Properties};

fn read(name: &str) -> Vec<u8> {
    std::fs::read(common::fixtures().join("text").join(name)).unwrap()
}

#[test]
fn properties_match_java() {
    let properties = Properties::load(&read("tricky.properties")).unwrap();
    let mut got: Vec<(String, String)> = properties.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    got.sort();
    let expected = String::from_utf8(read("tricky.properties.expected")).unwrap();
    let mut want: Vec<(String, String)> = expected
        .lines()
        .skip(1) // the header; a key may itself start with '#'
        .map(|l| {
            let (k, v) = l.split_once('\t').unwrap();
            (unescape(k), unescape(v))
        })
        .collect();
    // Java sorted by UTF-16 code units; all keys here are in the BMP, where
    // that order is the order of the strings' characters.
    want.sort();
    assert_eq!(got, want);
}

/// A raw entry as compared: its attributes sorted by name, and its tokens.
type RawEntry = (Vec<(String, String)>, Vec<String>);

struct DictionaryExpectation {
    case_sensitive: bool,
    size: usize,
    min_tokens: usize,
    max_tokens: usize,
    set: Vec<Vec<String>>,
    probes: Vec<(bool, Vec<String>)>,
    entries: Vec<RawEntry>,
}

fn dictionary_expectation() -> DictionaryExpectation {
    let text = String::from_utf8(read("tricky.dictionary.expected")).unwrap();
    let mut e = DictionaryExpectation {
        case_sensitive: false,
        size: 0,
        min_tokens: 0,
        max_tokens: 0,
        set: Vec::new(),
        probes: Vec::new(),
        entries: Vec::new(),
    };
    for line in text.lines() {
        let mut fields = line.split(['\t', ' ']);
        match fields.next().unwrap() {
            "case_sensitive" => e.case_sensitive = fields.next() == Some("true"),
            "size" => e.size = fields.next().unwrap().parse().unwrap(),
            "min_tokens" => e.min_tokens = fields.next().unwrap().parse().unwrap(),
            "max_tokens" => e.max_tokens = fields.next().unwrap().parse().unwrap(),
            _ => {
                let mut fields = line.split('\t');
                match fields.next().unwrap() {
                    "set" => e.set.push(fields.map(unescape).collect()),
                    "contains" => {
                        let found = fields.next() == Some("true");
                        e.probes.push((found, fields.map(unescape).collect()));
                    }
                    "entry" => {
                        let mut attributes = Vec::new();
                        let mut tokens = Vec::new();
                        for f in fields {
                            match f.strip_prefix('@') {
                                Some(a) => {
                                    let (k, v) = a.split_once('=').unwrap();
                                    attributes.push((unescape(k), unescape(v)));
                                }
                                None => tokens.push(unescape(f)),
                            }
                        }
                        e.entries.push((attributes, tokens));
                    }
                    other => panic!("unexpected line kind {other}"),
                }
            }
        }
    }
    e
}

#[test]
fn the_dictionary_set_matches_opennlp() {
    let want = dictionary_expectation();
    let d = Dictionary::read(&read("tricky.dictionary")).unwrap();
    assert_eq!(d.is_case_sensitive(), want.case_sensitive);
    assert_eq!(d.len(), want.size);
    assert_eq!((d.min_tokens(), d.max_tokens()), (want.min_tokens, want.max_tokens));

    // Java iterates the entries as first written; this dictionary is case
    // insensitive, so the set here holds them lowercased with Java's
    // per character mapping.
    let lower = |s: &str| -> String {
        s.chars().map(|c| char::from_u32(turbo_unicode::java::code_point::to_lower_case(c as u32)).unwrap()).collect()
    };
    let mut got: Vec<Vec<String>> = d.entries().map(|e| e.map(str::to_string).collect()).collect();
    got.sort();
    let mut set: Vec<Vec<String>> = want.set.iter().map(|e| e.iter().map(|t| lower(t)).collect()).collect();
    set.sort();
    assert_eq!(got, set);

    let mut scratch = String::new();
    for (found, probe) in &want.probes {
        let tokens: Vec<&str> = probe.iter().map(String::as_str).collect();
        assert_eq!(d.contains_tokens(&tokens, &mut scratch), *found, "probe {probe:?}");
        if let [token] = &tokens[..] {
            assert_eq!(d.contains(token), *found, "probe {token:?}");
        }
    }
    assert_eq!(want.probes.len(), 19);
}

#[test]
fn raw_entries_match_opennlp() {
    let want = dictionary_expectation();
    let entries = Entries::read(&read("tricky.dictionary")).unwrap();
    assert!(!entries.is_case_sensitive());
    let got: Vec<RawEntry> = entries
        .iter()
        .map(|e| {
            let mut attributes: Vec<(String, String)> =
                e.attributes.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            attributes.sort();
            (attributes, e.tokens.iter().map(|t| t.to_string()).collect())
        })
        .collect();
    assert_eq!(got, want.entries);
}
