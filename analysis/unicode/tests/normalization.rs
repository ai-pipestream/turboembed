//! Conformance with NormalizationTest.txt and checks of the run splitting.

use std::collections::HashSet;

use turbo_unicode::normalize::{
    Form, QuickCheck, canonical_combining_class, canonical_decomposition, compatibility_decomposition, compose,
    for_each_run, is_normalized, normalize, quick_check,
};

mod common;

const FORMS: [Form; 4] = [Form::Nfc, Form::Nfd, Form::Nfkc, Form::Nfkd];

fn norm(form: Form, text: &str) -> String {
    let mut out = String::new();
    normalize(form, text, &mut out);
    out
}

/// The five columns of every test line, and the code points of part 1.
fn test_lines() -> (Vec<[String; 5]>, HashSet<u32>) {
    let text = common::read("NormalizationTest.txt");
    let mut lines = Vec::new();
    let mut part1 = HashSet::new();
    let mut part = String::new();
    for line in text.lines() {
        if let Some(name) = line.strip_prefix('@') {
            part = name.split_whitespace().next().unwrap_or("").to_owned();
            continue;
        }
        let data = line.split('#').next().unwrap_or("");
        if data.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = data.split(';').collect();
        assert_eq!(fields.len(), 6, "five columns and a trailing semicolon: {line}");
        let columns = [0, 1, 2, 3, 4].map(|i| common::string(fields[i]));
        if part == "Part1" {
            part1.insert(common::cp(fields[0]));
        }
        lines.push(columns);
    }
    (lines, part1)
}

#[test]
fn every_line_of_normalization_test_holds() {
    let (lines, _) = test_lines();
    assert!(lines.len() > 19_000, "the whole file was read");
    for [c1, c2, c3, c4, c5] in &lines {
        let context = format!("{c1:?}");
        for source in [c1, c2, c3] {
            assert_eq!(&norm(Form::Nfc, source), c2, "c2 == NFC(c1..c3) for {context}");
            assert_eq!(&norm(Form::Nfd, source), c3, "c3 == NFD(c1..c3) for {context}");
        }
        for source in [c4, c5] {
            assert_eq!(&norm(Form::Nfc, source), c4, "c4 == NFC(c4..c5) for {context}");
            assert_eq!(&norm(Form::Nfd, source), c5, "c5 == NFD(c4..c5) for {context}");
        }
        for source in [c1, c2, c3, c4, c5] {
            assert_eq!(&norm(Form::Nfkc, source), c4, "c4 == NFKC(c1..c5) for {context}");
            assert_eq!(&norm(Form::Nfkd, source), c5, "c5 == NFKD(c1..c5) for {context}");
        }
    }
}

#[test]
fn characters_outside_part_1_are_unchanged_by_every_form() {
    let (_, part1) = test_lines();
    let mut buffer = [0u8; 4];
    for c in common::all_chars().filter(|&c| !part1.contains(&(c as u32))) {
        let text = c.encode_utf8(&mut buffer);
        for form in FORMS {
            assert_eq!(norm(form, text), *text, "{form:?} of U+{:04X}", c as u32);
            assert!(is_normalized(form, text), "{form:?} is_normalized U+{:04X}", c as u32);
        }
    }
}

#[test]
fn is_normalized_agrees_with_normalize_on_every_column() {
    let (lines, _) = test_lines();
    for columns in &lines {
        for column in columns {
            for form in FORMS {
                assert_eq!(is_normalized(form, column), norm(form, column) == *column, "{form:?} {column:?}");
            }
        }
    }
}

/// Normalization of the whole text at once, built only from the
/// single-level public functions: the reference the run splitting must
/// agree with.
fn reference(form: Form, text: &str) -> String {
    fn decompose(form: Form, c: char, out: &mut Vec<char>) {
        let cp = c as u32;
        if (0xAC00..0xAC00 + 11172).contains(&cp) {
            let index = cp - 0xAC00;
            out.push(char::from_u32(0x1100 + index / 588).unwrap());
            out.push(char::from_u32(0x1161 + (index % 588) / 28).unwrap());
            if !index.is_multiple_of(28) {
                out.push(char::from_u32(0x11A7 + index % 28).unwrap());
            }
            return;
        }
        let mapping = canonical_decomposition(c)
            .or_else(|| if matches!(form, Form::Nfkc | Form::Nfkd) { compatibility_decomposition(c) } else { None });
        match mapping {
            Some(mapping) => mapping.iter().for_each(|&m| decompose(form, m, out)),
            None => out.push(c),
        }
    }
    let mut chars = Vec::new();
    for c in text.chars() {
        decompose(form, c, &mut chars);
    }
    // Canonical ordering by repeated adjacent swaps (bubble sort, stable).
    let mut swapped = true;
    while swapped {
        swapped = false;
        for i in 1..chars.len() {
            let (a, b) = (canonical_combining_class(chars[i - 1]), canonical_combining_class(chars[i]));
            if b != 0 && a > b {
                chars.swap(i - 1, i);
                swapped = true;
            }
        }
    }
    if matches!(form, Form::Nfc | Form::Nfkc) {
        // Canonical composition as UAX #15 states it: a character composes
        // with the last starter when no character between them is blocking.
        let mut i = 1;
        while i < chars.len() {
            let class = canonical_combining_class(chars[i]);
            let starter = (0..i).rev().find(|&s| canonical_combining_class(chars[s]) == 0);
            if let Some(s) = starter {
                let blocked = (s + 1..i).any(|b| {
                    let between = canonical_combining_class(chars[b]);
                    between == 0 || between >= class
                });
                if !blocked && let Some(composite) = compose(chars[s], chars[i]) {
                    chars[s] = composite;
                    chars.remove(i);
                    continue;
                }
            }
            i += 1;
        }
    }
    chars.into_iter().collect()
}

#[test]
fn run_splitting_agrees_with_whole_text_normalization() {
    let (lines, _) = test_lines();
    // All source columns joined without separators: runs meet in every
    // combination the file has, so splitting at the wrong place shows.
    for column in 0..5 {
        let text: String = lines.iter().map(|columns| columns[column].as_str()).collect();
        for form in FORMS {
            assert!(norm(form, &text) == reference(form, &text), "{form:?} of column {} joined", column + 1);
        }
    }
}

#[test]
fn for_each_run_covers_the_text_and_reports_only_changes() {
    let (lines, _) = test_lines();
    let mut text: String = lines.iter().take(3000).map(|columns| columns[0].as_str()).collect();
    text.push_str("plain ascii tail");
    for form in FORMS {
        let mut next = 0;
        let mut rebuilt = String::new();
        let mut previous_unchanged = false;
        for_each_run(form, &text, |range, normalized| {
            assert_eq!(range.start, next, "ranges are contiguous");
            assert!(range.end > range.start, "ranges are not empty");
            next = range.end;
            match normalized {
                Some(normalized) => {
                    assert_ne!(normalized, &text[range.clone()], "a changed run differs");
                    rebuilt.push_str(normalized);
                    previous_unchanged = false;
                }
                None => {
                    assert!(!previous_unchanged, "unchanged ranges are merged");
                    rebuilt.push_str(&text[range]);
                    previous_unchanged = true;
                }
            }
        });
        assert_eq!(next, text.len(), "the ranges reach the end");
        assert_eq!(rebuilt, norm(form, &text));
    }
}

#[test]
fn for_each_run_reports_normalized_text_once_and_empty_text_never() {
    let mut calls = Vec::new();
    for_each_run(Form::Nfc, "caf\u{E9} na\u{EF}ve", |range, normalized| calls.push((range, normalized.is_some())));
    assert_eq!(calls, vec![(0..12, false)]);
    for_each_run(Form::Nfc, "", |_, _| panic!("no call for empty text"));
}

#[test]
fn for_each_run_reports_each_changed_run_with_its_source_range() {
    let text = "Cafe\u{301} and e\u{301}t\u{E9}";
    let mut calls = Vec::new();
    for_each_run(Form::Nfc, text, |range, normalized| calls.push((range, normalized.map(str::to_owned))));
    assert_eq!(
        calls,
        vec![
            (0..3, None),
            (3..6, Some("\u{E9}".to_owned())),
            (6..11, None),
            (11..14, Some("\u{E9}".to_owned())),
            (14..17, None),
        ]
    );
}

#[test]
fn single_level_decompositions_and_composition() {
    assert_eq!(canonical_decomposition('\u{E9}'), Some(&['e', '\u{301}'][..]));
    // U+1E09 decomposes to U+00E7 U+0301 one level deep, not to c U+0327 U+0301.
    assert_eq!(canonical_decomposition('\u{1E09}'), Some(&['\u{E7}', '\u{301}'][..]));
    assert_eq!(canonical_decomposition('a'), None);
    assert_eq!(canonical_decomposition('\u{AC00}'), None, "Hangul is algorithmic");
    assert_eq!(compatibility_decomposition('\u{FB01}'), Some(&['f', 'i'][..]));
    assert_eq!(compatibility_decomposition('\u{E9}'), None, "a canonical mapping is not a compatibility one");
    assert_eq!(compose('e', '\u{301}'), Some('\u{E9}'));
    assert_eq!(compose('\u{1100}', '\u{1161}'), Some('\u{AC00}'));
    assert_eq!(compose('\u{AC00}', '\u{11A8}'), Some('\u{AC01}'));
    assert_eq!(compose('\u{AC01}', '\u{11A8}'), None, "an LVT syllable takes no trailing consonant");
    // U+0958 DEVANAGARI LETTER QA is excluded from composition.
    assert_eq!(canonical_decomposition('\u{958}'), Some(&['\u{915}', '\u{93C}'][..]));
    assert_eq!(compose('\u{915}', '\u{93C}'), None);
    assert_eq!(canonical_combining_class('\u{301}'), 230);
    assert_eq!(canonical_combining_class('\u{327}'), 202);
    assert_eq!(canonical_combining_class('a'), 0);
}

#[test]
fn quick_check_values() {
    assert_eq!(quick_check(Form::Nfc, 'a'), QuickCheck::Yes);
    assert_eq!(quick_check(Form::Nfc, '\u{301}'), QuickCheck::Maybe);
    assert_eq!(quick_check(Form::Nfd, '\u{E9}'), QuickCheck::No);
    assert_eq!(quick_check(Form::Nfkc, '\u{FB01}'), QuickCheck::No);
    assert_eq!(quick_check(Form::Nfc, '\u{FB01}'), QuickCheck::Yes);
    assert_eq!(quick_check(Form::Nfc, '\u{958}'), QuickCheck::No);
    assert!(!is_normalized(Form::Nfc, "e\u{301}x"));
    assert!(is_normalized(Form::Nfd, "a\u{316}\u{301}"));
    assert!(!is_normalized(Form::Nfd, "a\u{301}\u{316}"), "marks out of canonical order");
}
