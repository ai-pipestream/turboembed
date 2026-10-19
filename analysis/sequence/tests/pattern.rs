//! `StringPattern` and `FeatureGeneratorUtil.tokenFeature` against the
//! JVM, in both modes (the goldens of `StringPatternGolden.java` in
//! `analysis/goldens-tool`).

mod common;

use common::{layer, lossless_units, strs, u32s};
use turbo_sequence::pattern::token_feature_utf16;
use turbo_sequence::{PatternMode, StringPattern};

/// The flag columns, in the order the checks below list the flags.
const FLAGS: [&str; 13] = [
    "all-letter",
    "initial-capital",
    "all-capital",
    "all-lower-case",
    "all-digit",
    "all-hiragana",
    "all-katakana",
    "period",
    "comma",
    "slash",
    "digit",
    "hyphen",
    "letters",
];

fn check(run: &str, mode: PatternMode) {
    let mut checked = 0;
    let mut failures = Vec::new();
    for id in ["tokens", "sentences"] {
        let g = common::read("string-pattern", run, "sequence", id);
        let l = layer(&g, "turbo:string-patterns");
        let tokens = strs(l, "token");
        let flags: Vec<&[u32]> = FLAGS.iter().map(|f| u32s(l, f)).collect();
        let digits = u32s(l, "digits");
        let classes = strs(l, "class");
        for (row, token) in tokens.iter().enumerate() {
            let units = lossless_units(token);
            let p = StringPattern::recognize_utf16(&units, mode);
            let actual = [
                p.is_all_letter(),
                p.is_initial_capital_letter(),
                p.is_all_capital_letter(),
                p.is_all_lower_case_letter(),
                p.is_all_digit(),
                p.is_all_hiragana(),
                p.is_all_katakana(),
                p.contains_period(),
                p.contains_comma(),
                p.contains_slash(),
                p.contains_digit(),
                p.contains_hyphen(),
                p.contains_letters(),
            ];
            let actual: String = actual.iter().map(|&b| if b { '1' } else { '0' }).collect();
            let expected: String = flags.iter().map(|f| if f[row] == 1 { '1' } else { '0' }).collect();
            let actual = format!("{actual}\t{}\t{}", p.digits(), token_feature_utf16(&units, mode));
            let expected = format!("{expected}\t{}\t{}", digits[row], classes[row]);
            if actual != expected {
                failures.push(format!("{id} {token}: expected {expected}, got {actual}"));
            }
            checked += 1;
        }
    }
    assert!(checked > 1000, "only {checked} tokens");
    assert!(failures.is_empty(), "{} of {checked} differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn utf16_units_match_opennlp_3x() {
    check("utf16-units", PatternMode::Utf16Units);
}

#[test]
fn code_points_match_the_unicode_branch() {
    check("code-points", PatternMode::CodePoints);
}

#[test]
fn str_and_utf16_entry_points_agree() {
    for token in ["Hello", "A.", "𝐀.", "1,234.5", "ひらがな", "カタカナ", "", "x-y/z"] {
        let units: Vec<u16> = token.encode_utf16().collect();
        for mode in [PatternMode::Utf16Units, PatternMode::CodePoints] {
            assert_eq!(StringPattern::recognize(token, mode), StringPattern::recognize_utf16(&units, mode));
            assert_eq!(turbo_sequence::token_feature(token, mode), token_feature_utf16(&units, mode));
        }
    }
}

fn class(token: &str) -> &'static str {
    turbo_sequence::token_feature(token, PatternMode::Utf16Units)
}

fn class_cp(token: &str) -> &'static str {
    turbo_sequence::token_feature(token, PatternMode::CodePoints)
}

#[test]
fn the_classes_of_common_tokens() {
    for (token, want) in [
        ("hello", "lc"),
        ("12", "2d"),
        ("1999", "4d"),
        ("A4", "an"),
        ("1-23", "dd"),
        ("1/23", "ds"),
        ("1,23", "dc"),
        ("3.55", "dp"),
        ("123", "num"),
        ("I", "sc"),
        ("IBM", "ac"),
        ("A.", "cp"),
        ("\u{C4}.", "cp"),
        ("Paris", "ic"),
        ("...", "other"),
        ("\u{3072}\u{3089}\u{304C}\u{306A}", "jah"),
        ("\u{30AB}\u{30BF}\u{30FC}", "jak"),
    ] {
        assert_eq!(class(token), want, "{token:?}");
    }
}

#[test]
fn an_empty_token_is_hiragana_by_units_and_nothing_by_code_points() {
    assert_eq!(class(""), "jah");
    assert_eq!(class_cp(""), "other");
}

#[test]
fn supplementary_letters_are_surrogates_by_units() {
    // U+1D400 MATHEMATICAL BOLD CAPITAL A.
    let p = StringPattern::recognize("\u{1D400}", PatternMode::Utf16Units);
    assert!(!p.contains_letters());
    assert_eq!(p.token_class(), "other");
    let p = StringPattern::recognize("\u{1D400}", PatternMode::CodePoints);
    assert!(p.contains_letters() && p.is_all_capital_letter());
    assert_eq!(p.token_class(), "ac");
    assert_eq!(class_cp("\u{1D400}."), "cp");
    assert_eq!(class("\u{1D400}."), "other");
}

#[test]
fn marks_continue_a_letter_by_code_points() {
    // "e" and a combining acute accent.
    assert_eq!(class("e\u{301}"), "other");
    assert_eq!(class_cp("e\u{301}"), "lc");
}

#[test]
fn lone_surrogates_are_malformed_by_code_points() {
    let p = StringPattern::recognize_utf16(&[0x41, 0xD800], PatternMode::CodePoints);
    assert!(!p.is_initial_capital_letter() && !p.is_all_capital_letter() && p.contains_letters());
    let p = StringPattern::recognize_utf16(&[0x41, 0xD800], PatternMode::Utf16Units);
    assert!(p.is_initial_capital_letter());
}
