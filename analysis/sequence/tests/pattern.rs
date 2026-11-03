//! Unicode pattern recognition invariants.
use turbo_sequence::pattern::token_feature_utf16;
use turbo_sequence::{PatternMode, StringPattern};

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
