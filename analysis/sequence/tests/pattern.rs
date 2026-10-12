//! `StringPattern` and `FeatureGeneratorUtil.tokenFeature` against the
//! JVM, in both modes.

mod common;

use common::{body, field, unescape_units};
use turbo_sequence::pattern::token_feature_utf16;
use turbo_sequence::{PatternMode, StringPattern};

fn check(file: &str, mode: PatternMode) {
    let text = common::read(file);
    let mut lines = body(&text, "string-pattern");
    field(lines.next().unwrap(), "mode");
    field(lines.next().unwrap(), "columns");
    let mut checked = 0;
    let mut failures = Vec::new();
    for line in lines {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 4, "{line}");
        let units = unescape_units(cols[0]);
        let p = StringPattern::recognize_utf16(&units, mode);
        let flags = [
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
        let flags: String = flags.iter().map(|&b| if b { '1' } else { '0' }).collect();
        let actual = format!("{flags}\t{}\t{}", p.digits(), token_feature_utf16(&units, mode));
        let expected = cols[1..].join("\t");
        if actual != expected {
            failures.push(format!("{}: expected {expected}, got {actual}", cols[0]));
        }
        checked += 1;
    }
    assert!(checked > 1000, "only {checked} tokens");
    assert!(failures.is_empty(), "{} of {checked} differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn utf16_units_match_opennlp_3x() {
    check("expected/pattern-units.txt", PatternMode::Utf16Units);
}

#[test]
fn code_points_match_the_unicode_branch() {
    check("expected/pattern-code-points.txt", PatternMode::CodePoints);
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
