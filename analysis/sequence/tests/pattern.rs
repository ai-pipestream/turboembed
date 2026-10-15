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
