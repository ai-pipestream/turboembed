//! The digit fold. The expectations are those of OpenNLP's
//! SetBasedNormalizerTest and AlignedNormalizerPipelineTest, with UTF-16
//! offsets written as UTF-8 byte offsets.

mod folds_support;

use folds_support::{Rng, check_alignment, covered, forward, per_char_edit, run_checked};
use turbo_normalize::{Digits, Normalizer, apply, normalize};

/// The value of a decimal digit for the digits these tests use, from the
/// zero of its block: Arabic-Indic, Devanagari, Thai, fullwidth,
/// mathematical bold and segmented digits.
fn digit_value(c: char) -> Option<u32> {
    let zeros = [0x660, 0x966, 0xE50, 0xFF10, 0x1D7CE, 0x1FBF0];
    zeros.iter().find_map(|&z| (z..z + 10).contains(&(c as u32)).then(|| c as u32 - z))
}

fn digit_fold(c: char) -> Option<&'static str> {
    const ASCII: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
    digit_value(c).map(|v| ASCII[v as usize])
}

fn digits(s: &str) -> String {
    run_checked(&Digits, s, per_char_edit(digit_fold))
}

#[test]
fn id() {
    assert_eq!(Digits::ID, "turbo:normalize:digits");
}

// SetBasedNormalizerTest.testDigitsMapDecimalDigitsToAscii
#[test]
fn decimal_digits_become_ascii() {
    assert_eq!(digits("\u{661}\u{662}\u{663}"), "123");
    assert_eq!(digits("\u{967}\u{968}"), "12");
    assert_eq!(digits("\u{FF11}\u{FF15}"), "15");
    assert_eq!(digits("a\u{665}b"), "a5b");
    assert_eq!(digits("\u{E53}\u{1FBF9}"), "39");
}

// SetBasedNormalizerTest.testDigitsLeaveAsciiAndNonDecimalNumeralsAlone
#[test]
fn ascii_digits_and_other_numerals_are_left_alone() {
    assert_eq!(digits("0123456789"), "0123456789");
    assert_eq!(digits("\u{B2}"), "\u{B2}");
    assert_eq!(digits("\u{2160}"), "\u{2160}");
    assert_eq!(digits("\u{BD}"), "\u{BD}");
    assert_eq!(digits("\u{2460}\u{3007}"), "\u{2460}\u{3007}");
    assert_eq!(digits("abc"), "abc");
    // Fullwidth letters have digit values of 10 and above, so radix 10
    // never maps them.
    assert_eq!(digits("\u{FF21}\u{FF41}"), "\u{FF21}\u{FF41}");
}

// SetBasedNormalizerTest.testDigitsReturnAsciiDigitTextUncopied
#[test]
fn ascii_digit_text_is_borrowed() {
    assert!(matches!(apply(&Digits, "version 42 of 2026"), std::borrow::Cow::Borrowed(_)));
}

// AlignedNormalizerPipelineTest.digitFoldOfSupplementaryDigitMapsSpanBackToOriginal
#[test]
fn a_supplementary_digit_maps_back_to_its_four_bytes() {
    let src = "a\u{1D7CE}b";
    let (out, a) = normalize(&Digits, src);
    assert_eq!(out, "a0b");
    assert_eq!(covered(src, &a, 2, 3), "b");
    assert_eq!(covered(src, &a, 1, 2), "\u{1D7CE}");
}

// AlignedNormalizerPipelineTest.digitFoldOfMixedAsciiAndNonAsciiDigitsMapsSpansBackToOriginal
#[test]
fn mixed_digits_map_one_for_one() {
    let src = "4\u{665}2\u{FF11}";
    let (out, a) = normalize(&Digits, src);
    assert_eq!(out, "4521");
    assert_eq!(covered(src, &a, 0, 1), "4");
    assert_eq!(covered(src, &a, 1, 2), "\u{665}");
    assert_eq!(covered(src, &a, 2, 3), "2");
    assert_eq!(covered(src, &a, 3, 4), "\u{FF11}");
    // The ASCII '2' is byte 3 of the source and byte 2 of the output.
    assert_eq!(forward(&a, 3, 4), (2, 3));
}

#[test]
fn neighbouring_digits_keep_their_own_edits() {
    // As in OpenNLP, "2" in "123" maps back to U+0662 alone: each digit is
    // its own edit.
    let src = "\u{661}\u{662}\u{663}";
    let (_, a) = normalize(&Digits, src);
    assert_eq!(a.edits().len(), 3);
    assert_eq!(covered(src, &a, 1, 2), "\u{662}");
}

#[test]
fn every_decimal_digit_folds_to_its_value() {
    // The Nd rows of UnicodeData.txt and their decimal values (field 6).
    let path: std::path::PathBuf =
        [env!("CARGO_MANIFEST_DIR"), "..", "unicode", "ucd", "UnicodeData.txt"].iter().collect();
    let data = std::fs::read_to_string(&path).unwrap();
    let mut nd = std::collections::BTreeMap::new();
    for line in data.lines() {
        let f: Vec<&str> = line.split(';').collect();
        if f[2] == "Nd" {
            nd.insert(char::from_u32(u32::from_str_radix(f[0], 16).unwrap()).unwrap(), f[6].to_string());
        }
    }
    assert!(nd.len() > 700);
    for c in (0x80..=0x10FFFFu32).filter_map(char::from_u32) {
        let s = c.to_string();
        let out = apply(&Digits, &s);
        match nd.get(&c) {
            Some(value) => assert_eq!(out, value.as_str(), "{c:?}"),
            None => assert_eq!(out, s, "{c:?}"),
        }
        assert_eq!(turbo_unicode::java::digit(c).map(|v| v.to_string()), nd.get(&c).cloned(), "{c:?}");
    }
}

#[test]
fn generated_texts_match_the_reference() {
    let pool = [
        "0",
        "7",
        "a",
        " ",
        "\u{660}",
        "\u{669}",
        "\u{96F}",
        "\u{E55}",
        "\u{FF19}",
        "\u{1D7D7}",
        "\u{1FBF3}",
        "\u{B2}",
        "\u{1F600}",
        "\u{E9}",
    ];
    let mut rng = Rng::new(12);
    for _ in 0..3000 {
        let src = rng.text(&pool, 20);
        let (out, a) = normalize(&Digits, &src);
        let expected: String =
            src.chars().map(|c| digit_fold(c).map_or_else(|| c.to_string(), str::to_string)).collect();
        assert_eq!(out, expected, "{src:?}");
        check_alignment(&src, &out, &a, per_char_edit(digit_fold));
    }
}
