use super::*;
use NumberNotation::{LatinEu, LatinUs};

/// The value as `BigDecimal.toPlainString` writes it, and the end.
fn scan(text: &str, apply_scale: bool, notation: NumberNotation) -> Option<(String, usize)> {
    let mut digits = String::new();
    let r = parse(text, 0, apply_scale, notation, &mut digits)?;
    let value = turbo_analysis_layers::DecimalRef { negative: false, digits: &digits, scale: r.scale.into() };
    Some((value.to_plain_string(), r.end))
}

fn scan_parity(text: &str, apply_scale: bool, notation: NumberNotation) -> Option<(String, usize)> {
    let mut digits = String::new();
    let r = parse_with_parity(text, 0, apply_scale, notation, true, &mut digits)?;
    let value = turbo_analysis_layers::DecimalRef { negative: false, digits: &digits, scale: r.scale.into() };
    Some((value.to_plain_string(), r.end))
}

#[test]
fn digits_without_leading_zeros() {
    let mut digits = String::from("left over");
    let r = parse("007", 0, false, LatinUs, &mut digits).unwrap();
    assert_eq!((digits.as_str(), r.scale, r.end), ("7", 0, 3));
    let r = parse("0.050", 0, false, LatinUs, &mut digits).unwrap();
    assert_eq!((digits.as_str(), r.scale), ("50", 3));
    let r = parse("0.00", 0, false, LatinUs, &mut digits).unwrap();
    assert_eq!((digits.as_str(), r.scale), ("0", 2));
    let r = parse("0k", 0, true, LatinUs, &mut digits).unwrap();
    assert_eq!((digits.as_str(), r.scale, r.multiplier), ("0", 0, Multiplier::Thousand));
}

#[test]
fn a_scan_starts_at_a_digit() {
    let mut digits = String::new();
    for text in ["", "x", "-5", ".5", " 5"] {
        assert_eq!(parse(text, 0, true, LatinUs, &mut digits), None, "{text:?}");
    }
    // Past the end of the text.
    assert_eq!(parse("5", 1, true, LatinUs, &mut digits), None);
    assert_eq!(parse("5", 7, true, LatinUs, &mut digits), None);
}

#[test]
fn the_scale_multiplies_and_keeps_the_fraction() {
    assert_eq!(scan("1.5k", true, LatinUs), Some(("1500.0".into(), 4)));
    assert_eq!(scan("1,5k", true, LatinEu), Some(("1500.0".into(), 4)));
    assert_eq!(scan("12k", true, LatinUs), Some(("12000".into(), 3)));
    assert_eq!(scan("12K", true, LatinUs), Some(("12000".into(), 3)));
    assert_eq!(scan("2m", true, LatinUs), Some(("2000000".into(), 2)));
    assert_eq!(scan("5b", true, LatinUs), Some(("5000000000".into(), 2)));
    assert_eq!(scan("5bn", true, LatinUs), Some(("5000000000".into(), 3)));
    assert_eq!(scan("5BN", true, LatinUs), Some(("5000000000".into(), 3)));
    assert_eq!(scan("3 million", true, LatinUs), Some(("3000000".into(), 9)));
    assert_eq!(scan("3 TRILLION", true, LatinUs), Some(("3000000000000".into(), 10)));
    assert_eq!(scan("0.25 Thousand", true, LatinUs), Some(("250.00".into(), 13)));
    // Without scaling the number ends with its digits.
    assert_eq!(scan("12k", false, LatinUs), Some(("12".into(), 2)));
    assert_eq!(scan("3 million", false, LatinUs), Some(("3".into(), 1)));
}

#[test]
fn markers_are_recorded_with_the_numeral_end() {
    let mut digits = String::new();
    let r = parse("1,234 billion.", 0, true, LatinUs, &mut digits).unwrap();
    assert_eq!(
        r,
        ScanResult { end: 13, numeral_end: 5, scale: 0, multiplier: Multiplier::Billion, marker: ScaleMarker::Word }
    );
    let r = parse("7bn)", 0, true, LatinUs, &mut digits).unwrap();
    assert_eq!((r.end, r.numeral_end, r.marker), (3, 1, ScaleMarker::Suffix));
    let r = parse("7 apples", 0, true, LatinUs, &mut digits).unwrap();
    assert_eq!((r.end, r.numeral_end, r.marker, r.multiplier), (1, 1, ScaleMarker::None, Multiplier::One));
}

#[test]
fn a_letter_after_the_digits_must_be_a_scale_suffix() {
    assert_eq!(scan("5x", true, LatinUs), None);
    assert_eq!(scan("5kg", true, LatinUs), None);
    assert_eq!(scan("5bnx", true, LatinUs), None);
    assert_eq!(scan("5k2", true, LatinUs), None);
    assert_eq!(scan("5m_", true, LatinUs), None);
    assert_eq!(scan("5é", true, LatinUs), None);
    // A digit that is not a letter ends the number before it.
    assert_eq!(scan("5\u{0661}m", true, LatinUs), Some(("5".into(), 1)));
    // A combining mark after a suffix is no boundary.
    assert_eq!(scan("5k\u{0301}", true, LatinUs), None);
}

#[test]
fn scale_words_take_one_whitespace_character() {
    assert_eq!(scan("3\u{00A0}million", true, LatinUs), Some(("3000000".into(), 10)));
    assert_eq!(scan("3\u{3000}million", true, LatinUs), Some(("3000000".into(), 11)));
    assert_eq!(scan("3\tthousand", true, LatinUs), Some(("3000".into(), 10)));
    assert_eq!(scan("3  million", true, LatinUs), Some(("3".into(), 1)));
    assert_eq!(scan("3 millions", true, LatinUs), Some(("3".into(), 1)));
    assert_eq!(scan("3 millionaire", true, LatinUs), Some(("3".into(), 1)));
    assert_eq!(scan("3 mil", true, LatinUs), Some(("3".into(), 1)));
    assert_eq!(scan("3 ", true, LatinUs), Some(("3".into(), 1)));
    // U+0085 is White_Space; U+200B is not.
    assert_eq!(scan("3\u{85}million", true, LatinUs), Some(("3000000".into(), 10)));
    assert_eq!(scan("3\u{200B}million", true, LatinUs), Some(("3".into(), 1)));
}

#[test]
fn scale_words_lowercase_by_code_point() {
    // U+0130's simple lowercase mapping is `i`, so Character.toLowerCase
    // turns the word into `million`.
    assert_eq!(scan("3 m\u{0130}LL\u{0130}ON", true, LatinUs), Some(("3000000".into(), 11)));
    // The Kelvin sign lowercases to `k`, which no scale word holds.
    assert_eq!(scan("3 \u{212A}", true, LatinUs), Some(("3".into(), 1)));
}

#[test]
fn grouping_is_strict() {
    assert_eq!(scan("1,234,567.89", false, LatinUs), Some(("1234567.89".into(), 12)));
    assert_eq!(scan("1.234.567,89", false, LatinEu), Some(("1234567.89".into(), 12)));
    assert_eq!(scan("1.234,5", false, LatinEu), Some(("1234.5".into(), 7)));
    // Read in the other notation, 1.234 is a fraction and the comma then
    // goes on with a digit.
    assert_eq!(scan("1.234,5", false, LatinUs), None);
    assert_eq!(scan("1,00,000", false, LatinUs), Some(("100000".into(), 8)));
    assert_eq!(scan_parity("1,00,000", false, LatinUs), None);
    assert_eq!(scan("1234,567", false, LatinUs), None);
    assert_eq!(scan("1,2345", false, LatinUs), None);
    assert_eq!(scan("1,23", false, LatinUs), None);
    assert_eq!(scan("1,", false, LatinUs), Some(("1".into(), 1)));
    assert_eq!(scan("1,23x", false, LatinUs), None);
    assert_eq!(scan("1,x", false, LatinUs), Some(("1".into(), 1)));
    // A group that ends at the end of the text, and one cut short.
    assert_eq!(scan("1,234", false, LatinUs), Some(("1234".into(), 5)));
    assert_eq!(scan("1,23", true, LatinUs), None);
    assert_eq!(scan("12,3", true, LatinUs), None);
    assert_eq!(scan("1.", false, LatinUs), Some(("1".into(), 1)));
}

#[test]
fn other_separators_before_a_digit_reject_the_scan() {
    for sep in ['\u{A0}', '\u{2009}', '\u{202F}', '\'', '\u{2019}', '\u{66B}', '\u{66C}'] {
        for notation in [LatinUs, LatinEu] {
            let text = format!("1{sep}234");
            let expected = if sep == '\u{66B}' { "1.234" } else { "1234" };
            assert_eq!(scan(&text, false, notation), Some((expected.into(), text.len())), "{text:?}");
            assert_eq!(scan_parity(&text, false, notation), None, "{text:?}");
            assert!(continues_number(&text, 1 + sep.len_utf8(), notation), "{text:?}");
            // Not followed by a digit, the separator ends the number.
            let text = format!("1{sep}x");
            assert_eq!(scan(&text, false, notation), Some(("1".into(), 1)), "{text:?}");
        }
    }
}

#[test]
fn exponents_reject_the_scan() {
    for (text, expected) in [("1e3", "1000"), ("1E3", "1000"), ("1e-3", "0.001"), ("1e+3", "1000"), ("1.5e3", "1500")] {
        assert_eq!(scan(text, true, LatinUs), Some((expected.into(), text.len())), "{text:?}");
        assert_eq!(scan_parity(text, true, LatinUs), None, "{text:?}");
    }
    for text in ["1.e3", "1,e3"] {
        assert_eq!(scan(text, true, LatinUs), None, "{text:?}");
    }
    assert_eq!(scan("1,e3", true, LatinEu), None);
    assert_eq!(scan("1e", true, LatinUs), None, "a letter that is no suffix");
    assert_eq!(scan("1e", false, LatinUs), Some(("1".into(), 1)));
    assert_eq!(scan("1e-", false, LatinUs), Some(("1".into(), 1)));
    assert_eq!(scan("1e-x", false, LatinUs), Some(("1".into(), 1)));
}

#[test]
fn continuation_after_separators_and_exponents() {
    for notation in [LatinUs, LatinEu] {
        assert!(continues_number("1,00,000", 5, notation));
        assert!(continues_number("1.00.000", 5, notation));
        assert!(continues_number("1.2.3", 4, notation));
        assert!(!continues_number("1 000", 2, notation));
        assert!(!continues_number(",000", 1, notation));
        assert!(!continues_number("id,000", 3, notation));
        assert!(continues_number("1e-3", 3, notation));
        assert!(continues_number("1e3", 2, notation));
        assert!(continues_number("1.e3", 3, notation));
        assert!(continues_number("1,e3", 3, notation));
        assert!(!continues_number("e3", 1, notation));
        assert!(!continues_number("-3", 1, notation));
        assert!(!continues_number("3", 0, notation));
        assert!(!continues_number("a.e3", 3, notation));
    }
}

#[test]
fn bounded_words() {
    // NumericBoundaryTest.testBoundedWord, with UTF-16 offsets turned
    // into bytes: U+1D48E is four bytes and two units.
    assert_eq!(word_end("cat!", 0, 3), Some(3));
    assert_eq!(word_end("cat3", 0, 3), None);
    assert_eq!(word_end("catastrophe", 0, 3), None);
    assert_eq!(word_end("xcat!", 1, 3), Some(4));
    assert_eq!(word_end("\u{1D48E}!", 0, 2), Some(4));
    assert_eq!(word_end("\u{1D48E}!", 0, 1), None);
    assert_eq!(word_end("\u{1D48E}x!", 0, 3), Some(5));
    assert_eq!(word_end("\u{1D48E}x!", 0, 2), None);
    assert_eq!(word_end("cat", 3, 3), None);
    assert_eq!(word_end("cat", 9, 3), None);
    assert_eq!(word_end("\u{3B1}\u{3B2}", 0, 2), Some(4));
}

#[test]
fn boundaries_skip_combining_marks() {
    assert!(boundary_before("", 0));
    assert!(boundary_before(" 5", 1));
    assert!(!boundary_before("a5", 1));
    assert!(!boundary_before("_5", 1));
    assert!(!boundary_before("\u{00B2}5", 2), "an Other_Number before");
    assert!(!boundary_before("\u{2160}5", 3), "a Letter_Number before");
    // The base character before the marks decides.
    assert!(boundary_before(" \u{0301}\u{0302}5", 5));
    assert!(!boundary_before("e\u{0301}5", 3));
    // Marks alone back to the start count as the start.
    assert!(boundary_before("\u{0301}5", 2));
    assert!(boundary_after("5", 1));
    assert!(boundary_after("5 ", 1));
    assert!(!boundary_after("5\u{0301}", 1));
    assert!(!boundary_after("5\u{1D7D9}", 1), "a mathematical digit");
    assert!(boundary_after("5\u{1F4E6}", 1), "an emoji");
}

#[test]
fn sign_boundaries() {
    assert!(sign_boundary_before("-5", 0));
    assert!(sign_boundary_before(" -5", 1));
    assert!(sign_boundary_before("(-5", 1));
    assert!(!sign_boundary_before("a-5", 1));
    assert!(!sign_boundary_before("5-5", 1));
    assert!(!sign_boundary_before("5%-10%", 2));
    assert!(!sign_boundary_before("50\u{20AC}-60", 5));
    assert!(!sign_boundary_before("$-5", 1));
    assert!(!sign_boundary_before("%\u{0301}-5", 3));
}

#[test]
fn whitespace_past_the_end() {
    assert!(whitespace_at("5", 1));
    assert!(whitespace_at("5", 9));
    assert!(whitespace_at("5\u{2009}", 1));
    assert!(!whitespace_at("5x", 1));
}
