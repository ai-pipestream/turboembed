//! The number scanning expectations of the OpenNLP numeric add-on's tests
//! (`NumberNotationTest`, `NumericBoundaryTest`, `NumericFalsePositiveTest`
//! and `NumericTokenIntegrityTest`), ported. Where a Java test checks a
//! money, quantity or date extractor, the expectation here is what the
//! number scan alone finds in the same text, as the Java `NumberScan` and
//! the candidate loop of `CursorQuantityExtractor` find it.

mod common;

use common::{Rng, eu, found, pairs, us};
use turbo_numeric::scan::{base_char_before, boundary_before, continues_number, sign_boundary_before, word_end};
use turbo_numeric::{DecimalRef, Multiplier, NumberNotation, parse};

use NumberNotation::{LatinEu, LatinUs};

const NOTATIONS: [NumberNotation; 2] = [LatinUs, LatinEu];

/// The value `parse` reads at the start of `text`, and its end.
fn value(text: &str, apply_scale: bool, notation: NumberNotation) -> Option<(String, usize)> {
    let mut digits = String::new();
    let r = parse(text, 0, apply_scale, notation, &mut digits)?;
    Some((DecimalRef { negative: false, digits: &digits, scale: r.scale }.to_plain_string(), r.end))
}

// NumberNotationTest

#[test]
fn separators_never_collide() {
    for n in NOTATIONS {
        assert_ne!(n.group_separator(), n.decimal_separator());
    }
}

#[test]
fn the_same_text_means_one_number_per_notation() {
    for (text, in_us, in_eu) in [
        ("1.500", "1.500", "1500"),
        ("1,500", "1500", "1.500"),
        ("12.345", "12.345", "12345"),
        ("12,345", "12345", "12.345"),
    ] {
        assert_eq!(value(text, false, LatinUs).unwrap().0, in_us, "{text}");
        assert_eq!(value(text, false, LatinEu).unwrap().0, in_eu, "{text}");
    }
}

#[test]
fn text_with_both_separators_scans_whole_in_one_notation_only() {
    for (text, written, expected) in [
        ("1.234,56", LatinEu, "1234.56"),
        ("1.234.567,89", LatinEu, "1234567.89"),
        ("1,234.56", LatinUs, "1234.56"),
        ("1,234,567.89", LatinUs, "1234567.89"),
    ] {
        assert_eq!(value(text, false, written), Some((expected.to_string(), text.len())), "{text}");
        let other = if written == LatinEu { LatinUs } else { LatinEu };
        assert!(value(text, false, other).is_none_or(|(_, end)| end < text.len()), "{text}");
    }
}

#[test]
fn grouping_neither_notation_knows_is_never_read_whole() {
    for text in ["1,00,000", "1.00.000", "12,3456,789", "12.3456.789"] {
        for n in NOTATIONS {
            assert!(value(text, false, n).is_none_or(|(_, end)| end < text.len()), "{text} in {n:?}");
            assert_eq!(found(text, n, true), [], "{text} in {n:?}");
        }
    }
}

#[test]
fn a_repeated_decimal_separator_rejects_the_whole_scan() {
    assert_eq!(value("1.2.3", false, LatinUs), None);
    assert_eq!(value("1,2,3", false, LatinEu), None);
}

#[test]
fn an_over_long_group_is_a_fraction_in_the_other_notation() {
    assert_eq!(value("1,2345", false, LatinEu), Some(("1.2345".into(), 6)));
    assert_eq!(value("1,2345", false, LatinUs), None);
    assert_eq!(value("1.2345", false, LatinUs), Some(("1.2345".into(), 6)));
    assert_eq!(value("1.2345", false, LatinEu), None);
}

#[test]
fn the_tail_of_an_earlier_number_is_recognized_in_each_notation() {
    for n in NOTATIONS {
        assert!(continues_number("1,00,000", 5, n));
        assert!(continues_number("1.00.000", 5, n));
        assert!(continues_number("1.2.3", 4, n));
        assert!(!continues_number("1 000", 2, n));
        assert!(!continues_number(",000", 1, n));
        assert!(!continues_number("id,000", 3, n));
    }
}

// NumericBoundaryTest

const UNSUPPORTED_SEPARATORS: [char; 7] = ['\u{A0}', '\u{2009}', '\u{202F}', '\'', '\u{2019}', '\u{66B}', '\u{66C}'];

#[test]
fn an_unsupported_separator_rejects_the_scan() {
    for sep in UNSUPPORTED_SEPARATORS {
        for n in NOTATIONS {
            let text = format!("1{sep}234");
            assert_eq!(value(&text, false, n), None, "{text:?}");
            assert!(continues_number(&text, 1 + sep.len_utf8(), n), "{text:?}");
            // Neither the prefix nor the tail is a number of its own.
            assert_eq!(found(&format!("{text} kg"), n, true), [], "{text:?}");
            assert_eq!(found(&format!("{text} kg"), n, false), [], "{text:?}");
        }
    }
}

const LETTERS_AND_DIGITS: [&str; 4] = ["a", "2", "\u{10400}", "\u{1D7D9}"];

#[test]
fn scale_markers_end_at_a_unicode_boundary() {
    for suffix in LETTERS_AND_DIGITS {
        assert_eq!(value(&format!("5m{suffix}"), true, LatinUs), None, "{suffix:?}");
        assert_eq!(value(&format!("5 million{suffix}"), true, LatinUs), Some(("5".into(), 1)), "{suffix:?}");
    }
}

#[test]
fn bounded_words() {
    // Byte offsets in place of the Java's UTF-16 ones: U+1D48E is four
    // bytes and two units.
    let cases: [(&str, usize, usize, Option<usize>); 9] = [
        ("cat!", 0, 3, Some(3)),
        ("cat3", 0, 3, None),
        ("catastrophe", 0, 3, None),
        ("xcat!", 1, 3, Some(4)),
        ("\u{1D48E}!", 0, 2, Some(4)),
        ("\u{1D48E}!", 0, 1, None),
        ("\u{1D48E}x!", 0, 3, Some(5)),
        ("\u{1D48E}x!", 0, 2, None),
        ("cat", 3, 3, None),
    ];
    for (text, start, limit, end) in cases {
        assert_eq!(word_end(text, start, limit), end, "{text:?} {start} {limit}");
    }
}

/// `DecimalFormat("#,##0.0000")` in the notation's symbols.
fn format_grouped(units: u64, notation: NumberNotation) -> String {
    let (int, frac) = (units / 10_000, units % 10_000);
    let digits = int.to_string();
    let mut out = String::new();
    for (i, d) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(notation.group_separator());
        }
        out.push(d);
    }
    out.push(notation.decimal_separator());
    out.push_str(&format!("{frac:04}"));
    out
}

#[test]
fn generated_decimals_keep_their_exact_value_and_scale() {
    for n in NOTATIONS {
        let mut rng = Rng(1909);
        for _ in 0..1000 {
            let units = rng.next() % 1_000_000_000_000;
            let text = format_grouped(units, n);
            let expected = format!("{}.{:04}", units / 10_000, units % 10_000);
            assert_eq!(value(&text, false, n), Some((expected.clone(), text.len())), "{text}");
            let mut digits = String::new();
            let scaled = format!("{text}M");
            let r = parse(&scaled, 0, true, n, &mut digits).unwrap();
            assert_eq!(r.end, text.len() + 1);
            assert_eq!(r.multiplier, Multiplier::Million);
            // The value times a million keeps the four fraction digits.
            let expected = if units == 0 { "0".to_string() } else { format!("{units}000000") };
            assert_eq!((digits.as_str(), r.scale), (expected.as_str(), 4), "{scaled}");
        }
    }
}

// NumericFalsePositiveTest: what the number scan finds in the near misses.

#[test]
fn near_misses_with_no_number() {
    for text in [
        "order 4,50,000 shipped",
        "the ticket is 1,00,000",
        "invoice 2.5.1 attached",
        "upgraded to 2.5.1",
        "release 10.15.7 notes",
        "1.2.3.4 is the address",
        "$1.2.3 is malformed",
        "1,2,3 euros is malformed",
        "1,00,000 USD",
        "capacity 1,00,000 kg",
    ] {
        assert_eq!(us(text), [], "{text}");
        assert_eq!(eu(text), [], "{text}");
    }
}

#[test]
fn near_misses_that_hold_plain_numbers() {
    let both = |text: &str, expected: &[(&str, &str)]| {
        assert_eq!(us(text), pairs(expected), "{text}");
        assert_eq!(eu(text), pairs(expected), "{text}");
    };
    both("order 12345678 confirmed", &[("12345678", "12345678")]);
    both("PIN 1234 was reset", &[("1234", "1234")]);
    both("HK$50 in Kowloon", &[("50", "50")]);
    both("the 100 things", &[("100", "100")]);
    both("chapter 7 of 12", &[("7", "7"), ("12", "12")]);
    both("5 apples a day", &[("5", "5")]);
    both("50 gb of it", &[("50", "50")]);
    both("2026-13-45", &[("2026", "2026"), ("13", "13"), ("45", "45")]);
    both("February 30, 2026", &[("30", "30"), ("2026", "2026")]);
    both("Q5 2024", &[("2024", "2024")]);
    both("07/14/2026", &[("07", "7"), ("14", "14"), ("2026", "2026")]);
    // A letter after the digits that is no scale suffix rejects the number.
    both("50kmh top speed", &[]);
    assert_eq!(found("50kmh top speed", LatinUs, false), pairs(&[("50", "50")]));
}

#[test]
fn near_misses_that_depend_on_the_notation() {
    assert_eq!(us("call 555,1234 now"), []);
    assert_eq!(eu("call 555,1234 now"), pairs(&[("555,1234", "555.1234")]));
    assert_eq!(us("extension 2,204"), pairs(&[("2,204", "2204")]));
    assert_eq!(eu("extension 2,204"), pairs(&[("2,204", "2.204")]));
}

#[test]
fn an_amount_in_the_other_notation_yields_nothing() {
    for (text, read_in) in [
        ("the invoice says 1.234,56 EUR", LatinUs),
        ("the invoice says 12,34 EUR", LatinUs),
        ("\u{20AC}1.234,56 total", LatinUs),
        ("the invoice says 1,234.56 USD", LatinEu),
        ("$1,234.56 total", LatinEu),
        ("capacity 1.234,5 kg", LatinUs),
        ("capacity 12,34 kg", LatinUs),
        ("capacity 1,234.5 kg", LatinEu),
    ] {
        assert_eq!(found(text, read_in, true), [], "{text}");
    }
    assert_eq!(eu("the invoice says 1.234,56 EUR"), pairs(&[("1.234,56", "1234.56")]));
    assert_eq!(us("capacity 1,234.5 kg"), pairs(&[("1,234.5", "1234.5")]));
}

#[test]
fn the_control_text_is_still_matched() {
    let control = "Berlin, 14 July 2026. The buyer paid $2,400,000 for 1,250 GB.";
    assert_eq!(us(control), pairs(&[("14", "14"), ("2026", "2026"), ("2,400,000", "2400000"), ("1,250", "1250")]));
    // 2,400,000 is no number in the European notation; 1,250 is 1.25.
    assert_eq!(eu(control), pairs(&[("14", "14"), ("2026", "2026"), ("1,250", "1.250")]));
    assert_eq!(eu("Berlin. Der Preis war 2.400.000 EUR."), pairs(&[("2.400.000", "2400000")]));
    assert_eq!(eu("Kapazitat 1.250,5 GB gemessen."), pairs(&[("1.250,5", "1250.5")]));
}

// NumericTokenIntegrityTest

#[test]
fn scientific_notation_is_not_split_into_numbers() {
    let all = "1e3 1e+3 1e-3 1E-03 2.5e-3 2,500e+3 2,5e-3 1e-3e-4 1.e-3";
    for n in NOTATIONS {
        for apply_scale in [true, false] {
            assert_eq!(found(all, n, apply_scale), [], "{n:?}");
            for number in all.split(' ') {
                assert_eq!(found(&format!("{number}kg"), n, apply_scale), [], "{number} in {n:?}");
                assert_eq!(found(&format!("{number} USD"), n, apply_scale), [], "{number} in {n:?}");
                assert_eq!(found(&format!("${number}"), n, apply_scale), [], "{number} in {n:?}");
            }
        }
    }
}

const MARKS: [&str; 5] = ["\u{0301}", "\u{0903}", "\u{20DD}", "\u{1D165}", "\u{FE0F}"];

#[test]
fn combining_marks_stay_with_their_neighbor() {
    for mark in MARKS {
        assert_eq!(us(&format!("a{mark}12kg")), [], "{mark:?}");
        assert_eq!(us(&format!("a{mark}12")), [], "{mark:?}");
        assert_eq!(us(&format!("5kg{mark}")), [], "{mark:?}");
        assert_eq!(us(&format!("5k{mark}")), [], "{mark:?}");
    }
}

#[test]
fn numeric_continuations_cannot_be_dropped_from_a_scale_suffix() {
    for suffix in ["\u{B2}", "\u{2163}", "\u{2460}", "_item"] {
        assert_eq!(us(&format!("5m{suffix}")), [], "{suffix:?}");
        assert_eq!(found(&format!("5m{suffix}"), LatinUs, false), pairs(&[("5", "5")]), "{suffix:?}");
    }
}

#[test]
fn money_and_measurement_texts() {
    let text = "Paid $3m for 3m of pipe and 5kg of fittings; USD 2m remained.";
    assert_eq!(us(text), pairs(&[("3m", "3000000"), ("3m", "3000000"), ("2m", "2000000")]));
    assert_eq!(found(text, LatinUs, false), pairs(&[("3", "3"), ("3", "3"), ("5", "5"), ("2", "2")]));
    let text = "$3m; USD 3m; 3m USD; 3m dollars; -\u{A3}3m; EUR 1,000m; ($3m); 3m\u{20AC}";
    let mut expected = vec![("3m", "3000000"); 5];
    expected.extend([("1,000m", "1000000000"), ("3m", "3000000"), ("3m", "3000000")]);
    assert_eq!(us(text), pairs(&expected));
    expected[5] = ("1,000m", "1000000.000");
    assert_eq!(eu(text), pairs(&expected));
    assert_eq!(us("$5kg; USD 8kg; 12kg USD; HK$3m"), pairs(&[("3m", "3000000")]));
}

#[test]
fn ranges_and_signs() {
    assert_eq!(us("3kg-5kg and -2kg"), []);
    assert_eq!(found("3kg-5kg and -2kg", LatinUs, false), pairs(&[("3", "3"), ("5", "5"), ("-2", "-2")]));
    assert_eq!(
        us("5%-10% and 50\u{20AC}-60\u{20AC} and a-5 and -5 and 5-5"),
        pairs(&[
            ("5", "5"),
            ("10", "10"),
            ("50", "50"),
            ("60", "60"),
            ("5", "5"),
            ("-5", "-5"),
            ("5", "5"),
            ("5", "5")
        ])
    );
}

#[test]
fn decorated_symbols_are_boundaries() {
    for prefix in ["\u{2615}\u{FE0F}", "\u{1F4E6}\u{FE0F}", "(\u{0301}"] {
        assert_eq!(us(&format!("{prefix}5")), pairs(&[("5", "5")]), "{prefix:?}");
        assert_eq!(us(&format!("{prefix}-5")), pairs(&[("-5", "-5")]), "{prefix:?}");
    }
}

#[test]
fn runs_of_combining_marks_use_the_base_before_them() {
    let cases: [(&str, bool, bool); 10] = [
        ("", true, true),
        ("a", false, false),
        ("9", false, false),
        ("_", false, false),
        ("\u{10400}", false, false),
        ("\u{B2}", false, false),
        ("(", true, true),
        ("\u{1F4E6}", true, true),
        ("$", true, false),
        ("%", true, false),
    ];
    for (base, boundary, signed) in cases {
        let prefix = format!("{base}\u{0301}\u{0903}\u{20DD}\u{1D165}\u{FE0F}");
        assert_eq!(base_char_before(&prefix, prefix.len()), base.chars().next(), "{base:?}");
        assert_eq!(boundary_before(&prefix, prefix.len()), boundary, "{base:?}");
        assert_eq!(sign_boundary_before(&prefix, prefix.len()), signed, "{base:?}");
    }
}
