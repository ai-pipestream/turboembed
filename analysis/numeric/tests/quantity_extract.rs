//! The quantity extractor: the cases of OpenNLP's
//! `CursorQuantityExtractorTest`, then edge cases of the port.
//!
//! Java spans count UTF-16 units; here they are UTF-8 bytes, so a span in
//! a text with a non-ASCII character is written out in bytes, with the
//! Java span beside it. Cases that pass null to the Java have no
//! counterpart: the Rust types cannot hold it. The checks on a mention's
//! own fields (`testQuantityValidation`, `testQuantityRejectsUnicodeBlankUnit`)
//! are `Quantities::check`, tested in turbo-analysis-layers.

mod money_common;

use money_common::{dec, same};
use turbo_analysis::{Analyzer, Annotator, Document, Ix, Layer, Span, ViewId};
use turbo_numeric::NumberNotation::{LatinEu, LatinUs};
use turbo_numeric::money::Decimal;
use turbo_numeric::{DEFAULT_UNITS, PERCENT, Quantities, QuantityExtractor, UnitError};

/// One extracted mention: its span, its value and its unit.
#[derive(Debug)]
struct Found {
    span: Span,
    value: Decimal,
    unit: String,
}

/// Every mention `x` finds in `text`, after checking the layer.
fn extract(x: &QuantityExtractor, text: &str) -> Vec<Found> {
    let mut q = Quantities::new(ViewId::SOURCE);
    x.scan(text, &mut q);
    assert_eq!(q.check(), Ok(()), "{text:?}");
    q.spans
        .iter()
        .map(|(i, span)| {
            let v = q.value(i);
            let value = Decimal { negative: v.negative, digits: v.digits.into(), scale: v.scale as i32 };
            Found { span, value, unit: q.unit(i).into() }
        })
        .collect()
}

/// The single mention `x` finds in `text`.
fn single_with(x: &QuantityExtractor, text: &str) -> Found {
    let mut all = extract(x, text);
    assert_eq!(all.len(), 1, "expected one mention in {text:?}: {all:?}");
    all.pop().unwrap()
}

/// The single mention the default extractor finds in `text`.
fn single(text: &str) -> Found {
    single_with(&QuantityExtractor::default(), text)
}

fn none_with(x: &QuantityExtractor, text: &str) {
    let found = extract(x, text);
    assert!(found.is_empty(), "{text:?}: {found:?}");
}

fn none(text: &str) {
    none_with(&QuantityExtractor::default(), text);
}

/// The span of the whole of `text`, in bytes.
fn whole(text: &str) -> Span {
    Span::new(0, text.len() as u32)
}

/// Asserts the mention's value, at any scale, and unit.
#[track_caller]
fn assert_mention(m: &Found, value: &str, unit: &str) {
    assert!(same(&dec(value), &m.value), "expected {value}, got {}", m.value);
    assert_eq!(m.unit, unit);
}

fn eu() -> QuantityExtractor {
    QuantityExtractor::new(LatinEu)
}

#[test]
fn recognized_shapes_cover_the_full_mention() {
    let cases = [
        ("50%", "50", "%"),
        ("3.5 %", "3.5", "%"),
        ("50 percent", "50", "%"),
        ("-2%", "-2", "%"),
        ("0%", "0", "%"),
        ("2.5km", "2.5", "km"),
        ("80 kg", "80", "kg"),
        ("1,250 GB", "1250", "GB"),
        ("3 GHz", "3", "GHz"),
        ("26.2 mi", "26.2", "mi"),
        ("500 ms", "500", "ms"),
    ];
    for (text, value, unit) in cases {
        let m = single(text);
        assert_eq!(m.span, whole(text), "{text:?}");
        assert_mention(&m, value, unit);
    }
}

#[test]
fn mention_inside_sentence_has_exact_span() {
    let m = single("the route is 26.2 mi long");
    assert_eq!(m.span, Span::new(13, 20));
    assert_eq!(m.unit, "mi");
}

#[test]
fn multiple_mentions() {
    let found = extract(&QuantityExtractor::default(), "lost 5 kg and 3% body fat");
    assert_eq!(found.len(), 2);
    assert_eq!(found[0].unit, "kg");
    assert_eq!(found[1].unit, "%");
}

/// The hyphen in a percentage range is not a minus: both amounts of
/// `5%-10%` are positive, though the character before the hyphen is a
/// percent sign rather than a digit.
#[test]
fn hyphenated_percent_range_yields_two_positive_mentions() {
    let found = extract(&QuantityExtractor::default(), "5%-10%");
    assert_eq!(found.len(), 2);
    assert_eq!(found[0].span, Span::new(0, 2));
    assert_mention(&found[0], "5", "%");
    assert_eq!(found[1].span, Span::new(3, 6));
    assert_mention(&found[1], "10", "%");
}

/// A quantity grouped in a convention the scan cannot read is rejected
/// whole, not cut short by a restart inside the group: `1,00,000 kg` is
/// not `0 kg` and `1,2345 km` not `2345 km`.
#[test]
fn grouping_the_scanner_cannot_parse_yields_no_mention() {
    for text in ["1,00,000 kg", "1,2345 km", "12,34 m", "1.234,5 kg"] {
        none(text);
    }
}

/// A comma not followed by a digit is punctuation after the mention.
#[test]
fn trailing_comma_without_digit_still_ends_the_match() {
    let m = single("5 kg, then");
    assert_eq!(m.span, Span::new(0, 4));
    assert_mention(&m, "5", "kg");
}

#[test]
fn rejected_shapes() {
    for text in [
        "the 100 things", // a bare number
        "$3 billion",     // money, not a unit quantity
        "100 USD",        // money, not a unit quantity
        "50kmh",          // not a known unit token
        "5 in the morning",
        "50 gb", // units are case-sensitive
        "50 km2",
        "version 2.5m1",
    ] {
        none(text);
    }
}

#[test]
fn custom_units() {
    let custom = QuantityExtractor::with_units(["bbl"], LatinUs).unwrap();
    assert_eq!(single_with(&custom, "500 bbl").unit, "bbl");
    none_with(&custom, "80 kg");
}

#[test]
fn custom_unit_validation() {
    let empty: [&str; 0] = [];
    assert_eq!(QuantityExtractor::with_units(empty, LatinUs), Err(UnitError::Empty));
    for bad in ["averyverylongunit", "m2", "kg/m", "\u{00B0}C", ""] {
        assert_eq!(
            QuantityExtractor::with_units(["kg", bad], LatinUs),
            Err(UnitError::NotAUnitToken(bad.into())),
            "{bad:?}"
        );
    }
}

#[test]
fn repeated_decimal_separator_yields_no_mention() {
    none_with(&QuantityExtractor::new(LatinUs), "1.2.3 kg");
    none_with(&eu(), "1,2,3 kg");
}

/// The European notation end to end: dots group digits and a comma marks
/// the fraction.
#[test]
fn european_notation_shapes_cover_the_full_mention() {
    let cases = [
        ("1.250 GB", "1250", "GB"),
        ("1.234,5 kg", "1234.5", "kg"),
        ("1.250,75 kWh", "1250.75", "kWh"),
        ("2,5km", "2.5", "km"),
        ("45,5 %", "45.5", "%"),
        ("-2,5 kg", "-2.5", "kg"),
    ];
    for (text, value, unit) in cases {
        let m = single_with(&eu(), text);
        assert_eq!(m.span, whole(text), "{text:?}");
        assert_mention(&m, value, unit);
    }
}

/// A number written in one notation gives no mention under the other,
/// nor does the Indian grouping, which neither notation knows.
#[test]
fn quantity_in_the_other_notation_yields_no_mention() {
    let cases = [
        ("1,234.56 kg", LatinEu),
        ("1,234,567.89 kg", LatinEu),
        ("1,00,000 kg", LatinEu),
        ("1.234,56 kg", LatinUs),
        ("1.234.567,89 kg", LatinUs),
        ("1.00.000 kg", LatinUs),
    ];
    for (text, notation) in cases {
        none_with(&QuantityExtractor::new(notation), text);
    }
}

/// Text valid in both notations is read as the notation says, a factor
/// of a thousand apart.
#[test]
fn quantity_valid_in_both_notations_means_what_the_notation_says() {
    assert_mention(&single_with(&eu(), "1.250 GB"), "1250", "GB");
    assert_mention(&single("1,250 GB"), "1250", "GB");
    assert_mention(&single_with(&eu(), "1,250 GB"), "1.250", "GB");
}

#[test]
fn notation_and_units_combine() {
    let custom = QuantityExtractor::with_units(["bbl"], LatinEu).unwrap();
    assert_mention(&single_with(&custom, "1.500 bbl"), "1500", "bbl");
    none_with(&custom, "80 kg");
}

/// The character before the number is read whole, so an emoji is an
/// ordinary boundary. U+1F4E6 is two UTF-16 units and four bytes: the
/// Java span 2..7 is 4..9 here.
#[test]
fn supplementary_plane_neighbor_is_a_boundary() {
    let m = single("\u{1F4E6}2.5km");
    assert_eq!(m.span, Span::new(4, 9));
    assert_mention(&m, "2.5", "km");
}

/// `testAnnotatorProvidesTheQuantityLayer`: the layer in a pipeline.
#[test]
fn annotator_provides_the_quantity_layer() {
    let analyzer = Analyzer::builder().add(QuantityExtractor::default()).build().unwrap();
    let mut docs = vec![Document::new("battery lasts 12 hr at 45% load"), Document::new("")];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let q = docs[0].get::<Quantities>().unwrap();
    assert_eq!(q.len(), 2);
    assert_eq!(q.unit(Ix::new(0)), "hr");
    assert_eq!(&docs[0].source()[q.spans.get(Ix::new(1)).range()], "45%");
    assert_eq!(docs[1].get::<Quantities>().unwrap().len(), 0);
}

/// The one gap between the number and its unit may be any `White_Space`
/// character, and gives the same mention as an ASCII space.
#[test]
fn unicode_whitespace_gap_matches_like_a_space() {
    for gap in ["\u{00A0}", "\u{202F}", "\u{2009}", "\u{3000}", "\t"] {
        for shape in ["3.5 %", "50 percent", "80 kg", "1,250 GB"] {
            let expected = single(shape);
            let text = shape.replace(' ', gap);
            let m = single(&text);
            // The gap may be longer than a space in bytes.
            let extra = (gap.len() - 1) as u32;
            assert_eq!(m.span, Span::new(expected.span.start, expected.span.end + extra), "{text:?}");
            assert!(same(&expected.value, &m.value), "{text:?}");
            assert_eq!(expected.unit, m.unit, "{text:?}");
        }
    }
}

// Edge cases of the port.

#[test]
fn the_default_units_are_the_javas() {
    let x = QuantityExtractor::default();
    assert_eq!(x.notation(), LatinUs);
    let units: Vec<&str> = x.units().collect();
    assert_eq!(units.len(), 37);
    assert!(units.windows(2).all(|w| w[0] < w[1]));
    for unit in DEFAULT_UNITS {
        let text = format!("7 {unit}");
        assert_mention(&single(&text), "7", unit);
        assert_mention(&single(&format!("7{unit}")), "7", unit);
    }
}

#[test]
fn empty_and_unitless_texts() {
    for text in ["", " ", "-", "%", "kg", "5", "5 ", "-5", "\u{1F4E6}", "percent"] {
        none(text);
    }
}

#[test]
fn the_text_ends_inside_a_shape() {
    assert_eq!(single("5%").span, whole("5%"));
    assert_eq!(single("ran 5 km").span, Span::new(4, 8));
    none("ran 5 ");
    none("ran 5\u{00A0}");
    none("ran -");
}

#[test]
fn the_percent_word_takes_any_case_and_units_do_not() {
    for text in ["50 PERCENT", "50 Percent", "50 pErCeNt", "50percent"] {
        let m = single(text);
        assert_eq!(m.span, whole(text), "{text:?}");
        assert_mention(&m, "50", PERCENT);
    }
    for text in ["50 percents", "50 per cent", "50 pct", "2.5 KM", "1 Kg", "5 hrs"] {
        none(text);
    }
    // A letter that is not ASCII never matches: the Kelvin sign
    // lowercases to `k`, the dotless i uppercases to `I`.
    none("50 \u{212A}ercent");
    none("50 perc\u{0131}nt");
}

#[test]
fn a_unit_must_end_at_a_word_boundary() {
    for text in ["50%x", "50%_", "5 kg2", "5 kg\u{0301}", "5 kg\u{00B2}", "5 m\u{1D7D9}", "5 kgs"] {
        none(text);
    }
    assert_eq!(single("5 kg.").span, Span::new(0, 4));
    assert_eq!(single("1 MB/s").span, Span::new(0, 4));
    assert_eq!(single("5 kg\u{1F4E6}").span, Span::new(0, 4));
    // A percent sign is no word character, so a second one ends the first.
    assert_eq!(single("50%%").span, Span::new(0, 3));
}

#[test]
fn a_mention_must_start_at_a_word_boundary() {
    for text in ["x5 kg", "_5 kg", "e\u{0301}5 kg", "\u{00B2}5 kg", "\u{1D7D9}5 kg"] {
        none(text);
    }
    // A combining mark alone back to the start counts as the start.
    assert_eq!(single("\u{0301}5 kg").span, Span::new(2, 6));
}

#[test]
fn only_one_gap_is_allowed() {
    for text in ["5  kg", "5 \u{00A0}kg", "5\u{200B}kg", "5 -kg", "5 - kg"] {
        none(text);
    }
}

#[test]
fn scale_markers_are_not_read() {
    // The number scan runs without scale markers, so a suffix or word is
    // where a unit would be, and none of them is one.
    for text in ["3 million km", "5k kg", "1.5k km", "2bn GB", "1e3 kg", "1.5e-3 kg"] {
        none(text);
    }
    // `m` as a suffix would be a million; here it is a metre.
    assert_mention(&single("3m"), "3", "m");
}

#[test]
fn money_takes_precedence() {
    for text in ["$3m", "$5 m", "USD 5 kg", "5 USD kg", "\u{20AC}5 kg", "3 pounds"] {
        none(text);
    }
    // The amount `$5` is left out and the quantity before it kept.
    let found = extract(&QuantityExtractor::default(), "10 kg-$5");
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].span, Span::new(0, 5));
    // A symbol after the unit is no amount.
    assert_eq!(single("5 m$").span, Span::new(0, 3));
    // A custom unit that is a currency code is money all the same.
    let custom = QuantityExtractor::with_units(["USD", "bbl"], LatinUs).unwrap();
    none_with(&custom, "5 USD");
    assert_mention(&single_with(&custom, "5 USD and 7 bbl"), "7", "bbl");
}

#[test]
fn signs() {
    let m = single("-5 kg");
    assert_eq!(m.span, whole("-5 kg"));
    assert_mention(&m, "-5", "kg");
    let m = single("(-5%)");
    assert_eq!(m.span, Span::new(1, 4));
    assert_mention(&m, "-5", "%");
    // After a letter, a digit, a currency symbol or a percent sign, a
    // hyphen is no minus.
    let m = single("a-5 kg");
    assert_eq!(m.span, Span::new(2, 6));
    assert_mention(&m, "5", "kg");
    assert_mention(&single("$-5%"), "5", "%");
    assert_mention(&single("\u{20AC}-5%"), "5", "%");
    let found = extract(&QuantityExtractor::default(), "5 kg-10 kg");
    assert_eq!(found.len(), 2);
    assert_mention(&found[1], "10", "kg");
    assert_eq!(found[1].span, Span::new(5, 10));
}

#[test]
fn a_negative_zero_is_zero() {
    let m = single("-0%");
    assert_eq!(m.span, whole("-0%"));
    assert!(!m.value.negative);
    let m = single("-0.0 kg");
    assert!(!m.value.negative);
    assert_eq!(m.value.to_plain_string(), "0.0");
}

#[test]
fn the_scale_is_the_fraction_written() {
    let m = single("1.250 GB");
    assert_eq!((m.value.digits.as_str(), m.value.scale), ("1250", 3));
    let m = single("007.50 kg");
    assert_eq!((m.value.digits.as_str(), m.value.scale), ("750", 2));
    let m = single("0.00%");
    assert_eq!((m.value.digits.as_str(), m.value.scale), ("0", 2));
    let long = "123456789012345678901234567890.5 kg";
    assert_eq!(single(long).value.to_plain_string(), "123456789012345678901234567890.5");
}

#[test]
fn custom_units_outside_the_basic_plane() {
    // Two mathematical bold letters: four UTF-16 units, eight bytes.
    let bold = "\u{1D424}\u{1D420}";
    let custom = QuantityExtractor::with_units([bold, "\u{E9}t"], LatinUs).unwrap();
    let m = single_with(&custom, &format!("5 {bold}"));
    assert_eq!(m.span, Span::new(0, 10));
    assert_eq!(m.unit, bold);
    assert_eq!(single_with(&custom, "3\u{E9}t").span, Span::new(0, 4));
    // Fourteen UTF-16 units is the limit (eight bold letters is sixteen).
    let four = bold.repeat(4);
    assert_eq!(QuantityExtractor::with_units([&four], LatinUs), Err(UnitError::NotAUnitToken(four.as_str().into())));
    assert!(QuantityExtractor::with_units(["abcdef"], LatinUs).is_ok());
    assert!(QuantityExtractor::with_units([format!("{bold}\u{1D41A}")], LatinUs).is_ok());
}

#[test]
fn a_custom_set_keeps_the_percent_marker_and_word() {
    let custom = QuantityExtractor::with_units(["bbl"], LatinUs).unwrap();
    assert_mention(&single_with(&custom, "5%"), "5", "%");
    assert_mention(&single_with(&custom, "5 percent"), "5", "%");
}

#[test]
fn a_unit_given_twice_is_kept_once() {
    let custom = QuantityExtractor::with_units(["bbl", "kg", "bbl"], LatinUs).unwrap();
    assert_eq!(custom.units().collect::<Vec<_>>(), ["bbl", "kg"]);
}

#[test]
fn the_annotator_names_its_configuration() {
    let us = QuantityExtractor::default();
    assert_ne!(us.config_hash(), eu().config_hash());
    assert_eq!(us.config_hash(), QuantityExtractor::new(LatinUs).config_hash());
    let ab = QuantityExtractor::with_units(["ab", "c"], LatinUs).unwrap();
    let a = QuantityExtractor::with_units(["a", "bc"], LatinUs).unwrap();
    assert_ne!(ab.config_hash(), a.config_hash());
    assert_ne!(us.config_hash(), QuantityExtractor::with_units(DEFAULT_UNITS, LatinEu).unwrap().config_hash());
}

#[test]
fn notations_are_independent_of_the_units() {
    for notation in [LatinUs, LatinEu] {
        let x = QuantityExtractor::new(notation);
        assert_eq!(x.notation(), notation);
        assert_eq!(x.units().count(), DEFAULT_UNITS.len());
    }
}

/// Behaviour of the Java that looks wrong and is kept, checked against a
/// JVM: see docs/quantities.md.
#[test]
fn opennlp_behaviour_kept() {
    // The hyphen of a range after a digit is no minus, and the first
    // number has no unit, so only the second is a mention.
    let m = single("5-10%");
    assert_eq!(m.span, Span::new(2, 5));
    assert_mention(&m, "10", "%");
    // The one gap may be a line break.
    assert_eq!(single("5\nkg").span, whole("5\nkg"));
    assert_eq!(single("5\u{2028}kg").span, whole("5\u{2028}kg"));
    // Short units are read wherever they follow a number.
    assert_mention(&single("at 5 g"), "5", "g");
    assert_mention(&single("5 t"), "5", "t");
    assert_mention(&single("100 m USD"), "100", "m");
    // Only the first unit of a compound unit is read.
    assert_mention(&single("1 MB/s"), "1", "MB");
}
