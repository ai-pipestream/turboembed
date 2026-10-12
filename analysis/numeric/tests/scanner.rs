//! The document scan: which mentions it finds, every column it writes,
//! and the annotator in a pipeline.

mod common;

use common::{eu, found, pairs, us};
use turbo_analysis::{Analyzer, Annotator, Document, Ix, Layer, Span, ViewId};
use turbo_numeric::{Multiplier, NumberNotation, NumberScanner, Numbers, ScaleMarker};

use NumberNotation::{LatinEu, LatinUs};

fn scan(text: &str, scanner: NumberScanner) -> Numbers {
    let mut out = Numbers::new(ViewId::SOURCE);
    scanner.scan(text, &mut out);
    assert_eq!(out.check(), Ok(()), "{text:?}");
    out
}

#[test]
fn empty_and_numberless_texts() {
    for text in ["", " ", "-", "--", "abc", "\u{1F4E6}", "e-", ",.", "k"] {
        assert_eq!(us(text), [], "{text:?}");
        assert_eq!(eu(text), [], "{text:?}");
    }
}

#[test]
fn a_text_that_ends_inside_a_number() {
    assert_eq!(us("total 1,23"), []);
    assert_eq!(us("total 1,2"), []);
    assert_eq!(us("total 12,"), pairs(&[("12", "12")]));
    assert_eq!(us("total 12."), pairs(&[("12", "12")]));
    assert_eq!(us("total 1,234"), pairs(&[("1,234", "1234")]));
    assert_eq!(us("total -"), []);
    assert_eq!(us("total 5 "), pairs(&[("5", "5")]));
    assert_eq!(us("total 5 m"), pairs(&[("5", "5")]));
}

#[test]
fn grouped_numbers() {
    assert_eq!(us("1,234,567.89"), pairs(&[("1,234,567.89", "1234567.89")]));
    assert_eq!(eu("1,234,567.89"), []);
    assert_eq!(eu("1.234,5"), pairs(&[("1.234,5", "1234.5")]));
    // In the US notation 1.234 is a fraction that a comma and a digit go on
    // from, so nothing is read, and no tail starts a number of its own.
    assert_eq!(us("1.234,5"), []);
    assert_eq!(us("1,00,000"), []);
    assert_eq!(eu("1,00,000"), []);
    assert_eq!(us("pay 1,00,000 now, or 7"), pairs(&[("7", "7")]));
    // Four leading digits cannot be grouped.
    assert_eq!(us("1234,567"), []);
}

#[test]
fn scale_markers() {
    assert_eq!(us("12k"), pairs(&[("12k", "12000")]));
    assert_eq!(us("3 million"), pairs(&[("3 million", "3000000")]));
    assert_eq!(us("3\u{00A0}million"), pairs(&[("3\u{00A0}million", "3000000")]));
    assert_eq!(us("5bn"), pairs(&[("5bn", "5000000000")]));
    assert_eq!(us("-1.5k"), pairs(&[("-1.5k", "-1500.0")]));
    assert_eq!(eu("2,5 Billion"), pairs(&[("2,5 Billion", "2500000000.0")]));
    assert_eq!(us("4 trillion"), pairs(&[("4 trillion", "4000000000000")]));
    assert_eq!(us("5x"), []);
    assert_eq!(us("12abc and 5"), pairs(&[("5", "5")]));
    assert_eq!(us("1e-3"), []);
    // Without scaling, the number ends with its digits and the letter after
    // it is not looked at.
    let plain = |text| found(text, LatinUs, false);
    assert_eq!(plain("12k"), pairs(&[("12", "12")]));
    assert_eq!(plain("3 million"), pairs(&[("3", "3")]));
    assert_eq!(plain("5x"), pairs(&[("5", "5")]));
    assert_eq!(plain("1e-3"), []);
}

#[test]
fn signs() {
    assert_eq!(us("-5"), pairs(&[("-5", "-5")]));
    assert_eq!(us("x -5"), pairs(&[("-5", "-5")]));
    // After a letter the minus is a hyphen and the number is positive.
    assert_eq!(us("a-5"), pairs(&[("5", "5")]));
    assert_eq!(us("5%-10%"), pairs(&[("5", "5"), ("10", "10")]));
    assert_eq!(us("$-5"), pairs(&[("5", "5")]));
    assert_eq!(us("--5"), pairs(&[("-5", "-5")]));
    assert_eq!(us("-0.00"), pairs(&[("-0.00", "0.00")]));
}

#[test]
fn combining_marks_around_numbers() {
    // After a digit a mark is no letter, so the number ends before it: the
    // Java checks no boundary after a number with no scale marker.
    assert_eq!(us("5\u{0301}"), pairs(&[("5", "5")]));
    assert_eq!(us("12\u{0301}34"), pairs(&[("12", "12")]));
    // Before a number the base character before the marks decides.
    assert_eq!(us("e\u{0301}5"), []);
    assert_eq!(us(" \u{0301}5"), pairs(&[("5", "5")]));
    assert_eq!(us("\u{0301}\u{0302}5"), pairs(&[("5", "5")]));
    assert_eq!(us("%\u{0301}-5"), pairs(&[("5", "5")]));
}

#[test]
fn multibyte_text_around_numbers() {
    let text = "\u{65E5}\u{672C} 12,500 \u{5186}, \u{20AC}3.5k \u{B7} \u{1F4E6}7 \u{6570}\u{5B57}12 \u{3B1}-4";
    let n = scan(text, NumberScanner::default());
    let spans: Vec<&str> = n.spans.iter().map(|(_, s)| &text[s.range()]).collect();
    // After a Greek letter the minus is a hyphen.
    assert_eq!(spans, ["12,500", "3.5k", "7", "4"]);
    assert!(!n.negative[3]);
    let at = |s: &str| text.find(s).unwrap() as u32;
    assert_eq!(n.spans.get(Ix::new(0)), Span::new(at("12,500"), at("12,500") + 6));
    assert_eq!(n.spans.get(Ix::new(1)), Span::new(at("3.5k"), at("3.5k") + 4));
    assert_eq!(n.numeral_end[1], at("3.5k") + 3);
    assert_eq!(n.value(Ix::new(1)).to_plain_string(), "3500.0");
    // A digit after a letter of any script is part of a word.
    assert_eq!(us("\u{6570}\u{5B57}12"), []);
    assert_eq!(us("\u{3B1}12"), []);
}

#[test]
fn digits_after_a_thin_space() {
    assert_eq!(us("x\u{2009}123 1\u{2009}3 \u{2009}123"), pairs(&[("123", "123"), ("123", "123")]));
    assert_eq!(us("1\u{2009}234"), []);
    assert_eq!(us("1\u{2009}million"), pairs(&[("1\u{2009}million", "1000000")]));
}

#[test]
fn every_column() {
    let text = "-3 million, 1.5k and 42.";
    let n = scan(text, NumberScanner::default());
    assert_eq!(n.len(), 3);
    let spans: Vec<Span> = n.spans.iter().map(|(_, s)| s).collect();
    assert_eq!(spans, [Span::new(0, 10), Span::new(12, 16), Span::new(21, 23)]);
    assert_eq!(n.numeral_end, [2, 15, 23]);
    assert_eq!(n.negative, [true, false, false]);
    assert_eq!(n.coefficient.iter().collect::<Vec<_>>(), ["3000000", "15000", "42"]);
    assert_eq!(n.scale, [0, 1, 0]);
    assert_eq!(n.multiplier, [Multiplier::Million, Multiplier::Thousand, Multiplier::One]);
    assert_eq!(n.marker, [ScaleMarker::Word, ScaleMarker::Suffix, ScaleMarker::None]);
    assert_eq!(n.notation, [LatinUs; 3]);
    assert_eq!(n.numeral(Ix::new(0)), Span::new(1, 2));
    assert_eq!(n.value(Ix::new(0)).to_plain_string(), "-3000000");
    assert_eq!(n.value(Ix::new(0)).to_f64(), -3e6);
    assert_eq!(n.value(Ix::new(1)).to_f64(), 1500.0);

    let n = scan("1,5", NumberScanner::new(LatinEu));
    assert_eq!(n.notation, [LatinEu]);
    assert_eq!(n.value(Ix::new(0)).to_plain_string(), "1.5");
}

#[test]
fn a_scan_appends_to_what_is_there() {
    let mut out = Numbers::new(ViewId::SOURCE);
    let scanner = NumberScanner::default();
    scanner.scan("1 2", &mut out);
    scanner.scan("", &mut out);
    assert_eq!(out.len(), 2);
    assert_eq!(out.check(), Ok(()));
}

#[test]
fn the_annotator_writes_the_layer() {
    let analyzer = Analyzer::builder().add(NumberScanner::default()).build().unwrap();
    assert_eq!(analyzer.plan(), [vec!["turbo-numeric:numbers"]]);
    let mut docs = vec![Document::new("Revenue rose 12% to $3.5bn."), Document::new("")];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let n = docs[0].get::<Numbers>().unwrap();
    let text = docs[0].source();
    let spans: Vec<&str> = n.spans.iter().map(|(_, s)| &text[s.range()]).collect();
    assert_eq!(spans, ["12", "3.5bn"]);
    assert_eq!(n.value(Ix::new(1)).to_plain_string(), "3500000000.0");
    assert!(docs[1].get::<Numbers>().unwrap().is_empty());
}

#[test]
fn the_configuration_is_in_the_hash() {
    let hashes: Vec<u64> = [
        NumberScanner::new(LatinUs),
        NumberScanner::new(LatinEu),
        NumberScanner { notation: LatinUs, apply_scale: false },
        NumberScanner { notation: LatinEu, apply_scale: false },
    ]
    .iter()
    .map(Annotator::config_hash)
    .collect();
    for (i, a) in hashes.iter().enumerate() {
        assert!(hashes[i + 1..].iter().all(|b| a != b), "{hashes:?}");
    }
    assert_eq!(NumberScanner::default(), NumberScanner::new(LatinUs));
}
