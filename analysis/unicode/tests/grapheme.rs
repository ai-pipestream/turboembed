//! Extended grapheme clusters and the properties behind them, checked
//! against the UCD files directly (not through the generator).

mod common;

use turbo_unicode::grapheme::{
    GraphemeBreak, IndicConjunctBreak, grapheme_boundaries, grapheme_break, indic_conjunct_break,
};
use turbo_unicode::props::complex_context;

#[test]
fn grapheme_break_matches_the_property_file() {
    let mut expected = vec![GraphemeBreak::Other; 0x110000];
    for fields in common::records(&common::read("GraphemeBreakProperty.txt")) {
        let value = GraphemeBreak::ALL
            .into_iter()
            .find(|v| v.property_name() == fields[1])
            .unwrap_or_else(|| panic!("unknown value {}", fields[1]));
        for cp in common::range(&fields[0]) {
            expected[cp as usize] = value;
        }
    }
    for cp in 0..0x110000u32 {
        assert_eq!(grapheme_break(cp), expected[cp as usize], "U+{cp:04X}");
    }
    assert_eq!(grapheme_break(0x110000), GraphemeBreak::Other);
}

#[test]
fn indic_conjunct_break_matches_derived_core_properties() {
    let mut expected = vec![IndicConjunctBreak::None; 0x110000];
    let mut seen = 0;
    for fields in common::records(&common::read("DerivedCoreProperties.txt")) {
        if fields.len() != 3 || fields[1] != "InCB" {
            continue;
        }
        let value = IndicConjunctBreak::ALL
            .into_iter()
            .find(|v| v.property_name() == fields[2])
            .unwrap_or_else(|| panic!("unknown value {}", fields[2]));
        for cp in common::range(&fields[0]) {
            expected[cp as usize] = value;
            seen += 1;
        }
    }
    assert!(seen > 0);
    for cp in 0..0x110000u32 {
        assert_eq!(indic_conjunct_break(cp), expected[cp as usize], "U+{cp:04X}");
    }
}

#[test]
fn complex_context_matches_line_break() {
    let expected = common::binary_property("LineBreak.txt", "SA");
    for c in common::all_chars() {
        assert_eq!(complex_context(c), expected[c as usize], "U+{:04X}", c as u32);
    }
    assert!(complex_context('\u{0E01}') && complex_context('\u{1000}') && !complex_context('a'));
}

/// Every case of `GraphemeBreakTest.txt`: `÷` marks a boundary and `×`
/// none, between hexadecimal code points.
#[test]
fn grapheme_break_test() {
    let text = common::read("GraphemeBreakTest.txt");
    let mut cases = 0;
    for line in text.lines() {
        let data = line.split('#').next().unwrap().trim();
        if data.is_empty() {
            continue;
        }
        let mut input = String::new();
        let mut expected = Vec::new();
        for item in data.split_whitespace() {
            match item {
                "÷" => expected.push(input.len()),
                "×" => {}
                hex => input.push(char::from_u32(common::cp(hex)).expect("not a surrogate")),
            }
        }
        let actual: Vec<usize> = grapheme_boundaries(&input).collect();
        assert_eq!(actual, expected, "{line}");
        cases += 1;
    }
    assert_eq!(cases, 853);
}
