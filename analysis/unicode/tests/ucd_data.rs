//! The tables against the UCD files, parsed again here independently of
//! the generator.

use std::collections::HashMap;

use turbo_unicode::category::general_category;
use turbo_unicode::normalize::{canonical_combining_class, canonical_decomposition, compatibility_decomposition};
use turbo_unicode::script::{Script, script};
use turbo_unicode::{casefold, confusables, java, props};

mod common;

fn chars_of(field: &str) -> Vec<char> {
    common::string(field).chars().collect()
}

#[test]
fn general_category_of_every_code_point() {
    let mut assigned = vec![false; 0x110000];
    for row in common::unicode_data() {
        assigned[row.cp as usize] = true;
        let Some(c) = char::from_u32(row.cp) else { continue };
        assert_eq!(general_category(c).abbreviation(), row.fields[2], "U+{:04X}", row.cp);
    }
    for c in common::all_chars().filter(|&c| !assigned[c as usize]) {
        assert_eq!(general_category(c).abbreviation(), "Cn", "unassigned U+{:04X}", c as u32);
    }
}

#[test]
fn combining_class_and_decompositions_of_every_code_point() {
    let mut seen = vec![false; 0x110000];
    for row in common::unicode_data() {
        seen[row.cp as usize] = true;
        let Some(c) = char::from_u32(row.cp) else { continue };
        let ccc: u8 = row.fields[3].parse().expect("a number");
        assert_eq!(canonical_combining_class(c), ccc, "U+{:04X}", row.cp);
        let field = &row.fields[5];
        let (canonical, compatibility) = match field.strip_prefix('<') {
            Some(tagged) => (None, Some(chars_of(tagged.split_once('>').expect("closed tag").1))),
            None if field.is_empty() => (None, None),
            None => (Some(chars_of(field)), None),
        };
        assert_eq!(canonical_decomposition(c).map(<[char]>::to_vec), canonical, "U+{:04X}", row.cp);
        assert_eq!(compatibility_decomposition(c).map(<[char]>::to_vec), compatibility, "U+{:04X}", row.cp);
    }
    for c in common::all_chars().filter(|&c| !seen[c as usize]) {
        assert_eq!(canonical_combining_class(c), 0);
        assert_eq!(canonical_decomposition(c), None);
    }
}

#[test]
fn simple_case_mappings_and_digits_of_every_code_point() {
    let mut rows = HashMap::new();
    for row in common::unicode_data() {
        rows.insert(row.cp, row.fields);
    }
    for c in common::all_chars() {
        let fields = rows.get(&(c as u32));
        let mapped = |i: usize| {
            fields.map(|f| &f[i]).filter(|f| !f.is_empty()).map(|f| char::from_u32(common::cp(f)).unwrap()).unwrap_or(c)
        };
        assert_eq!(java::to_upper_char(c), mapped(12), "upper U+{:04X}", c as u32);
        assert_eq!(java::to_lower_char(c), mapped(13), "lower U+{:04X}", c as u32);
        let nd = fields.is_some_and(|f| f[2] == "Nd");
        let digit = if nd { Some(fields.unwrap()[6].parse::<u8>().expect("a digit value")) } else { None };
        assert_eq!(java::digit(c), digit, "digit U+{:04X}", c as u32);
        assert_eq!(java::is_digit(c), nd, "is_digit U+{:04X}", c as u32);
    }
}

#[test]
fn case_folding_of_every_entry() {
    let mut full: HashMap<u32, String> = HashMap::new();
    let mut simple: HashMap<u32, String> = HashMap::new();
    let mut turkic: HashMap<u32, String> = HashMap::new();
    for fields in common::records(&common::read("CaseFolding.txt")) {
        let code = common::cp(&fields[0]);
        let mapping = common::string(&fields[2]);
        match fields[1].as_str() {
            "C" => {
                full.insert(code, mapping.clone());
                simple.insert(code, mapping);
            }
            "F" => {
                full.insert(code, mapping);
            }
            "S" => {
                simple.insert(code, mapping);
            }
            "T" => {
                turkic.insert(code, mapping);
            }
            status => panic!("unknown status {status}"),
        }
    }
    for c in common::all_chars() {
        let cp = c as u32;
        let expected_full = full.get(&cp).map(String::as_str);
        assert_eq!(casefold::full(c), expected_full, "full U+{cp:04X}");
        let expected_simple = simple.get(&cp).map_or(c, |s| s.chars().next().unwrap());
        assert_eq!(casefold::simple(c), expected_simple, "simple U+{cp:04X}");
        let expected_turkic = turkic.get(&cp).map(String::as_str).or(expected_full);
        assert_eq!(casefold::turkic_full(c), expected_turkic, "turkic full U+{cp:04X}");
        let expected_turkic_simple = turkic.get(&cp).map_or(expected_simple, |s| s.chars().next().unwrap());
        assert_eq!(casefold::turkic_simple(c), expected_turkic_simple, "turkic simple U+{cp:04X}");
    }
}

#[test]
fn fold_of_a_string() {
    let mut out = String::new();
    casefold::fold("Stra\u{DF}e \u{130}stanbul \u{3A3}\u{39F}\u{3A3} \u{FB01}NE \u{10400}", &mut out);
    assert_eq!(out, "strasse i\u{307}stanbul \u{3C3}\u{3BF}\u{3C3} fine \u{10428}");
}

/// A property check: the file, the property name, and the function under test.
type PropertyCheck = (&'static str, &'static str, fn(char) -> bool);

#[test]
fn binary_properties_of_every_code_point() {
    let checks: [PropertyCheck; 17] = [
        ("PropList.txt", "White_Space", props::white_space),
        ("PropList.txt", "Dash", props::dash),
        ("PropList.txt", "Quotation_Mark", props::quotation_mark),
        ("PropList.txt", "Regional_Indicator", props::regional_indicator),
        ("PropList.txt", "Variation_Selector", props::variation_selector),
        ("DerivedCoreProperties.txt", "Default_Ignorable_Code_Point", props::default_ignorable),
        ("DerivedCoreProperties.txt", "Alphabetic", props::alphabetic),
        ("DerivedCoreProperties.txt", "Lowercase", props::lowercase),
        ("DerivedCoreProperties.txt", "Uppercase", props::uppercase),
        ("DerivedCoreProperties.txt", "Cased", props::cased),
        ("DerivedCoreProperties.txt", "Case_Ignorable", props::case_ignorable),
        ("emoji-data.txt", "Emoji", props::emoji),
        ("emoji-data.txt", "Emoji_Presentation", props::emoji_presentation),
        ("emoji-data.txt", "Emoji_Modifier", props::emoji_modifier),
        ("emoji-data.txt", "Emoji_Modifier_Base", props::emoji_modifier_base),
        ("emoji-data.txt", "Emoji_Component", props::emoji_component),
        ("emoji-data.txt", "Extended_Pictographic", props::extended_pictographic),
    ];
    for (file, property, test) in checks {
        let set = common::binary_property(file, property);
        for c in common::all_chars() {
            assert_eq!(test(c), set[c as usize], "{property} of U+{:04X}", c as u32);
        }
    }
}

#[test]
fn script_of_every_code_point() {
    let mut expected = vec!["Unknown".to_owned(); 0x110000];
    for fields in common::records(&common::read("Scripts.txt")) {
        for cp in common::range(&fields[0]) {
            expected[cp as usize] = fields[1].clone();
        }
    }
    for c in common::all_chars() {
        assert_eq!(script(c).name(), expected[c as usize], "U+{:04X}", c as u32);
    }
    assert_eq!(script('a'), Script::Latin);
    assert_eq!(script('\u{3B1}'), Script::Greek);
    assert_eq!(script('\u{416}'), Script::Cyrillic);
    assert_eq!(script(' '), Script::Common);
    assert_eq!(script('\u{301}'), Script::Inherited);
    assert_eq!(script('\u{378}'), Script::Unknown);
    assert_eq!(Script::OldItalic.name(), "Old_Italic");
}

#[test]
fn confusable_prototype_of_every_entry() {
    let mut expected: HashMap<u32, String> = HashMap::new();
    for fields in common::records(&common::read("confusables.txt").replace('\u{FEFF}', "")) {
        expected.insert(common::cp(&fields[0]), common::string(&fields[1]));
    }
    for c in common::all_chars() {
        assert_eq!(confusables::prototype(c), expected.get(&(c as u32)).map(String::as_str), "U+{:04X}", c as u32);
    }
}
