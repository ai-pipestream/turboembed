//! Checks every code point of the Java character class tables against the
//! UCD files they were generated from, read here with a parser of their
//! own, so a fault in the generator or the table readers shows up without
//! a JDK.

use std::path::Path;

use turbo_unicode::java::{self, GeneralCategory, Script};

const CODE_POINTS: usize = 0x11_0000;

fn ucd_file(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("ucd").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// Calls `f(cp, value)` for every code point a property file lists.
fn for_each_listed(text: &str, mut f: impl FnMut(usize, &str)) {
    for line in text.lines() {
        let content = line.split('#').next().unwrap().trim();
        if content.is_empty() {
            continue;
        }
        let (points, value) = content.split_once(';').unwrap();
        let value = value.split(';').next().unwrap().trim();
        let points = points.trim();
        let (start, end) = points.split_once("..").unwrap_or((points, points));
        for cp in usize::from_str_radix(start, 16).unwrap()..=usize::from_str_radix(end, 16).unwrap() {
            f(cp, value);
        }
    }
}

/// The code points whose value in the file is `property`.
fn property(text: &str, property: &str) -> Vec<bool> {
    let mut set = vec![false; CODE_POINTS];
    for_each_listed(text, |cp, value| set[cp] |= value == property);
    set
}

#[test]
fn general_category_matches_derived_general_category() {
    let mut expected = vec![GeneralCategory::Unassigned; CODE_POINTS];
    for_each_listed(&ucd_file("DerivedGeneralCategory.txt"), |cp, value| {
        expected[cp] = GeneralCategory::from_abbreviation(value).unwrap();
    });
    for (cp, &category) in expected.iter().enumerate() {
        let cp32 = cp as u32;
        assert_eq!(java::general_category(cp32), category, "U+{cp:04X}");
        let letter = matches!(
            category,
            GeneralCategory::UppercaseLetter
                | GeneralCategory::LowercaseLetter
                | GeneralCategory::TitlecaseLetter
                | GeneralCategory::ModifierLetter
                | GeneralCategory::OtherLetter
        );
        assert_eq!(java::is_letter(cp32), letter, "U+{cp:04X}");
        assert_eq!(java::is_digit(cp32), category == GeneralCategory::DecimalDigitNumber, "U+{cp:04X}");
    }
}

#[test]
fn flags_match_the_property_files() {
    let alphabetic = property(&ucd_file("DerivedCoreProperties.txt"), "Alphabetic");
    let other_upper = property(&ucd_file("PropList.txt"), "Other_Uppercase");
    for cp in 0..CODE_POINTS {
        let cp32 = cp as u32;
        assert_eq!(java::is_alphabetic(cp32), alphabetic[cp], "U+{cp:04X}");
        let upper = other_upper[cp] || java::general_category(cp32) == GeneralCategory::UppercaseLetter;
        assert_eq!(java::is_upper_case(cp32), upper, "U+{cp:04X}");
    }
}

#[test]
fn lower_case_matches_unicode_data() {
    let mut expected: Vec<u32> = (0..CODE_POINTS as u32).collect();
    for line in ucd_file("UnicodeData.txt").lines() {
        let fields: Vec<&str> = line.split(';').collect();
        if !fields[13].is_empty() {
            let cp = usize::from_str_radix(fields[0], 16).unwrap();
            expected[cp] = u32::from_str_radix(fields[13], 16).unwrap();
        }
    }
    for (cp, &lower) in expected.iter().enumerate() {
        assert_eq!(java::to_lower_case(cp as u32), lower, "U+{cp:04X}");
    }
}

#[test]
fn script_matches_scripts_txt() {
    let mut expected = vec![Script::Other; CODE_POINTS];
    for_each_listed(&ucd_file("Scripts.txt"), |cp, value| {
        expected[cp] = Script::from_property_name(value).unwrap_or(Script::Other);
    });
    for (cp, &script) in expected.iter().enumerate() {
        assert_eq!(java::script(cp as u32), script, "U+{cp:04X}");
    }
}
