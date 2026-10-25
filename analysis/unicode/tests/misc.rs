//! Spot checks of the small modules: categories, properties and
//! confusable skeletons.

use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::{confusables, props};

#[test]
fn category_groups() {
    assert_eq!(general_category('A'), GeneralCategory::Lu);
    assert!(general_category('\u{1C5}').is_letter());
    assert!(general_category('\u{301}').is_mark());
    assert!(general_category('\u{2160}').is_number());
    assert!(general_category('\u{2014}').is_punctuation());
    assert!(general_category('$').is_symbol());
    assert!(general_category('\u{2028}').is_separator());
    assert!(general_category('\u{E000}').is_other());
    assert_eq!(general_category('\u{E000}'), GeneralCategory::Co);
    assert_eq!(general_category('\u{10FFFF}'), GeneralCategory::Cn);
    assert_eq!(GeneralCategory::Pd.abbreviation(), "Pd");
}

#[test]
fn property_spot_checks() {
    assert!(props::white_space('\u{85}'), "NEXT LINE is White_Space though not Java whitespace");
    assert!(props::default_ignorable('\u{200B}'));
    assert!(props::dash('\u{2014}'));
    assert!(props::quotation_mark('\u{201C}'));
    assert!(props::alphabetic('\u{4E00}'));
    assert!(props::lowercase('\u{AA}'));
    assert!(props::uppercase('\u{2160}'));
    assert!(props::regional_indicator('\u{1F1E6}'));
    assert!(props::variation_selector('\u{FE0F}'));
    assert!(props::emoji('\u{A9}') && !props::emoji_presentation('\u{A9}'));
    assert!(props::emoji_presentation('\u{1F600}'));
    assert!(props::emoji_modifier('\u{1F3FB}'));
    assert!(props::emoji_modifier_base('\u{1F44D}'));
    assert!(props::emoji_component('\u{200D}'));
    assert!(props::extended_pictographic('\u{1F600}'));
    assert!(!props::extended_pictographic('a'));
}

fn skeleton(text: &str) -> String {
    let mut out = String::new();
    confusables::skeleton(text, &mut out);
    out
}

#[test]
fn skeletons_of_confusable_strings_are_equal() {
    // Cyrillic a (U+0430) looks like Latin a.
    assert_eq!(confusables::prototype('\u{430}'), Some("a"));
    assert_eq!(confusables::prototype('a'), None);
    assert_eq!(skeleton("p\u{430}ypal"), skeleton("paypal"));
    assert_eq!(skeleton("m"), "rn");
    assert_ne!(skeleton("paypal"), skeleton("paypa1x"));
    // The skeleton is in NFD: a precomposed letter and its decomposition agree.
    assert_eq!(skeleton("\u{E9}"), skeleton("e\u{301}"));
}
