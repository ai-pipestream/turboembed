//! Simple case mapping: one character to one character from the Unicode
//! 18.0 tables, with no special casing and no context.

use turbo_unicode::casemap::{has_upper, lower, lowercase, upper};

fn lowered(s: &str) -> String {
    let mut out = String::new();
    lowercase(s, &mut out);
    out
}

#[test]
fn ascii_latin_greek_and_cyrillic_capitals_lower_to_their_small_letters() {
    assert_eq!(lowered("Dogs"), "dogs");
    assert_eq!(lowered("ÜBER Straße"), "über straße");
    assert_eq!(lowered("\u{3A3}\u{39F}\u{3A6}\u{38A}\u{391}"), "\u{3C3}\u{3BF}\u{3C6}\u{3AF}\u{3B1}");
    assert_eq!(lowered("\u{41F}\u{420}\u{418}\u{412}\u{415}\u{422}"), "\u{43F}\u{440}\u{438}\u{432}\u{435}\u{442}");
    assert_eq!(lowered("already lower 123 \u{1F600}"), "already lower 123 \u{1F600}");
    assert_eq!(lowered(""), "");
}

#[test]
fn the_mapping_is_simple_and_without_context() {
    // U+0130 lowers to U+0069 by its simple mapping; SpecialCasing's
    // `i\u{307}` is not applied, so the text keeps its length.
    assert_eq!(lower('\u{130}'), 'i');
    assert_eq!(lowered("\u{130}stanbul"), "istanbul");
    // Final sigma is a context rule of full case mapping, not of this one.
    assert_eq!(lowered("\u{39F}\u{3A3}"), "\u{3BF}\u{3C3}");
    // U+1E9E CAPITAL SHARP S lowers to U+00DF; U+00DF has no uppercase
    // simple mapping.
    assert_eq!(lower('\u{1E9E}'), '\u{DF}');
    assert_eq!(upper('\u{DF}'), '\u{DF}');
    assert_eq!(upper('ü'), 'Ü');
    assert_eq!(upper('\u{3C2}'), '\u{3A3}', "final sigma uppers to sigma");
    for text in ["", "abc", "\u{1F600}", "\u{DF}"] {
        assert!(!has_upper(text), "{text:?}");
    }
    for text in ["Abc", "ab\u{C4}", "\u{3A3}", "\u{1E9E}"] {
        assert!(has_upper(text), "{text:?}");
    }
}

#[test]
fn every_character_keeps_its_count_and_lowers_to_a_fixed_point() {
    for cp in 0..=0x10FFFFu32 {
        let Some(c) = char::from_u32(cp) else { continue };
        let l = lower(c);
        assert_eq!(lower(l), l, "{c:?} lowers to {l:?}, which lowers again");
        let mut s = String::new();
        lowercase(c.encode_utf8(&mut [0; 4]), &mut s);
        assert_eq!(s.chars().count(), 1, "{c:?}");
        assert_eq!(s.chars().next(), Some(l));
    }
}
