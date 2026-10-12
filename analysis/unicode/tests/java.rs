//! `java` against the JDK's documented behaviour. The expected values were
//! taken from OpenJDK 21 (Unicode 15.0); every character used is in that
//! repertoire.

use turbo_unicode::java;

fn lower(text: &str) -> String {
    let mut out = String::new();
    java::to_lowercase(text, &mut out);
    out
}

#[test]
fn whitespace_follows_character_is_whitespace() {
    for c in ['\t', '\n', '\u{B}', '\u{C}', '\r', '\u{1C}', '\u{1D}', '\u{1E}', '\u{1F}', ' ', '\u{1680}', '\u{2000}'] {
        assert!(java::is_whitespace(c), "U+{:04X}", c as u32);
    }
    for c in ['\u{2028}', '\u{2029}', '\u{205F}', '\u{3000}'] {
        assert!(java::is_whitespace(c), "U+{:04X}", c as u32);
    }
    // No-break spaces, NEXT LINE and the zero-width characters are not.
    for c in ['\u{A0}', '\u{2007}', '\u{202F}', '\u{85}', '\u{200B}', '\u{FEFF}', '\u{180E}', 'a', '\0'] {
        assert!(!java::is_whitespace(c), "U+{:04X}", c as u32);
    }
}

#[test]
fn space_char_is_every_separator() {
    for c in [' ', '\u{A0}', '\u{2007}', '\u{202F}', '\u{2028}', '\u{2029}', '\u{3000}'] {
        assert!(java::is_space_char(c), "U+{:04X}", c as u32);
    }
    for c in ['\t', '\n', '\u{85}', '\u{200B}'] {
        assert!(!java::is_space_char(c), "U+{:04X}", c as u32);
    }
}

#[test]
fn letters_and_digits_follow_the_jdk() {
    for c in ['a', 'Z', '\u{AA}', '\u{B5}', '\u{2B0}', '\u{1C5}', '\u{3B1}', '\u{5D0}', '\u{4E00}', '\u{10400}'] {
        assert!(java::is_letter(c), "U+{:04X}", c as u32);
        assert!(java::is_letter_or_digit(c), "U+{:04X}", c as u32);
        assert!(!java::is_digit(c), "U+{:04X}", c as u32);
    }
    for c in ['\u{301}', '\u{2160}', '\u{1F600}', '_', '1'] {
        assert!(!java::is_letter(c), "U+{:04X}", c as u32);
    }
    for (c, value) in [('7', 7), ('\u{660}', 0), ('\u{FF10}', 0), ('\u{1D7CE}', 0), ('\u{1D7D7}', 9), ('\u{96F}', 9)] {
        assert!(java::is_digit(c), "U+{:04X}", c as u32);
        assert!(java::is_letter_or_digit(c), "U+{:04X}", c as u32);
        assert_eq!(java::digit(c), Some(value), "U+{:04X}", c as u32);
    }
    // Roman numerals and superscripts are numbers but not decimal digits.
    for c in ['\u{2160}', '\u{B2}', 'a', 'A'] {
        assert_eq!(java::digit(c), None, "U+{:04X}", c as u32);
    }
}

#[test]
fn supplementary_and_simple_case_mappings() {
    assert!(java::is_supplementary('\u{10000}'));
    assert!(!java::is_supplementary('\u{FFFF}'));
    assert_eq!(java::to_lower_char('A'), 'a');
    assert_eq!(java::to_lower_char('\u{130}'), 'i', "the simple mapping of U+0130");
    assert_eq!(java::to_lower_char('\u{10400}'), '\u{10428}');
    assert_eq!(java::to_upper_char('\u{DF}'), '\u{DF}', "sharp s has no simple uppercase");
    assert_eq!(java::to_upper_char('\u{3C2}'), '\u{3A3}');
    assert_eq!(java::to_upper_char('1'), '1');
}

#[test]
fn to_lowercase_applies_final_sigma() {
    let cases = [
        ("\u{39F}\u{394}\u{39F}\u{3A3}", "\u{3BF}\u{3B4}\u{3BF}\u{3C2}"),
        ("\u{3A3}\u{391}", "\u{3C3}\u{3B1}"),
        ("\u{3A3}", "\u{3C3}"),
        ("\u{391}\u{3A3}.", "\u{3B1}\u{3C2}."),
        ("\u{391}\u{3A3}'\u{391}", "\u{3B1}\u{3C3}'\u{3B1}"),
        ("\u{391}'\u{3A3}", "\u{3B1}'\u{3C2}"),
        ("\u{391}\u{3A3}'", "\u{3B1}\u{3C2}'"),
        ("\u{391}\u{3A3}\u{301}", "\u{3B1}\u{3C2}\u{301}"),
        ("\u{391}\u{3A3}\u{301}\u{391}", "\u{3B1}\u{3C3}\u{301}\u{3B1}"),
        ("\u{391}.\u{3A3}", "\u{3B1}.\u{3C2}"),
        ("\u{391}\u{3A3}1", "\u{3B1}\u{3C2}1"),
        ("\u{391}\u{3A3} \u{391}", "\u{3B1}\u{3C2} \u{3B1}"),
        ("\u{391}\u{3A3}\u{AD}\u{391}", "\u{3B1}\u{3C3}\u{AD}\u{3B1}"),
        ("1\u{3A3}", "1\u{3C3}"),
        ("\u{391}\u{3A3}\u{391}", "\u{3B1}\u{3C3}\u{3B1}"),
        // U+02B0 is both cased and case-ignorable; it counts as cased.
        ("A\u{2B0}\u{3A3}", "a\u{2B0}\u{3C2}"),
        ("\u{3A3}\u{2B0}", "\u{3C3}\u{2B0}"),
    ];
    for (text, expected) in cases {
        assert_eq!(lower(text), expected, "{text:?}");
    }
}

#[test]
fn to_lowercase_applies_unconditional_special_casing_only() {
    assert_eq!(lower("\u{130}STANBUL"), "i\u{307}stanbul");
    // Turkish and Lithuanian mappings are language-specific and not applied.
    assert_eq!(lower("I"), "i");
    assert_eq!(lower("\u{CC}"), "\u{EC}");
    assert_eq!(lower("Mixed ASCII and \u{C4}\u{D6}\u{DC} \u{10400}"), "mixed ascii and \u{E4}\u{F6}\u{FC} \u{10428}");
    let mut out = String::from("kept ");
    java::to_lowercase("ABC", &mut out);
    assert_eq!(out, "kept abc", "appends to the output");
}
