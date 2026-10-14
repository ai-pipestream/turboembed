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

/// The JDK's condition walks the sigma's word, as its word break iterator
/// for the root locale delimits it, passing over every character that is
/// not cased; Unicode's Final_Sigma skips only case-ignorable ones. These
/// are the cases where the two differ, each checked on OpenJDK 21.
#[test]
fn to_lowercase_applies_the_jdks_final_sigma_condition_not_unicodes() {
    // Final: the sigma's word ends, or a digit, mark or uncased letter is
    // passed over on the way to a cased letter.
    for text in [
        "\u{391}\u{3A3}1",
        "\u{391}1\u{3A3}",
        "\u{391}\u{3A3}123",
        "\u{391}\u{3A3},a",
        "\u{391}\u{3A3})a",
        "\u{391}\u{3A3}\u{A0}a",
        "\u{391}\u{3A3}\u{751F}a",
        "\u{391}\u{3A3}\u{3042}a",
        "\u{391}\u{3A3}\u{30A2}a",
        "\u{391}\u{3A3}\u{4E00}a",
        "\u{391}\u{3A3}\u{FA2D}a",
        "\u{391}\u{3A3}\u{3005}a",
        "\u{391}\u{3A3}\u{964}a",
        "\u{391}\u{3A3}\u{1F600}a",
        "\u{391}\u{3A3}%a",
        "\u{391}\u{3A3}\u{20AC}a",
        "\u{391}\u{3A3}\u{B7}a",
        "\u{391}\u{3A3}\u{387}a",
        "\u{391}\u{3A3}\u{2019}a",
        "\u{391}\u{3A3}\u{66B}a",
        "\u{391}\u{3A3}.1a",
        "\u{391}\u{3A3}1.a",
        "\u{391}\u{3A3}'1a",
        "\u{391}\u{3A3}1'a",
        "\u{391}\u{3A3}1-1a",
        "\u{391}\u{3A3}--a",
        "\u{391}\u{3A3}__a",
        "\u{391}\u{3A3}_1",
        "\u{391}_\u{3A3}",
        "\u{391}-\u{3A3}",
        "\u{391}\u{3A3}.\u{301}a",
        "\u{391}\u{3A3}\u{E02} a",
        "\u{391}\u{3A3}\u{E01}\u{E02}\u{E03}\u{E04} a",
        "\u{391}\u{3A3}\u{F7E}",
        "\u{391}\u{3A3}\u{300}",
        "\u{391}\u{300}\u{3A3}",
        "1\u{345}\u{3A3}",
        // Other_Lowercase characters the JDK does not count as cased.
        "\u{391}\u{3A3}\u{AA}",
        "\u{391}\u{3A3}\u{1D62}",
        "\u{391}\u{3A3}\u{1D78}",
        "\u{391}\u{3A3}\u{2071}",
        "\u{391}\u{3A3}\u{A770}",
        "\u{391}\u{3A3}\u{2180}",
        "\u{391}\u{3A3}\u{2B9}",
        "\u{391}\u{3A3}\u{24B6}",
    ] {
        let at = text.find('\u{3A3}').unwrap();
        assert_eq!(lower(text)[at..].chars().next(), Some('\u{3C2}'), "{text:?} should end in final sigma");
    }
    // Not final: a cased character is reached within the word.
    for text in [
        "\u{391}\u{3A3}1a",
        "\u{391}\u{3A3}12a",
        "\u{391}\u{3A3}.a",
        "\u{391}\u{3A3}-a",
        "\u{391}\u{3A3}_a",
        "\u{391}\u{3A3}'a",
        "\u{391}\u{3A3}\"a",
        "\u{391}\u{3A3}\u{2013}a",
        "\u{391}\u{3A3}\u{2014}a",
        "\u{391}\u{3A3}\u{203F}a",
        "\u{391}\u{3A3}\u{FF3F}a",
        "\u{391}\u{3A3}\u{2027}a",
        "\u{391}\u{3A3}\u{AD}a",
        "\u{391}\u{3A3}1,1a",
        "\u{391}\u{3A3}1.1a",
        "\u{391}\u{3A3}1'1a",
        "\u{391}\u{3A3}\u{E02}a",
        "\u{391}\u{3A3}\u{E02}\u{E48}a",
        "\u{391}\u{3A3}\u{E02}.a",
        "\u{391}\u{3A3}a\u{E02}",
        "\u{391}\u{3A3}\u{9FFF}a",
        "\u{391}\u{3A3}\u{3400}a",
        "\u{391}\u{3A3}\u{FA2E}a",
        "\u{391}\u{3A3}\u{AC00}a",
        "\u{391}\u{3A3}\u{903}a",
        "\u{391}\u{3A3}\u{B2}a",
        "\u{391}\u{3A3}\u{2460}a",
        "\u{391}\u{3A3}\u{301}.a",
        "\u{391}\u{3A3}\u{200D}\u{301}a",
        "\u{391}\u{3A3}\u{200B}a",
        "\u{391}\u{3A3}\u{3A3}",
        "\u{AA}\u{3A3}",
        "\u{345}\u{3A3}",
        "\u{391},\u{3A3}",
        // The JDK's own list of cased characters.
        "\u{391}\u{3A3}\u{2160}",
        "\u{391}\u{3A3}\u{217F}",
        "\u{391}\u{3A3}\u{2B0}",
        "\u{391}\u{3A3}\u{2C0}",
        "\u{391}\u{3A3}\u{37A}",
        "\u{391}\u{3A3}\u{1D2C}",
        "\u{391}\u{3A3}\u{1D61}",
        "\u{391}\u{3A3}\u{10400}",
        "\u{391}\u{3A3}\u{10428}",
        "\u{391}\u{3A3}\u{1E9E}",
        "\u{391}\u{3A3}\u{1C5}",
    ] {
        let at = text.find('\u{3A3}').unwrap();
        assert_eq!(lower(text)[at..].chars().next(), Some('\u{3C3}'), "{text:?} should end in sigma");
    }
}

/// Pairs on which the JDK's word break iterator decides, each checked on
/// OpenJDK 21: a mark after a joiner breaks the word and one before it does
/// not; the kana marks U+3099 and U+309A are marks, not kana; a danda
/// chains letters to the digits after it; a spacing mark is a letter; the
/// soft hyphen is a joiner, not a format character, so it does not join a
/// Roman numeral; a supplementary character joins the word, but
/// `isBoundary` right after one answers true unless it starts the text;
/// U+0295 is a cased letter to JDK 21.
#[test]
fn the_word_iterators_corners_decide_the_sigma() {
    let decisions = |text: &str| -> String {
        lower(text)
            .chars()
            .zip(text.chars())
            .filter(|(_, c)| *c == '\u{3A3}')
            .map(|(l, _)| if l == '\u{3C2}' { 'F' } else { 'S' })
            .collect()
    };
    for (text, expected) in [
        ("\u{3A3}.\u{301}\u{3A3}", "SS"),
        ("\u{3A3}\u{301}.\u{3A3}", "SF"),
        ("A\u{3099}\u{3A3}", "F"),
        ("\u{3A3}\u{964}\u{2460}\u{3A3}", "SF"),
        ("\u{391}\u{964}1\u{3A3}", "F"),
        ("K\u{203F}\u{F7E}\u{3A3}", "S"),
        ("\u{2160}\u{AD}\u{3A3}", "S"),
        ("\u{2160}\u{3A3}", "F"),
        ("\u{391}\u{20000}\u{3A3}", "S"),
        ("\u{391}\u{10400}\u{3A3}", "S"),
        ("a\u{1D400}\u{3A3}", "S"),
        ("\u{3A3}\u{109B2}\u{3A3}", "SS"),
        ("\u{391}\u{3A3}\u{20000}a", "F"),
        ("\u{10400}\u{3A3}", "F"),
        ("\u{1E9E}_\u{3A3}-\u{180E}\u{3A3}", "SF"),
        ("\u{295}\u{3A3}", "F"),
        ("\u{391}\u{3A3}\u{200B}a", "S"),
        ("\u{391}\u{3A3}\u{2060}\u{301}a", "S"),
    ] {
        assert_eq!(decisions(text), expected, "{text:?}");
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
