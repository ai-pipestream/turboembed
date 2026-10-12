//! `java.lang.Character`'s classes on a code point given as `u32`.
//!
//! OpenNLP's tokenizers and sentence detector classify characters with
//! the `int` overloads of `java.lang.Character` while walking a string in
//! UTF-16 code units, and their features and splits depend on the
//! answers, so the ports ask the same questions here. Each function names
//! the Java method it reproduces. A surrogate code unit (U+D800 to
//! U+DFFF) is a valid argument: its category is [`GeneralCategory::Cs`],
//! its script is [`Script::Unknown`] and it is in no class, as in Java. A
//! value above U+10FFFF is treated as unassigned.
//!
//! Every answer is read from the crate's general category, script,
//! property and case tables, the same ones the `char` functions of
//! [`crate::java`] read:
//!
//! | Function | Java method | Definition from the UCD |
//! |---|---|---|
//! | [`general_category`], [`get_type`] | `getType(int)` | `UnicodeData.txt` field 2; [`java_type`] gives Java's number |
//! | [`is_letter`] | `isLetter(int)` | general category `Lu`, `Ll`, `Lt`, `Lm` or `Lo` |
//! | [`is_digit`] | `isDigit(int)` | general category `Nd` |
//! | [`is_upper_case`] | `isUpperCase(int)` | `Uppercase` (`DerivedCoreProperties.txt`): `Lu` or `Other_Uppercase` |
//! | [`is_alphabetic`] | `isAlphabetic(int)` | `Alphabetic` (`DerivedCoreProperties.txt`) |
//! | [`is_whitespace`] | `isWhitespace(int)` | `Zs`, `Zl` or `Zp` except U+00A0, U+2007 and U+202F; or U+0009 to U+000D; or U+001C to U+001F |
//! | [`is_space_char`] | `isSpaceChar(int)` | general category `Zs`, `Zl` or `Zp` |
//! | [`to_lower_case`] | `toLowerCase(int)` | the simple lowercase mapping, field 13 of `UnicodeData.txt` |
//! | [`script`] | `UnicodeScript.of(int)` | `Scripts.txt` |
//!
//! JDK 21, which OpenNLP's golden files are made with, implements Unicode
//! 15.0, and these tables Unicode 18.0. A character added in Unicode 15.1
//! or later is unassigned to that JDK and in no class, while here it is
//! classified by the 18.0 data. Where both assign a code point, the
//! answers differ for the code points listed in the `jdk` test of this
//! crate, all from changes Unicode made after 15.0: U+0295 is `Lo` here
//! and `Ll` in JDK 21 (a letter either way), and the combining Latin small
//! letters U+0363..U+036F and U+1DD3..U+1DE6 are alphabetic here and not
//! in JDK 21, and U+1171E AHOM CONSONANT SIGN MEDIAL RA is `Mc` here and
//! `Mn` in JDK 21 (changed in Unicode 16.0; alphabetic either way): 35
//! code points. With `TURBO_TEST_JDK` set, that test compares the general
//! category, the six class tests, the simple lowercase mapping and the
//! script of every code point from U+0000 to U+10FFFF with a running JDK
//! 21, skipping the code points the JDK reports as unassigned, and fails
//! on any other difference.

use crate::category::{ALL as CATEGORIES, GeneralCategory};
use crate::script::Script;
use crate::tables::case::{LOWER_KEYS, LOWER_VALUES};
use crate::tables::props::{ALPHABETIC, HAS_LOWER, UPPERCASE};
use crate::tables::{category, props, script as scripts};
use crate::trie::{find, lookup_code_point};

/// The general category: `Character.getType(int)` as an enum value.
#[inline]
pub fn general_category(cp: u32) -> GeneralCategory {
    CATEGORIES[lookup_code_point(&category::INDEX, &category::BLOCKS, cp) as usize]
}

/// `Character.getType(int)`: the general category in Java's numbering.
#[inline]
pub fn get_type(cp: u32) -> u8 {
    java_type(general_category(cp))
}

/// Java's number for a general category, the value of the matching
/// `Character` constant (`UPPERCASE_LETTER` is 1, `FINAL_QUOTE_PUNCTUATION`
/// is 30). Java does not use 17.
#[inline]
pub fn java_type(category: GeneralCategory) -> u8 {
    JAVA_TYPES[category as usize]
}

/// The general category with Java's number `java_type`, `None` for 17 and
/// for numbers above 30.
pub fn from_java_type(java_type: u8) -> Option<GeneralCategory> {
    JAVA_TYPES.iter().position(|&number| number == java_type).map(|i| CATEGORIES[i])
}

/// Java's number of each general category, in the order of
/// [`GeneralCategory`]'s variants (Lu, Ll, Lt, Lm, Lo, Mn, Mc, Me, Nd, Nl,
/// No, Pc, Pd, Ps, Pe, Pi, Pf, Po, Sm, Sc, Sk, So, Zs, Zl, Zp, Cc, Cf, Cs,
/// Co, Cn).
const JAVA_TYPES: [u8; 30] =
    [1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 23, 20, 21, 22, 29, 30, 24, 25, 26, 27, 28, 12, 13, 14, 15, 16, 19, 18, 0];

/// The property bits of `cp`, one per constant of `tables::props`.
#[inline]
fn bits(cp: u32) -> u32 {
    lookup_code_point(&props::INDEX, &props::BLOCKS, cp)
}

/// `Character.isLetter(int)`: general category `L*`.
#[inline]
pub fn is_letter(cp: u32) -> bool {
    if cp < 0x80 {
        return (cp as u8).is_ascii_alphabetic();
    }
    general_category(cp).is_letter()
}

/// `Character.isDigit(int)`: general category `Nd`. Narrower than Rust's
/// `char::is_numeric`, which also accepts `Nl` and `No`.
#[inline]
pub fn is_digit(cp: u32) -> bool {
    if cp < 0x80 {
        return (cp as u8).is_ascii_digit();
    }
    general_category(cp) == GeneralCategory::Nd
}

/// `Character.isLetterOrDigit(int)`.
#[inline]
pub fn is_letter_or_digit(cp: u32) -> bool {
    if cp < 0x80 {
        return (cp as u8).is_ascii_alphanumeric();
    }
    let category = general_category(cp);
    category.is_letter() || category == GeneralCategory::Nd
}

/// `Character.isUpperCase(int)`: general category `Lu` or the
/// `Other_Uppercase` property (Roman numerals, circled letters), which is
/// the `Uppercase` property.
#[inline]
pub fn is_upper_case(cp: u32) -> bool {
    if cp < 0x80 {
        return (cp as u8).is_ascii_uppercase();
    }
    bits(cp) & UPPERCASE != 0
}

/// `Character.isAlphabetic(int)`: the `Alphabetic` property.
#[inline]
pub fn is_alphabetic(cp: u32) -> bool {
    if cp < 0x80 {
        return (cp as u8).is_ascii_alphabetic();
    }
    bits(cp) & ALPHABETIC != 0
}

/// `Character.isWhitespace(int)`: space, line and paragraph separators
/// other than the no-break spaces, plus the ASCII controls U+0009 to
/// U+000D and U+001C to U+001F. U+0085 is not whitespace by this test.
#[inline]
pub fn is_whitespace(cp: u32) -> bool {
    match cp {
        0x20 | 0x09..=0x0D | 0x1C..=0x1F => true,
        0x00..=0x7F | 0xA0 | 0x2007 | 0x202F => false,
        _ => general_category(cp).is_separator(),
    }
}

/// `Character.getType(int) == Character.SPACE_SEPARATOR`: general
/// category `Zs`.
#[inline]
pub fn is_space_separator(cp: u32) -> bool {
    general_category(cp) == GeneralCategory::Zs
}

/// `Character.isSpaceChar(int)`: general category `Zs`, `Zl` or `Zp`.
#[inline]
pub fn is_space_char(cp: u32) -> bool {
    general_category(cp).is_separator()
}

/// `Character.toLowerCase(int)`: the simple one-to-one lowercase mapping,
/// or the code point itself when it has none. U+0130 maps to U+0069, as
/// in Java; the full mapping to "i" plus a combining dot is not used.
#[inline]
pub fn to_lower_case(cp: u32) -> u32 {
    if cp < 0x80 {
        return u32::from((cp as u8).to_ascii_lowercase());
    }
    if bits(cp) & HAS_LOWER == 0 {
        return cp;
    }
    find(&LOWER_KEYS, &LOWER_VALUES, cp).map_or(cp, u32::from)
}

/// `Character.UnicodeScript.of(int)`: the `Scripts.txt` value, with
/// [`Script::Unknown`] for unassigned code points and surrogates.
#[inline]
pub fn script(cp: u32) -> Script {
    scripts::ALL[lookup_code_point(&scripts::INDEX, &scripts::BLOCKS, cp) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_use_java_numbers() {
        use GeneralCategory::*;
        assert_eq!(java_type(Lu), 1);
        assert_eq!(java_type(Cc), 15);
        assert_eq!(java_type(Co), 18);
        assert_eq!(java_type(Cs), 19);
        assert_eq!(java_type(Pf), 30);
        assert_eq!(java_type(Cn), 0);
        assert_eq!(from_java_type(17), None);
        assert_eq!(from_java_type(31), None);
        let mut numbers: Vec<u8> = CATEGORIES.iter().map(|&category| java_type(category)).collect();
        numbers.sort_unstable();
        assert_eq!(numbers, (0..=30).filter(|&n| n != 17).collect::<Vec<u8>>());
        for category in CATEGORIES {
            assert_eq!(from_java_type(java_type(category)), Some(category));
        }
        assert_eq!(get_type('A' as u32), 1);
        assert_eq!(get_type(0xD800), 19);
    }

    #[test]
    fn general_category_values() {
        use GeneralCategory::*;
        let cases = [
            (0x0041, Lu),
            (0x0061, Ll),
            (0x01C5, Lt),
            (0x02B0, Lm),
            (0x4E2D, Lo),
            (0x0301, Mn),
            (0x0030, Nd),
            (0x2167, Nl),
            (0x00B2, No),
            (0x0020, Zs),
            (0x2028, Zl),
            (0x2029, Zp),
            (0x0009, Cc),
            (0x200B, Cf),
            (0xE000, Co),
            (0xD800, Cs),
            (0xDBFF, Cs),
            (0xDC00, Cs),
            (0xDFFF, Cs),
            (0x002D, Pd),
            (0x0028, Ps),
            (0x0029, Pe),
            (0x005F, Pc),
            (0x002E, Po),
            (0x002B, Sm),
            (0x0024, Sc),
            (0x005E, Sk),
            (0x1F600, So),
            (0x00AB, Pi),
            (0x00BB, Pf),
            (0x0378, Cn),
            (0x10FFFF, Cn),
            (0x110000, Cn),
            (u32::MAX, Cn),
        ];
        for (cp, expected) in cases {
            assert_eq!(general_category(cp), expected, "U+{cp:04X}");
        }
    }

    #[test]
    fn agrees_with_the_char_functions() {
        for c in (0..=0x10_FFFF).filter_map(char::from_u32) {
            let cp = c as u32;
            assert_eq!(general_category(cp), crate::category::general_category(c), "U+{cp:04X}");
            assert_eq!(is_letter(cp), crate::java::is_letter(c), "U+{cp:04X}");
            assert_eq!(is_digit(cp), crate::java::is_digit(c), "U+{cp:04X}");
            assert_eq!(is_letter_or_digit(cp), crate::java::is_letter_or_digit(c), "U+{cp:04X}");
            assert_eq!(is_whitespace(cp), crate::java::is_whitespace(c), "U+{cp:04X}");
            assert_eq!(is_space_char(cp), crate::java::is_space_char(c), "U+{cp:04X}");
            assert_eq!(is_upper_case(cp), crate::props::uppercase(c), "U+{cp:04X}");
            assert_eq!(is_alphabetic(cp), crate::props::alphabetic(c), "U+{cp:04X}");
            assert_eq!(to_lower_case(cp), crate::java::to_lower_char(c) as u32, "U+{cp:04X}");
            assert_eq!(script(cp), crate::script::script(c), "U+{cp:04X}");
        }
    }

    #[test]
    fn surrogates_and_values_beyond_the_code_space_are_in_no_class() {
        for cp in [0xD800, 0xDB7F, 0xDB80, 0xDBFF, 0xDC00, 0xDFFF, 0x110000, 0xFFFF_FFFF] {
            assert!(!is_letter(cp) && !is_digit(cp) && !is_letter_or_digit(cp), "U+{cp:04X}");
            assert!(!is_upper_case(cp) && !is_alphabetic(cp), "U+{cp:04X}");
            assert!(!is_whitespace(cp) && !is_space_char(cp) && !is_space_separator(cp), "U+{cp:04X}");
            assert_eq!(to_lower_case(cp), cp, "U+{cp:04X}");
            assert_eq!(script(cp), Script::Unknown, "U+{cp:04X}");
        }
    }

    #[test]
    fn letters_and_digits() {
        assert!(is_letter('A' as u32) && is_letter(0x00AA) && is_letter(0x4E2D) && is_letter(0x10400));
        assert!(!is_letter('1' as u32) && !is_letter(0x2167) && !is_letter(0x0301));
        assert!(is_digit('7' as u32) && is_digit(0x0660) && is_digit(0xFF10) && is_digit(0x1D7CE));
        // Superscript two is No and the Roman numeral eight is Nl: neither is a Java digit.
        assert!(!is_digit(0x00B2) && !is_digit(0x2167) && !is_digit('a' as u32));
        assert!(is_letter_or_digit('z' as u32) && is_letter_or_digit('0' as u32) && !is_letter_or_digit('_' as u32));
    }

    #[test]
    fn upper_case_and_alphabetic() {
        assert!(is_upper_case('A' as u32) && is_upper_case(0x10400));
        // Other_Uppercase: Roman numeral eight and circled capital A.
        assert!(is_upper_case(0x2167) && is_upper_case(0x24B6));
        // Title case is not upper case.
        assert!(!is_upper_case(0x01C5) && !is_upper_case('a' as u32));
        assert!(is_alphabetic('a' as u32) && is_alphabetic(0x2167) && is_alphabetic(0x0345));
        assert!(!is_alphabetic('1' as u32) && !is_alphabetic(0x0301));
    }

    #[test]
    fn java_whitespace() {
        for cp in [
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x1680, 0x2000, 0x200A, 0x2028, 0x2029, 0x205F,
            0x3000,
        ] {
            assert!(is_whitespace(cp), "U+{cp:04X}");
        }
        for cp in [0x00, 0x08, 0x0E, 0x85, 0xA0, 0x2007, 0x202F, 0x200B, 0xFEFF, 0x180E, 'a' as u32] {
            assert!(!is_whitespace(cp), "U+{cp:04X}");
        }
        assert!(is_space_separator(0xA0) && is_space_separator(0x2007) && is_space_separator(0x3000));
        assert!(!is_space_separator(0x2028) && !is_space_separator(0x09));
        assert!(is_space_char(0x2028) && is_space_char(0x2029) && is_space_char(0xA0) && !is_space_char(0x0A));
    }

    #[test]
    fn simple_lower_case() {
        let cases = [
            (0x0041, 0x0061),
            (0x0061, 0x0061),
            (0x00C0, 0x00E0),
            (0x0130, 0x0069),
            (0x0049, 0x0069),
            (0x01C5, 0x01C6),
            (0x0391, 0x03B1),
            (0x0410, 0x0430),
            (0x10A0, 0x2D00),
            (0x1E9E, 0x00DF),
            (0x212A, 0x006B),
            (0x2126, 0x03C9),
            (0x2167, 0x2177),
            (0x24B6, 0x24D0),
            (0xFF21, 0xFF41),
            (0x10400, 0x10428),
            (0x1E900, 0x1E922),
            (0x4E2D, 0x4E2D),
            (0x10FFFF, 0x10FFFF),
        ];
        for (cp, lower) in cases {
            assert_eq!(to_lower_case(cp), lower, "U+{cp:04X}");
        }
    }

    #[test]
    fn scripts() {
        let cases = [
            (0x0041, Script::Latin),
            (0x4E2D, Script::Han),
            (0x20000, Script::Han),
            (0x3042, Script::Hiragana),
            (0x30A2, Script::Katakana),
            // The prolonged sound mark is Common, not Katakana.
            (0x30FC, Script::Common),
            (0xD55C, Script::Hangul),
            (0x0E01, Script::Thai),
            (0x0E81, Script::Lao),
            (0x1000, Script::Myanmar),
            (0x1780, Script::Khmer),
            (0x1950, Script::TaiLe),
            (0x1980, Script::NewTaiLue),
            (0xAA80, Script::TaiViet),
            (0x0301, Script::Inherited),
            (0x3000, Script::Common),
            (0x0378, Script::Unknown),
        ];
        for (cp, expected) in cases {
            assert_eq!(script(cp), expected, "U+{cp:04X}");
        }
    }
}
