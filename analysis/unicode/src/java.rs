//! `java.lang.Character`'s classes, on a code point given as `u32`.
//!
//! OpenNLP classifies characters with `java.lang.Character`, and its
//! features and splits depend on the answers, so the ports ask the same
//! questions here. Each function names the Java method it reproduces.
//! A surrogate code unit (U+D800 to U+DFFF) is a valid argument: its
//! category is [`GeneralCategory::Surrogate`] and it is in no class, as in
//! Java, so a caller walking UTF-16 code units can classify each unit. A
//! value above U+10FFFF is treated as unassigned.
//!
//! The tables are built from the Unicode 17.0 character database, with
//! each Java class derived from it the way the JDK defines it:
//!
//! | Function | Java method | Definition from the UCD |
//! |---|---|---|
//! | [`general_category`] | `getType(int)` | `DerivedGeneralCategory.txt`, in Java's numbering |
//! | [`is_letter`] | `isLetter(int)` | general category `Lu`, `Ll`, `Lt`, `Lm` or `Lo` |
//! | [`is_digit`] | `isDigit(int)` | general category `Nd` |
//! | [`is_upper_case`] | `isUpperCase(int)` | general category `Lu`, or `Other_Uppercase` |
//! | [`is_alphabetic`] | `isAlphabetic(int)` | `Alphabetic` (`DerivedCoreProperties.txt`) |
//! | [`is_whitespace`] | `isWhitespace(int)` | `Zs`, `Zl` or `Zp` except U+00A0, U+2007 and U+202F; or U+0009 to U+000D; or U+001C to U+001F |
//! | [`is_space_char`] | `isSpaceChar(int)` | general category `Zs`, `Zl` or `Zp` |
//! | [`to_lower_case`] | `toLowerCase(int)` | the simple lowercase mapping, field 13 of `UnicodeData.txt` |
//! | [`script`] | `UnicodeScript.of(int)` | `Scripts.txt`, for the scripts in [`Script`] |
//!
//! JDK 21, which OpenNLP's golden files are made with, implements Unicode
//! 15.0. A character added in Unicode 15.1, 16.0 or 17.0 is unassigned to
//! that JDK and in no class, while here it is classified by the 17.0 data.
//! Where both assign a code point, the answers differ for 34 code points,
//! all from changes Unicode made after 15.0: U+0295 is `Lo` here and `Ll`
//! in JDK 21 (a letter either way), and the combining Latin small letters
//! U+0363..U+036F and U+1DD3..U+1DE6 are alphabetic here and not in JDK
//! 21. The `jdk` test compares every BMP code point and every 257th
//! above with a running JDK and fails on any other difference.

use crate::tables::java::{self as table, flag};

pub use crate::tables::java::{GeneralCategory, Script};

/// The general category: `Character.getType(int)`.
#[inline]
pub fn general_category(cp: u32) -> GeneralCategory {
    table::general_category(cp)
}

/// `Character.isLetter(int)`: general category `L*`.
#[inline]
pub fn is_letter(cp: u32) -> bool {
    table::flags(cp) & flag::LETTER != 0
}

/// `Character.isDigit(int)`: general category `Nd`. Narrower than Rust's
/// `char::is_numeric`, which also accepts `Nl` and `No`.
#[inline]
pub fn is_digit(cp: u32) -> bool {
    table::flags(cp) & flag::DIGIT != 0
}

/// `Character.isLetterOrDigit(int)`.
#[inline]
pub fn is_letter_or_digit(cp: u32) -> bool {
    table::flags(cp) & (flag::LETTER | flag::DIGIT) != 0
}

/// `Character.isUpperCase(int)`: general category `Lu` or the
/// `Other_Uppercase` property (Roman numerals, circled letters).
#[inline]
pub fn is_upper_case(cp: u32) -> bool {
    table::flags(cp) & flag::UPPER_CASE != 0
}

/// `Character.isAlphabetic(int)`: the `Alphabetic` property.
#[inline]
pub fn is_alphabetic(cp: u32) -> bool {
    table::flags(cp) & flag::ALPHABETIC != 0
}

/// `Character.isWhitespace(int)`: space, line and paragraph separators
/// other than the no-break spaces, plus the ASCII controls U+0009 to
/// U+000D and U+001C to U+001F. U+0085 is not whitespace by this test.
#[inline]
pub fn is_whitespace(cp: u32) -> bool {
    table::flags(cp) & flag::WHITESPACE != 0
}

/// `Character.getType(int) == Character.SPACE_SEPARATOR`: general
/// category `Zs`.
#[inline]
pub fn is_space_separator(cp: u32) -> bool {
    general_category(cp) == GeneralCategory::SpaceSeparator
}

/// `Character.isSpaceChar(int)`: general category `Zs`, `Zl` or `Zp`.
#[inline]
pub fn is_space_char(cp: u32) -> bool {
    matches!(
        general_category(cp),
        GeneralCategory::SpaceSeparator | GeneralCategory::LineSeparator | GeneralCategory::ParagraphSeparator
    )
}

/// `Character.toLowerCase(int)`: the simple one-to-one lowercase mapping,
/// or the code point itself when it has none. U+0130 maps to U+0069, as
/// in Java; the full mapping to "i" plus a combining dot is not used.
#[inline]
pub fn to_lower_case(cp: u32) -> u32 {
    table::to_lower_case(cp)
}

/// `Character.UnicodeScript.of(int)`, reduced to the scripts the UAX #29
/// word type distinguishes; every other script is [`Script::Other`].
#[inline]
pub fn script(cp: u32) -> Script {
    table::script(cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_use_java_numbers() {
        assert_eq!(GeneralCategory::ALL.len(), 30);
        assert_eq!(GeneralCategory::UppercaseLetter.java_type(), 1);
        assert_eq!(GeneralCategory::Control.java_type(), 15);
        assert_eq!(GeneralCategory::PrivateUse.java_type(), 18);
        assert_eq!(GeneralCategory::FinalQuotePunctuation.java_type(), 30);
        assert_eq!(GeneralCategory::from_java_type(17), None);
        assert_eq!(GeneralCategory::from_java_type(31), None);
        for category in GeneralCategory::ALL {
            assert_eq!(GeneralCategory::from_java_type(category.java_type()), Some(category));
            assert_eq!(GeneralCategory::from_abbreviation(category.abbreviation()), Some(category));
        }
    }

    #[test]
    fn general_category_values() {
        use GeneralCategory::*;
        let cases = [
            (0x0041, UppercaseLetter),
            (0x0061, LowercaseLetter),
            (0x01C5, TitlecaseLetter),
            (0x02B0, ModifierLetter),
            (0x4E2D, OtherLetter),
            (0x0301, NonSpacingMark),
            (0x0030, DecimalDigitNumber),
            (0x2167, LetterNumber),
            (0x00B2, OtherNumber),
            (0x0020, SpaceSeparator),
            (0x2028, LineSeparator),
            (0x2029, ParagraphSeparator),
            (0x0009, Control),
            (0x200B, Format),
            (0xE000, PrivateUse),
            (0xD800, Surrogate),
            (0xDFFF, Surrogate),
            (0x002D, DashPunctuation),
            (0x0028, StartPunctuation),
            (0x0029, EndPunctuation),
            (0x005F, ConnectorPunctuation),
            (0x002E, OtherPunctuation),
            (0x002B, MathSymbol),
            (0x0024, CurrencySymbol),
            (0x005E, ModifierSymbol),
            (0x1F600, OtherSymbol),
            (0x00AB, InitialQuotePunctuation),
            (0x00BB, FinalQuotePunctuation),
            (0x0378, Unassigned),
            (0x10FFFF, Unassigned),
            (0x110000, Unassigned),
            (u32::MAX, Unassigned),
        ];
        for (cp, expected) in cases {
            assert_eq!(general_category(cp), expected, "U+{cp:04X}");
        }
    }

    #[test]
    fn letters_and_digits() {
        assert!(is_letter('A' as u32) && is_letter(0x00AA) && is_letter(0x4E2D) && is_letter(0x10400));
        assert!(!is_letter('1' as u32) && !is_letter(0x2167) && !is_letter(0x0301) && !is_letter(0xD800));
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
        assert!(!is_upper_case(0x01C5) && !is_upper_case('a' as u32) && !is_upper_case(0xD801));
        assert!(is_alphabetic('a' as u32) && is_alphabetic(0x2167) && is_alphabetic(0x0345));
        assert!(!is_alphabetic('1' as u32) && !is_alphabetic(0x0301) && !is_alphabetic(0xDC00));
    }

    #[test]
    fn java_whitespace() {
        for cp in [
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x1680, 0x2000, 0x200A, 0x2028, 0x2029, 0x205F,
            0x3000,
        ] {
            assert!(is_whitespace(cp), "U+{cp:04X}");
        }
        for cp in [0x00, 0x08, 0x0E, 0x85, 0xA0, 0x2007, 0x202F, 0x200B, 0xFEFF, 0x180E, 'a' as u32, 0xD800] {
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
            (0xD800, 0xD800),
            (0x10FFFF, 0x10FFFF),
        ];
        for (cp, lower) in cases {
            assert_eq!(to_lower_case(cp), lower, "U+{cp:04X}");
        }
    }

    #[test]
    fn scripts() {
        let cases = [
            (0x0041, Script::Other),
            (0x4E2D, Script::Han),
            (0x20000, Script::Han),
            (0x3042, Script::Hiragana),
            (0x30A2, Script::Katakana),
            // The prolonged sound mark is Common, not Katakana.
            (0x30FC, Script::Other),
            (0xD55C, Script::Hangul),
            (0x0E01, Script::Thai),
            (0x0E81, Script::Lao),
            (0x1000, Script::Myanmar),
            (0x1780, Script::Khmer),
            (0x1950, Script::TaiLe),
            (0x1980, Script::NewTaiLue),
            (0xAA80, Script::TaiViet),
            (0x3000, Script::Other),
        ];
        for (cp, expected) in cases {
            assert_eq!(script(cp), expected, "U+{cp:04X}");
        }
        assert_eq!(Script::TaiLe.java_name(), "TAI_LE");
        assert_eq!(Script::from_property_name("New_Tai_Lue"), Some(Script::NewTaiLue));
        assert_eq!(Script::from_property_name("Latin"), None);
    }
}
