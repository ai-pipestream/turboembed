//! The UAX #29 word boundary properties: `Word_Break` from
//! `WordBreakProperty.txt`, and `Extended_Pictographic` from
//! `emoji-data.txt`, which rule WB3c uses.
//!
//! OpenNLP 3.x's UAX #29 tokenizer bundles its own
//! `WordBreakProperty-17.0.0.txt`; the tables here are built from the
//! Unicode 18.0 files in `ucd/`. The two differ on 558 code points, all
//! newly assigned in Unicode 18.0 and `Other` in the 17.0 file: 515 are
//! `ALetter` here, 37 `Extend` and 6 `Katakana`. On every other code point
//! the `Word_Break` values are the same, so a golden file mismatch on text
//! with a character new in 18.0 is expected. Unlike the Java character
//! classes, these properties do not depend on the JDK.

use crate::tables::props::{self, EXTENDED_PICTOGRAPHIC};
use crate::tables::word_break::{BLOCKS, INDEX, NAMES};
use crate::trie::lookup_code_point;

/// A UAX #29 `Word_Break` property value, in the order of OpenNLP's
/// `WordBreak` enum.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WordBreak {
    /// No `Word_Break` value: every code point the property file does not list.
    #[default]
    Other = 0,
    /// Carriage return (U+000D).
    Cr = 1,
    /// Line feed (U+000A).
    Lf = 2,
    /// The other mandatory breaks: U+000B, U+000C, U+0085, U+2028, U+2029.
    Newline = 3,
    /// Combining marks and other characters that extend the one before.
    Extend = 4,
    /// Zero width joiner (U+200D).
    Zwj = 5,
    /// Regional indicator symbols, used in pairs for flags.
    RegionalIndicator = 6,
    /// Format characters.
    Format = 7,
    /// Katakana.
    Katakana = 8,
    /// Hebrew letters.
    HebrewLetter = 9,
    /// Alphabetic letters.
    ALetter = 10,
    /// The apostrophe (U+0027).
    SingleQuote = 11,
    /// The quotation mark (U+0022).
    DoubleQuote = 12,
    /// Characters that join letters or numbers, such as the full stop.
    MidNumLet = 13,
    /// Characters that join letters, such as the middle dot.
    MidLetter = 14,
    /// Characters that join numbers, such as the comma.
    MidNum = 15,
    /// Digits.
    Numeric = 16,
    /// Characters that extend letters or numbers, such as the low line.
    ExtendNumLet = 17,
    /// Spaces that separate words.
    WSegSpace = 18,
}

impl WordBreak {
    /// Every value, in discriminant order, which is also the order of the
    /// numbers the generated table stores.
    pub const ALL: [WordBreak; 19] = [
        WordBreak::Other,
        WordBreak::Cr,
        WordBreak::Lf,
        WordBreak::Newline,
        WordBreak::Extend,
        WordBreak::Zwj,
        WordBreak::RegionalIndicator,
        WordBreak::Format,
        WordBreak::Katakana,
        WordBreak::HebrewLetter,
        WordBreak::ALetter,
        WordBreak::SingleQuote,
        WordBreak::DoubleQuote,
        WordBreak::MidNumLet,
        WordBreak::MidLetter,
        WordBreak::MidNum,
        WordBreak::Numeric,
        WordBreak::ExtendNumLet,
        WordBreak::WSegSpace,
    ];

    /// The value name as `WordBreakProperty.txt` writes it, such as `ALetter`.
    pub const fn property_name(self) -> &'static str {
        NAMES[self as usize]
    }

    /// The value with this `WordBreakProperty.txt` name.
    pub fn from_property_name(name: &str) -> Option<WordBreak> {
        NAMES.iter().position(|&n| n == name).map(|i| WordBreak::ALL[i])
    }
}

/// The `Word_Break` value of a code point; [`WordBreak::Other`] for every
/// code point the property file does not list, surrogates and values
/// above U+10FFFF included.
#[inline]
pub fn word_break(cp: u32) -> WordBreak {
    WordBreak::ALL[lookup_code_point(&INDEX, &BLOCKS, cp) as usize]
}

/// Whether a code point has the `Extended_Pictographic` property; false
/// for surrogates and values above U+10FFFF.
#[inline]
pub fn is_extended_pictographic(cp: u32) -> bool {
    lookup_code_point(&props::INDEX, &props::BLOCKS, cp) & EXTENDED_PICTOGRAPHIC != 0
}

/// Whether a code point is a regional indicator symbol (U+1F1E6 to
/// U+1F1FF), two of which make a flag.
#[inline]
pub fn is_regional_indicator(cp: u32) -> bool {
    (0x1F1E6..=0x1F1FF).contains(&cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ucd_file(name: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("ucd").join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    /// The `(start, end, value)` ranges of a UCD property file.
    fn ranges(text: &str) -> Vec<(u32, u32, String)> {
        let mut out = Vec::new();
        for line in text.lines() {
            let content = line.split('#').next().unwrap().trim();
            if content.is_empty() {
                continue;
            }
            let (points, value) = content.split_once(';').unwrap();
            let value = value.split(';').next().unwrap().trim().to_string();
            let (start, end) = points.trim().split_once("..").unwrap_or((points.trim(), points.trim()));
            out.push((u32::from_str_radix(start, 16).unwrap(), u32::from_str_radix(end, 16).unwrap(), value));
        }
        out
    }

    #[test]
    fn values_follow_opennlp_order() {
        assert_eq!(WordBreak::ALL.len(), 19);
        assert_eq!(WordBreak::ALL.len(), NAMES.len());
        assert_eq!(WordBreak::Other as u8, 0);
        assert_eq!(WordBreak::ALetter as u8, 10);
        assert_eq!(WordBreak::WSegSpace as u8, 18);
        for (i, value) in WordBreak::ALL.into_iter().enumerate() {
            assert_eq!(value as usize, i);
            assert_eq!(WordBreak::from_property_name(value.property_name()), Some(value));
        }
        assert_eq!(WordBreak::from_property_name("Hebrew_Letter"), Some(WordBreak::HebrewLetter));
        assert_eq!(WordBreak::from_property_name("Aletter"), None);
    }

    #[test]
    fn word_break_values() {
        use WordBreak::*;
        let cases = [
            (0x000D, Cr),
            (0x000A, Lf),
            (0x000B, Newline),
            (0x0085, Newline),
            (0x2028, Newline),
            (0x0301, Extend),
            (0x200D, Zwj),
            (0x1F1E6, RegionalIndicator),
            (0x00AD, Format),
            (0x30A2, Katakana),
            (0x05D0, HebrewLetter),
            (0x0041, ALetter),
            (0x0027, SingleQuote),
            (0x0022, DoubleQuote),
            (0x002E, MidNumLet),
            (0x003A, MidLetter),
            (0x002C, MidNum),
            (0x0030, Numeric),
            (0x005F, ExtendNumLet),
            (0x0020, WSegSpace),
            (0x3000, WSegSpace),
            (0x4E2D, Other),
            (0x3042, Other),
            (0xD800, Other),
            (0x110000, Other),
        ];
        for (cp, expected) in cases {
            assert_eq!(word_break(cp), expected, "U+{cp:04X}");
        }
    }

    #[test]
    fn word_break_matches_the_property_file() {
        let mut expected = vec![WordBreak::Other; 0x11_0000];
        for (start, end, value) in ranges(&ucd_file("WordBreakProperty.txt")) {
            let value = WordBreak::from_property_name(&value).unwrap();
            expected[start as usize..=end as usize].fill(value);
        }
        for (cp, &value) in expected.iter().enumerate() {
            assert_eq!(word_break(cp as u32), value, "U+{cp:04X}");
        }
    }

    #[test]
    fn extended_pictographic_matches_emoji_data() {
        let mut expected = vec![false; 0x11_0000];
        for (start, end, value) in ranges(&ucd_file("emoji-data.txt")) {
            if value == "Extended_Pictographic" {
                expected[start as usize..=end as usize].fill(true);
            }
        }
        for (cp, &value) in expected.iter().enumerate() {
            assert_eq!(is_extended_pictographic(cp as u32), value, "U+{cp:04X}");
        }
        assert!(is_extended_pictographic(0x00A9) && is_extended_pictographic(0x1F600));
        assert!(!is_extended_pictographic('a' as u32) && !is_extended_pictographic(0x1F1E6));
        assert!(!is_extended_pictographic(0x110000));
    }

    #[test]
    fn regional_indicators() {
        assert!(is_regional_indicator(0x1F1E6) && is_regional_indicator(0x1F1FF));
        assert!(!is_regional_indicator(0x1F1E5) && !is_regional_indicator(0x1F200));
    }
}
