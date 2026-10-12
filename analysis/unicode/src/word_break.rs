//! The UAX #29 word boundary properties: `Word_Break` from the Unicode
//! 17.0 `WordBreakProperty.txt`, and `Extended_Pictographic` from the
//! Unicode 17.0 `emoji-data.txt`, which rule WB3c uses. These are the
//! same files OpenNLP's UAX #29 tokenizer bundles, so its boundaries and
//! the ones built on these tables come from the same data.
//!
//! Unlike the Java character classes, these properties do not depend on
//! the JDK: OpenNLP reads them from its own copy of the files.

use crate::tables::word_break as table;

pub use crate::tables::word_break::WordBreak;

/// The `Word_Break` value of a code point; [`WordBreak::Other`] for every
/// code point the property file does not list, surrogates and values
/// above U+10FFFF included.
#[inline]
pub fn word_break(cp: u32) -> WordBreak {
    table::word_break(cp)
}

/// Whether a code point has the `Extended_Pictographic` property.
#[inline]
pub fn is_extended_pictographic(cp: u32) -> bool {
    table::is_extended_pictographic(cp)
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
