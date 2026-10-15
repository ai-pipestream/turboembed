//! The UAX #29 sentence boundary property `Sentence_Break`, from
//! `SentenceBreakProperty.txt` (Unicode 18.0). The rule sentence splitter
//! in `turbo-sentdetect` runs the UAX #29 sentence rules over it.

use crate::tables::sentence_break::{BLOCKS, INDEX, NAMES};
use crate::trie::lookup_code_point;

/// A UAX #29 `Sentence_Break` property value.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SentenceBreak {
    /// Every code point the property file does not list.
    #[default]
    Other = 0,
    /// Carriage return (U+000D).
    Cr = 1,
    /// Line feed (U+000A).
    Lf = 2,
    /// Combining marks and other characters that extend the one before.
    Extend = 3,
    /// The other paragraph separators: U+0085, U+2028, U+2029.
    Sep = 4,
    /// Format characters.
    Format = 5,
    /// Spaces other than the separators.
    Sp = 6,
    /// Lower case letters.
    Lower = 7,
    /// Upper and title case letters.
    Upper = 8,
    /// Other letters, such as CJK ideographs and letters of scripts without case.
    OLetter = 9,
    /// Digits.
    Numeric = 10,
    /// The ambiguous terminators: full stops that also end abbreviations.
    ATerm = 11,
    /// Punctuation that continues a sentence, such as the comma and colon.
    SContinue = 12,
    /// The other sentence terminators: `!`, `?`, danda, the ideographic full stop.
    STerm = 13,
    /// Closing and opening punctuation and quotation marks.
    Close = 14,
}

impl SentenceBreak {
    /// Every value, in discriminant order, which is also the order of the
    /// numbers the generated table stores.
    pub const ALL: [SentenceBreak; 15] = [
        SentenceBreak::Other,
        SentenceBreak::Cr,
        SentenceBreak::Lf,
        SentenceBreak::Extend,
        SentenceBreak::Sep,
        SentenceBreak::Format,
        SentenceBreak::Sp,
        SentenceBreak::Lower,
        SentenceBreak::Upper,
        SentenceBreak::OLetter,
        SentenceBreak::Numeric,
        SentenceBreak::ATerm,
        SentenceBreak::SContinue,
        SentenceBreak::STerm,
        SentenceBreak::Close,
    ];

    /// The value name as `SentenceBreakProperty.txt` writes it, such as `ATerm`.
    pub const fn property_name(self) -> &'static str {
        NAMES[self as usize]
    }

    /// The value with this `SentenceBreakProperty.txt` name.
    pub fn from_property_name(name: &str) -> Option<SentenceBreak> {
        NAMES.iter().position(|&n| n == name).map(|i| SentenceBreak::ALL[i])
    }
}

/// The `Sentence_Break` value of a code point; [`SentenceBreak::Other`]
/// for every code point the property file does not list, surrogates and
/// values above U+10FFFF included.
#[inline]
pub fn sentence_break(cp: u32) -> SentenceBreak {
    SentenceBreak::ALL[lookup_code_point(&INDEX, &BLOCKS, cp) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_follow_the_table_order() {
        assert_eq!(SentenceBreak::ALL.len(), NAMES.len());
        for (i, value) in SentenceBreak::ALL.into_iter().enumerate() {
            assert_eq!(value as usize, i);
            assert_eq!(SentenceBreak::from_property_name(value.property_name()), Some(value));
        }
        assert_eq!(SentenceBreak::from_property_name("Aterm"), None);
    }

    #[test]
    fn sentence_break_values() {
        use SentenceBreak::*;
        let cases = [
            (0x000D, Cr),
            (0x000A, Lf),
            (0x0085, Sep),
            (0x2028, Sep),
            (0x2029, Sep),
            (0x0020, Sp),
            (0x00A0, Sp),
            (0x0009, Sp),
            (0x0301, Extend),
            (0x00AD, Format),
            (0x0061, Lower),
            (0x0041, Upper),
            (0x4E2D, OLetter),
            (0x0030, Numeric),
            (0x002E, ATerm),
            (0x2024, ATerm),
            (0xFF0E, ATerm),
            (0x002C, SContinue),
            (0x003A, SContinue),
            (0x0021, STerm),
            (0x003F, STerm),
            (0x3002, STerm),
            (0x0964, STerm),
            (0x0022, Close),
            (0x0028, Close),
            (0x00AB, Close),
            (0x0024, Other),
            (0xD800, Other),
            (0x110000, Other),
        ];
        for (cp, expected) in cases {
            assert_eq!(sentence_break(cp), expected, "U+{cp:04X}");
        }
    }

    #[test]
    fn sentence_break_matches_the_property_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("ucd/SentenceBreakProperty.txt");
        let text = std::fs::read_to_string(&path).unwrap();
        let mut expected = vec![SentenceBreak::Other; 0x11_0000];
        for line in text.lines() {
            let content = line.split('#').next().unwrap().trim();
            if content.is_empty() {
                continue;
            }
            let (points, value) = content.split_once(';').unwrap();
            let (start, end) = points.trim().split_once("..").unwrap_or((points.trim(), points.trim()));
            let value = SentenceBreak::from_property_name(value.trim()).unwrap();
            let (start, end) = (u32::from_str_radix(start, 16).unwrap(), u32::from_str_radix(end, 16).unwrap());
            expected[start as usize..=end as usize].fill(value);
        }
        for (cp, &value) in expected.iter().enumerate() {
            assert_eq!(sentence_break(cp as u32), value, "U+{cp:04X}");
        }
    }
}
