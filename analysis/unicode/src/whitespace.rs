//! OpenNLP's two definitions of whitespace and Python's str.isspace.
//!
//! OpenNLP 3.x picks one process-wide through its `WhitespaceMode`
//! (`opennlp.tools.util.WhitespaceMode`, read by `StringUtil.isWhitespace`).
//! The choice changes tokens, trimmed spans and model features, so here
//! it is a value every annotator takes as a parameter, never a global.

use crate::java::code_point;

/// Python's `str.isspace` and Unicode-pattern `\s`: Unicode `White_Space`
/// plus the four information separators U+001C through U+001F. In
/// particular this includes U+0085, unlike OpenNLP's legacy mode.
#[inline]
pub fn is_python_space(c: char) -> bool {
    if c.is_ascii() { matches!(c, '\u{09}'..='\u{0d}' | '\u{1c}'..='\u{20}') } else { crate::props::white_space(c) }
}

/// Which characters count as whitespace.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum WhitespaceMode {
    /// OpenNLP 1.x and 2.x: `Character.isWhitespace` or general category
    /// `Zs`. This includes U+001C to U+001F and the no-break spaces, and
    /// leaves out U+0085.
    Legacy,
    /// The Unicode `White_Space` property, OpenNLP's default since 3.0:
    /// the 25 code points in [`UNICODE_WHITESPACE`].
    #[default]
    Unicode,
}

/// The code points with the Unicode `White_Space` property, in order.
/// The list is OpenNLP's `UnicodeWhitespace` table, which matches
/// `PropList.txt`.
pub const UNICODE_WHITESPACE: [u32; 25] = [
    0x0009, // character tabulation
    0x000A, // line feed
    0x000B, // line tabulation
    0x000C, // form feed
    0x000D, // carriage return
    0x0020, // space
    0x0085, // next line
    0x00A0, // no-break space
    0x1680, // ogham space mark
    0x2000, // en quad
    0x2001, // em quad
    0x2002, // en space
    0x2003, // em space
    0x2004, // three-per-em space
    0x2005, // four-per-em space
    0x2006, // six-per-em space
    0x2007, // figure space
    0x2008, // punctuation space
    0x2009, // thin space
    0x200A, // hair space
    0x2028, // line separator
    0x2029, // paragraph separator
    0x202F, // narrow no-break space
    0x205F, // medium mathematical space
    0x3000, // ideographic space
];

/// Builds the table of ASCII whitespace for one mode.
const fn ascii_table(legacy: bool) -> [bool; 128] {
    let mut table = [false; 128];
    let mut b = 0x09;
    while b <= 0x0D {
        table[b] = true;
        b += 1;
    }
    table[0x20] = true;
    if legacy {
        // Java's isWhitespace includes the file, group, record and unit separators.
        let mut b = 0x1C;
        while b <= 0x1F {
            table[b] = true;
            b += 1;
        }
    }
    table
}

const LEGACY_ASCII: [bool; 128] = ascii_table(true);
const UNICODE_ASCII: [bool; 128] = ascii_table(false);

impl WhitespaceMode {
    /// OpenNLP StringUtil's blank check in this mode. Empty text is blank.
    #[inline]
    pub fn is_blank(self, text: &str) -> bool {
        text.chars().all(|c| self.is_whitespace(c as u32))
    }

    /// Borrows `text` without leading and trailing whitespace in this mode.
    #[inline]
    pub fn trim(self, text: &str) -> &str {
        text.trim_start_matches(|c: char| self.is_whitespace(c as u32))
            .trim_end_matches(|c: char| self.is_whitespace(c as u32))
    }

    /// Whether `cp` is whitespace in this mode. `cp` may be a surrogate
    /// code unit, which is never whitespace. A code point below U+0080 is
    /// answered from a 128-entry table.
    #[inline]
    pub fn is_whitespace(self, cp: u32) -> bool {
        if cp < 0x80 {
            return self.is_ascii_whitespace(cp as u8);
        }
        match self {
            WhitespaceMode::Legacy => code_point::is_whitespace(cp) || code_point::is_space_separator(cp),
            WhitespaceMode::Unicode => {
                matches!(cp, 0x0085 | 0x00A0 | 0x1680 | 0x2000..=0x200A | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000)
            }
        }
    }

    /// Whether the ASCII byte `byte` is whitespace in this mode. A byte of
    /// 0x80 or above is never ASCII whitespace.
    #[inline]
    pub fn is_ascii_whitespace(self, byte: u8) -> bool {
        let table = match self {
            WhitespaceMode::Legacy => &LEGACY_ASCII,
            WhitespaceMode::Unicode => &UNICODE_ASCII,
        };
        byte < 0x80 && table[(byte & 0x7F) as usize]
    }
}

/// StringUtil.isUnicodeBlank as a function suitable for callbacks/re-exports.
#[inline]
pub fn is_unicode_blank(text: &str) -> bool {
    WhitespaceMode::Unicode.is_blank(text)
}

/// StringUtil.trimUnicodeWhitespace as a function suitable for callbacks.
#[inline]
pub fn trim_unicode_whitespace(text: &str) -> &str {
    WhitespaceMode::Unicode.trim(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use WhitespaceMode::{Legacy, Unicode};

    #[test]
    fn default_is_unicode() {
        assert_eq!(WhitespaceMode::default(), Unicode);
    }

    #[test]
    fn python_space_is_exactly_the_29_code_points() {
        // Independently enumerated by Python 3.14 str.isspace over all code points.
        let expected: Vec<_> = (0x09..=0x0d)
            .chain(0x1c..=0x20)
            .chain([0x85, 0xa0, 0x1680])
            .chain(0x2000..=0x200a)
            .chain([0x2028, 0x2029, 0x202f, 0x205f, 0x3000])
            .collect();
        assert_eq!(expected.len(), 29);
        for cp in 0..=0x10_ffff {
            let actual = char::from_u32(cp).is_some_and(is_python_space);
            assert_eq!(actual, expected.contains(&cp), "U+{cp:04X}");
        }
    }

    #[test]
    fn modes_differ_where_opennlp_says() {
        // Next line: White_Space, but Java's isWhitespace says no and it is Cc, not Zs.
        assert!(Unicode.is_whitespace(0x85) && !Legacy.is_whitespace(0x85));
        // The information separators are Java whitespace only.
        for cp in 0x1C..=0x1F {
            assert!(Legacy.is_whitespace(cp) && !Unicode.is_whitespace(cp));
        }
        // No-break spaces are Zs, so legacy mode includes them through the Zs test.
        for cp in [0xA0, 0x2007, 0x202F] {
            assert!(Legacy.is_whitespace(cp) && Unicode.is_whitespace(cp));
        }
        // Format characters that look like spaces are whitespace in neither mode.
        for cp in [0x180E, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF] {
            assert!(!Legacy.is_whitespace(cp) && !Unicode.is_whitespace(cp));
        }
        for cp in [0xD800, 0xDFFF, 'a' as u32, 0x110000] {
            assert!(!Legacy.is_whitespace(cp) && !Unicode.is_whitespace(cp));
        }
    }

    #[test]
    fn ascii_tables() {
        let unicode: Vec<u8> = (0u8..0x80).filter(|&b| Unicode.is_ascii_whitespace(b)).collect();
        assert_eq!(unicode, [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20]);
        let legacy: Vec<u8> = (0u8..0x80).filter(|&b| Legacy.is_ascii_whitespace(b)).collect();
        assert_eq!(legacy, [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20]);
        assert!(!Unicode.is_ascii_whitespace(0xA0) && !Legacy.is_ascii_whitespace(0x85));
    }

    #[test]
    fn unicode_mode_is_the_list() {
        for cp in 0..=0x10_FFFF {
            assert_eq!(Unicode.is_whitespace(cp), UNICODE_WHITESPACE.contains(&cp), "U+{cp:04X}");
        }
    }

    #[test]
    fn unicode_mode_matches_prop_list() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("ucd/PropList.txt");
        let text = std::fs::read_to_string(path).unwrap();
        let mut listed = Vec::new();
        for line in text.lines() {
            let content = line.split('#').next().unwrap();
            if let Some((points, "White_Space")) = content.split_once(';').map(|(p, v)| (p.trim(), v.trim())) {
                let (start, end) = points.split_once("..").unwrap_or((points, points));
                listed.extend(u32::from_str_radix(start, 16).unwrap()..=u32::from_str_radix(end, 16).unwrap());
            }
        }
        assert_eq!(listed, UNICODE_WHITESPACE);
    }

    #[test]
    fn legacy_mode_is_java_whitespace_or_zs() {
        for cp in 0..=0x10_FFFF {
            let expected = code_point::is_whitespace(cp) || code_point::is_space_separator(cp);
            assert_eq!(Legacy.is_whitespace(cp), expected, "U+{cp:04X}");
        }
        // The full legacy set, worked out from the Java definition.
        let legacy: Vec<u32> = (0..=0x10_FFFF).filter(|&cp| Legacy.is_whitespace(cp)).collect();
        let mut expected: Vec<u32> = (0x09..=0x0D).chain(0x1C..=0x20).collect();
        expected.extend([0xA0, 0x1680]);
        expected.extend(0x2000..=0x200A);
        expected.extend([0x2028, 0x2029, 0x202F, 0x205F, 0x3000]);
        assert_eq!(legacy, expected);
    }
}
