//! What counts as whitespace: the Unicode `White_Space` property, and
//! Python's `str.isspace` for the stages that must match Python.
//!
//! Every annotator, tokenizer, reader and splitter in the workspace uses
//! [`is_whitespace`]. It is a fixed definition, not a setting: the 25 code
//! points of [`UNICODE_WHITESPACE`]. In particular U+0085 (next line) is
//! whitespace, and the information separators U+001C to U+001F are not.

/// Python's `str.isspace` and Unicode-pattern `\s`: Unicode `White_Space`
/// plus the four information separators U+001C through U+001F.
#[inline]
pub fn is_python_space(c: char) -> bool {
    if c.is_ascii() { matches!(c, '\u{09}'..='\u{0d}' | '\u{1c}'..='\u{20}') } else { crate::props::white_space(c) }
}

/// The code points with the Unicode `White_Space` property, in order, as
/// listed in `PropList.txt`.
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

/// Whether `cp` has the Unicode `White_Space` property. `cp` may be a
/// surrogate code unit, which is never whitespace.
#[inline]
pub const fn is_whitespace(cp: u32) -> bool {
    if cp < 0x80 {
        return is_ascii_whitespace(cp as u8);
    }
    matches!(cp, 0x0085 | 0x00A0 | 0x1680 | 0x2000..=0x200A | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000)
}

/// Whether the byte `byte` is an ASCII `White_Space` character: tab, line
/// feed, line tabulation, form feed, carriage return or space. A byte of
/// 0x80 or above is never ASCII whitespace.
#[inline]
pub const fn is_ascii_whitespace(byte: u8) -> bool {
    matches!(byte, 0x09..=0x0D | 0x20)
}

/// Whether every character of `text` is whitespace. Empty text is blank.
#[inline]
pub fn is_blank(text: &str) -> bool {
    text.chars().all(|c| is_whitespace(c as u32))
}

/// Borrows `text` without leading and trailing whitespace. Blank text
/// gives the empty slice at its end.
#[inline]
pub fn trim(text: &str) -> &str {
    text.trim_start_matches(|c: char| is_whitespace(c as u32)).trim_end_matches(|c: char| is_whitespace(c as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn next_line_is_whitespace_and_information_separators_are_not() {
        assert!(is_whitespace(0x85));
        for cp in 0x1C..=0x1F {
            assert!(!is_whitespace(cp), "U+{cp:04X}");
            assert!(!is_ascii_whitespace(cp as u8), "U+{cp:04X}");
        }
        // Format characters that look like spaces are not whitespace.
        for cp in [0x180E, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF] {
            assert!(!is_whitespace(cp), "U+{cp:04X}");
        }
        for cp in [0xD800, 0xDFFF, 'a' as u32, 0x110000] {
            assert!(!is_whitespace(cp), "U+{cp:04X}");
        }
    }

    #[test]
    fn ascii_table() {
        let ascii: Vec<u8> = (0u8..=0xFF).filter(|&b| is_ascii_whitespace(b)).collect();
        assert_eq!(ascii, [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20]);
    }

    #[test]
    fn whitespace_is_the_list() {
        for cp in 0..=0x10_FFFF {
            assert_eq!(is_whitespace(cp), UNICODE_WHITESPACE.contains(&cp), "U+{cp:04X}");
        }
    }

    #[test]
    fn the_list_matches_prop_list() {
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
}
