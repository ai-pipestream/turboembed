//! The escaping of a `str` field. A field never holds a tab or a line end,
//! since those separate fields and rows, so the writer escapes them:
//!
//! - `\t`, `\n`, `\r` and `\\` for a tab, a line feed, a carriage return
//!   and a backslash;
//! - `\uXXXX`, four lowercase hex digits naming one UTF-16 unit, for every
//!   other character below U+0020.
//!
//! Nothing else is escaped: every other character, a character outside
//! the Basic Multilingual Plane included, is written as its UTF-8 bytes.
//! The reader accepts `\uXXXX` for any UTF-16 unit, and a character outside
//! the BMP written as two of them (a high surrogate, then a low one) is
//! that one character. A lone surrogate is refused, as is a raw character
//! below U+0020, which a conforming writer never leaves unescaped.

use std::fmt;

use crate::hex::digit;

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Appends `s` to `out` with the escapes above.
pub fn escape_into(s: &str, out: &mut String) {
    if !s.bytes().any(|b| b < 0x20 || b == b'\\') {
        out.push_str(s);
        return;
    }
    for c in s.chars() {
        match c {
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let b = c as u32 as u8;
                out.push_str("\\u00");
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 15) as usize] as char);
            }
            c => out.push(c),
        }
    }
}

/// `s` with the escapes above.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    escape_into(s, &mut out);
    out
}

/// Appends the text `field` stands for to `out`. On an error `out` holds
/// what was decoded before it.
pub fn unescape_into(field: &str, out: &mut String) -> Result<(), EscapeError> {
    let b = field.as_bytes();
    let mut run = 0;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c >= 0x20 && c != b'\\' {
            i += 1;
            continue;
        }
        if c != b'\\' {
            return Err(EscapeError { at: i, kind: EscapeErrorKind::Raw(c as char) });
        }
        out.push_str(&field[run..i]);
        let (decoded, used) = escape_at(field, i)?;
        out.push(decoded);
        i += used;
        run = i;
    }
    out.push_str(&field[run..]);
    Ok(())
}

/// The text `field` stands for.
pub fn unescape(field: &str) -> Result<String, EscapeError> {
    let mut out = String::with_capacity(field.len());
    unescape_into(field, &mut out)?;
    Ok(out)
}

/// The UTF-16 units `field` stands for. Unlike `unescape`, it takes a
/// lone surrogate written as `\uXXXX`: a decision point dump's contexts
/// are cut at UTF-16 units and can hold half of a pair.
pub fn unescape_utf16(field: &str) -> Result<Vec<u16>, EscapeError> {
    let b = field.as_bytes();
    let mut out = Vec::with_capacity(field.len());
    let mut run = 0;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c >= 0x20 && c != b'\\' {
            i += 1;
            continue;
        }
        if c != b'\\' {
            return Err(EscapeError { at: i, kind: EscapeErrorKind::Raw(c as char) });
        }
        out.extend(field[run..i].encode_utf16());
        let unit = match b.get(i + 1) {
            None => return Err(EscapeError { at: i, kind: EscapeErrorKind::Trailing }),
            Some(b't') => u16::from(b'\t'),
            Some(b'n') => u16::from(b'\n'),
            Some(b'r') => u16::from(b'\r'),
            Some(b'\\') => u16::from(b'\\'),
            Some(b'u') => unit_at(b, i).ok_or(EscapeError { at: i, kind: EscapeErrorKind::Unit })?,
            Some(_) => {
                let c = field[i + 1..].chars().next().expect("a character after the backslash");
                return Err(EscapeError { at: i, kind: EscapeErrorKind::Unknown(c) });
            }
        };
        out.push(unit);
        i += if b[i + 1] == b'u' { 6 } else { 2 };
        run = i;
    }
    out.extend(field[run..].encode_utf16());
    Ok(out)
}

/// The character the escape starting at byte `at` (a backslash) stands
/// for, and how many bytes the escape takes.
fn escape_at(field: &str, at: usize) -> Result<(char, usize), EscapeError> {
    let b = field.as_bytes();
    let err = |kind| Err(EscapeError { at, kind });
    match b.get(at + 1) {
        None => err(EscapeErrorKind::Trailing),
        Some(b't') => Ok(('\t', 2)),
        Some(b'n') => Ok(('\n', 2)),
        Some(b'r') => Ok(('\r', 2)),
        Some(b'\\') => Ok(('\\', 2)),
        Some(b'u') => {
            let Some(unit) = unit_at(b, at) else { return err(EscapeErrorKind::Unit) };
            match unit {
                0xD800..=0xDBFF => {
                    if b.get(at + 6) != Some(&b'\\') || b.get(at + 7) != Some(&b'u') {
                        return err(EscapeErrorKind::LoneSurrogate(unit));
                    }
                    let Some(low) = unit_at(b, at + 6) else {
                        return Err(EscapeError { at: at + 6, kind: EscapeErrorKind::Unit });
                    };
                    if !(0xDC00..=0xDFFF).contains(&low) {
                        return err(EscapeErrorKind::LoneSurrogate(unit));
                    }
                    let c = 0x10000 + ((unit as u32 - 0xD800) << 10) + (low as u32 - 0xDC00);
                    Ok((char::from_u32(c).expect("a surrogate pair is a character"), 12))
                }
                0xDC00..=0xDFFF => err(EscapeErrorKind::LoneSurrogate(unit)),
                _ => Ok((char::from_u32(unit as u32).expect("a unit outside the surrogates is a character"), 6)),
            }
        }
        Some(_) => {
            let c = field[at + 1..].chars().next().expect("a character after the backslash");
            err(EscapeErrorKind::Unknown(c))
        }
    }
}

/// The four lowercase hex digits after `\u` at `at`.
fn unit_at(b: &[u8], at: usize) -> Option<u16> {
    let digits = b.get(at + 2..at + 6)?;
    digits.iter().try_fold(0u16, |v, &d| Some((v << 4) | digit(d)? as u16))
}

/// Why a `str` field could not be decoded, and where: `at` is the byte
/// offset in the field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EscapeError {
    pub at: usize,
    pub kind: EscapeErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EscapeErrorKind {
    /// A backslash at the end of the field.
    Trailing,
    /// A backslash before a character that starts no escape.
    Unknown(char),
    /// `\u` not followed by four lowercase hex digits.
    Unit,
    /// A surrogate that is not half of a high-then-low pair.
    LoneSurrogate(u16),
    /// A character below U+0020 that is not escaped.
    Raw(char),
}

impl fmt::Display for EscapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {} of the field: ", self.at)?;
        match self.kind {
            EscapeErrorKind::Trailing => f.write_str("a backslash ends the field, expected an escape after it"),
            EscapeErrorKind::Unknown(c) => {
                write!(f, "expected one of \\t \\n \\r \\\\ \\uXXXX, found \\{}", c.escape_debug())
            }
            EscapeErrorKind::Unit => f.write_str("expected four lowercase hex digits after \\u"),
            EscapeErrorKind::LoneSurrogate(u) => {
                write!(f, "\\u{u:04x} is a surrogate without its other half, expected a high then a low surrogate")
            }
            EscapeErrorKind::Raw(c) => write!(f, "U+{:04X} must be escaped", c as u32),
        }
    }
}

impl std::error::Error for EscapeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_control_character_is_escaped_and_comes_back() {
        for c in (0u32..0x20).map(|u| char::from_u32(u).unwrap()) {
            let s = format!("a{c}b");
            let e = escape(&s);
            let expected = match c {
                '\t' => "a\\tb".to_string(),
                '\n' => "a\\nb".to_string(),
                '\r' => "a\\rb".to_string(),
                c => format!("a\\u{:04x}b", c as u32),
            };
            assert_eq!(e, expected, "U+{:04X}", c as u32);
            assert!(!e.bytes().any(|b| b < 0x20), "no raw control character left for U+{:04X}", c as u32);
            assert_eq!(unescape(&e).unwrap(), s, "U+{:04X}", c as u32);
        }
    }

    #[test]
    fn backslash_and_plain_text() {
        assert_eq!(escape("a\\b"), "a\\\\b");
        assert_eq!(unescape("a\\\\b").unwrap(), "a\\b");
        assert_eq!(escape("\\u0041"), "\\\\u0041");
        assert_eq!(unescape("\\\\u0041").unwrap(), "\\u0041");
        // DEL and every character from U+0020 up are written as they are.
        let plain = "Zürich \u{7f} \u{a0} \u{2028} 東京 😀";
        assert_eq!(escape(plain), plain);
        assert_eq!(unescape(plain).unwrap(), plain);
        assert_eq!(escape(""), "");
        assert_eq!(unescape("").unwrap(), "");
    }

    #[test]
    fn a_surrogate_pair_is_one_character() {
        // U+1F600 is D83D DE00 in UTF-16.
        assert_eq!(unescape("a\\ud83d\\ude00b").unwrap(), "a😀b");
        // The writer leaves it as UTF-8, and that reads back too.
        assert_eq!(escape("a😀b"), "a😀b");
        assert_eq!(unescape(&escape("a😀b")).unwrap(), "a😀b");
        // The highest pair, U+10FFFF.
        assert_eq!(unescape("\\udbff\\udfff").unwrap(), "\u{10FFFF}");
        // A BMP unit written as an escape is that character.
        assert_eq!(unescape("\\u00e9").unwrap(), "é");
    }

    #[test]
    fn utf16_units_keep_a_lone_surrogate() {
        let units = |s: &str| unescape_utf16(s).unwrap();
        assert_eq!(units("a\\ud83d"), vec![0x61, 0xd83d]);
        assert_eq!(units("\\ude00b"), vec![0xde00, 0x62]);
        assert_eq!(units("a\\ud83d\\ude00"), "a😀".encode_utf16().collect::<Vec<_>>());
        assert_eq!(units("😀\\t\\\\\\n\\r\\u001f"), "😀\t\\\n\r\u{1f}".encode_utf16().collect::<Vec<_>>());
        assert_eq!(units(""), Vec::<u16>::new());
        // What `unescape` takes, this gives as UTF-16.
        for s in ["Zürich \\u0000 東京", "a\\ud83d\\ude00b"] {
            assert_eq!(units(s), unescape(s).unwrap().encode_utf16().collect::<Vec<_>>(), "{s:?}");
        }
        let err = |s: &str| unescape_utf16(s).unwrap_err();
        assert_eq!(err("ab\\"), EscapeError { at: 2, kind: EscapeErrorKind::Trailing });
        assert_eq!(err("a\\x"), EscapeError { at: 1, kind: EscapeErrorKind::Unknown('x') });
        assert_eq!(err("\\uD83D"), EscapeError { at: 0, kind: EscapeErrorKind::Unit });
        assert_eq!(err("a\tb"), EscapeError { at: 1, kind: EscapeErrorKind::Raw('\t') });
    }

    #[test]
    fn malformed_escapes_are_refused_where_they_are() {
        let kind = |s: &str| unescape(s).unwrap_err();
        assert_eq!(kind("ab\\"), EscapeError { at: 2, kind: EscapeErrorKind::Trailing });
        assert_eq!(kind("a\\x"), EscapeError { at: 1, kind: EscapeErrorKind::Unknown('x') });
        assert_eq!(kind("\\é"), EscapeError { at: 0, kind: EscapeErrorKind::Unknown('é') });
        assert_eq!(kind("\\u00E9"), EscapeError { at: 0, kind: EscapeErrorKind::Unit });
        assert_eq!(kind("\\u00e"), EscapeError { at: 0, kind: EscapeErrorKind::Unit });
        assert_eq!(kind("x\\ud83d"), EscapeError { at: 1, kind: EscapeErrorKind::LoneSurrogate(0xd83d) });
        assert_eq!(kind("\\ud83dx"), EscapeError { at: 0, kind: EscapeErrorKind::LoneSurrogate(0xd83d) });
        assert_eq!(kind("\\ud83d\\u0041"), EscapeError { at: 0, kind: EscapeErrorKind::LoneSurrogate(0xd83d) });
        assert_eq!(kind("\\ude00"), EscapeError { at: 0, kind: EscapeErrorKind::LoneSurrogate(0xde00) });
        assert_eq!(kind("\\ud83d\\uzz00"), EscapeError { at: 6, kind: EscapeErrorKind::Unit });
        assert_eq!(kind("a\rb"), EscapeError { at: 1, kind: EscapeErrorKind::Raw('\r') });
        let shown = kind("a\\x").to_string();
        assert!(shown.contains("at byte 1") && shown.contains("found \\x"), "{shown}");
    }
}
