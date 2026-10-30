//! UTF-8 decoding and small ASCII parsers used by MASC and brat readers.

const REPLACEMENT: char = '\u{FFFD}';

/// Decodes as the JDK's UTF-8 reader does: each malformed sequence becomes
/// one U+FFFD, preserving the annotation's UTF-16 offset boundaries.
pub(crate) fn decode_utf8(bytes: &[u8]) -> String {
    let cont = |b: u8| b & 0xC0 == 0x80;
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b1 = bytes[i];
        let rest = &bytes[i + 1..];
        let (c, len) = match b1 {
            0x00..=0x7F => (char::from(b1), 1),
            0xC2..=0xDF => match rest.first() {
                None => (REPLACEMENT, 1),
                Some(&b2) if !cont(b2) => (REPLACEMENT, 1),
                Some(&b2) => (char::from_u32(u32::from(b1 & 0x1F) << 6 | u32::from(b2 & 0x3F)).unwrap(), 2),
            },
            0xE0..=0xEF => {
                let overlong = |b2: u8| b1 == 0xE0 && b2 & 0xE0 == 0x80;
                match *rest {
                    [b2, _, ..] if overlong(b2) || !cont(b2) => (REPLACEMENT, 1),
                    [_, b3, ..] if !cont(b3) => (REPLACEMENT, 2),
                    [b2, b3, ..] => {
                        let u = u32::from(b1 & 0x0F) << 12 | u32::from(b2 & 0x3F) << 6 | u32::from(b3 & 0x3F);
                        (char::from_u32(u).unwrap_or(REPLACEMENT), 3)
                    }
                    [b2] if overlong(b2) || !cont(b2) => (REPLACEMENT, 1),
                    _ => (REPLACEMENT, 1 + rest.len()),
                }
            }
            0xF0..=0xF7 => {
                let second_bad = |b2: u8| {
                    b1 > 0xF4
                        || b1 == 0xF0 && !(0x90..=0xBF).contains(&b2)
                        || b1 == 0xF4 && b2 & 0xF0 != 0x80
                        || !cont(b2)
                };
                match *rest {
                    [b2, b3, b4, ..] => {
                        let u = u32::from(b1 & 0x07) << 18
                            | u32::from(b2 & 0x3F) << 12
                            | u32::from(b3 & 0x3F) << 6
                            | u32::from(b4 & 0x3F);
                        if cont(b2) && cont(b3) && cont(b4) && (0x1_0000..=0x10_FFFF).contains(&u) {
                            (char::from_u32(u).unwrap(), 4)
                        } else if second_bad(b2) {
                            (REPLACEMENT, 1)
                        } else if !cont(b3) {
                            (REPLACEMENT, 2)
                        } else {
                            (REPLACEMENT, 3)
                        }
                    }
                    _ if b1 > 0xF4 => (REPLACEMENT, 1),
                    [b2, ..] if second_bad(b2) => (REPLACEMENT, 1),
                    [_, b3] if !cont(b3) => (REPLACEMENT, 2),
                    _ => (REPLACEMENT, 1 + rest.len()),
                }
            }
            _ => (REPLACEMENT, 1),
        };
        out.push(c);
        i += len;
    }
    out
}

pub(crate) fn parse_int(s: &str) -> Option<i32> {
    let (negative, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let limit: i64 = if negative { 1 << 31 } else { (1 << 31) - 1 };
    let mut value: i64 = 0;
    for c in digits.chars() {
        if u32::from(c) > 0xFFFF {
            return None;
        }
        value = value * 10 + i64::from(turbo_unicode::java::digit(c)?);
        if value > limit {
            return None;
        }
    }
    let value = if negative { -value } else { value };
    i32::try_from(value).ok()
}

pub(crate) use turbo_unicode::java::equals_ignore_case as equals_ignore_case_ascii;

pub(crate) fn split_on_spaces(value: &str) -> Vec<&str> {
    value.split(' ').filter(|s| !s.is_empty()).collect()
}

pub(crate) fn split_space(value: &str) -> Vec<&str> {
    if value.is_empty() {
        return vec![""];
    }
    let mut fields: Vec<&str> = value.split(' ').collect();
    while fields.last().is_some_and(|f| f.is_empty()) {
        fields.pop();
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_replaces_malformed_sequences_at_jdk_boundaries() {
        let decoded = |bytes: &[u8]| decode_utf8(bytes).replace('\u{FFFD}', "?");
        assert_eq!(decode_utf8("a\u{e9}\u{20ac}\u{1f600}\u{feff}".as_bytes()), "a\u{e9}\u{20ac}\u{1f600}\u{feff}");
        assert_eq!(decoded(b"\xED\xA0\x80x"), "?x");
        assert_eq!(decoded(b"\xED\xA0x"), "?x");
        assert_eq!(decoded(b"\xC0\xAFx"), "??x");
        assert_eq!(decoded(b"\xE0\x80\xAFx"), "???x");
        assert_eq!(decoded(b"\xF0\x80\x80\xAFx"), "????x");
        assert_eq!(decoded(b"\xF4\x90\x80\x80x"), "????x");
        assert_eq!(decoded(b"\xF5\x80\x80\x80x"), "????x");
        assert_eq!(decoded(b"\xF8\x88\x80\x80\x80x"), "?????x");
        assert_eq!(decoded(b"\xE2\x28\xA1x"), "?(?x");
        assert_eq!(decoded(b"\xE2\x82x"), "?x");
        assert_eq!(decoded(b"\xF0\x9F\x98x"), "?x");
        assert_eq!(decoded(b"\xF0\x9Fx"), "?x");
        assert_eq!(decoded(b"\xF0\x9F\x98\xF0\x9F\x98\x80"), "?\u{1f600}");
        assert_eq!(decoded(b"\x80\xBF"), "??");
        assert_eq!(decoded(b"x\xE2\x82"), "x?");
        assert_eq!(decoded(b"x\xF0\x9F\x98"), "x?");
        assert_eq!(decoded(b"x\xF0\x9F"), "x?");
        assert_eq!(decoded(b"x\xE0\x80"), "x??");
        assert_eq!(decoded(b"x\xF4\x90"), "x??");
        assert_eq!(decoded(b"x\xC3"), "x?");
    }

    #[test]
    fn integers_are_signed_i32_values_with_bmp_digits() {
        assert_eq!(parse_int("42"), Some(42));
        assert_eq!(parse_int("+7"), Some(7));
        assert_eq!(parse_int("-0"), Some(0));
        assert_eq!(parse_int("\u{0662}\u{0664}"), Some(24));
        assert_eq!(parse_int("\u{ff13}"), Some(3));
        assert_eq!(parse_int("2147483647"), Some(i32::MAX));
        assert_eq!(parse_int("-2147483648"), Some(i32::MIN));
        assert_eq!(parse_int("2147483648"), None);
        for value in ["", "+", " 1", "1a", "\u{1d7cf}"] {
            assert_eq!(parse_int(value), None, "{value:?}");
        }
    }

    #[test]
    fn unicode_case_and_space_splitting() {
        assert!(equals_ignore_case_ascii("REGION", "region"));
        assert!(equals_ignore_case_ascii("reg\u{131}on", "region"));
        assert!(equals_ignore_case_ascii("reg\u{130}on", "region"));
        assert!(equals_ignore_case_ascii("\u{17f}", "s"));
        assert!(equals_ignore_case_ascii("\u{212a}", "k"));
        assert!(equals_ignore_case_ascii("primarydata", "primaryData"));
        assert!(!equals_ignore_case_ascii("regions", "region"));
        assert!(!equals_ignore_case_ascii("regio", "region"));
        assert!(!equals_ignore_case_ascii("g:region", "region"));
        assert!(!equals_ignore_case_ascii("r\u{e9}gion", "region"));
        assert_eq!(split_on_spaces(" 1  2 "), ["1", "2"]);
        assert_eq!(split_space("1  2 "), ["1", "", "2"]);
        assert_eq!(split_space(""), [""]);
    }
}
