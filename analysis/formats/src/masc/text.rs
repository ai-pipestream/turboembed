//! UTF-8 decoding and small ASCII parsers used by MASC and brat readers.

pub(crate) fn decode_utf8(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

pub(crate) fn parse_int(s: &str) -> Option<i32> {
    s.parse().ok()
}

pub(crate) fn equals_ignore_case_ascii(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

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
    fn utf8_uses_unicode_replacement_and_offsets_remain_deterministic() {
        assert_eq!(decode_utf8("a\u{e9}\u{20ac}\u{1f600}".as_bytes()), "a\u{e9}\u{20ac}\u{1f600}");
        assert_eq!(decode_utf8(b"a\xED\xA0\x80b"), "a\u{fffd}\u{fffd}\u{fffd}b");
        assert_eq!(decode_utf8(b"a\xE2\x82"), "a\u{fffd}");
    }

    #[test]
    fn integers_are_signed_ascii_i32_values() {
        assert_eq!(parse_int("42"), Some(42));
        assert_eq!(parse_int("+7"), Some(7));
        assert_eq!(parse_int("-2147483648"), Some(i32::MIN));
        assert_eq!(parse_int("2147483648"), None);
        for value in ["", "+", " 1", "1a", "\u{0662}\u{0664}", "\u{ff13}"] {
            assert_eq!(parse_int(value), None, "{value:?}");
        }
    }

    #[test]
    fn ascii_case_and_space_splitting() {
        assert!(equals_ignore_case_ascii("REGION", "region"));
        assert!(!equals_ignore_case_ascii("reg\u{131}on", "region"));
        assert_eq!(split_on_spaces(" 1  2 "), ["1", "2"]);
        assert_eq!(split_space("1  2 "), ["1", "", "2"]);
        assert_eq!(split_space(""), [""]);
    }
}
