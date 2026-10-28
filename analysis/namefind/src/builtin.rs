//! The patterns of `RegexNameFinderFactory.DEFAULT_REGEX_NAME_FINDER`.
//!
//! Each of the five built-in patterns (`USA_PHONE_NUM`, `EMAIL`, `URL`,
//! `MGRS`, `DEGREES_MIN_SEC_LAT_LON`) is matched directly over the UTF-8
//! bytes of the space-joined tokens by a hand-written scanner in
//! [`search`], with no regex compilation and no per-sentence stand-in
//! buffer.
//!
//! All five Java patterns match ASCII characters (plus U+00B0, U+2019 and
//! U+201D in `DEGREES_MIN_SEC_LAT_LON`), fold ASCII case only, and use
//! Java's default ASCII `\d`, `\w`, `\s` and `\b` classes.
//!
//! `\b` in `java.util.regex` also counts a nonspacing mark after a letter
//! or digit as a word character. It makes no difference to the names:
//! `URL`, the one pattern with `\b`, matches no space, so a match lies
//! within one token, and a match that is kept starts and ends at the
//! token's edges, next to a space or the end of the text. A boundary
//! inside a token only decides matches that are dropped, and they cover
//! nothing past the token.
//!
//! `EMAIL` starts with a lookbehind, `(?<![a-z0-9!#$%&'*+/=?^_`{|}~.-])`;
//! [`search`] checks the byte before each candidate start and moves on one
//! byte when the check fails, which is the order the Java matcher tries
//! positions in.

mod geo;
mod net;

use self::geo::{match_dms_lat_lon, match_mgrs};
use self::net::{match_email, match_url, match_usa_phone};

/// `RegexNameFinderFactory.DEFAULT_REGEX_NAME_FINDER`, in Java's order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BuiltIn {
    UsaPhoneNum,
    Email,
    Url,
    Mgrs,
    DegreesMinSecLatLon,
}

impl BuiltIn {
    pub const ALL: [BuiltIn; 5] =
        [BuiltIn::UsaPhoneNum, BuiltIn::Email, BuiltIn::Url, BuiltIn::Mgrs, BuiltIn::DegreesMinSecLatLon];

    /// The enum constant's Java name.
    pub const fn java_name(self) -> &'static str {
        match self {
            BuiltIn::UsaPhoneNum => "USA_PHONE_NUM",
            BuiltIn::Email => "EMAIL",
            BuiltIn::Url => "URL",
            BuiltIn::Mgrs => "MGRS",
            BuiltIn::DegreesMinSecLatLon => "DEGREES_MIN_SEC_LAT_LON",
        }
    }

    /// The entity type its spans get (`getType()`): the Java name, except
    /// `PHONE_NUM` for `USA_PHONE_NUM`.
    pub const fn kind(self) -> &'static str {
        match self {
            BuiltIn::UsaPhoneNum => "PHONE_NUM",
            other => other.java_name(),
        }
    }

    /// The Java pattern, as `Pattern.compile` receives it.
    pub const fn java_pattern(self) -> &'static str {
        match self {
            BuiltIn::UsaPhoneNum => r"((\(\d{3}\) ?)|(\d{3}-))?\d{3}-\d{4}",
            BuiltIn::Email => concat!(
                r"(?<![a-z0-9!#$%&'*+/=?^_`{|}~.-])",
                r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]{1,64}(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]{1,64}){0,10}",
                r#"|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]"#,
                r#"|\\[\x01-\x09\x0b\x0c\x0e-\x7f]){0,255}")"#,
                r"@(?:(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){1,20}",
                r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
                r"|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}",
                r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\])",
            ),
            BuiltIn::Url => concat!(
                r"\b(?:(?:ht|f)tps?://|~/|/|www\.)",
                r"(?:\w{1,63}:\w{1,63}@)?",
                r"(?:[-\w]{1,63}\.){1,20}(?:com|org|net|gov",
                r"|mil|biz|info|mobi|name|aero|jobs|museum",
                r"|travel|[a-z]{2})(?::\d{1,5})?",
                r"(?:/(?:[-\w~!$+|.,=]|%[a-f\d]{2}){0,255}){0,50}",
                r"(?:\?(?:[-\w~!$+|.,*:=]|%[a-f\d]{2}){0,255}",
                r"(?:&(?:[-\w~!$+|.,*:=]|%[a-f\d]{2}){0,255}){0,50})?",
                r"(?:#(?:[-\w~!$+|.,*:=]|%[a-f\d]{2}){0,255})?\b",
            ),
            BuiltIn::Mgrs => r"\d{1,2}[A-Za-z]\s*[A-Za-z]{2}\s*\d{1,5}\s*\d{1,5}",
            BuiltIn::DegreesMinSecLatLon => concat!(
                r#"([-|\+]?\d{1,3}[d|D|\u00B0|\s](\s*\d{1,2}['|\u2019|\s])"#,
                r#"?(\s*\d{1,2}[\"|\u201d])?\s*[N|n|S|s]?)(\s*|,|,\s*)([-|\+]?\d{1,3}[d|D|\u00B0|"#,
                r#"\s](\s*\d{1,2}['|\u2019|\s])?(\s*\d{1,2}[\"|\u201d])?\s*[E|e|W|w]?)"#,
            ),
        }
    }

    /// Whether Java compiles it with `Pattern.CASE_INSENSITIVE`.
    pub const fn case_insensitive(self) -> bool {
        !matches!(self, BuiltIn::UsaPhoneNum)
    }
}

/// The leftmost match of `pattern` in `text` that starts at or after
/// `from`, as `Matcher.find` reports it.
pub(crate) fn search(pattern: BuiltIn, text: &str, from: usize) -> Option<(usize, usize)> {
    let b = text.as_bytes();
    let match_at: fn(&[u8], usize) -> Option<usize> = match pattern {
        BuiltIn::UsaPhoneNum => match_usa_phone,
        BuiltIn::Email => match_email,
        BuiltIn::Url => match_url,
        BuiltIn::Mgrs => match_mgrs,
        BuiltIn::DegreesMinSecLatLon => match_dms_lat_lon,
    };
    for start in from..b.len() {
        // Every built-in pattern starts with an ASCII byte.
        if b[start] < 0x80
            && let Some(end) = match_at(b, start)
        {
            return Some((start, end));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_and_flags() {
        for b in BuiltIn::ALL {
            assert!(!b.java_pattern().is_empty());
        }
        assert_eq!(BuiltIn::UsaPhoneNum.kind(), "PHONE_NUM");
        assert_eq!(BuiltIn::Mgrs.kind(), "MGRS");
        assert!(!BuiltIn::UsaPhoneNum.case_insensitive());
        assert!(BuiltIn::Email.case_insensitive());
    }

    #[test]
    fn email_lookbehind_moves_one_character() {
        // "." may not precede a match, and may not start one either.
        assert_eq!(search(BuiltIn::Email, ".a@b.com", 0), None);
        assert_eq!(search(BuiltIn::Email, ".a@b.com c@d.org", 0), Some((9, 16)));
        assert_eq!(search(BuiltIn::Email, "x.a@b.com", 0), Some((0, 9)));
        // A space in a quoted local part must be escaped.
        assert_eq!(search(BuiltIn::Email, "\"a b\"@c.de", 0), None);
        assert_eq!(search(BuiltIn::Email, r#""a\ b"@c.de"#, 0), Some((0, 11)));
    }

    #[test]
    fn url_needs_ascii_word_boundaries() {
        let url = |token: &str| search(BuiltIn::Url, token, 0);
        assert_eq!(url("www.example.com"), Some((0, 15)));
        assert_eq!(url("www.example.com\u{E9}"), Some((0, 15)));
        assert_eq!(url("www.example.co_"), None);
        // The slash form needs a word character before the slash.
        assert_eq!(url("/a.com"), None);
        assert_eq!(url("x/a.com"), Some((1, 7)));
    }
}
