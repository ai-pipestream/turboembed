//! The patterns of `RegexNameFinderFactory.DEFAULT_REGEX_NAME_FINDER`.
//!
//! Each is a `regex` crate pattern that matches exactly where the Java
//! pattern does, run over a stand-in for the joined tokens rather than the
//! tokens themselves. The Java patterns match ASCII and three punctuation
//! marks (U+00B0, U+2019 and U+201D, in `DEGREES_MIN_SEC_LAT_LON`), so the
//! stand-in keeps those and replaces every other character by U+0000,
//! which none of the patterns matches, and which is not a word character.
//!
//! Case-insensitive Java patterns fold ASCII only, and the `regex`
//! crate's `(?i)` folds Unicode too (`k` matches U+212A KELVIN SIGN); the
//! stand-in holds no character that folds to ASCII, so the two agree.
//! `\d`, `\w` and `\s` are written as their ASCII classes and `\b` as the
//! ASCII word boundary.
//!
//! `\b` in `java.util.regex` also counts a nonspacing mark after a letter
//! or digit as a word character, where the stand-in has U+0000. It makes
//! no difference to the names: `URL`, the one pattern with `\b`, matches
//! no space, so a match lies within one token, and a match that is kept
//! starts and ends at the token's edges, next to a space or the end of
//! the text. A boundary inside a token only decides matches that are
//! dropped, and they cover nothing past the token.
//!
//! `EMAIL` starts with a lookbehind, `(?<![a-z0-9!#$%&'*+/=?^_`{|}~.-])`,
//! which the `regex` crate lacks; [`search`] finds the leftmost match of
//! the rest and checks the character before it, moving on one character
//! when the check fails, which is the order the Java matcher tries
//! positions in. A kept match starts after a space, where the lookbehind
//! always holds, so it decides only which dropped matches are found; the
//! golden files have no sentence where that changes the names, but it is
//! reproduced so the matcher agrees with Java's on any text.

use std::sync::OnceLock;

use regex::{Regex, RegexBuilder};

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

    /// The `regex` translation, compiled once.
    pub(crate) fn regex(self) -> &'static Regex {
        static COMPILED: [OnceLock<Regex>; 5] = [const { OnceLock::new() }; 5];
        COMPILED[self as usize].get_or_init(|| {
            RegexBuilder::new(&self.translated()).size_limit(1 << 30).build().expect("the built-in patterns compile")
        })
    }

    /// The pattern in `regex` syntax, over the stand-in text. For `EMAIL`
    /// it is the pattern after the lookbehind.
    pub fn translated(self) -> String {
        // Java's \s and \w by default.
        const S: &str = r"[\t\n\x0B\x0C\r ]";
        const W: &str = "0-9A-Za-z_";
        match self {
            BuiltIn::UsaPhoneNum => r"(?:\([0-9]{3}\) ?|[0-9]{3}-)?[0-9]{3}-[0-9]{4}".into(),
            BuiltIn::Email => concat!(
                r"(?i)(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]{1,64}(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]{1,64}){0,10}",
                r#"|"(?:[\x01-\x08\x0B\x0C\x0E-\x1F\x21\x23-\x5B\x5D-\x7F]|\\[\x01-\x09\x0B\x0C\x0E-\x7F]){0,255}")"#,
                r"@(?:(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){1,20}[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
                r"|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\])",
            )
            .into(),
            BuiltIn::Url => {
                let part = |extra: &str| format!(r"(?:[-{W}~!$+|.,{extra}=]|%[a-f0-9]{{2}}){{0,255}}");
                let (path, query) = (part(""), part("*:"));
                format!(
                    concat!(
                        r"(?i)(?-u:\b)(?:(?:ht|f)tps?://|~/|/|www\.)(?:[{W}]{{1,63}}:[{W}]{{1,63}}@)?",
                        r"(?:[-{W}]{{1,63}}\.){{1,20}}",
                        r"(?:com|org|net|gov|mil|biz|info|mobi|name|aero|jobs|museum|travel|[a-z]{{2}})",
                        r"(?::[0-9]{{1,5}})?(?:/{path}){{0,50}}(?:\?{query}(?:&{query}){{0,50}})?(?:#{query})?(?-u:\b)",
                    ),
                    W = W,
                    path = path,
                    query = query,
                )
            }
            BuiltIn::Mgrs => format!(r"[0-9]{{1,2}}[A-Za-z]{S}*[A-Za-z]{{2}}{S}*[0-9]{{1,5}}{S}*[0-9]{{1,5}}"),
            BuiltIn::DegreesMinSecLatLon => {
                let half = |hemisphere: &str| {
                    format!(
                        concat!(
                            r"(?:[-|+]?[0-9]{{1,3}}[dD|\x{{B0}}\t\n\x0B\x0C\r ]",
                            r"(?:{S}*[0-9]{{1,2}}['|\x{{2019}}\t\n\x0B\x0C\r ])?",
                            r#"(?:{S}*[0-9]{{1,2}}["|\x{{201D}}])?{S}*[{h}|]?)"#,
                        ),
                        S = S,
                        h = hemisphere,
                    )
                };
                format!(r"{}(?:{S}*|,|,{S}*){}", half("NnSs"), half("EeWw"))
            }
        }
    }
}

/// Appends the stand-in for `token`: ASCII, U+00B0, U+2019 and U+201D
/// kept, every other character U+0000.
pub(crate) fn push_stand_in(out: &mut String, token: &str) {
    if token.is_ascii() {
        out.push_str(token);
        return;
    }
    out.extend(token.chars().map(|c| match c {
        '\0'..='\x7F' | '\u{B0}' | '\u{2019}' | '\u{201D}' => c,
        _ => '\0',
    }));
}

/// The characters the `EMAIL` lookbehind refuses before a match, with
/// `CASE_INSENSITIVE`.
fn email_lookbehind(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"!#$%&'*+/=?^_`{|}~.-".contains(&b)
}

/// The leftmost match of `pattern` in the stand-in `text` that starts at
/// or after `from`, as `Matcher.find` would report it.
pub(crate) fn search(pattern: BuiltIn, text: &str, mut from: usize) -> Option<(usize, usize)> {
    let regex = pattern.regex();
    loop {
        let m = regex.find_at(text, from)?;
        if pattern != BuiltIn::Email || m.start() == 0 || !email_lookbehind(text.as_bytes()[m.start() - 1]) {
            return Some((m.start(), m.end()));
        }
        // A match starts with an ASCII character.
        from = m.start() + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pattern_compiles() {
        for b in BuiltIn::ALL {
            let _ = b.regex();
        }
        assert_eq!(BuiltIn::UsaPhoneNum.kind(), "PHONE_NUM");
        assert_eq!(BuiltIn::Mgrs.kind(), "MGRS");
    }

    #[test]
    fn stand_in() {
        let mut out = String::new();
        push_stand_in(&mut out, "a\u{B0}b\u{2019}\u{201D}\u{E9}\u{1F600}\u{301}");
        assert_eq!(out, "a\u{B0}b\u{2019}\u{201D}\0\0\0");
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
        let url = |token: &str| {
            let mut text = String::new();
            push_stand_in(&mut text, token);
            search(BuiltIn::Url, &text, 0)
        };
        assert_eq!(url("www.example.com"), Some((0, 15)));
        assert_eq!(url("www.example.com\u{E9}"), Some((0, 15)));
        assert_eq!(url("www.example.co_"), None);
        // The slash form needs a word character before the slash.
        assert_eq!(url("/a.com"), None);
        assert_eq!(url("x/a.com"), Some((1, 7)));
    }
}
