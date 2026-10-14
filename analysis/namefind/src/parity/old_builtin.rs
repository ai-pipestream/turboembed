//! `builtin.rs` as it was before the hand-written scanners (main at
//! 1fc349f7): each built-in pattern translated to the `regex` crate and
//! run over a stand-in text. Kept unchanged apart from this header and the
//! trimmed tests, as the reference the scanners are checked against.

#![allow(dead_code)]

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
