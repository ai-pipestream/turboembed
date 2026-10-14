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

/// Java's default `\s`: `[\t\n\x0B\x0C\r ]`.
#[inline]
fn is_java_space(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | 0x0B | 0x0C | b'\r' | b' ')
}

/// Java's default `\w`: `[0-9A-Za-z_]`.
#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline]
fn skip_spaces(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && is_java_space(b[i]) {
        i += 1;
    }
    i
}

#[inline]
fn count_digits(b: &[u8], i: usize, max: usize) -> usize {
    let limit = (i + max).min(b.len());
    let mut j = i;
    while j < limit && b[j].is_ascii_digit() {
        j += 1;
    }
    j - i
}

/// `\d{3}-\d{4}` at `i`.
#[inline]
fn match_phone_core(b: &[u8], i: usize) -> Option<usize> {
    if i + 8 <= b.len()
        && b[i].is_ascii_digit()
        && b[i + 1].is_ascii_digit()
        && b[i + 2].is_ascii_digit()
        && b[i + 3] == b'-'
        && b[i + 4].is_ascii_digit()
        && b[i + 5].is_ascii_digit()
        && b[i + 6].is_ascii_digit()
        && b[i + 7].is_ascii_digit()
    {
        Some(i + 8)
    } else {
        None
    }
}

/// `((\(\d{3}\) ?)|(\d{3}-))?\d{3}-\d{4}` at `pos`.
fn match_usa_phone(b: &[u8], pos: usize) -> Option<usize> {
    if b[pos] == b'(' {
        if pos + 5 <= b.len()
            && b[pos + 1].is_ascii_digit()
            && b[pos + 2].is_ascii_digit()
            && b[pos + 3].is_ascii_digit()
            && b[pos + 4] == b')'
        {
            if b.get(pos + 5) == Some(&b' ')
                && let Some(end) = match_phone_core(b, pos + 6)
            {
                return Some(end);
            }
            return match_phone_core(b, pos + 5);
        }
        return None;
    }
    if pos + 4 <= b.len()
        && b[pos].is_ascii_digit()
        && b[pos + 1].is_ascii_digit()
        && b[pos + 2].is_ascii_digit()
        && b[pos + 3] == b'-'
    {
        if let Some(end) = match_phone_core(b, pos + 4) {
            return Some(end);
        }
        return match_phone_core(b, pos);
    }
    None
}

#[inline]
fn is_email_atom(b: u8) -> bool {
    b.is_ascii_alphanumeric()
        || matches!(
            b,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'/'
                | b'='
                | b'?'
                | b'^'
                | b'_'
                | b'`'
                | b'{'
                | b'|'
                | b'}'
                | b'~'
                | b'-'
        )
}

#[inline]
fn is_email_lookbehind(b: u8) -> bool {
    is_email_atom(b) || b == b'.'
}

/// `25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?` followed by `delim`.
fn match_ipv4_octet(b: &[u8], i: usize, delim: u8) -> Option<usize> {
    let n = count_digits(b, i, 3);
    if n == 0 || b.get(i + n) != Some(&delim) {
        return None;
    }
    let val = b[i..i + n].iter().fold(0u16, |acc, &d| acc * 10 + u16::from(d - b'0'));
    (val <= 255).then_some(i + n + 1)
}

/// `[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.` when the byte after `.` is an
/// ASCII letter or digit.
fn match_domain_label_dot(b: &[u8], i: usize) -> Option<usize> {
    if !b.get(i).is_some_and(u8::is_ascii_alphanumeric) {
        return None;
    }
    let limit = (i + 63).min(b.len());
    let mut j = i + 1;
    while j < limit && (b[j].is_ascii_alphanumeric() || b[j] == b'-') {
        j += 1;
    }
    if b[j - 1].is_ascii_alphanumeric()
        && b.get(j) == Some(&b'.')
        && b.get(j + 1).is_some_and(u8::is_ascii_alphanumeric)
    {
        Some(j + 1)
    } else {
        None
    }
}

/// `BuiltIn::Email` at `pos`.
fn match_email(b: &[u8], pos: usize) -> Option<usize> {
    if pos > 0 && is_email_lookbehind(b[pos - 1]) {
        return None;
    }
    let at = if b[pos] == b'"' {
        let mut i = pos + 1;
        let mut count = 0usize;
        while i < b.len() && b[i] != b'"' {
            if count == 255 {
                return None;
            }
            if b[i] == b'\\' {
                let next = *b.get(i + 1)?;
                if !matches!(next, 0x01..=0x09 | 0x0B | 0x0C | 0x0E..=0x7F) {
                    return None;
                }
                i += 2;
            } else if matches!(b[i], 0x01..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F | 0x21 | 0x23..=0x5B | 0x5D..=0x7F) {
                i += 1;
            } else {
                return None;
            }
            count += 1;
        }
        if b.get(i) != Some(&b'"') || b.get(i + 1) != Some(&b'@') {
            return None;
        }
        i + 2
    } else if is_email_atom(b[pos]) {
        let mut i = pos;
        let limit = (i + 64).min(b.len());
        while i < limit && is_email_atom(b[i]) {
            i += 1;
        }
        for _ in 0..10 {
            if b.get(i) == Some(&b'.') && b.get(i + 1).is_some_and(|&c| is_email_atom(c)) {
                i += 1;
                let limit = (i + 64).min(b.len());
                while i < limit && is_email_atom(b[i]) {
                    i += 1;
                }
            } else {
                break;
            }
        }
        if b.get(i) != Some(&b'@') {
            return None;
        }
        i + 1
    } else {
        return None;
    };

    if b.get(at) == Some(&b'[') {
        let i = match_ipv4_octet(b, at + 1, b'.')?;
        let i = match_ipv4_octet(b, i, b'.')?;
        let i = match_ipv4_octet(b, i, b'.')?;
        return match_ipv4_octet(b, i, b']');
    }

    let mut i = match_domain_label_dot(b, at)?;
    for _ in 1..20 {
        let Some(next) = match_domain_label_dot(b, i) else { break };
        i = next;
    }
    // Final `[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?`.
    let limit = (i + 63).min(b.len());
    let mut end = i + 1;
    while end < limit && (b[end].is_ascii_alphanumeric() || b[end] == b'-') {
        end += 1;
    }
    while b[end - 1] == b'-' {
        end -= 1;
    }
    Some(end)
}

#[inline]
fn word_boundary(b: &[u8], i: usize) -> bool {
    let left = i > 0 && is_word_byte(b[i - 1]);
    let right = i < b.len() && is_word_byte(b[i]);
    left != right
}

#[inline]
fn starts_with_ignore_ascii_case(b: &[u8], i: usize, lit: &[u8]) -> bool {
    b.get(i..i + lit.len()).is_some_and(|s| s.eq_ignore_ascii_case(lit))
}

#[inline]
fn is_url_path_char(b: u8) -> bool {
    is_word_byte(b) || matches!(b, b'-' | b'~' | b'!' | b'$' | b'+' | b'|' | b'.' | b',' | b'=')
}

#[inline]
fn is_url_query_char(b: u8) -> bool {
    is_url_path_char(b) || matches!(b, b'*' | b':')
}

/// One atom of a URL path, query or fragment (`char` or `%[a-fA-F0-9]{2}`).
#[inline]
fn step_url_atom(b: &[u8], i: usize, allow: fn(u8) -> bool) -> Option<usize> {
    let c = *b.get(i)?;
    if allow(c) {
        Some(i + 1)
    } else if c == b'%'
        && b.get(i + 1).is_some_and(u8::is_ascii_hexdigit)
        && b.get(i + 2).is_some_and(u8::is_ascii_hexdigit)
    {
        Some(i + 3)
    } else {
        None
    }
}

/// Matches the optional port, path, query, fragment and trailing `\b`
/// after a URL TLD ending at `tld_end`.
fn match_url_tail(b: &[u8], tld_end: usize) -> Option<usize> {
    let mut last_valid = word_boundary(b, tld_end).then_some(tld_end);
    let mut record = |p: usize| {
        if word_boundary(b, p) {
            last_valid = Some(p);
        }
    };

    let mut i = tld_end;
    if b.get(i) == Some(&b':') {
        let digits = count_digits(b, i + 1, 5);
        if digits > 0 {
            for d in 1..=digits {
                record(i + 1 + d);
            }
            i += 1 + digits;
        }
    }

    for _ in 0..50 {
        if b.get(i) != Some(&b'/') {
            break;
        }
        i += 1;
        record(i);
        for _ in 0..255 {
            let Some(next) = step_url_atom(b, i, is_url_path_char) else { break };
            i = next;
            record(i);
        }
    }

    if b.get(i) == Some(&b'?') {
        i += 1;
        record(i);
        for _ in 0..255 {
            let Some(next) = step_url_atom(b, i, is_url_query_char) else { break };
            i = next;
            record(i);
        }
        for _ in 0..50 {
            if b.get(i) != Some(&b'&') {
                break;
            }
            i += 1;
            record(i);
            for _ in 0..255 {
                let Some(next) = step_url_atom(b, i, is_url_query_char) else { break };
                i = next;
                record(i);
            }
        }
    }

    if b.get(i) == Some(&b'#') {
        i += 1;
        record(i);
        for _ in 0..255 {
            let Some(next) = step_url_atom(b, i, is_url_query_char) else { break };
            i = next;
            record(i);
        }
    }

    last_valid
}

/// `BuiltIn::Url` at `pos`.
fn match_url(b: &[u8], pos: usize) -> Option<usize> {
    if !word_boundary(b, pos) {
        return None;
    }
    let scheme_len = if starts_with_ignore_ascii_case(b, pos, b"http://") {
        7
    } else if starts_with_ignore_ascii_case(b, pos, b"https://") {
        8
    } else if starts_with_ignore_ascii_case(b, pos, b"ftp://") {
        6
    } else if starts_with_ignore_ascii_case(b, pos, b"ftps://") {
        7
    } else if b.get(pos..pos + 2) == Some(b"~/") {
        2
    } else if b[pos] == b'/' {
        1
    } else if starts_with_ignore_ascii_case(b, pos, b"www.") {
        4
    } else {
        return None;
    };

    let mut i = pos + scheme_len;
    // Optional `(?:\w{1,63}:\w{1,63}@)?`.
    {
        let mut u = i;
        let limit = (u + 63).min(b.len());
        while u < limit && is_word_byte(b[u]) {
            u += 1;
        }
        if u > i && b.get(u) == Some(&b':') {
            let p_start = u + 1;
            let mut p = p_start;
            let limit = (p + 63).min(b.len());
            while p < limit && is_word_byte(b[p]) {
                p += 1;
            }
            if p > p_start && b.get(p) == Some(&b'@') {
                i = p + 1;
            }
        }
    }

    // Collect up to 20 `(?:[-\w]{1,63}\.)` end offsets.
    let mut dot_ends = [0usize; 20];
    let mut num_dots = 0usize;
    let mut scan = i;
    while num_dots < 20 {
        let limit = (scan + 63).min(b.len());
        let mut j = scan;
        while j < limit && (is_word_byte(b[j]) || b[j] == b'-') {
            j += 1;
        }
        if j == scan || b.get(j) != Some(&b'.') {
            break;
        }
        scan = j + 1;
        dot_ends[num_dots] = scan;
        num_dots += 1;
    }
    if num_dots == 0 {
        return None;
    }

    const TLDS: [&[u8]; 13] = [
        b"com", b"org", b"net", b"gov", b"mil", b"biz", b"info", b"mobi", b"name", b"aero", b"jobs", b"museum",
        b"travel",
    ];
    for &host_end in dot_ends[..num_dots].iter().rev() {
        for tld in TLDS {
            if starts_with_ignore_ascii_case(b, host_end, tld)
                && let Some(end) = match_url_tail(b, host_end + tld.len())
            {
                return Some(end);
            }
        }
        if b.get(host_end).is_some_and(u8::is_ascii_alphabetic)
            && b.get(host_end + 1).is_some_and(u8::is_ascii_alphabetic)
            && let Some(end) = match_url_tail(b, host_end + 2)
        {
            return Some(end);
        }
    }
    None
}

/// `\d{1,2}[A-Za-z]\s*[A-Za-z]{2}\s*\d{1,5}\s*\d{1,5}` at `pos`.
fn match_mgrs(b: &[u8], pos: usize) -> Option<usize> {
    if !b[pos].is_ascii_digit() {
        return None;
    }
    let after_zone =
        if b.get(pos + 1).is_some_and(u8::is_ascii_digit) && b.get(pos + 2).is_some_and(u8::is_ascii_alphabetic) {
            pos + 3
        } else if b.get(pos + 1).is_some_and(u8::is_ascii_alphabetic) {
            pos + 2
        } else {
            return None;
        };
    let i = skip_spaces(b, after_zone);
    if !b.get(i).is_some_and(u8::is_ascii_alphabetic) || !b.get(i + 1).is_some_and(u8::is_ascii_alphabetic) {
        return None;
    }
    let i = skip_spaces(b, i + 2);
    let d1 = count_digits(b, i, 5);
    if d1 == 0 {
        return None;
    }
    let after_d1 = i + d1;
    let after_ws = skip_spaces(b, after_d1);
    let d2 = count_digits(b, after_ws, 5);
    if d2 > 0 {
        Some(after_ws + d2)
    } else if d1 >= 2 {
        // `\s*` matched 0 spaces and the two `\d{1,5}` groups split `d1`.
        Some(after_d1)
    } else {
        None
    }
}

/// `[d|D|\u00B0|\s]` at `i`.
#[inline]
fn step_deg_unit(b: &[u8], i: usize) -> Option<usize> {
    let c = *b.get(i)?;
    if matches!(c, b'd' | b'D' | b'|') || is_java_space(c) {
        Some(i + 1)
    } else if b.get(i..i + 2) == Some(&[0xC2, 0xB0]) {
        // U+00B0 DEGREE SIGN in UTF-8.
        Some(i + 2)
    } else {
        None
    }
}

/// `\s*\d{1,2}['|\u2019|\s]` at `i`.
fn match_dms_minutes(b: &[u8], i: usize) -> Option<usize> {
    let j = skip_spaces(b, i);
    let d = count_digits(b, j, 2);
    if d == 0 {
        return None;
    }
    let k = j + d;
    let c = *b.get(k)?;
    if matches!(c, b'\'' | b'|') || is_java_space(c) {
        Some(k + 1)
    } else if b.get(k..k + 3) == Some(&[0xE2, 0x80, 0x99]) {
        // U+2019 RIGHT SINGLE QUOTATION MARK in UTF-8.
        Some(k + 3)
    } else {
        None
    }
}

/// `\s*\d{1,2}[\"|\u201d]` at `i`.
fn match_dms_seconds(b: &[u8], i: usize) -> Option<usize> {
    let j = skip_spaces(b, i);
    let d = count_digits(b, j, 2);
    if d == 0 {
        return None;
    }
    let k = j + d;
    let c = *b.get(k)?;
    if matches!(c, b'"' | b'|') {
        Some(k + 1)
    } else if b.get(k..k + 3) == Some(&[0xE2, 0x80, 0x9D]) {
        // U+201D RIGHT DOUBLE QUOTATION MARK in UTF-8.
        Some(k + 3)
    } else {
        None
    }
}

/// Matches the longitude half starting at `pos`:
/// `[-|\+]?\d{1,3}[d|D|\u00B0|\s](\s*\d{1,2}['|\u2019|\s])?(\s*\d{1,2}[\"|\u201d])?\s*[E|e|W|w]?`.
fn match_dms_lon(b: &[u8], mut i: usize) -> Option<usize> {
    if b.get(i).is_some_and(|&c| matches!(c, b'-' | b'|' | b'+')) {
        i += 1;
    }
    let deg = count_digits(b, i, 3);
    if deg == 0 {
        return None;
    }
    let mut i = step_deg_unit(b, i + deg)?;
    if let Some(after_min) = match_dms_minutes(b, i) {
        i = after_min;
    }
    if let Some(after_sec) = match_dms_seconds(b, i) {
        i = after_sec;
    }
    i = skip_spaces(b, i);
    if b.get(i).is_some_and(|&c| matches!(c, b'E' | b'e' | b'W' | b'w' | b'|')) {
        i += 1;
    }
    Some(i)
}

/// Matches `\s*[N|n|S|s]?(\s*|,|,\s*)` followed by `match_dms_lon` from
/// `after_lat_body`.
fn match_dms_after_lat(b: &[u8], after_lat_body: usize) -> Option<usize> {
    let ws = skip_spaces(b, after_lat_body);
    if b.get(ws).is_some_and(|&c| matches!(c, b'N' | b'n' | b'S' | b's' | b'|')) {
        let after_hemi = ws + 1;
        // Separator `(\s*|,|,\s*)` after hemisphere.
        let after_sep_ws = skip_spaces(b, after_hemi);
        if let Some(end) = match_dms_lon(b, after_sep_ws) {
            return Some(end);
        }
        if b.get(after_hemi) == Some(&b',') {
            if let Some(end) = match_dms_lon(b, after_hemi + 1) {
                return Some(end);
            }
            let after_comma_ws = skip_spaces(b, after_hemi + 1);
            if let Some(end) = match_dms_lon(b, after_comma_ws) {
                return Some(end);
            }
        }
    }
    // Without `[N|n|S|s]`: `ws` already consumed all `\s*`.
    if let Some(end) = match_dms_lon(b, ws) {
        return Some(end);
    }
    if b.get(ws) == Some(&b',') {
        if let Some(end) = match_dms_lon(b, ws + 1) {
            return Some(end);
        }
        let after_comma_ws = skip_spaces(b, ws + 1);
        if let Some(end) = match_dms_lon(b, after_comma_ws) {
            return Some(end);
        }
    }
    None
}

/// `BuiltIn::DegreesMinSecLatLon` at `pos`.
fn match_dms_lat_lon(b: &[u8], pos: usize) -> Option<usize> {
    let mut i = pos;
    if matches!(b[i], b'-' | b'|' | b'+') {
        i += 1;
    }
    let deg = count_digits(b, i, 3);
    if deg == 0 {
        return None;
    }
    let after_deg = step_deg_unit(b, i + deg)?;

    if let Some(after_min) = match_dms_minutes(b, after_deg) {
        if let Some(after_sec) = match_dms_seconds(b, after_min)
            && let Some(end) = match_dms_after_lat(b, after_sec)
        {
            return Some(end);
        }
        if let Some(end) = match_dms_after_lat(b, after_min) {
            return Some(end);
        }
    }
    if let Some(after_sec) = match_dms_seconds(b, after_deg)
        && let Some(end) = match_dms_after_lat(b, after_sec)
    {
        return Some(end);
    }
    match_dms_after_lat(b, after_deg)
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
