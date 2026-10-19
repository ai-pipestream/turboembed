//! Hand-written byte scanners for `USA_PHONE_NUM`, `EMAIL`, and `URL`.

/// Java's default `\s`: `[\t\n\x0B\x0C\r ]`.
#[inline]
pub(super) fn is_java_space(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | 0x0B | 0x0C | b'\r' | b' ')
}

/// Java's default `\w`: `[0-9A-Za-z_]`.
#[inline]
pub(super) fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline]
pub(super) fn count_digits(b: &[u8], i: usize, max: usize) -> usize {
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
pub(super) fn match_usa_phone(b: &[u8], pos: usize) -> Option<usize> {
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
pub(super) fn match_email(b: &[u8], pos: usize) -> Option<usize> {
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
pub(super) fn match_url(b: &[u8], pos: usize) -> Option<usize> {
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
