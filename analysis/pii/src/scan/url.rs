//! Web addresses, `WebPiiExtractor`: an `http`, `https`, `ftp` or `ftps`
//! URL (scheme in any case), or a host starting `www.`, at a token start.
//!
//! The host is dot-separated labels of ASCII letters, digits and inner
//! hyphens ending in a registered top-level domain, or, after a scheme, a
//! dotted IPv4 address; then an optional port and the path, query and
//! fragment characters of RFC 3986. Trailing sentence punctuation and an
//! unbalanced closing bracket are left out. A URL with userinfo
//! (`user:password@`) is left to the `url-credential` scan. The
//! normalized form has the scheme and host in lower case and the rest as
//! written.

use turbo_analysis_layers::{PiiEvidence, PiiKind};

use super::secrets::on_token_start;
use crate::hits::Hits;
use crate::tables::tlds;

const SCHEMES: [&[u8]; 4] = [b"https://", b"http://", b"ftps://", b"ftp://"];
const MAX_LABEL: usize = 63;

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if !matches!(b[i], b'h' | b'H' | b'f' | b'F' | b'w' | b'W')
            || !on_token_start(text, i, false)
            || after_joiner(b, i)
        {
            i += 1;
            continue;
        }
        let Some((host_end, end)) = url_at(b, i) else {
            i += 1;
            continue;
        };
        hits.add(i, end, PiiKind::Url, PiiEvidence::Shape, |s| {
            s.extend(b[i..host_end].iter().map(|&c| c.to_ascii_lowercase() as char));
            s.push_str(&text[host_end..end]);
        });
        i = end;
    }
}

/// A `.`, `@`, `/` or `:` right before `i`: `i` is inside a host, path or
/// address, not at the start of a URL.
fn after_joiner(b: &[u8], i: usize) -> bool {
    i > 0 && matches!(b[i - 1], b'.' | b'@' | b'/' | b':' | b'-')
}

fn scheme_len(b: &[u8], i: usize) -> Option<usize> {
    let rest = &b[i..];
    if rest.len() >= 4 && rest[..4].eq_ignore_ascii_case(b"www.") {
        return Some(0);
    }
    SCHEMES.iter().find(|s| rest.len() >= s.len() && rest[..s.len()].eq_ignore_ascii_case(s)).map(|s| s.len())
}

/// The host's end and the URL's end, when a URL starts at `i`.
fn url_at(b: &[u8], i: usize) -> Option<(usize, usize)> {
    let scheme = scheme_len(b, i)?;
    let host_start = i + scheme;
    if has_userinfo(b, host_start) {
        return None;
    }
    let host_end = domain_end(b, host_start).or_else(|| (scheme > 0).then(|| ipv4_end(b, host_start)).flatten())?;
    let mut p = host_end;
    if b.get(p) == Some(&b':') {
        let digits = b[p + 1..].iter().take(6).take_while(|c| c.is_ascii_digit()).count();
        if (1..=5).contains(&digits) {
            p += 1 + digits;
        }
    }
    if matches!(b.get(p), Some(b'/' | b'?' | b'#')) {
        while b.get(p).is_some_and(|&c| is_url_char(c)) {
            p += 1;
        }
    }
    let end = trim_end(b, host_end, p);
    let next = b.get(end).copied();
    (!next.is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80)).then_some((host_end, end))
}

/// Whether the authority starting at `at` has an `@` before its end.
fn has_userinfo(b: &[u8], at: usize) -> bool {
    b[at..].iter().take_while(|&&c| is_url_char(c) && !matches!(c, b'/' | b'?' | b'#')).any(|&c| c == b'@')
}

/// The end of a host name of two or more labels whose last label is a
/// registered top-level domain.
fn domain_end(b: &[u8], start: usize) -> Option<usize> {
    let mut p = start;
    let mut labels = 0;
    let mut last = start..start;
    loop {
        let from = p;
        while p < b.len() && p - from < MAX_LABEL && (b[p].is_ascii_alphanumeric() || b[p] == b'-') {
            p += 1;
        }
        if p == from || b[from] == b'-' || b[p - 1] == b'-' {
            return None;
        }
        labels += 1;
        last = from..p;
        if b.get(p) == Some(&b'.') && b.get(p + 1).is_some_and(u8::is_ascii_alphanumeric) {
            p += 1;
        } else {
            break;
        }
    }
    let tld = &b[last];
    (labels >= 2 && tld.iter().all(u8::is_ascii_alphabetic) && tlds::registered(tld)).then_some(p)
}

/// The end of a dotted-decimal IPv4 address of four octets up to 255.
fn ipv4_end(b: &[u8], start: usize) -> Option<usize> {
    let mut p = start;
    for octet in 0..4 {
        if octet > 0 {
            if b.get(p) != Some(&b'.') {
                return None;
            }
            p += 1;
        }
        let digits = b[p..].iter().take(4).take_while(|c| c.is_ascii_digit()).count();
        let value = b[p..p + digits].iter().fold(0u32, |v, &d| v * 10 + u32::from(d - b'0'));
        if !(1..=3).contains(&digits) || value > 255 || (digits > 1 && b[p] == b'0') {
            return None;
        }
        p += digits;
    }
    Some(p)
}

/// RFC 3986 unreserved, reserved and percent characters.
fn is_url_char(c: u8) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(
            c,
            b'-' | b'.'
                | b'_'
                | b'~'
                | b':'
                | b'/'
                | b'?'
                | b'#'
                | b'['
                | b']'
                | b'@'
                | b'!'
                | b'$'
                | b'&'
                | b'\''
                | b'('
                | b')'
                | b'*'
                | b'+'
                | b','
                | b';'
                | b'='
                | b'%'
        )
}

/// `end` with trailing sentence punctuation, quotes and unbalanced
/// closing brackets taken off, never before `floor`.
fn trim_end(b: &[u8], floor: usize, mut end: usize) -> usize {
    while end > floor {
        let c = b[end - 1];
        let unbalanced = |open: u8, close: u8| {
            let s = &b[floor..end];
            s.iter().filter(|&&x| x == close).count() > s.iter().filter(|&&x| x == open).count()
        };
        let drop = match c {
            b'.' | b',' | b';' | b':' | b'!' | b'?' | b'\'' | b'*' => true,
            b')' => unbalanced(b'(', b')'),
            b']' => unbalanced(b'[', b']'),
            _ => false,
        };
        if !drop {
            break;
        }
        end -= 1;
    }
    end
}
