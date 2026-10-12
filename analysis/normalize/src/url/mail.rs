//! The e-mail address scan both URL normalizers use, ported from OpenNLP's
//! `MailAddressScan`.
//!
//! An address is a maximal run of local-part characters `[-+_.0-9A-Za-z]`
//! (the character before it is not one of them), an `@`, and a domain run
//! of `[-.0-9A-Za-z]` that does not start with a dot and is at least two
//! characters long. Every character involved is ASCII, so the scan works
//! on bytes: a byte of a multi-byte UTF-8 sequence is never in any of the
//! sets, which is how OpenNLP's UTF-16 scan treats non-ASCII characters.

/// `[-+_.0-9A-Za-z]`: the local part, and the characters that may not
/// stand right before an address.
pub(crate) fn is_local(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'+' | b'_' | b'.')
}

/// `[-0-9A-Za-z]`: the first character of the domain.
fn is_domain_start(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-'
}

/// `[-.0-9A-Za-z]`: the rest of the domain.
fn is_domain(b: u8) -> bool {
    is_domain_start(b) || b == b'.'
}

/// An address found by [`match_at`]: where its domain starts and where
/// the address ends.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Address {
    pub domain_start: usize,
    pub end: usize,
}

/// The shortest domain OpenNLP accepts, in characters (all ASCII, so also
/// in bytes).
pub(crate) const MIN_DOMAIN_LEN: usize = 2;

/// The address that starts at byte `start`, if one does.
pub(crate) fn match_at(text: &[u8], start: usize) -> Option<Address> {
    if start > 0 && is_local(text[start - 1]) {
        return None;
    }
    if !is_local(*text.get(start)?) {
        return None;
    }
    let mut at = start + 1;
    while at < text.len() && is_local(text[at]) {
        at += 1;
    }
    if text.get(at) != Some(&b'@') {
        return None;
    }
    let domain_start = at + 1;
    if !is_domain_start(*text.get(domain_start)?) {
        return None;
    }
    let mut end = domain_start + 1;
    while end < text.len() && is_domain(text[end]) {
        end += 1;
    }
    (end - domain_start >= MIN_DOMAIN_LEN).then_some(Address { domain_start, end })
}

/// A character of an internationalized address (RFC 6531 local part, RFC
/// 5890 U-label domain): any letter, digit or combining mark beyond ASCII.
fn is_international(c: char) -> bool {
    !c.is_ascii()
        && (turbo_unicode::java::is_letter_or_digit(c) || turbo_unicode::category::general_category(c).is_mark())
}

fn is_local_char(c: char) -> bool {
    if c.is_ascii() { is_local(c as u8) } else { is_international(c) }
}

fn is_domain_start_char(c: char) -> bool {
    if c.is_ascii() { is_domain_start(c as u8) } else { is_international(c) }
}

fn is_domain_char(c: char) -> bool {
    if c.is_ascii() { is_domain(c as u8) } else { is_international(c) }
}

/// The address that starts at byte `start` of `text`, with the sets of
/// [`match_at`] widened to every non-ASCII letter, digit and combining
/// mark, so `José@example.com` and `用户@例子.广告` are addresses. The
/// domain must still be two characters long.
pub(crate) fn match_international_at(text: &str, start: usize) -> Option<Address> {
    if text[..start].chars().next_back().is_some_and(is_local_char) {
        return None;
    }
    let local = text[start..].char_indices().take_while(|&(_, c)| is_local_char(c)).last()?;
    let at = start + local.0 + local.1.len_utf8();
    if text.as_bytes().get(at) != Some(&b'@') {
        return None;
    }
    let domain_start = at + 1;
    let mut domain = text[domain_start..].char_indices();
    let (_, first) = domain.next()?;
    if !is_domain_start_char(first) {
        return None;
    }
    let mut end = domain_start + first.len_utf8();
    let mut chars = 1;
    for (k, c) in domain.take_while(|&(_, c)| is_domain_char(c)) {
        end = domain_start + k + c.len_utf8();
        chars += 1;
    }
    (chars >= MIN_DOMAIN_LEN).then_some(Address { domain_start, end })
}
