//! E-mail addresses, `CursorPiiExtractor.scanEmails`: expanded around
//! each `@`, an ASCII local part and a dotted domain whose last label is a
//! registered top-level domain.

use turbo_analysis_layers::{PiiConfidence, PiiKind};

use crate::hits::Hits;
use crate::tables::tlds;
use crate::text::{before, letter_or_digit, on_end};

const LOCAL_MAX: usize = 64;
const ADDRESS_MAX: usize = 254;
/// RFC 1035: 253 characters in presentation form.
const DOMAIN_MAX: usize = 253;
const LABEL_MAX: usize = 63;

fn is_local(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'%' | b'+' | b'-')
}

fn is_domain(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'.' || c == b'-'
}

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    for at in b.iter().enumerate().filter(|(_, c)| **c == b'@').map(|(i, _)| i) {
        let mut start = at;
        while start > 0 && is_local(b[start - 1]) {
            start -= 1;
        }
        if start == at || !valid_local(&b[start..at]) {
            continue;
        }
        let mut end = at + 1;
        while end < b.len() && is_domain(b[end]) {
            end += 1;
        }
        // The boundary is checked before trailing punctuation is dropped.
        if !on_end(text, end) {
            continue;
        }
        while end > at + 1 && matches!(b[end - 1], b'.' | b'-') {
            end -= 1;
        }
        if end == at + 1
            || end - start > ADDRESS_MAX
            || !valid_domain(&b[at + 1..end])
            || before(text, start).is_some_and(letter_or_digit)
        {
            continue;
        }
        hits.add(start, end, PiiKind::Email, PiiConfidence::Format, |s| {
            s.push_str(&text[start..=at]);
            s.extend(b[at + 1..end].iter().map(|c| c.to_ascii_lowercase() as char));
        });
    }
}

/// At most 64 characters, no dot first, last or doubled.
fn valid_local(local: &[u8]) -> bool {
    local.len() <= LOCAL_MAX
        && local[0] != b'.'
        && local[local.len() - 1] != b'.'
        && !local.windows(2).any(|w| w == b"..")
}

/// At most 253 characters, two labels or more of 1 to 63 characters
/// without a hyphen first or last, the last one a registered TLD.
fn valid_domain(domain: &[u8]) -> bool {
    if domain.len() > DOMAIN_MAX {
        return false;
    }
    let mut labels = 0;
    let mut last: &[u8] = &[];
    for label in domain.split(|&c| c == b'.') {
        if label.is_empty() || label.len() > LABEL_MAX || label[0] == b'-' || label[label.len() - 1] == b'-' {
            return false;
        }
        labels += 1;
        last = label;
    }
    labels >= 2 && tlds::registered(last)
}
