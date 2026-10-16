//! National identifiers by country.
//!
//! - United States, `UsIdentityPiiExtractor`: SSNs and ITINs written
//!   `ddd-dd-dddd` or `ddd dd dddd` with one separator throughout.
//! - United Kingdom and Germany, `EuIdentityPiiExtractor`: NHS numbers
//!   (`ddd ddd dddd` or ten bare digits, mod 11) and Steuer-IDs
//!   (`dd ddd ddd ddd` or eleven bare digits, ISO 7064 MOD 11,10 with
//!   the repetition rules).

use turbo_analysis_layers::{PiiConfidence, PiiKind, PiiKinds};

use crate::hits::Hits;
use crate::text::{on_end, on_number_start};

/// No letter or digit after `end`, and none of `joiners` going on with a
/// digit.
fn on_number_end(text: &str, end: usize, joiners: &[u8]) -> bool {
    let b = text.as_bytes();
    on_end(text, end) && (end + 1 >= b.len() || !joiners.contains(&b[end]) || !b[end + 1].is_ascii_digit())
}

fn value(digits: &[u8]) -> u32 {
    digits.iter().fold(0, |v, &d| v * 10 + u32::from(d - b'0'))
}

/// SSNs and ITINs of `kinds`.
pub(crate) fn scan_us(text: &str, kinds: PiiKinds, hits: &mut Hits) {
    const LENGTH: usize = 11;
    let b = text.as_bytes();
    let mut i = 0;
    while i + LENGTH <= b.len() {
        let found = us_at(text, i).filter(|&kind| kinds.contains(kind));
        let Some(kind) = found else {
            i += 1;
            continue;
        };
        let separator = b[i + 3];
        hits.add(i, i + LENGTH, kind, PiiConfidence::Format, |s| {
            s.extend(b[i..i + LENGTH].iter().map(|&c| if c == separator { '-' } else { c as char }));
        });
        i += LENGTH;
    }
}

/// The kind of the 11 bytes at `i`, when they are one.
fn us_at(text: &str, i: usize) -> Option<PiiKind> {
    let b = text.as_bytes();
    if !b[i].is_ascii_digit() || !on_number_start(text, i) {
        return None;
    }
    let separator = b[i + 3];
    if (separator != b'-' && separator != b' ') || b[i + 6] != separator {
        return None;
    }
    let (area, group, serial) = (&b[i..i + 3], &b[i + 4..i + 6], &b[i + 7..i + 11]);
    if ![area, group, serial].iter().all(|d| d.iter().all(u8::is_ascii_digit)) || !on_number_end(text, i + 11, b".,-") {
        return None;
    }
    let (area, group, serial) = (value(area), value(group), value(serial));
    if group == 0 {
        return None;
    }
    if area >= 900 {
        return matches!(group, 50..=65 | 70..=88 | 90..=92 | 94..=99).then_some(PiiKind::UsItin);
    }
    (area != 0 && area != 666 && serial != 0).then_some(PiiKind::UsSsn)
}

const NHS_GROUPS: &[usize] = &[3, 3, 4];
const STEUER_GROUPS: &[usize] = &[2, 3, 3, 3];
const JOINERS: &[u8] = b".,- ";

/// NHS numbers and Steuer-IDs of `kinds`: the Steuer-ID scan first, as
/// in the Java.
pub(crate) fn scan_eu(text: &str, kinds: PiiKinds, hits: &mut Hits) {
    if kinds.contains(PiiKind::DeSteuerId) {
        scan_grouped(text, STEUER_GROUPS, PiiKind::DeSteuerId, hits);
    }
    if kinds.contains(PiiKind::UkNhs) {
        scan_grouped(text, NHS_GROUPS, PiiKind::UkNhs, hits);
    }
}

fn scan_grouped(text: &str, groups: &[usize], kind: PiiKind, hits: &mut Hits) {
    let b = text.as_bytes();
    let length: usize = groups.iter().sum();
    let mut value = [0u8; 11];
    let value = &mut value[..length];
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() || !on_number_start(text, i) {
            i += 1;
            continue;
        }
        let end = read_grouped(text, i, groups, value).or_else(|| read_bare(text, i, value));
        let valid = |value: &[u8]| if kind == PiiKind::UkNhs { valid_nhs(value) } else { valid_steuer_id(value) };
        match end {
            Some(end) if valid(value) => {
                hits.add(i, end, kind, PiiConfidence::Checksum, |s| {
                    s.extend(value.iter().map(|&d| (b'0' + d) as char));
                });
                i = end;
            }
            _ => i += 1,
        }
    }
}

/// Reads groups split by one separator, a space or a hyphen, the same
/// one throughout, into `value`; the end, or `None`.
fn read_grouped(text: &str, start: usize, groups: &[usize], value: &mut [u8]) -> Option<usize> {
    let b = text.as_bytes();
    let mut p = start;
    let mut n = 0;
    let mut separator = 0;
    for (g, &size) in groups.iter().enumerate() {
        if g > 0 {
            let c = *b.get(p)?;
            if (c != b' ' && c != b'-') || (g > 1 && c != separator) {
                return None;
            }
            separator = c;
            p += 1;
        }
        for _ in 0..size {
            let c = *b.get(p).filter(|c| c.is_ascii_digit())?;
            value[n] = c - b'0';
            n += 1;
            p += 1;
        }
    }
    on_number_end(text, p, JOINERS).then_some(p)
}

/// Reads the digits with no separator into `value`; the end, or `None`.
fn read_bare(text: &str, start: usize, value: &mut [u8]) -> Option<usize> {
    let b = text.as_bytes();
    let end = start + value.len();
    // The digits are checked first, so `end` is a character boundary.
    let digits = b.get(start..end).filter(|d| d.iter().all(u8::is_ascii_digit))?;
    if !on_number_end(text, end, JOINERS) {
        return None;
    }
    for (v, &c) in value.iter_mut().zip(digits) {
        *v = c - b'0';
    }
    Some(end)
}

/// Mod 11 with weights 10 down to 2; a check of 10 is never issued, and
/// ten equal digits are refused.
fn valid_nhs(v: &[u8]) -> bool {
    if v.iter().all(|&d| d == v[0]) {
        return false;
    }
    let sum: u32 = v[..9].iter().enumerate().map(|(i, &d)| u32::from(d) * (10 - i as u32)).sum();
    let check = match 11 - sum % 11 {
        11 => 0,
        c => c,
    };
    check != 10 && check == u32::from(v[9])
}

/// ISO 7064 MOD 11,10 over the first ten digits, no leading zero, and
/// among them exactly one digit twice or three times, never three in a
/// row.
fn valid_steuer_id(v: &[u8]) -> bool {
    if v[0] == 0 || !repetition_valid(&v[..10]) {
        return false;
    }
    let mut product = 10u32;
    for &d in &v[..10] {
        let mut sum = (u32::from(d) + product) % 10;
        if sum == 0 {
            sum = 10;
        }
        product = sum * 2 % 11;
    }
    let check = match 11 - product {
        10 => 0,
        c => c,
    };
    check == u32::from(v[10])
}

fn repetition_valid(v: &[u8]) -> bool {
    let mut counts = [0u8; 10];
    for &d in v {
        counts[d as usize] += 1;
    }
    let mut repeated = None;
    for (digit, &count) in counts.iter().enumerate() {
        if count < 2 {
            continue;
        }
        if repeated.is_some() || count > 3 {
            return false;
        }
        repeated = Some(digit as u8);
    }
    let Some(r) = repeated else { return false };
    counts[r as usize] < 3 || !v.windows(3).any(|w| w == [r, r, r])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nhs_check() {
        assert!(valid_nhs(&[9, 4, 3, 4, 7, 6, 5, 9, 1, 9]));
        assert!(!valid_nhs(&[9, 4, 3, 4, 7, 6, 5, 9, 1, 8]));
        assert!(!valid_nhs(&[0; 10]));
    }

    #[test]
    fn steuer_id_check_and_repetitions() {
        assert!(valid_steuer_id(&[6, 5, 9, 2, 9, 9, 7, 0, 4, 8, 9]));
        assert!(!valid_steuer_id(&[6, 5, 9, 2, 9, 9, 7, 0, 4, 8, 8]));
        assert!(repetition_valid(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 1]));
        assert!(repetition_valid(&[1, 1, 2, 1, 3, 4, 5, 6, 7, 8]));
        assert!(!repetition_valid(&[1, 1, 1, 2, 3, 4, 5, 6, 7, 8]));
        assert!(!repetition_valid(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]));
        assert!(!repetition_valid(&[1, 1, 2, 2, 3, 4, 5, 6, 7, 8]));
        assert!(!repetition_valid(&[1, 1, 1, 1, 3, 4, 5, 6, 7, 8]));
    }
}
