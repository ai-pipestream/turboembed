//! Vehicle and parcel identifiers, `LogisticsPiiExtractor`.
//!
//! - Vehicle identification numbers (`vin`): 17 uppercase letters and
//!   digits without `I`, `O` or `Q`, at least one of each, a model year
//!   character (the tenth) that is not `U`, `Z` or `0`, and the ninth
//!   character the ISO 3779 / 49 CFR 565 check character.
//! - Tracking numbers (`tracking-number`), compact or in groups of four
//!   split by single spaces: UPS `1Z` and 16 letters or digits (UPS mod
//!   10); FedEx Express 12 digits (weights 3, 1, 7 mod 11) and FedEx Ground
//!   15 digits, both only when they fail Luhn (a Luhn-valid one is a card
//!   or an IMEI); USPS 20 digits starting `0` or `7` (GS1 mod 10, also with
//!   the `91` prefix it implies); 22 digits starting `91` to `95` (USPS, GS1 mod
//!   10 over 21 digits) or `96` (FedEx Ground, GS1 mod 10 over the 14-digit
//!   serial).

use turbo_analysis_layers::{ChecksumAlgorithm, PiiEvidence, PiiKind, PiiKinds};

use crate::checksum::{luhn, parcel, vin};
use crate::hits::Hits;
use crate::text::{on_end, on_number_start, on_word_start, starts_with};

const VIN_LENGTH: usize = 17;
const UPS_LENGTH: usize = 18;
const MAX_DIGITS: usize = 22;

/// The vehicle and tracking numbers of `kinds`.
pub(crate) fn scan(text: &str, kinds: PiiKinds, hits: &mut Hits) {
    if kinds.contains(PiiKind::Vin) {
        scan_vin(text, hits);
    }
    if kinds.contains(PiiKind::TrackingNumber) {
        scan_ups(text, hits);
        scan_numeric(text, hits);
    }
}

fn scan_vin(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + VIN_LENGTH <= b.len() {
        let found = (b[i].is_ascii_uppercase() || b[i].is_ascii_digit())
            && on_word_start(text, i)
            && !matches!(i.checked_sub(1).map(|p| b[p]), Some(b'-' | b'_' | b'/'))
            && vin_at(text, i);
        if found {
            let algorithm = ChecksumAlgorithm::Vin;
            hits.add_text(text, i, i + VIN_LENGTH, PiiKind::Vin, PiiEvidence::Checksum { algorithm, passed: true });
            i += VIN_LENGTH;
        } else {
            i += 1;
        }
    }
}

fn vin_at(text: &str, i: usize) -> bool {
    let b = text.as_bytes();
    let Ok(w) = <&[u8; VIN_LENGTH]>::try_from(&b[i..i + VIN_LENGTH]) else { return false };
    let end = i + VIN_LENGTH;
    on_end(text, end)
        && !matches!(b.get(end), Some(b'-' | b'_'))
        && w.iter().all(|&c| vin::value(c).is_some())
        && w.iter().any(u8::is_ascii_digit)
        && w.iter().any(u8::is_ascii_uppercase)
        && !matches!(w[9], b'U' | b'Z' | b'0')
        && vin::valid(w)
}

/// No letter or digit at `end`, nor a space going on with a digit.
fn on_group_end(text: &str, end: usize) -> bool {
    let b = text.as_bytes();
    on_end(text, end) && !(b.get(end) == Some(&b' ') && b.get(end + 1).is_some_and(u8::is_ascii_digit))
}

fn scan_ups(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + UPS_LENGTH <= b.len() {
        if !starts_with(b, i, b"1Z") || !on_word_start(text, i) {
            i += 1;
            continue;
        }
        let mut chars = [0u8; UPS_LENGTH];
        match read_ups(text, i, &mut chars) {
            Some(end) => {
                hits.add(i, end, PiiKind::TrackingNumber, checksum(ChecksumAlgorithm::UpsMod10), |s| {
                    s.extend(chars.iter().map(|&c| c as char));
                });
                i = end;
            }
            None => i += 1,
        }
    }
}

/// `1Z` and 16 uppercase letters or digits, the last a digit, with at most
/// one space between two characters, and the UPS check digit.
fn read_ups(text: &str, start: usize, chars: &mut [u8; UPS_LENGTH]) -> Option<usize> {
    let b = text.as_bytes();
    let mut p = start;
    let mut n = 0;
    while n < UPS_LENGTH {
        if n > 0 && b.get(p) == Some(&b' ') {
            p += 1;
        }
        let c = *b.get(p).filter(|c| c.is_ascii_uppercase() || c.is_ascii_digit())?;
        chars[n] = c;
        n += 1;
        p += 1;
    }
    let check = chars[UPS_LENGTH - 1];
    (check.is_ascii_digit() && on_group_end(text, p) && parcel::ups(&chars[2..UPS_LENGTH - 1]) == Some(check - b'0'))
        .then_some(p)
}

fn checksum(algorithm: ChecksumAlgorithm) -> PiiEvidence {
    PiiEvidence::Checksum { algorithm, passed: true }
}

fn scan_numeric(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut digits = [0u8; MAX_DIGITS];
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() || !on_number_start(text, i) {
            i += 1;
            continue;
        }
        let Some((end, n)) = read_digits(text, i, &mut digits) else {
            i += 1;
            continue;
        };
        if let Some(algorithm) = numeric_check(&digits[..n]) {
            hits.add(i, end, PiiKind::TrackingNumber, checksum(algorithm), |s| {
                s.extend(digits[..n].iter().map(|&d| d as char));
            });
        }
        i = end;
    }
}

/// A run of digits, compact or in groups of four split by single spaces
/// (the last group may be shorter), of at most 22 digits, at a number end.
fn read_digits(text: &str, start: usize, digits: &mut [u8; MAX_DIGITS]) -> Option<(usize, usize)> {
    let b = text.as_bytes();
    let mut p = start;
    let mut n = 0;
    let mut grouped = false;
    let mut group = 0;
    loop {
        match b.get(p) {
            Some(c) if c.is_ascii_digit() => {
                if n == MAX_DIGITS || (grouped && group == 4) {
                    return None;
                }
                digits[n] = *c;
                n += 1;
                group += 1;
                p += 1;
            }
            Some(b' ') if b.get(p + 1).is_some_and(u8::is_ascii_digit) && group == 4 && (grouped || n == 4) => {
                grouped = true;
                group = 0;
                p += 1;
            }
            _ => break,
        }
    }
    (on_end(text, p) && !matches!(b.get(p), Some(b'-' | b'.' | b',') if b.get(p + 1).is_some_and(u8::is_ascii_digit)))
        .then_some((p, n))
}

/// The check that a run of `digits` passes as a tracking number.
fn numeric_check(d: &[u8]) -> Option<ChecksumAlgorithm> {
    let last = d.last()? - b'0';
    let gs1 = |data: &[u8]| parcel::gs1_mod10(data) == Some(last);
    // A Luhn-valid 12- or 15-digit number is read as a card or an IMEI,
    // not as a FedEx number.
    let passed = match d.len() {
        12 if !luhn::valid(d) => {
            return (parcel::fedex_mod11(&d[..11]) == Some(last)).then_some(ChecksumAlgorithm::FedexMod11);
        }
        15 => gs1(&d[..14]) && !luhn::valid(d),
        // The USPS 20-digit numbers in use start `0` or `7` (certified
        // mail); a dated build number such as `2024 0131 ...` does not.
        20 if matches!(d[0], b'0' | b'7') => gs1(&d[..19]) || {
            let mut implied = [0u8; 21];
            implied[..2].copy_from_slice(b"91");
            implied[2..].copy_from_slice(&d[..19]);
            gs1(&implied)
        },
        22 if d[0] == b'9' && (b'1'..=b'5').contains(&d[1]) => gs1(&d[..21]),
        22 if d.starts_with(b"96") => gs1(&d[7..21]),
        _ => false,
    };
    passed.then_some(ChecksumAlgorithm::Gs1Mod10)
}
