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

fn scan_ups(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + UPS_LENGTH <= b.len() {
        let joined = matches!(i.checked_sub(1).map(|p| b[p]), Some(b'-' | b'_' | b'/'));
        if !starts_with(b, i, b"1Z") || !on_word_start(text, i) || joined {
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
/// one space between two characters, and the UPS check digit. A compact
/// number ends at a word end that is not `-` or `_`; a spaced one also is
/// not followed by a space and a digit, which would make it part of a
/// longer spaced run.
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
    let spaced = p - start > UPS_LENGTH;
    let continues = spaced && b.get(p) == Some(&b' ') && b.get(p + 1).is_some_and(u8::is_ascii_digit);
    let ends = on_end(text, p) && !matches!(b.get(p), Some(b'-' | b'_')) && !continues;
    (check.is_ascii_digit() && ends && parcel::ups(&chars[2..UPS_LENGTH - 1]) == Some(check - b'0')).then_some(p)
}

fn checksum(algorithm: ChecksumAlgorithm) -> PiiEvidence {
    PiiEvidence::Checksum { algorithm, passed: true }
}

fn scan_numeric(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let joined = i >= 2 && b[i - 1] == b'-' && b[i - 2].is_ascii_digit();
        if !b[i].is_ascii_digit() || !on_number_start(text, i) || joined {
            i += 1;
            continue;
        }
        let run = read_run(text, i);
        if let Some((start, end, digits)) = run.candidate() {
            let algorithm = numeric_check(&digits).expect("candidate passed");
            hits.add(start, end, PiiKind::TrackingNumber, checksum(algorithm), |s| {
                s.extend(digits.iter().map(|&d| d as char));
            });
        }
        // Never read a number from inside the run: a suffix of a longer
        // grouped number is not a tracking number.
        i = run.end;
    }
}

const MAX_GROUPS: usize = 8;

/// A run of digits, compact or in groups split by single spaces where
/// every group but the last has four digits.
struct Run {
    /// Start and end of each group.
    groups: [(usize, usize); MAX_GROUPS],
    count: usize,
    /// End of the whole run, past any digits it refused.
    end: usize,
    /// The run has a usable shape and ends at a number end.
    ok: bool,
    digits: [u8; MAX_DIGITS],
    len: usize,
}

fn read_run(text: &str, start: usize) -> Run {
    let b = text.as_bytes();
    let mut run = Run { groups: [(0, 0); MAX_GROUPS], count: 0, end: start, ok: true, digits: [0; MAX_DIGITS], len: 0 };
    let mut p = start;
    loop {
        let g = p;
        while p < b.len() && b[p].is_ascii_digit() {
            if run.len < run.digits.len() {
                run.digits[run.len] = b[p];
            }
            run.len += 1;
            p += 1;
        }
        if run.count == MAX_GROUPS {
            run.ok = false;
        } else {
            run.groups[run.count] = (g, p);
            run.count += 1;
        }
        let four = p - g == 4;
        if !(four && b.get(p) == Some(&b' ') && b.get(p + 1).is_some_and(u8::is_ascii_digit)) {
            break;
        }
        p += 1;
    }
    run.end = p;
    let grouped_long = run.count > 1 && run.groups[..run.count].iter().any(|(s, e)| e - s > 4);
    let ends = on_end(text, p)
        && !matches!(b.get(p), Some(b'-' | b'.' | b',') if b.get(p + 1).is_some_and(u8::is_ascii_digit));
    run.ok &= ends && !grouped_long && run.len <= MAX_DIGITS;
    run
}

impl Run {
    /// The tracking number in the run: the whole run, or, for a grouped
    /// run, the run without a leading year (`2026 8416 ...`) or without a
    /// trailing short group (`... 3682 2 boxes`), or both.
    fn candidate(&self) -> Option<(usize, usize, Vec<u8>)> {
        if !self.ok {
            return None;
        }
        let n = self.count;
        let groups = &self.groups[..n];
        let year = n > 1 && {
            let (s, e) = groups[0];
            e - s == 4 && matches!(&self.digits[..2], b"19" | b"20")
        };
        let short_tail = n > 1 && groups[n - 1].1 - groups[n - 1].0 < 4;
        let options = [(0, n, true), (1, n, year), (0, n - 1, short_tail), (1, n - 1, year && short_tail)];
        options.into_iter().filter(|&(a, z, allowed)| allowed && a < z).find_map(|(a, z, _)| {
            let from = 4 * a;
            let to = groups[..z].iter().map(|(s, e)| e - s).sum::<usize>();
            let digits = self.digits[from..to].to_vec();
            numeric_check(&digits).map(|_| (groups[a].0, groups[z - 1].1, digits))
        })
    }
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
        20 if matches!(d[0], b'0' | b'7') => {
            gs1(&d[..19]) || {
                let mut implied = [0u8; 21];
                implied[..2].copy_from_slice(b"91");
                implied[2..].copy_from_slice(&d[..19]);
                gs1(&implied)
            }
        }
        22 if d[0] == b'9' && (b'1'..=b'5').contains(&d[1]) => gs1(&d[..21]),
        22 if d.starts_with(b"96") => gs1(&d[7..21]),
        _ => false,
    };
    passed.then_some(ChecksumAlgorithm::Gs1Mod10)
}
