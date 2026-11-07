//! Identifiers gated by a context label in [`crate::window`], Luhn checked.
//! The scanners add every checked candidate; the detector's label policy
//! pass records the label fact and rejects the unlabelled ones.
//!
//! - Canadian Social Insurance Numbers, `CaIdentityPiiExtractor`: nine
//!   digits, bare or as three groups split by one space or hyphen, paired
//!   with `social insurance number`, `social insurance no` or `SIN`.
//! - IMEIs, `DevicePiiExtractor`: fifteen digits, split by spaces or by
//!   hyphens (not both), paired with `IMEI`.

use turbo_analysis_layers::{ChecksumAlgorithm, PiiEvidence, PiiKind};

use crate::checksum::{bic, luhn};
use crate::hits::Hits;
use crate::text::{before, on_end, on_number_start, on_word_start};

const SIN_DIGITS: usize = 9;
const IMEI_DIGITS: usize = 15;

pub(crate) fn scan_sin(text: &str, hits: &mut Hits) {
    scan(text, PiiKind::CaSin, hits);
}

pub(crate) fn scan_imei(text: &str, hits: &mut Hits) {
    scan(text, PiiKind::Imei, hits);
}

pub(crate) fn scan_bic(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + 8 <= b.len() {
        if !b[i].is_ascii_uppercase()
            || !on_word_start(text, i)
            || before(text, i).is_some_and(|c| matches!(c, '-' | '/' | '.' | '@'))
        {
            i += 1;
            continue;
        }
        let len = [11usize, 8].into_iter().find(|&len| {
            let end = i + len;
            end <= b.len() && on_end(text, end) && !matches!(b.get(end), Some(b'-' | b'/')) && bic::check(&b[i..end])
        });
        if let Some(len) = len {
            let end = i + len;
            hits.add_text(text, i, end, PiiKind::Bic, PiiEvidence::Shape);
            i = end;
        } else {
            i += 1;
        }
    }
}

fn scan(text: &str, kind: PiiKind, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut digits = [0u8; IMEI_DIGITS];
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() || !on_number_start(text, i) {
            i += 1;
            continue;
        }
        let (end, n) = match kind {
            PiiKind::CaSin => (read_sin(text, i, &mut digits), SIN_DIGITS),
            _ => (read_imei(text, i, &mut digits), IMEI_DIGITS),
        };
        match end {
            Some(end) if luhn::valid(&digits[..n]) => {
                hits.add(
                    i,
                    end,
                    kind,
                    PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: true },
                    |s| {
                        s.extend(digits[..n].iter().map(|&d| d as char));
                    },
                );
                i = end;
            }
            _ => i += 1,
        }
    }
}

/// No letter or digit after `end`, and no space or hyphen going on with a
/// digit.
fn on_digits_end(text: &str, end: usize) -> bool {
    let b = text.as_bytes();
    on_end(text, end) && !(end + 1 < b.len() && matches!(b[end], b'-' | b' ') && b[end + 1].is_ascii_digit())
}

/// Nine digits, with a separator after the third and the sixth when one
/// follows the third.
fn read_sin(text: &str, start: usize, digits: &mut [u8]) -> Option<usize> {
    let b = text.as_bytes();
    let mut p = start;
    let mut separator = 0;
    for (count, digit) in digits.iter_mut().enumerate().take(SIN_DIGITS) {
        if count == 3 && matches!(b.get(p), Some(b'-' | b' ')) {
            separator = b[p];
        }
        if count > 0 && count % 3 == 0 && separator != 0 {
            if b.get(p) != Some(&separator) {
                return None;
            }
            p += 1;
        }
        *digit = *b.get(p).filter(|c| c.is_ascii_digit())?;
        p += 1;
    }
    on_digits_end(text, p).then_some(p)
}

/// Fifteen digits; a space or hyphen may stand between two digits, the
/// same one each time.
fn read_imei(text: &str, start: usize, digits: &mut [u8]) -> Option<usize> {
    let b = text.as_bytes();
    let mut p = start;
    let mut n = 0;
    let mut separator = 0;
    while p < b.len() && n < IMEI_DIGITS {
        let c = b[p];
        if c.is_ascii_digit() {
            digits[n] = c;
            n += 1;
        } else if (c == b'-' || c == b' ') && b.get(p + 1).is_some_and(u8::is_ascii_digit) {
            if separator != 0 && separator != c {
                return None;
            }
            separator = c;
        } else {
            break;
        }
        p += 1;
    }
    (n == IMEI_DIGITS && on_digits_end(text, p)).then_some(p)
}
