//! Identifiers read only after a label naming them, Luhn checked.
//!
//! - Canadian Social Insurance Numbers, `CaIdentityPiiExtractor`: nine
//!   digits, bare or as three groups split by one space or hyphen, after
//!   `social insurance number`, `social insurance no` or `SIN`.
//! - IMEIs, `DevicePiiExtractor`: fifteen digits, split by spaces or by
//!   hyphens (not both), after `IMEI`.
//!
//! Labels match in any ASCII case, at a word start, with whitespace and
//! `:`, `#` or `=` (and `.` for a SIN) between label and number.

use turbo_analysis_layers::{PiiConfidence, PiiKind};

use crate::checksum::luhn;
use crate::hits::Hits;
use crate::text::{before, on_end, on_number_start, on_word_start, whitespace};

const SIN_LABELS: [&[u8]; 3] = [b"social insurance number", b"social insurance no", b"sin"];
const SIN_DIGITS: usize = 9;
const IMEI_LABEL: &[u8] = b"imei";
const IMEI_DIGITS: usize = 15;

pub(crate) fn scan_sin(text: &str, hits: &mut Hits) {
    scan(text, PiiKind::CaSin, hits);
}

pub(crate) fn scan_imei(text: &str, hits: &mut Hits) {
    scan(text, PiiKind::Imei, hits);
}

fn scan(text: &str, kind: PiiKind, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut digits = [0u8; IMEI_DIGITS];
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() || !on_number_start(text, i) || !has_label(text, i, kind) {
            i += 1;
            continue;
        }
        let (end, n) = match kind {
            PiiKind::CaSin => (read_sin(text, i, &mut digits), SIN_DIGITS),
            _ => (read_imei(text, i, &mut digits), IMEI_DIGITS),
        };
        match end {
            Some(end) if luhn::valid(&digits[..n]) => {
                hits.add(i, end, kind, PiiConfidence::LabelledChecksum, |s| {
                    s.extend(digits[..n].iter().map(|&d| d as char));
                });
                i = end;
            }
            _ => i += 1,
        }
    }
}

/// Whether a label of `kind` ends before `start`, past the separators.
fn has_label(text: &str, start: usize, kind: PiiKind) -> bool {
    let sin = kind == PiiKind::CaSin;
    let mut end = start;
    while let Some(c) = before(text, end) {
        if !(whitespace(c) || matches!(c, ':' | '#' | '=') || (sin && c == '.')) {
            break;
        }
        end -= c.len_utf8();
    }
    let b = text.as_bytes();
    let labels: &[&[u8]] = if sin { &SIN_LABELS } else { &[IMEI_LABEL] };
    labels.iter().any(|label| {
        end >= label.len()
            && b[end - label.len()..end].eq_ignore_ascii_case(label)
            && on_word_start(text, end - label.len())
    })
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
