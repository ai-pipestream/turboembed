//! Identifiers gated by a context label in [`crate::window`], Luhn checked.
//!
//! - Canadian Social Insurance Numbers, `CaIdentityPiiExtractor`: nine
//!   digits, bare or as three groups split by one space or hyphen, paired
//!   with `social insurance number`, `social insurance no` or `SIN`.
//! - IMEIs, `DevicePiiExtractor`: fifteen digits, split by spaces or by
//!   hyphens (not both), paired with `IMEI`.

use turbo_analysis::Span;
use turbo_analysis_layers::{PiiConfidence, PiiKind};

use crate::checksum::{bic, luhn};
use crate::hits::Hits;
use crate::text::{before, on_end, on_number_start, on_word_start};
use crate::window::{WindowConfig, has_any_label};

const SIN_DIGITS: usize = 9;
const IMEI_DIGITS: usize = 15;
const BIC_LABELS: [&[u8]; 6] = [b"bic/swift", b"swift/bic", b"bic code", b"swift code", b"swift", b"bic"];

pub(crate) fn scan_sin(text: &str, window: &WindowConfig, hits: &mut Hits) {
    scan(text, PiiKind::CaSin, window, hits);
}

pub(crate) fn scan_imei(text: &str, window: &WindowConfig, hits: &mut Hits) {
    scan(text, PiiKind::Imei, window, hits);
}

pub(crate) fn scan_bic(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + 8 <= b.len() {
        if !b[i].is_ascii_uppercase()
            || !on_word_start(text, i)
            || before(text, i).is_some_and(|c| matches!(c, '-' | '/' | '.' | '@'))
            || !has_any_label(text, i, &BIC_LABELS, true)
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
            hits.add(i, end, PiiKind::Bic, PiiConfidence::Format, |s| s.push_str(&text[i..end]));
            i = end;
        } else {
            i += 1;
        }
    }
}

fn scan(text: &str, kind: PiiKind, window: &WindowConfig, hits: &mut Hits) {
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
                let span = Span::new(i as u32, end as u32);
                if let Some(conf) = window.check(text, span, kind, PiiConfidence::Checksum) {
                    hits.add(i, end, kind, conf, |s| {
                        s.extend(digits[..n].iter().map(|&d| d as char));
                    });
                    i = end;
                } else {
                    i += 1;
                }
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
