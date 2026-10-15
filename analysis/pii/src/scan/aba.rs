//! ABA routing transit numbers, `BankingPiiExtractor`: nine contiguous
//! digits with an allocated prefix (00 to 12, 21 to 32, 61 to 72, 80) and
//! a check digit under the weights 3, 7, 1. All zeros are refused.

use turbo_analysis_layers::{PiiConfidence, PiiKind};

use crate::hits::Hits;
use crate::text::{on_end, on_number_start};

const DIGITS: usize = 9;

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + DIGITS <= b.len() {
        let end = i + DIGITS;
        let found = b[i].is_ascii_digit()
            && on_number_start(text, i)
            && b[i..end].iter().all(u8::is_ascii_digit)
            && on_number_end(text, end)
            && allocated(b[i], b[i + 1])
            && check_digit(&b[i..end]);
        if found {
            hits.add_text(text, i, end, PiiKind::AbaRouting, PiiConfidence::Checksum);
            i = end;
        } else {
            i += 1;
        }
    }
}

/// No letter or digit after, and no `.` or `,` going on with a digit.
fn on_number_end(text: &str, end: usize) -> bool {
    let b = text.as_bytes();
    on_end(text, end) && (end + 1 >= b.len() || !matches!(b[end], b'.' | b',') || !b[end + 1].is_ascii_digit())
}

fn allocated(first: u8, second: u8) -> bool {
    let prefix = (first - b'0') * 10 + (second - b'0');
    matches!(prefix, 0..=12 | 21..=32 | 61..=72 | 80)
}

fn check_digit(digits: &[u8]) -> bool {
    let sum: u32 = digits.iter().zip([3, 7, 1].iter().cycle()).map(|(&d, &w)| w * u32::from(d - b'0')).sum();
    sum > 0 && sum.is_multiple_of(10)
}
