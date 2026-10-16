//! Payment card numbers, `CursorPiiExtractor.scanCards`: 13 to 19 digits
//! split by single spaces or hyphens, starting with 2 to 6 or with
//! UnionPay's 81, passing Luhn. Separator-delimited prefixes are tried
//! longest first, so an expiry date after the number does not hide it.

use turbo_analysis_layers::{PiiConfidence, PiiKind};

use crate::checksum::luhn;
use crate::hits::Hits;
use crate::text::{on_end, on_number_start};

const MIN_DIGITS: usize = 13;
const MAX_DIGITS: usize = 19;

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut digits = [0u8; MAX_DIGITS + 1];
    // Where each group ends in the text, and the digits before it.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let first = b[i];
        if !(matches!(first, b'2'..=b'6') || first == b'8') || !on_number_start(text, i) {
            i += 1;
            continue;
        }
        let mut count = 0;
        groups.clear();
        // The end after the last digit; the first character is one.
        let mut end = i;
        let mut previous_separator = false;
        let mut p = i;
        while p < b.len() && count <= MAX_DIGITS {
            let c = b[p];
            if c.is_ascii_digit() {
                digits[count] = c;
                count += 1;
                end = p + 1;
                previous_separator = false;
            } else if (c == b' ' || c == b'-') && !previous_separator {
                groups.push((end, count));
                previous_separator = true;
            } else {
                break;
            }
            p += 1;
        }
        groups.push((end, count));
        let found = groups.iter().rev().find(|&&(end, n)| {
            (MIN_DIGITS..=MAX_DIGITS).contains(&n)
                && (first != b'8' || digits[1] == b'1')
                && on_end(text, end)
                && luhn::valid(&digits[..n])
        });
        match found {
            Some(&(end, n)) => {
                hits.add(i, end, PiiKind::Card, PiiConfidence::Checksum, |s| {
                    s.extend(digits[..n].iter().map(|&d| d as char));
                });
                i = end;
            }
            None => i += 1,
        }
    }
}
