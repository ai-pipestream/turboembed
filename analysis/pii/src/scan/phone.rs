//! Phone numbers, `CursorPiiExtractor.scanPhones`: `+` and digits that
//! split into an assigned calling code and a national length used under
//! it, or 10 or 11 digits with a space, hyphen or parenthesis between
//! digits. Separator-delimited prefixes are tried longest first.

use turbo_analysis_layers::{PiiConfidence, PiiKind};

use crate::hits::Hits;
use crate::tables::phone_lengths::plausible_international;
use crate::text::{on_end, on_number_start};

const MAX_DIGITS: usize = 15;
const DOMESTIC_MIN: usize = 10;
const DOMESTIC_MAX: usize = 11;

/// One place the candidate may be cut: its end, the digits before it,
/// whether a separator came between digits, and the parentheses opened
/// and closed.
#[derive(Copy, Clone)]
struct Cut {
    end: usize,
    digits: usize,
    separated: bool,
    open: u8,
    close: u8,
}

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut last_end = usize::MAX;
    let mut digits = [0u8; MAX_DIGITS + 1];
    let mut cuts: Vec<Cut> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let plus = c == b'+';
        // Adjacent phones may start at the end of the one before.
        if (!plus && !c.is_ascii_digit() && c != b'(')
            || (i != last_end && !on_number_start(text, i))
            || (i > 0 && b[i - 1] == b'+')
        {
            i += 1;
            continue;
        }
        let mut count = 0;
        let mut end = i;
        let (mut open, mut close) = (0u8, 0u8);
        let mut separated = false;
        let mut previous_separator = false;
        cuts.clear();
        let mut p = if plus { i + 1 } else { i };
        while p < b.len() && count <= MAX_DIGITS {
            let ch = b[p];
            if ch.is_ascii_digit() {
                if count > 0 && p > i && !b[p - 1].is_ascii_digit() {
                    separated = true;
                }
                digits[count] = ch;
                count += 1;
                end = p + 1;
                previous_separator = false;
            } else if (ch == b' ' || ch == b'-') && !previous_separator {
                cuts.push(Cut { end, digits: count, separated, open, close });
                previous_separator = true;
            } else if ch == b'(' && open == 0 {
                open += 1;
                previous_separator = false;
            } else if ch == b')' && close == 0 && open == 1 {
                close += 1;
                end = p + 1;
                previous_separator = false;
            } else {
                break;
            }
            p += 1;
        }
        cuts.push(Cut { end, digits: count, separated, open, close });
        let found = cuts.iter().rev().find(|cut| {
            if cut.digits == 0 || cut.open != cut.close {
                return false;
            }
            let run = &digits[..cut.digits];
            let length_ok = if plus {
                cut.digits <= MAX_DIGITS && plausible_international(run)
            } else {
                (DOMESTIC_MIN..=DOMESTIC_MAX).contains(&cut.digits) && cut.separated
            };
            length_ok
                && on_end(text, cut.end)
                && !(cut.end + 1 < b.len() && b[cut.end] == b'.' && b[cut.end + 1].is_ascii_digit())
        });
        match found {
            Some(&cut) => {
                hits.add(i, cut.end, PiiKind::Phone, PiiConfidence::Format, |s| {
                    if plus {
                        s.push('+');
                    }
                    s.extend(digits[..cut.digits].iter().map(|&d| d as char));
                });
                last_end = cut.end;
                i = cut.end;
            }
            None => i += 1,
        }
    }
}
