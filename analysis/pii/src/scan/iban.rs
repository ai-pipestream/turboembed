//! IBANs, `CursorPiiExtractor.scanIbans`: two capitals, two check digits
//! and more capitals or digits, in groups split by single spaces, of the
//! length the country's registry entry gives, passing mod 97.

use turbo_analysis_layers::{PiiConfidence, PiiKind};

use crate::checksum::mod97;
use crate::hits::Hits;
use crate::tables::iban_lengths::registered_length;
use crate::text::{on_end, on_word_start};

const MAX_LENGTH: usize = 34;

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut compact = [0u8; MAX_LENGTH + 1];
    // Where each group ends in the text, and the characters before it.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_uppercase()
            || !on_word_start(text, i)
            || i + 3 >= b.len()
            || !b[i + 1].is_ascii_uppercase()
            || !b[i + 2].is_ascii_digit()
            || !b[i + 3].is_ascii_digit()
        {
            i += 1;
            continue;
        }
        let mut length = 0;
        groups.clear();
        let mut p = i;
        while p < b.len() && length <= MAX_LENGTH {
            let c = b[p];
            if c.is_ascii_uppercase() || c.is_ascii_digit() {
                compact[length] = c;
                length += 1;
            } else if c == b' ' && b.get(p + 1).is_some_and(|n| n.is_ascii_uppercase() || n.is_ascii_digit()) {
                groups.push((p, length));
            } else {
                break;
            }
            p += 1;
        }
        groups.push((p, length));
        let registered = registered_length(b[i], b[i + 1]);
        let found = groups
            .iter()
            .rev()
            .find(|&&(end, n)| n == registered && on_end(text, end) && mod97::remainder(&compact[..n]) == 1);
        match found {
            Some(&(end, n)) => {
                hits.add(i, end, PiiKind::Iban, PiiConfidence::Checksum, |s| {
                    s.extend(compact[..n].iter().map(|&c| c as char));
                });
                i = end;
            }
            None => i += 1,
        }
    }
}
