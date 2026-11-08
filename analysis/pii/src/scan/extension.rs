//! Telephone extensions, `PhoneExtensionPiiExtractor`: one to six digits
//! after a phone number the phone scan finds, introduced by `ext`, `ext.`,
//! `extn`, `extension`, `x` or `#` (words in any case), with spaces, a
//! comma or a colon between. The mention is the digits alone, so the
//! phone and its extension are two mentions side by side.

use turbo_analysis_layers::{PiiEvidence, PiiKind};

use super::phone;
use crate::hits::Hits;
use crate::text::on_end;

const MAX_DIGITS: usize = 6;
const WORDS: [&[u8]; 4] = [b"extension", b"extn", b"ext", b"x"];

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let mut phones = Hits::default();
    phone::scan(text, &mut phones);
    let mut ends: Vec<usize> = phones.ends().collect();
    ends.sort_unstable();
    ends.dedup();
    for end in ends {
        if let Some((start, stop)) = extension_after(text, end) {
            hits.add(start, stop, PiiKind::PhoneExtension, PiiEvidence::Shape, |s| s.push_str(&text[start..stop]));
        }
    }
}

fn skip_gap(b: &[u8], mut p: usize, max: usize) -> usize {
    let from = p;
    while p < b.len() && p - from < max && matches!(b[p], b' ' | b'\t' | b',' | b':' | b'(') {
        p += 1;
    }
    p
}

/// The digits of an extension introduced right after `end`.
fn extension_after(text: &str, end: usize) -> Option<(usize, usize)> {
    let b = text.as_bytes();
    let mut p = skip_gap(b, end, 3);
    if b.get(p) == Some(&b'#') {
        p += 1;
    } else {
        let word = WORDS.iter().find(|w| b.len() >= p + w.len() && b[p..p + w.len()].eq_ignore_ascii_case(w))?;
        // A phone never ends right before a letter, so a word always
        // follows a gap. A word does not run on into letters (`extra`),
        // and only `x` runs straight into the digits (`x2083`).
        p += word.len();
        let next = b.get(p).copied().unwrap_or(b' ');
        if next.is_ascii_alphabetic() || (word.len() > 1 && next.is_ascii_digit()) {
            return None;
        }
        if b.get(p) == Some(&b'.') {
            p += 1;
        }
    }
    let start = skip_gap(b, p, 2);
    let digits = b[start..].iter().take(MAX_DIGITS + 1).take_while(|c| c.is_ascii_digit()).count();
    let stop = start + digits;
    ((1..=MAX_DIGITS).contains(&digits) && on_end(text, stop) && b.get(stop) != Some(&b'-')).then_some((start, stop))
}
