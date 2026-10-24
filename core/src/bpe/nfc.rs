//! NFC for a byte-level BPE whose tokenizer file normalizes with it, as
//! Qwen2's does, writing the source of each byte, so that spans point
//! into the caller's text.

use std::cell::Cell;

use unicode_normalization_alignments::{IsNormalized, UnicodeNormalization, is_nfc_quick};

use crate::tokenizer::Span;

/// The parts of `s` NFC can change, by byte range: each run of
/// characters that are not ASCII, with the ASCII character before it,
/// which marks after it can compose with. No composition joins a
/// character to an ASCII one after it, and an ASCII character stops the
/// reordering of marks, so the text outside these runs stays as it is.
fn runs(s: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    let b = s.as_bytes();
    let mut i = 0;
    std::iter::from_fn(move || {
        let a = i + b[i..].iter().position(|&x| x >= 0x80)?;
        let end = a + b[a..].iter().position(|&x| x < 0x80).unwrap_or(b.len() - a);
        let start = if a > i { a - 1 } else { a };
        i = end;
        Some((start, end))
    })
}

fn run_is_nfc(s: &str) -> bool {
    match is_nfc_quick(s.chars()) {
        IsNormalized::Yes => true,
        IsNormalized::No => false,
        IsNormalized::Maybe => s.chars().eq(s.nfc().map(|(c, _)| c)),
    }
}

/// Whether `s` is in NFC already.
pub(super) fn is_nfc(s: &str) -> bool {
    runs(s).all(|(a, b)| run_is_nfc(&s[a..b]))
}

/// `s` as it is, each character from itself; `at` is where `s` starts
/// in the caller's text. With `sources` `None`, no source is written.
fn copy(s: &str, at: usize, out: &mut String, sources: &mut Option<&mut Vec<Span>>) {
    out.push_str(s);
    let Some(sources) = sources else { return };
    for (i, c) in s.char_indices() {
        let from = (at + i) as u32;
        sources.extend(std::iter::repeat_n([from, from + c.len_utf8() as u32], c.len_utf8()));
    }
}

/// `piece` in NFC into `out`, and in `sources` the span in the caller's
/// text of each byte written, `base` being where `piece` starts there.
/// A character NFC composes from several comes from all of them, where
/// upstream's offsets keep the first alone; one NFC writes beside another
/// comes from the same source. With `sources` `None`, only the text is
/// written.
pub(super) fn aligned(piece: &str, base: usize, out: &mut String, mut sources: Option<&mut Vec<Span>>) {
    out.clear();
    if let Some(s) = sources.as_mut() {
        s.clear();
    }
    let mut done = 0;
    for (a, b) in runs(piece) {
        copy(&piece[done..a], base + done, out, &mut sources);
        let run = &piece[a..b];
        if run_is_nfc(run) {
            copy(run, base + a, out, &mut sources);
        } else {
            compose(run, base + a, out, &mut sources);
        }
        done = b;
    }
    copy(&piece[done..], base + done, out, &mut sources);
}

fn compose(run: &str, base: usize, out: &mut String, sources: &mut Option<&mut Vec<Span>>) {
    let Some(sources) = sources else {
        out.extend(run.nfc().map(|(c, _)| c));
        return;
    };
    let starts: Vec<(usize, char)> = run.char_indices().collect();
    let mut next = 0;
    let mut last: Span = [base as u32; 2];
    // A change of 0 is one character for one, -n one for n + 1, and 1
    // one more beside the last.
    for (c, change) in run.nfc() {
        if change <= 0 && next < starts.len() {
            let n = change.unsigned_abs().min(starts.len() - 1 - next);
            let (z, zc) = starts[next + n];
            last = [(base + starts[next].0) as u32, (base + z + zc.len_utf8()) as u32];
            next += 1 + n;
        }
        out.push(c);
        sources.extend(std::iter::repeat_n(last, c.len_utf8()));
    }
}

/// The largest buffers a thread keeps between texts.
const KEPT: usize = 64 << 10;

thread_local! {
    static TEXT: Cell<String> = const { Cell::new(String::new()) };
    static SOURCES: Cell<Vec<Span>> = const { Cell::new(Vec::new()) };
}

/// This thread's buffers, which `keep` gives back.
pub(super) fn buffers() -> (String, Vec<Span>) {
    (TEXT.take(), SOURCES.take())
}

pub(super) fn keep(text: String, sources: Vec<Span>) {
    if text.capacity() <= KEPT {
        TEXT.set(text);
    }
    if sources.capacity() <= KEPT {
        SOURCES.set(sources);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composed_characters_come_from_all_their_sources() {
        let (mut s, mut src) = (String::new(), Vec::new());
        // e, U+0302 and U+0301 compose to U+1EBF; then a lone x.
        aligned("ae\u{302}\u{301}x", 10, &mut s, Some(&mut src));
        assert_eq!(s, "a\u{1ebf}x");
        assert_eq!(src, [[10, 11], [11, 16], [11, 16], [11, 16], [16, 17]]);
        // Already composed: one for one.
        aligned("\u{e9}", 0, &mut s, Some(&mut src));
        assert_eq!((s.as_str(), src.as_slice()), ("\u{e9}", &[[0, 2], [0, 2]][..]));
        // U+0344 decomposes to two marks: the first composes with the a,
        // and the second is written beside it.
        aligned("a\u{344}", 0, &mut s, Some(&mut src));
        assert_eq!(s, "\u{e4}\u{301}");
        assert_eq!(src, [[0, 3]; 4]);
        assert!(is_nfc("plain") && is_nfc("\u{e9}") && !is_nfc("e\u{301}") && !is_nfc("\u{344}"));
        assert!(is_nfc("x \u{e9}t\u{e9} y") && !is_nfc("x \u{e9}te\u{301} y"));
        // Only the run that needs it is composed, around text that does not.
        aligned("ab e\u{301}\u{e9} cd", 0, &mut s, Some(&mut src));
        assert_eq!(s, "ab \u{e9}\u{e9} cd");
        assert_eq!(&src[..3], [[0, 1], [1, 2], [2, 3]]);
        assert_eq!(&src[3..7], [[3, 6], [3, 6], [6, 8], [6, 8]]);
        assert_eq!(&src[7..], [[8, 9], [9, 10], [10, 11]]);
    }
}
