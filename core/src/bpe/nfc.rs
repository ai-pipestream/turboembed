//! NFC for a byte-level BPE whose tokenizer file normalizes with it, as
//! Qwen2's does, writing the source of each byte, so that spans point
//! into the caller's text.

use std::cell::Cell;

use unicode_normalization_alignments::char::{canonical_combining_class, compose as char_compose, decompose_canonical};
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

/// `run` in NFC, each character from every source character it holds a
/// part of. NFC decomposes, puts marks in canonical order and composes;
/// each step here carries the sources along, so a mark reordered past
/// another and composed with the starter still counts, which the
/// normalizer's own change counts cannot say.
fn compose(run: &str, base: usize, out: &mut String, sources: &mut Option<&mut Vec<Span>>) {
    let Some(sources) = sources else {
        out.extend(run.nfc().map(|(c, _)| c));
        return;
    };
    let mut parts: Vec<(char, Span)> = Vec::with_capacity(run.len());
    for (i, c) in run.char_indices() {
        let from = [(base + i) as u32, (base + i + c.len_utf8()) as u32];
        decompose_canonical(c, |d| parts.push((d, from)));
    }
    // Canonical order: a stable sort of each run of marks by class.
    let mut i = 0;
    while i < parts.len() {
        let n = parts[i..].iter().take_while(|p| canonical_combining_class(p.0) != 0).count();
        parts[i..i + n].sort_by_key(|p| canonical_combining_class(p.0));
        i += n.max(1);
    }
    // Composition, as UAX #15 has it: a character joins the last starter
    // unless a character between them has its class or a higher one.
    let mut done: Vec<(char, Span)> = Vec::with_capacity(parts.len());
    let (mut starter, mut last): (Option<usize>, Option<u8>) = (None, None);
    for (c, from) in parts {
        let class = canonical_combining_class(c);
        if let Some(s) = starter
            && last.is_none_or(|l| l < class)
            && let Some(r) = char_compose(done[s].0, c)
        {
            let to: &mut (char, Span) = &mut done[s];
            *to = (r, [to.1[0].min(from[0]), to.1[1].max(from[1])]);
            continue;
        }
        if class == 0 {
            (starter, last) = (Some(done.len()), None);
        } else {
            last = Some(class);
        }
        done.push((c, from));
    }
    // Spans in text order, both ends: a character whose span starts or
    // ends before the one before it, as a mark composed past another
    // does, joins it, and the group comes from all their sources.
    let mut groups: Vec<(usize, Span)> = Vec::new();
    for (i, &(_, from)) in done.iter().enumerate() {
        let mut group = (i, from);
        while let Some(&(at, last)) = groups.last() {
            if group.1[0] >= last[0] && group.1[1] >= last[1] {
                break;
            }
            group = (at, [last[0].min(group.1[0]), last[1].max(group.1[1])]);
            groups.pop();
        }
        groups.push(group);
    }
    let mut groups = groups.into_iter().peekable();
    let mut from = [base as u32; 2];
    for (i, &(c, _)) in done.iter().enumerate() {
        if groups.peek().is_some_and(|g| g.0 == i) {
            from = groups.next().map_or(from, |g| g.1);
        }
        out.push(c);
        sources.extend(std::iter::repeat_n(from, c.len_utf8()));
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
        // and the second is written beside it, from U+0344 alone.
        aligned("a\u{344}", 0, &mut s, Some(&mut src));
        assert_eq!(s, "\u{e4}\u{301}");
        assert_eq!(src, [[0, 3], [0, 3], [1, 3], [1, 3]]);
        // The cedilla (class 202) does not block the acute (230) after it,
        // so the acute composes with the a, whose span reaches over both.
        // The cedilla's own span would then end before the a's, so the two
        // share it, to keep spans in text order.
        aligned("a\u{327}\u{301}", 0, &mut s, Some(&mut src));
        assert_eq!(s, "\u{e1}\u{327}");
        assert_eq!(src, [[0, 5]; 4]);
        // Marks out of order are reordered with their sources.
        aligned("a\u{301}\u{327}", 0, &mut s, Some(&mut src));
        assert_eq!(s, "\u{e1}\u{327}");
        assert_eq!(src, [[0, 3], [0, 3], [3, 5], [3, 5]]);
        assert!(is_nfc("plain") && is_nfc("\u{e9}") && !is_nfc("e\u{301}") && !is_nfc("\u{344}"));
        assert!(is_nfc("x \u{e9}t\u{e9} y") && !is_nfc("x \u{e9}te\u{301} y"));
        // Only the run that needs it is composed, around text that does not.
        aligned("ab e\u{301}\u{e9} cd", 0, &mut s, Some(&mut src));
        assert_eq!(s, "ab \u{e9}\u{e9} cd");
        assert_eq!(&src[..3], [[0, 1], [1, 2], [2, 3]]);
        assert_eq!(&src[3..7], [[3, 6], [3, 6], [6, 8], [6, 8]]);
        assert_eq!(&src[7..], [[8, 9], [9, 10], [10, 11]]);
    }

    /// The text written is the normalizer's NFC, sources or none, on
    /// starters, marks of several classes, Hangul and characters that
    /// decompose to more than one.
    #[test]
    fn the_text_is_nfc() {
        let chars = [
            'a', 'e', '\u{e9}', '\u{301}', '\u{302}', '\u{327}', '\u{323}', '\u{344}', '\u{1100}', '\u{1161}',
            '\u{11a8}', '\u{ac00}', '\u{1e0b}', '\u{f900}', ' ',
        ];
        let (mut s, mut src) = (String::new(), Vec::new());
        let mut text = String::new();
        for i in 0..chars.len().pow(4) {
            text.clear();
            let mut k = i;
            for _ in 0..4 {
                text.push(chars[k % chars.len()]);
                k /= chars.len();
            }
            let nfc: String = text.nfc().map(|(c, _)| c).collect();
            aligned(&text, 0, &mut s, Some(&mut src));
            assert_eq!(s, nfc, "{text:?}");
            assert_eq!(src.len(), s.len());
            assert!(src.windows(2).all(|w| w[0][0] <= w[1][0] && w[0][1] <= w[1][1]), "{text:?}: {src:?}");
            for (&[a, b], (at, c)) in
                src.iter().zip(s.char_indices().flat_map(|(i, c)| std::iter::repeat_n((i, c), c.len_utf8())))
            {
                assert!(
                    a < b && text.is_char_boundary(a as usize) && text.is_char_boundary(b as usize),
                    "{text:?} {at} {c:?}"
                );
            }
            aligned(&text, 0, &mut s, None);
            assert_eq!(s, nfc, "{text:?}");
        }
    }
}
