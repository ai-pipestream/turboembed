//! The single pass shared by the normalizers that replace or delete single
//! code points: dashes, quotes, bullets, ellipsis, digits and invisible
//! characters.
//!
//! Each of those normalizers changes only non-ASCII code points, so the
//! scan skips ASCII bytes without decoding them and asks the mapping only
//! about the non-ASCII characters it meets. Text that is all ASCII is
//! left untouched at the cost of one byte scan. Stretches the mapping
//! leaves alone are copied by the builder in one piece when the next
//! change (or the end of the text) is reached.

use turbo_analysis::ViewBuilder;

/// Copies `text` into `out`, writing `f(c)` in place of every non-ASCII
/// character `c` for which `f` returns `Some`. ASCII characters are always
/// kept and `f` is never called for them, so `f` must not map an ASCII
/// character: no table that uses this helper has one.
///
/// Neighbouring replacements merge into one edit in the alignment (the
/// builder joins touching edits), so a span inside a run of replaced
/// characters maps back to the whole run.
pub fn substitute(text: &str, out: &mut ViewBuilder<'_>, f: impl Fn(char) -> Option<&'static str>) {
    for_each_non_ascii(text, |at, c| {
        if let Some(with) = f(c) {
            out.keep(at);
            out.replace(at + c.len_utf8(), with);
        }
    });
}

/// Copies `text` into `out` without the non-ASCII characters for which
/// `f` returns true. As with [`substitute`], ASCII is always kept and `f`
/// is only asked about non-ASCII characters. Each deletion is an edit to
/// the empty string; neighbouring deletions merge into one.
pub fn remove(text: &str, out: &mut ViewBuilder<'_>, f: impl Fn(char) -> bool) {
    for_each_non_ascii(text, |at, c| {
        if f(c) {
            out.keep(at);
            out.replace(at + c.len_utf8(), "");
        }
    });
}

/// Calls `visit(offset, c)` for each non-ASCII character of `text`, in
/// order. ASCII bytes are skipped a byte at a time without decoding;
/// a lead byte of a multi-byte sequence is decoded from the slice.
#[inline]
pub fn for_each_non_ascii(text: &str, mut visit: impl FnMut(usize, char)) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] < 0x80 {
            i += 1;
            continue;
        }
        // `i` is always on a character boundary: it only moves past ASCII
        // bytes or past whole characters.
        let c = decode_at(text, i);
        visit(i, c);
        i += c.len_utf8();
    }
}

/// The encoded length of the character whose first byte is `lead`. `lead`
/// must be the first byte of a character, not a continuation byte.
#[inline]
pub fn utf8_len(lead: u8) -> usize {
    match lead {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

/// The character that starts at byte `at` of `text`. `at` must be a
/// character boundary inside the text.
#[inline]
pub fn decode_at(text: &str, at: usize) -> char {
    match text[at..].chars().next() {
        Some(c) => c,
        None => panic!("no character at byte {at} of a {}-byte text", text.len()),
    }
}

/// The character that ends at byte `at` of `text`, and the byte where it
/// starts. `at` must be a character boundary greater than zero.
#[inline]
pub fn decode_before(text: &str, at: usize) -> (usize, char) {
    match text[..at].char_indices().next_back() {
        Some(found) => found,
        None => panic!("no character before byte {at}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str, f: impl Fn(char) -> Option<&'static str>) -> (String, Vec<(u32, u32, u32, u32)>) {
        let mut b = ViewBuilder::new(text);
        substitute(text, &mut b, f);
        let (out, a) = b.finish();
        (out, a.edits().iter().map(|e| (e.parent.start, e.parent.end, e.child.start, e.child.end)).collect())
    }

    #[test]
    fn ascii_text_is_never_offered_to_the_mapping() {
        let (out, edits) = run("plain ASCII - with 'quotes'", |c| panic!("asked about {c:?}"));
        assert_eq!(out, "plain ASCII - with 'quotes'");
        assert!(edits.is_empty());
    }

    #[test]
    fn replacements_are_written_where_the_mapping_fires() {
        // U+00E9 (2 bytes) stays, U+2026 (3 bytes) becomes "...".
        let (out, edits) = run("caf\u{E9}\u{2026} ok", |c| (c == '\u{2026}').then_some("..."));
        assert_eq!(out, "caf\u{E9}... ok");
        assert_eq!(edits, [(5, 8, 5, 8)]);
    }

    #[test]
    fn neighbouring_replacements_form_one_edit() {
        let (out, edits) = run("\u{201C}\u{201D}x", |c| Some(if c == '\u{201C}' { "<" } else { ">" }));
        assert_eq!(out, "<>x");
        assert_eq!(edits, [(0, 6, 0, 2)]);
    }

    #[test]
    fn remove_deletes_and_keeps_the_rest() {
        let text = "a\u{200B}b\u{FEFF}\u{200B}c\u{1F600}";
        let mut b = ViewBuilder::new(text);
        remove(text, &mut b, |c| c == '\u{200B}' || c == '\u{FEFF}');
        let (out, a) = b.finish();
        assert_eq!(out, "abc\u{1F600}");
        let edits: Vec<_> = a.edits().iter().map(|e| (e.parent.start, e.parent.end, e.child.start)).collect();
        assert_eq!(edits, [(1, 4, 1), (5, 11, 2)]);
    }

    #[test]
    fn decoding_either_side_of_a_boundary() {
        let text = "a\u{10400}-";
        assert_eq!(decode_at(text, 1), '\u{10400}');
        assert_eq!(decode_before(text, 5), (1, '\u{10400}'));
        assert_eq!(decode_before(text, 1), (0, 'a'));
    }
}
