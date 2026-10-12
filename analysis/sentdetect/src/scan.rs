//! The byte level scans both detectors are built on: finding the next
//! line break, and trimming whitespace from either end of a stretch of
//! text.
//!
//! A line break here is `\n` or `\r`. Both are ASCII, and in UTF-8 no byte
//! of a multi-byte character is below 0x80, so a scan over bytes finds
//! exactly the characters a scan over code points would. Whitespace is
//! decoded per character only where a byte is not ASCII; every character
//! with the `White_Space` property or Java's whitespace classes is in the
//! Basic Multilingual Plane, so testing whole code points gives the same
//! answer as OpenNLP's test on each UTF-16 code unit (a surrogate half is
//! never whitespace).

use turbo_unicode::WhitespaceMode;

const ONES: u64 = 0x0101_0101_0101_0101;
const HIGHS: u64 = 0x8080_8080_8080_8080;
const LF_LANES: u64 = ONES * b'\n' as u64;
const CR_LANES: u64 = ONES * b'\r' as u64;

/// A mask with the high bit set in the lowest zero byte of `x`. Bytes above
/// that one may be flagged too (the subtraction borrows into them), but the
/// lowest flagged byte is always a real zero.
#[inline]
fn zero_bytes(x: u64) -> u64 {
    x.wrapping_sub(ONES) & !x & HIGHS
}

/// The offset of the first `\n` or `\r` at or after `from`, or the length
/// of `bytes` when there is none. Eight bytes are tested per step.
#[inline]
pub(crate) fn next_break(bytes: &[u8], from: usize) -> usize {
    let mut i = from;
    while let Some(chunk) = bytes.get(i..i + 8) {
        let word = u64::from_le_bytes(chunk.try_into().expect("eight bytes"));
        let hits = zero_bytes(word ^ LF_LANES) | zero_bytes(word ^ CR_LANES);
        if hits != 0 {
            return i + (hits.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < bytes.len() {
        if bytes[i] == b'\n' || bytes[i] == b'\r' {
            return i;
        }
        i += 1;
    }
    bytes.len()
}

/// Decodes the character that starts at byte `i` of valid UTF-8, with its
/// length in bytes. The byte at `i` is not ASCII.
#[inline]
fn decode_multibyte(bytes: &[u8], i: usize) -> (u32, usize) {
    let b0 = u32::from(bytes[i]);
    let b1 = u32::from(bytes[i + 1] & 0x3F);
    if b0 < 0xE0 {
        return (((b0 & 0x1F) << 6) | b1, 2);
    }
    let b2 = u32::from(bytes[i + 2] & 0x3F);
    if b0 < 0xF0 {
        return (((b0 & 0x0F) << 12) | (b1 << 6) | b2, 3);
    }
    let b3 = u32::from(bytes[i + 3] & 0x3F);
    (((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3, 4)
}

/// The offset of the first character in `text[start..end]` that is not
/// whitespace in `mode`, or `end` when every character is. `start` and
/// `end` are character boundaries.
#[inline]
pub(crate) fn skip_whitespace(text: &str, start: usize, end: usize, mode: WhitespaceMode) -> usize {
    let bytes = text.as_bytes();
    let mut i = start;
    while i < end {
        let b = bytes[i];
        if b < 0x80 {
            if !mode.is_ascii_whitespace(b) {
                return i;
            }
            i += 1;
        } else {
            let (cp, len) = decode_multibyte(bytes, i);
            if !mode.is_whitespace(cp) {
                return i;
            }
            i += len;
        }
    }
    end
}

/// The offset just past the last character in `text[start..end]` that is
/// not whitespace in `mode`, or `start` when every character is. `start`
/// and `end` are character boundaries.
#[inline]
pub(crate) fn skip_whitespace_back(text: &str, start: usize, end: usize, mode: WhitespaceMode) -> usize {
    let bytes = text.as_bytes();
    let mut j = end;
    while j > start {
        let b = bytes[j - 1];
        if b < 0x80 {
            if !mode.is_ascii_whitespace(b) {
                return j;
            }
            j -= 1;
        } else {
            let mut lead = j - 1;
            while bytes[lead] & 0xC0 == 0x80 {
                lead -= 1;
            }
            let (cp, _) = decode_multibyte(bytes, lead);
            if !mode.is_whitespace(cp) {
                return j;
            }
            j = lead;
        }
    }
    start
}

/// `text[start..end]` with whitespace in `mode` removed from both ends, as
/// `Span.trim` does, or `None` when nothing remains.
#[inline]
pub(crate) fn trim(text: &str, start: usize, end: usize, mode: WhitespaceMode) -> Option<(usize, usize)> {
    let first = skip_whitespace(text, start, end, mode);
    if first == end {
        return None;
    }
    Some((first, skip_whitespace_back(text, first, end, mode)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use WhitespaceMode::{Legacy, Unicode};

    #[test]
    fn finds_every_break_at_every_alignment() {
        // A break at each position of a buffer longer than two words, so
        // both the word loop and the tail loop are exercised.
        for len in 0..40 {
            for at in 0..len {
                for brk in *b"\n\r" {
                    let mut bytes = vec![b'x'; len];
                    bytes[at] = brk;
                    assert_eq!(next_break(&bytes, 0), at, "len {len} at {at}");
                    assert_eq!(next_break(&bytes, at + 1), len);
                }
            }
            assert_eq!(next_break(&vec![b'a'; len], 0), len);
        }
    }

    #[test]
    fn bytes_next_to_a_break_are_not_breaks() {
        // 0x0B, 0x0C, 0x0E and 0x8A differ from LF or CR in one bit, and
        // 0x09 sits just below LF: none of them is a break.
        let bytes = [0x09, 0x0B, 0x0C, 0x0E, 0x8A, 0x8D, 0x00, 0xFF, 0x0A];
        assert_eq!(next_break(&bytes, 0), 8);
        // The lowest of two breaks in one word wins, whatever lies above it.
        let bytes = *b"ab\rc\nd\n\r";
        assert_eq!(next_break(&bytes, 0), 2);
        assert_eq!(next_break(&bytes, 3), 4);
    }

    #[test]
    fn breaks_inside_multibyte_text() {
        let text = "\u{1F600}\u{0A0A}\u{2028}x\r";
        // U+0A0A is E0 A8 8A: no byte of it is a line feed.
        assert_eq!(next_break(text.as_bytes(), 0), text.len() - 1);
    }

    #[test]
    fn trims_by_mode() {
        let text = "\u{85}\u{A0} a b \u{2028}\u{1C}";
        // Unicode: U+0085, U+00A0 and U+2028 are White_Space, U+001C is not.
        assert_eq!(trim(text, 0, text.len(), Unicode), Some((5, text.len())));
        // Legacy: U+0085 is not whitespace; U+001C is.
        assert_eq!(trim(text, 0, text.len(), Legacy), Some((0, 8)));
        assert_eq!(trim(" \t\u{3000}", 0, 5, Unicode), None);
        assert_eq!(trim("", 0, 0, Legacy), None);
        // A four-byte character is never whitespace, at either end.
        let text = " \u{1D400} ";
        assert_eq!(trim(text, 0, text.len(), Unicode), Some((1, 5)));
    }
}
