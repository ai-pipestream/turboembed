//! Mojibake by the byte round trip of OpenNLP's `CursorArtifactDetector`:
//! a maximal run of non-ASCII characters that Windows-1252 (ISO-8859-1
//! for the C1 controls) can encode is reported when its bytes are one or
//! more strictly well-formed UTF-8 sequences, at least one of them
//! multi-byte. Such a run is most likely UTF-8 that was decoded with the
//! single-byte code page; it is a heuristic, not proof.

use turbo_normalize_layers::ArtifactKind;

use super::Artifact;
use super::codepage::CodePage;

/// Reports every run that round-trips.
pub(crate) fn scan(text: &str, out: &mut Vec<Artifact>) {
    let page = CodePage::Windows1252;
    let high = |c: char| page.byte(c).is_some_and(|b| b >= 0x80);
    let mut chars = text.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        if !high(c) {
            continue;
        }
        let mut end = start + c.len_utf8();
        while let Some(&(at, next)) = chars.peek() {
            if !high(next) {
                break;
            }
            end = at + next.len_utf8();
            chars.next();
        }
        if is_utf8(text[start..end].chars().filter_map(|c| page.byte(c))) {
            out.push(Artifact::new(start, end, ArtifactKind::Mojibake));
        }
    }
}

/// How many continuation bytes a UTF-8 lead byte takes, and the smallest
/// and largest scalar a sequence of that length may encode (RFC 3629,
/// section 3: overlong forms and values past U+10FFFF are not UTF-8).
/// `None` for a byte that cannot start a multi-byte sequence.
pub(crate) fn lead(b: u8) -> Option<(usize, u32, u32)> {
    match b {
        0xC2..=0xDF => Some((1, 0x80, 0x7FF)),
        0xE0..=0xEF => Some((2, 0x800, 0xFFFF)),
        0xF0..=0xF4 => Some((3, 0x10000, 0x10FFFF)),
        _ => None,
    }
}

/// Whether `b` is a UTF-8 continuation byte.
pub(crate) fn is_continuation(b: u8) -> bool {
    (0x80..=0xBF).contains(&b)
}

/// Whether the bytes are complete, strictly well-formed UTF-8 that
/// encodes at least one non-ASCII scalar: no overlong form, no encoded
/// surrogate, nothing past U+10FFFF, no sequence cut off at the end.
fn is_utf8(mut bytes: impl Iterator<Item = u8>) -> bool {
    let mut multi_byte = false;
    while let Some(b) = bytes.next() {
        if b < 0x80 {
            continue;
        }
        let Some((continuations, min, max)) = lead(b) else { return false };
        let mut scalar = u32::from(b) & (0x3F >> continuations);
        for _ in 0..continuations {
            match bytes.next() {
                Some(c) if is_continuation(c) => scalar = (scalar << 6) | u32::from(c & 0x3F),
                _ => return false,
            }
        }
        if scalar < min || scalar > max || (0xD800..=0xDFFF).contains(&scalar) {
            return false;
        }
        multi_byte = true;
    }
    multi_byte
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_validation_matches_the_standard_library_on_every_two_byte_pair() {
        for a in 0x80..=0xFFu8 {
            for b in 0x80..=0xFFu8 {
                let bytes = [a, b];
                assert_eq!(is_utf8(bytes.into_iter()), std::str::from_utf8(&bytes).is_ok(), "{a:02X} {b:02X}");
            }
        }
    }

    #[test]
    fn utf8_validation_matches_the_standard_library_on_three_and_four_byte_forms() {
        for a in [0xE0u8, 0xE1, 0xED, 0xEE, 0xEF, 0xF0, 0xF1, 0xF4, 0xF5] {
            for b in 0x80..=0xBFu8 {
                for c in [0x80u8, 0xBF, 0x41] {
                    let three = [a, b, c];
                    let four = [a, b, c, 0x80];
                    assert_eq!(is_utf8(three.into_iter()), std::str::from_utf8(&three).is_ok(), "{three:02X?}");
                    assert_eq!(is_utf8(four.into_iter()), std::str::from_utf8(&four).is_ok(), "{four:02X?}");
                }
            }
        }
    }
}
