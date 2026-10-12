//! Decoding one character of UTF-8, for the scanners that walk bytes and
//! only decode what is not ASCII.

/// The code point and byte length of the character that starts at byte
/// `i` of valid UTF-8. `i` must be on a character boundary.
#[inline(always)]
pub(crate) fn decode(bytes: &[u8], i: usize) -> (u32, usize) {
    let b0 = u32::from(bytes[i]);
    if b0 < 0x80 {
        return (b0, 1);
    }
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

#[cfg(test)]
mod tests {
    use super::decode;

    #[test]
    fn decodes_every_length() {
        let text = "a\u{e9}\u{20ac}\u{1f600}";
        let bytes = text.as_bytes();
        assert_eq!(decode(bytes, 0), (0x61, 1));
        assert_eq!(decode(bytes, 1), (0xE9, 2));
        assert_eq!(decode(bytes, 3), (0x20AC, 3));
        assert_eq!(decode(bytes, 6), (0x1F600, 4));
        // The first and last code point of every length.
        let edges = "\u{7f}\u{80}\u{7ff}\u{800}\u{ffff}\u{10000}\u{10ffff}";
        for (i, c) in edges.char_indices() {
            assert_eq!(decode(edges.as_bytes(), i), (c as u32, c.len_utf8()));
        }
    }
}
