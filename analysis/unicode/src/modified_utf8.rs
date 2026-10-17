//! Java's modified UTF-8, the encoding `DataOutputStream.writeUTF` writes
//! and OpenNLP model files store their predicate strings in.
//!
//! It differs from UTF-8 in two ways: U+0000 is the two bytes `C0 80`, so
//! an encoded string holds no zero byte, and a character outside the BMP
//! is written as its two UTF-16 surrogates, each encoded as a three-byte
//! sequence. A lone surrogate, which a Java string can hold, encodes the
//! same way. For text inside the BMP without U+0000, modified UTF-8 and
//! UTF-8 are the same bytes.

/// Appends the three-byte encoding of a value in U+0800..=U+FFFF.
#[inline]
fn push3(out: &mut Vec<u8>, unit: u32) {
    out.extend_from_slice(&[0xE0 | (unit >> 12) as u8, 0x80 | ((unit >> 6) & 0x3F) as u8, 0x80 | (unit & 0x3F) as u8]);
}

/// Appends the modified UTF-8 encoding of one code unit or code point.
///
/// `cp` may be a surrogate code unit (U+D800..=U+DFFF), which is written
/// as three bytes like any other value up to U+FFFF; a code point above
/// U+FFFF is written as its two surrogates. `cp` must be at most U+10FFFF.
#[inline]
pub fn push_modified_utf8(out: &mut Vec<u8>, cp: u32) {
    assert!(cp <= 0x10_FFFF, "U+{cp:X} is not a code point");
    match cp {
        0 => out.extend_from_slice(&[0xC0, 0x80]),
        0x01..=0x7F => out.push(cp as u8),
        0x80..=0x7FF => out.extend_from_slice(&[0xC0 | (cp >> 6) as u8, 0x80 | (cp & 0x3F) as u8]),
        0x800..=0xFFFF => push3(out, cp),
        _ => {
            let offset = cp - 0x1_0000;
            push3(out, 0xD800 + (offset >> 10));
            push3(out, 0xDC00 + (offset & 0x3FF));
        }
    }
}

/// Appends the modified UTF-8 encoding of `s`. Runs of characters below
/// U+10000 that hold no U+0000 are copied as they are; only U+0000 and
/// four-byte characters are re-encoded.
pub fn push_str_modified_utf8(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let mut run = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == 0 {
            out.extend_from_slice(&bytes[run..i]);
            out.extend_from_slice(&[0xC0, 0x80]);
            i += 1;
            run = i;
        } else if b >= 0xF0 {
            out.extend_from_slice(&bytes[run..i]);
            let cp = (u32::from(b & 0x07) << 18)
                | (u32::from(bytes[i + 1] & 0x3F) << 12)
                | (u32::from(bytes[i + 2] & 0x3F) << 6)
                | u32::from(bytes[i + 3] & 0x3F);
            push_modified_utf8(out, cp);
            i += 4;
            run = i;
        } else {
            i += 1;
        }
    }
    out.extend_from_slice(&bytes[run..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(cp: u32) -> Vec<u8> {
        let mut out = Vec::new();
        push_modified_utf8(&mut out, cp);
        out
    }

    fn encode_str(s: &str) -> Vec<u8> {
        let mut out = Vec::new();
        push_str_modified_utf8(&mut out, s);
        out
    }

    // The expected bytes are what Java's DataOutputStream.writeUTF wrote
    // for the same strings, without its two-byte length prefix.

    #[test]
    fn boundaries_match_write_utf() {
        assert_eq!(encode(0x0000), [0xC0, 0x80]);
        assert_eq!(encode(0x0001), [0x01]);
        assert_eq!(encode(0x007F), [0x7F]);
        assert_eq!(encode(0x0080), [0xC2, 0x80]);
        assert_eq!(encode(0x07FF), [0xDF, 0xBF]);
        assert_eq!(encode(0x0800), [0xE0, 0xA0, 0x80]);
        assert_eq!(encode(0xFFFF), [0xEF, 0xBF, 0xBF]);
        assert_eq!(encode(0x10000), [0xED, 0xA0, 0x80, 0xED, 0xB0, 0x80]);
        assert_eq!(encode(0x10FFFF), [0xED, 0xAF, 0xBF, 0xED, 0xBF, 0xBF]);
    }

    #[test]
    fn lone_surrogate_is_three_bytes() {
        assert_eq!(encode(0xD800), [0xED, 0xA0, 0x80]);
        assert_eq!(encode(0xDFFF), [0xED, 0xBF, 0xBF]);
    }

    #[test]
    fn surrogate_halves_concatenate_to_the_pair() {
        let mut halves = encode(0xD83D);
        halves.extend(encode(0xDE00));
        assert_eq!(halves, encode(0x1F600));
    }

    #[test]
    fn string_matches_write_utf() {
        assert_eq!(
            encode_str("a\u{0}\u{e9}\u{1f600}b"),
            [0x61, 0xC0, 0x80, 0xC3, 0xA9, 0xED, 0xA0, 0xBD, 0xED, 0xB8, 0x80, 0x62]
        );
    }

    #[test]
    fn bmp_text_without_nul_is_utf8() {
        let text = "Grüße, 東京 \u{2028} end";
        assert_eq!(encode_str(text), text.as_bytes());
        assert_eq!(encode_str(""), b"");
    }

    #[test]
    fn string_equals_per_character_encoding() {
        let text = "\u{0}\u{0}x\u{10400}\u{10FFFF}\u{7FF}\u{800}\u{0}";
        let mut expected = Vec::new();
        for c in text.chars() {
            push_modified_utf8(&mut expected, c as u32);
        }
        assert_eq!(encode_str(text), expected);
    }

    #[test]
    fn appends_to_existing_bytes() {
        let mut out = b"pre=".to_vec();
        push_str_modified_utf8(&mut out, "\u{0}");
        assert_eq!(out, b"pre=\xC0\x80");
    }

    #[test]
    #[should_panic(expected = "U+110000 is not a code point")]
    fn code_point_above_max_panics() {
        let mut out = Vec::new();
        push_modified_utf8(&mut out, 0x11_0000);
    }
}
