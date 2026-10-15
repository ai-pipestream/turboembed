//! The byte-level steps of ftfy's encoding fix, over bytes that each
//! carry the source span they came from: putting back 0xA0 bytes that
//! became spaces (`restore_byte_a0`), standing U+FFFD in for UTF-8
//! sequences that lost a byte (`replace_lossy_sequences`), and decoding
//! UTF-8 strictly, with or without the CESU-8 and Java forms ftfy calls
//! "utf-8-variants". Ported from ftfy 6.3.1 by Robyn Speer (Apache-2.0;
//! see NOTICE).

use super::{Src, Tracked};

/// Bytes, each with the span of the source text it came from.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(crate) struct TrackedBytes {
    pub bytes: Vec<u8>,
    pub src: Vec<Src>,
}

impl TrackedBytes {
    fn push(&mut self, b: u8, src: Src) {
        self.bytes.push(b);
        self.src.push(src);
    }

    fn at(&self, i: usize) -> Option<u8> {
        self.bytes.get(i).copied()
    }

    fn matches(&self, i: usize, pattern: &[fn(u8) -> bool]) -> bool {
        pattern.iter().enumerate().all(|(k, f)| self.at(i + k).is_some_and(f))
    }
}

/// A continuation byte that is not 0x85 or 0xA0 (bytes that some
/// programs turn into a newline or a space).
fn likely_continuation(b: u8) -> bool {
    matches!(b, 0x80..=0x84 | 0x86..=0x9F | 0xA1..=0xBF)
}

fn continuation(b: u8) -> bool {
    (0x80..=0xBF).contains(&b)
}

/// A regex alternative over bytes: one test per byte.
type BytePattern<'a> = &'a [fn(u8) -> bool];

fn space(b: u8) -> bool {
    b == b' '
}

/// ftfy's `ALTERED_UTF8_RE` at `i`: the length of a UTF-8 sequence with a
/// space where 0xA0 belongs, or 0. The alternatives are tried in the
/// regex's order.
fn altered_utf8_at(b: &TrackedBytes, i: usize) -> usize {
    let alternatives: [BytePattern; 6] = [
        &[|x| matches!(x, 0xC2 | 0xC3 | 0xC5 | 0xCE | 0xD0 | 0xD9), space],
        &[|x| matches!(x, 0xE2 | 0xE3), space, likely_continuation],
        &[|x| matches!(x, 0xE0..=0xE3), likely_continuation, space],
        &[|x| x == 0xF0, space, continuation, continuation],
        &[|x| x == 0xF0, continuation, space, continuation],
        &[|x| x == 0xF0, continuation, continuation, space],
    ];
    alternatives.iter().find(|p| b.matches(i, p)).map_or(0, |p| p.len())
}

/// ftfy's `ALTERED_UTF8_RE.search`.
pub(crate) fn has_altered_utf8(b: &TrackedBytes) -> bool {
    (0..b.bytes.len()).any(|i| altered_utf8_at(b, i) > 0)
}

/// ftfy's `restore_byte_a0`: "Ã " becomes "Ã\u{A0} " unless a word in
/// Portuguese or Catalan follows; then every UTF-8 sequence with a space
/// for 0xA0 gets its 0xA0 back. An inserted 0xA0 carries the span of the
/// 0xC3 before it, so the two decode to one character from the "Ã".
pub(crate) fn restore_byte_a0(b: &TrackedBytes) -> TrackedBytes {
    let mut grave = TrackedBytes::default();
    let mut i = 0;
    while i < b.bytes.len() {
        if b.bytes[i] == 0xC3 && b.at(i + 1) == Some(b' ') {
            let rest = &b.bytes[i + 2..];
            let word = [&b" "[..], b"quele", b"quela", b"quilo", b"s "].iter().any(|w| rest.starts_with(w));
            if !word {
                grave.push(0xC3, b.src[i]);
                grave.push(0xA0, b.src[i]);
                grave.push(b' ', b.src[i + 1]);
                i += 2;
                continue;
            }
        }
        grave.push(b.bytes[i], b.src[i]);
        i += 1;
    }
    let mut out = TrackedBytes::default();
    let mut i = 0;
    while i < grave.bytes.len() {
        let len = altered_utf8_at(&grave, i).max(1);
        let restore = len > 1;
        for k in i..i + len {
            let byte = grave.bytes[k];
            out.push(if restore && byte == b' ' { 0xA0 } else { byte }, grave.src[k]);
        }
        i += len;
    }
    out
}

fn sub(b: u8) -> bool {
    b == 0x1A
}

fn sub_or_question(b: u8) -> bool {
    b == 0x1A || b == b'?'
}

fn sub_or_continuation(b: u8) -> bool {
    b == 0x1A || continuation(b)
}

fn sub_question_or_continuation(b: u8) -> bool {
    b == 0x1A || b == b'?' || continuation(b)
}

/// ftfy's `LOSSY_UTF8_RE` at `i`: the length of a UTF-8 or CESU-8
/// sequence that lost bytes to 0x1A (a sloppy code page's stand-in for
/// U+FFFD) or, once, to '?', or 0.
fn lossy_utf8_at(b: &TrackedBytes, i: usize) -> usize {
    let ed = |x: u8| x == 0xED;
    let alternatives: [BytePattern; 10] = [
        &[|x| matches!(x, 0xC2..=0xDF), sub],
        &[|x| matches!(x, 0xC2 | 0xC3), |x| x == b'?'],
        &[
            ed,
            |x| matches!(x, 0xA0..=0xAF),
            sub_or_question,
            ed,
            |x| matches!(x, 0xB0..=0xBF),
            sub_question_or_continuation,
        ],
        &[
            ed,
            |x| matches!(x, 0xA0..=0xAF),
            sub_question_or_continuation,
            ed,
            |x| matches!(x, 0xB0..=0xBF),
            sub_or_question,
        ],
        &[|x| matches!(x, 0xE0..=0xEF), sub_or_question, sub_or_continuation],
        &[|x| matches!(x, 0xE0..=0xEF), sub_or_continuation, sub_or_question],
        &[|x| matches!(x, 0xF0..=0xF4), sub_or_question, sub_or_continuation, sub_or_continuation],
        &[|x| matches!(x, 0xF0..=0xF4), sub_or_continuation, sub_or_question, sub_or_continuation],
        &[|x| matches!(x, 0xF0..=0xF4), sub_or_continuation, sub_or_continuation, sub_or_question],
        &[sub],
    ];
    alternatives.iter().find(|p| b.matches(i, p)).map_or(0, |p| p.len())
}

/// ftfy's `replace_lossy_sequences`: each lossy sequence becomes the
/// UTF-8 of U+FFFD, its three bytes carrying the span of the whole
/// sequence.
pub(crate) fn replace_lossy_sequences(b: &TrackedBytes) -> TrackedBytes {
    let mut out = TrackedBytes::default();
    let mut i = 0;
    while i < b.bytes.len() {
        let len = lossy_utf8_at(b, i);
        if len == 0 {
            out.push(b.bytes[i], b.src[i]);
            i += 1;
            continue;
        }
        let src = b.src[i..i + len].iter().copied().reduce(Src::join).expect("a match is not empty");
        for byte in "\u{FFFD}".bytes() {
            out.push(byte, src);
        }
        i += len;
    }
    out
}

/// Decodes `b` as strict UTF-8, or with `variants` also as CESU-8
/// surrogate pairs and Java's two-byte U+0000; None at the first byte
/// that does not decode, as Python's strict decoders fail. Each
/// character carries the span of its bytes.
pub(crate) fn decode_utf8(b: &TrackedBytes, variants: bool) -> Option<Tracked> {
    let bytes = &b.bytes;
    let mut out = Tracked::default();
    let mut i = 0;
    while i < bytes.len() {
        let (c, len) = if variants && bytes[i] == 0xC0 {
            (bytes.get(i + 1) == Some(&0x80)).then_some(('\0', 2))?
        } else if variants && is_cesu_pair(&bytes[i..]) {
            let cp = ((u32::from(bytes[i + 1]) & 0x0F) << 16)
                + ((u32::from(bytes[i + 2]) & 0x3F) << 10)
                + ((u32::from(bytes[i + 4]) & 0x0F) << 6)
                + (u32::from(bytes[i + 5]) & 0x3F)
                + 0x10000;
            (char::from_u32(cp)?, 6)
        } else {
            utf8_char(&bytes[i..])?
        };
        let src = b.src[i..i + len].iter().copied().reduce(Src::join).expect("a character has bytes");
        out.push(c, src);
        i += len;
    }
    Some(out)
}

/// ftfy's `CESU8_RE` at the start of `b`: a UTF-16 surrogate pair with
/// each surrogate encoded as UTF-8 on its own.
fn is_cesu_pair(b: &[u8]) -> bool {
    matches!(b, [0xED, 0xA0..=0xAF, 0x80..=0xBF, 0xED, 0xB0..=0xBF, 0x80..=0xBF, ..])
}

/// The valid UTF-8 character at the start of `b` and its length.
fn utf8_char(b: &[u8]) -> Option<(char, usize)> {
    let (len, lower, upper) = match b[0] {
        0x00..=0x7F => return Some((char::from(b[0]), 1)),
        0xC2..=0xDF => (2, 0x80, 0xBF),
        0xE0 => (3, 0xA0, 0xBF),
        0xED => (3, 0x80, 0x9F),
        0xE1..=0xEF => (3, 0x80, 0xBF),
        0xF0 => (4, 0x90, 0xBF),
        0xF4 => (4, 0x80, 0x8F),
        0xF1..=0xF3 => (4, 0x80, 0xBF),
        _ => return None,
    };
    let seq = b.get(..len)?;
    if !(lower..=upper).contains(&seq[1]) || !seq[2..].iter().all(|&x| continuation(x)) {
        return None;
    }
    let c = std::str::from_utf8(seq).ok()?.chars().next()?;
    Some((c, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracked(bytes: &[u8]) -> TrackedBytes {
        let src = (0..bytes.len() as u32).map(|i| Src { start: i, end: i + 1 }).collect();
        TrackedBytes { bytes: bytes.to_vec(), src }
    }

    #[test]
    fn a_grave_gets_its_a0_back_except_before_portuguese_words() {
        let out = restore_byte_a0(&tracked(b"voil\xC3 le"));
        assert_eq!(out.bytes, b"voil\xC3\xA0 le");
        // The inserted byte belongs to the 0xC3.
        assert_eq!(out.src[5], out.src[4]);
        // Before "quele" no byte is inserted: the space itself is taken
        // for the lost 0xA0 ("àquele").
        assert_eq!(restore_byte_a0(&tracked(b"\xC3 quele")).bytes, b"\xC3\xA0quele");
    }

    #[test]
    fn lossy_sequences_become_one_replacement_character() {
        let out = replace_lossy_sequences(&tracked(b"a\xE2\x80\x1Ab"));
        assert_eq!(out.bytes, b"a\xEF\xBF\xBDb");
        assert_eq!(out.src[1], Src { start: 1, end: 4 });
        let decoded = decode_utf8(&out, false).unwrap();
        assert_eq!(decoded.chars, vec!['a', '\u{FFFD}', 'b']);
        assert_eq!(decoded.src[1], Src { start: 1, end: 4 });
    }

    #[test]
    fn variants_decode_cesu8_and_java_null() {
        let cesu = tracked(b"\xED\xA0\xBD\xED\xB8\x8D\xC0\x80");
        assert_eq!(decode_utf8(&cesu, false), None);
        let decoded = decode_utf8(&cesu, true).unwrap();
        assert_eq!(decoded.chars, vec!['\u{1F60D}', '\0']);
        assert_eq!(decoded.src, vec![Src { start: 0, end: 6 }, Src { start: 6, end: 8 }]);
        assert_eq!(decode_utf8(&tracked(b"\xC0"), true), None);
        assert_eq!(decode_utf8(&tracked(b"\xED\xA0\x80"), true), None);
    }
}
