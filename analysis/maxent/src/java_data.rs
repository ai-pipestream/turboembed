//! Reading binary data written by Java's `DataOutputStream`: big-endian
//! integers and doubles, and strings in Java's modified UTF-8.
//!
//! Modified UTF-8 encodes each UTF-16 code unit on its own: U+0000 as the
//! two bytes `C0 80`, units up to U+07FF in two bytes, the rest in three.
//! A character outside the Basic Multilingual Plane is therefore two
//! three-byte sequences, one per surrogate, and a lone surrogate can be
//! represented. Like `DataInputStream.readUTF`, the decoder here accepts
//! overlong forms and a raw zero byte, and refuses anything else that is
//! not one of the three sequence shapes.

use std::borrow::Cow;

use crate::error::{Error, Result};

/// The prefix OpenNLP's `ModelParameterChunker` writes when a string's
/// modified UTF-8 is longer than the 65535 bytes `writeUTF` allows. The
/// prefix is followed by the number of chunks, and the chunks follow as
/// ordinary strings.
pub const CHUNKED_SIGNATURE: &[u8] = b"CHUNKED-MODEL-PARAMS:";

/// A cursor over Java `DataInput` data.
#[derive(Debug, Clone)]
pub struct JavaData<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> JavaData<'a> {
    pub fn new(data: &'a [u8]) -> JavaData<'a> {
        JavaData { data, pos: 0 }
    }

    /// The offset of the next byte to read.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// The number of bytes not yet read.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn bytes(&mut self, n: usize, what: &'static str) -> Result<&'a [u8]> {
        let Some(b) = self.data.get(self.pos..self.pos + n) else {
            return Err(Error::Data { offset: self.pos, message: what });
        };
        self.pos += n;
        Ok(b)
    }

    /// `DataInput.readInt`: a big-endian `i32`.
    pub fn read_int(&mut self) -> Result<i32> {
        let b = self.bytes(4, "the data ends inside an int")?;
        Ok(i32::from_be_bytes(b.try_into().expect("four bytes")))
    }

    /// `DataInput.readDouble`: a big-endian IEEE 754 `f64`.
    pub fn read_double(&mut self) -> Result<f64> {
        let b = self.bytes(8, "the data ends inside a double")?;
        Ok(f64::from_be_bytes(b.try_into().expect("eight bytes")))
    }

    /// `DataInput.readUTF`, without decoding: a big-endian `u16` length,
    /// then that many bytes of modified UTF-8, which are checked to be
    /// well formed and returned as they are.
    pub fn read_utf(&mut self) -> Result<&'a [u8]> {
        let len = self.bytes(2, "the data ends inside a string length")?;
        let len = usize::from(u16::from_be_bytes([len[0], len[1]]));
        let start = self.pos;
        let bytes = self.bytes(len, "the data ends inside a string")?;
        if let Err(at) = check_modified_utf8(bytes) {
            return Err(Error::Data { offset: start + at, message: "malformed modified UTF-8" });
        }
        Ok(bytes)
    }

    /// `DataInput.readUTF`, decoded to a `String`. A lone surrogate, which
    /// a Rust string cannot hold, becomes U+FFFD.
    pub fn read_utf_string(&mut self) -> Result<String> {
        self.read_utf().map(decode_modified_utf8)
    }

    /// OpenNLP's `ModelParameterChunker.readUTF`: a string that does not
    /// start with [`CHUNKED_SIGNATURE`] is returned as it is. One that does
    /// gives a chunk count after the signature, and the result is that many
    /// following strings joined. Joining the modified UTF-8 bytes is the
    /// same as joining the strings, because each code unit is encoded on
    /// its own.
    pub fn read_chunked_utf(&mut self) -> Result<Cow<'a, [u8]>> {
        let start = self.pos;
        let first = self.read_utf()?;
        let Some(rest) = first.strip_prefix(CHUNKED_SIGNATURE) else {
            return Ok(Cow::Borrowed(first));
        };
        let count = parse_chunk_count(rest).ok_or(Error::Data {
            offset: start + 2,
            message: "the chunk count after the signature is not an integer",
        })?;
        let mut joined = Vec::new();
        for _ in 0..count {
            joined.extend_from_slice(self.read_utf()?);
        }
        Ok(Cow::Owned(joined))
    }
}

/// Java's `data.replace(SIGNATURE, "")` followed by `Integer.parseInt`:
/// every further copy of the signature is removed, then the rest must be
/// an optional sign and decimal digits that fit an `i32`. A negative
/// count reads no chunks, as the Java loop does. Only ASCII digits are
/// accepted; `Integer.parseInt` would also accept other Unicode decimal
/// digits, which no OpenNLP writer produces.
fn parse_chunk_count(rest: &[u8]) -> Option<u32> {
    let mut digits: Vec<u8> = Vec::with_capacity(rest.len());
    let mut i = 0;
    while i < rest.len() {
        if rest[i..].starts_with(CHUNKED_SIGNATURE) {
            i += CHUNKED_SIGNATURE.len();
        } else {
            digits.push(rest[i]);
            i += 1;
        }
    }
    let (negative, body) = match digits.split_first() {
        Some((b'-', body)) => (true, body),
        Some((b'+', body)) => (false, body),
        _ => (false, &digits[..]),
    };
    if body.is_empty() || !body.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut value: i64 = 0;
    for &d in body {
        value = value * 10 + i64::from(d - b'0');
        if value > i64::from(i32::MAX) + 1 {
            return None;
        }
    }
    if negative {
        return Some(0);
    }
    u32::try_from(i32::try_from(value).ok()?).ok()
}

/// Steps through modified UTF-8, one UTF-16 code unit per sequence.
/// Yields `Err(offset)` at the first malformed sequence.
fn units(bytes: &[u8]) -> impl Iterator<Item = std::result::Result<u16, usize>> + '_ {
    let mut i = 0;
    std::iter::from_fn(move || {
        let &c = bytes.get(i)?;
        let at = i;
        let unit = match c >> 4 {
            0..=7 => {
                i += 1;
                Ok(u16::from(c))
            }
            12 | 13 => match bytes.get(i + 1) {
                Some(&c2) if c2 & 0xC0 == 0x80 => {
                    i += 2;
                    Ok(u16::from(c & 0x1F) << 6 | u16::from(c2 & 0x3F))
                }
                _ => Err(at),
            },
            14 => match (bytes.get(i + 1), bytes.get(i + 2)) {
                (Some(&c2), Some(&c3)) if c2 & 0xC0 == 0x80 && c3 & 0xC0 == 0x80 => {
                    i += 3;
                    Ok(u16::from(c & 0x0F) << 12 | u16::from(c2 & 0x3F) << 6 | u16::from(c3 & 0x3F))
                }
                _ => Err(at),
            },
            _ => Err(at),
        };
        if unit.is_err() {
            i = bytes.len();
        }
        Some(unit)
    })
}

/// Checks that `bytes` is modified UTF-8 as `DataInputStream.readUTF`
/// accepts it; on failure, the offset of the first bad sequence.
pub fn check_modified_utf8(bytes: &[u8]) -> std::result::Result<(), usize> {
    for unit in units(bytes) {
        unit?;
    }
    Ok(())
}

/// Decodes well-formed modified UTF-8 (see [`check_modified_utf8`]) to a
/// `String`. Surrogate pairs become one character; a lone surrogate or a
/// malformed sequence becomes U+FFFD.
pub fn decode_modified_utf8(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let utf16 = units(bytes).map(|u| u.unwrap_or(0xFFFD));
    for c in char::decode_utf16(utf16) {
        out.push(c.unwrap_or(char::REPLACEMENT_CHARACTER));
    }
    out
}

/// The canonical encoding of well-formed modified UTF-8: the bytes
/// `DataOutputStream.writeUTF` would write for the same string. Borrowed
/// when the input is already canonical, which is always the case for data
/// Java wrote; a raw zero byte or an overlong sequence is re-encoded, so
/// two encodings of one Java string become the same bytes.
pub fn canonical_modified_utf8(bytes: &[u8]) -> Cow<'_, [u8]> {
    let mut i = 0;
    let mut canonical = true;
    for unit in units(bytes) {
        let Ok(unit) = unit else { break };
        let len = match bytes[i] >> 4 {
            0..=7 => 1,
            12 | 13 => 2,
            _ => 3,
        };
        if encoded_len(unit) != len {
            canonical = false;
            break;
        }
        i += len;
    }
    if canonical {
        return Cow::Borrowed(bytes);
    }
    let mut out = Vec::with_capacity(bytes.len());
    for unit in units(bytes) {
        let Ok(unit) = unit else { break };
        push_unit(&mut out, unit);
    }
    Cow::Owned(out)
}

fn encoded_len(unit: u16) -> usize {
    match unit {
        0x0001..=0x007F => 1,
        0x0000 | 0x0080..=0x07FF => 2,
        _ => 3,
    }
}

fn push_unit(out: &mut Vec<u8>, unit: u16) {
    match encoded_len(unit) {
        1 => out.push(unit as u8),
        2 => out.extend_from_slice(&[0xC0 | (unit >> 6) as u8, 0x80 | (unit & 0x3F) as u8]),
        _ => out.extend_from_slice(&[
            0xE0 | (unit >> 12) as u8,
            0x80 | ((unit >> 6) & 0x3F) as u8,
            0x80 | (unit & 0x3F) as u8,
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf(bytes: &[u8]) -> Vec<u8> {
        let mut v = (bytes.len() as u16).to_be_bytes().to_vec();
        v.extend_from_slice(bytes);
        v
    }

    #[test]
    fn ints_and_doubles_are_big_endian() {
        let mut d = vec![0xFF, 0xFF, 0xFF, 0xFE, 0x00, 0x00, 0x00, 0x01];
        d.extend_from_slice(&(-0.25f64).to_bits().to_be_bytes());
        let mut r = JavaData::new(&d);
        assert_eq!(r.read_int().unwrap(), -2);
        assert_eq!(r.read_int().unwrap(), 1);
        assert_eq!(r.read_double().unwrap(), -0.25);
        assert_eq!(r.remaining(), 0);
        assert!(matches!(r.read_int(), Err(Error::Data { offset: 16, .. })));
    }

    #[test]
    fn modified_utf8_decodes_nul_pairs_and_lone_surrogates() {
        // "a", NUL as C0 80, "é", U+1F600 as two surrogate sequences, then a
        // lone high surrogate U+D800.
        let bytes = [b'a', 0xC0, 0x80, 0xC3, 0xA9, 0xED, 0xA0, 0xBD, 0xED, 0xB8, 0x80, 0xED, 0xA0, 0x80];
        let d = utf(&bytes);
        let mut r = JavaData::new(&d);
        assert_eq!(r.read_utf_string().unwrap(), "a\0é\u{1F600}\u{FFFD}");
        let mut r = JavaData::new(&d);
        assert_eq!(r.read_utf().unwrap(), &bytes);
    }

    #[test]
    fn malformed_modified_utf8_names_its_offset() {
        let d = utf(&[b'a', 0xC3, 0x29]);
        assert!(matches!(JavaData::new(&d).read_utf(), Err(Error::Data { offset: 3, .. })));
        let d = utf(&[0xF0, 0x9F, 0x98, 0x80]);
        assert!(matches!(JavaData::new(&d).read_utf(), Err(Error::Data { offset: 2, .. })));
        let d = utf(&[b'a', 0xE2, 0x82]);
        assert!(matches!(JavaData::new(&d).read_utf(), Err(Error::Data { offset: 3, .. })));
        let mut short = utf(b"abc");
        short.pop();
        assert!(matches!(JavaData::new(&short).read_utf(), Err(Error::Data { offset: 2, .. })));
    }

    #[test]
    fn chunked_strings_are_joined() {
        let mut d = utf(b"CHUNKED-MODEL-PARAMS:3");
        d.extend(utf(b"ab"));
        d.extend(utf(&[0xC3, 0xA9]));
        d.extend(utf(b"z"));
        d.extend(utf(b"after"));
        let mut r = JavaData::new(&d);
        assert_eq!(&*r.read_chunked_utf().unwrap(), "abéz".as_bytes());
        assert_eq!(&*r.read_chunked_utf().unwrap(), b"after");
        assert!(matches!(JavaData::new(&utf(b"plain")).read_chunked_utf().unwrap(), Cow::Borrowed(b"plain")));
    }

    #[test]
    fn chunk_counts_follow_integer_parse_int() {
        assert_eq!(parse_chunk_count(b"12"), Some(12));
        assert_eq!(parse_chunk_count(b"+2"), Some(2));
        assert_eq!(parse_chunk_count(b"-5"), Some(0));
        assert_eq!(parse_chunk_count(b"2147483647"), Some(i32::MAX as u32));
        assert_eq!(parse_chunk_count(b"-2147483648"), Some(0));
        assert_eq!(parse_chunk_count(b"2147483648"), None);
        assert_eq!(parse_chunk_count(b""), None);
        assert_eq!(parse_chunk_count(b"-"), None);
        assert_eq!(parse_chunk_count(b"1 "), None);
        assert_eq!(parse_chunk_count(b"1CHUNKED-MODEL-PARAMS:2"), Some(12));
        let d = utf(b"CHUNKED-MODEL-PARAMS:x");
        assert!(matches!(JavaData::new(&d).read_chunked_utf(), Err(Error::Data { offset: 2, .. })));
    }

    #[test]
    fn canonical_forms() {
        assert!(matches!(canonical_modified_utf8(b"abc"), Cow::Borrowed(_)));
        assert!(matches!(canonical_modified_utf8(&[0xC0, 0x80, 0xED, 0xA0, 0x80]), Cow::Borrowed(_)));
        // A raw zero byte and an overlong "A" (C1 81) are re-encoded.
        assert_eq!(&*canonical_modified_utf8(&[0x00, 0xC1, 0x81]), &[0xC0, 0x80, b'A']);
        // An overlong three-byte "é".
        assert_eq!(&*canonical_modified_utf8(&[0xE0, 0x83, 0xA9]), &[0xC3, 0xA9]);
    }
}
