//! Encoded payloads: the base64 and hex alphabets, the scan of one base64
//! run, the decoded length, and a streaming decoder that reads a decoded
//! prefix straight from the text without copying the payload.

use turbo_normalize_layers::AssetEncoding;

/// `[A-Za-z0-9+/]`: the standard base64 alphabet without padding.
pub(crate) fn is_base64(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'+' || b == b'/'
}

/// `[A-Za-z0-9_-]`: the URL-safe base64 alphabet without padding.
pub(crate) fn is_base64_url(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_'
}

/// A character of either base64 alphabet.
pub(crate) fn is_any_base64(b: u8) -> bool {
    is_base64(b) || b == b'-' || b == b'_'
}

/// The six-bit value of a digit of either base64 alphabet; `None` for
/// padding, line breaks and every other byte.
pub(crate) fn digit(b: u8) -> Option<u8> {
    match b {
        b'A'..=b'Z' => Some(b - b'A'),
        b'a'..=b'z' => Some(b - b'a' + 26),
        b'0'..=b'9' => Some(b - b'0' + 52),
        b'+' | b'-' => Some(62),
        b'/' | b'_' => Some(63),
        _ => None,
    }
}

/// The value of an ASCII hexadecimal digit.
pub(crate) fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// The end of one CRLF or LF line break at `at`, or `at` when none starts
/// there.
pub(crate) fn line_break_end(text: &[u8], at: usize) -> usize {
    match text.get(at) {
        Some(b'\n') => at + 1,
        Some(b'\r') if text.get(at + 1) == Some(&b'\n') => at + 2,
        _ => at,
    }
}

/// One scanned payload and the counts its decoded length follows from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Payload {
    pub start: usize,
    pub end: usize,
    pub encoding: AssetEncoding,
    /// Base64: the characters, padding included, line breaks excluded.
    /// Hex: the digits.
    pub encoded_len: usize,
    /// The trailing `=` characters of a base64 payload.
    pub padding: usize,
    /// Whether the alphabet, wrapping and padding decode together.
    pub valid: bool,
}

impl Payload {
    /// The number of decoded bytes, as OpenNLP's `decodedLength` computes
    /// it for base64: three bytes per group of four, less the padding,
    /// plus one or two for an unpadded tail of two or three characters.
    pub fn decoded_len(&self) -> u64 {
        if self.encoding == AssetEncoding::Hex {
            return (self.encoded_len / 2) as u64;
        }
        let n = self.encoded_len as u64;
        let tail = match n % 4 {
            2 => 1,
            3 => 2,
            _ => 0,
        };
        n / 4 * 3 - self.padding as u64 + tail
    }

    /// A decoder over the payload that stops after `max` bytes.
    pub fn decoder<'t>(&self, text: &'t [u8], max: usize) -> Decoder<'t> {
        Decoder {
            text,
            at: self.start,
            end: self.end,
            remaining: max,
            bits: 0,
            held: 0,
            hex: self.encoding == AssetEncoding::Hex,
        }
    }
}

/// Where a base64 run may continue across a line break.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Wrapping {
    /// MIME (76 columns) and PEM (64 columns) wrapping of standard base64:
    /// every line but the last has the same one of those two widths.
    pub mime: bool,
    /// Quoted-printable soft line breaks: `=` then CRLF or LF, at any
    /// column, in standard base64 only (MIME never carries the URL-safe
    /// alphabet), with a standard base64 character after the break.
    pub soft: bool,
}

/// Scans the base64 or base64url run that starts at `start`, OpenNLP's
/// `CursorAssetDetector.payload`.
///
/// The run takes characters of either alphabet, then at most two `=`. With
/// `wrapping.mime` a standard run continues after a CRLF or LF when the
/// line before it is 64 or 76 characters long, the same width as every
/// earlier line, and the next line starts with a standard base64
/// character. With `wrapping.soft` it also continues after a
/// quoted-printable soft line break. A run of length 1 modulo 4 without
/// padding loses its last character, which no base64 text can end with.
///
/// The result is invalid when it is shorter than two characters, mixes
/// the alphabets, wraps URL-safe text, has padding at a length that is
/// not a multiple of four, or is followed by a third `=`.
pub(crate) fn scan_base64(text: &[u8], start: usize, wrapping: Wrapping) -> Payload {
    let len = text.len();
    let mut i = start;
    let (mut encoded, mut line, mut width, mut padding) = (0usize, 0usize, 0usize, 0usize);
    let (mut url_safe, mut standard_only, mut soft_wrapped) = (false, false, false);
    while i < len {
        let c = text[i];
        if is_any_base64(c) && padding == 0 {
            url_safe |= c == b'-' || c == b'_';
            standard_only |= c == b'+' || c == b'/';
            encoded += 1;
            line += 1;
            i += 1;
            continue;
        }
        if c == b'=' && padding == 0 && wrapping.soft {
            let next = line_break_end(text, i + 1);
            if next > i + 1 && next < len && !url_safe && is_base64(text[next]) {
                soft_wrapped = true;
                line = 0;
                i = next;
                continue;
            }
        }
        if c == b'=' && padding < 2 {
            padding += 1;
            encoded += 1;
            line += 1;
            i += 1;
            continue;
        }
        let next = if wrapping.mime && !url_safe && padding == 0 { line_break_end(text, i) } else { i };
        if next > i
            && (line == 64 || line == 76)
            && (width == 0 || width == line)
            && next < len
            && is_base64(text[next])
        {
            width = line;
            line = 0;
            i = next;
            continue;
        }
        break;
    }
    if encoded % 4 == 1 && padding == 0 {
        i -= 1;
        encoded -= 1;
    }
    let valid = encoded >= 2
        && !(url_safe && (standard_only || width > 0 || soft_wrapped))
        && (padding == 0 || encoded.is_multiple_of(4))
        && text.get(i) != Some(&b'=');
    let encoding = if url_safe { AssetEncoding::Base64Url } else { AssetEncoding::Base64 };
    Payload { start, end: i, encoding, encoded_len: encoded, padding, valid }
}

/// Reads the decoded bytes of a payload one at a time, straight from the
/// text, OpenNLP's `Base64PayloadInputStream`. Base64 digits of either
/// alphabet are read and everything else (line breaks, padding, the `=`
/// of a soft line break) is skipped; hex payloads read digit pairs and
/// skip separators and `0x` prefixes. The payload has been validated, so
/// the decoder does not check it again.
pub(crate) struct Decoder<'t> {
    text: &'t [u8],
    at: usize,
    end: usize,
    remaining: usize,
    bits: u32,
    held: u32,
    hex: bool,
}

impl Iterator for Decoder<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.remaining == 0 {
            return None;
        }
        let width = if self.hex { 4 } else { 6 };
        while self.at < self.end {
            let c = self.text[self.at];
            self.at += 1;
            let value = if self.hex {
                if c == b'0' && matches!(self.text.get(self.at), Some(b'x' | b'X')) && self.at < self.end {
                    self.at += 1;
                    continue;
                }
                hex_digit(c)
            } else {
                digit(c)
            };
            let Some(value) = value else { continue };
            self.bits = (self.bits << width) | value as u32;
            self.held += width;
            if self.held >= 8 {
                self.held -= 8;
                self.remaining -= 1;
                return Some((self.bits >> self.held) as u8);
            }
        }
        None
    }
}

/// The longest decoded header any check reads: 45 bytes, the 60 base64
/// characters OpenNLP decodes for an EMF candidate.
pub(crate) const MAX_HEADER: usize = 45;

/// The decoded bytes a check of the format reads: `HEADER_ENCODED_LENGTH`
/// (32 base64 characters, 24 bytes) for most payloads,
/// `EMF_HEADER_ENCODED_LENGTH` (60 characters, 45 bytes) for one that
/// starts with the little-endian EMR_HEADER record type,
/// `JPEG2000_HEADER_ENCODED_LENGTH` and `PEM_HEADER_ENCODED_LENGTH` (48
/// characters, 36 bytes) for one that starts with the JPEG 2000 signature
/// box or with `-----BEGIN `.
///
/// OpenNLP chooses the length by matching base64 prefixes of the text
/// (`AQAAA`, `AAAADGpQICANCocK`, `LS0tLS1CRUdJTi`); each prefix fixes the
/// bits tested here, so the choice is the same.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Header {
    bytes: [u8; MAX_HEADER],
    len: usize,
}

impl Header {
    /// Decodes the header of a valid payload.
    pub fn read(text: &[u8], payload: &Payload) -> Header {
        let mut bytes = [0u8; MAX_HEADER];
        let mut len = 0;
        for b in payload.decoder(text, MAX_HEADER) {
            bytes[len] = b;
            len += 1;
        }
        let limit = if bytes[0] == 1 && bytes[1] == 0 && bytes[2] == 0 && bytes[3] < 4 {
            45
        } else if bytes[..12] == *b"\x00\x00\x00\x0cjP  \r\n\x87\n"
            || (bytes[..10] == *b"-----BEGIN" && bytes[10] >> 4 == 2)
        {
            36
        } else {
            24
        };
        Header { bytes, len: len.min(limit) }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

/// Why a payload could not be decoded.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The text ends before the payload span does.
    TextTooShort,
    /// The payload holds a character outside its alphabet.
    NotInAlphabet,
    /// The payload uses `+` or `/` and also `-` or `_`.
    MixedAlphabets,
    /// URL-safe base64 broken across lines.
    WrappedUrlSafe,
    /// Padding in the middle, more than two `=`, or a length padding does
    /// not complete.
    Padding,
    /// A base64 length of 1 modulo 4, or an odd number of hex digits.
    Length,
}

impl std::fmt::Display for DecodeError {
    /// OpenNLP's messages where it has one.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DecodeError::TextTooShort => "text is shorter than the payload span",
            DecodeError::NotInAlphabet => "payload contains a non-base64 character",
            DecodeError::MixedAlphabets => "payload mixes base64 and base64url alphabets",
            DecodeError::WrappedUrlSafe => "wrapped base64url payloads are not supported",
            DecodeError::Padding => "payload has misplaced or incomplete padding",
            DecodeError::Length => "payload length cannot be decoded",
        })
    }
}

impl std::error::Error for DecodeError {}

/// Decodes the payload `text[start..end]`, as OpenNLP's
/// `EmbeddedAsset.decode` does for base64: either alphabet but not both,
/// CR and LF skipped except in URL-safe text, padding only at the end.
/// A quoted-printable soft line break (`=` before CRLF or LF) is skipped
/// as well; OpenNLP does not accept one, but its detector never reports a
/// payload that holds one.
pub(crate) fn decode_base64(text: &str, start: usize, end: usize) -> Result<Vec<u8>, DecodeError> {
    let bytes = text.as_bytes().get(start..end).ok_or(DecodeError::TextTooShort)?;
    let (mut standard, mut url, mut wrapped) = (false, false, false);
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3 + 2);
    let (mut bits, mut held, mut digits, mut padding) = (0u32, 0u32, 0usize, 0usize);
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        i += 1;
        match c {
            b'\r' | b'\n' => wrapped = true,
            b'=' if line_break_end(bytes, i) > i => {
                wrapped = true;
                i = line_break_end(bytes, i);
            }
            b'=' => padding += 1,
            _ => {
                let Some(value) = digit(c) else { return Err(DecodeError::NotInAlphabet) };
                if padding > 0 {
                    return Err(DecodeError::Padding);
                }
                standard |= c == b'+' || c == b'/';
                url |= c == b'-' || c == b'_';
                digits += 1;
                bits = (bits << 6) | value as u32;
                held += 6;
                if held >= 8 {
                    held -= 8;
                    out.push((bits >> held) as u8);
                }
            }
        }
    }
    if standard && url {
        return Err(DecodeError::MixedAlphabets);
    }
    if wrapped && url {
        return Err(DecodeError::WrappedUrlSafe);
    }
    if padding > 2 || (padding > 0 && !(digits + padding).is_multiple_of(4)) {
        return Err(DecodeError::Padding);
    }
    if digits % 4 == 1 {
        return Err(DecodeError::Length);
    }
    Ok(out)
}

/// Decodes the hex payload `text[start..end]`: digit pairs, with the
/// separators and `0x` prefixes the hex scan accepts skipped.
pub(crate) fn decode_hex(text: &str, start: usize, end: usize) -> Result<Vec<u8>, DecodeError> {
    let bytes = text.as_bytes().get(start..end).ok_or(DecodeError::TextTooShort)?;
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let (mut high, mut i) = (None, 0);
    while i < bytes.len() {
        let c = bytes[i];
        i += 1;
        if c == b'0' && high.is_none() && matches!(bytes.get(i), Some(b'x' | b'X')) {
            i += 1;
            continue;
        }
        match (hex_digit(c), high) {
            (Some(low), Some(h)) => {
                out.push((h << 4) | low);
                high = None;
            }
            (Some(h), None) => high = Some(h),
            (None, _) if matches!(c, b' ' | b':' | b',' | b'\t' | b'\r' | b'\n') && high.is_none() => {}
            (None, _) => return Err(DecodeError::NotInAlphabet),
        }
    }
    if high.is_some() { Err(DecodeError::Length) } else { Ok(out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STANDARD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /// Base64 written out here, apart from the decoder: standard padded,
    /// URL-safe unpadded, or standard wrapped at `width` with `separator`.
    fn encode(bytes: &[u8], transport: &str) -> String {
        let mut flat = String::new();
        for group in bytes.chunks(3) {
            let n = group.iter().enumerate().fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
            for k in 0..=group.len() {
                flat.push(char::from(STANDARD[(n >> (18 - 6 * k) & 63) as usize]));
            }
            for _ in group.len()..3 {
                flat.push('=');
            }
        }
        let wrap = |width: usize, separator: &str| {
            flat.as_bytes().chunks(width).map(|l| std::str::from_utf8(l).unwrap()).collect::<Vec<_>>().join(separator)
        };
        match transport {
            "standard" => flat,
            "url" => flat.trim_end_matches('=').replace('+', "-").replace('/', "_"),
            "mime64" => wrap(64, "\n"),
            "mime76" => wrap(76, "\r\n"),
            _ => panic!("transport is unsupported"),
        }
    }

    /// A payload over `text[start..end]`; the decoder reads it as base64.
    fn base64_payload(start: usize, end: usize) -> Payload {
        Payload { start, end, encoding: AssetEncoding::Base64, encoded_len: end - start, padding: 0, valid: true }
    }

    // Base64PayloadInputStreamTest: the decoded prefix up to a limit, with
    // other base64 text after the payload that the decoder must not read.
    #[test]
    fn decoded_prefix() {
        for length in [0usize, 1, 2, 3, 4, 5, 23, 24, 25, 63, 64, 65, 256, 1024] {
            let expected: Vec<u8> = (0..length).map(|i| (i * 73 + 19) as u8).collect();
            for transport in ["standard", "url", "mime64", "mime76"] {
                let encoded = encode(&expected, transport);
                let before = "Other payload: ";
                let text = format!("{before}{encoded} /AAAA");
                let payload = base64_payload(before.len(), before.len() + encoded.len());
                for limit in [0, 1, length.saturating_sub(1), length, usize::MAX] {
                    let mut decoder = payload.decoder(text.as_bytes(), limit);
                    let read: Vec<u8> = decoder.by_ref().collect();
                    assert_eq!(read, expected[..length.min(limit)], "{length} {transport} {limit}");
                    assert_eq!(decoder.next(), None);
                    assert_eq!(decoder.next(), None);
                }
            }
        }
    }

    #[test]
    fn soft_line_breaks_are_skipped() {
        let text = "iVBORw0K=\r\nGgo=\nAAAA";
        let decoded: Vec<u8> = base64_payload(0, text.len()).decoder(text.as_bytes(), usize::MAX).collect();
        assert_eq!(decoded, b"\x89PNG\r\n\x1a\n\0\0\0");
    }

    #[test]
    fn hex_pairs_skip_separators_and_prefixes() {
        for text in ["89504e47", "89 50 4e 47", "89:50:4E:47", "0x89, 0x50, 0x4e, 0X47", "89 50\n  4e 47"] {
            let payload = Payload { encoding: AssetEncoding::Hex, ..base64_payload(0, text.len()) };
            let decoded: Vec<u8> = payload.decoder(text.as_bytes(), usize::MAX).collect();
            assert_eq!(decoded, b"\x89PNG", "{text}");
            assert_eq!(decode_hex(text, 0, text.len()).unwrap(), b"\x89PNG", "{text}");
        }
        assert_eq!(decode_hex("895", 0, 3), Err(DecodeError::Length));
        assert_eq!(decode_hex("8g", 0, 2), Err(DecodeError::NotInAlphabet));
        assert_eq!(decode_hex("89", 0, 3), Err(DecodeError::TextTooShort));
    }

    #[test]
    fn base64_decode_errors() {
        let cases = [
            ("QUJD", Ok(&b"ABC"[..])),
            ("QUI=", Ok(b"AB")),
            ("QQ==", Ok(b"A")),
            ("QQ", Ok(b"A")),
            ("QUJ\r\nD", Ok(b"ABC")),
            ("+_AA", Err(DecodeError::MixedAlphabets)),
            ("QU-_\nAA", Err(DecodeError::WrappedUrlSafe)),
            ("QUJD=", Err(DecodeError::Padding)),
            ("QQ=A", Err(DecodeError::Padding)),
            ("QQ===", Err(DecodeError::Padding)),
            ("QUJDQ", Err(DecodeError::Length)),
            ("QU*D", Err(DecodeError::NotInAlphabet)),
        ];
        for (text, expected) in cases {
            assert_eq!(decode_base64(text, 0, text.len()).as_deref().map_err(|e| *e), expected, "{text}");
        }
    }
}
