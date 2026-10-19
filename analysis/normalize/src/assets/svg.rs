//! Whether a payload is an SVG image: its first element is `svg` in the
//! SVG namespace. OpenNLP's `SvgHeader` asks the JDK's StAX parser for the
//! first start element, with DTDs and external entities disabled. This is
//! a hand-written scan of the same well-formedness rules up to the end of
//! that start tag; it resolves nothing and builds no tree.
//!
//! What is read: an optional byte-order mark and the encoding it or the
//! first four bytes imply (XML 1.0 appendix F); the XML declaration; then
//! white space, comments, processing instructions and one document type
//! declaration, in any order the XML grammar allows; then the root start
//! tag, its attributes and the namespace declarations on it. Any
//! well-formedness error before the end of that tag, or the end of the
//! 64 KiB decoded prefix, means the payload is not SVG.

mod scan;

use self::scan::Xml;
use super::payload::{Decoder, Payload};

/// The decoded prefix the check reads (`SvgHeader.MAX_HEADER_BYTES`).
pub(crate) const MAX_HEADER_BYTES: usize = 64 * 1024;

const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";
const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";
const XMLNS_NAMESPACE: &str = "http://www.w3.org/2000/xmlns/";

/// Whether a decoded header can start an XML document: `<`, XML white
/// space, `<` in UTF-16BE without a byte-order mark, or a UTF-8 or UTF-16
/// byte-order mark (`SvgHeader.isCandidate`).
pub(crate) fn is_candidate(header: &[u8]) -> bool {
    matches!(
        header,
        [b'<' | b' ' | b'\t' | b'\r' | b'\n', ..]
            | [0, b'<', ..]
            | [0xEF, 0xBB, 0xBF, ..]
            | [0xFE, 0xFF, ..]
            | [0xFF, 0xFE, ..]
    )
}

/// Whether the payload's first element is the SVG root.
pub(crate) fn is_svg(text: &[u8], payload: &Payload) -> bool {
    Xml::new(payload.decoder(text, MAX_HEADER_BYTES)).svg_root() == Some(true)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Encoding {
    Utf8,
    Utf16Be,
    Utf16Le,
    Latin1,
    Ascii,
}

/// The characters of the decoded bytes: decoded in the document's
/// encoding, with line ends normalized to LF (XML 1.0 section 2.11).
/// A malformed sequence or a code point outside the XML `Char` production
/// ends the stream, which makes every unfinished construct fail.
struct Chars<'t> {
    bytes: Decoder<'t>,
    pending: [u8; 4],
    pending_len: usize,
    pending_at: usize,
    encoding: Encoding,
    held: Option<char>,
    ended: bool,
}

impl<'t> Chars<'t> {
    /// Detects the encoding from the first four bytes and skips a
    /// byte-order mark.
    fn new(mut bytes: Decoder<'t>) -> Chars<'t> {
        let mut pending = [0u8; 4];
        let mut pending_len = 0;
        for b in bytes.by_ref().take(4) {
            pending[pending_len] = b;
            pending_len += 1;
        }
        let (encoding, skip) = match &pending[..pending_len] {
            [0xEF, 0xBB, 0xBF, ..] => (Encoding::Utf8, 3),
            [0xFE, 0xFF, ..] => (Encoding::Utf16Be, 2),
            [0xFF, 0xFE, ..] => (Encoding::Utf16Le, 2),
            [0x00, b'<', 0x00, b'?'] => (Encoding::Utf16Be, 0),
            [b'<', 0x00, b'?', 0x00] => (Encoding::Utf16Le, 0),
            _ => (Encoding::Utf8, 0),
        };
        Chars { bytes, pending, pending_len, pending_at: skip, encoding, held: None, ended: false }
    }

    fn byte(&mut self) -> Option<u8> {
        if self.pending_at < self.pending_len {
            self.pending_at += 1;
            return Some(self.pending[self.pending_at - 1]);
        }
        self.bytes.next()
    }

    fn decode(&mut self) -> Option<char> {
        let first = self.byte()?;
        match self.encoding {
            Encoding::Ascii => first.is_ascii().then_some(first as char),
            Encoding::Latin1 => Some(first as char),
            Encoding::Utf8 => self.utf8(first),
            Encoding::Utf16Be | Encoding::Utf16Le => {
                let unit = self.unit(first)?;
                match unit {
                    0xD800..=0xDBFF => {
                        let next = self.byte()?;
                        let low = self.unit(next)?;
                        if !(0xDC00..=0xDFFF).contains(&low) {
                            return None;
                        }
                        char::from_u32(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00))
                    }
                    _ => char::from_u32(unit),
                }
            }
        }
    }

    /// A UTF-16 code unit whose first byte is `first`.
    fn unit(&mut self, first: u8) -> Option<u32> {
        let second = self.byte()?;
        let (high, low) = if self.encoding == Encoding::Utf16Be { (first, second) } else { (second, first) };
        Some(((high as u32) << 8) | low as u32)
    }

    /// Strict UTF-8: no overlong forms, surrogates or values past U+10FFFF.
    fn utf8(&mut self, first: u8) -> Option<char> {
        let (len, init, min) = match first {
            0x00..=0x7F => return Some(first as char),
            0xC2..=0xDF => (2, (first & 0x1F) as u32, 0x80),
            0xE0..=0xEF => (3, (first & 0x0F) as u32, 0x800),
            0xF0..=0xF4 => (4, (first & 0x07) as u32, 0x10000),
            _ => return None,
        };
        let mut value = init;
        for _ in 1..len {
            let b = self.byte()?;
            if b & 0xC0 != 0x80 {
                return None;
            }
            value = (value << 6) | (b & 0x3F) as u32;
        }
        if value < min {
            return None;
        }
        char::from_u32(value)
    }

    fn next(&mut self) -> Option<char> {
        if self.ended {
            return None;
        }
        let c = match self.held.take() {
            Some(c) => Some(c),
            None => self.decode().filter(|&c| is_xml_char(c)),
        };
        let Some(c) = c else {
            self.ended = true;
            return None;
        };
        if c != '\r' {
            return Some(c);
        }
        match self.decode().filter(|&c| is_xml_char(c)) {
            Some('\n') => {}
            Some(other) => self.held = Some(other),
            None => self.ended = true,
        }
        Some('\n')
    }
}

/// The XML `Char` production.
fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..)
}

/// XML white space: space, tab, LF (CR has been normalized away).
fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// A `PubidChar` of XML 1.0.
fn is_pubid_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(
            c,
            ' ' | '\r'
                | '\n'
                | '-'
                | '\''
                | '('
                | ')'
                | '+'
                | ','
                | '.'
                | '/'
                | ':'
                | '='
                | '?'
                | ';'
                | '!'
                | '*'
                | '#'
                | '@'
                | '$'
                | '_'
                | '%'
        )
}

/// `NameStartChar` of XML 1.0, fifth edition.
fn is_name_start(c: char) -> bool {
    matches!(c,
        ':' | 'A'..='Z' | '_' | 'a'..='z' | '\u{C0}'..='\u{D6}' | '\u{D8}'..='\u{F6}' | '\u{F8}'..='\u{2FF}'
        | '\u{370}'..='\u{37D}' | '\u{37F}'..='\u{1FFF}' | '\u{200C}'..='\u{200D}' | '\u{2070}'..='\u{218F}'
        | '\u{2C00}'..='\u{2FEF}' | '\u{3001}'..='\u{D7FF}' | '\u{F900}'..='\u{FDCF}' | '\u{FDF0}'..='\u{FFFD}'
        | '\u{10000}'..='\u{EFFFF}')
}

/// `NameChar` of XML 1.0, fifth edition.
fn is_name_char(c: char) -> bool {
    is_name_start(c) || matches!(c, '-' | '.' | '0'..='9' | '\u{B7}' | '\u{300}'..='\u{36F}' | '\u{203F}'..='\u{2040}')
}

/// The encoding after a declaration names `name`, given the one the first
/// bytes implied. A declaration may not move between 8-bit and 16-bit
/// encodings or flip UTF-16 byte order. Beyond UTF-8, UTF-16, US-ASCII and
/// ISO-8859-1 the payload is taken not to be SVG.
fn declared_encoding(detected: Encoding, name: &str) -> Option<Encoding> {
    let name = name.to_ascii_uppercase();
    let sixteen = matches!(detected, Encoding::Utf16Be | Encoding::Utf16Le);
    match name.as_str() {
        "UTF-16" if sixteen => Some(detected),
        "UTF-16BE" if detected == Encoding::Utf16Be => Some(Encoding::Utf16Be),
        "UTF-16LE" if detected == Encoding::Utf16Le => Some(Encoding::Utf16Le),
        "UTF-8" if !sixteen => Some(Encoding::Utf8),
        "US-ASCII" | "ASCII" | "ANSI_X3.4-1968" | "ANSI_X3.4-1986" | "ISO-IR-6" | "ISO646-US" | "US" | "IBM367"
        | "CP367" | "CSASCII"
            if !sixteen =>
        {
            Some(Encoding::Ascii)
        }
        "ISO-8859-1" | "ISO_8859-1" | "LATIN1" | "ISO-IR-100" | "L1" | "IBM819" | "CP819" | "CSISOLATIN1"
            if !sixteen =>
        {
            Some(Encoding::Latin1)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_encoding_rejects_mismatched_utf16_endianness() {
        assert_eq!(declared_encoding(Encoding::Utf16Be, "UTF-16"), Some(Encoding::Utf16Be));
        assert_eq!(declared_encoding(Encoding::Utf16Le, "UTF-16"), Some(Encoding::Utf16Le));
        assert_eq!(declared_encoding(Encoding::Utf16Be, "UTF-16BE"), Some(Encoding::Utf16Be));
        assert_eq!(declared_encoding(Encoding::Utf16Le, "UTF-16LE"), Some(Encoding::Utf16Le));
        assert_eq!(declared_encoding(Encoding::Utf16Be, "UTF-16LE"), None);
        assert_eq!(declared_encoding(Encoding::Utf16Le, "UTF-16BE"), None);
    }
}
