//! Decoding a corpus file into the text the readers read.
//!
//! OpenNLP's format factories take the file's encoding as a parameter
//! (`-encoding`, `PlainTextByLineStream(factory, charset)`). The readers
//! here read decoded text, and a sample's spans are byte offsets into
//! that text. Two encodings are decoded here: UTF-8 and ISO-8859-1, the
//! encoding of several older corpora (CoNLL-2002 Spanish and Dutch,
//! Floresta Sinta(c)tica).
//!
//! Java's decoder replaces a malformed UTF-8 sequence with U+FFFD and
//! reads on; [`decode`] refuses the input instead and says where, since a
//! replaced character in training data is a corrupted sample. A byte
//! order mark is kept, as Java keeps it: it is not whitespace, so it ends
//! up at the start of the first token.

use std::borrow::Cow;
use std::fmt;

/// The encoding of a corpus file.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Encoding {
    #[default]
    Utf8,
    /// ISO-8859-1: each byte is the code point of the same value.
    Latin1,
}

/// Bytes that are not valid UTF-8.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DecodeError {
    /// The offset of the first byte of the malformed sequence.
    pub offset: usize,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "malformed UTF-8 at byte {}", self.offset)
    }
}

impl std::error::Error for DecodeError {}

/// The text `bytes` hold in `encoding`; borrowed when the bytes are
/// already that text.
pub fn decode(bytes: &[u8], encoding: Encoding) -> Result<Cow<'_, str>, DecodeError> {
    match encoding {
        Encoding::Utf8 => {
            std::str::from_utf8(bytes).map(Cow::Borrowed).map_err(|e| DecodeError { offset: e.valid_up_to() })
        }
        Encoding::Latin1 => {
            if bytes.is_ascii() {
                // ASCII is valid UTF-8, and the same text in both.
                Ok(Cow::Borrowed(std::str::from_utf8(bytes).expect("ASCII is UTF-8")))
            } else {
                Ok(Cow::Owned(bytes.iter().map(|&b| char::from(b)).collect()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_both_encodings() {
        assert_eq!(decode(b"a\xc3\xa9", Encoding::Utf8).unwrap(), "a\u{e9}");
        assert_eq!(decode(b"a\xe9", Encoding::Latin1).unwrap(), "a\u{e9}");
        assert!(matches!(decode(b"ab", Encoding::Latin1).unwrap(), Cow::Borrowed("ab")));
        assert_eq!(decode(b"ab\xe9c", Encoding::Utf8), Err(DecodeError { offset: 2 }));
        assert_eq!(decode(b"\xef\xbb\xbfa", Encoding::Utf8).unwrap(), "\u{feff}a");
    }
}
