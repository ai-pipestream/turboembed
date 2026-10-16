//! Decoders for every encoding of the WHATWG Encoding Standard, each
//! writing the text through a `DecodeBuilder` so the result carries an
//! alignment from the source bytes to the text.
//!
//! The decoders follow the standard's algorithms step for step, including
//! where an error "restores" bytes to the input: those bytes are decoded
//! again, so a replacement character maps to exactly the bytes the error
//! consumed. A lead byte followed by an ASCII byte that does not complete
//! it, for example, gives one U+FFFD for the lead byte and then the ASCII
//! character for its own byte.

pub(crate) mod big5;
pub(crate) mod euc_jp;
pub(crate) mod euc_kr;
pub(crate) mod gb18030;
mod iso_2022_jp;
pub(crate) mod shift_jis;
mod single_byte;
mod utf16;
mod utf8;

pub(crate) use big5::Big5;
pub(crate) use euc_jp::EucJp;
pub(crate) use euc_kr::EucKr;
pub(crate) use gb18030::Gb18030;
pub(crate) use iso_2022_jp::Iso2022Jp;
pub(crate) use shift_jis::ShiftJis;

use std::sync::Arc;

use turbo_analysis::{Alignment, DecodeBuilder, Document, Error, SourceError};
use turbo_encoding_layers::Encoding;

/// The replacement character, as the text a decoding error writes.
pub(crate) const REPLACEMENT: &str = "\u{FFFD}";

/// A decoded text and where each part of it came from in the bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub text: String,
    /// From the bytes (parent) to `text` (child).
    pub alignment: Alignment,
    /// The encoding used: the one asked for, or the one a byte order mark
    /// named.
    pub encoding: Encoding,
    /// Whether a byte order mark was found and dropped from the text.
    pub bom: bool,
    /// How many replacement characters decoding errors wrote.
    pub errors: u32,
}

impl Decoded {
    /// The document whose view 0 is this text, keeping `bytes` (the bytes
    /// that were decoded) below it.
    pub fn into_document(self, bytes: impl Into<Arc<[u8]>>) -> Result<Document, Error> {
        Document::decoded(bytes, self.text, self.alignment)
    }
}

/// What the start of `bytes` says the encoding is: a UTF-8, UTF-16BE or
/// UTF-16LE byte order mark, and its length.
pub fn bom_sniff(bytes: &[u8]) -> Option<(Encoding, usize)> {
    match bytes {
        [0xEF, 0xBB, 0xBF, ..] => Some((Encoding::Utf8, 3)),
        [0xFE, 0xFF, ..] => Some((Encoding::Utf16Be, 2)),
        [0xFF, 0xFE, ..] => Some((Encoding::Utf16Le, 2)),
        _ => None,
    }
}

/// The standard's "decode": a byte order mark wins over `encoding` and is
/// dropped from the text (one deletion edit); the rest is decoded with the
/// encoding the mark named, or `encoding` when there is none.
pub fn decode(bytes: &[u8], encoding: Encoding) -> Result<Decoded, SourceError> {
    let mut out = DecodeBuilder::new(bytes)?;
    let (encoding, bom) = match bom_sniff(bytes) {
        Some((e, len)) => {
            out.replace(len, "");
            (e, true)
        }
        None => (encoding, false),
    };
    let errors = decode_into(bytes, encoding, &mut out);
    let (text, alignment) = out.finish()?;
    Ok(Decoded { text, alignment, encoding, bom, errors })
}

/// Decodes with `encoding` whatever the bytes start with: a byte order
/// mark, if any, is decoded as characters (U+FEFF for UTF-8).
pub fn decode_without_bom_handling(bytes: &[u8], encoding: Encoding) -> Result<Decoded, SourceError> {
    let mut out = DecodeBuilder::new(bytes)?;
    let errors = decode_into(bytes, encoding, &mut out);
    let (text, alignment) = out.finish()?;
    Ok(Decoded { text, alignment, encoding, bom: false, errors })
}

/// Decodes `bytes` from `out`'s position to the end. Returns the number of
/// errors.
fn decode_into(bytes: &[u8], encoding: Encoding, out: &mut DecodeBuilder<'_>) -> u32 {
    match encoding {
        Encoding::Utf8 => utf8::decode(bytes, out),
        Encoding::Utf16Be => drive(bytes, utf16::Utf16::new(true), out),
        Encoding::Utf16Le => drive(bytes, utf16::Utf16::new(false), out),
        Encoding::Gbk | Encoding::Gb18030 => drive(bytes, Gb18030::default(), out),
        Encoding::Big5 => drive(bytes, Big5::default(), out),
        Encoding::EucJp => drive(bytes, EucJp::default(), out),
        Encoding::Iso2022Jp => drive(bytes, Iso2022Jp::default(), out),
        Encoding::ShiftJis => drive(bytes, ShiftJis::default(), out),
        Encoding::EucKr => drive(bytes, EucKr::default(), out),
        Encoding::XUserDefined => single_byte::x_user_defined(bytes, out),
        Encoding::Replacement => {
            if out.position() < bytes.len() {
                out.replace(bytes.len(), REPLACEMENT);
                1
            } else {
                0
            }
        }
        _ => {
            let table = crate::tables::single_byte_table(encoding).expect("every other encoding is single-byte");
            single_byte::decode(bytes, table, out)
        }
    }
}

/// What a stateful decoder made of one byte.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// The byte starts or continues a sequence.
    Pending,
    /// The sequence ends in this character.
    Char(char),
    /// The sequence ends in two characters (four Big5 pointers).
    Pair(char, char),
    /// The sequence ends and writes nothing (an ISO-2022-JP escape).
    Drop,
    /// The sequence is an error; its last `restore` bytes are decoded
    /// again.
    Error { restore: u8 },
}

/// A decoder that takes one byte at a time, in the standard's terms.
pub(crate) trait Stepper {
    /// Whether an ASCII byte outside a sequence always decodes to itself,
    /// so runs of ASCII can be copied without stepping.
    const ASCII_PASSES: bool;

    fn step(&mut self, byte: u8) -> Step;

    /// Whether no sequence is open.
    fn idle(&self) -> bool;

    /// The end of the input: None when no sequence is open, otherwise the
    /// error (an `Error` step) for the open one, which resets the state.
    fn eof(&mut self) -> Option<Step>;
}

/// Runs a stepping decoder over `src` from `out`'s position. Every
/// character is one edit from the bytes of its sequence (none for ASCII
/// that passes through); every error is one U+FFFD over the bytes it
/// consumed. Returns the number of errors.
pub(crate) fn drive<S: Stepper>(src: &[u8], mut s: S, out: &mut DecodeBuilder<'_>) -> u32 {
    let mut errors = 0;
    let mut i = out.position();
    loop {
        if S::ASCII_PASSES && s.idle() {
            let run = src[i..].iter().position(|b| !b.is_ascii()).unwrap_or(src.len() - i);
            if run > 0 {
                i += run;
                out.keep(i);
            }
        }
        if i == src.len() {
            match s.eof() {
                None => break,
                Some(Step::Error { restore }) => {
                    i = src.len() - restore as usize;
                    out.replace(i, REPLACEMENT);
                    errors += 1;
                    continue;
                }
                Some(other) => unreachable!("end of input gives an error or nothing, not {other:?}"),
            }
        }
        let byte = src[i];
        i += 1;
        match s.step(byte) {
            Step::Pending => {}
            Step::Char(c) => out.push(i, c),
            Step::Pair(a, b) => {
                let mut buf = [0u8; 8];
                let len = a.encode_utf8(&mut buf).len();
                let len = len + b.encode_utf8(&mut buf[len..]).len();
                out.replace(i, std::str::from_utf8(&buf[..len]).expect("two encoded characters"));
            }
            Step::Drop => out.replace(i, ""),
            Step::Error { restore } => {
                i -= restore as usize;
                debug_assert!(i > out.position(), "an error consumes at least one byte");
                out.replace(i, REPLACEMENT);
                errors += 1;
            }
        }
    }
    errors
}
