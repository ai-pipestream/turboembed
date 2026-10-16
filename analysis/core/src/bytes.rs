//! The bytes a document was read from, below its first text view.
//!
//! A file arrives as bytes in some encoding (Windows-1252, Shift_JIS,
//! UTF-16LE, UTF-8 with a byte order mark). Decoding turns them into the
//! UTF-8 text of view 0, and the decoder records, through a
//! [`DecodeBuilder`], which run of source bytes each piece of text came
//! from. The record is an ordinary [`Alignment`] whose parent side is byte
//! offsets in the file and whose child side is byte offsets in view 0, so
//! an offset in any view maps to the byte in the original file by one more
//! step up the chain of views.
//!
//! Where the bytes are already the UTF-8 of the text (ASCII in any
//! ASCII-compatible encoding, valid UTF-8 in a UTF-8 file) the alignment
//! has no edits at all. Each other character is one edit from the bytes
//! that encoded it to its UTF-8 (four Big5 codes decode to two characters,
//! which share one edit), and each replacement character is one edit from
//! the bytes it replaced. An edit is 16 bytes, so text that is mostly
//! non-ASCII in a legacy encoding, or any UTF-16 file, holds about 16
//! bytes of alignment per character beside its text.
//!
//! `SourceBytes::new` checks the edits' bounds and order and that the
//! bytes outside every edit are the text's own; the bytes inside an edit
//! are the decoder's to vouch for.

use std::fmt;
use std::sync::Arc;

use crate::span::Span;
use crate::text::{Alignment, Edit};

/// The largest source a document takes: offsets are `u32`.
pub const MAX_SOURCE_LEN: usize = u32::MAX as usize;

/// Why source bytes and their decoded text were refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceError {
    /// The bytes, or the text decoded from them, are longer than a `u32`
    /// offset can reach.
    TooLong { len: u64 },
    /// The alignment does not describe these bytes and this text: an edit
    /// out of bounds or off a character boundary, or an unchanged stretch
    /// whose bytes differ from the text.
    Misaligned(String),
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceError::TooLong { len } => {
                write!(f, "{len} bytes is more than the {MAX_SOURCE_LEN} a u32 offset can reach")
            }
            SourceError::Misaligned(detail) => write!(f, "source bytes and text do not align: {detail}"),
        }
    }
}

impl std::error::Error for SourceError {}

/// A document's original bytes and how its first text view lines up with
/// them: the alignment's parent side is offsets in `bytes`, its child side
/// offsets in the view 0 text.
#[derive(Clone, Debug)]
pub struct SourceBytes {
    bytes: Arc<[u8]>,
    alignment: Alignment,
}

impl SourceBytes {
    /// Checks that `alignment` turns `bytes` into `text`: both short
    /// enough for `u32` offsets, every edit inside both and on character
    /// boundaries of `text`, and every unchanged stretch the same bytes on
    /// both sides.
    pub fn new(bytes: Arc<[u8]>, text: &str, alignment: Alignment) -> Result<SourceBytes, SourceError> {
        check_len(bytes.len())?;
        check_len(text.len())?;
        let (mut parent_at, mut child_at) = (0usize, 0usize);
        for (i, e) in alignment.edits().iter().enumerate() {
            let (p, c) = (e.parent.range(), e.child.range());
            if p.start > p.end || c.start > c.end {
                return Err(SourceError::Misaligned(format!("edit {i} ends before it starts")));
            }
            if p.start < parent_at || c.start < child_at {
                return Err(SourceError::Misaligned(format!("edit {i} starts before the edit before it ends")));
            }
            if p.end > bytes.len() || c.end > text.len() {
                return Err(SourceError::Misaligned(format!("edit {i} ends past the bytes or the text")));
            }
            if !text.is_char_boundary(c.start) || !text.is_char_boundary(c.end) {
                return Err(SourceError::Misaligned(format!("edit {i} cuts a character of the text")));
            }
            same_bytes(&bytes[parent_at..p.start], &text.as_bytes()[child_at..c.start], parent_at)?;
            (parent_at, child_at) = (p.end, c.end);
        }
        same_bytes(&bytes[parent_at..], &text.as_bytes()[child_at..], parent_at)?;
        Ok(SourceBytes { bytes, alignment })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// From the bytes (parent) to the view 0 text (child).
    pub fn alignment(&self) -> &Alignment {
        &self.alignment
    }
}

fn check_len(len: usize) -> Result<(), SourceError> {
    if len > MAX_SOURCE_LEN { Err(SourceError::TooLong { len: len as u64 }) } else { Ok(()) }
}

/// An unchanged stretch: the same bytes on both sides, and as many.
fn same_bytes(source: &[u8], text: &[u8], at: usize) -> Result<(), SourceError> {
    if source.len() != text.len() {
        return Err(SourceError::Misaligned(format!(
            "the unchanged stretch at byte {at} is {} bytes in the source and {} in the text",
            source.len(),
            text.len()
        )));
    }
    match source.iter().zip(text).position(|(a, b)| a != b) {
        Some(k) => Err(SourceError::Misaligned(format!("byte {} is outside every edit but differs", at + k))),
        None => Ok(()),
    }
}

/// Builds the view 0 text from source bytes while recording the
/// alignment, the way a `ViewBuilder` builds a view from its parent. A
/// decoder calls it once per decoded character (or per run of bytes that
/// are already UTF-8), so no text exists without the bytes it came from.
pub struct DecodeBuilder<'a> {
    src: &'a [u8],
    out: String,
    edits: Vec<Edit>,
    at: usize,
}

impl<'a> DecodeBuilder<'a> {
    /// Refuses a source too long for `u32` offsets.
    pub fn new(src: &'a [u8]) -> Result<DecodeBuilder<'a>, SourceError> {
        check_len(src.len())?;
        Ok(DecodeBuilder { src, out: String::with_capacity(src.len()), edits: Vec::new(), at: 0 })
    }

    /// Where the builder is in the source.
    pub fn position(&self) -> usize {
        self.at
    }

    /// The text written so far.
    pub fn text(&self) -> &str {
        &self.out
    }

    /// Copies the source up to `end`, which must be UTF-8 there: no edit.
    ///
    /// # Panics
    ///
    /// If those bytes are not UTF-8.
    pub fn keep(&mut self, end: usize) {
        let run = std::str::from_utf8(&self.src[self.at..end]).expect("kept bytes are UTF-8");
        self.out.push_str(run);
        self.at = end;
    }

    /// Copies `run`, which is the source's own next bytes already checked
    /// as UTF-8 (a valid stretch of a UTF-8 file): no edit.
    ///
    /// # Panics
    ///
    /// If `run` is not the source's next bytes.
    pub fn keep_str(&mut self, run: &str) {
        let end = self.at + run.len();
        let here = &self.src[self.at..end];
        assert!(std::ptr::eq(here.as_ptr(), run.as_ptr()) || here == run.as_bytes(), "kept text is the source's");
        self.out.push_str(run);
        self.at = end;
    }

    /// Writes `c` for the source bytes up to `end`. Bytes that already are
    /// `c` in UTF-8 (ASCII in an ASCII-compatible encoding) are copied with
    /// no edit; anything else is one edit, the smallest unit an offset maps
    /// to.
    pub fn push(&mut self, end: usize, c: char) {
        let mut buf = [0u8; 4];
        let utf8 = c.encode_utf8(&mut buf);
        if &self.src[self.at..end] == utf8.as_bytes() {
            self.out.push_str(utf8);
            self.at = end;
        } else {
            self.replace(end, utf8);
        }
    }

    /// Writes `with` for the source bytes up to `end`, as one edit. An
    /// empty `with` drops the bytes (a byte order mark); `end` at the
    /// current position inserts. A call that neither consumes nor writes
    /// anything records nothing.
    pub fn replace(&mut self, end: usize, with: &str) {
        let out_end = self.out.len() + with.len();
        if out_end > MAX_SOURCE_LEN {
            // The text has outgrown u32 offsets; `finish` refuses it, so
            // no span is recorded.
            self.out.push_str(with);
            self.at = end;
            return;
        }
        let parent = Span::new(self.at as u32, end as u32);
        let child = Span::new(self.out.len() as u32, out_end as u32);
        self.out.push_str(with);
        self.at = end;
        if !(parent.is_empty() && child.is_empty()) {
            self.edits.push(Edit { parent, child });
        }
    }

    /// The text and its alignment to the source. Every source byte must
    /// have been consumed, and the text must fit `u32` offsets; the spans
    /// recorded on the way are only kept when it does.
    pub fn finish(self) -> Result<(String, Alignment), SourceError> {
        check_len(self.out.len())?;
        if self.at != self.src.len() {
            return Err(SourceError::Misaligned(format!(
                "the decoder stopped at byte {} of {}",
                self.at,
                self.src.len()
            )));
        }
        Ok((self.out, Alignment::from_builder(self.edits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::Side;

    #[test]
    fn ascii_and_valid_utf8_cost_no_edits() {
        let src = "plain caf\u{E9}".as_bytes();
        let mut b = DecodeBuilder::new(src).unwrap();
        b.keep(6);
        b.keep_str(std::str::from_utf8(&src[6..]).unwrap());
        let (text, a) = b.finish().unwrap();
        assert_eq!(text, "plain caf\u{E9}");
        assert!(a.edits().is_empty());
        SourceBytes::new(src.into(), &text, a).unwrap();
    }

    #[test]
    fn a_legacy_character_is_one_edit_from_its_bytes() {
        // "caf\xE9!" in Windows-1252: the 1 byte of é becomes 2 in UTF-8.
        let src = b"caf\xE9!";
        let mut b = DecodeBuilder::new(src).unwrap();
        for (i, &byte) in src.iter().enumerate() {
            b.push(i + 1, if byte == 0xE9 { '\u{E9}' } else { byte as char });
        }
        let (text, a) = b.finish().unwrap();
        assert_eq!(text, "caf\u{E9}!");
        assert_eq!(a.edits(), [Edit { parent: Span::new(3, 4), child: Span::new(3, 5) }]);
        assert_eq!(a.span_to_parent(Span::new(3, 5)), Span::new(3, 4));
        assert_eq!(a.to_parent(6, Side::End), 5);
        assert_eq!(a.span_to_child(Span::new(0, 5)), Span::new(0, 6));
        SourceBytes::new(src.as_slice().into(), &text, a).unwrap();
    }

    #[test]
    fn an_unfinished_decode_is_refused() {
        let mut b = DecodeBuilder::new(b"ab").unwrap();
        b.keep(1);
        assert!(matches!(b.finish(), Err(SourceError::Misaligned(_))));
    }

    #[test]
    fn source_bytes_check_the_unchanged_stretches() {
        let a = Alignment::from_edits(vec![Edit { parent: Span::new(1, 2), child: Span::new(1, 3) }]).unwrap();
        // "a\xE9b" decoded as "a" + U+00E9 + "b" is fine; "a" + U+00E9 + "c" is not.
        SourceBytes::new(b"a\xE9b".as_slice().into(), "a\u{E9}b", a.clone()).unwrap();
        let bad = SourceBytes::new(b"a\xE9b".as_slice().into(), "a\u{E9}c", a.clone());
        assert!(matches!(bad, Err(SourceError::Misaligned(_))));
        // An edit that cuts a character of the text is refused.
        let cut = Alignment::from_edits(vec![Edit { parent: Span::new(1, 2), child: Span::new(1, 2) }]).unwrap();
        assert!(SourceBytes::new(b"a\xE9b".as_slice().into(), "a\u{E9}b", cut).is_err());
        // A stretch longer on one side is refused.
        assert!(SourceBytes::new(b"ab".as_slice().into(), "abc", Alignment::identity()).is_err());
    }

    #[test]
    fn lengths_past_u32_are_refused() {
        assert_eq!(check_len(MAX_SOURCE_LEN), Ok(()));
        assert_eq!(check_len(MAX_SOURCE_LEN + 1), Err(SourceError::TooLong { len: MAX_SOURCE_LEN as u64 + 1 }));
        let message = SourceError::TooLong { len: MAX_SOURCE_LEN as u64 + 1 }.to_string();
        assert!(message.contains("4294967296 bytes"), "{message}");
    }

    #[test]
    fn misordered_edits_are_refused() {
        let e = |p: (u32, u32), c: (u32, u32)| Edit { parent: Span::new(p.0, p.1), child: Span::new(c.0, c.1) };
        assert!(Alignment::from_edits(vec![e((2, 3), (2, 4)), e((1, 2), (1, 2))]).is_err());
        assert!(Alignment::from_edits(vec![e((1, 1), (1, 1))]).is_err());
        // Two bytes unchanged before in the parent, one in the child.
        assert!(Alignment::from_edits(vec![e((2, 3), (1, 2))]).is_err());
        assert!(Alignment::from_edits(vec![e((0, 2), (0, 1)), e((3, 4), (2, 5))]).is_ok());
        // Spans ending before they start, built without `Span::new`.
        let back = |p: (u32, u32), c: (u32, u32)| Edit {
            parent: Span { start: p.0, end: p.1 },
            child: Span { start: c.0, end: c.1 },
        };
        assert!(Alignment::from_edits(vec![back((1, 0), (1, 0))]).is_err());
        assert!(Alignment::from_edits(vec![back((3, 0), (3, 0))]).is_err());
    }

    #[test]
    fn edits_ending_before_they_start_are_refused_not_a_panic() {
        // An alignment built in this crate skips `from_edits`; `new` still
        // refuses it.
        let back = Edit { parent: Span { start: 3, end: 0 }, child: Span { start: 3, end: 0 } };
        let a = Alignment::from_builder(vec![back]);
        assert!(matches!(SourceBytes::new(b"a"[..].into(), "abc", a), Err(SourceError::Misaligned(_))));
    }
}
