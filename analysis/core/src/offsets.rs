//! UTF-16 and code point offsets, for Java and the wire. Inside the
//! library every offset is a UTF-8 byte offset into one text view; these
//! are conversions made at the edge, through a table built once per view.
//!
//! The table keeps a checkpoint every `STRIDE` bytes: a byte offset and the
//! UTF-16 and code point offsets of the same place. A conversion finds the
//! nearest checkpoint at or before its input by binary search and counts
//! forward over at most one stride of the text, without allocating. A text
//! that is all ASCII keeps no checkpoints, since all three units agree.

use std::fmt;

use crate::error::Error;
use crate::span::Span;

/// Bytes between checkpoints: a conversion walks at most this many bytes
/// (plus the rest of one character) past the checkpoint it starts from.
pub const STRIDE: u32 = 256;

/// An offset in UTF-16 code units, as Java's `String` indexes text. A
/// distinct type, so it cannot be passed where a byte offset is expected.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Utf16Offset(pub u32);

/// An offset in Unicode code points (Unicode scalar values), as Java's
/// `codePointCount` and Python count them.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CodePointOffset(pub u32);

/// The unit of an offset that could not be converted to bytes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OffsetUnit {
    Utf16,
    CodePoint,
}

impl fmt::Display for OffsetUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            OffsetUnit::Utf16 => "UTF-16 offset",
            OffsetUnit::CodePoint => "code point offset",
        })
    }
}

/// Why an offset could not be converted to bytes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OffsetError {
    /// The offset is past the end of the view, which is `len` long in the
    /// offset's unit.
    PastEnd { len: u32 },
    /// The offset falls between the two UTF-16 units of a surrogate pair,
    /// that is, inside a character outside the Basic Multilingual Plane.
    InsideCharacter,
}

/// The checkpoint table of one text view. It does not hold the text: every
/// method takes the view's text, which must be the text it was built from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OffsetTable {
    ascii: bool,
    /// The text's length in bytes, UTF-16 units and code points.
    len: u32,
    utf16_len: u32,
    code_point_len: u32,
    /// The checkpoints, by column: the first character boundary at or
    /// after each multiple of `STRIDE` below the text's length, starting
    /// at byte 0. All three columns are strictly increasing.
    byte: Vec<u32>,
    utf16: Vec<u32>,
    code_point: Vec<u32>,
}

impl Default for OffsetTable {
    fn default() -> OffsetTable {
        OffsetTable::build("")
    }
}

/// Counts the characters starting in `bytes` (UTF-8 bytes from a character
/// boundary), as (UTF-16 units, code points). A byte that is not a
/// continuation byte (0b10xxxxxx) starts a character; one of 0xF0 or more
/// starts a four byte character, which is two UTF-16 units.
#[inline]
fn count(bytes: &[u8]) -> (u32, u32) {
    let mut chars = 0u32;
    let mut astral = 0u32;
    for &b in bytes {
        chars += ((b as i8) >= -0x40) as u32;
        astral += (b >= 0xF0) as u32;
    }
    (chars + astral, chars)
}

/// The length in bytes of the character a UTF-8 lead byte starts.
#[inline]
fn width(lead: u8) -> u32 {
    match lead {
        0..0x80 => 1,
        0x80..0xE0 => 2,
        0xE0..0xF0 => 3,
        _ => 4,
    }
}

impl OffsetTable {
    /// Builds the table for `text` in one pass.
    pub fn build(text: &str) -> OffsetTable {
        let len = u32::try_from(text.len()).expect("a view shorter than 4 GiB");
        if text.is_ascii() {
            return OffsetTable {
                ascii: true,
                len,
                utf16_len: len,
                code_point_len: len,
                byte: Vec::new(),
                utf16: Vec::new(),
                code_point: Vec::new(),
            };
        }
        let n = len.div_ceil(STRIDE) as usize;
        let (mut byte, mut utf16, mut code_point) =
            (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let bytes = text.as_bytes();
        let (mut at, mut u, mut c) = (0u32, 0u32, 0u32);
        while at < len {
            // `at` is a character boundary at or after this stride's start.
            byte.push(at);
            utf16.push(u);
            code_point.push(c);
            let next = (at / STRIDE + 1) * STRIDE;
            let mut end = next.min(len);
            while (end as usize) < bytes.len() && !text.is_char_boundary(end as usize) {
                end += 1;
            }
            let (du, dc) = count(&bytes[at as usize..end as usize]);
            u += du;
            c += dc;
            at = end;
        }
        OffsetTable { ascii: false, len, utf16_len: u, code_point_len: c, byte, utf16, code_point }
    }

    /// Whether the text is all ASCII, so every conversion is the identity.
    pub fn is_ascii(&self) -> bool {
        self.ascii
    }

    /// How many checkpoints the table keeps: none for ASCII text.
    pub fn checkpoints(&self) -> usize {
        self.byte.len()
    }

    /// The text's length in bytes.
    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The text's length in UTF-16 units.
    pub fn utf16_len(&self) -> Utf16Offset {
        Utf16Offset(self.utf16_len)
    }

    /// The text's length in code points.
    pub fn code_point_len(&self) -> CodePointOffset {
        CodePointOffset(self.code_point_len)
    }

    /// The last checkpoint at or before byte `byte`.
    #[inline]
    fn checkpoint_for_byte(&self, byte: u32) -> usize {
        // Checkpoint 0 is byte 0, so the point is at least 1.
        self.byte.partition_point(|&b| b <= byte) - 1
    }

    #[inline]
    fn check_text(&self, text: &str) {
        debug_assert_eq!(text.len(), self.len as usize, "the text the table was built from");
    }

    /// A byte offset in UTF-16 units. `byte` must be a character boundary
    /// in the text, as every span on the document is.
    pub fn to_utf16(&self, text: &str, byte: u32) -> Utf16Offset {
        self.check_text(text);
        debug_assert!(text.is_char_boundary(byte as usize), "byte {byte} is not a character boundary");
        if self.ascii {
            return Utf16Offset(byte);
        }
        let k = self.checkpoint_for_byte(byte);
        let (du, _) = count(&text.as_bytes()[self.byte[k] as usize..byte as usize]);
        Utf16Offset(self.utf16[k] + du)
    }

    /// A byte offset in code points. `byte` must be a character boundary in
    /// the text.
    pub fn to_code_point(&self, text: &str, byte: u32) -> CodePointOffset {
        self.check_text(text);
        debug_assert!(text.is_char_boundary(byte as usize), "byte {byte} is not a character boundary");
        if self.ascii {
            return CodePointOffset(byte);
        }
        let k = self.checkpoint_for_byte(byte);
        let (_, dc) = count(&text.as_bytes()[self.byte[k] as usize..byte as usize]);
        CodePointOffset(self.code_point[k] + dc)
    }

    /// A UTF-16 offset in bytes. Refused past the end of the text, and
    /// between the two units of a surrogate pair, which is no place in
    /// UTF-8 text.
    pub fn from_utf16(&self, text: &str, at: Utf16Offset) -> Result<u32, Error> {
        self.check_text(text);
        let x = at.0;
        if x > self.utf16_len {
            return Err(Error::Offset {
                unit: OffsetUnit::Utf16,
                at: x,
                reason: OffsetError::PastEnd { len: self.utf16_len },
            });
        }
        if self.ascii {
            return Ok(x);
        }
        let k = self.utf16.partition_point(|&u| u <= x) - 1;
        let bytes = text.as_bytes();
        let (mut pos, mut u) = (self.byte[k], self.utf16[k]);
        while u < x {
            let lead = bytes[pos as usize];
            u += 1 + (lead >= 0xF0) as u32;
            pos += width(lead);
        }
        if u > x {
            return Err(Error::Offset { unit: OffsetUnit::Utf16, at: x, reason: OffsetError::InsideCharacter });
        }
        Ok(pos)
    }

    /// A code point offset in bytes. Refused past the end of the text.
    pub fn from_code_point(&self, text: &str, at: CodePointOffset) -> Result<u32, Error> {
        self.check_text(text);
        let x = at.0;
        if x > self.code_point_len {
            return Err(Error::Offset {
                unit: OffsetUnit::CodePoint,
                at: x,
                reason: OffsetError::PastEnd { len: self.code_point_len },
            });
        }
        if self.ascii {
            return Ok(x);
        }
        let k = self.code_point.partition_point(|&c| c <= x) - 1;
        let bytes = text.as_bytes();
        let mut pos = self.byte[k];
        for _ in self.code_point[k]..x {
            pos += width(bytes[pos as usize]);
        }
        Ok(pos)
    }

    /// A span in UTF-16 units, `[start, end)`.
    pub fn span_to_utf16(&self, text: &str, s: Span) -> (Utf16Offset, Utf16Offset) {
        let mut c = OffsetCursor::new(self, text);
        (c.to_utf16(s.start), c.to_utf16(s.end))
    }

    /// A span in code points, `[start, end)`.
    pub fn span_to_code_point(&self, text: &str, s: Span) -> (CodePointOffset, CodePointOffset) {
        let mut c = OffsetCursor::new(self, text);
        (c.to_code_point(s.start), c.to_code_point(s.end))
    }

    /// Converts a column of byte offsets (a layer's starts or its ends) to
    /// UTF-16 units, appending to `out`. A column in text order is
    /// converted in one forward pass over the text; an offset before the
    /// one preceding it is converted on its own, so any column is accepted.
    pub fn column_to_utf16(&self, text: &str, bytes: &[u32], out: &mut Vec<Utf16Offset>) {
        out.reserve(bytes.len());
        let mut c = OffsetCursor::new(self, text);
        for &b in bytes {
            out.push(if b >= c.byte { c.to_utf16(b) } else { self.to_utf16(text, b) });
        }
    }

    /// Converts a column of byte offsets to code points, appending to
    /// `out`, as `column_to_utf16` does.
    pub fn column_to_code_point(&self, text: &str, bytes: &[u32], out: &mut Vec<CodePointOffset>) {
        out.reserve(bytes.len());
        let mut c = OffsetCursor::new(self, text);
        for &b in bytes {
            out.push(if b >= c.byte { c.to_code_point(b) } else { self.to_code_point(text, b) });
        }
    }
}

/// Converts byte offsets given in text order, walking the text forward
/// once in all: each call counts from where the last one stopped, or from
/// a checkpoint when one lies between the two. Every offset must be a
/// character boundary at or after the one before it.
pub struct OffsetCursor<'a> {
    table: &'a OffsetTable,
    bytes: &'a [u8],
    /// Where the cursor is, in all three units.
    byte: u32,
    utf16: u32,
    code_point: u32,
    /// The first checkpoint after `byte`.
    next: usize,
}

impl<'a> OffsetCursor<'a> {
    /// A cursor at the start of `text`, the text `table` was built from.
    pub fn new(table: &'a OffsetTable, text: &'a str) -> OffsetCursor<'a> {
        table.check_text(text);
        let next = table.byte.len().min(1);
        OffsetCursor { table, bytes: text.as_bytes(), byte: 0, utf16: 0, code_point: 0, next }
    }

    fn advance(&mut self, byte: u32) {
        debug_assert!(byte >= self.byte, "offsets in text order: {byte} after {}", self.byte);
        debug_assert!(
            (byte as usize) == self.bytes.len()
                || (byte as usize) < self.bytes.len() && (self.bytes[byte as usize] as i8) >= -0x40,
            "byte {byte} is not a character boundary"
        );
        let t = self.table;
        if t.byte.get(self.next).is_some_and(|&b| b <= byte) {
            // Jump to the last checkpoint at or before `byte`.
            let j = self.next + t.byte[self.next..].partition_point(|&b| b <= byte);
            self.byte = t.byte[j - 1];
            self.utf16 = t.utf16[j - 1];
            self.code_point = t.code_point[j - 1];
            self.next = j;
        }
        let (du, dc) = count(&self.bytes[self.byte as usize..byte as usize]);
        self.byte = byte;
        self.utf16 += du;
        self.code_point += dc;
    }

    /// The UTF-16 offset of `byte`, at or after the last offset given.
    pub fn to_utf16(&mut self, byte: u32) -> Utf16Offset {
        if self.table.ascii {
            return Utf16Offset(byte);
        }
        self.advance(byte);
        Utf16Offset(self.utf16)
    }

    /// The code point offset of `byte`, at or after the last offset given.
    pub fn to_code_point(&mut self, byte: u32) -> CodePointOffset {
        if self.table.ascii {
            return CodePointOffset(byte);
        }
        self.advance(byte);
        CodePointOffset(self.code_point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every character boundary of `text`, with its offset in all three
    /// units, counted plainly.
    fn boundaries(text: &str) -> Vec<(u32, u32, u32)> {
        let mut out = vec![(0, 0, 0)];
        let (mut u, mut c) = (0, 0);
        for (i, ch) in text.char_indices() {
            u += ch.len_utf16() as u32;
            c += 1;
            out.push(((i + ch.len_utf8()) as u32, u, c));
        }
        out
    }

    #[test]
    fn ascii_keeps_no_checkpoints() {
        let text = "plain ASCII text";
        let t = OffsetTable::build(text);
        assert!(t.is_ascii());
        assert_eq!(t.checkpoints(), 0);
        assert_eq!(t.to_utf16(text, 6), Utf16Offset(6));
        assert_eq!(t.from_code_point(text, CodePointOffset(16)), Ok(16));
    }

    #[test]
    fn checkpoints_land_on_boundaries_one_stride_apart() {
        // 3 byte characters do not divide 256, so most checkpoints are past
        // their stride's start.
        let text = "\u{20AC}".repeat(400);
        let t = OffsetTable::build(&text);
        assert_eq!(t.checkpoints(), text.len().div_ceil(STRIDE as usize));
        for (k, &b) in t.byte.iter().enumerate() {
            assert!(text.is_char_boundary(b as usize));
            assert!(b >= k as u32 * STRIDE && b < k as u32 * STRIDE + 4);
            assert_eq!(t.utf16[k], b / 3);
            assert_eq!(t.code_point[k], b / 3);
        }
    }

    #[test]
    fn every_boundary_both_ways() {
        let text = "a\u{E9}\u{20AC}\u{1D11E}b\u{1F600}\u{4E2D}";
        let t = OffsetTable::build(text);
        for (b, u, c) in boundaries(text) {
            assert_eq!(t.to_utf16(text, b), Utf16Offset(u));
            assert_eq!(t.to_code_point(text, b), CodePointOffset(c));
            assert_eq!(t.from_utf16(text, Utf16Offset(u)), Ok(b));
            assert_eq!(t.from_code_point(text, CodePointOffset(c)), Ok(b));
        }
    }

    #[test]
    fn default_table_matches_empty_text() {
        let t = OffsetTable::default();
        assert_eq!(t, OffsetTable::build(""));
        assert!(t.is_ascii());
        assert_eq!(t.to_utf16("", 0), Utf16Offset(0));
        assert_eq!(t.to_code_point("", 0), CodePointOffset(0));
        assert_eq!(t.from_utf16("", Utf16Offset(0)), Ok(0));
        assert_eq!(t.from_code_point("", CodePointOffset(0)), Ok(0));
    }
}
