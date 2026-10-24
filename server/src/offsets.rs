//! Edge offset conversion at the gRPC boundary: internal UTF-8 byte
//! offsets to and from UTF-16 code units (Java string indices) and
//! Unicode code points, through a view text's checkpoint table
//! (`turbo_analysis::OffsetTable`). An offset that is out of bounds,
//! inverted, inside a code point, or inside a UTF-16 surrogate pair
//! is refused with `TURBO_E_INVALID_ARGUMENT`; nothing is snapped to
//! a boundary or truncated.

use turbo_analysis::{CodePointOffset, CodePointSpan, OffsetTable, Span, Utf16Offset, Utf16Span};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::OffsetMode as ProtoOffsetMode;

/// The unit span offsets on the wire are counted in.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum OffsetMode {
    #[default]
    Utf8,
    Utf16,
    CodePoint,
}

impl OffsetMode {
    pub fn from_proto(mode: i32) -> Result<OffsetMode, Failure> {
        match ProtoOffsetMode::try_from(mode) {
            Ok(ProtoOffsetMode::Utf8) => Ok(OffsetMode::Utf8),
            Ok(ProtoOffsetMode::Utf16) => Ok(OffsetMode::Utf16),
            Ok(ProtoOffsetMode::CodePoint) => Ok(OffsetMode::CodePoint),
            Err(_) => Err(invalid(format!("offset mode {mode} is not a known OffsetMode"))),
        }
    }

    pub fn to_proto(self) -> i32 {
        let mode = match self {
            OffsetMode::Utf8 => ProtoOffsetMode::Utf8,
            OffsetMode::Utf16 => ProtoOffsetMode::Utf16,
            OffsetMode::CodePoint => ProtoOffsetMode::CodePoint,
        };
        mode as i32
    }
}

fn invalid(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

/// Converts spans between wire offsets and UTF-8 byte offsets for
/// one view text.
pub struct Offsetter {
    table: OffsetTable,
    mode: OffsetMode,
}

impl Offsetter {
    pub fn new(text: &str, mode: OffsetMode) -> Offsetter {
        Offsetter { table: OffsetTable::build(text), mode }
    }

    pub fn mode(&self) -> OffsetMode {
        self.mode
    }

    /// One wire span to its byte span in `text`. The core table
    /// refuses out-of-bounds and mid-codepoint/surrogate ends; the
    /// order and boundary checks for UTF-8 mode are the same rules.
    pub fn from_wire(&self, text: &str, what: &str, start: u32, end: u32) -> Result<Span, Failure> {
        let parsed = match self.mode {
            OffsetMode::Utf8 => {
                if start > end || end as usize > text.len() {
                    return Err(invalid(format!("{what} [{start}, {end}) is outside the {}-byte text", text.len())));
                }
                if !text.is_char_boundary(start as usize) || !text.is_char_boundary(end as usize) {
                    return Err(invalid(format!("{what} [{start}, {end}) splits a code point")));
                }
                Ok(Span::new(start, end))
            }
            OffsetMode::Utf16 => {
                // try_new, never new: an inverted wire span is a
                // refusal, not a panic.
                let Some(wire) = Utf16Span::try_new(Utf16Offset(start), Utf16Offset(end)) else {
                    return Err(invalid(format!("{what} [{start}, {end}) ends before it starts")));
                };
                self.table.span_from_utf16(text, wire).map_err(|e| invalid(format!("{what}: {e}")))
            }
            OffsetMode::CodePoint => {
                let Some(wire) = CodePointSpan::try_new(CodePointOffset(start), CodePointOffset(end)) else {
                    return Err(invalid(format!("{what} [{start}, {end}) ends before it starts")));
                };
                self.table.span_from_code_point(text, wire).map_err(|e| invalid(format!("{what}: {e}")))
            }
        };
        let span = parsed?;
        if span.start > span.end {
            return Err(invalid(format!("{what} [{start}, {end}) ends before it starts")));
        }
        Ok(span)
    }

    /// One internal byte span (already proven on char boundaries by
    /// its layer's check) to wire offsets.
    pub fn to_wire(&self, text: &str, span: Span) -> (u32, u32) {
        match self.mode {
            OffsetMode::Utf8 => (span.start, span.end),
            OffsetMode::Utf16 => {
                let s = self.table.span_to_utf16_typed(text, span);
                (s.start.0, s.end.0)
            }
            OffsetMode::CodePoint => {
                let s = self.table.span_to_code_point_typed(text, span);
                (s.start.0, s.end.0)
            }
        }
    }

    /// Converts a column of internal byte offsets (already proven on char
    /// boundaries) to wire offsets, using [`OffsetTable::column_to_utf16`]
    /// and [`OffsetTable::column_to_code_point`] for linear-time forward
    /// cursor conversion on ordered columns with automatic fallback for
    /// out-of-order entries.
    pub fn column_to_wire(&self, text: &str, bytes: &[u32]) -> Vec<u32> {
        match self.mode {
            OffsetMode::Utf8 => bytes.to_vec(),
            OffsetMode::Utf16 => {
                let mut out = Vec::with_capacity(bytes.len());
                self.table.column_to_utf16(text, bytes, &mut out);
                out.into_iter().map(|u| u.0).collect()
            }
            OffsetMode::CodePoint => {
                let mut out = Vec::with_capacity(bytes.len());
                self.table.column_to_code_point(text, bytes, &mut out);
                out.into_iter().map(|c| c.0).collect()
            }
        }
    }

    /// Converts a slice of internal byte spans to `(start, end)` wire
    /// columns via [`Offsetter::column_to_wire`].
    pub fn spans_to_wire(&self, text: &str, spans: &[Span]) -> (Vec<u32>, Vec<u32>) {
        let starts: Vec<u32> = spans.iter().map(|s| s.start).collect();
        let ends: Vec<u32> = spans.iter().map(|s| s.end).collect();
        (self.column_to_wire(text, &starts), self.column_to_wire(text, &ends))
    }

    /// Converts parallel `(start, end)` wire columns to internal byte
    /// spans in `text`, refusing out-of-bounds, inverted, mid-codepoint,
    /// or mid-surrogate offsets.
    pub fn spans_from_wire(&self, text: &str, what: &str, starts: &[u32], ends: &[u32]) -> Result<Vec<Span>, Failure> {
        starts.iter().zip(ends).map(|(&s, &e)| self.from_wire(text, what, s, e)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // "a" + e-acute (2 bytes, 1 UTF-16 unit) + grinning face
    // (4 bytes, a surrogate pair) + CJK (3 bytes, 1 unit).
    const TEXT: &str = "a\u{e9}\u{1f600}\u{4e2d}";

    #[test]
    fn utf16_and_code_point_round_trip_every_boundary() {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let o = Offsetter::new(TEXT, mode);
            let mut bounds: Vec<u32> = TEXT.char_indices().map(|(i, _)| i as u32).collect();
            bounds.push(TEXT.len() as u32);
            for &a in &bounds {
                for &b in &bounds {
                    if a > b {
                        continue;
                    }
                    let span = Span::new(a, b);
                    let (ws, we) = o.to_wire(TEXT, span);
                    assert_eq!(o.from_wire(TEXT, "span", ws, we).unwrap(), span, "{mode:?} {a}..{b}");
                }
            }
        }
        // Java UTF-16 indices, counted by hand: a=0, e-acute=1,
        // surrogate pair=2..4, CJK=4, end=5.
        let o = Offsetter::new(TEXT, OffsetMode::Utf16);
        assert_eq!(o.to_wire(TEXT, Span::new(3, 7)), (2, 4));
        assert_eq!(o.from_wire(TEXT, "span", 2, 4).unwrap(), Span::new(3, 7));
        let o = Offsetter::new(TEXT, OffsetMode::CodePoint);
        assert_eq!(o.to_wire(TEXT, Span::new(3, 7)), (2, 3));
    }

    #[test]
    fn malformed_wire_offsets_are_refused_not_snapped() {
        let o16 = Offsetter::new(TEXT, OffsetMode::Utf16);
        assert!(o16.from_wire(TEXT, "span", 3, 4).is_err(), "mid-surrogate start");
        assert!(o16.from_wire(TEXT, "span", 2, 3).is_err(), "mid-surrogate end");
        assert!(o16.from_wire(TEXT, "span", 0, 99).is_err(), "out of bounds");
        assert!(o16.from_wire(TEXT, "span", 4, 2).is_err(), "inverted");
        let o8 = Offsetter::new(TEXT, OffsetMode::Utf8);
        assert!(o8.from_wire(TEXT, "span", 2, 3).is_err(), "mid-codepoint");
        assert!(o8.from_wire(TEXT, "span", 0, 100).is_err(), "out of bounds");
        let ocp = Offsetter::new(TEXT, OffsetMode::CodePoint);
        assert!(ocp.from_wire(TEXT, "span", 0, 5).is_err(), "past the last code point");
        assert!(OffsetMode::from_proto(99).is_err());
    }
}
