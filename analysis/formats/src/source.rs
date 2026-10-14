//! The source text a reader reads: its lines, the fields cut from them,
//! and the Java string operations the OpenNLP readers cut them with.
//!
//! Every offset is a UTF-8 byte offset into the source text, as a
//! [`Span`]. Java works in UTF-16 code units; the operations here give the
//! same pieces as Java's on any text that is valid Unicode, because every
//! character they test (whitespace, the separators, `_`, the tab) is in
//! the Basic Multilingual Plane and none of them is a surrogate.

use std::borrow::Cow;
use std::fmt;

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use crate::error::FormatError;

/// A piece of a source text: the text and the bytes it occupies there.
/// A sample's tokens and tags are fields, so each maps back to the exact
/// bytes it was read from.
///
/// The text is almost always the source's own bytes, borrowed. A reader
/// that joins several pieces of the source into one value (the CoNLL-U
/// reader joins the tags of a multiword token's words with `+`) gives the
/// field a text of its own, and the span of the pieces it was made from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Field<'a> {
    pub text: Cow<'a, str>,
    pub span: Span,
}

impl<'a> Field<'a> {
    /// The field at `span` of `source`.
    ///
    /// # Panics
    ///
    /// When `span` is not a range of character boundaries in `source`.
    pub fn of(source: &'a str, span: Span) -> Field<'a> {
        Field { text: Cow::Borrowed(&source[span.range()]), span }
    }

    /// A field with a text of its own, made from the source bytes at
    /// `span`.
    pub fn made(text: String, span: Span) -> Field<'a> {
        Field { text: Cow::Owned(text), span }
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Whether the text is the source's bytes at the span.
    pub fn is_borrowed(&self) -> bool {
        matches!(self.text, Cow::Borrowed(_))
    }

    /// The part of this field from byte `start` to byte `end` of its
    /// text. A borrowed field's part has that part's span; a field with a
    /// text of its own keeps its whole span.
    ///
    /// # Panics
    ///
    /// When the range is not a range of character boundaries in the text.
    pub fn slice(&self, start: usize, end: usize) -> Field<'a> {
        match self.text {
            Cow::Borrowed(text) => {
                let base = self.span.start;
                Field {
                    text: Cow::Borrowed(&text[start..end]),
                    span: Span::new(base + start as u32, base + end as u32),
                }
            }
            Cow::Owned(ref text) => Field { text: Cow::Owned(text[start..end].to_owned()), span: self.span },
        }
    }
}

impl<'a> From<Piece<'a>> for Field<'a> {
    fn from(piece: Piece<'a>) -> Field<'a> {
        Field { text: Cow::Borrowed(piece.text), span: piece.span }
    }
}

impl fmt::Display for Field<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// A borrowed piece of a source text, which the readers cut lines into
/// before they keep the pieces as [`Field`]s.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Piece<'a> {
    pub text: &'a str,
    pub span: Span,
}

impl<'a> Piece<'a> {
    /// The part from byte `start` to byte `end` of the piece's text.
    ///
    /// # Panics
    ///
    /// When the range is not a range of character boundaries in the text.
    pub fn slice(self, start: usize, end: usize) -> Piece<'a> {
        let base = self.span.start;
        Piece { text: &self.text[start..end], span: Span::new(base + start as u32, base + end as u32) }
    }

    pub fn field(self) -> Field<'a> {
        self.into()
    }
}

/// Checks that a source text fits the `u32` offsets of [`Span`].
pub(crate) fn check_len(source: &str) -> Result<(), FormatError> {
    if u32::try_from(source.len()).is_err() {
        return Err(FormatError::TooLong { len: source.len() });
    }
    Ok(())
}

/// One line of a source text, without its line terminator.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Line<'a> {
    /// The line as a piece of the source.
    pub piece: Piece<'a>,
    /// The line's number, counting from 1.
    pub number: u32,
}

impl<'a> Line<'a> {
    pub fn text(&self) -> &'a str {
        self.piece.text
    }

    pub fn span(&self) -> Span {
        self.piece.span
    }

    pub fn field(&self) -> Field<'a> {
        self.piece.into()
    }
}

/// The lines of a source text as `BufferedReader.readLine` returns them,
/// which is what `PlainTextByLineStream` hands every OpenNLP sample
/// stream: a line ends at `\n`, `\r` or `\r\n`, the terminator is not
/// part of it, and text after the last terminator is a last line, while
/// a terminator at the very end does not start an empty one. U+0085,
/// U+2028 and U+2029 do not end a line.
#[derive(Clone, Debug)]
pub struct Lines<'a> {
    source: &'a str,
    pos: usize,
    number: u32,
}

impl<'a> Lines<'a> {
    /// The lines of `source`, which must fit `u32` offsets
    /// ([`FormatError::TooLong`]).
    pub fn new(source: &'a str) -> Result<Lines<'a>, FormatError> {
        check_len(source)?;
        Ok(Lines { source, pos: 0, number: 0 })
    }

    /// The source text.
    pub fn source(&self) -> &'a str {
        self.source
    }
}

impl<'a> Iterator for Lines<'a> {
    type Item = Line<'a>;

    fn next(&mut self) -> Option<Line<'a>> {
        let bytes = self.source.as_bytes();
        if self.pos >= bytes.len() {
            return None;
        }
        let start = self.pos;
        let end = bytes[start..].iter().position(|&b| b == b'\n' || b == b'\r').map_or(bytes.len(), |i| start + i);
        self.pos = match bytes.get(end) {
            Some(b'\r') if bytes.get(end + 1) == Some(&b'\n') => end + 2,
            Some(_) => end + 1,
            None => end,
        };
        self.number += 1;
        let span = Span::new(start as u32, end as u32);
        Some(Line { piece: Piece { text: &self.source[start..end], span }, number: self.number })
    }
}

/// Whether `c` is removed by Java's `String.trim`: U+0000 to U+0020.
fn is_trimmed(c: char) -> bool {
    c <= ' '
}

/// Java's `String.trim()` of a field: every leading and trailing
/// character up to U+0020 removed.
pub(crate) fn java_trim(field: Piece<'_>) -> Piece<'_> {
    let t = field.text;
    let start = t.len() - t.trim_start_matches(is_trimmed).len();
    let end = t.trim_end_matches(is_trimmed).len().max(start);
    field.slice(start, end)
}

/// The tokens `WhitespaceTokenizer.INSTANCE.tokenizePos` finds in a
/// field: the runs of characters that are not whitespace in `mode`.
pub(crate) fn whitespace_tokens<'a>(field: Piece<'a>, mode: WhitespaceMode) -> impl Iterator<Item = Piece<'a>> {
    let text = field.text;
    let mut chars = text.char_indices().peekable();
    std::iter::from_fn(move || {
        while chars.next_if(|&(_, c)| mode.is_whitespace(c as u32)).is_some() {}
        let (start, _) = chars.next()?;
        let mut end = text.len();
        while let Some(&(i, c)) = chars.peek() {
            if mode.is_whitespace(c as u32) {
                end = i;
                break;
            }
            chars.next();
        }
        Some(field.slice(start, end))
    })
}

/// `StringUtil.split(line, separator)` with limit 0: the pieces between
/// every `separator`, with the empty pieces at the end removed. An empty
/// field gives one empty piece.
pub(crate) fn split_fields<'a>(field: Piece<'a>, separator: char) -> Vec<Piece<'a>> {
    if field.text.is_empty() {
        return vec![field];
    }
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in field.text.char_indices() {
        if c == separator {
            out.push(field.slice(start, i));
            start = i + c.len_utf8();
        }
    }
    out.push(field.slice(start, field.text.len()));
    while out.last().is_some_and(|f| f.text.is_empty()) {
        out.pop();
    }
    out
}

/// The byte offset in `text` after its first `units` UTF-16 code units,
/// as Java's `substring(units)` cuts it. `Err(len)` when the text has
/// fewer units (`len` is its length in units), and `Ok(None)` when the
/// cut falls between the two halves of a surrogate pair, which Java cuts
/// and a Rust string cannot hold.
pub(crate) fn utf16_cut(text: &str, units: usize) -> Result<Option<usize>, usize> {
    let mut seen = 0;
    for (i, c) in text.char_indices() {
        if seen == units {
            return Ok(Some(i));
        }
        seen += c.len_utf16();
        if seen > units {
            return Ok(None);
        }
    }
    if seen == units { Ok(Some(text.len())) } else { Err(seen) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(s: &str) -> Vec<(&str, u32, u32)> {
        Lines::new(s).unwrap().map(|l| (l.text(), l.span().start, l.span().end)).collect()
    }

    #[test]
    fn lines_end_as_buffered_reader_ends_them() {
        assert_eq!(lines(""), []);
        assert_eq!(lines("a"), [("a", 0, 1)]);
        assert_eq!(lines("a\n"), [("a", 0, 1)]);
        assert_eq!(lines("a\n\n"), [("a", 0, 1), ("", 2, 2)]);
        assert_eq!(lines("\r\n"), [("", 0, 0)]);
        assert_eq!(lines("a\rb\r\nc\n\rd"), [("a", 0, 1), ("b", 2, 3), ("c", 5, 6), ("", 7, 7), ("d", 8, 9)]);
        // Unicode line separators are part of the line.
        assert_eq!(lines("a\u{2028}b\u{85}"), [("a\u{2028}b\u{85}", 0, 7)]);
    }

    #[test]
    fn line_numbers_count_from_one() {
        let n: Vec<u32> = Lines::new("a\n\nb").unwrap().map(|l| l.number).collect();
        assert_eq!(n, [1, 2, 3]);
    }

    #[test]
    fn trim_removes_control_characters_and_space_only() {
        let s = "\u{0}\t a \u{a0}\u{1f}";
        let t = java_trim(Piece { text: s, span: Span::new(0, s.len() as u32) });
        assert_eq!(t.text, "a \u{a0}");
        assert_eq!(t.span, Span::new(3, 7));
        let blank = " \t ";
        assert_eq!(java_trim(Piece { text: blank, span: Span::new(0, 3) }).text, "");
    }

    #[test]
    fn whitespace_tokens_follow_the_mode() {
        let s = " a\u{85}b\u{1c}c  d ";
        let f = Piece { text: s, span: Span::new(0, s.len() as u32) };
        let unicode: Vec<&str> = whitespace_tokens(f, WhitespaceMode::Unicode).map(|t| t.text).collect();
        let legacy: Vec<&str> = whitespace_tokens(f, WhitespaceMode::Legacy).map(|t| t.text).collect();
        assert_eq!(unicode, ["a", "b\u{1c}c", "d"]);
        assert_eq!(legacy, ["a\u{85}b", "c", "d"]);
        let spans: Vec<Span> = whitespace_tokens(f, WhitespaceMode::Unicode).map(|t| t.span).collect();
        assert_eq!(spans, [Span::new(1, 2), Span::new(4, 7), Span::new(9, 10)]);
    }

    #[test]
    fn split_drops_trailing_empty_pieces() {
        let piece = |s: &'static str| -> Vec<&'static str> {
            split_fields(Piece { text: s, span: Span::new(0, s.len() as u32) }, '\t')
                .into_iter()
                .map(|f| f.text)
                .collect()
        };
        assert_eq!(piece(""), [""]);
        assert_eq!(piece("a\tb\tc\t"), ["a", "b", "c"]);
        assert_eq!(piece("a\t\tc"), ["a", "", "c"]);
        assert_eq!(piece("\t\t"), Vec::<&str>::new());
        assert_eq!(piece("\ta"), ["", "a"]);
    }

    #[test]
    fn borrowed_and_made_fields_slice() {
        let s = "xB-NPy";
        let f = Field::of(s, Span::new(1, 5));
        assert!(f.is_borrowed());
        let part = f.slice(2, 4);
        assert_eq!((part.as_str(), part.span), ("NP", Span::new(3, 5)));
        let made = Field::made("ADP+DET".to_owned(), Span::new(0, 6));
        assert!(!made.is_borrowed());
        let part = made.slice(4, 7);
        assert_eq!((part.as_str(), part.span), ("DET", Span::new(0, 6)));
    }

    #[test]
    fn utf16_cuts() {
        assert_eq!(utf16_cut("B-NP", 2), Ok(Some(2)));
        assert_eq!(utf16_cut("ab", 2), Ok(Some(2)));
        assert_eq!(utf16_cut("a", 2), Err(1));
        assert_eq!(utf16_cut("", 2), Err(0));
        assert_eq!(utf16_cut("\u{1F600}x", 2), Ok(Some(4)));
        assert_eq!(utf16_cut("a\u{1F600}", 2), Ok(None));
        assert_eq!(utf16_cut("\u{e9}\u{e9}x", 2), Ok(Some(4)));
    }
}
