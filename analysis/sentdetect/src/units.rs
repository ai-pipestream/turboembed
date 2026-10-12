//! Walking a text by UTF-16 code unit, with a fast form for ASCII text.
//!
//! The context generator follows the Java unit by unit through
//! [`Utf16Cursor`]. In a text that is all ASCII every character is one
//! byte and one code unit, so a byte index is the cursor; [`AsciiCursor`]
//! is that, and lets the generator skip the decoding the general cursor
//! does at every step. Both implement [`Units`], and the generator is
//! written once over it.

use turbo_unicode::Utf16Cursor;

use crate::eos::Position;

/// The cursor operations the context generator uses; see [`Utf16Cursor`]
/// for what each means.
pub(crate) trait Units<'t>: Clone {
    /// A cursor on the character at `at`, whose byte and unit indexes the
    /// caller knows to agree (as [`Utf16Cursor::at`] takes them).
    fn place(text: &'t str, at: Position) -> Self;
    fn text(&self) -> &'t str;
    fn unit(&self) -> Option<u16>;
    fn peek_prev_unit(&self) -> Option<u16>;
    fn peek_next_unit(&self) -> Option<u16>;
    fn step_back(&mut self) -> bool;
    fn step_forward(&mut self) -> bool;
    fn unit_index(&self) -> usize;
    fn byte_offset(&self) -> usize;
    fn char_start(&self) -> bool;
}

impl<'t> Units<'t> for Utf16Cursor<'t> {
    #[inline]
    fn place(text: &'t str, at: Position) -> Self {
        Utf16Cursor::at(text, at.byte as usize, at.unit as usize)
    }
    #[inline]
    fn text(&self) -> &'t str {
        Utf16Cursor::text(self)
    }
    #[inline]
    fn unit(&self) -> Option<u16> {
        Utf16Cursor::unit(self)
    }
    #[inline]
    fn peek_prev_unit(&self) -> Option<u16> {
        Utf16Cursor::peek_prev_unit(self)
    }
    #[inline]
    fn peek_next_unit(&self) -> Option<u16> {
        Utf16Cursor::peek_next_unit(self)
    }
    #[inline]
    fn step_back(&mut self) -> bool {
        Utf16Cursor::step_back(self)
    }
    #[inline]
    fn step_forward(&mut self) -> bool {
        Utf16Cursor::step_forward(self)
    }
    #[inline]
    fn unit_index(&self) -> usize {
        Utf16Cursor::unit_index(self)
    }
    #[inline]
    fn byte_offset(&self) -> usize {
        Utf16Cursor::byte_offset(self)
    }
    #[inline]
    fn char_start(&self) -> bool {
        Utf16Cursor::char_start(self)
    }
}

/// A position in a text that is all ASCII: byte index and unit index are
/// the same number.
#[derive(Copy, Clone, Debug)]
pub(crate) struct AsciiCursor<'t> {
    text: &'t str,
    i: usize,
}

impl<'t> AsciiCursor<'t> {
    /// A cursor at byte `i` of `text`, which must be all ASCII.
    #[inline]
    pub(crate) fn at(text: &'t str, i: usize) -> AsciiCursor<'t> {
        debug_assert!(i <= text.len());
        AsciiCursor { text, i }
    }
}

impl<'t> Units<'t> for AsciiCursor<'t> {
    #[inline]
    fn place(text: &'t str, at: Position) -> Self {
        debug_assert_eq!(at.byte, at.unit, "one byte per unit in ASCII text");
        AsciiCursor::at(text, at.byte as usize)
    }
    #[inline]
    fn text(&self) -> &'t str {
        self.text
    }
    #[inline]
    fn unit(&self) -> Option<u16> {
        self.text.as_bytes().get(self.i).map(|&b| u16::from(b))
    }
    #[inline]
    fn peek_prev_unit(&self) -> Option<u16> {
        self.i.checked_sub(1).map(|j| u16::from(self.text.as_bytes()[j]))
    }
    #[inline]
    fn peek_next_unit(&self) -> Option<u16> {
        self.text.as_bytes().get(self.i + 1).map(|&b| u16::from(b))
    }
    #[inline]
    fn step_back(&mut self) -> bool {
        if self.i == 0 {
            return false;
        }
        self.i -= 1;
        true
    }
    #[inline]
    fn step_forward(&mut self) -> bool {
        if self.i >= self.text.len() {
            return false;
        }
        self.i += 1;
        true
    }
    #[inline]
    fn unit_index(&self) -> usize {
        self.i
    }
    #[inline]
    fn byte_offset(&self) -> usize {
        self.i
    }
    #[inline]
    fn char_start(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walks both cursors over an ASCII text in both directions and
    /// compares every answer.
    #[test]
    fn ascii_cursor_agrees_with_the_general_one() {
        let text = "ab. c\td";
        let mut a = AsciiCursor::at(text, 0);
        let mut u = Utf16Cursor::new(text);
        let same = |a: &AsciiCursor<'_>, u: &Utf16Cursor<'_>| {
            assert_eq!(
                (a.unit(), a.peek_prev_unit(), a.peek_next_unit(), a.unit_index(), a.byte_offset(), a.char_start()),
                (u.unit(), u.peek_prev_unit(), u.peek_next_unit(), u.unit_index(), u.byte_offset(), u.char_start())
            );
        };
        loop {
            same(&a, &u);
            let (x, y) = (Units::step_forward(&mut a), Units::step_forward(&mut u));
            assert_eq!(x, y);
            if !x {
                break;
            }
        }
        loop {
            same(&a, &u);
            let (x, y) = (Units::step_back(&mut a), Units::step_back(&mut u));
            assert_eq!(x, y);
            if !x {
                break;
            }
        }
        assert_eq!(Units::text(&a), text);
    }
}
