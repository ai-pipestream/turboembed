//! Walking UTF-8 text as UTF-16 code units.
//!
//! OpenNLP's algorithms index text by Java `char`, a UTF-16 code unit. A
//! character outside the Basic Multilingual Plane is two units there, a
//! high surrogate then a low one. [`Utf16Cursor`] gives those units from
//! a `&str` without converting or copying it, and each unit knows the
//! UTF-8 byte offset of the character it belongs to, so a port can follow
//! the Java logic unit by unit and still report byte offsets.

/// The UTF-8 length of the character with code point `cp`.
#[inline]
fn utf8_len(cp: u32) -> usize {
    1 + usize::from(cp >= 0x80) + usize::from(cp >= 0x800) + usize::from(cp >= 0x1_0000)
}

/// Decodes the character that starts at byte `i` of valid UTF-8.
#[inline]
fn decode_at(bytes: &[u8], i: usize) -> u32 {
    let b0 = u32::from(bytes[i]);
    if b0 < 0x80 {
        return b0;
    }
    let b1 = u32::from(bytes[i + 1] & 0x3F);
    if b0 < 0xE0 {
        return ((b0 & 0x1F) << 6) | b1;
    }
    let b2 = u32::from(bytes[i + 2] & 0x3F);
    if b0 < 0xF0 {
        return ((b0 & 0x0F) << 12) | (b1 << 6) | b2;
    }
    let b3 = u32::from(bytes[i + 3] & 0x3F);
    ((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3
}

/// The start of the character that ends at byte `end` of valid UTF-8.
/// `end` must be above 0 and on a character boundary.
#[inline]
fn prev_char_start(bytes: &[u8], end: usize) -> usize {
    let mut start = end - 1;
    while bytes[start] & 0xC0 == 0x80 {
        start -= 1;
    }
    start
}

#[inline]
fn high_surrogate(cp: u32) -> u16 {
    (0xD800 + ((cp - 0x1_0000) >> 10)) as u16
}

#[inline]
fn low_surrogate(cp: u32) -> u16 {
    (0xDC00 + ((cp - 0x1_0000) & 0x3FF)) as u16
}

/// The UTF-16 code units of the bytes of a run of whole characters: one
/// per character, two for each four-byte character.
#[inline]
fn units_in(bytes: &[u8]) -> usize {
    bytes.iter().map(|&b| usize::from(b & 0xC0 != 0x80) + usize::from(b >= 0xF0)).sum()
}

/// A position among the UTF-16 code units of a `&str`.
///
/// The positions run from 0 to the text's UTF-16 length; the last one is
/// the end, where there is no unit. The cursor holds the byte offset and
/// code point of the character under it and whether it is on the second
/// unit of that character, so stepping decodes one UTF-8 character and
/// allocates nothing.
#[derive(Clone, Debug)]
pub struct Utf16Cursor<'a> {
    text: &'a str,
    /// The byte offset of the current character, or the text length at the end.
    byte: usize,
    /// The current character's code point; 0 at the end.
    cp: u32,
    /// On the low surrogate of a character outside the BMP.
    low: bool,
    /// The UTF-16 index of the current position.
    unit: usize,
}

impl<'a> Utf16Cursor<'a> {
    /// A cursor on the first unit of `text`.
    pub fn new(text: &'a str) -> Utf16Cursor<'a> {
        let mut cursor = Utf16Cursor { text, byte: 0, cp: 0, low: false, unit: 0 };
        cursor.load();
        cursor
    }

    /// Decodes the character at `self.byte`.
    #[inline]
    fn load(&mut self) {
        self.cp = if self.byte < self.text.len() { decode_at(self.text.as_bytes(), self.byte) } else { 0 };
    }

    /// The text the cursor walks.
    pub fn text(&self) -> &'a str {
        self.text
    }

    /// Whether the cursor is past the last unit.
    #[inline]
    pub fn at_end(&self) -> bool {
        self.byte >= self.text.len()
    }

    /// The code unit at the cursor, or `None` at the end.
    #[inline]
    pub fn unit(&self) -> Option<u16> {
        if self.at_end() {
            None
        } else if self.cp < 0x1_0000 {
            Some(self.cp as u16)
        } else if self.low {
            Some(low_surrogate(self.cp))
        } else {
            Some(high_surrogate(self.cp))
        }
    }

    /// The code point of the character the current unit belongs to, or
    /// `None` at the end. On either half of a surrogate pair it is the
    /// whole character's code point.
    #[inline]
    pub fn code_point(&self) -> Option<u32> {
        if self.at_end() { None } else { Some(self.cp) }
    }

    /// The UTF-8 byte offset of the character the current unit belongs
    /// to; on a low surrogate that is the offset of the pair. At the end
    /// it is the text length.
    #[inline]
    pub fn byte_offset(&self) -> usize {
        self.byte
    }

    /// Whether the current position starts a character: true everywhere
    /// except on a low surrogate. The end counts as a character start.
    #[inline]
    pub fn char_start(&self) -> bool {
        !self.low
    }

    /// The UTF-16 index of the current position, counted from the start
    /// of the text.
    #[inline]
    pub fn unit_index(&self) -> usize {
        self.unit
    }

    /// Moves one unit forward. Returns false, without moving, at the end.
    #[inline]
    pub fn step_forward(&mut self) -> bool {
        if self.at_end() {
            return false;
        }
        self.unit += 1;
        if self.cp >= 0x1_0000 && !self.low {
            self.low = true;
        } else {
            self.byte += utf8_len(self.cp);
            self.low = false;
            self.load();
        }
        true
    }

    /// Moves one unit back. Returns false, without moving, at the start.
    #[inline]
    pub fn step_back(&mut self) -> bool {
        if self.low {
            self.low = false;
            self.unit -= 1;
            return true;
        }
        if self.byte == 0 {
            return false;
        }
        self.byte = prev_char_start(self.text.as_bytes(), self.byte);
        self.load();
        self.low = self.cp >= 0x1_0000;
        self.unit -= 1;
        true
    }

    /// The unit before the cursor, without moving, or `None` at the start.
    #[inline]
    pub fn peek_prev_unit(&self) -> Option<u16> {
        if self.low {
            return Some(high_surrogate(self.cp));
        }
        if self.byte == 0 {
            return None;
        }
        let bytes = self.text.as_bytes();
        let cp = decode_at(bytes, prev_char_start(bytes, self.byte));
        Some(if cp < 0x1_0000 { cp as u16 } else { low_surrogate(cp) })
    }

    /// The unit after the cursor, without moving, or `None` when the
    /// cursor is on the last unit or at the end.
    #[inline]
    pub fn peek_next_unit(&self) -> Option<u16> {
        if self.at_end() {
            return None;
        }
        if self.cp >= 0x1_0000 && !self.low {
            return Some(low_surrogate(self.cp));
        }
        let next = self.byte + utf8_len(self.cp);
        if next >= self.text.len() {
            return None;
        }
        let cp = decode_at(self.text.as_bytes(), next);
        Some(if cp < 0x1_0000 { cp as u16 } else { high_surrogate(cp) })
    }

    /// Moves to the first unit of the character at byte `byte`, or to the
    /// end when `byte` is the text length. The UTF-16 index is updated by
    /// counting the units between the old and the new position.
    ///
    /// # Panics
    ///
    /// When `byte` is not a character boundary of the text.
    pub fn seek_byte(&mut self, byte: usize) {
        assert!(self.text.is_char_boundary(byte), "byte {byte} is not a character boundary");
        let bytes = self.text.as_bytes();
        let char_unit = self.unit - usize::from(self.low);
        self.unit = if byte >= self.byte {
            char_unit + units_in(&bytes[self.byte..byte])
        } else {
            char_unit - units_in(&bytes[byte..self.byte])
        };
        self.byte = byte;
        self.low = false;
        self.load();
    }
}

/// The length of `s` in UTF-16 code units.
pub fn utf16_len(s: &str) -> usize {
    units_in(s.as_bytes())
}

/// The UTF-8 byte offset of UTF-16 index `unit_index` in `s`. `None` when
/// the index falls between the two halves of a surrogate pair or past the
/// end; the text length is a valid index.
pub fn utf16_to_byte(s: &str, unit_index: usize) -> Option<usize> {
    let mut units = 0;
    for (byte, c) in s.char_indices() {
        if units == unit_index {
            return Some(byte);
        }
        units += c.len_utf16();
        if units > unit_index {
            return None;
        }
    }
    (units == unit_index).then_some(s.len())
}

/// The UTF-16 index of UTF-8 byte offset `byte` in `s`, or `None` when
/// `byte` is not a character boundary.
pub fn byte_to_utf16(s: &str, byte: usize) -> Option<usize> {
    s.is_char_boundary(byte).then(|| units_in(&s.as_bytes()[..byte]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "a", "é" (2 bytes), "€" (3 bytes), U+1F600 (4 bytes), "b".
    const TEXT: &str = "a\u{e9}\u{20ac}\u{1f600}b";

    /// (unit, byte offset, unit index, char start) at every position.
    const STEPS: [(Option<u16>, usize, usize, bool); 7] = [
        (Some(0x0061), 0, 0, true),
        (Some(0x00E9), 1, 1, true),
        (Some(0x20AC), 3, 2, true),
        (Some(0xD83D), 6, 3, true),
        (Some(0xDE00), 6, 4, false),
        (Some(0x0062), 10, 5, true),
        (None, 11, 6, true),
    ];

    fn state(cursor: &Utf16Cursor) -> (Option<u16>, usize, usize, bool) {
        (cursor.unit(), cursor.byte_offset(), cursor.unit_index(), cursor.char_start())
    }

    #[test]
    fn steps_forward_and_back() {
        let mut cursor = Utf16Cursor::new(TEXT);
        for (i, expected) in STEPS.iter().enumerate() {
            assert_eq!(state(&cursor), *expected, "forward step {i}");
            assert_eq!(cursor.step_forward(), i + 1 < STEPS.len());
        }
        assert!(cursor.at_end());
        for (i, expected) in STEPS.iter().enumerate().rev() {
            assert_eq!(state(&cursor), *expected, "backward step {i}");
            assert_eq!(cursor.step_back(), i > 0);
        }
        assert_eq!(state(&cursor), STEPS[0]);
    }

    #[test]
    fn units_match_encode_utf16() {
        let expected: Vec<u16> = TEXT.encode_utf16().collect();
        let mut cursor = Utf16Cursor::new(TEXT);
        let mut units = Vec::new();
        while let Some(unit) = cursor.unit() {
            units.push(unit);
            cursor.step_forward();
        }
        assert_eq!(units, expected);
        assert_eq!(utf16_len(TEXT), expected.len());
    }

    #[test]
    fn code_point_on_both_halves() {
        let mut cursor = Utf16Cursor::new(TEXT);
        cursor.seek_byte(6);
        assert_eq!(cursor.code_point(), Some(0x1F600));
        cursor.step_forward();
        assert_eq!(cursor.code_point(), Some(0x1F600));
        cursor.step_forward();
        assert_eq!(cursor.code_point(), Some('b' as u32));
        cursor.step_forward();
        assert_eq!(cursor.code_point(), None);
    }

    #[test]
    fn peeks_do_not_move() {
        let mut cursor = Utf16Cursor::new(TEXT);
        let mut prev = Vec::new();
        let mut next = Vec::new();
        loop {
            let before = state(&cursor);
            prev.push(cursor.peek_prev_unit());
            next.push(cursor.peek_next_unit());
            assert_eq!(state(&cursor), before);
            if !cursor.step_forward() {
                break;
            }
        }
        assert_eq!(prev, [None, Some(0x61), Some(0xE9), Some(0x20AC), Some(0xD83D), Some(0xDE00), Some(0x62)]);
        assert_eq!(next, [Some(0xE9), Some(0x20AC), Some(0xD83D), Some(0xDE00), Some(0x62), None, None]);
    }

    #[test]
    fn seek_byte_sets_the_unit_index() {
        let mut cursor = Utf16Cursor::new(TEXT);
        for &(byte, unit) in &[(10, 5), (3, 2), (11, 6), (0, 0), (6, 3), (1, 1)] {
            cursor.seek_byte(byte);
            assert_eq!((cursor.byte_offset(), cursor.unit_index(), cursor.char_start()), (byte, unit, true));
        }
        // From the low half of the pair, backward and forward.
        cursor.seek_byte(6);
        cursor.step_forward();
        assert_eq!(cursor.unit_index(), 4);
        cursor.seek_byte(3);
        assert_eq!((cursor.unit(), cursor.unit_index()), (Some(0x20AC), 2));
        cursor.seek_byte(6);
        cursor.step_forward();
        cursor.seek_byte(10);
        assert_eq!((cursor.unit(), cursor.unit_index()), (Some(0x62), 5));
    }

    #[test]
    #[should_panic(expected = "not a character boundary")]
    fn seek_inside_a_character_panics() {
        Utf16Cursor::new(TEXT).seek_byte(2);
    }

    #[test]
    fn empty_text() {
        let mut cursor = Utf16Cursor::new("");
        assert!(cursor.at_end());
        assert_eq!((cursor.unit(), cursor.peek_prev_unit(), cursor.peek_next_unit()), (None, None, None));
        assert!(!cursor.step_forward() && !cursor.step_back());
        assert_eq!(utf16_len(""), 0);
        assert_eq!(utf16_to_byte("", 0), Some(0));
        assert_eq!(utf16_to_byte("", 1), None);
    }

    #[test]
    fn conversions() {
        let to_byte: Vec<Option<usize>> = (0..8).map(|i| utf16_to_byte(TEXT, i)).collect();
        assert_eq!(to_byte, [Some(0), Some(1), Some(3), Some(6), None, Some(10), Some(11), None]);
        let to_unit: Vec<Option<usize>> = (0..=11).map(|b| byte_to_utf16(TEXT, b)).collect();
        assert_eq!(to_unit, [Some(0), Some(1), None, Some(2), None, None, Some(3), None, None, None, Some(5), Some(6)]);
        assert_eq!(byte_to_utf16(TEXT, 12), None);
    }

    #[test]
    fn consecutive_pairs() {
        // Two characters outside the BMP, then one inside it.
        let text = "\u{10400}\u{1F1E6}x";
        let mut cursor = Utf16Cursor::new(text);
        let mut seen = Vec::new();
        while !cursor.at_end() {
            seen.push((cursor.unit().unwrap(), cursor.byte_offset(), cursor.unit_index()));
            cursor.step_forward();
        }
        assert_eq!(seen, [(0xD801, 0, 0), (0xDC00, 0, 1), (0xD83C, 4, 2), (0xDDE6, 4, 3), (0x78, 8, 4)]);
        while cursor.step_back() {}
        assert_eq!((cursor.unit(), cursor.unit_index()), (Some(0xD801), 0));
    }
}
