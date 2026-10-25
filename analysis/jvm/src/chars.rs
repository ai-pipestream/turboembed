//! Java's view of text: UTF-16 code units and `String.compareTo` order.
//!
//! The analysis crates hold text as UTF-8 and every offset as a UTF-8 byte
//! offset. Java code sees the same text as a sequence of `char` values,
//! UTF-16 code units, where a character outside the Basic Multilingual
//! Plane is two units (a surrogate pair). Code that ports a Java loop over
//! `charAt` iterates [`Utf16Units`], which yields the same units with the
//! byte offset of the character each belongs to, so a result found on
//! units maps straight back to a byte span.

use std::cmp::Ordering;
use std::iter::FusedIterator;

/// Which part of a character a UTF-16 code unit is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Half {
    /// The character is in the Basic Multilingual Plane and this unit is
    /// all of it.
    Whole,
    /// The first unit of a surrogate pair (U+D800 to U+DBFF).
    High,
    /// The second unit of a surrogate pair (U+DC00 to U+DFFF).
    Low,
}

/// One UTF-16 code unit of a string, as Java's `String.charAt` returns it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Utf16Unit {
    /// The code unit.
    pub unit: u16,
    /// The UTF-8 byte offset of the character the unit belongs to. Both
    /// units of a surrogate pair carry the offset of their character.
    pub byte: u32,
    /// Whether the unit is a whole character or one half of a pair.
    pub half: Half,
}

/// An iterator over the UTF-16 code units of a `&str`, in order, each with
/// the UTF-8 byte offset of its character.
///
/// It decodes the UTF-8 bytes directly. The input is a `&str`, so it is
/// valid UTF-8 and is not validated again.
#[derive(Clone, Debug)]
pub struct Utf16Units<'a> {
    bytes: &'a [u8],
    /// Byte offset of the next character to decode.
    pos: usize,
    /// The low surrogate of the pair whose high surrogate was yielded
    /// last, with its character's byte offset; zero when there is none
    /// (zero is never a low surrogate).
    low: u16,
    low_byte: u32,
}

impl<'a> Utf16Units<'a> {
    /// Iterates the units of `text`.
    ///
    /// # Panics
    ///
    /// If `text` is longer than `u32::MAX` bytes, since byte offsets are
    /// `u32` throughout the analysis crates.
    pub fn new(text: &'a str) -> Utf16Units<'a> {
        assert!(u32::try_from(text.len()).is_ok(), "text of {} bytes does not fit u32 offsets", text.len());
        Utf16Units { bytes: text.as_bytes(), pos: 0, low: 0, low_byte: 0 }
    }
}

impl Iterator for Utf16Units<'_> {
    type Item = Utf16Unit;

    #[inline]
    fn next(&mut self) -> Option<Utf16Unit> {
        if self.low != 0 {
            let unit = Utf16Unit { unit: self.low, byte: self.low_byte, half: Half::Low };
            self.low = 0;
            return Some(unit);
        }
        let start = self.pos;
        let b0 = *self.bytes.get(start)?;
        // The length is below u32::MAX (checked in `new`).
        let byte = start as u32;
        if b0 < 0x80 {
            self.pos = start + 1;
            return Some(Utf16Unit { unit: b0 as u16, byte, half: Half::Whole });
        }
        let b = self.bytes;
        if b0 < 0xE0 {
            self.pos = start + 2;
            let cp = ((b0 as u16 & 0x1F) << 6) | (b[start + 1] as u16 & 0x3F);
            return Some(Utf16Unit { unit: cp, byte, half: Half::Whole });
        }
        if b0 < 0xF0 {
            self.pos = start + 3;
            let cp = ((b0 as u16 & 0x0F) << 12) | ((b[start + 1] as u16 & 0x3F) << 6) | (b[start + 2] as u16 & 0x3F);
            return Some(Utf16Unit { unit: cp, byte, half: Half::Whole });
        }
        self.pos = start + 4;
        let cp = ((b0 as u32 & 0x07) << 18)
            | ((b[start + 1] as u32 & 0x3F) << 12)
            | ((b[start + 2] as u32 & 0x3F) << 6)
            | (b[start + 3] as u32 & 0x3F);
        let v = cp - 0x1_0000;
        self.low = 0xDC00 | (v & 0x3FF) as u16;
        self.low_byte = byte;
        Some(Utf16Unit { unit: 0xD800 | (v >> 10) as u16, byte, half: Half::High })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // A unit takes at most three bytes (a three-byte character) and at
        // least one; a four-byte character is two units, two bytes each.
        let rest = self.bytes.len() - self.pos;
        let pending = usize::from(self.low != 0);
        (rest.div_ceil(3) + pending, Some(rest + pending))
    }
}

impl FusedIterator for Utf16Units<'_> {}

/// The number of UTF-16 code units in `text`: Java's `String.length()` of
/// the same text. Counts from the bytes without decoding or allocating.
pub fn utf16_len(text: &str) -> usize {
    // Every character starts with one byte that is not a continuation byte
    // (10xxxxxx), and a character of four bytes (lead byte 11110xxx) is two
    // units.
    text.as_bytes().iter().map(|&b| usize::from(b & 0xC0 != 0x80) + usize::from(b >= 0xF0)).sum()
}

/// Exactly Java's `a.compareTo(b)`: the difference of the first UTF-16 code
/// units that differ, or else the difference of the lengths in UTF-16 code
/// units.
///
/// This is not the order of `str::cmp`, which is byte order and equal to
/// code point order. The two disagree when a character outside the Basic
/// Multilingual Plane (whose first unit is a surrogate, U+D800 to U+DBFF)
/// meets one from U+E000 to U+FFFF: by code point the first is greater, by
/// UTF-16 unit the second is.
///
/// Java strings are shorter than 2^31 units, so the length difference
/// always fits an `i32` there. For longer Rust strings the difference is
/// clamped to the `i32` range, which keeps its sign.
pub fn compare_to(a: &str, b: &str) -> i32 {
    let (x, y) = (a.as_bytes(), b.as_bytes());
    // Before the first differing byte the texts are equal, so their
    // character boundaries are the same: the character holding the first
    // difference starts at the same offset in both.
    let n = x.len().min(y.len());
    let mut i = x.iter().zip(y).position(|(p, q)| p != q).unwrap_or(n);
    if i == n {
        // One is a prefix of the other (or they are equal).
        let d = utf16_len(a) as i64 - utf16_len(b) as i64;
        return d.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    while !a.is_char_boundary(i) {
        i -= 1;
    }
    // Both characters exist, since a byte differs inside them, and they
    // differ, so one of their first two units differs.
    let ca = a[i..].chars().next().expect("character at differing byte");
    let cb = b[i..].chars().next().expect("character at differing byte");
    let (mut buf_a, mut buf_b) = ([0u16; 2], [0u16; 2]);
    for (&p, &q) in ca.encode_utf16(&mut buf_a).iter().zip(cb.encode_utf16(&mut buf_b).iter()) {
        if p != q {
            return i32::from(p) - i32::from(q);
        }
    }
    unreachable!("two different characters have different UTF-16 units")
}

/// Java's `String.compareTo` order as an [`Ordering`]; see [`compare_to`].
pub fn compare(a: &str, b: &str) -> Ordering {
    compare_to(a, b).cmp(&0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Text with characters of one, two, three and four UTF-8 bytes, and
    /// characters on each side of the surrogate range.
    const MIXED: &str = "a\u{7F}\u{80}\u{E9}\u{7FF}\u{800}\u{D7FF}\u{E000}\u{FFFF}\u{10000}\u{1F600}\u{10FFFF}z";

    #[test]
    fn units_match_the_standard_encoder() {
        let ours: Vec<u16> = Utf16Units::new(MIXED).map(|u| u.unit).collect();
        let std: Vec<u16> = MIXED.encode_utf16().collect();
        assert_eq!(ours, std);
        assert_eq!(utf16_len(MIXED), std.len());
        assert_eq!(utf16_len(""), 0);
        assert_eq!(Utf16Units::new("").next(), None);
    }

    #[test]
    fn units_carry_their_characters_byte_offsets() {
        let mut want = Vec::new();
        for (byte, c) in MIXED.char_indices() {
            let mut buf = [0u16; 2];
            let units = c.encode_utf16(&mut buf);
            if units.len() == 1 {
                want.push(Utf16Unit { unit: units[0], byte: byte as u32, half: Half::Whole });
            } else {
                want.push(Utf16Unit { unit: units[0], byte: byte as u32, half: Half::High });
                want.push(Utf16Unit { unit: units[1], byte: byte as u32, half: Half::Low });
            }
        }
        let got: Vec<Utf16Unit> = Utf16Units::new(MIXED).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn size_hint_bounds_the_count() {
        let mut it = Utf16Units::new(MIXED);
        loop {
            let (lo, hi) = it.size_hint();
            let left = it.clone().count();
            assert!(lo <= left && left <= hi.unwrap(), "{lo} <= {left} <= {hi:?}");
            if it.next().is_none() {
                break;
            }
        }
    }

    fn naive_compare_to(a: &str, b: &str) -> i32 {
        let (x, y): (Vec<u16>, Vec<u16>) = (a.encode_utf16().collect(), b.encode_utf16().collect());
        for (p, q) in x.iter().zip(&y) {
            if p != q {
                return *p as i32 - *q as i32;
            }
        }
        x.len() as i32 - y.len() as i32
    }

    #[test]
    fn compare_differs_from_str_order_outside_the_bmp() {
        // U+1F600 is above U+FFFD by code point, but its first unit, the
        // high surrogate U+D83D, is below U+FFFD.
        let (a, b) = ("\u{1F600}", "\u{FFFD}");
        assert_eq!(a.cmp(b), Ordering::Greater);
        assert_eq!(compare(a, b), Ordering::Less);
        assert_eq!(compare_to(a, b), 0xD83D - 0xFFFD);
    }

    #[test]
    fn compare_matches_a_unit_by_unit_comparison() {
        let alphabet = ["", "a", "b", "\u{E9}", "\u{FFFD}", "\u{E000}", "\u{1F600}", "\u{1F601}", "\u{10000}", "ab"];
        let mut strings = Vec::new();
        for x in alphabet {
            for y in alphabet {
                for z in ["", "a", "\u{1F600}"] {
                    strings.push(format!("{x}{y}{z}"));
                }
            }
        }
        for a in &strings {
            for b in &strings {
                assert_eq!(compare_to(a, b), naive_compare_to(a, b), "{a:?} vs {b:?}");
            }
        }
    }
}
