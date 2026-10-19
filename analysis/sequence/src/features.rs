//! [`Features`]: the feature strings of one context, in the order they
//! were generated.
//!
//! OpenNLP collects a context's features in a `List<String>` and the
//! model sums their parameters in that order, so the order is part of the
//! result: the same features in another order can give probabilities that
//! differ in the last bits. The features are kept as Java modified UTF-8,
//! the encoding the model keys its predicates by, in one byte buffer with
//! the end of each feature recorded beside it, so building a context
//! allocates nothing once the buffers have grown, and evaluating it
//! ([`MaxentModel::eval_features`]) looks each feature up without
//! re-encoding it.
//!
//! A feature is built in pieces: the `push_*` methods append to the open
//! feature and [`Features::finish`] closes it, which is how Java's string
//! concatenations (`"w=" + token`) map onto it.
//!
//! [`MaxentModel::eval_features`]: turbo_maxent::MaxentModel::eval_features

use std::fmt::Write as _;

use turbo_unicode::modified_utf8::{push_modified_utf8, push_str_modified_utf8};

use crate::java::{push_lower, push_lower_units, push_units};

/// The features of one context, as Java modified UTF-8.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Features {
    bytes: Vec<u8>,
    /// The end of each finished feature in `bytes`. The open feature runs
    /// from the last end to the end of `bytes`.
    ends: Vec<u32>,
}

impl Features {
    pub fn new() -> Features {
        Features::default()
    }

    /// Removes every feature, keeping the buffers.
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.ends.clear();
    }

    /// The number of finished features.
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Feature `i`, as modified UTF-8.
    pub fn get(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.bytes[start..self.ends[i] as usize]
    }

    /// The finished features in order, as modified UTF-8.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u8]> + Clone + '_ {
        (0..self.ends.len()).map(|i| self.get(i))
    }

    /// Appends text to the open feature.
    pub fn push_str(&mut self, s: &str) -> &mut Features {
        push_str_modified_utf8(&mut self.bytes, s);
        self
    }

    /// Appends `StringUtil.toLowerCase(s)` to the open feature: every code
    /// point lowercased by `Character.toLowerCase(int)`.
    pub fn push_lower(&mut self, s: &str) -> &mut Features {
        push_lower(&mut self.bytes, s);
        self
    }

    /// Appends `s.substring(start, end)` to the open feature, with `start`
    /// and `end` in UTF-16 code units. The substring can end inside a
    /// surrogate pair, as Java's can.
    ///
    /// # Panics
    ///
    /// If the range is not within `s`.
    pub fn push_units(&mut self, s: &str, start: usize, end: usize) -> &mut Features {
        push_units(&mut self.bytes, s, start, end);
        self
    }

    /// Appends text given as UTF-16 code units, which may be lone
    /// surrogates.
    pub fn push_utf16(&mut self, units: &[u16]) -> &mut Features {
        for &u in units {
            push_modified_utf8(&mut self.bytes, u32::from(u));
        }
        self
    }

    /// Appends the lowercase form of text given as UTF-16 code units.
    pub fn push_lower_utf16(&mut self, units: &[u16]) -> &mut Features {
        push_lower_units(&mut self.bytes, units);
        self
    }

    /// Appends text already in modified UTF-8.
    pub fn push_modified_utf8(&mut self, bytes: &[u8]) -> &mut Features {
        self.bytes.extend_from_slice(bytes);
        self
    }

    /// Appends a number as Java's `String.valueOf(int)` writes it.
    pub fn push_int(&mut self, n: usize) -> &mut Features {
        let mut digits = [0u8; 20];
        let mut i = digits.len();
        let mut n = n;
        loop {
            i -= 1;
            digits[i] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        self.bytes.extend_from_slice(&digits[i..]);
        self
    }

    /// Appends an optional value as Java's string concatenation does,
    /// which writes `null` for a null reference.
    pub fn push_or_null(&mut self, s: Option<&str>) -> &mut Features {
        self.push_str(s.unwrap_or("null"))
    }

    /// Closes the open feature, which may be empty.
    pub fn finish(&mut self) {
        let end = u32::try_from(self.bytes.len()).expect("features of fewer than 4 GiB");
        self.ends.push(end);
    }

    /// Adds `s` as one feature.
    pub fn add(&mut self, s: &str) {
        self.push_str(s).finish();
    }

    /// Adds every feature of `other`, each with `prefix` and the number
    /// `n` written before it (`WindowFeatureGenerator`'s `"p" + i +
    /// feature`).
    pub fn extend_numbered(&mut self, prefix: &str, n: usize, other: &Features) {
        for feature in other.iter() {
            self.push_str(prefix).push_int(n).push_modified_utf8(feature).finish();
        }
    }

    /// Adds every feature of `other`.
    pub fn extend_from(&mut self, other: &Features) {
        for feature in other.iter() {
            self.push_modified_utf8(feature).finish();
        }
    }

    /// Feature `i` as a Rust string, with each lone surrogate replaced by
    /// U+FFFD. For messages and tests; the model looks features up by
    /// [`get`](Self::get).
    pub fn to_string_lossy(&self, i: usize) -> String {
        let units = utf16_units(self.get(i));
        crate::java::code_points(&units).map(|cp| char::from_u32(cp).unwrap_or('\u{FFFD}')).collect()
    }

    /// Feature `i` with every UTF-16 code unit outside U+0020..U+007E,
    /// and the backslash, written as `\uXXXX` (upper case hex), and the
    /// tab, line feed and carriage return as `\t`, `\n` and `\r`: the
    /// escaping the tag dictionary goldens hash a dictionary's words and
    /// tags in, and the tests compare and print features in.
    pub fn escaped(&self, i: usize) -> String {
        let mut out = String::new();
        for unit in utf16_units(self.get(i)) {
            escape_unit(&mut out, unit);
        }
        out
    }
}

/// Writes one UTF-16 code unit as [`Features::escaped`] does.
pub fn escape_unit(out: &mut String, u: u16) {
    match u {
        0x5C => out.push_str("\\\\"),
        0x09 => out.push_str("\\t"),
        0x0A => out.push_str("\\n"),
        0x0D => out.push_str("\\r"),
        0x20..=0x7E => out.push(u as u8 as char),
        _ => {
            let _ = write!(out, "\\u{u:04X}");
        }
    }
}

/// The UTF-16 code units of modified UTF-8. The input is what this
/// module wrote, so it is well formed.
fn utf16_units(bytes: &[u8]) -> Vec<u16> {
    let mut units: Vec<u16> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let (unit, len) = if b < 0x80 {
            (u16::from(b), 1)
        } else if b < 0xE0 {
            ((u16::from(b & 0x1F) << 6) | u16::from(bytes[i + 1] & 0x3F), 2)
        } else {
            ((u16::from(b & 0x0F) << 12) | (u16::from(bytes[i + 1] & 0x3F) << 6) | u16::from(bytes[i + 2] & 0x3F), 3)
        };
        units.push(unit);
        i += len;
    }
    units
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_built_in_pieces() {
        let mut f = Features::new();
        f.push_str("w=").push_lower("ABC").finish();
        f.add("");
        f.push_str("p").push_int(12).push_or_null(None).finish();
        f.push_str("pre=").push_units("\u{1F600}x", 0, 1).finish();
        assert_eq!(f.len(), 4);
        assert_eq!(f.get(0), b"w=abc");
        assert_eq!(f.get(1), b"");
        assert_eq!(f.get(2), b"p12null");
        assert_eq!(f.escaped(3), "pre=\\uD83D");
        assert_eq!(f.to_string_lossy(3), "pre=\u{FFFD}");
        assert_eq!(f.iter().count(), 4);
    }

    #[test]
    fn numbered_features_copy_the_bytes() {
        let mut inner = Features::new();
        inner.add("w=\u{E9}\0");
        inner.add("x");
        let mut f = Features::new();
        f.extend_numbered("n", 3, &inner);
        assert_eq!(f.escaped(0), "n3w=\\u00E9\\u0000");
        assert_eq!(f.get(1), b"n3x");
        f.clear();
        assert!(f.is_empty());
        f.extend_from(&inner);
        assert_eq!(f, inner);
    }

    #[test]
    fn escaping_writes_each_unit_outside_printable_ascii_in_hex() {
        let mut f = Features::new();
        f.add("a\\b\t\n\r~\u{7F}\u{1F600}");
        assert_eq!(f.escaped(0), "a\\\\b\\t\\n\\r~\\u007F\\uD83D\\uDE00");
    }
}
