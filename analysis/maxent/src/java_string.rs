//! A Java string that may hold a lone surrogate.
//!
//! OpenNLP's context generators cut text at UTF-16 code units: the
//! tokenizer's features name the character before and after a candidate
//! split as one `char`, which for a character outside the BMP is half a
//! surrogate pair, and the lemmatizer's classes are edit scripts over the
//! units of a word, so a class can insert or delete half a pair. Such a
//! predicate or outcome is a Java string no Rust `String` can hold.
//! [`JavaString`] holds any Java string as Java's modified UTF-8, the
//! encoding model files store their labels in and the one a
//! [`MaxentModel`](crate::MaxentModel) keys its predicates by, so a label
//! travels from training events to a written model, and from a model file
//! to the features looked up in it, without re-encoding. Two Java strings
//! are equal when their bytes are, and a `JavaString` orders as
//! `String.compareTo` does (by UTF-16 code units).
//!
//! Comparing with a `str` ([`JavaString::eq_str`], [`JavaString::starts_with`],
//! [`JavaString::ends_with`]) compares bytes when the `str`'s UTF-8 is its
//! modified UTF-8 (no U+0000 and nothing outside the BMP), which every
//! label a validator compares against is, and compares UTF-16 units
//! otherwise; neither allocates.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;

use turbo_unicode::modified_utf8::{push_modified_utf8, push_str_modified_utf8};

use crate::java_data::{canonical_modified_utf8, check_modified_utf8};

/// A Java string, in modified UTF-8 as `DataOutputStream.writeUTF` writes
/// it (U+0000 as `C0 80`, every UTF-16 unit above U+07FF as three bytes,
/// so a character outside the BMP is its two surrogates).
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct JavaString {
    bytes: Vec<u8>,
}

/// Whether the UTF-8 of `s` is also its modified UTF-8: no U+0000 (two
/// bytes there) and no character outside the BMP (two surrogates there).
#[inline]
pub(crate) fn utf8_is_modified_utf8(s: &str) -> bool {
    !s.bytes().any(|b| b == 0 || b >= 0xF0)
}

impl JavaString {
    /// The Java string with the same characters as `s`.
    pub fn new(s: &str) -> JavaString {
        let mut bytes = Vec::with_capacity(s.len());
        push_str_modified_utf8(&mut bytes, s);
        JavaString { bytes }
    }

    /// The Java string of these UTF-16 code units, lone surrogates
    /// included.
    pub fn from_utf16(units: &[u16]) -> JavaString {
        let mut bytes = Vec::with_capacity(units.len());
        for &u in units {
            push_modified_utf8(&mut bytes, u32::from(u));
        }
        JavaString { bytes }
    }

    /// A Java string from modified UTF-8 as a model file or `readUTF`
    /// gives it; `None` when the bytes are not well formed. A non-canonical
    /// encoding (an overlong sequence, a raw zero byte) is made canonical,
    /// so equal strings have equal bytes.
    pub fn from_modified_utf8(bytes: &[u8]) -> Option<JavaString> {
        check_modified_utf8(bytes).ok()?;
        Some(JavaString::from_checked(bytes))
    }

    /// A Java string from modified UTF-8 already checked to be well formed
    /// (what [`JavaData::read_utf`](crate::java_data::JavaData::read_utf)
    /// returns), made canonical.
    pub(crate) fn from_checked(bytes: &[u8]) -> JavaString {
        JavaString { bytes: canonical_modified_utf8(bytes).into_owned() }
    }

    /// The modified UTF-8 bytes: what `writeUTF` writes after the length.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The length in modified UTF-8 bytes, which `writeUTF` limits to 65535.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Keeps the first `len` bytes, which must end a unit's encoding, as
    /// `substring(0, n)` keeps the first n units.
    pub fn truncate(&mut self, len: usize) {
        debug_assert!(len >= self.bytes.len() || self.bytes[len] & 0xC0 != 0x80, "byte {len} is inside a unit");
        self.bytes.truncate(len);
    }

    /// The UTF-16 code units, as `String.charAt` gives them.
    pub fn utf16_units(&self) -> Utf16Units<'_> {
        Utf16Units { bytes: &self.bytes }
    }

    /// The UTF-16 code units from byte `start` on, which must begin a
    /// unit: `substring(n)` as `charAt` gives it.
    pub fn utf16_units_from(&self, start: usize) -> Utf16Units<'_> {
        debug_assert!(start >= self.bytes.len() || self.bytes[start] & 0xC0 != 0x80, "byte {start} is inside a unit");
        Utf16Units { bytes: &self.bytes[start..] }
    }

    /// `String.length()`.
    pub fn utf16_len(&self) -> usize {
        self.utf16_units().count()
    }

    /// `equals(s)`.
    pub fn eq_str(&self, s: &str) -> bool {
        if utf8_is_modified_utf8(s) {
            return self.bytes == s.as_bytes();
        }
        self.utf16_units().eq(s.encode_utf16())
    }

    /// `startsWith(prefix)`.
    pub fn starts_with(&self, prefix: &str) -> bool {
        if utf8_is_modified_utf8(prefix) {
            return self.bytes.starts_with(prefix.as_bytes());
        }
        let mut units = self.utf16_units();
        prefix.encode_utf16().all(|p| units.next() == Some(p))
    }

    /// `endsWith(suffix)`.
    pub fn ends_with(&self, suffix: &str) -> bool {
        if utf8_is_modified_utf8(suffix) {
            return self.bytes.ends_with(suffix.as_bytes());
        }
        let suffix: Vec<u16> = suffix.encode_utf16().collect();
        let units: Vec<u16> = self.utf16_units().collect();
        units.ends_with(&suffix)
    }

    /// The string as `&str` when it holds no lone surrogate; modified
    /// UTF-8 is then UTF-8, except for U+0000 and characters outside the
    /// BMP, which are re-encoded.
    pub fn to_str(&self) -> Option<Cow<'_, str>> {
        if let Ok(s) = std::str::from_utf8(&self.bytes) {
            return Some(Cow::Borrowed(s));
        }
        let mut s = String::with_capacity(self.bytes.len());
        for c in char::decode_utf16(self.utf16_units()) {
            s.push(c.ok()?);
        }
        Some(Cow::Owned(s))
    }

    /// The string with every lone surrogate replaced by U+FFFD.
    pub fn to_string_lossy(&self) -> String {
        char::decode_utf16(self.utf16_units()).map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER)).collect()
    }

    /// Appends what `getBytes(StandardCharsets.UTF_8)` returns: UTF-8, with
    /// `?` for each lone surrogate, which is the encoder's replacement.
    pub fn push_java_utf8(&self, out: &mut String) {
        for c in char::decode_utf16(self.utf16_units()) {
            out.push(c.unwrap_or('?'));
        }
    }
}

/// The UTF-16 code units of a [`JavaString`], decoded from its canonical
/// modified UTF-8.
#[derive(Clone)]
pub struct Utf16Units<'a> {
    bytes: &'a [u8],
}

impl Iterator for Utf16Units<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        let (&b, rest) = self.bytes.split_first()?;
        let (unit, len) = match b {
            0x00..=0x7F => (u16::from(b), 1),
            0xC0..=0xDF => ((u16::from(b & 0x1F) << 6) | u16::from(rest[0] & 0x3F), 2),
            _ => ((u16::from(b & 0x0F) << 12) | (u16::from(rest[0] & 0x3F) << 6) | u16::from(rest[1] & 0x3F), 3),
        };
        self.bytes = &self.bytes[len..];
        Some(unit)
    }
}

/// `String.compareTo`: by UTF-16 code units, then by length.
impl Ord for JavaString {
    fn cmp(&self, other: &JavaString) -> Ordering {
        self.utf16_units().cmp(other.utf16_units())
    }
}

impl PartialOrd for JavaString {
    fn partial_cmp(&self, other: &JavaString) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<&str> for JavaString {
    fn from(s: &str) -> JavaString {
        JavaString::new(s)
    }
}

impl From<String> for JavaString {
    fn from(s: String) -> JavaString {
        JavaString::new(&s)
    }
}

impl From<&String> for JavaString {
    fn from(s: &String) -> JavaString {
        JavaString::new(s)
    }
}

impl PartialEq<str> for JavaString {
    fn eq(&self, other: &str) -> bool {
        self.eq_str(other)
    }
}

impl PartialEq<&str> for JavaString {
    fn eq(&self, other: &&str) -> bool {
        self.eq_str(other)
    }
}

impl PartialEq<String> for JavaString {
    fn eq(&self, other: &String) -> bool {
        self.eq_str(other)
    }
}

impl PartialEq<JavaString> for str {
    fn eq(&self, other: &JavaString) -> bool {
        other.eq_str(self)
    }
}

impl PartialEq<JavaString> for &str {
    fn eq(&self, other: &JavaString) -> bool {
        other.eq_str(self)
    }
}

impl PartialEq<JavaString> for String {
    fn eq(&self, other: &JavaString) -> bool {
        other.eq_str(self)
    }
}

impl fmt::Display for JavaString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_lossy())
    }
}

/// The string with each lone surrogate written as `\u{XXXX}`.
impl fmt::Debug for JavaString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("\"")?;
        for c in char::decode_utf16(self.utf16_units()) {
            match c {
                Ok(c) => write!(f, "{}", c.escape_debug())?,
                Err(e) => write!(f, "\\u{{{:04x}}}", e.unpaired_surrogate())?,
            }
        }
        f.write_str("\"")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_as_write_utf_does() {
        assert_eq!(JavaString::new("a\0é€😀").as_bytes(), b"a\xC0\x80\xC3\xA9\xE2\x82\xAC\xED\xA0\xBD\xED\xB8\x80");
        assert_eq!(JavaString::new("a\0é€😀").utf16_len(), 6);
        assert_eq!(JavaString::new("a\0é€😀").len(), 14);
        let lone = JavaString::from_utf16(&[0x70, 0x3D, 0xD83C]);
        assert_eq!(lone.as_bytes(), b"p=\xED\xA0\xBC");
        assert_eq!(lone.utf16_units().collect::<Vec<_>>(), [0x70, 0x3D, 0xD83C]);
        assert_eq!(JavaString::from_modified_utf8(b"p=\xED\xA0\xBC").unwrap(), lone);
        assert_eq!(JavaString::from_modified_utf8(b"a\0").unwrap(), JavaString::new("a\0"));
        assert_eq!(JavaString::new("a\0é€😀").utf16_units_from(3).collect::<Vec<_>>(), [0xE9, 0x20AC, 0xD83D, 0xDE00]);
        assert!(JavaString::from_modified_utf8(b"\xFF").is_none());
    }

    #[test]
    fn converts_back_to_text() {
        let s = JavaString::new("a\0😀");
        assert_eq!(s.to_str().unwrap(), "a\0😀");
        assert!(matches!(s.to_str(), Some(Cow::Owned(_))));
        assert!(matches!(JavaString::new("plain").to_str(), Some(Cow::Borrowed("plain"))));
        let lone = JavaString::from_utf16(&[0x61, 0xD83C, 0x62]);
        assert_eq!(lone.to_str(), None);
        assert_eq!(lone.to_string_lossy(), "a\u{FFFD}b");
        let mut utf8 = String::new();
        lone.push_java_utf8(&mut utf8);
        assert_eq!(utf8, "a?b");
        assert_eq!(format!("{lone:?}"), "\"a\\u{d83c}b\"");
        assert_eq!(lone.to_string(), "a\u{FFFD}b");
        assert!(JavaString::new("x") == "x");
    }

    #[test]
    fn compares_with_text_without_encoding_it() {
        let plain = JavaString::new("B-per");
        let owned = String::from("B-per");
        assert!(plain == "B-per" && "B-per" == plain && plain == owned);
        assert!(plain != "B-pe" && plain != "b-per");
        assert!(plain.starts_with("B-") && plain.starts_with("") && !plain.starts_with("B-x"));
        assert!(plain.ends_with("per") && !plain.ends_with("B-per-"));
        // Text whose UTF-8 is not its modified UTF-8 is compared by units.
        let astral = JavaString::new("a\0😀b");
        assert!(astral == "a\0😀b" && astral != "a\0😀" && !astral.eq_str("a"));
        assert!(astral.starts_with("a\0😀") && !astral.starts_with("a\0😁"));
        assert!(astral.ends_with("😀b") && !astral.ends_with("\0b"));
        assert!(!JavaString::new("a").starts_with("a\0") && !JavaString::new("b").ends_with("😀b"));
        // A lone surrogate equals no text, but its prefix and suffix can.
        let lone = JavaString::from_utf16(&[0x42, 0x2D, 0xD83D]);
        assert!(lone != "B-" && !lone.eq_str("B-\u{FFFD}"));
        assert!(lone.starts_with("B-") && !lone.ends_with("-"));
    }

    #[test]
    fn orders_by_utf16_units() {
        let mut v: Vec<JavaString> =
            ["z", "😀", "Z", "a\0", "a", "\u{FFFF}"].iter().map(|s| JavaString::new(s)).collect();
        v.sort();
        let order: Vec<String> = v.iter().map(|s| format!("{s:?}")).collect();
        // U+0000 after "a" by length; the emoji's high surrogate (U+D83D)
        // before U+FFFF; upper case before lower case.
        assert_eq!(order, ["\"Z\"", "\"a\"", "\"a\\0\"", "\"z\"", "\"😀\"", "\"\\u{ffff}\""]);
        assert_eq!(JavaString::new("\0").cmp(&JavaString::new("a")), Ordering::Less);
        for (a, b) in [("ab", "b"), ("", "a"), ("é", "z"), ("\u{10000}", "\u{FFFF}")] {
            let expected = turbo_jvm::chars::compare(a, b);
            assert_eq!(JavaString::new(a).cmp(&JavaString::new(b)), expected, "{a:?} vs {b:?}");
        }
    }
}
