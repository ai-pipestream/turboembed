//! A Java string that may hold a lone surrogate.
//!
//! OpenNLP's context generators cut text at UTF-16 code units: the
//! tokenizer's features name the character before and after a candidate
//! split as one `char`, which for a character outside the BMP is half a
//! surrogate pair. Such a predicate is a Java string no Rust `String` can
//! hold. [`JavaString`] holds any Java string as Java's modified UTF-8,
//! the encoding model files store predicates in and the one `turbo-maxent`
//! keys a model's predicates by, so a label travels from events to the
//! index to a written model without re-encoding. Two Java strings are
//! equal when their bytes are, and a `JavaString` orders as
//! `String.compareTo` does (by UTF-16 code units).

use std::cmp::Ordering;
use std::fmt;

use turbo_maxent::java_data::{canonical_modified_utf8, check_modified_utf8};
use turbo_unicode::modified_utf8::{push_modified_utf8, push_str_modified_utf8};

/// A Java string, in modified UTF-8 as `DataOutputStream.writeUTF` writes
/// it (U+0000 as `C0 80`, every UTF-16 unit above U+07FF as three bytes,
/// so a character outside the BMP is its two surrogates).
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct JavaString {
    bytes: Vec<u8>,
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
        Some(JavaString { bytes: canonical_modified_utf8(bytes).into_owned() })
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

    /// `String.length()`.
    pub fn utf16_len(&self) -> usize {
        self.utf16_units().count()
    }

    /// The string as `&str` when it holds no lone surrogate; modified
    /// UTF-8 is then UTF-8, except for U+0000, which is re-encoded.
    pub fn to_str(&self) -> Option<std::borrow::Cow<'_, str>> {
        if let Ok(s) = std::str::from_utf8(&self.bytes) {
            return Some(std::borrow::Cow::Borrowed(s));
        }
        let mut s = String::with_capacity(self.bytes.len());
        for c in char::decode_utf16(self.utf16_units()) {
            s.push(c.ok()?);
        }
        Some(std::borrow::Cow::Owned(s))
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

impl PartialEq<str> for JavaString {
    fn eq(&self, other: &str) -> bool {
        *self == JavaString::new(other)
    }
}

impl PartialEq<&str> for JavaString {
    fn eq(&self, other: &&str) -> bool {
        *self == JavaString::new(other)
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
        assert!(JavaString::from_modified_utf8(b"\xFF").is_none());
    }

    #[test]
    fn converts_back_to_text() {
        let s = JavaString::new("a\0😀");
        assert_eq!(s.to_str().unwrap(), "a\0😀");
        assert_eq!(JavaString::new("plain").to_str().unwrap(), "plain");
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
