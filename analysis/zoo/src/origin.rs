//! Validated URLs, relative artifact paths, and origin containment checks.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An `https://` address of printable ASCII with no spaces and no
/// backslash, whose host is not empty: `https:///x` and
/// `https://user@/x` are refused.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Url(String);

impl Url {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for Url {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let Some(rest) = s.strip_prefix("https://") else {
            return Err(format!("{s:?} is not an https:// address"));
        };
        if !s.bytes().all(|c| c.is_ascii_graphic() && c != b'\\') {
            return Err(format!("{s:?} is not an https:// address of printable ASCII without a backslash"));
        }
        let authority = &rest[..rest.find(['/', '?', '#']).unwrap_or(rest.len())];
        let host = authority.rsplit_once('@').map_or(authority, |(_, host)| host);
        if host.is_empty() || host.starts_with(':') {
            return Err(format!("{s:?} is not an https:// address: it names no host"));
        }
        Ok(Url(s.to_owned()))
    }
}

impl fmt::Display for Url {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A file's path inside its artifact's directory: relative, `/`
/// separated, no empty, `.` or `..` component, no backslash and no
/// control character.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelPath(String);

impl RelPath {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for RelPath {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let ok = !s.is_empty()
            && !s.starts_with('/')
            && !s.chars().any(|c| c == '\\' || c.is_control())
            && s.split('/').all(|p| !p.is_empty() && p != "." && p != "..");
        if ok { Ok(RelPath(s.to_owned())) } else { Err(format!("{s:?} is not a relative path inside the directory")) }
    }
}

impl fmt::Display for RelPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for Url {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Url {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
    }
}

impl Serialize for RelPath {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for RelPath {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
    }
}

/// An `https://` address split after its host: the host and the rest,
/// from the first `/`, `?` or `#` on (empty for a bare host).
/// The scheme is always `https`, which [`Url`] checks.
fn split_host(url: &Url) -> (&str, &str) {
    let rest = &url.as_str()["https://".len()..];
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    (&rest[..end], &rest[end..])
}

/// Strips one `.` or `%2e` (case-insensitive) from the front of `s`.
fn strip_dot(s: &[u8]) -> Option<&[u8]> {
    if let Some(rest) = s.strip_prefix(b".") {
        Some(rest)
    } else if s.len() >= 3 && s[..3].eq_ignore_ascii_case(b"%2e") {
        Some(&s[3..])
    } else {
        None
    }
}

/// The byte a `%` escape's two hex digits give, in either case; `None`
/// unless both are hex digits.
fn hex_byte(digits: &[u8]) -> Option<u8> {
    let [high, low] = digits else { return None };
    let value = |c: u8| (c as char).to_digit(16);
    Some((value(*high)? * 16 + value(*low)?) as u8)
}

/// Whether `path` is free of the bytes and escapes this check knows a
/// server may turn into a step out of a directory, and decodes to valid
/// UTF-8. Scans once, without allocating:
///
/// - every `%` starts an escape of two hex digits, so a malformed escape
///   a lenient decoder might repair into another escape (`%%32%65` or
///   `%2%65` into `%2e`) and IIS's `%u002e` form are refused;
/// - no `\` or `;` is in the path, plain or escaped (`%5c`, `%3b`): some
///   servers read `\` as `/` and strip a `;` with what follows it from a
///   segment (`..;` as `..`);
/// - no escape gives `/`, `%` (`%252e` decodes twice to `.`) or NUL;
/// - the decoded bytes are valid UTF-8 as RFC 3629 defines it, so there
///   is no overlong form (`%c0%ae` for `.`), no surrogate (`%ed%a0%80`),
///   no five- or six-byte form, no byte from `%f5` up, no lone or
///   missing continuation byte, and nothing past U+10FFFF. A lax decoder
///   reads an overlong form as the ASCII character it encodes; the other
///   forms are not UTF-8 at all.
///
/// A `.` or `..` segment is [`plain_path`]'s to refuse.
fn decodes_plainly(path: &[u8]) -> bool {
    // Continuation bytes the character being decoded still needs, and the
    // range the next one must fall in.
    let mut needed = 0u8;
    let (mut low, mut high) = (0x80u8, 0xBFu8);
    let mut i = 0;
    while i < path.len() {
        let byte = match path[i] {
            b'\\' | b';' => return false,
            b'%' => match path.get(i + 1..i + 3).and_then(hex_byte) {
                Some(b'/' | b'\\' | b'%' | b';' | 0) | None => return false,
                Some(decoded) => {
                    i += 3;
                    decoded
                }
            },
            plain => {
                i += 1;
                plain
            }
        };
        if needed > 0 {
            if !(low..=high).contains(&byte) {
                return false;
            }
            needed -= 1;
            (low, high) = (0x80, 0xBF);
            continue;
        }
        (needed, low, high) = match byte {
            0x00..=0x7F => (0, 0x80, 0xBF),
            0xC2..=0xDF => (1, 0x80, 0xBF),
            0xE0 => (2, 0xA0, 0xBF),
            0xE1..=0xEC | 0xEE..=0xEF => (2, 0x80, 0xBF),
            0xED => (2, 0x80, 0x9F),
            0xF0 => (3, 0x90, 0xBF),
            0xF1..=0xF3 => (3, 0x80, 0xBF),
            0xF4 => (3, 0x80, 0x8F),
            // A continuation byte with no lead, `%c0` and `%c1` (always
            // overlong), and `%f5` to `%ff`.
            _ => return false,
        };
    }
    needed == 0
}

/// Whether the path of `rest` (what [`split_host`] leaves, up to a `?` or
/// `#`) is free of the forms this check knows a server may read as a step
/// out of the directory the path names: a `.` or `..` segment, plain or
/// with its dots percent-encoded, and whatever [`decodes_plainly`]
/// refuses. A query or fragment is not checked.
pub(super) fn plain_path(rest: &str) -> bool {
    let path = &rest[..rest.find(['?', '#']).unwrap_or(rest.len())];
    decodes_plainly(path.as_bytes())
        && path.split('/').all(|segment| match strip_dot(segment.as_bytes()) {
            None => true,
            Some(b"") => false,
            Some(after_first) => strip_dot(after_first) != Some(b""),
        })
}

/// Whether `origin` names a location a file can lie under: no user
/// information before its host (`https://a.org@b.net/` is on `b.net`), no
/// query or fragment, a plain path ([`plain_path`]), and at least one
/// non-empty path segment, so not a whole host (`https://example.org` or
/// `https://example.org/`).
pub(super) fn origin_is_a_location(origin: &Url) -> bool {
    let (host, rest) = split_host(origin);
    !host.contains('@')
        && !rest.contains(['?', '#'])
        && plain_path(rest)
        && rest.split('/').any(|segment| !segment.is_empty())
}

/// Whether `file` lies under `origin` ([`crate::index::Artifact::origin`]):
/// the same host, compared without regard to case, a plain path
/// ([`plain_path`]), and a path that is the origin's, without any trailing
/// `/`, or continues it after a `/` with at least one non-empty segment.
/// Paths are compared byte for byte.
pub(super) fn under_origin(origin: &Url, file: &Url) -> bool {
    let (origin_host, origin_rest) = split_host(origin);
    let (file_host, file_rest) = split_host(file);
    let prefix = origin_rest.trim_end_matches('/');
    origin_host.eq_ignore_ascii_case(file_host)
        && plain_path(file_rest)
        && file_rest.strip_prefix(prefix).is_some_and(|after| {
            after.is_empty() || {
                let after_path = &after[..after.find(['?', '#']).unwrap_or(after.len())];
                after_path.starts_with('/') && !after_path.trim_matches('/').is_empty()
            }
        })
}

#[cfg(test)]
mod tests {
    use super::plain_path;

    /// The percent-encoding of every byte of `bytes`, between `/a/x` and
    /// `y`, so no segment is a dot segment.
    fn encoded<'a>(bytes: &[u8], into: &'a mut String) -> &'a str {
        into.clear();
        into.push_str("/a/x");
        for b in bytes {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            into.extend(['%', HEX[usize::from(b >> 4)] as char, HEX[usize::from(b & 15)] as char]);
        }
        into.push('y');
        into
    }

    /// What the path check should say of these decoded bytes: valid UTF-8
    /// holding no `/`, `\`, `%`, `;` or NUL.
    fn expected(bytes: &[u8]) -> bool {
        std::str::from_utf8(bytes).is_ok() && !bytes.iter().any(|b| matches!(b, b'/' | b'\\' | b'%' | b';' | 0))
    }

    /// Every scalar value's UTF-8, percent-encoded, is accepted, except
    /// the five characters a path may not hold escaped.
    #[test]
    fn every_scalar_value_is_accepted_but_five() {
        let mut path = String::new();
        let mut utf8 = [0u8; 4];
        for c in (0..=0x10FFFF).filter_map(char::from_u32) {
            let bytes = c.encode_utf8(&mut utf8).as_bytes();
            let refused = matches!(c, '/' | '\\' | '%' | ';' | '\0');
            assert_eq!(plain_path(encoded(bytes, &mut path)), !refused, "U+{:04X}", c as u32);
        }
    }

    /// The check agrees with the standard library's UTF-8 validation on
    /// every sequence of one or two bytes, every three-byte sequence with
    /// a three-byte lead, and four-byte sequences over every lead and
    /// second byte with a few third and fourth bytes at the edges of the
    /// continuation range.
    #[test]
    fn decoded_bytes_are_checked_as_strictly_as_utf8() {
        let mut path = String::new();
        let mut check = |bytes: &[u8]| {
            assert_eq!(plain_path(encoded(bytes, &mut path)), expected(bytes), "{bytes:02x?}");
        };
        for a in 0..=255u8 {
            check(&[a]);
            for b in 0..=255u8 {
                check(&[a, b]);
                for c in [0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0] {
                    check(&[a, b, c, 0x80]);
                    check(&[a, b, 0x80, c]);
                }
            }
        }
        for a in 0xE0..=0xEFu8 {
            for b in 0..=255u8 {
                for c in 0..=255u8 {
                    check(&[a, b, c]);
                }
            }
        }
    }

    /// A character cut off by the end of the path, or by its query, is
    /// refused.
    #[test]
    fn a_character_cut_off_at_the_end_is_refused() {
        for path in ["/a/x%c3", "/a/x%e0%a0", "/a/x%f0%90%80", "/a/x%c3?%a9", "/a/x%c3#%a9"] {
            assert!(!plain_path(path), "{path}");
        }
        assert!(plain_path("/a/x%c3%a9?%c3"));
    }

    /// Escapes must be well formed: a `%` with fewer than two hex digits
    /// after it is refused, whatever a lenient decoder would make of it.
    #[test]
    fn malformed_escapes_are_refused() {
        for path in ["/a/%", "/a/%2", "/a/%zz", "/a/%2g", "/a/%%32%65", "/a/%2%65", "/a/%%35%63", "/a/%u002e", "/a/x%"]
        {
            assert!(!plain_path(path), "{path}");
        }
        for path in ["/a/%2E%2e.x", "/a/%41%7e", "/a/%3F", "/a/%23"] {
            assert!(plain_path(path), "{path}");
        }
    }
}
