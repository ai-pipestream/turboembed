//! Helpers shared by the integration tests.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// The fixtures directory.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Undoes the `\uXXXX` escaping the Java fixture generators apply: each
/// escape is one UTF-16 code unit, and a pair of them can form one
/// character.
pub fn unescape(s: &str) -> String {
    let mut units: Vec<u16> = Vec::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("\\u") {
        units.extend(rest[..i].encode_utf16());
        units.push(u16::from_str_radix(&rest[i + 2..i + 6], 16).expect("four hex digits"));
        rest = &rest[i + 6..];
    }
    units.extend(rest.encode_utf16());
    String::from_utf16(&units).expect("the fixtures hold no lone surrogates")
}

/// A Java string's modified UTF-8, as model predicates are keyed.
pub fn modified_utf8(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    turbo_unicode::modified_utf8::push_str_modified_utf8(&mut out, s);
    out
}

/// Parses probabilities printed with Java's `Double.toString`. Rust's
/// parser reads the shortest round-trip form exactly.
pub fn doubles(s: &str) -> Vec<f64> {
    s.split(' ').map(|d| d.parse::<f64>().expect("a double")).collect()
}
