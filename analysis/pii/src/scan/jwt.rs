//! JSON Web Tokens in compact form, `SecretsPiiExtractor.scanJwts`: three
//! non-empty unpadded base64url segments split by dots, with valid
//! lengths and zero unused bits, whose first segment decodes to UTF-8
//! JSON with one top-level `alg` that is a non-empty ASCII string.

use turbo_analysis_layers::{PiiConfidence, PiiKind};

use super::jws;
use super::secrets::{on_token_end, on_token_start};
use crate::hits::Hits;

fn is_base64url(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'-' || c == b'_'
}

fn value(c: u8) -> u32 {
    match c {
        b'A'..=b'Z' => u32::from(c - b'A'),
        b'a'..=b'z' => u32::from(c - b'a') + 26,
        b'0'..=b'9' => u32::from(c - b'0') + 52,
        b'-' => 62,
        _ => 63,
    }
}

/// A non-empty run of a valid unpadded length whose unused low bits are
/// zero.
fn valid_encoding(run: &[u8]) -> bool {
    let unused = match run.len() % 4 {
        1 => return false,
        2 => 0x0f,
        3 => 0x03,
        _ => 0,
    };
    !run.is_empty() && value(run[run.len() - 1]) & unused == 0
}

/// Not inside a longer base64url or dotted value.
fn on_jwt_start(text: &str, start: usize) -> bool {
    let b = text.as_bytes();
    let mut p = start;
    while p > 0 && b[p - 1] == b'.' {
        p -= 1;
    }
    on_token_start(text, start, true) && on_token_start(text, p, true)
}

/// No further segment, padding or standard base64 character after `end`.
fn on_jwt_end(text: &str, end: usize) -> bool {
    let b = text.as_bytes();
    if !on_token_end(text, end) || matches!(b.get(end), Some(b'=' | b'+' | b'/')) {
        return false;
    }
    let mut p = end;
    while p < b.len() && b[p] == b'.' {
        p += 1;
    }
    on_token_end(text, p) && b.get(p) != Some(&b'-')
}

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut header = Vec::new();
    let mut i = 0;
    while i < b.len() {
        // `e`, `I`, `C` and `D` start the encoding of `{`, a space, a
        // carriage return and a line feed or tab.
        if !matches!(b[i], b'e' | b'I' | b'C' | b'D') || !on_jwt_start(text, i) {
            i += 1;
            continue;
        }
        let mut p = i;
        let mut header_end = i;
        let mut segments = true;
        for segment in 0..3 {
            if segment > 0 {
                if b.get(p) != Some(&b'.') {
                    segments = false;
                    break;
                }
                p += 1;
            }
            let from = p;
            while p < b.len() && is_base64url(b[p]) {
                p += 1;
            }
            if !valid_encoding(&b[from..p]) {
                segments = false;
                break;
            }
            if segment == 0 {
                header_end = p;
            }
        }
        if segments && on_jwt_end(text, p) && {
            decode(&b[i..header_end], &mut header);
            std::str::from_utf8(&header).is_ok_and(jws::valid_header)
        } {
            hits.add_text(text, i, p, PiiKind::Jwt, PiiConfidence::Format);
            i = p;
        } else {
            i += 1;
        }
    }
}

/// Decodes a checked unpadded base64url run into `out`.
fn decode(run: &[u8], out: &mut Vec<u8>) {
    out.clear();
    let (mut accumulator, mut bits) = (0u32, 0);
    for &c in run {
        accumulator = accumulator << 6 | value(c);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((accumulator >> bits) as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings_and_decoding() {
        assert!(valid_encoding(b"eyI"));
        assert!(!valid_encoding(b"eyJ"));
        assert!(!valid_encoding(b"eyJh1"));
        assert!(!valid_encoding(b""));
        assert!(valid_encoding(b"QQ"));
        assert!(!valid_encoding(b"QR"));
        let mut out = Vec::new();
        decode(b"eyJhbGciOiJIUzI1NiJ9", &mut out);
        assert_eq!(out, br#"{"alg":"HS256"}"#);
    }
}
