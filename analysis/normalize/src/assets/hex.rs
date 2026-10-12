//! Hex dumps: runs of hexadecimal byte pairs, an addition to OpenNLP's
//! detector (which reads only base64). A dump is reported only when its
//! decoded bytes start with a recognized signature, like a bare base64
//! run, so prose with hex digits in it (`deadbeef`, a hash, a UUID) is
//! never one.
//!
//! The forms read are those programs print and people paste:
//!
//! - digit pairs back to back: `89504e470d0a1a0a...`;
//! - pairs separated by one space, colon, comma, or comma and space:
//!   `89 50 4e 47`, `89:50:4e:47`, `0x89, 0x50, 0x4e`;
//! - each pair prefixed with `0x` or `0X` (all pairs or none);
//! - any of these continued on the next line: a CRLF or LF, optionally
//!   after the separator, then optional indentation of spaces and tabs.
//!
//! The first gap between two pairs fixes the separator, and every later
//! gap on the same line uses it. The run must stand alone: no ASCII letter,
//! digit or `_` right before it or right after its last pair.

use turbo_normalize_layers::AssetEncoding;

use super::payload::{Payload, hex_digit, line_break_end};

/// The fewest bytes a dump must hold, the bytes of the shortest bare base64
/// run OpenNLP considers (32 characters, 24 bytes) rounded up to the
/// 32-byte line `xxd -p` and `od` print.
pub(crate) const MIN_HEX_BYTES: usize = 32;

/// The separators a dump may use between pairs on one line.
const SEPARATORS: [&[u8]; 5] = [b", ", b",", b" ", b":", b""];

/// The hex dump that starts at `start`, as a payload whose `encoded_len`
/// is its number of hex digits.
pub(crate) fn scan(text: &[u8], start: usize) -> Option<Payload> {
    if start > 0 && is_word(text[start - 1]) {
        return None;
    }
    let prefixed = matches!(text.get(start..start + 2), Some(b"0x" | b"0X"));
    let mut end = pair_end(text, start, prefixed)?;
    let mut pairs = 1;
    let mut separator: Option<&[u8]> = None;
    loop {
        let candidates: &[&[u8]] = match separator {
            Some(s) => &[s][..],
            None if prefixed => &SEPARATORS[..4],
            None => &SEPARATORS[..],
        };
        let same_line = candidates.iter().find_map(|&s| {
            let next = pair_end(text, end + s.len(), prefixed).filter(|_| text[end..].starts_with(s))?;
            Some((s, next))
        });
        if let Some((s, next)) = same_line {
            separator = Some(s);
            end = next;
            pairs += 1;
            continue;
        }
        match next_line_pair(text, end, separator.unwrap_or(b""), prefixed) {
            Some(next) => {
                end = next;
                pairs += 1;
            }
            None => break,
        }
    }
    if pairs < MIN_HEX_BYTES || text.get(end).is_some_and(|&b| is_word(b)) {
        return None;
    }
    Some(Payload { start, end, encoding: AssetEncoding::Hex, encoded_len: pairs * 2, padding: 0, valid: true })
}

/// The end of the pair (with its `0x` when `prefixed`) at `at`.
fn pair_end(text: &[u8], at: usize, prefixed: bool) -> Option<usize> {
    let digits = if prefixed {
        let rest = text.get(at..)?;
        if !(rest.starts_with(b"0x") || rest.starts_with(b"0X")) {
            return None;
        }
        at + 2
    } else {
        at
    };
    let pair = text.get(digits..digits + 2)?;
    pair.iter().all(|&b| hex_digit(b).is_some()).then_some(digits + 2)
}

/// The end of the first pair of the next line, after the separator or
/// its form without trailing white space (a trailing `,` stays on its
/// line), a line break and indentation.
fn next_line_pair(text: &[u8], end: usize, separator: &[u8], prefixed: bool) -> Option<usize> {
    let trimmed = separator.trim_ascii_end();
    let mut at = [separator, trimmed].iter().find(|s| text[end..].starts_with(s)).map_or(end, |s| end + s.len());
    let after_break = line_break_end(text, at);
    if after_break == at {
        return None;
    }
    at = after_break;
    while matches!(text.get(at), Some(b' ' | b'\t')) {
        at += 1;
    }
    pair_end(text, at, prefixed)
}

/// An ASCII letter, digit or `_`, which may not touch a dump.
fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
