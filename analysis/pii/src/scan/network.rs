//! Network addresses, `NetworkPiiExtractor`: dotted decimal IPv4, the
//! RFC 4291 text form of IPv6 with an optional IPv4 tail, and 48-bit MAC
//! addresses as six pairs split by `:` or `-` or three quads split by
//! `.`. The MAC scan runs first, then IPv6, then IPv4, as in the Java.

use turbo_analysis_layers::{PiiConfidence, PiiKind, PiiKinds};

use crate::hits::Hits;
use crate::text::{before, hex_value, letter_or_digit, on_end_before, on_number_start, on_word_start};

pub(crate) fn scan(text: &str, kinds: PiiKinds, hits: &mut Hits) {
    if kinds.contains(PiiKind::Mac) {
        scan_macs(text, hits);
    }
    if kinds.contains(PiiKind::Ipv6) {
        scan_ipv6(text, hits);
    }
    if kinds.contains(PiiKind::Ipv4) {
        scan_ipv4(text, hits);
    }
}

/// Whether dots before `start` join it to a letter or digit before them.
fn continues_dotted_value(text: &str, start: usize) -> bool {
    let b = text.as_bytes();
    let mut p = start;
    while p > 0 && b[p - 1] == b'.' {
        p -= 1;
    }
    p < start && before(text, p).is_some_and(letter_or_digit)
}

/// No word or dotted value goes on into `start`, and no `:` or `-` after
/// a hexadecimal digit makes it the middle of a grouped value.
fn on_group_start(text: &str, start: usize) -> bool {
    if !on_word_start(text, start) || continues_dotted_value(text, start) {
        return false;
    }
    let b = text.as_bytes();
    start == 0 || !matches!(b[start - 1], b':' | b'-') || start < 2 || !b[start - 2].is_ascii_hexdigit()
}

fn scan_ipv4(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut octets = [0u32; 4];
    let mut i = 0;
    while i < b.len() {
        let end = if b[i].is_ascii_digit() && on_number_start(text, i) && !continues_dotted_value(text, i) {
            parse_ipv4(b, i, &mut octets).filter(|&end| on_end_before(text, end, b'.') && !reserved_ipv4(&octets))
        } else {
            None
        };
        match end {
            Some(end) => {
                let o = octets;
                hits.add(i, end, PiiKind::Ipv4, PiiConfidence::Format, |s| {
                    use std::fmt::Write;
                    let _ = write!(s, "{}.{}.{}.{}", o[0], o[1], o[2], o[3]);
                });
                i = end;
            }
            None => i += 1,
        }
    }
}

fn reserved_ipv4(octets: &[u32; 4]) -> bool {
    octets.iter().all(|&o| o == 0) || octets.iter().all(|&o| o == 255)
}

/// Four decimal octets of 0 to 255 split by dots, no leading zero, no
/// fourth digit; the end, or `None`.
fn parse_ipv4(b: &[u8], start: usize, octets: &mut [u32; 4]) -> Option<usize> {
    let mut p = start;
    for (o, octet) in octets.iter_mut().enumerate() {
        if o > 0 {
            if b.get(p) != Some(&b'.') {
                return None;
            }
            p += 1;
        }
        if !b.get(p).is_some_and(u8::is_ascii_digit) {
            return None;
        }
        let leading_zero = b[p] == b'0';
        let mut digits = 0;
        let mut value = 0;
        while p < b.len() && b[p].is_ascii_digit() && digits < 3 {
            value = value * 10 + u32::from(b[p] - b'0');
            digits += 1;
            p += 1;
        }
        if (digits > 1 && leading_zero) || value > 255 || b.get(p).is_some_and(u8::is_ascii_digit) {
            return None;
        }
        *octet = value;
    }
    Some(p)
}

/// An IPv6 candidate: its end, its eight groups, the groups written out
/// (not from an IPv4 tail), the digits of the longest of those, and
/// whether an IPv4 tail gave the last 32 bits.
struct Ipv6 {
    end: usize,
    groups: [u16; 8],
    written: usize,
    longest: usize,
    embedded: bool,
}

fn scan_ipv6(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let double_colon = c == b':' && b.get(i + 1) == Some(&b':');
        let found = if (c.is_ascii_hexdigit() || double_colon) && on_group_start(text, i) {
            parse_ipv6(b, i).filter(|v| on_end_before(text, v.end, b'.') && plausible_ipv6(v))
        } else {
            None
        };
        match found {
            Some(v) => {
                hits.add(i, v.end, PiiKind::Ipv6, PiiConfidence::Format, |s| format_ipv6(&v.groups, s));
                i = v.end;
            }
            None => i += 1,
        }
    }
}

fn parse_ipv6(b: &[u8], start: usize) -> Option<Ipv6> {
    // The groups before a `::` and after it.
    let (mut head, mut tail) = ([0u16; 8], [0u16; 8]);
    let (mut heads, mut tails) = (0usize, 0usize);
    let mut octets = [0u32; 4];
    let mut compressed = false;
    let mut embedded = false;
    let mut written = 0;
    let mut longest = 0;
    let mut p = start;
    if b[p] == b':' {
        compressed = true;
        p += 2;
    }
    while p < b.len() {
        if heads + tails >= 8 {
            return None;
        }
        if (heads > 0 || compressed)
            && b[p].is_ascii_digit()
            && let Some(quad_end) = parse_ipv4(b, p, &mut octets)
        {
            let pair = [(octets[0] << 8 | octets[1]) as u16, (octets[2] << 8 | octets[3]) as u16];
            for g in pair {
                if compressed {
                    // An IPv4 tail can make nine groups; the count check
                    // below refuses that, so the tenth slot is never asked.
                    if tails < 8 {
                        tail[tails] = g;
                    }
                    tails += 1;
                } else {
                    if heads < 8 {
                        head[heads] = g;
                    }
                    heads += 1;
                }
            }
            embedded = true;
            p = quad_end;
            break;
        }
        let mut digits = 0;
        let mut value = 0u32;
        while p < b.len() && b[p].is_ascii_hexdigit() && digits < 4 {
            value = value * 16 + hex_value(b[p]);
            digits += 1;
            p += 1;
        }
        // A trailing `::` may end before punctuation or whitespace.
        if digits == 0 && compressed && tails == 0 && b[p] != b':' {
            break;
        }
        if digits == 0 || b.get(p).is_some_and(u8::is_ascii_hexdigit) {
            return None;
        }
        if compressed {
            tail[tails] = value as u16;
            tails += 1;
        } else {
            head[heads] = value as u16;
            heads += 1;
        }
        written += 1;
        longest = longest.max(digits);
        if b.get(p) != Some(&b':') {
            break;
        }
        if b.get(p + 1) == Some(&b':') {
            if compressed {
                return None;
            }
            compressed = true;
            p += 2;
            continue;
        }
        if b.get(p + 1).is_some_and(u8::is_ascii_hexdigit) {
            p += 1;
            continue;
        }
        break;
    }
    let total = heads + tails;
    if if compressed { total >= 8 } else { total != 8 } {
        return None;
    }
    // An IPv4 tail gives the last 32 bits.
    if embedded && b.get(p) == Some(&b':') {
        return None;
    }
    let mut groups = [0u16; 8];
    groups[..heads].copy_from_slice(&head[..heads]);
    groups[8 - tails..].copy_from_slice(&tail[..tails]);
    Some(Ipv6 { end: p, groups, written, longest, embedded })
}

/// Not the unspecified address; without an IPv4 tail, two groups or more
/// written, with a group of three digits or more or all eight written.
/// The filter keeps times and namespaces out, and short forms such as
/// `::1` with them.
fn plausible_ipv6(v: &Ipv6) -> bool {
    if v.groups.iter().all(|&g| g == 0) {
        return false;
    }
    v.embedded || (v.written >= 2 && (v.longest >= 3 || v.written == 8))
}

/// RFC 5952: lower case, no leading zeros, the first longest run of two
/// zero groups or more as `::`; an IPv4 tail written as two groups.
fn format_ipv6(groups: &[u16; 8], out: &mut String) {
    use std::fmt::Write;
    let (mut run_start, mut run_length) = (usize::MAX, 0);
    let mut i = 0;
    while i < 8 {
        if groups[i] != 0 {
            i += 1;
            continue;
        }
        let j = (i..8).find(|&j| groups[j] != 0).unwrap_or(8);
        if j - i > run_length {
            run_length = j - i;
            run_start = i;
        }
        i = j;
    }
    if run_length < 2 {
        run_start = usize::MAX;
    }
    let from = out.len();
    let mut i = 0;
    while i < 8 {
        if i == run_start {
            out.push_str("::");
            i += run_length;
            continue;
        }
        if out.len() > from && !out.ends_with(':') {
            out.push(':');
        }
        let _ = write!(out, "{:x}", groups[i]);
        i += 1;
    }
}

fn scan_macs(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut bytes = [0u8; 6];
    let mut i = 0;
    while i < b.len() {
        let found = if b[i].is_ascii_hexdigit() && on_group_start(text, i) {
            [(b':', 2, 6), (b'-', 2, 6), (b'.', 4, 3)]
                .into_iter()
                .find_map(|(separator, digits, groups)| {
                    parse_mac(b, i, separator, digits, groups, &mut bytes).map(|end| (end, separator))
                })
                .filter(|&(end, separator)| on_end_before(text, end, separator) && !reserved_mac(&bytes))
        } else {
            None
        };
        match found {
            Some((end, _)) => {
                hits.add(i, end, PiiKind::Mac, PiiConfidence::Format, |s| {
                    use std::fmt::Write;
                    for (k, byte) in bytes.iter().enumerate() {
                        if k > 0 {
                            s.push(':');
                        }
                        let _ = write!(s, "{byte:02x}");
                    }
                });
                i = end;
            }
            None => i += 1,
        }
    }
}

/// `groups` groups of `digits` hexadecimal digits split by `separator`,
/// no digit after a group; the end, or `None`.
fn parse_mac(
    b: &[u8],
    start: usize,
    separator: u8,
    digits: usize,
    groups: usize,
    bytes: &mut [u8; 6],
) -> Option<usize> {
    let mut p = start;
    let mut n = 0;
    for g in 0..groups {
        if g > 0 {
            if b.get(p) != Some(&separator) {
                return None;
            }
            p += 1;
        }
        for _ in 0..digits {
            let c = *b.get(p).filter(|c| c.is_ascii_hexdigit())?;
            let v = hex_value(c) as u8;
            if n % 2 == 0 {
                bytes[n / 2] = v << 4;
            } else {
                bytes[n / 2] |= v;
            }
            n += 1;
            p += 1;
        }
        if b.get(p).is_some_and(u8::is_ascii_hexdigit) {
            return None;
        }
    }
    Some(p)
}

fn reserved_mac(bytes: &[u8; 6]) -> bool {
    bytes.iter().all(|&v| v == 0) || bytes.iter().all(|&v| v == 0xff)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn formatted(groups: [u16; 8]) -> String {
        let mut s = String::new();
        format_ipv6(&groups, &mut s);
        s
    }

    #[test]
    fn rfc_5952_forms() {
        assert_eq!(formatted([0x2001, 0xdb8, 0, 0, 0, 0, 0, 1]), "2001:db8::1");
        assert_eq!(formatted([0x2001, 0xdb8, 0, 1, 0, 0, 0, 1]), "2001:db8:0:1::1");
        assert_eq!(formatted([0x2001, 0, 0, 1, 0, 0, 1, 1]), "2001::1:0:0:1:1");
        assert_eq!(formatted([0x2001, 0xdb8, 0, 1, 1, 1, 1, 1]), "2001:db8:0:1:1:1:1:1");
        assert_eq!(formatted([0; 8]), "::");
        assert_eq!(formatted([0, 0, 0, 0, 0, 0, 0, 1]), "::1");
        assert_eq!(formatted([1, 0, 0, 0, 0, 0, 0, 0]), "1::");
    }
}
