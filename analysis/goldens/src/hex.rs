//! Lowercase hex, as the golden format writes commits, digests and the
//! bits of an f64.

const DIGITS: &[u8; 16] = b"0123456789abcdef";

/// Bytes as lowercase hex, two digits each.
pub fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 15) as usize] as char);
    }
    s
}

/// Exactly `2 * N` lowercase hex digits as `N` bytes. Uppercase digits are
/// refused: the format writes lowercase only, so a file with uppercase
/// digits was not written by a conforming writer.
pub fn parse_hex<const N: usize>(s: &str) -> Option<[u8; N]> {
    let b = s.as_bytes();
    if b.len() != 2 * N {
        return None;
    }
    let mut out = [0u8; N];
    for (o, pair) in out.iter_mut().zip(b.chunks_exact(2)) {
        *o = (digit(pair[0])? << 4) | digit(pair[1])?;
    }
    Some(out)
}

/// One lowercase hex digit's value.
pub(crate) fn digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let bytes = [0x00, 0x0f, 0xa5, 0xff];
        assert_eq!(to_hex(&bytes), "000fa5ff");
        assert_eq!(parse_hex::<4>("000fa5ff"), Some(bytes));
    }

    #[test]
    fn refuses_uppercase_wrong_length_and_other_characters() {
        assert_eq!(parse_hex::<2>("00FF"), None);
        assert_eq!(parse_hex::<2>("00f"), None);
        assert_eq!(parse_hex::<2>("00fff"), None);
        assert_eq!(parse_hex::<2>("00fg"), None);
        assert_eq!(parse_hex::<2>("+0ff"), None);
    }
}
