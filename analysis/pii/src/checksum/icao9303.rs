//! ICAO Doc 9303 (`7-3-1` modulus 10) check digit and German Personalausweis
//! document number validation.

/// Computes the ICAO Doc 9303 Part 3 (`4.9`) check digit (`b'0'..=b'9'`) over
/// `field`, mapping `0..=9` to `0..=9`, `A..=Z` to `10..=35`, and `<` to `0`
/// with repeating weights `[7, 3, 1]`.
pub fn check_digit(field: &[u8]) -> Option<u8> {
    const WEIGHTS: [u32; 3] = [7, 3, 1];
    if field.is_empty() {
        return None;
    }
    let mut sum = 0u32;
    for (i, &b) in field.iter().enumerate() {
        let v = match b {
            b'0'..=b'9' => (b - b'0') as u32,
            b'A'..=b'Z' => (b - b'A' + 10) as u32,
            b'<' => 0,
            _ => return None,
        };
        sum += v * WEIGHTS[i % 3];
    }
    Some(b'0' + (sum % 10) as u8)
}

/// Checks a field whose final byte is an ICAO Doc 9303 check digit.
pub fn check(field_with_check: &[u8]) -> bool {
    let Some((&last, body)) = field_with_check.split_last() else {
        return false;
    };
    last.is_ascii_digit() && check_digit(body) == Some(last)
}

/// Returns whether `b` belongs to the German Personalausweis Restricted
/// Alphanumeric Set (`0123456789CFGHJKLMNPRTVWXY`).
#[inline]
pub const fn is_npa_char(b: u8) -> bool {
    matches!(
        b,
        b'0'..=b'9'
            | b'C'
            | b'F'
            | b'G'
            | b'H'
            | b'J'
            | b'K'
            | b'L'
            | b'M'
            | b'N'
            | b'P'
            | b'R'
            | b'T'
            | b'V'
            | b'W'
            | b'X'
            | b'Y'
    )
}

/// Checks a 10-character German Personalausweis document number (Ausweisnummer):
/// leading authority-code letter in `L, M, N, P, R, T, V, W, X, Y`, body in
/// the restricted alphanumeric set with at least one non-zero digit, and a
/// valid ICAO 9303 `7-3-1` check digit in the 10th position.
pub fn de_personalausweis(chars: &[u8; 10]) -> bool {
    if !matches!(chars[0], b'L' | b'M' | b'N' | b'P' | b'R' | b'T' | b'V' | b'W' | b'X' | b'Y') {
        return false;
    }
    if !chars[..9].iter().all(|&b| is_npa_char(b)) {
        return false;
    }
    if !chars[1..9].iter().any(|&b| b.is_ascii_digit() && b != b'0') {
        return false;
    }
    check(chars)
}
