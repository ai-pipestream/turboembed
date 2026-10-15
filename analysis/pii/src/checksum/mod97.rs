//! The ISO 7064 MOD 97-10 check of ISO 13616, as `CursorPiiExtractor`
//! computes it.

/// The remainder of an IBAN of upper case letters and digits: the first
/// four characters moved to the end, letters read as 10 to 35. A valid
/// IBAN leaves 1. `compact` has at least five characters.
pub fn remainder(compact: &[u8]) -> u32 {
    let n = compact.len();
    let mut r = 0u32;
    for i in 0..n {
        let c = compact[(i + 4) % n];
        r = if c.is_ascii_digit() {
            (r * 10 + u32::from(c - b'0')) % 97
        } else {
            (r * 100 + u32::from(c - b'A') + 10) % 97
        };
    }
    r
}

#[cfg(test)]
mod tests {
    use super::remainder;

    #[test]
    fn documented_test_ibans() {
        assert_eq!(remainder(b"DE89370400440532013000"), 1);
        assert_eq!(remainder(b"GB82WEST12345698765432"), 1);
        assert_ne!(remainder(b"GB82WEST12345698765431"), 1);
        assert_eq!(remainder(b"NL91ABNA0417164300"), 1);
    }
}
