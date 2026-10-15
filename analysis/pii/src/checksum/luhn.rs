//! The Luhn check (ISO/IEC 7812, GSMA TS.06), OpenNLP's `Luhn`.

/// Whether the ASCII digits pass the Luhn check. All zeros pass.
pub fn valid(digits: &[u8]) -> bool {
    let mut sum = 0u32;
    let mut doubled = false;
    for &d in digits.iter().rev() {
        let mut v = u32::from(d - b'0');
        if doubled {
            v *= 2;
            if v > 9 {
                v -= 9;
            }
        }
        sum += v;
        doubled = !doubled;
    }
    sum.is_multiple_of(10)
}

#[cfg(test)]
mod tests {
    use super::valid;

    #[test]
    fn known_values() {
        assert!(valid(b"79927398713"));
        assert!(!valid(b"79927398710"));
        assert!(valid(b"4111111111111111"));
        assert!(valid(b"490154203237518"));
        assert!(valid(b"0000"));
        assert!(valid(b""));
        assert!(valid(b"0"));
        assert!(!valid(b"1"));
    }
}
