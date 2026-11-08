//! The check digits of parcel tracking numbers: the UPS `1Z` modulus 10,
//! the FedEx Express weights 3, 1, 7 modulus 11, and the GS1 modulus 10
//! with weights 3 and 1 (USPS and FedEx Ground).

/// The UPS check digit of the 15 characters after `1Z`: letters count
/// `(c - 63) % 10`, characters at odd places (from 0) are doubled, and
/// the digit makes the sum a multiple of 10. `None` for a character that
/// is neither an uppercase letter nor a digit.
pub fn ups(serial: &[u8]) -> Option<u8> {
    let mut sum = 0u32;
    for (i, &c) in serial.iter().enumerate() {
        let v = match c {
            b'0'..=b'9' => u32::from(c - b'0'),
            b'A'..=b'Z' => (u32::from(c) - 63) % 10,
            _ => return None,
        };
        sum += if i % 2 == 1 { v * 2 } else { v };
    }
    Some(((10 - sum % 10) % 10) as u8)
}

/// The FedEx Express check digit of 11 digits: weights 3, 1, 7 from the
/// left, the sum modulo 11, then modulo 10.
pub fn fedex_mod11(digits: &[u8]) -> Option<u8> {
    let mut sum = 0u32;
    for (i, &c) in digits.iter().enumerate() {
        if !c.is_ascii_digit() {
            return None;
        }
        sum += u32::from(c - b'0') * [3, 1, 7][i % 3];
    }
    Some((sum % 11 % 10) as u8)
}

/// The GS1 check digit of `digits`: weight 3 on the rightmost digit, then
/// 1 and 3 alternating, and the digit that makes the sum a multiple of 10.
pub fn gs1_mod10(digits: &[u8]) -> Option<u8> {
    let mut sum = 0u32;
    for (i, &c) in digits.iter().rev().enumerate() {
        if !c.is_ascii_digit() {
            return None;
        }
        sum += u32::from(c - b'0') * if i % 2 == 0 { 3 } else { 1 };
    }
    Some(((10 - sum % 10) % 10) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        // The UPS sample number, then made up numbers with valid checks.
        assert_eq!(ups(b"999AA1012345678"), Some(4));
        assert_eq!(ups(b"99-AA1012345678"), None);
        assert_eq!(fedex_mod11(b"84165465368"), Some(2));
        assert_eq!(fedex_mod11(b"84165465368x"), None);
        assert_eq!(gs1_mod10(b"12151330242714"), Some(4));
        assert_eq!(gs1_mod10(b"940022291175169408035"), Some(0));
        assert_eq!(gs1_mod10(b"0307496115017593819"), Some(3));
    }
}
