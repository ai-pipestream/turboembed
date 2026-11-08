//! The vehicle identification number check digit of ISO 3779 and
//! 49 CFR 565.15: every character is transliterated to a value, weighted
//! `8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2`, and the sum
//! modulo 11 is the ninth character (`X` for 10).

const WEIGHTS: [u32; 17] = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2];

/// The value of a VIN character: digits as themselves, letters by the
/// 49 CFR 565 table. `I`, `O`, `Q` and anything else have none.
pub const fn value(c: u8) -> Option<u32> {
    Some(match c {
        b'0'..=b'9' => (c - b'0') as u32,
        b'A'..=b'H' => (c - b'A') as u32 + 1,
        b'J'..=b'N' => (c - b'J') as u32 + 1,
        b'P' => 7,
        b'R' => 9,
        b'S'..=b'Z' => (c - b'S') as u32 + 2,
        _ => return None,
    })
}

/// The check character the first, and the tenth to seventeenth,
/// characters of `vin` call for, or `None` when one is no VIN character.
pub fn check_char(vin: &[u8; 17]) -> Option<u8> {
    let mut sum = 0;
    for (&c, w) in vin.iter().zip(WEIGHTS) {
        sum += value(c)? * w;
    }
    Some(match sum % 11 {
        10 => b'X',
        r => b'0' + r as u8,
    })
}

/// Whether the ninth character of `vin` is its check character.
pub fn valid(vin: &[u8; 17]) -> bool {
    check_char(vin) == Some(vin[8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_examples() {
        assert!(valid(b"1M8GDM9AXKP042788"));
        assert!(valid(b"1HGCM82633A004352"));
        assert!(!valid(b"1HGCM82643A004352"));
        assert!(!valid(b"1HGCM8263IA004352"));
        assert_eq!(value(b'I'), None);
        assert_eq!(value(b'O'), None);
        assert_eq!(value(b'Q'), None);
        assert_eq!(value(b'Z'), Some(9));
    }
}
