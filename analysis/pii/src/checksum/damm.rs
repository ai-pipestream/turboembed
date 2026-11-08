//! The Damm check digit, over a totally anti-symmetric quasigroup of
//! order 10: it catches every single-digit error and every adjacent
//! transposition, and the check digit of a valid number is 0.

const T: [[u8; 10]; 10] = [
    [0, 3, 1, 7, 5, 9, 8, 6, 4, 2],
    [7, 0, 9, 2, 1, 5, 4, 8, 6, 3],
    [4, 2, 0, 6, 8, 7, 1, 3, 5, 9],
    [1, 7, 5, 0, 9, 8, 3, 4, 2, 6],
    [6, 1, 2, 3, 0, 4, 5, 9, 7, 8],
    [3, 6, 7, 4, 2, 0, 9, 5, 8, 1],
    [5, 8, 6, 9, 7, 2, 0, 1, 3, 4],
    [8, 9, 4, 5, 3, 6, 2, 0, 1, 7],
    [9, 4, 3, 8, 6, 1, 7, 2, 0, 5],
    [2, 5, 8, 1, 4, 3, 6, 7, 9, 0],
];

/// The interim digit of the ASCII `digits`: their check digit, or 0 when
/// they end with a valid one.
pub fn check_digit(digits: &[u8]) -> Option<u8> {
    let mut c = 0u8;
    for &d in digits {
        if !d.is_ascii_digit() {
            return None;
        }
        c = T[c as usize][(d - b'0') as usize];
    }
    Some(c)
}

/// Whether the ASCII `digits` end with their Damm check digit. `false`
/// for an empty input or a byte that is no digit.
pub fn valid(digits: &[u8]) -> bool {
    !digits.is_empty() && check_digit(digits) == Some(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_example() {
        assert_eq!(check_digit(b"572"), Some(4));
        assert!(valid(b"5724"));
        assert!(!valid(b"5742"));
        assert!(!valid(b"5725"));
        assert!(!valid(b""));
    }
}
