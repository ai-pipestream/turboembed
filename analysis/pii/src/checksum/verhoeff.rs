//! The Verhoeff check digit, over the dihedral group D5: it catches every
//! single-digit error and every adjacent transposition. Used by India's
//! Aadhaar number, among others.

const D: [[u8; 10]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];

const P: [[u8; 10]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 8, 7, 0, 6],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

const INV: [u8; 10] = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9];

fn fold(digits: &[u8], shift: usize) -> Option<u8> {
    let mut c = 0u8;
    for (i, &d) in digits.iter().rev().enumerate() {
        if !d.is_ascii_digit() {
            return None;
        }
        c = D[c as usize][P[(i + shift) % 8][(d - b'0') as usize] as usize];
    }
    Some(c)
}

/// Whether the last digit of the ASCII `digits` is their Verhoeff check
/// digit. `false` for an empty input or a byte that is no digit.
pub fn valid(digits: &[u8]) -> bool {
    !digits.is_empty() && fold(digits, 0) == Some(0)
}

/// The Verhoeff check digit to append to the ASCII `digits`.
pub fn check_digit(digits: &[u8]) -> Option<u8> {
    fold(digits, 1).map(|c| INV[c as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_example() {
        assert_eq!(check_digit(b"236"), Some(3));
        assert!(valid(b"2363"));
        assert!(!valid(b"2364"));
        assert!(!valid(b"3263"));
        assert!(!valid(b""));
    }
}
