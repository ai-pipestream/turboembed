//! Small ASCII helpers the normalizers share.

/// `c` is an ASCII letter or digit.
#[inline]
pub fn is_ascii_alnum(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

/// `c` with A to Z mapped to a to z; every other character unchanged.
#[inline]
pub fn ascii_lower(c: char) -> char {
    c.to_ascii_lowercase()
}

/// `a` and `b` are equal when A to Z are taken as a to z. Characters
/// outside ASCII must match exactly.
#[inline]
pub fn ascii_eq_ignore_case(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}
