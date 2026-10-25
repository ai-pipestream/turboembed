//! ASCII classification used by frequency segmentation.

/// `c` is an ASCII letter or digit.
#[inline]
pub fn is_ascii_alnum(c: char) -> bool {
    c.is_ascii_alphanumeric()
}
