//! Java String helpers that use this crate's character definitions.

/// String.isBlank: empty text or only Character.isWhitespace characters.
/// Unlike Unicode blank, no-break spaces and U+0085 are not blank here.
#[inline]
pub fn is_blank(text: &str) -> bool {
    text.chars().all(super::is_whitespace)
}
