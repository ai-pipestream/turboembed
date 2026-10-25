//! Java String.equalsIgnoreCase on valid UTF-8 text.
use super::{to_lower_char, to_upper_char};
use crate::utf16::utf16_len;

#[inline]
fn same(x: char, y: char) -> bool {
    if x == y {
        return true;
    }
    let (ux, uy) = (to_upper_char(x), to_upper_char(y));
    ux == uy || to_lower_char(ux) == to_lower_char(uy)
}

/// Java `String.equalsIgnoreCase`: equal UTF-16 lengths, with each code
/// point compared as written, uppercased, or lowercased after uppercasing.
/// No locale rules or multi-character expansions apply. ASCII targets
/// take a byte-based path; non-ASCII matches still use the Unicode tables.
pub fn equals_ignore_case(a: &str, b: &str) -> bool {
    if a.is_ascii() && b.is_ascii() {
        return a.eq_ignore_ascii_case(b);
    }
    if b.is_ascii() {
        return equals_ascii_target(a, b);
    }
    if a.is_ascii() {
        return equals_ascii_target(b, a);
    }
    if utf16_len(a) != utf16_len(b) {
        return false;
    }
    let mut right = b.chars();
    a.chars().all(|left| right.next().is_some_and(|r| same(left, r))) && right.next().is_none()
}

fn equals_ascii_target(text: &str, target: &str) -> bool {
    let mut chars = text.chars();
    for t in target.bytes() {
        let Some(c) = chars.next() else {
            return false;
        };
        if !(if c.is_ascii() { (c as u8).eq_ignore_ascii_case(&t) } else { same(c, char::from(t)) }) {
            return false;
        }
    }
    chars.next().is_none()
}
