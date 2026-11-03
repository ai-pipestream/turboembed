//! Simple case mapping (UnicodeData.txt fields 13 and 14): each character
//! maps to one character, with no SpecialCasing.txt entry and no context,
//! so a mapped text keeps its count of scalar values and a character maps
//! the same wherever it stands (U+03A3 SIGMA lowers to U+03C3 at the end
//! of a word too). The edit trees of a lemmatizer head read a word through
//! [`lowercase`], so a sentence-initial capital shares its tree with the
//! lowercase word (`turbo::head::edit_tree`).

use crate::ascii::push_lowercase_run;
use crate::props::bits;
use crate::tables::case::{LOWER_KEYS, LOWER_VALUES, UPPER_KEYS, UPPER_VALUES};
use crate::tables::props::{HAS_LOWER, HAS_UPPER};
use crate::trie::find;

/// The simple lowercase mapping of `c`, `c` itself when it has none.
#[inline]
pub fn lower(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_lowercase();
    }
    if bits(c) & HAS_LOWER == 0 {
        return c;
    }
    find(&LOWER_KEYS, &LOWER_VALUES, c as u32).unwrap_or(c)
}

/// The simple uppercase mapping of `c`, `c` itself when it has none.
#[inline]
pub fn upper(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_uppercase();
    }
    if bits(c) & HAS_UPPER == 0 {
        return c;
    }
    find(&UPPER_KEYS, &UPPER_VALUES, c as u32).unwrap_or(c)
}

/// Appends the simple lowercase mapping of each character of `text` to
/// `out`: as many characters as `text` has, in its order.
pub fn lowercase(text: &str, out: &mut String) {
    out.reserve(text.len());
    let mut i = 0;
    while i < text.len() {
        let ascii_end = push_lowercase_run(text, i, out);
        if ascii_end > i {
            i = ascii_end;
            continue;
        }
        let c = text[i..].chars().next().expect("i is at a character boundary");
        out.push(lower(c));
        i += c.len_utf8();
    }
}

/// Whether [`lowercase`] changes `text`: some character of it lowers to
/// another.
pub fn has_upper(text: &str) -> bool {
    text.chars().any(|c| lower(c) != c)
}
