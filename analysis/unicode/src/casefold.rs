//! Case folding (CaseFolding.txt).
//!
//! Full folding uses the entries of status C and F, simple folding those
//! of status C and S. The Turkic variants replace the folding of `I` and
//! U+0130 by the status T entries (`I` folds to dotless `ı`, U+0130 to `i`).

use crate::ascii::push_lowercase_run;
use crate::props::bits;
use crate::tables::case::{
    FOLD_FULL_KEYS, FOLD_FULL_VALUES, FOLD_SIMPLE_KEYS, FOLD_SIMPLE_VALUES, FOLD_TURKIC_KEYS, FOLD_TURKIC_VALUES,
};
use crate::tables::props::{HAS_FOLD, HAS_TURKIC_FOLD};
use crate::trie::find;

/// The full case folding of `c`, `None` when `c` folds to itself.
#[inline]
pub fn full(c: char) -> Option<&'static str> {
    if bits(c) & HAS_FOLD == 0 {
        return None;
    }
    find(&FOLD_FULL_KEYS, &FOLD_FULL_VALUES, c as u32)
}

/// The simple case folding of `c`: one character, `c` itself when it
/// folds to itself.
#[inline]
pub fn simple(c: char) -> char {
    if bits(c) & HAS_FOLD == 0 {
        return c;
    }
    find(&FOLD_SIMPLE_KEYS, &FOLD_SIMPLE_VALUES, c as u32).unwrap_or(c)
}

/// The full case folding with the Turkic entries applied.
pub fn turkic_full(c: char) -> Option<&'static str> {
    turkic(c).or_else(|| full(c))
}

/// The simple case folding with the Turkic entries applied.
pub fn turkic_simple(c: char) -> char {
    match turkic(c) {
        Some(folded) => folded.chars().next().expect("a Turkic folding is one character"),
        None => simple(c),
    }
}

fn turkic(c: char) -> Option<&'static str> {
    if bits(c) & HAS_TURKIC_FOLD == 0 {
        return None;
    }
    find(&FOLD_TURKIC_KEYS, &FOLD_TURKIC_VALUES, c as u32)
}

/// Appends the full case folding of `text` to `out`.
pub fn fold(text: &str, out: &mut String) {
    out.reserve(text.len());
    let mut i = 0;
    while i < text.len() {
        let ascii_end = push_lowercase_run(text, i, out);
        if ascii_end > i {
            i = ascii_end;
            continue;
        }
        let c = text[i..].chars().next().expect("i is at a character boundary");
        match full(c) {
            Some(folded) => out.push_str(folded),
            None => out.push(c),
        }
        i += c.len_utf8();
    }
}
