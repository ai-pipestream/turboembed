//! JDK 21 assertion semantics at UTF-8 character boundaries. A backward
//! mark walk stops at a supplementary character because Java encounters
//! its low surrogate first. A supplementary mark is a word side only
//! before the mark, not after it.
use super::compile::Assertion;
use turbo_unicode::GeneralCategory;
use turbo_unicode::java::code_point::{general_category, is_letter_or_digit};

impl Assertion {
    #[inline(always)]
    pub(super) fn holds(self, text: &str, pos: usize) -> bool {
        match self {
            Assertion::StartText => pos == 0,
            Assertion::EndText => pos == text.len(),
            Assertion::EndOrFinalTerminator => final_end(text, pos),
            Assertion::AsciiWordBoundary => boundary(text, pos),
            Assertion::AsciiNonWordBoundary => !boundary(text, pos),
        }
    }
}

#[inline(always)]
fn ascii_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline(always)]
fn boundary(text: &str, pos: usize) -> bool {
    let b = text.as_bytes();
    let prev = if pos == 0 { 0 } else { b[pos - 1] };
    let next = if pos == b.len() { 0 } else { b[pos] };
    if (prev | next).is_ascii() {
        return ascii_word(prev) != ascii_word(next);
    }
    unicode_boundary(text, pos)
}

#[cold]
fn unicode_boundary(text: &str, pos: usize) -> bool {
    let left = text[..pos].char_indices().next_back().is_some_and(|(i, c)| {
        (c.is_ascii() && ascii_word(c as u8)) || (c as u32 <= 0xFFFF && mark(c) && base(&text[..i]))
    });
    let right = text[pos..]
        .chars()
        .next()
        .is_some_and(|c| (c.is_ascii() && ascii_word(c as u8)) || (mark(c) && base(&text[..pos])));
    left != right
}

fn base(prefix: &str) -> bool {
    for c in prefix.chars().rev() {
        if c as u32 > 0xFFFF {
            return false;
        }
        if is_letter_or_digit(c as u32) {
            return true;
        }
        if !mark(c) {
            return false;
        }
    }
    false
}

fn mark(c: char) -> bool {
    // U+1171E changed from Mn to Mc after JDK 21's Unicode 15.0.
    general_category(c as u32) == GeneralCategory::Mn || c as u32 == 0x1171E
}

#[inline(always)]
fn final_end(text: &str, pos: usize) -> bool {
    let remaining = text.len() - pos;
    if remaining == 0 {
        return true;
    }
    if remaining > 3 {
        return false;
    }
    let rest = &text[pos..];
    match rest {
        "" | "\r\n" | "\r" | "\u{85}" | "\u{2028}" | "\u{2029}" => true,
        "\n" => pos == 0 || text.as_bytes()[pos - 1] != b'\r',
        _ => false,
    }
}
