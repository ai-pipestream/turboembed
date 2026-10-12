//! Buffer helpers the light and minimal stemmers share: OpenNLP's
//! `StemmerUtil`, adapted from Lucene. Each works on the first `len` code
//! units of a buffer, the convention the stemmers stem in place by.

/// Whether the first `len` units of `s` end with `suffix`.
#[inline]
pub(crate) fn ends_with(s: &[u16], len: usize, suffix: &[u16]) -> bool {
    suffix.len() <= len && s[len - suffix.len()..len] == *suffix
}

/// Whether the first `len` units of `s` end with one of `suffixes`.
#[inline]
pub(crate) fn ends_any(s: &[u16], len: usize, suffixes: &[&[u16]]) -> bool {
    suffixes.iter().any(|x| ends_with(s, len, x))
}

/// Deletes the unit at `pos` (before `len`) and returns the new length.
#[inline]
pub(crate) fn delete(s: &mut [u16], pos: usize, len: usize) -> usize {
    s.copy_within(pos + 1..len, pos);
    len - 1
}

/// One of the unaccented vowels a, e, i, o, u and y: the vowel test of the
/// Finnish and Hungarian stemmers.
#[inline]
pub(crate) fn is_vowel(c: u16) -> bool {
    matches!(c, 0x61 | 0x65 | 0x69 | 0x6F | 0x75 | 0x79)
}

/// `Character.isLetter(char)` on one code unit: a surrogate is not a
/// letter.
#[inline]
pub(crate) fn is_letter(c: u16) -> bool {
    char::from_u32(c as u32).is_some_and(turbo_unicode::java::is_letter)
}

/// A code unit as a `char` for matching against letters; a surrogate,
/// which matches none of them, becomes U+FFFD.
#[inline]
pub(crate) fn ch(c: u16) -> char {
    char::from_u32(c as u32).unwrap_or(char::REPLACEMENT_CHARACTER)
}

/// A letter as a code unit. Every letter the stemmers write is in the
/// Basic Multilingual Plane.
#[inline]
pub(crate) const fn unit(c: char) -> u16 {
    c as u16
}

/// Replaces each unit of `s` found in `from` with the unit at the same
/// place in `to`.
#[inline]
pub(crate) fn fold(s: &mut [u16], from: &[u16], to: &[u16]) {
    debug_assert_eq!(from.len(), to.len());
    for c in s {
        if let Some(i) = from.iter().position(|f| f == c) {
            *c = to[i];
        }
    }
}
