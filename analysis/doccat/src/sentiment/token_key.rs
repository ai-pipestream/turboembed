//! The lookup form of a token, shared by the sentiment lexicons.

use std::borrow::Cow;

use turbo_unicode::casefold::fold;
use turbo_unicode::category::general_category;

/// What a caller trims from a token's edges before lookup.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Trim {
    /// Unicode punctuation only: symbols (an emoji) stay part of the key.
    Punctuation,
    /// Everything that is not alphanumeric: the word lexicons' old
    /// rule, under which a trailing emoji or currency sign comes off.
    NonAlphanumeric,
}

fn trimmable(c: char, keep_hyphen: bool, trim: Trim) -> bool {
    if c == '\'' || (keep_hyphen && c == '-') {
        return false;
    }
    match trim {
        Trim::Punctuation => general_category(c).is_punctuation(),
        Trim::NonAlphanumeric => !c.is_alphanumeric(),
    }
}

/// A token's lookup form: typographic apostrophes (U+2019, U+02BC)
/// read as `'`, Unicode punctuation trimmed from both ends but
/// for the apostrophe, and the rest case-folded; the token as it
/// is when that leaves nothing (an emoji). Already-clean ASCII
/// lowercase tokens are returned borrowed, with no allocation.
pub(crate) fn token_key_cow(token: &str, keep_hyphen: bool) -> Cow<'_, str> {
    key_with(token, keep_hyphen, Trim::Punctuation)
}

fn key_with(token: &str, keep_hyphen: bool, trim: Trim) -> Cow<'_, str> {
    let clean_ascii = !token.is_empty()
        && token.is_ascii()
        && !token.bytes().any(|b| b.is_ascii_uppercase())
        && token.chars().next().is_some_and(|c| !trimmable(c, keep_hyphen, trim))
        && token.chars().last().is_some_and(|c| !trimmable(c, keep_hyphen, trim));
    if clean_ascii {
        return Cow::Borrowed(token);
    }
    let plain: String = token.chars().map(|c| if matches!(c, '\u{2019}' | '\u{02BC}') { '\'' } else { c }).collect();
    let trimmed = plain.trim_matches(|c: char| trimmable(c, keep_hyphen, trim));
    if trimmed.is_empty() {
        return Cow::Borrowed(token);
    }
    let mut out = String::with_capacity(trimmed.len());
    fold(trimmed, &mut out);
    Cow::Owned(out)
}

/// [`token_key_cow`] as an owned string.
pub fn token_key(token: &str) -> String {
    token_key_cow(token, false).into_owned()
}

/// Like [`token_key_cow`], but a token of nothing but
/// punctuation (but for apostrophes) reads as empty, for the
/// token streams that match lexicons word by word.
pub(crate) fn token_key_strict(token: &str, keep_hyphen: bool) -> String {
    strict_with(token, keep_hyphen, Trim::Punctuation)
}

/// [`token_key_strict`] with the word lexicons' trim: symbols come
/// off the edges too, so a word carrying a trailing emoji still
/// finds its entry.
pub(crate) fn token_key_word(token: &str, keep_hyphen: bool) -> String {
    strict_with(token, keep_hyphen, Trim::NonAlphanumeric)
}

fn strict_with(token: &str, keep_hyphen: bool, trim: Trim) -> String {
    if !token.is_empty() && token.chars().all(|c| trimmable(c, keep_hyphen, trim)) {
        return String::new();
    }
    key_with(token, keep_hyphen, trim).into_owned()
}

/// The lookup form for aspect work in `turbo-analysis-embed`,
/// which shares this crate's sentiment tables.
pub fn embed_token_key(token: &str) -> String {
    token_key_word(token, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curly_apostrophes_and_unicode_punctuation_normalize() {
        assert_eq!(token_key("don\u{2019}t"), "don't");
        assert_eq!(token_key("don\u{02BC}t"), "don't");
        assert_eq!(token_key("\u{201C}great\u{201D}"), "great");
        assert_eq!(token_key("GREAT!"), "great");
        assert_eq!(token_key("\u{1F600}"), "\u{1F600}");
        assert_eq!(token_key_strict("-well-known-", true), "-well-known-");
        assert_eq!(token_key("-well-known-"), "well-known");
    }

    #[test]
    fn word_keys_trim_symbols_but_plain_keys_keep_them() {
        assert_eq!(token_key_word("idiot\u{1F621}", false), "idiot");
        assert_eq!(token_key_word("\u{1F621}IDIOT", false), "idiot");
        assert_eq!(token_key_word("\u{1F621}", false), "", "a bare emoji is no word");
        assert_eq!(token_key_word("-well-known-", true), "-well-known-");
        assert_eq!(token_key("idiot\u{1F621}"), "idiot\u{1F621}", "the plain key keeps symbols");
        assert_eq!(embed_token_key("great\u{1F600}"), "great");
    }

    #[test]
    fn clean_tokens_are_borrowed() {
        assert!(matches!(token_key_cow("great", false), Cow::Borrowed("great")));
        assert!(matches!(token_key_cow("don't", false), Cow::Borrowed("don't")));
        assert!(matches!(token_key_cow("Great", false), Cow::Owned(_)));
    }
}
