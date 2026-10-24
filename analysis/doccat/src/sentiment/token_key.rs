//! The lookup form of a token, shared by the sentiment lexicons.

use std::borrow::Cow;

use turbo_unicode::casefold::fold;
use turbo_unicode::category::general_category;

fn trimmable(c: char, keep_hyphen: bool) -> bool {
    c != '\'' && !(keep_hyphen && c == '-') && general_category(c).is_punctuation()
}

/// A token's lookup form: typographic apostrophes (U+2019, U+02BC)
/// read as `'`, Unicode punctuation trimmed from both ends but
/// for the apostrophe, and the rest case-folded; the token as it
/// is when that leaves nothing (an emoji). Already-clean ASCII
/// lowercase tokens are returned borrowed, with no allocation.
pub(crate) fn token_key_cow(token: &str, keep_hyphen: bool) -> Cow<'_, str> {
    let clean_ascii = !token.is_empty()
        && token.is_ascii()
        && !token.bytes().any(|b| b.is_ascii_uppercase())
        && token.chars().next().is_some_and(|c| !trimmable(c, keep_hyphen))
        && token.chars().last().is_some_and(|c| !trimmable(c, keep_hyphen));
    if clean_ascii {
        return Cow::Borrowed(token);
    }
    let plain: String = token.chars().map(|c| if matches!(c, '\u{2019}' | '\u{02BC}') { '\'' } else { c }).collect();
    let trimmed = plain.trim_matches(|c: char| trimmable(c, keep_hyphen));
    if trimmed.is_empty() {
        return Cow::Borrowed(token);
    }
    let mut out = String::with_capacity(trimmed.len());
    fold(trimmed, &mut out);
    Cow::Owned(out)
}

/// [`token_key_cow`] as an owned string.
pub(crate) fn token_key(token: &str) -> String {
    token_key_cow(token, false).into_owned()
}

/// Like [`token_key_cow`], but a token of nothing but
/// punctuation (but for apostrophes) reads as empty, for the
/// token streams that match lexicons word by word.
pub(crate) fn token_key_strict(token: &str, keep_hyphen: bool) -> String {
    if !token.is_empty() && token.chars().all(|c| trimmable(c, keep_hyphen)) {
        return String::new();
    }
    token_key_cow(token, keep_hyphen).into_owned()
}

/// The lookup form for aspect work in `turbo-analysis-embed`,
/// which shares this crate's sentiment tables.
pub fn embed_token_key(token: &str) -> String {
    token_key_strict(token, true)
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
    fn clean_tokens_are_borrowed() {
        assert!(matches!(token_key_cow("great", false), Cow::Borrowed("great")));
        assert!(matches!(token_key_cow("don't", false), Cow::Borrowed("don't")));
        assert!(matches!(token_key_cow("Great", false), Cow::Owned(_)));
    }
}
