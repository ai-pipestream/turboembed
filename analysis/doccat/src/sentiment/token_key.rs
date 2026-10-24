//! The lookup form of a token, shared by the sentiment lexicons.

use turbo_unicode::casefold::fold;
use turbo_unicode::category::general_category;

/// A token's lookup form: typographic apostrophes (U+2019, U+02BC) read
/// as `'`, Unicode punctuation trimmed from both ends but for the
/// apostrophe, and the rest case-folded; the token as it is when that
/// leaves nothing (an emoji).
pub(crate) fn token_key(token: &str) -> String {
    let plain: String = token.chars().map(|c| if matches!(c, '\u{2019}' | '\u{02BC}') { '\'' } else { c }).collect();
    let trimmed = plain.trim_matches(|c: char| c != '\'' && general_category(c).is_punctuation());
    if trimmed.is_empty() {
        return token.to_owned();
    }
    let mut out = String::with_capacity(trimmed.len());
    fold(trimmed, &mut out);
    out
}
