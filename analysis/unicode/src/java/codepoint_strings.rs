//! OpenNLP StringUtil's context-independent lowercase operations.
use super::to_lower_char;
use crate::ascii::push_lowercase_run;

/// Appends `StringUtil.toLowerCase(text)` to `out`: one simple lowercase
/// mapping per code point. Unlike String.toLowerCase(Locale.ROOT), this
/// maps U+0130 to `i` and always maps capital sigma to U+03C3.
pub fn to_lowercase_code_points(text: &str, out: &mut String) {
    out.reserve(text.len());
    let mut i = 0;
    while i < text.len() {
        let end = push_lowercase_run(text, i, out);
        if end > i {
            i = end;
            continue;
        }
        let c = text[i..].chars().next().expect("i is a character boundary");
        out.push(to_lower_char(c));
        i += c.len_utf8();
    }
}

/// The allocating form of [`to_lowercase_code_points`].
pub fn lowercase_code_points(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    to_lowercase_code_points(text, &mut out);
    out
}

/// Whether StringUtil's code-point lowercase changes `text`.
#[inline]
pub fn lowercase_code_points_changes(text: &str) -> bool {
    if text.is_ascii() {
        return text.bytes().any(|b| b.is_ascii_uppercase());
    }
    text.chars().any(|c| to_lower_char(c) != c)
}

/// Whether StringUtil's lowercase of `word` equals `lower`, without
/// allocation. The second argument is compared as supplied, not folded.
#[inline]
pub fn eq_lowercase_code_points(word: &str, lower: &str) -> bool {
    if word.is_ascii() && lower.is_ascii() {
        return word.len() == lower.len() && word.bytes().zip(lower.bytes()).all(|(w, l)| w.to_ascii_lowercase() == l);
    }
    word.chars().map(to_lower_char).eq(lower.chars())
}
