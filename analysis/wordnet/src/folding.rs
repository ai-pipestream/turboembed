//! Canonical lemma normalization and space-splitting helpers (`LemmaFolding`).

use turbo_unicode::java::to_lower_char;

/// Folds a written form to canonical WordNet lemma form: replaces underscores (`_`)
/// with spaces (` `) and lowercases each code point using OpenNLP's locale-independent
/// `StringUtil.toLowerCase` (`Character.toLowerCase`).
#[inline]
pub fn fold(written_form: &str) -> String {
    let mut out = String::with_capacity(written_form.len());
    fold_into(written_form, &mut out);
    out
}

/// Folds `written_form` into `out` after clearing `out`.
#[inline]
pub fn fold_into(written_form: &str, out: &mut String) {
    out.clear();
    out.reserve(written_form.len());
    for c in written_form.chars() {
        if c == '_' {
            out.push(' ');
        } else {
            out.push(to_lower_char(c));
        }
    }
}

/// Splits `value` on runs of ASCII space (`' '`), ignoring leading and trailing
/// spaces while preserving non-space characters such as tabs or non-breaking spaces.
#[inline]
pub fn split_on_spaces(value: &str) -> Vec<&str> {
    value.split(' ').filter(|s| !s.is_empty()).collect()
}
