use std::borrow::Cow;

use super::{lowercase_code_points_changes, to_lowercase};

/// Java ROOT lowercasing, borrowing the input when no character changes.
pub fn lowercase_cow(text: &str) -> Cow<'_, str> {
    if !lowercase_code_points_changes(text) {
        Cow::Borrowed(text)
    } else {
        let mut lower = String::with_capacity(text.len());
        to_lowercase(text, &mut lower);
        Cow::Owned(lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrows_unchanged_unicode_and_keeps_contextual_mappings() {
        for word in ["dogs", "café", "σοφος", "123"] {
            assert!(matches!(lowercase_cow(word), Cow::Borrowed(_)));
        }
        for word in ["DOGS", "CAFÉ", "ΣΟΦΟΣ", "İs"] {
            let mut expected = String::new();
            to_lowercase(word, &mut expected);
            assert_eq!(lowercase_cow(word), expected);
        }
    }
}
