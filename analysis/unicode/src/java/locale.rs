//! Turkish and Azerbaijani locale lowercase, using SpecialCasing.txt's
//! Before_Dot and After_I contexts on the original text. Other mappings
//! retain the shared JDK lowercase behavior, including final sigma.

use super::{Lowercase, lowercase_at};
use crate::normalize::canonical_combining_class;

/// Mapping of the character at `at` for either `tr` or `az`.
/// The empty string means deletion of U+0307 after capital I. A combining
/// character of class 0 or 230 blocks the context; the target I/dot is
/// tested before this barrier rule. Input must point to `c` in `text`.
pub fn turkic_lowercase_at(text: &str, at: usize, c: char) -> Lowercase {
    assert!(text.get(at..).is_some_and(|s| s.starts_with(c)), "c is not at byte at");
    match c {
        '\u{130}' => Lowercase::Char('i'),
        'I' => {
            let before_dot = text[at + 1..]
                .chars()
                .find(|&next| next == '\u{307}' || matches!(canonical_combining_class(next), 0 | 230))
                == Some('\u{307}');
            Lowercase::Char(if before_dot { 'i' } else { '\u{131}' })
        }
        '\u{307}' => {
            let after_i = text[..at]
                .chars()
                .rev()
                .find(|&prev| prev == 'I' || matches!(canonical_combining_class(prev), 0 | 230))
                == Some('I');
            if after_i { Lowercase::Str("") } else { Lowercase::Char(c) }
        }
        _ => lowercase_at(text, at, c),
    }
}

/// Appends Turkish/Azerbaijani lowercase to `out`.
pub fn to_turkic_lowercase(text: &str, out: &mut String) {
    for (at, c) in text.char_indices() {
        match turkic_lowercase_at(text, at, c) {
            Lowercase::Char(c) => out.push(c),
            Lowercase::Str(s) => out.push_str(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_context_controls_the_dot_and_capital_i() {
        for (text, want) in [
            ("İSTANBUL IĞDIR", "istanbul ığdır"),
            ("I\u{307}", "i"),
            ("I\u{323}\u{307}", "i\u{323}"),
            ("I\u{315}\u{307}", "i\u{315}"),
            ("I\u{301}\u{307}", "ı\u{301}\u{307}"),
            ("I \u{307}", "ı \u{307}"),
            ("I\u{307}\u{307}", "i\u{307}"),
            ("i\u{307}", "i\u{307}"),
            ("ΟΣ Μάιος", "ος μάιος"),
        ] {
            let mut out = String::new();
            to_turkic_lowercase(text, &mut out);
            assert_eq!(out, want, "{text:?}");
        }
    }
}
