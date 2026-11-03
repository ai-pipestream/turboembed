//! Normalization with the source-change markers used by aligned tokenizers.
//!
//! The first character of a source character's decomposition has change 0;
//! its remaining characters have change 1. Canonical ordering keeps those
//! markers with their characters. Composition adds `change - 1` to the
//! starter's marker whenever another character joins it.

use super::{Form, canonical_combining_class, canonical_order, data, full_decomposition, hangul};

/// An iterator of normalized characters and their source-change markers.
pub struct Aligned {
    chars: std::vec::IntoIter<(char, isize)>,
}

impl Iterator for Aligned {
    type Item = (char, isize);

    fn next(&mut self) -> Option<Self::Item> {
        self.chars.next()
    }
}

/// Normalizes `text` to `form`, retaining source alignment for each character.
pub fn aligned(form: Form, text: &str) -> Aligned {
    aligned_chars(form, text.chars())
}

/// Normalizes an iterator of characters with source-change markers.
pub fn aligned_chars(form: Form, chars: impl Iterator<Item = char>) -> Aligned {
    let mut normalized = Vec::with_capacity(chars.size_hint().0);
    for c in chars {
        decompose(form, c, &mut normalized);
    }
    canonical_order(&mut normalized, |c| c.1);
    if form.composes() {
        compose(&mut normalized);
    }
    Aligned { chars: normalized.into_iter().map(|(c, _, change)| (c, change)).collect::<Vec<_>>().into_iter() }
}

fn decompose(form: Form, c: char, out: &mut Vec<(char, u8, isize)>) {
    if let Some((jamo, len)) = hangul::decompose(c) {
        out.extend(jamo[..len].iter().enumerate().map(|(i, &c)| (c, 0, isize::from(i != 0))));
        return;
    }
    match full_decomposition(form, c, data(c)) {
        Some(mapping) => {
            out.extend(mapping.iter().enumerate().map(|(i, &c)| (c, canonical_combining_class(c), isize::from(i != 0))))
        }
        None => out.push((c, canonical_combining_class(c), 0)),
    }
}

fn compose(chars: &mut Vec<(char, u8, isize)>) {
    let mut kept = 0;
    let mut starter: Option<usize> = None;
    let mut last_class = 0;
    for i in 0..chars.len() {
        let (c, class, change) = chars[i];
        let combined = starter.and_then(|at| {
            let unblocked = last_class == 0 || last_class < class;
            if !unblocked {
                return None;
            }
            super::compose(chars[at].0, c).map(|composite| (at, composite))
        });
        if let Some((at, composite)) = combined {
            chars[at].0 = composite;
            chars[at].2 += change - 1;
            continue;
        }
        if class == 0 {
            starter = Some(kept);
        }
        last_class = class;
        chars[kept] = (c, class, change);
        kept += 1;
    }
    chars.truncate(kept);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize;

    #[test]
    fn every_form_has_the_same_text_with_and_without_alignment() {
        let texts = ["plain", "e\u{301}", "ae\u{302}\u{301}", "a\u{344}", "\u{1100}\u{1161}\u{11a8}", "ﬃ ①"];
        for form in [Form::Nfc, Form::Nfd, Form::Nfkc, Form::Nfkd] {
            for text in texts {
                let got: String = aligned(form, text).map(|(c, _)| c).collect();
                let mut want = String::new();
                normalize(form, text, &mut want);
                assert_eq!(got, want, "{form:?} {text:?}");
            }
        }
    }

    #[test]
    fn decomposition_and_composition_keep_source_changes() {
        let nfd: Vec<(char, isize)> = aligned(Form::Nfd, "e\u{301}").collect();
        assert_eq!(nfd, [('e', 0), ('\u{301}', 0)]);
        let reordered: Vec<(char, isize)> = aligned(Form::Nfd, "a\u{301}\u{327}").collect();
        assert_eq!(reordered, [('a', 0), ('\u{327}', 0), ('\u{301}', 0)]);
        let composed: Vec<(char, isize)> = aligned(Form::Nfc, "ae\u{302}\u{301}").collect();
        assert_eq!(composed, [('a', 0), ('\u{1ebf}', -2)]);
        let split: Vec<(char, isize)> = aligned(Form::Nfc, "a\u{344}").collect();
        assert_eq!(split, [('ä', -1), ('\u{301}', 1)]);
    }
}
