//! The frozen Unicode behavior of Hugging Face `tokenizers` 0.23.
//!
//! Its category and aligned-normalization crates predate this workspace's
//! Unicode data. Small deltas retain their behavior without restoring those
//! external dependencies. Unassigned code points are not in `is_other`,
//! matching `unicode_categories` 0.1.1.

use super::{Form, data, full_decomposition, hangul};
use crate::GeneralCategory as G;
use crate::category::general_category;

mod tables;
use tables::{CATEGORIES, CCC, COMPOSE_BLACKLIST, NEW_NFD};

fn override_at(table: &[(u32, u32, u8)], c: char) -> Option<u8> {
    let u = c as u32;
    table
        .binary_search_by(|&(a, b, _)| {
            if b < u {
                std::cmp::Ordering::Less
            } else if a > u {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .ok()
        .map(|i| table[i].2)
}

fn categories(c: char) -> u8 {
    let g = general_category(c);
    let current = (matches!(g, G::Cc | G::Cf | G::Co) as u8)
        | ((matches!(g, G::Mn) as u8) << 1)
        | ((g.is_punctuation() as u8) << 2)
        | ((g.is_mark() as u8) << 3)
        | ((matches!(g, G::Nd) as u8) << 4)
        | ((matches!(g, G::Pc) as u8) << 5);
    override_at(&CATEGORIES, c).unwrap_or(current)
}

/// Whether `c` is in the frozen Cc, Cf, or Co categories.
pub fn is_other(c: char) -> bool {
    categories(c) & 1 != 0
}

/// Whether `c` is in the frozen nonspacing-mark category.
pub fn is_mark_nonspacing(c: char) -> bool {
    categories(c) & 2 != 0
}

/// Whether `c` is in any frozen punctuation category.
pub fn is_punctuation(c: char) -> bool {
    categories(c) & 4 != 0
}

/// Whether `c` is in any frozen mark category.
pub fn is_mark(c: char) -> bool {
    categories(c) & 8 != 0
}

/// Whether `c` is in the frozen decimal-number category.
pub fn is_number_decimal_digit(c: char) -> bool {
    categories(c) & 16 != 0
}

/// Whether `c` is in the frozen connector-punctuation category.
pub fn is_punctuation_connector(c: char) -> bool {
    categories(c) & 32 != 0
}

/// The canonical combining class used by upstream's aligned normalization.
pub fn canonical_combining_class(c: char) -> u8 {
    override_at(&CCC, c).unwrap_or_else(|| super::canonical_combining_class(c))
}

/// The upstream primary composite, or `None` when its tables had none.
pub fn compose(first: char, second: char) -> Option<char> {
    let key = (first as u32, second as u32);
    if COMPOSE_BLACKLIST.binary_search(&key).is_ok() { None } else { super::compose(first, second) }
}

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

/// Normalizes text to upstream NFC or NFD with source-change markers.
pub fn aligned(form: Form, text: &str) -> Aligned {
    aligned_chars(form, text.chars())
}

/// Normalizes characters to upstream NFC or NFD with source-change markers.
pub fn aligned_chars(form: Form, chars: impl Iterator<Item = char>) -> Aligned {
    assert!(matches!(form, Form::Nfc | Form::Nfd), "HF compatibility covers NFC and NFD");
    let mut normalized = Vec::with_capacity(chars.size_hint().0);
    for c in chars {
        decompose(form, c, &mut normalized);
    }
    canonical_order(&mut normalized);
    if form.composes() {
        compose_aligned(&mut normalized);
    }
    Aligned { chars: normalized.into_iter().map(|(c, _, change)| (c, change)).collect::<Vec<_>>().into_iter() }
}

/// Whether text is already in upstream NFC.
pub fn is_nfc(text: &str) -> bool {
    aligned(Form::Nfc, text).map(|(c, _)| c).eq(text.chars())
}

fn decompose(form: Form, c: char, out: &mut Vec<(char, u8, isize)>) {
    if let Some((jamo, len)) = hangul::decompose(c) {
        out.extend(jamo[..len].iter().enumerate().map(|(i, &c)| (c, 0, isize::from(i != 0))));
        return;
    }
    let mapping = if NEW_NFD.binary_search(&(c as u32)).is_ok() { None } else { full_decomposition(form, c, data(c)) };
    match mapping {
        Some(mapping) => {
            out.extend(mapping.iter().enumerate().map(|(i, &c)| (c, canonical_combining_class(c), isize::from(i != 0))))
        }
        None => out.push((c, canonical_combining_class(c), 0)),
    }
}

fn canonical_order(chars: &mut [(char, u8, isize)]) {
    for i in 1..chars.len() {
        let current = chars[i];
        if current.1 == 0 {
            continue;
        }
        let mut j = i;
        while j > 0 && chars[j - 1].1 > current.1 {
            chars[j] = chars[j - 1];
            j -= 1;
        }
        chars[j] = current;
    }
}

fn compose_aligned(chars: &mut Vec<(char, u8, isize)>) {
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
            compose(chars[at].0, c).map(|composite| (at, composite))
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

    #[test]
    fn frozen_categories_and_nfd_cover_newer_unicode_assignments() {
        assert!(!is_other('\u{378}'));
        assert!(!is_mark_nonspacing('\u{5c8}'));
        assert!(!is_mark_nonspacing('\u{7fd}'));
        assert_eq!(aligned(Form::Nfd, "\u{105c9}\u{11383}\u{113c5}").count(), 3);
        let nfc: String = aligned(Form::Nfc, "\u{105d2}\u{307}").map(|(c, _)| c).collect();
        assert_eq!(nfc, "\u{105d2}\u{307}");
        assert_eq!(canonical_combining_class('\u{5c8}'), 0);
    }
}
