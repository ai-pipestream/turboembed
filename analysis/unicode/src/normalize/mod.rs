//! The Unicode normalization forms NFC, NFD, NFKC and NFKD (UAX #15).
//!
//! Normalizing decomposes every character (canonical mappings for NFC and
//! NFD, canonical and compatibility mappings for NFKC and NFKD, Hangul
//! syllables by the algorithm of Unicode section 3.12), puts each run of
//! combining marks in canonical order by combining class, and for NFC and
//! NFKC composes the result again with the primary composites that are not
//! excluded from composition.
//!
//! Text is processed in runs. A run starts at a character with combining
//! class 0 whose quick check value for the form is Yes: nothing before such
//! a character can reorder or compose with anything after it, so each run
//! normalizes on its own. A run whose characters all have quick check Yes
//! and are already in canonical order is in the form and is copied without
//! further work; only the other runs are decomposed and recomposed.

mod alignment;
mod hangul;
pub mod hf;
mod run;
mod segment;

use crate::tables::normalize::*;
use crate::trie::{find, lookup};

pub use alignment::{Aligned, aligned, aligned_chars};
pub use segment::{for_each_run, is_normalized, normalize};

/// A normalization form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Form {
    /// Canonical decomposition followed by canonical composition.
    Nfc,
    /// Canonical decomposition.
    Nfd,
    /// Compatibility decomposition followed by canonical composition.
    Nfkc,
    /// Compatibility decomposition.
    Nfkd,
}

impl Form {
    /// The form composes after decomposing (NFC, NFKC).
    pub fn composes(self) -> bool {
        matches!(self, Form::Nfc | Form::Nfkc)
    }

    /// The form applies compatibility mappings (NFKC, NFKD).
    pub fn compatibility(self) -> bool {
        matches!(self, Form::Nfkc | Form::Nfkd)
    }

    fn quick_check_shift(self) -> u32 {
        match self {
            Form::Nfc => NFC_QC_SHIFT,
            Form::Nfd => NFD_QC_SHIFT,
            Form::Nfkc => NFKC_QC_SHIFT,
            Form::Nfkd => NFKD_QC_SHIFT,
        }
    }
}

/// A quick check value (the NFC_QC, NFD_QC, NFKC_QC and NFKD_QC properties).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuickCheck {
    /// The character can occur in the form.
    Yes,
    /// The character cannot occur in the form.
    No,
    /// The character can occur in the form or not, depending on what
    /// precedes it (only for NFC and NFKC).
    Maybe,
}

/// The packed normalization value of `c`: combining class, quick check
/// values and decomposition and composition flags.
#[inline]
fn data(c: char) -> u32 {
    lookup(&INDEX, &BLOCKS, c)
}

/// The quick check value of `form` in the packed value `data`.
#[inline]
fn quick_check_of(form: Form, data: u32) -> QuickCheck {
    match (data >> form.quick_check_shift()) & 3 {
        QC_NO => QuickCheck::No,
        QC_MAYBE => QuickCheck::Maybe,
        _ => QuickCheck::Yes,
    }
}

/// The canonical combining class of `c`.
#[inline]
pub fn canonical_combining_class(c: char) -> u8 {
    (data(c) & CCC_MASK) as u8
}

/// The quick check value of `c` for `form`.
#[inline]
pub fn quick_check(form: Form, c: char) -> QuickCheck {
    quick_check_of(form, data(c))
}

/// The canonical decomposition mapping of `c`, one level deep, as
/// UnicodeData.txt lists it; `None` when `c` has none. Hangul syllables
/// return `None` here: they decompose by algorithm, which the
/// normalization functions apply.
pub fn canonical_decomposition(c: char) -> Option<&'static [char]> {
    if data(c) & HAS_CANONICAL == 0 {
        return None;
    }
    find(&CANONICAL_KEYS, &CANONICAL_VALUES, c as u32).map(pool_slice)
}

/// The compatibility decomposition mapping of `c` (a mapping with a tag
/// such as `<font>` in UnicodeData.txt), one level deep; `None` when `c`
/// has none or only a canonical mapping.
pub fn compatibility_decomposition(c: char) -> Option<&'static [char]> {
    if data(c) & HAS_COMPATIBILITY == 0 {
        return None;
    }
    find(&COMPATIBILITY_KEYS, &COMPATIBILITY_VALUES, c as u32).map(pool_slice)
}

/// The primary composite of `first` followed by `second`, when there is
/// one and it is not excluded from composition. Hangul syllables compose
/// by algorithm (L + V, LV + T).
pub fn compose(first: char, second: char) -> Option<char> {
    if let Some(syllable) = hangul::compose(first, second) {
        return Some(syllable);
    }
    if data(first) & COMPOSES_FIRST == 0 || data(second) & COMPOSES_SECOND == 0 {
        return None;
    }
    let key = (u64::from(first) << 32) | u64::from(second);
    COMPOSITION_KEYS.binary_search(&key).ok().map(|i| COMPOSITION_VALUES[i])
}

/// The full decomposition of `c` for `form` (canonical, or canonical and
/// compatibility), before canonical ordering; `None` when `c` decomposes
/// to itself. Hangul syllables are not covered.
fn full_decomposition(form: Form, c: char, data: u32) -> Option<&'static [char]> {
    if form.compatibility() {
        if data & (HAS_CANONICAL | HAS_COMPATIBILITY) == 0 {
            return None;
        }
        find(&FULL_COMPATIBILITY_KEYS, &FULL_COMPATIBILITY_VALUES, c as u32).map(pool_slice)
    } else {
        if data & HAS_CANONICAL == 0 {
            return None;
        }
        find(&FULL_CANONICAL_KEYS, &FULL_CANONICAL_VALUES, c as u32).map(pool_slice)
    }
}

/// The characters a slot (offset << 8 | length) of `POOL` refers to.
#[inline]
fn pool_slice(slot: u32) -> &'static [char] {
    let offset = (slot >> 8) as usize;
    &POOL[offset..offset + (slot & 0xFF) as usize]
}
