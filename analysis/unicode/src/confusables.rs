//! Confusable detection data of UTS #39 (confusables.txt).

use crate::normalize::{Form, normalize};
use crate::props::bits;
use crate::tables::confusables::{PROTOTYPE_KEYS, PROTOTYPE_VALUES};
use crate::tables::props::HAS_PROTOTYPE;
use crate::trie::find;

/// The prototype `c` maps to in confusables.txt: the string it can be
/// visually confused with. `None` when `c` is its own prototype.
#[inline]
pub fn prototype(c: char) -> Option<&'static str> {
    if bits(c) & HAS_PROTOTYPE == 0 {
        return None;
    }
    find(&PROTOTYPE_KEYS, &PROTOTYPE_VALUES, c as u32)
}

/// Appends the skeleton of `text` (UTS #39 section 4) to `out`: the NFD
/// form of the text with every character replaced by its prototype, put
/// in NFD again. Two strings are confusable when their skeletons are
/// equal.
pub fn skeleton(text: &str, out: &mut String) {
    let mut decomposed = String::with_capacity(text.len());
    normalize(Form::Nfd, text, &mut decomposed);
    let mut mapped = String::with_capacity(decomposed.len());
    for c in decomposed.chars() {
        match prototype(c) {
            Some(target) => mapped.push_str(target),
            None => mapped.push(c),
        }
    }
    normalize(Form::Nfd, &mapped, out);
}
