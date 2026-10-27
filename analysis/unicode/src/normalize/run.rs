//! Normalizing one run: decomposition, canonical ordering and composition.

use super::{Form, canonical_order, data, full_decomposition, hangul};
use crate::tables::normalize::CCC_MASK;

/// The buffers of run normalization, kept between runs so that a long
/// text allocates them once.
#[derive(Default)]
pub(super) struct RunNormalizer {
    /// The run's characters with their combining classes.
    chars: Vec<(char, u8)>,
    /// The normalized run.
    pub(super) out: String,
}

impl RunNormalizer {
    /// Normalizes `run` to `form` into `self.out`.
    pub(super) fn normalize(&mut self, form: Form, run: &str) {
        self.chars.clear();
        for c in run.chars() {
            self.decompose(form, c);
        }
        canonical_order(&mut self.chars, |c| c.1);
        if form.composes() {
            compose(&mut self.chars);
        }
        self.out.clear();
        self.out.extend(self.chars.iter().map(|&(c, _)| c));
    }

    fn decompose(&mut self, form: Form, c: char) {
        if let Some((jamo, len)) = hangul::decompose(c) {
            self.chars.extend(jamo[..len].iter().map(|&j| (j, 0)));
            return;
        }
        let packed = data(c);
        match full_decomposition(form, c, packed) {
            Some(mapping) => self.chars.extend(mapping.iter().map(|&m| (m, (data(m) & CCC_MASK) as u8))),
            None => self.chars.push((c, (packed & CCC_MASK) as u8)),
        }
    }
}

/// The canonical composition algorithm: each character composes with the
/// last starter before it when the pair has a primary composite and no
/// character between them blocks it (a character blocks when its class is
/// 0 or not lower than the class of the character being composed).
fn compose(chars: &mut Vec<(char, u8)>) {
    let Some(&(first, first_class)) = chars.first() else { return };
    let mut starter = 0;
    let mut starter_char = first;
    // The class of the last character kept after the starter; 0 when the
    // starter is the last character kept. A run that begins with a
    // non-starter has no starter to compose with: 256 blocks everything.
    let mut last_class: u16 = if first_class == 0 { 0 } else { 256 };
    let mut kept = 1;
    for i in 1..chars.len() {
        let (c, class) = chars[i];
        let unblocked = last_class == 0 || last_class < u16::from(class);
        if unblocked && let Some(composite) = super::compose(starter_char, c) {
            chars[starter] = (composite, 0);
            starter_char = composite;
            continue;
        }
        if class == 0 {
            starter = kept;
            starter_char = c;
        }
        last_class = u16::from(class);
        chars[kept] = (c, class);
        kept += 1;
    }
    chars.truncate(kept);
}
