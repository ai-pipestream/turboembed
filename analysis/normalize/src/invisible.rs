//! Deletes invisible format and bidirectional control characters.

use turbo_analysis::ViewBuilder;

use crate::Normalizer;
use crate::substitute::remove;
use crate::tables::is_invisible;

/// Deletes the 26 code points of [`INVISIBLE`](crate::tables::INVISIBLE):
/// the soft hyphen, zero width space, word joiner, the invisible math
/// operators, the byte order mark, the bidirectional marks, embeddings,
/// overrides and isolates, and the six deprecated format characters
/// U+206A to U+206F. Each deletion is an edit to the empty string, so a
/// span in the output never covers a deleted character unless the span
/// crosses it.
///
/// The zero width joiner and non-joiner, the Mongolian vowel separator and
/// the variation selectors are kept: they change how text is shaped and
/// rendered, and the joiner holds emoji ZWJ sequences together.
///
/// OpenNLP counterpart: `InvisibleCharSequenceNormalizer`, which deletes the
/// first 20; the six deprecated format characters are an addition.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Invisible;

impl Normalizer for Invisible {
    const ID: &'static str = "turbo:normalize:invisible";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        remove(text, out, is_invisible);
    }
}
