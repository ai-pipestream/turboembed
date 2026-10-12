//! English contraction expansion: `can't` to `can not`, `we're` to
//! `we are`, written through the view builder so a span over the
//! expansion maps back to the whole contraction.
//!
//! [`EnglishContractions`] expands the forms whose reading does not
//! depend on context. [`EnglishContractionsWithPos`] also expands `'s`
//! and `'d` when part-of-speech tags say which reading is meant.

mod pos;
mod rules;
mod scan;
mod word_break;

pub use pos::EnglishContractionsWithPos;

use turbo_analysis::ViewBuilder;

use crate::Normalizer;

/// Expands unambiguous English contractions into separate words.
///
/// A contraction is a run of ASCII letters, an apostrophe (U+0027,
/// U+2019, U+02BC or U+FF07) and a run of ASCII letters, and it must be a
/// complete word under the UAX #29 word boundaries: `_can't`, `can't1`,
/// `écan't`, `I.don't` and `can't've` are left alone, while `'can't'`,
/// `(don't)` and `can't,` are expanded. The forms:
///
/// | Form | Expansion |
/// |---|---|
/// | `can't` | `can not` |
/// | `won't`, `shan't` | `will not`, `shall not` |
/// | `<aux>n't` for are, could, dare, did, do, does, had, has, have, is, may, might, must, need, ought, should, was, were, would | `<aux> not` |
/// | `let's` | `let us` |
/// | `y'all` | `you all` |
/// | `'re` after how, here, there, these, they, those, we, what, when, where, who, why, you | ` are` |
/// | `'ve` after could, I, may, might, must, should, there, these, they, those, we, what, where, who, would, you | ` have` |
/// | `'ll` after he, here, how, I, it, she, that, there, these, they, this, those, we, what, when, where, which, who, why, you | ` will` |
/// | `I'm` | `I am` |
///
/// The stem is compared ASCII case-insensitively and kept as written. An
/// all-uppercase word gets an uppercase expansion (`CAN'T` to
/// `CAN NOT`); the replaced stems of `won't`, `shan't` and `y'all` follow
/// the case of what they replace (`Won't` to `Will not`). `'s`, `'d` and
/// `ain't` are ambiguous and left alone.
///
/// OpenNLP counterpart: `EnglishContractionCharSequenceNormalizer`. This
/// normalizer adds `y'all` and the words marked in docs/normalizers.md to
/// the subject lists.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EnglishContractions;

impl Normalizer for EnglishContractions {
    const ID: &'static str = "turbo:normalize:english-contractions";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        scan::expand(text, out, None);
    }
}
