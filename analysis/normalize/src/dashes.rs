//! Folds dash and hyphen characters to U+002D HYPHEN-MINUS.

use turbo_analysis::ViewBuilder;

use crate::substitute::substitute;
use crate::tables::{is_default_dash, is_math_minus};
use crate::{Normalizer, hash_config};

/// Replaces every dash in OpenNLP's default dash set (the `UnicodeDash`
/// table without its three mathematical minus signs) with U+002D. The
/// hyphen-minus itself is a member and stays as it is, so ASCII text
/// produces no edits.
///
/// [`Dashes::with_math_minus`] also folds the minus signs U+207B, U+208B
/// and U+2212. The default leaves them alone because in "5\u{2212}3" the
/// minus sign is an operator, not punctuation.
///
/// OpenNLP counterpart: `DashCharSequenceNormalizer`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dashes {
    math_minus: bool,
}

impl Dashes {
    /// The OpenNLP default set.
    pub fn new() -> Dashes {
        Dashes { math_minus: false }
    }

    /// The default set plus the three minus signs.
    pub fn with_math_minus() -> Dashes {
        Dashes { math_minus: true }
    }

    /// Whether the minus signs are folded too.
    pub fn folds_math_minus(&self) -> bool {
        self.math_minus
    }
}

impl Normalizer for Dashes {
    const ID: &'static str = "turbo:normalize:dashes";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let math_minus = self.math_minus;
        substitute(text, out, |c| (is_default_dash(c) || (math_minus && is_math_minus(c))).then_some("-"));
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.math_minus)
    }
}
