//! Folds typographic quotation marks to the ASCII apostrophe and quotation
//! mark.

use turbo_analysis::ViewBuilder;

use crate::Normalizer;
use crate::substitute::substitute;
use crate::tables::quote_fold;

/// Replaces the single quotation marks and apostrophes of
/// [`SINGLE_QUOTES`](crate::tables::SINGLE_QUOTES) with U+0027 and the
/// double quotation marks of [`DOUBLE_QUOTES`](crate::tables::DOUBLE_QUOTES)
/// with U+0022, in one pass. OpenNLP runs the two sets as two passes; the
/// sets do not overlap and no replacement is a member of either, so one
/// pass writes the same text.
///
/// The CJK corner brackets, which also have the Unicode `Quotation_Mark`
/// property, are left alone (see
/// [`CORNER_BRACKETS`](crate::tables::CORNER_BRACKETS)).
///
/// OpenNLP counterpart: `QuoteCharSequenceNormalizer`. This set adds
/// U+2E42 DOUBLE LOW-REVERSED-9 QUOTATION MARK.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Quotes;

impl Normalizer for Quotes {
    const ID: &'static str = "turbo:normalize:quotes";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        substitute(text, out, quote_fold);
    }
}
