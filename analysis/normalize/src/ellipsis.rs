//! Expands ellipses and dot leaders to ASCII full stops.

use turbo_analysis::ViewBuilder;

use crate::Normalizer;
use crate::substitute::substitute;
use crate::tables::ellipsis_fold;

/// Replaces each leader and ellipsis of [`ELLIPSES`](crate::tables::ELLIPSES)
/// with its full stops: U+2026 HORIZONTAL ELLIPSIS with "...", U+2025 TWO
/// DOT LEADER with "..", U+2024 ONE DOT LEADER with ".", and the vertical
/// presentation forms U+FE19 and U+FE30 like the characters they present.
/// Every byte of an expansion maps back to the one character it came from.
///
/// OpenNLP counterpart: `EllipsisCharSequenceNormalizer`, which expands
/// U+2025 and U+2026 only.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ellipsis;

impl Normalizer for Ellipsis {
    const ID: &'static str = "turbo:normalize:ellipsis";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        substitute(text, out, ellipsis_fold);
    }
}
