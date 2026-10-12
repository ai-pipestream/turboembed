//! Folds decimal digits of every script to ASCII digits.

use turbo_analysis::ViewBuilder;

use crate::Normalizer;
use crate::substitute::substitute;

/// The ASCII digit strings, indexed by value.
const ASCII_DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// Replaces every decimal digit (general category Nd) that is not an ASCII
/// digit with the ASCII digit of the same value: Arabic-Indic U+0663
/// becomes "3", fullwidth U+FF15 becomes "5", mathematical bold U+1D7CE
/// becomes "0". Other numerals (superscripts, fractions, Roman numerals,
/// circled digits) are not decimal digits and are left alone.
///
/// The digit values come from `turbo_unicode::java::digit`, which follows
/// `Character.digit(c, 10)` on Unicode 18.0 data.
///
/// OpenNLP counterpart: `DigitCharSequenceNormalizer`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Digits;

impl Normalizer for Digits {
    const ID: &'static str = "turbo:normalize:digits";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        // Only non-ASCII characters reach the mapping, so an ASCII digit is
        // never replaced by itself.
        substitute(text, out, |c| turbo_unicode::java::digit(c).map(|v| ASCII_DIGITS[usize::from(v)]));
    }
}
