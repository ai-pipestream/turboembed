//! English: numbers written in digits or words, fractions, negatives,
//! scales, ordinals (relative ones included) and percentages.
//!
//! docs/grammar.md lists what the rules read and how a reading is chosen.

mod lexicon;
mod rules;

use std::sync::OnceLock;

use crate::engine::{Ctx, Dim, Grammar, Lexicon, Node, NumeralStyle, Output, Rule};
use crate::value::{Value, flag};

// Word classes, one per lexicon category.
pub(crate) const UNIT: Dim = Dim(0);
pub(crate) const TEEN: Dim = Dim(1);
pub(crate) const TENS: Dim = Dim(2);
pub(crate) const HUNDRED: Dim = Dim(3);
pub(crate) const SCALE: Dim = Dim(4);
pub(crate) const DOZEN: Dim = Dim(5);
pub(crate) const ARTICLE: Dim = Dim(6);
pub(crate) const ORD_WORD: Dim = Dim(7);
pub(crate) const ORD_TENS: Dim = Dim(8);
pub(crate) const ORD_SCALE: Dim = Dim(9);
pub(crate) const FRAC_WORD: Dim = Dim(10);
pub(crate) const PCT_WORD: Dim = Dim(11);
pub(crate) const NEG_WORD: Dim = Dim(12);
pub(crate) const REL_WORD: Dim = Dim(13);
pub(crate) const VULGAR: Dim = Dim(14);
pub(crate) const ORD_SUFFIX: Dim = Dim(15);
// What the rules build.
/// A numeral lexeme.
pub(crate) const NUMERAL: Dim = Dim(16);
/// An integer that can still take part in a larger one: `three hundred`,
/// `4 thousand`.
pub(crate) const INT: Dim = Dim(17);
/// A finished number: the numbers output.
pub(crate) const NUM: Dim = Dim(18);
/// The denominator of a spelled fraction: `fifths`, `quarter`.
pub(crate) const DEN: Dim = Dim(19);
/// Digits spoken one by one after `point`: `zero three`.
pub(crate) const DIGSEQ: Dim = Dim(20);
/// A run of `+` and `-` before an exponent.
pub(crate) const SIGN: Dim = Dim(21);
pub(crate) const ORD: Dim = Dim(22);
pub(crate) const REL: Dim = Dim(23);
pub(crate) const PCT: Dim = Dim(24);
const DIMS: u16 = 25;

/// An ordinal word that is only `first` or `second`, which never names a
/// denominator (`a second` is no half).
pub(crate) const SIMPLE: u8 = flag::GRAMMAR;
/// Relative-ordinal words: what they count from.
pub(crate) const TO_END: u8 = 1;
pub(crate) const TO_CURRENT: u8 = 2;

/// Words after which a lone `one` is a pronoun, not a number: `the one
/// you mentioned`, `which one`.
const DETERMINERS: &[&str] = &["any", "each", "every", "no", "that", "the", "this", "which"];

/// The English grammar.
pub struct English {
    lexicon: OnceLock<Lexicon>,
}

/// The one instance; its lexicon is sorted on first use.
pub static ENGLISH: English = English { lexicon: OnceLock::new() };

impl Grammar for English {
    fn name(&self) -> &'static str {
        "en"
    }

    fn lexicon(&self) -> &Lexicon {
        self.lexicon.get_or_init(|| Lexicon::new(lexicon::ENTRIES))
    }

    fn rules(&self) -> &'static [Rule] {
        rules::RULES
    }

    fn dims(&self) -> u16 {
        DIMS
    }

    fn numeral(&self) -> Dim {
        NUMERAL
    }

    fn style(&self) -> NumeralStyle {
        NumeralStyle { mixed_separators: true, space_groups: true, indian_groups: true }
    }

    fn output(&self, dim: Dim) -> Option<Output> {
        match dim {
            NUM => Some(Output::Number),
            ORD | REL => Some(Output::Ordinal),
            PCT => Some(Output::Percent),
            _ => None,
        }
    }

    fn keep(&self, output: Output, node: &Node, ctx: &Ctx<'_>) -> bool {
        let before = ctx.before().map(|p| ctx.word(p));
        match output {
            Output::Number => {
                let lone_one = ctx.end - ctx.start == 1 && ctx.word(ctx.start) == "one";
                !(lone_one && before.is_some_and(|w| DETERMINERS.contains(&w)))
            }
            Output::Ordinal => match node.value {
                // `fifths` is a denominator, never a position.
                Value::Ordinal(o) if o.has(flag::PLURAL) => false,
                // `a hundredth` is a fraction.
                Value::Ordinal(_) => {
                    let bare_scale =
                        self.rules().get(usize::from(node.rule)).is_some_and(|r| r.name == "ord_scale_bare");
                    !(bare_scale && matches!(before, Some("a" | "an")))
                }
                _ => true,
            },
            Output::Percent => true,
        }
    }
}

#[cfg(test)]
mod tests;
