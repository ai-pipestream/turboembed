//! English: numbers written in digits or words, fractions, negatives,
//! scales, ordinals (relative ones included) and percentages.
//!
//! docs/grammar.md lists what the rules read and how a reading is chosen.

mod lexicon;
pub(crate) mod rules;
pub(crate) mod time;

pub use time::{ENGLISH_TIMES, EnglishTimes};

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
pub(crate) const DIMS: u16 = 25;

/// An ordinal word that is only `first` or `second`, which never names a
/// denominator (`a second` is no half).
pub(crate) const SIMPLE: u8 = flag::GRAMMAR;
/// Relative-ordinal words: what they count from.
pub(crate) const TO_END: u8 = 1;
pub(crate) const TO_CURRENT: u8 = 2;

/// Words after which a lone `one` is a pronoun, not a number: `the one
/// you mentioned`, `which one`. Ordinal words do the same (see
/// [`English::picks_out`]).
const DETERMINERS: &[&str] = &["any", "each", "every", "no", "that", "the", "this", "which"];

/// The English grammar.
pub struct English {
    lexicon: OnceLock<Lexicon>,
}

/// The one instance; its lexicon is sorted on first use.
pub static ENGLISH: English = English { lexicon: OnceLock::new() };

impl English {
    /// Whether the lexeme at `i` makes a `one` after it a pronoun: a
    /// determiner, an ordinal (`the first one`, `the 3rd one`) or a
    /// relative ordinal (`the last one`).
    fn picks_out(&self, ctx: &Ctx<'_>, i: usize) -> bool {
        let word = ctx.word(i);
        DETERMINERS.contains(&word)
            || self.lexicon().lookup(word).iter().any(|e| match e.dim {
                ORD_WORD | ORD_TENS | ORD_SCALE | REL_WORD => true,
                ORD_SUFFIX => ctx.lexeme(i).glued,
                _ => false,
            })
    }
}

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
        NumeralStyle { mixed_separators: true, space_groups: true, indian_groups: true, decimal_comma: false }
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
                !(lone_one && ctx.before().is_some_and(|p| self.picks_out(ctx, p)))
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
            Output::Percent | Output::Time => true,
        }
    }

    /// A count before a scale ordinal stands as a number: `two millionth`
    /// also reads as two millionths, as `100 millionths` does.
    fn shadows(&self, ordinal: &Node) -> bool {
        self.rules().get(usize::from(ordinal.rule)).is_none_or(|r| r.name != "ord_scale")
    }
}

#[cfg(test)]
mod tests;
