//! Dutch: numbers written in digits (`1.234,56`) or words, compounds
//! included (`tweeëntwintig`, `driehonderdvijftig`, `vijfentwintighonderd`),
//! fractions (`drie vijfde`, `driekwart`, `tweeënhalf`, `anderhalf`),
//! spoken decimals (`vijf komma vijf`), ordinals (`eerste`,
//! `honderdvijfentwintigste`, `1ste`, `11e`), relative ordinals
//! (`voorlaatste`, `twee na laatste`) and percentages (`twintig procent`).
//!
//! The lexer folds accents, so the lexicon and the rules hold folded words
//! (`een` for `één`, `en` for `ën`), and cuts compound number words into
//! glued pieces ([`split`]). docs/grammar.md lists what the rules read.

mod lexicon;
mod rules;
mod split;
pub(crate) mod time;

pub use time::{DUTCH_TIMES, DutchTimes};

use std::sync::OnceLock;

use turbo_analysis_layers::NumberNotation;

use crate::engine::{Ctx, Dim, Grammar, Lexicon, Node, NumeralStyle, Output, Rule, WordForms};
use crate::value::{Value, flag};

// Word classes, one per lexicon category.
pub(crate) const UNIT: Dim = Dim(0);
pub(crate) const TEEN: Dim = Dim(1);
pub(crate) const TENS: Dim = Dim(2);
pub(crate) const HUNDRED: Dim = Dim(3);
pub(crate) const SCALE: Dim = Dim(4);
/// `dozijn` and `gros`, with the size of the group.
pub(crate) const DOZEN: Dim = Dim(5);
/// `anderhalf`: one and a half.
pub(crate) const ONE_HALF: Dim = Dim(6);
pub(crate) const ORD_WORD: Dim = Dim(7);
pub(crate) const ORD_TENS: Dim = Dim(8);
pub(crate) const ORD_SCALE: Dim = Dim(9);
/// `half`, `halve`, `kwart`: a denominator only.
pub(crate) const FRAC_WORD: Dim = Dim(10);
pub(crate) const PCT_WORD: Dim = Dim(11);
pub(crate) const NEG_WORD: Dim = Dim(12);
pub(crate) const REL_WORD: Dim = Dim(13);
pub(crate) const VULGAR: Dim = Dim(14);
pub(crate) const ORD_SUFFIX: Dim = Dim(15);
// What the rules build.
pub(crate) const NUMERAL: Dim = Dim(16);
/// An integer that can still take part in a larger one.
pub(crate) const INT: Dim = Dim(17);
/// A finished number: the numbers output.
pub(crate) const NUM: Dim = Dim(18);
/// The denominator of a spelled fraction.
pub(crate) const DEN: Dim = Dim(19);
/// Digits said after `komma`.
pub(crate) const DIGSEQ: Dim = Dim(20);
/// A run of `+` and `-` before an exponent.
pub(crate) const SIGN: Dim = Dim(21);
pub(crate) const ORD: Dim = Dim(22);
pub(crate) const REL: Dim = Dim(23);
pub(crate) const PCT: Dim = Dim(24);
pub(crate) const DIMS: u16 = 25;

/// An ordinal word that names no denominator: `eerste`, `nulde`.
pub(crate) const SIMPLE: u8 = flag::GRAMMAR;
/// An ordinal glued to a word before it, as in `honderdachtste`: no
/// denominator of a fraction, which stands apart (`drie vijfde`).
pub(crate) const GLUED: u8 = flag::GRAMMAR << 1;
/// A fraction said as a ratio: `een op de drie`.
pub(crate) const RATIO: u8 = flag::GRAMMAR << 2;
/// Relative-ordinal words: what they count from.
pub(crate) const TO_END: u8 = 1;
pub(crate) const TO_CURRENT: u8 = 2;

/// Words after which a lone `een` or `één` is a pronoun, not a number:
/// `de een`, `die ene`.
const DETERMINERS: &[&str] = &["de", "die", "deze", "dat", "het", "elke", "ieder", "iedere", "welke", "geen"];

/// The Dutch grammar.
pub struct Dutch {
    lexicon: OnceLock<Lexicon>,
}

/// The one instance; its lexicon is sorted on first use.
pub static DUTCH: Dutch = Dutch { lexicon: OnceLock::new() };

impl WordForms for Dutch {
    fn folds(&self) -> bool {
        true
    }

    fn split(&self, word: &str, cuts: &mut Vec<u32>) {
        split::split(self.lexicon(), word, cuts);
    }
}

impl Grammar for Dutch {
    fn name(&self) -> &'static str {
        "nl"
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
        NumeralStyle { mixed_separators: true, space_groups: true, indian_groups: false, decimal_comma: true }
    }

    fn words(&self) -> &dyn WordForms {
        self
    }

    fn notation(&self) -> NumberNotation {
        NumberNotation::LatinEu
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
        keep(self.rules(), output, node, ctx)
    }

    /// A count before a scale ordinal stands as a number: `drie
    /// biljoenste` also reads as three trillionths.
    fn shadows(&self, ordinal: &Node) -> bool {
        self.rules().get(usize::from(ordinal.rule)).is_none_or(|r| r.name != "ord_scale")
    }
}

/// Whether the lone word of a match is `een` written without accents: the
/// article, where `één` is the number.
pub(crate) fn plain_een(ctx: &Ctx<'_>) -> bool {
    ctx.end - ctx.start == 1 && ctx.word(ctx.start) == "een" && ctx.raw(ctx.start).eq_ignore_ascii_case("een")
}

/// Whether a candidate stands in its context.
pub(crate) fn keep(rules: &[Rule], output: Output, node: &Node, ctx: &Ctx<'_>) -> bool {
    let before = ctx.before().map(|p| ctx.word(p));
    match output {
        // `een` is the article; `één` is a number unless a determiner
        // makes it a pronoun (`de één`).
        Output::Number => {
            let lone_one = ctx.end - ctx.start == 1 && ctx.word(ctx.start) == "een";
            !(plain_een(ctx) || lone_one && before.is_some_and(|b| DETERMINERS.contains(&b)))
        }
        Output::Ordinal => match node.value {
            // `vijfden` is a denominator, never a position.
            Value::Ordinal(o) if o.has(flag::PLURAL) => false,
            // `een miljoenste` is a fraction.
            Value::Ordinal(_) => {
                let name = rules.get(usize::from(node.rule)).map_or("", |r| r.name);
                let one_scale = name == "ord_scale" && ctx.end - ctx.start == 2 && ctx.word(ctx.start) == "een";
                !(one_scale || name == "ord_scale_bare" && before == Some("een"))
            }
            _ => true,
        },
        Output::Percent | Output::Time => true,
    }
}

#[cfg(test)]
mod tests;
