//! Spanish: numbers written in digits (`1.234,56`) or words (`dos mil
//! trescientos cuarenta y siete`, `veintiuna`, `mil millones`), fractions
//! (`tres quintos`, `cinco con un cuarto`, `doscientosavos`), spoken
//! decimals (`siete con cincuenta`), ordinals (`vigésimo primero`,
//! `tresmillonésimo`, `1.º`, `2da`), relative ordinals (`penúltimo`) and
//! percentages (`cien por cien`).
//!
//! The lexer folds accents, so the lexicon and the rules hold folded words
//! (`dieciseis`, `ultimo`), and cuts compound number words into glued
//! pieces ([`split`]). docs/grammar.md lists what the rules read.

mod lexicon;
mod rules;
mod split;
pub(crate) mod time;
pub(crate) mod unit;

pub use time::{SPANISH_TIMES, SpanishTimes};

use std::sync::OnceLock;

use turbo_analysis_layers::NumberNotation;

use crate::engine::{Ctx, Dim, Grammar, Kind, Lexicon, Node, NumeralStyle, Output, Rule, WordForms};
use crate::value::{Value, flag};

// Word classes, one per lexicon category.
pub(crate) const UNIT: Dim = Dim(0);
/// `diez` to `veintinueve`, each one word.
pub(crate) const TEEN: Dim = Dim(1);
pub(crate) const TENS: Dim = Dim(2);
/// `cien`, `ciento`, `doscientos` to `novecientos`.
pub(crate) const HUNDREDS: Dim = Dim(3);
/// `mil`, `millón`, `billón`, with their power of ten.
pub(crate) const SCALE: Dim = Dim(4);
pub(crate) const DOZEN: Dim = Dim(5);
pub(crate) const ORD_WORD: Dim = Dim(6);
pub(crate) const ORD_TENS: Dim = Dim(7);
pub(crate) const ORD_HUND: Dim = Dim(8);
pub(crate) const ORD_SCALE: Dim = Dim(9);
/// A word that names a denominator only: `medio`, `tercio`, `treintavo`.
pub(crate) const FRAC_WORD: Dim = Dim(10);
/// The `-avo` ending cut from a compound.
pub(crate) const AVO: Dim = Dim(11);
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
/// Decimal digits said after `coma` or `con`: `cero tres`, `cincuenta`.
pub(crate) const DIGSEQ: Dim = Dim(20);
/// A run of `+` and `-` before an exponent.
pub(crate) const SIGN: Dim = Dim(21);
pub(crate) const ORD: Dim = Dim(22);
pub(crate) const REL: Dim = Dim(23);
pub(crate) const PCT: Dim = Dim(24);
pub(crate) const DIMS: u16 = 25;

/// An ordinal word that names no denominator: `primero`, `segundo`,
/// `tercero` (a third is a `tercio`).
pub(crate) const SIMPLE: u8 = flag::GRAMMAR;
/// A number that takes no smaller part after it: `cien`.
pub(crate) const CLOSED: u8 = flag::GRAMMAR << 1;
/// A fraction said with `un` or `una` as its numerator, which can follow
/// `y` in a mixed number: `cinco y un cuarto`.
pub(crate) const ONE_PART: u8 = flag::GRAMMAR << 2;
/// Relative-ordinal words: what they count from.
pub(crate) const TO_END: u8 = 1;
pub(crate) const TO_CURRENT: u8 = 2;

/// The Spanish grammar.
pub struct Spanish {
    lexicon: OnceLock<Lexicon>,
}

/// The one instance; its lexicon is sorted on first use.
pub static SPANISH: Spanish = Spanish { lexicon: OnceLock::new() };

impl WordForms for Spanish {
    fn folds(&self) -> bool {
        true
    }

    fn split(&self, word: &str, cuts: &mut Vec<u32>) {
        split::split(self.lexicon(), word, cuts);
    }
}

impl Grammar for Spanish {
    fn name(&self) -> &'static str {
        "es"
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

    /// A count before a scale ordinal stands as a number: `dos
    /// millonésimo` holds the number 2.
    fn shadows(&self, ordinal: &Node) -> bool {
        self.rules().get(usize::from(ordinal.rule)).is_none_or(|r| r.name != "ord_scale")
    }
}

/// Whether a candidate stands in its context.
pub(crate) fn keep(rules: &[Rule], output: Output, node: &Node, ctx: &Ctx<'_>) -> bool {
    let before = ctx.before().map(|p| ctx.word(p));
    let one_word = ctx.end - ctx.start == 1;
    match output {
        Output::Number => {
            let word = ctx.word(ctx.start);
            // `por ciento` and `por cien` name a percentage, not a hundred.
            let hundred_of_pct = before == Some("por") && matches!(word, "cien" | "ciento");
            // `un` or `una` before a count is an article: `subieron un 12,5
            // por ciento`. Alone it counts one: `tirate un paso`.
            let article = matches!(word, "un" | "una") && ctx.after().is_some_and(|a| counts(ctx, a));
            !(one_word && (hundred_of_pct || article))
        }
        // `un quinto` and `tres quintos` are fractions; `un primer paso`
        // holds an ordinal.
        Output::Ordinal => match node.value {
            Value::Ordinal(o) if o.has(flag::SPELLED) && !o.has(SIMPLE) => {
                if o.has(flag::PLURAL) && ctx.before().is_some_and(|p| counts(ctx, p)) {
                    return false;
                }
                let after_article = matches!(before, Some("un" | "una"));
                let single = one_word || rules.get(usize::from(node.rule)).is_some_and(|r| r.name == "ord_scale_bare");
                !(after_article && single && ctx.lexeme(ctx.start).kind == Kind::Word)
            }
            _ => true,
        },
        Output::Percent | Output::Quantity | Output::Time => true,
    }
}

/// Whether the lexeme at `i` is a count: digits or a number word.
pub(crate) fn counts(ctx: &Ctx<'_>, i: usize) -> bool {
    ctx.lexeme(i).kind == Kind::Numeral
        || SPANISH.lexicon().lookup(ctx.word(i)).iter().any(|e| matches!(e.dim, UNIT | TEEN | TENS | HUNDREDS | SCALE))
}

#[cfg(test)]
mod tests;
