//! Portuguese quantity grammar and the number forms needed by its rules.
mod numbers;
pub(crate) mod unit;
use crate::engine::{Ctx, Dim, Grammar, Lexicon, Node, NumeralStyle, Output, Rule, WordForms};
use std::sync::OnceLock;
const NUMERAL: Dim = Dim(0);
const WORD: Dim = Dim(1);
const SCALE: Dim = Dim(4);
const NUM: Dim = Dim(6);
const DIMS: u16 = 25;
struct Portuguese {
    lexicon: OnceLock<Lexicon>,
}
static PORTUGUESE: Portuguese = Portuguese { lexicon: OnceLock::new() };
impl WordForms for Portuguese {
    fn folds(&self) -> bool {
        true
    }
}
impl Grammar for Portuguese {
    fn name(&self) -> &'static str {
        "pt"
    }
    fn lexicon(&self) -> &Lexicon {
        self.lexicon.get_or_init(|| Lexicon::new(numbers::ENTRIES))
    }
    fn rules(&self) -> &'static [Rule] {
        numbers::RULES
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
    fn notation(&self) -> turbo_analysis_layers::NumberNotation {
        turbo_analysis_layers::NumberNotation::LatinEu
    }
    fn output(&self, d: Dim) -> Option<Output> {
        (d == NUM).then_some(Output::Number)
    }
    fn keep(&self, _: Output, _: &Node, _: &Ctx<'_>) -> bool {
        true
    }
}
fn counts(ctx: &Ctx<'_>, i: usize) -> bool {
    ctx.lexeme(i).kind == crate::engine::Kind::Numeral
        || PORTUGUESE.lexicon().lookup(ctx.word(i)).iter().any(|e| matches!(e.dim, WORD | SCALE))
}
