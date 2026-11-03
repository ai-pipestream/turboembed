//! Spanish quantities and currencies, composed with the Spanish number grammar.
mod lexicon;
mod rules;

use super::{DIMS, NUMERAL, SPANISH};
use crate::engine::{Ctx, Dim, Grammar, Kind, Lexicon, Node, NumeralStyle, Output, Rule, WordForms};
use crate::unit::{Angle, CurrencyUnit, Generic, Temperature, Unit, Weight};
use crate::value::Value;
use std::sync::OnceLock;

const RAW: Dim = Dim(DIMS);
const UNIT: Dim = Dim(DIMS + 1);
const QTY: Dim = Dim(DIMS + 2);
const CODE: Dim = Dim(DIMS + 3);
const SCALE_WORD: Dim = Dim(DIMS + 4);
const TIME: Dim = Dim(DIMS + 5);
const COUNTRY: Dim = Dim(DIMS + 6);
const BARE: u8 = 1;
const PREFIX: u8 = 2;

/// Spanish measures, money, temperatures and ages, with original source offsets.
pub struct SpanishUnits {
    lexicon: OnceLock<Lexicon>,
}
pub static SPANISH_UNITS: SpanishUnits = SpanishUnits { lexicon: OnceLock::new() };
static RULES: OnceLock<Vec<Rule>> = OnceLock::new();

impl Grammar for SpanishUnits {
    fn name(&self) -> &'static str {
        "es-units"
    }
    fn lexicon(&self) -> &Lexicon {
        self.lexicon.get_or_init(|| {
            let mut entries = super::lexicon::ENTRIES.to_vec();
            entries.extend(lexicon::entries());
            Lexicon::new(&entries)
        })
    }
    fn rules(&self) -> &'static [Rule] {
        RULES.get_or_init(|| {
            let mut rules = SPANISH.rules().to_vec();
            rules.extend_from_slice(rules::RULES);
            rules
        })
    }
    fn dims(&self) -> u16 {
        DIMS + 7
    }
    fn numeral(&self) -> Dim {
        NUMERAL
    }
    fn style(&self) -> NumeralStyle {
        SPANISH.style()
    }
    fn words(&self) -> &dyn WordForms {
        SPANISH.words()
    }
    fn notation(&self) -> turbo_analysis_layers::NumberNotation {
        SPANISH.notation()
    }
    fn output(&self, dim: Dim) -> Option<Output> {
        if matches!(dim, UNIT | QTY) { Some(Output::Quantity) } else { SPANISH.output(dim) }
    }
    fn shadows(&self, ordinal: &Node) -> bool {
        SPANISH.shadows(ordinal)
    }
    fn keep(&self, output: Output, node: &Node, ctx: &Ctx<'_>) -> bool {
        if output != Output::Quantity {
            return SPANISH.keep(output, node, ctx);
        }
        let (unit, amount) = match node.value {
            Value::Unit(u) => (u, None),
            Value::Quantity(q) => (q.unit(), Some(q.value())),
            _ => return false,
        };
        if amount.is_none()
            && !lexicon::WORDS.iter().any(|w| w.unit == unit && w.flags & BARE != 0 && w.word == ctx.word(ctx.end - 1))
        {
            return false;
        }
        let near = |words: &[&str]| {
            (ctx.start.saturating_sub(5)..(ctx.end + 3).min(ctx.lexed.len())).any(|i| words.contains(&ctx.word(i)))
        };
        if matches!(unit, Unit::Age(_))
            && ctx.before().is_some_and(|p| {
                ctx.word(p) == "menos" || ctx.is_punct(p, '-') && (p == 0 || !super::counts(ctx, p - 1))
            })
        {
            return false;
        }
        if matches!(unit, Unit::Temperature(Temperature::Degree) | Unit::Angle(Angle::Degree))
            && ctx.raw(ctx.end - 1) == "º"
            && !near(&["temperatura", "calor", "calentar", "termostato"])
        {
            return false;
        }
        if matches!(unit, Unit::Currency(_)) && amount.is_some() {
            let first = ctx.start;
            let last = ctx.end - 1;
            if ctx.lexeme(first).kind == Kind::Numeral
                && ctx.lexeme(first).glued
                && ctx.before().is_some_and(|p| money_token(ctx, p))
            {
                return false;
            }
            if money_token(ctx, last)
                && ctx.after().is_some_and(|i| ctx.lexeme(i).kind == Kind::Numeral && ctx.lexeme(i).glued)
            {
                return false;
            }
        }
        match unit {
            Unit::Angle(Angle::Degree) | Unit::Temperature(Temperature::Degree) => {
                let angle = near(&["girar", "giro", "gira", "angulo", "rotar", "rotacion", "vuelta"])
                    && !near(&["temperatura", "calentar", "calor", "termostato", "hierve", "frio"]);
                angle == matches!(unit, Unit::Angle(_))
            }
            Unit::Weight(Weight::Pound) | Unit::Currency(CurrencyUnit::Generic(Generic::Pound))
                if matches!(ctx.word(ctx.end - 1), "libra" | "libras") =>
            {
                let money = amount.is_some_and(|a| a >= crate::dec::Dec::from_i64(1_000_000))
                    || near(&["precio", "cuesta", "costo", "pago", "pagaron", "pagado", "ingresos", "beneficio"]);
                money == matches!(unit, Unit::Currency(_))
            }
            _ => true,
        }
    }
}

/// A currency prefix lexeme, without constructing a second normalized string.
fn money_token(ctx: &Ctx<'_>, at: usize) -> bool {
    SPANISH_UNITS.lexicon().lookup(ctx.word(at)).iter().any(|e| {
        e.dim == CODE
            || e.dim == RAW
                && matches!(lexicon::WORDS[e.value as usize].unit, Unit::Currency(_))
                && e.flags & PREFIX != 0
    })
}
