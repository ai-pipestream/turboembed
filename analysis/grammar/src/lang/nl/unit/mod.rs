//! Dutch quantities and currencies, composed with the Dutch number grammar.
mod lexicon;
mod rules;

use super::{DIMS, DUTCH, NUMERAL};
use crate::engine::{Ctx, Dim, Grammar, Kind, Lexicon, Node, NumeralStyle, Output, Rule, WordForms};
use crate::lang::unit_pack::{self, *};
use crate::unit::{Angle, Temperature, Unit};
use crate::value::Value;
use std::sync::OnceLock;

const _: () = assert!(DIMS == NUMBER_DIMS, "the unit dims follow the number grammar's");

/// Dutch measures, money, temperatures and ages, with original source offsets.
pub struct DutchUnits {
    lexicon: OnceLock<Lexicon>,
}
pub static DUTCH_UNITS: DutchUnits = DutchUnits { lexicon: OnceLock::new() };
static RULES: OnceLock<Vec<Rule>> = OnceLock::new();

impl Grammar for DutchUnits {
    fn name(&self) -> &'static str {
        "nl-units"
    }
    fn lexicon(&self) -> &Lexicon {
        self.lexicon.get_or_init(|| {
            let mut entries = super::lexicon::ENTRIES.to_vec();
            entries.extend(unit_pack::entries::<lexicon::Words>());
            Lexicon::new(&entries)
        })
    }
    fn rules(&self) -> &'static [Rule] {
        RULES.get_or_init(|| {
            let mut rules = DUTCH.rules().to_vec();
            rules.extend_from_slice(rules::RULES);
            rules
        })
    }
    fn dims(&self) -> u16 {
        UNIT_DIMS
    }
    fn numeral(&self) -> Dim {
        NUMERAL
    }
    fn style(&self) -> NumeralStyle {
        DUTCH.style()
    }
    fn words(&self) -> &dyn WordForms {
        DUTCH.words()
    }
    fn notation(&self) -> turbo_analysis_layers::NumberNotation {
        DUTCH.notation()
    }
    fn output(&self, dim: Dim) -> Option<Output> {
        if matches!(dim, UNIT | QTY) { Some(Output::Quantity) } else { DUTCH.output(dim) }
    }
    fn shadows(&self, ordinal: &Node) -> bool {
        DUTCH.shadows(ordinal)
    }
    fn keep(&self, output: Output, node: &Node, ctx: &Ctx<'_>) -> bool {
        if output != Output::Quantity {
            return DUTCH.keep(output, node, ctx);
        }
        let (unit, amount) = match node.value {
            Value::Unit(u) => (u, None),
            Value::Quantity(q) => (q.unit(), Some(q.value())),
            _ => return false,
        };
        if amount.is_none()
            && !matches!(unit, Unit::Speed(_) | Unit::Area(_))
            && !lexicon::WORDS.iter().any(|w| w.unit == unit && w.flags & BARE != 0 && w.word == ctx.word(ctx.end - 1))
        {
            return false;
        }
        let near = |words: &[&str]| {
            (ctx.start.saturating_sub(5)..(ctx.end + 3).min(ctx.lexed.len())).any(|i| words.contains(&ctx.word(i)))
        };
        if matches!(unit, Unit::Age(_))
            && ctx.before().is_some_and(|p| {
                ctx.word(p) == "min" || ctx.is_punct(p, '-') && (p == 0 || ctx.lexeme(p - 1).kind != Kind::Numeral)
            })
        {
            return false;
        }
        if matches!(unit, Unit::Temperature(Temperature::Degree) | Unit::Angle(Angle::Degree))
            && ctx.raw(ctx.end - 1) == "º"
            && !near(&["temperatuur", "warmte", "verwarm", "thermostaat"])
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
                let angle = near(&["draai", "draaien", "hoek", "roteer", "rotatie", "slagen"])
                    && !near(&["temperatuur", "verwarm", "warmte", "thermostaat", "kookt", "koud"]);
                angle == matches!(unit, Unit::Angle(_))
            }
            _ => true,
        }
    }
}

/// A currency prefix lexeme, without constructing a second normalized string.
fn money_token(ctx: &Ctx<'_>, at: usize) -> bool {
    DUTCH_UNITS.lexicon().lookup(ctx.word(at)).iter().any(|e| {
        e.dim == CODE
            || matches!(ctx.word(at), "$" | "€" | "£" | "元")
            || e.dim == RAW
                && matches!(lexicon::WORDS[e.value as usize].unit, Unit::Currency(_))
                && e.flags & PREFIX != 0
    })
}
