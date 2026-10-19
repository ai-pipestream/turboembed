//! Spanish dates, times, durations, ranges and sets, built on the Spanish
//! number rules: `3 de enero del 2019`, `mañana a las 8:00 a.m.`, `de las
//! 5 a las 6 de la tarde`, `los próximos tres días`, `cada lunes a las 4
//! p.m.`, `a finales de 2000`.
//!
//! The categories, the values and most builders are those of the English
//! time grammar ([`crate::lang::en::time`]): only the words and their
//! order are Spanish. docs/grammar.md lists what the rules read.

mod clock;
mod compose;
mod date;
mod duration;
mod lexicon;
mod range;
mod set;

use std::sync::OnceLock;

use crate::engine::{Ctx, Dim, Grammar, Kind, Lexicon, Node, NumeralStyle, Output, Rule, WordForms};

pub(crate) use crate::lang::en::time::{ABBR, AMBIG, PLURAL, SHORT, TONIGHT};

use super::{DIMS as NUMBER_DIMS, NUMERAL, SPANISH};

// Word classes.
pub(crate) const MONTH: Dim = Dim(NUMBER_DIMS);
pub(crate) const WEEKDAY: Dim = Dim(NUMBER_DIMS + 1);
pub(crate) const UNIT: Dim = Dim(NUMBER_DIMS + 2);
pub(crate) const PART: Dim = Dim(NUMBER_DIMS + 3);
pub(crate) const DAYWORD: Dim = Dim(NUMBER_DIMS + 4);
pub(crate) const RELW: Dim = Dim(NUMBER_DIMS + 5);
pub(crate) const SEASON: Dim = Dim(NUMBER_DIMS + 6);
pub(crate) const HOLIDAY: Dim = Dim(NUMBER_DIMS + 7);
pub(crate) const SETWORD: Dim = Dim(NUMBER_DIMS + 8);
/// `de`, `del`.
pub(crate) const DE: Dim = Dim(NUMBER_DIMS + 9);
/// `el`, `la`, `los`, `las`.
pub(crate) const ART: Dim = Dim(NUMBER_DIMS + 10);
// Pieces, as in the English grammar.
pub(crate) const AMPM: Dim = Dim(NUMBER_DIMS + 11);
pub(crate) const DIG: Dim = Dim(NUMBER_DIMS + 12);
pub(crate) const YEAR: Dim = Dim(NUMBER_DIMS + 13);
pub(crate) const MDAY: Dim = Dim(NUMBER_DIMS + 14);
/// An hour that is a time only after `las` or `la`: `a las 5`.
pub(crate) const BARE: Dim = Dim(NUMBER_DIMS + 15);
pub(crate) const TO: Dim = Dim(NUMBER_DIMS + 16);
pub(crate) const QTR: Dim = Dim(NUMBER_DIMS + 17);
pub(crate) const MD: Dim = Dim(NUMBER_DIMS + 18);
pub(crate) const WD: Dim = Dim(NUMBER_DIMS + 19);
pub(crate) const DOM: Dim = Dim(NUMBER_DIMS + 20);
pub(crate) const AMOUNT: Dim = Dim(NUMBER_DIMS + 21);
pub(crate) const LRANGE: Dim = Dim(NUMBER_DIMS + 22);
pub(crate) const DPOINT: Dim = Dim(NUMBER_DIMS + 23);
pub(crate) const TPOINT: Dim = Dim(NUMBER_DIMS + 24);
/// A part of the day said after a clock time: `de la tarde`.
pub(crate) const OF_PART: Dim = Dim(NUMBER_DIMS + 25);
// Readings.
pub(crate) const CLOCK: Dim = Dim(NUMBER_DIMS + 26);
pub(crate) const PARTD: Dim = Dim(NUMBER_DIMS + 27);
pub(crate) const DATE: Dim = Dim(NUMBER_DIMS + 28);
pub(crate) const HOL: Dim = Dim(NUMBER_DIMS + 29);
pub(crate) const MONTHD: Dim = Dim(NUMBER_DIMS + 30);
pub(crate) const PERIOD: Dim = Dim(NUMBER_DIMS + 31);
pub(crate) const DT: Dim = Dim(NUMBER_DIMS + 32);
pub(crate) const DTP: Dim = Dim(NUMBER_DIMS + 33);
pub(crate) const DUR: Dim = Dim(NUMBER_DIMS + 34);
pub(crate) const TRANGE: Dim = Dim(NUMBER_DIMS + 35);
pub(crate) const RANGE: Dim = Dim(NUMBER_DIMS + 36);
pub(crate) const SET: Dim = Dim(NUMBER_DIMS + 37);
pub(crate) const MODED: Dim = Dim(NUMBER_DIMS + 38);
pub(crate) const SEP: Dim = Dim(NUMBER_DIMS + 39);
/// The kind of a year: `fiscal`, `escolar`, `calendario`.
pub(crate) const KIND: Dim = Dim(NUMBER_DIMS + 40);
/// A `DPOINT` or `TPOINT` that may take an article: a bound after `de`,
/// `entre` or a connector, as in `entre el 1 de diciembre y el 4 de
/// febrero`.
pub(crate) const ADPOINT: Dim = Dim(NUMBER_DIMS + 41);
pub(crate) const ATPOINT: Dim = Dim(NUMBER_DIMS + 42);
const DIMS: u16 = NUMBER_DIMS + 43;

/// The Spanish time grammar.
pub struct SpanishTimes {
    lexicon: OnceLock<Lexicon>,
}

/// The one instance; its lexicon and rules are put together on first use.
pub static SPANISH_TIMES: SpanishTimes = SpanishTimes { lexicon: OnceLock::new() };

static RULES: OnceLock<Vec<Rule>> = OnceLock::new();

impl WordForms for SpanishTimes {
    fn folds(&self) -> bool {
        true
    }

    fn split(&self, word: &str, cuts: &mut Vec<u32>) {
        SPANISH.split(word, cuts);
    }
}

impl Grammar for SpanishTimes {
    fn name(&self) -> &'static str {
        "es-times"
    }

    fn lexicon(&self) -> &Lexicon {
        self.lexicon.get_or_init(|| {
            let mut entries = super::lexicon::ENTRIES.to_vec();
            entries.extend_from_slice(lexicon::ENTRIES);
            Lexicon::new(&entries)
        })
    }

    fn rules(&self) -> &'static [Rule] {
        RULES.get_or_init(|| {
            let mut all = SPANISH.rules().to_vec();
            for part in [clock::RULES, date::RULES, compose::RULES, duration::RULES, set::RULES, range::RULES] {
                all.extend_from_slice(part);
            }
            all
        })
    }

    fn dims(&self) -> u16 {
        DIMS
    }

    fn numeral(&self) -> Dim {
        NUMERAL
    }

    fn style(&self) -> NumeralStyle {
        SPANISH.style()
    }

    fn words(&self) -> &dyn WordForms {
        self
    }

    fn output(&self, dim: Dim) -> Option<Output> {
        match dim {
            BARE | CLOCK | PARTD | DATE | HOL | MONTHD | PERIOD | DT | DTP | DUR | TRANGE | RANGE | SET | MODED
            | WD | DOM | MD => Some(Output::Time),
            _ => None,
        }
    }

    fn keep(&self, output: Output, node: &Node, ctx: &Ctx<'_>) -> bool {
        output == Output::Time && keep(self.rules(), node, ctx)
    }
}

/// Whether the lexeme before the match says a clock time follows: `a las
/// 5`, `de la una`.
pub(crate) fn after_article(ctx: &Ctx<'_>) -> bool {
    ctx.before().is_some_and(|p| matches!(ctx.word(p), "las" | "la"))
}

fn rule_name(rules: &[Rule], node: &Node) -> &'static str {
    rules.get(usize::from(node.rule)).map_or("", |r| r.name)
}

/// Words that make an ambiguous month or weekday a time: `hasta mar`,
/// but `el mar` is the sea.
const CUES: &[&str] = &["desde", "hasta", "este", "proximo", "cada"];

/// Determiners before which `mañana` is the morning, not tomorrow: `la
/// mañana fue fría`.
const MORNING_CUES: &[&str] = &["la", "una", "esta", "esa", "cada", "toda", "buena", "misma"];

fn starts_ambiguous(ctx: &Ctx<'_>, dim: Dim) -> bool {
    SPANISH_TIMES.lexicon().lookup(ctx.word(ctx.start)).iter().any(|e| e.dim == dim && e.flags & AMBIG != 0)
}

/// Whether a candidate stands as a mention in its context.
fn keep(rules: &[Rule], node: &Node, ctx: &Ctx<'_>) -> bool {
    let before = ctx.before().map(|p| ctx.word(p));
    let after = ctx.after().map(|a| ctx.word(a));
    // Not inside a run of numbers joined by hyphens or slashes: a phone
    // number, a code, an identifier.
    let joined = |p: usize, q: usize| {
        let (a, b) = (ctx.lexeme(p), ctx.lexeme(q));
        b.glued && matches!(ctx.punct(if a.kind == Kind::Punct { p } else { q }), Some('-' | '/'))
    };
    if let Some(p) = ctx.before()
        && p > 0
        && ctx.lexeme(p).kind == Kind::Punct
        && joined(p - 1, p)
        && ctx.lexeme(ctx.start).glued
        && ctx.lexeme(p - 1).kind == Kind::Numeral
    {
        return false;
    }
    if let Some(a) = ctx.after()
        && a + 1 < ctx.lexed.len()
        && ctx.lexeme(a).kind == Kind::Punct
        && ctx.lexeme(a).glued
        && joined(a, a + 1)
        && ctx.lexeme(a + 1).kind == Kind::Numeral
    {
        return false;
    }
    match rule_name(rules, node) {
        _ if node.dim == BARE => after_article(ctx) && !matches!(after, Some("%" | "por")),
        // `el día 18`, `el 4.º`.
        "dom_ordinal" | "date_dom" => before == Some("el"),
        "monthd_month" | "wd" | "date_wd" => {
            !starts_ambiguous(ctx, MONTH) && !starts_ambiguous(ctx, WEEKDAY)
                || before.is_some_and(|b| CUES.contains(&b))
        }
        "date_dayword" => !(ctx.word(ctx.start) == "mañana" && before.is_some_and(|b| MORNING_CUES.contains(&b))),
        // `de la tarde` sets a clock's half of the day; alone it is no
        // mention.
        "partd_of" => false,
        // `el año anterior a la revolución` relates the year to what
        // follows.
        "period_rel_after" | "period_art_rel_after" => !matches!(after, Some("a" | "al" | "de" | "del")),
        // `el año`, `la semana`: the reference's own.
        "period_the_unit" => matches!(before, Some("el" | "la" | "del")),
        // `y después de las 6` starts another mention.
        "moded_y_despues" | "moded_o_despues" => !matches!(after, Some("de" | "del")),
        _ => true,
    }
}

#[cfg(test)]
mod tests;
