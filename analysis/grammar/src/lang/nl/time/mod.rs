//! Dutch dates, times, durations, ranges and sets, built on the Dutch
//! number rules: `22 april`, `morgen om 8:00`, `half acht`, `kwart voor
//! drie`, `van 5 tot 6 's middags`, `maandagmiddag`, `de komende drie
//! dagen`, `over twee weken`, `elke maandag om 16.00`, `eind van de
//! maand`.
//!
//! The categories, the values and most builders are those of the English
//! time grammar ([`crate::lang::en::time`]): the words, their order and
//! the readings Dutch has of its own (`half acht` is 7:30, a date puts
//! the day first) are Dutch. The lexer cuts a day and a part of it
//! written as one word (`morgenavond`, `vanmiddag`) into glued words.
//! docs/grammar.md lists what the rules read.

mod clock;
mod compose;
mod date;
mod duration;
mod lexicon;
mod range;
mod set;

use std::sync::OnceLock;

use crate::engine::{Ctx, Dim, Grammar, Kind, Lexicon, Node, NumeralStyle, Output, Rule, WordForms};

pub(crate) use crate::lang::en::time::{ABBR, AMBIG, PLURAL, SHORT};

use super::{DIMS as NUMBER_DIMS, DUTCH, NUMERAL};

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
// Pieces, as in the English grammar.
pub(crate) const DIG: Dim = Dim(NUMBER_DIMS + 9);
pub(crate) const YEAR: Dim = Dim(NUMBER_DIMS + 10);
pub(crate) const MDAY: Dim = Dim(NUMBER_DIMS + 11);
/// An hour, perhaps with minutes, that is a time only where the text
/// around it says so: `om 5`, `5 tot 6 's middags`.
pub(crate) const BARE: Dim = Dim(NUMBER_DIMS + 12);
pub(crate) const TO: Dim = Dim(NUMBER_DIMS + 13);
pub(crate) const QTR: Dim = Dim(NUMBER_DIMS + 14);
pub(crate) const MD: Dim = Dim(NUMBER_DIMS + 15);
pub(crate) const WD: Dim = Dim(NUMBER_DIMS + 16);
pub(crate) const DOM: Dim = Dim(NUMBER_DIMS + 17);
pub(crate) const AMOUNT: Dim = Dim(NUMBER_DIMS + 18);
pub(crate) const LRANGE: Dim = Dim(NUMBER_DIMS + 19);
pub(crate) const DPOINT: Dim = Dim(NUMBER_DIMS + 20);
pub(crate) const TPOINT: Dim = Dim(NUMBER_DIMS + 21);
/// A part of the day said after a clock time: `'s middags`, `in de
/// avond`.
pub(crate) const OF_PART: Dim = Dim(NUMBER_DIMS + 22);
/// Half an hour before an hour: `half acht` is 7:30.
pub(crate) const HALF: Dim = Dim(NUMBER_DIMS + 23);
// Readings.
pub(crate) const CLOCK: Dim = Dim(NUMBER_DIMS + 24);
pub(crate) const PARTD: Dim = Dim(NUMBER_DIMS + 25);
pub(crate) const DATE: Dim = Dim(NUMBER_DIMS + 26);
pub(crate) const HOL: Dim = Dim(NUMBER_DIMS + 27);
pub(crate) const MONTHD: Dim = Dim(NUMBER_DIMS + 28);
pub(crate) const PERIOD: Dim = Dim(NUMBER_DIMS + 29);
pub(crate) const DT: Dim = Dim(NUMBER_DIMS + 30);
pub(crate) const DTP: Dim = Dim(NUMBER_DIMS + 31);
pub(crate) const DUR: Dim = Dim(NUMBER_DIMS + 32);
pub(crate) const TRANGE: Dim = Dim(NUMBER_DIMS + 33);
pub(crate) const RANGE: Dim = Dim(NUMBER_DIMS + 34);
pub(crate) const SET: Dim = Dim(NUMBER_DIMS + 35);
pub(crate) const MODED: Dim = Dim(NUMBER_DIMS + 36);
pub(crate) const SEP: Dim = Dim(NUMBER_DIMS + 37);
/// `'s`, with a straight or a curly apostrophe.
pub(crate) const APOS: Dim = Dim(NUMBER_DIMS + 38);
const DIMS: u16 = NUMBER_DIMS + 39;

/// The `-s` form of a part of the day, said after `'s`: `'s avonds`.
pub(crate) const GENITIVE: u8 = 1 << 6;

/// The Dutch time grammar.
pub struct DutchTimes {
    lexicon: OnceLock<Lexicon>,
}

/// The one instance; its lexicon and rules are put together on first use.
pub static DUTCH_TIMES: DutchTimes = DutchTimes { lexicon: OnceLock::new() };

static RULES: OnceLock<Vec<Rule>> = OnceLock::new();

impl WordForms for DutchTimes {
    fn folds(&self) -> bool {
        true
    }

    fn split(&self, word: &str, cuts: &mut Vec<u32>) {
        DUTCH.split(word, cuts);
        if cuts.is_empty() {
            split_day_part(self.lexicon(), word, cuts);
        }
    }
}

/// Cuts a day and a part of it written as one word: `morgenavond`,
/// `maandagmiddag`, `vanochtend`, `zondagmorgen`.
fn split_day_part(lexicon: &Lexicon, word: &str, cuts: &mut Vec<u32>) {
    if !word.bytes().all(|b| b.is_ascii_lowercase()) || !lexicon.lookup(word).is_empty() {
        return;
    }
    let day = |w: &str| {
        w == "van" || lexicon.lookup(w).iter().any(|e| (e.dim == WEEKDAY || e.dim == DAYWORD) && e.flags & PLURAL == 0)
    };
    let part = |w: &str| w == "morgen" || lexicon.lookup(w).iter().any(|e| e.dim == PART && e.flags == 0);
    if let Some(i) = (2..word.len().saturating_sub(3)).find(|&i| day(&word[..i]) && part(&word[i..])) {
        cuts.push(i as u32);
    }
}

impl Grammar for DutchTimes {
    fn name(&self) -> &'static str {
        "nl-times"
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
            let mut all = DUTCH.rules().to_vec();
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
        DUTCH.style()
    }

    fn words(&self) -> &dyn WordForms {
        self
    }

    fn notation(&self) -> turbo_analysis_layers::NumberNotation {
        DUTCH.notation()
    }

    fn output(&self, dim: Dim) -> Option<Output> {
        match dim {
            BARE | CLOCK | PARTD | DATE | HOL | MONTHD | PERIOD | DT | DTP | DUR | TRANGE | RANGE | SET | MODED
            | WD | DOM | MD => Some(Output::Time),
            _ => None,
        }
    }

    fn time_zone_context(&self, dim: Dim) -> bool {
        dim == LRANGE
    }

    fn keep(&self, output: Output, node: &Node, ctx: &Ctx<'_>) -> bool {
        output == Output::Time && keep(self.rules(), node, ctx)
    }
}

/// Words before a clock time said as a bare hour: `om 5`, `rond 11 uur`.
const CLOCK_CUES: &[&str] = &["om", "rond", "omstreeks", "tegen"];

/// Whether a bare hour opens a range of hours that ends in `uur`:
/// `tussen 3 en 5 uur`, `van 9 tot 11 uur`.
fn opens_hour_range(ctx: &Ctx<'_>) -> bool {
    let w = |i: usize| (i < ctx.lexed.len()).then(|| ctx.word(i));
    ctx.before().is_some_and(|b| matches!(ctx.word(b), "tussen" | "van"))
        && matches!(w(ctx.end), Some("en" | "tot"))
        && ctx.end + 1 < ctx.lexed.len()
        && ctx.lexeme(ctx.end + 1).kind == Kind::Numeral
        && w(ctx.end + 2) == Some("uur")
}

/// Whether an hour in `uur` closes such a range.
fn closes_hour_range(ctx: &Ctx<'_>) -> bool {
    ctx.start >= 3
        && matches!(ctx.word(ctx.start - 1), "en" | "tot")
        && ctx.lexeme(ctx.start - 2).kind == Kind::Numeral
        && matches!(ctx.word(ctx.start - 3), "tussen" | "van")
}

/// Words that make an ambiguous month or weekday (`zo`, `di`, `jan`) a
/// time.
const CUES: &[&str] = &[
    "op",
    "van",
    "vanaf",
    "tot",
    "tussen",
    "sinds",
    "na",
    "voor",
    "in",
    "elke",
    "iedere",
    "deze",
    "volgende",
    "vorige",
    "komende",
    "afgelopen",
    "aanstaande",
];

/// Whether the first word of the mention is a word with another common
/// sense, as `zo` and `jan` are.
fn starts_ambiguous(ctx: &Ctx<'_>, dim: Dim) -> bool {
    DUTCH_TIMES.lexicon().lookup(ctx.word(ctx.start)).iter().any(|e| e.dim == dim && e.flags & AMBIG != 0)
}

/// Whether a mention ends in an abbreviation's period with the sentence
/// going on after it: `zat. om 5 uur`, not `... in de problemen zat.`.
fn abbreviation_dot(ctx: &Ctx<'_>) -> bool {
    ctx.is_punct(ctx.end - 1, '.') && ctx.after().is_some_and(|a| ctx.lexeme(a).kind == Kind::Word)
}

fn rule_name(rules: &[Rule], node: &Node) -> &'static str {
    rules.get(usize::from(node.rule)).map_or("", |r| r.name)
}

fn is_currency(c: char) -> bool {
    turbo_unicode::category::general_category(c) == turbo_unicode::category::GeneralCategory::Sc
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
    // Not money: `$12 december`.
    let start = ctx.lexeme(ctx.start);
    if (start.glued || start.kind == Kind::Numeral) && ctx.before().and_then(|p| ctx.punct(p)).is_some_and(is_currency)
    {
        return false;
    }
    let not_amount = !matches!(after, Some("%" | "procent" | "per" | "euro" | "dollar" | "cent" | "€" | "$"));
    match rule_name(rules, node) {
        // `om 5`, `om 6.45`, never `6.45 procent`.
        _ if node.dim == BARE => {
            (before.is_some_and(|b| CLOCK_CUES.contains(&b)) || opens_hour_range(ctx)) && not_amount
        }
        // `om 9 uur of 10 uur`: a list of clock times goes on.
        "clock_uur" => {
            before.is_some_and(|b| CLOCK_CUES.contains(&b))
                || matches!(before, Some("of" | "en")) && ctx.start >= 2 && ctx.word(ctx.start - 2) == "uur"
                || closes_hour_range(ctx)
        }
        "clock_dot" | "clock_om_dot" => not_amount,
        // `op de 18e`, `op 29e`.
        "dom_the" | "dom_ordinal" | "date_dom" => before == Some("op"),
        "monthd_month" | "wd" | "wd_dot" | "date_wd" => {
            !starts_ambiguous(ctx, MONTH) && !starts_ambiguous(ctx, WEEKDAY)
                || before.is_some_and(|b| CUES.contains(&b))
                || (rule_name(rules, node) == "monthd_month"
                    && after.is_some_and(|a| DUTCH_TIMES.lexicon().lookup(a).iter().any(|e| e.dim == MONTH)))
                || abbreviation_dot(ctx)
        }
        "moded_before" => {
            !matches!(before, Some("plan" | "meeting" | "boeking" | "reservering" | "afspraak"))
                || matches!(ctx.word(ctx.start + 1), "dec" | "volgende")
        }
        "moded_until" => before != Some("terug"),
        "moded_the_start" => before != Some("aan"),
        "dt_clock_date" => !(ctx.word(ctx.start) == "om" && ctx.word(ctx.end - 1) == "vandaag"),
        // A meal alone is no time: `lunch eten`.
        "partd" => !matches!(ctx.word(ctx.start), "lunch" | "diner" | "ontbijt" | "avondeten" | "middageten"),
        // `30 minuten later`, but `later deze week`.
        "dt_later" => !matches!(after, Some("deze" | "dit" | "volgende" | "vandaag" | "morgen")),
        // `2018 of later`, a bound that ends the clause.
        "moded_or_later" | "moded_and_later" | "moded_and_after" | "moded_and_higher" => {
            ctx.after().is_none_or(|a| ctx.lexeme(a).kind == Kind::Punct)
        }
        // `de week van 10 april` is the week of a day, `de week 31` the
        // week numbered.
        "period_the_unit" => after != Some("van") && ctx.after().is_none_or(|a| ctx.lexeme(a).kind != Kind::Numeral),
        // A year in words needs `in`: `in tweeduizend twintig`.
        "period_year_spelled" => before == Some("in"),
        _ => true,
    }
}

#[cfg(test)]
mod tests;
