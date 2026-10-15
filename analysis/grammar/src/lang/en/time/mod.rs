//! English dates, times, durations, ranges and sets, built on the number
//! rules: `May twenty nine`, `tomorrow at 8am`, `from 5 to 6pm of April
//! 22`, `3.5 years`, `every Monday at 4pm`, `before the end of 2018`.
//!
//! A time node's value is a [`TimeValue`] kept in the parse's
//! [`crate::engine::Interned`] table; resolution against a reference date
//! happens after selection ([`crate::time::resolve`]).
//!
//! The categories go from words (`MONTH`, `WEEKDAY`) through pieces that
//! are no mention on their own (`MDAY`, `BARE`) to the readings the
//! output takes: a day (`DATE`), a month (`MONTHD`), a longer period
//! (`PERIOD`), a time of day (`CLOCK`, `PARTD`), both (`DT`, `DTP`), a
//! duration, a range, a set or a modified reading. docs/grammar.md lists
//! what they read.

pub(crate) mod clock;
pub(crate) mod compose;
pub(crate) mod date;
pub(crate) mod duration;
pub(crate) mod lexicon;
pub(crate) mod range;
pub(crate) mod set;

use std::sync::OnceLock;

use turbo_analysis_layers::Modifier;

use crate::engine::{Ctx, Dim, Grammar, Kind, Lexicon, Node, NumeralStyle, Output, Rule};
use crate::time::TimeValue;
use crate::time::spec::{Clock, DateSpec, Point, TimeOfDay, YearRef};
use crate::value::Value;

use super::{DIMS as NUMBER_DIMS, ENGLISH, NUMERAL};

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
pub(crate) const DECADEW: Dim = Dim(NUMBER_DIMS + 9);
// Pieces.
/// `am`, `p.m.`: 1 for the morning, 2 for the afternoon, 3 for either.
pub(crate) const AMPM: Dim = Dim(NUMBER_DIMS + 10);
/// An integer numeral written in plain digits, with their count.
pub(crate) const DIG: Dim = Dim(NUMBER_DIMS + 11);
pub(crate) const YEAR: Dim = Dim(NUMBER_DIMS + 12);
/// A day of the month, 1 to 31.
pub(crate) const MDAY: Dim = Dim(NUMBER_DIMS + 13);
/// An hour, perhaps with minutes, that is a time only where the text
/// around it says so: `at 5`, `5 to 6pm`.
pub(crate) const BARE: Dim = Dim(NUMBER_DIMS + 14);
/// A range connector: `to`, `-`, `till`, `through`.
pub(crate) const TO: Dim = Dim(NUMBER_DIMS + 15);
/// A quarter number: `q3`.
pub(crate) const QTR: Dim = Dim(NUMBER_DIMS + 16);
/// A month and a day of an open year.
pub(crate) const MD: Dim = Dim(NUMBER_DIMS + 17);
/// A weekday, alone or placed: `Friday`, `next Monday`, `Monday 21`.
pub(crate) const WD: Dim = Dim(NUMBER_DIMS + 18);
/// A day of an open month: `the 18th`.
pub(crate) const DOM: Dim = Dim(NUMBER_DIMS + 19);
/// A count for a duration, in thousandths: `3`, `a few`, `an`.
pub(crate) const AMOUNT: Dim = Dim(NUMBER_DIMS + 20);
/// A clock range that needs a date or a part of the day to stand:
/// `3-8`, `five to seven`.
pub(crate) const LRANGE: Dim = Dim(NUMBER_DIMS + 21);
/// Any reading that names a date or a period, for modifiers and ranges.
pub(crate) const DPOINT: Dim = Dim(NUMBER_DIMS + 22);
/// A clock time or a date with one, for modifiers.
pub(crate) const TPOINT: Dim = Dim(NUMBER_DIMS + 23);
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
/// A separator inside a written date: `/`, `-`.
pub(crate) const SEP: Dim = Dim(NUMBER_DIMS + 37);
const DIMS: u16 = NUMBER_DIMS + 38;

// Flags on lexicon terms and pieces.
/// An abbreviation (`Sept`), which may take a period.
pub(crate) const ABBR: u8 = 1;
/// A word with a common other sense (`may`, `sun`), a time only beside
/// words that make it one.
pub(crate) const AMBIG: u8 = 1 << 1;
/// A plural (`Mondays`, `mornings`).
pub(crate) const PLURAL: u8 = 1 << 2;
/// A year written in two digits; a day of the month written as an
/// ordinal.
pub(crate) const SHORT: u8 = 1 << 3;
pub(crate) const ORDINAL: u8 = 1 << 3;
/// Written in words.
pub(crate) const SPELLED: u8 = 1 << 4;
/// `tonight`: today's night.
pub(crate) const TONIGHT: u8 = 1 << 5;

/// The English time grammar.
pub struct EnglishTimes {
    lexicon: OnceLock<Lexicon>,
}

/// The one instance; its lexicon and rules are put together on first use.
pub static ENGLISH_TIMES: EnglishTimes = EnglishTimes { lexicon: OnceLock::new() };

static RULES: OnceLock<Vec<Rule>> = OnceLock::new();

impl Grammar for EnglishTimes {
    fn name(&self) -> &'static str {
        "en-times"
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
            let mut all = ENGLISH.rules().to_vec();
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
        ENGLISH.style()
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

/// Words that make a bare `may` a month: `in May`, `since May`.
const MONTH_CUES: &[&str] = &[
    "in", "of", "since", "until", "till", "from", "to", "by", "before", "after", "during", "through", "early", "late",
    "mid", "next", "last",
];

/// Words after which `this may` is a month: the end of a clause.
fn ends_clause(ctx: &Ctx<'_>) -> bool {
    match ctx.after() {
        None => true,
        Some(a) => ctx.lexeme(a).kind != Kind::Word || matches!(ctx.word(a), "to" | "and" | "or" | "through" | "until"),
    }
}

/// Whether the first word of the mention is a word with another common
/// sense, as `may` and `sun` are.
fn starts_ambiguous(ctx: &Ctx<'_>, dim: Dim) -> bool {
    ENGLISH_TIMES.lexicon().lookup(ctx.word(ctx.start)).iter().any(|e| e.dim == dim && e.flags & AMBIG != 0)
}

/// Words before which `last` is a verb: `will last`, `shouldn't last`.
const VERB_CUES: &[&str] = &[
    "will", "would", "should", "shall", "can", "could", "may", "might", "must", "to", "t", "not", "does", "did", "do",
    "it", "they", "which", "that",
];

fn rule_name(rules: &[Rule], node: &Node) -> &'static str {
    rules.get(usize::from(node.rule)).map_or("", |r| r.name)
}

/// Whether a candidate stands as a mention in its context.
fn keep(rules: &[Rule], node: &Node, ctx: &Ctx<'_>) -> bool {
    let before = ctx.before().map(|p| ctx.word(p));
    let after = ctx.after().map(|a| ctx.word(a));
    // Not inside a run of numbers joined by hyphens or slashes: a phone
    // number, a code, an identifier (`138-2010-2015`).
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
    // Not money: `$12 December 27`.
    let start = ctx.lexeme(ctx.start);
    if (start.glued || start.kind == Kind::Numeral) && ctx.before().and_then(|p| ctx.punct(p)).is_some_and(is_currency)
    {
        return false;
    }
    match rule_name(rules, node) {
        // `at 6`, `at 6.45`, never `6.45 percent`.
        _ if node.dim == BARE => before == Some("at") && !matches!(after, Some("%" | "percent" | "per")),
        // `on the 18th`, `on 29th`.
        "dom_the" | "dom_ordinal" | "date_dom" => before == Some("on"),
        // `on 5.12`.
        "md_dot" | "date_md_dot" => before == Some("on"),
        // `in May`, not `May I`.
        "monthd_month" => !starts_ambiguous(ctx, MONTH) || before.is_some_and(|b| MONTH_CUES.contains(&b)),
        // `this May`, but not `this may take longer`.
        "monthd_rel" => {
            let ambiguous = ENGLISH_TIMES
                .lexicon()
                .lookup(ctx.word(ctx.start + 1))
                .iter()
                .any(|e| e.dim == MONTH && e.flags & AMBIG != 0);
            !ambiguous || ends_clause(ctx)
        }
        // `on Sun`, not `the sun`.
        "wd" | "date_wd" => !starts_ambiguous(ctx, WEEKDAY) || !matches!(before, Some("the" | "a")),
        // `lunch` alone is a meal, `at lunch` a time.
        "partd" => before != Some("good") && !matches!(ctx.word(ctx.start), "lunch" | "dinner" | "breakfast"),
        // `in the year 2008` names `year 2008`; `before the year 2019`
        // takes the article.
        "period_the_year_word" => false,
        // `the late eighties` stands only after a word such as `by`; alone
        // the mention starts at `late`.
        "moded_the_early" | "moded_the_mid" | "moded_the_late" => false,
        // `at 6.45`, not `at 6.45 percent`.
        "clock_at_dot" | "clock_at_dot_ampm" => !matches!(after, Some("%" | "percent" | "per")),
        // `30 minutes later`, but `30 minutes` and `later this week`.
        "dt_later" => !matches!(after, Some("this" | "next" | "last" | "in" | "today" | "tomorrow")),
        // `2018 or later`, a bound that ends the clause.
        "moded_or_later" | "moded_and_later" | "moded_and_after" | "moded_or_after" | "moded_and_greater" => {
            ctx.after().is_none_or(|a| ctx.lexeme(a).kind == Kind::Punct)
        }
        // `at the end of 2000` is `the end of 2000` without its article.
        "moded_end_of_the" | "moded_beginning_of_the" | "moded_middle_of_the" | "moded_start_of_the" => {
            !matches!(before, Some("at" | "from"))
        }
        // `will last two hours` is a duration.
        "range_next_n" | "range_next_dur" | "range_next_unit" if ctx.word(ctx.start) == "last" => {
            !before.is_some_and(|b| VERB_CUES.contains(&b))
        }
        // `the year of 2008` is the year 2008.
        "period_the_unit" => after != Some("of"),
        // A year in words needs `in`: `in nineteen ninety six`.
        "period_year_spelled" => before == Some("in"),
        _ => true,
    }
}

fn is_currency(c: char) -> bool {
    turbo_unicode::category::general_category(c) == turbo_unicode::category::GeneralCategory::Sc
}

// Reading children. A rule's items fix each child's category, and each
// category holds one kind of value.

pub(crate) fn term(v: &Value) -> (i64, u8) {
    v.term().unwrap_or((0, 0))
}

pub(crate) fn tval(c: &Ctx<'_>, v: &Value) -> Option<TimeValue> {
    c.time_of(v)
}

/// A child's point when it carries no modifier.
pub(crate) fn plain(c: &Ctx<'_>, v: &Value) -> Option<Point> {
    match tval(c, v)? {
        TimeValue::Point { point, modifier, clip: false } if modifier.is_none() => Some(point),
        _ => None,
    }
}

/// A child's point, modifier or not.
pub(crate) fn any_point(c: &Ctx<'_>, v: &Value) -> Option<(Point, Modifier)> {
    match tval(c, v)? {
        TimeValue::Point { point, modifier, .. } => Some((point, modifier)),
        _ => None,
    }
}

pub(crate) fn clock_of(c: &Ctx<'_>, v: &Value) -> Option<Clock> {
    match plain(c, v)? {
        Point { date: DateSpec::None, time: TimeOfDay::Clock(k), shift } if shift.is_zero() => Some(k),
        _ => None,
    }
}

pub(crate) fn date_of(c: &Ctx<'_>, v: &Value) -> Option<DateSpec> {
    match plain(c, v)? {
        Point { date, time: TimeOfDay::None, shift } if shift.is_zero() && date != DateSpec::None => Some(date),
        _ => None,
    }
}

pub(crate) fn emit(c: &Ctx<'_>, t: TimeValue) -> Option<Value> {
    c.time(t)
}

pub(crate) fn emit_point(c: &Ctx<'_>, p: Point) -> Option<Value> {
    emit(c, TimeValue::point(p))
}

pub(crate) fn emit_date(c: &Ctx<'_>, d: DateSpec) -> Option<Value> {
    emit_point(c, Point::date(d))
}

pub(crate) fn emit_clock(c: &Ctx<'_>, k: Clock) -> Option<Value> {
    emit_point(c, Point::time(TimeOfDay::Clock(k)))
}

pub(crate) fn term_value(value: i64, flags: u8) -> Option<Value> {
    Some(Value::Term { value, flags })
}

/// The year a `YEAR` term names: two digits read as Recognizers-Text
/// reads them, 30 to 39 left open, below that this century, above it the
/// last.
pub(crate) fn year_ref(v: &Value) -> YearRef {
    let (y, f) = term(v);
    if f & SHORT == 0 {
        return YearRef::Abs(y as i16);
    }
    match y {
        30..=39 => YearRef::Short(y as u8),
        0..=29 => YearRef::Abs(2000 + y as i16),
        _ => YearRef::Abs(1900 + y as i16),
    }
}

/// A rule from its parts.
pub(crate) const fn r(
    name: &'static str,
    dim: Dim,
    weight: i16,
    items: &'static [crate::engine::Item],
    build: crate::engine::Build,
) -> Rule {
    Rule { name, dim, items, weight, build }
}

#[cfg(test)]
mod tests;
