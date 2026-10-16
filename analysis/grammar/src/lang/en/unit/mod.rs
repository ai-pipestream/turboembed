//! English numbers with units, built on the number rules: lengths, areas,
//! volumes, weights, speeds, information, angles, temperatures, ages and
//! money (`6 ft 2 in`, `120 km/h`, `2 GB`, `minus 40 degrees`, `a
//! 5-year-old`, `$3.5m`, `C$ 16.4 million`, `13 dollars and 45 cents`).
//!
//! A unit word's lexicon entry points into a table of [`lexicon`]; rules
//! put units together (`square`, `per`, `/`, `cubic`, `metric`, a
//! country before a currency) and a number before or after a unit makes
//! a quantity. All quantities go to one output, [`Output::Quantity`], so
//! one reading wins where a text allows several (`two pounds` is a weight,
//! `361 million pounds` money). A unit named with no amount is a mention
//! too (`convert to fahrenheit`), unless the word has a common other
//! sense. docs/grammar-units.md lists what the rules read.

mod lexicon;
mod rules;

use std::sync::OnceLock;

use crate::dec::Dec;
use crate::engine::{Ctx, Dim, Grammar, Kind, Lexicon, Node, NumeralStyle, Output, Rule};
use crate::unit::{Angle, CurrencyUnit, Generic, Temperature, Unit, Weight};
use crate::value::Value;

use super::{DIMS as NUMBER_DIMS, ENGLISH, NUMERAL};

// Word classes.
/// A unit of measure (`km`, `pounds`), by its index in [`lexicon::UNITS`].
pub(crate) const UWORD: Dim = Dim(NUMBER_DIMS);
/// A money word or sign (`$`, `euros`, `cents`), by its index in
/// [`lexicon::MONEY`].
pub(crate) const CURW: Dim = Dim(NUMBER_DIMS + 1);
/// A currency code (`usd`), by its index in the code table.
pub(crate) const CODE: Dim = Dim(NUMBER_DIMS + 2);
/// A code for millions of a currency (`musd`).
pub(crate) const MCODE: Dim = Dim(NUMBER_DIMS + 3);
/// A country word (`swiss`), by its first row in [`lexicon::NATIONS`].
pub(crate) const NATION: Dim = Dim(NUMBER_DIMS + 4);
/// An SI prefix written as a word (`kilo`), with its power of ten.
pub(crate) const SIPRE: Dim = Dim(NUMBER_DIMS + 5);
/// A time word, a [`Time`].
pub(crate) const TIMEW: Dim = Dim(NUMBER_DIMS + 6);
/// A temperature scale (`celsius`, `f`).
pub(crate) const TSCALE: Dim = Dim(NUMBER_DIMS + 7);
/// `degree`, `deg`, `°`.
pub(crate) const DEG: Dim = Dim(NUMBER_DIMS + 8);
/// A multiplier glued to an amount of money (`m`, `bn`), with its power
/// of ten.
pub(crate) const MULT: Dim = Dim(NUMBER_DIMS + 9);
/// `in` and `"` for inches.
pub(crate) const INCHW: Dim = Dim(NUMBER_DIMS + 10);
// Pieces and readings.
/// A unit of measure, perhaps of several words.
pub(crate) const UNIT: Dim = Dim(NUMBER_DIMS + 11);
/// A currency.
pub(crate) const CUR: Dim = Dim(NUMBER_DIMS + 12);
/// A currency that may come before its amount (`$`, `USD`, `C$`).
pub(crate) const CURP: Dim = Dim(NUMBER_DIMS + 13);
/// A measured quantity.
pub(crate) const QTY: Dim = Dim(NUMBER_DIMS + 14);
pub(crate) const MONEY: Dim = Dim(NUMBER_DIMS + 15);
pub(crate) const AGE: Dim = Dim(NUMBER_DIMS + 16);
const DIMS: u16 = NUMBER_DIMS + 17;

// Flags on unit words.
/// A word with a common other sense: a unit only after a number.
pub(crate) const AMBIG: u8 = 1;
/// A unit only in lowercase (`m`, not the `M` of a million).
pub(crate) const LOWER: u8 = 1 << 1;
/// An abbreviation that may take a period (`deg.`, `tsp.`).
pub(crate) const DOT: u8 = 1 << 2;
/// Money that may come before its amount.
pub(crate) const PREFIX: u8 = 1 << 3;
/// An information unit whose `b` says bits and `B` bytes.
pub(crate) const BITCASE: u8 = 1 << 4;
/// A unit written as a symbol or abbreviation.
pub(crate) const ABBR: u8 = 1 << 5;
/// Money only after a country word (`won`).
pub(crate) const NATION_ONLY: u8 = 1 << 6;
/// A country's short form that goes only before `$` (`C$`).
pub(crate) const ONLY_DOLLAR: u8 = 1 << 7;

/// A time word's meaning.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Time {
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

impl Time {
    pub(crate) fn of(v: i64) -> Time {
        use Time::*;
        [Millisecond, Second, Minute, Hour, Day, Week, Month, Year][v as usize]
    }
}

/// The English unit grammar: the number grammar's rules and words, and
/// the unit rules after them.
pub struct EnglishUnits {
    lexicon: OnceLock<Lexicon>,
}

/// The one instance; its lexicon and rules are put together on first use.
pub static ENGLISH_UNITS: EnglishUnits = EnglishUnits { lexicon: OnceLock::new() };

static RULES: OnceLock<Vec<Rule>> = OnceLock::new();

impl Grammar for EnglishUnits {
    fn name(&self) -> &'static str {
        "en-units"
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
            let mut all = ENGLISH.rules().to_vec();
            all.extend_from_slice(rules::RULES);
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
            UNIT | CUR | QTY | MONEY | AGE => Some(Output::Quantity),
            _ => ENGLISH.output(dim),
        }
    }

    fn keep(&self, output: Output, node: &Node, ctx: &Ctx<'_>) -> bool {
        match output {
            Output::Quantity => match node.value {
                Value::Unit(u) => self.bare(u, ctx),
                Value::Quantity(q) => self.stands(q.value(), q.unit(), ctx),
                _ => false,
            },
            _ => ENGLISH.keep(output, node, ctx),
        }
    }

    fn shadows(&self, ordinal: &Node) -> bool {
        ENGLISH.shadows(ordinal)
    }
}

/// Words near a bare `degrees` that make it an angle.
const ANGLE_CUES: &[&str] = &[
    "angle",
    "angled",
    "angles",
    "anticlockwise",
    "arc",
    "bank",
    "bend",
    "clockwise",
    "counterclockwise",
    "incline",
    "inclined",
    "latitude",
    "longitude",
    "pivot",
    "rotate",
    "rotated",
    "rotates",
    "rotating",
    "rotation",
    "slope",
    "spin",
    "tilt",
    "tilted",
    "turn",
    "turned",
    "turning",
    "turns",
];

/// Words near a bare `degrees` that make it a temperature, whatever else
/// is near.
const TEMPERATURE_CUES: &[&str] = &[
    "boil",
    "boiling",
    "boils",
    "cold",
    "colder",
    "cool",
    "fever",
    "freezing",
    "heat",
    "heated",
    "hot",
    "hotter",
    "oven",
    "temperature",
    "temperatures",
    "thermometer",
    "thermostat",
    "warm",
    "warmer",
    "weather",
];

/// Words near `pounds` that make it money.
const MONEY_CUES: &[&str] = &[
    "bid", "bought", "budget", "buy", "charge", "charged", "cost", "costing", "costs", "debt", "earn", "earned",
    "earns", "fee", "fees", "fine", "fined", "invest", "invested", "loan", "offer", "paid", "pay", "pays", "price",
    "priced", "prices", "profit", "profits", "raised", "rent", "revenue", "revenues", "salary", "sell", "sold",
    "spend", "spent", "wage", "wages", "worth",
];

impl EnglishUnits {
    /// Whether any word within a few lexemes of the match is in `cues`.
    fn near(&self, ctx: &Ctx<'_>, cues: &[&str], before: usize, after: usize) -> bool {
        let from = ctx.start.saturating_sub(before);
        let to = (ctx.end + after).min(ctx.lexed.len());
        (from..to).any(|i| ctx.lexeme(i).kind == Kind::Word && cues.contains(&ctx.word(i)))
    }

    /// The flags of lexeme `i`'s entries of category `dim`, or-ed.
    fn flags(&self, ctx: &Ctx<'_>, i: usize, dim: Dim) -> Option<u8> {
        let mut found = None;
        for e in self.lexicon().lookup(ctx.word(i)) {
            if e.dim == dim {
                found = Some(found.unwrap_or(0) | e.flags);
            }
        }
        found
    }

    /// Whether the token ending at lexeme `i`, its glued letters and
    /// digits, holds a word.
    fn word_in_token(&self, ctx: &Ctx<'_>, mut i: usize) -> bool {
        loop {
            let l = ctx.lexeme(i);
            if l.kind == Kind::Word {
                return true;
            }
            if l.kind != Kind::Numeral || !l.glued || i == 0 {
                return false;
            }
            i -= 1;
        }
    }

    /// Whether lexeme `i` is money that may come before an amount: `$`,
    /// `USD`.
    fn money_sign(&self, ctx: &Ctx<'_>, i: usize) -> bool {
        self.flags(ctx, i, CURW).is_some_and(|f| f & PREFIX != 0)
            || self.flags(ctx, i, CODE).is_some_and(|f| f & PREFIX != 0)
    }

    /// Whether a unit named with no amount is a mention: its last word is
    /// a unit with no other common sense, and `cent` does not follow
    /// `per`.
    fn bare(&self, unit: Unit, ctx: &Ctx<'_>) -> bool {
        let last = ctx.end - 1;
        if ctx.lexeme(last).kind != Kind::Word {
            return false;
        }
        let plain = |dim: Dim| self.flags(ctx, last, dim).is_some_and(|f| f & (AMBIG | NATION_ONLY) == 0);
        let named = plain(UWORD) || plain(CURW) || plain(CODE) || self.flags(ctx, last, TSCALE) == Some(0);
        let per_cent = matches!(unit, Unit::Currency(CurrencyUnit::Minor(_)))
            && ctx.before().is_some_and(|p| ctx.word(p) == "per");
        named && !per_cent
    }

    /// Whether a quantity stands in its context.
    fn stands(&self, value: Dec, unit: Unit, ctx: &Ctx<'_>) -> bool {
        let (first, last) = (ctx.start, ctx.end - 1);
        // Not part of a code: `SCP-173 m`, `scp2-173m`, `24703 L-202`.
        if let Some(p) = ctx.before()
            && p > 0
            && ctx.lexeme(first).glued
            && ctx.is_punct(p, '-')
            && ctx.lexeme(p).glued
            && self.word_in_token(ctx, p - 1)
        {
            return false;
        }
        // Nor the start of a hyphenated word or code: `3 t-shirts`.
        if let Some(a) = ctx.after()
            && a + 1 < ctx.lexed.len()
            && ctx.is_punct(a, '-')
            && ctx.lexeme(a).glued
            && ctx.lexeme(a + 1).glued
            && ctx.lexeme(a + 1).is_alnum()
            && self.flags(ctx, last, UWORD).is_some_and(|f| f & ABBR != 0)
        {
            return false;
        }
        match unit {
            Unit::Temperature(Temperature::Degree) => {
                self.near(ctx, TEMPERATURE_CUES, 5, 3) || !self.near(ctx, ANGLE_CUES, 5, 3)
            }
            Unit::Angle(Angle::Degree) => !self.near(ctx, TEMPERATURE_CUES, 5, 3) && self.near(ctx, ANGLE_CUES, 5, 3),
            // `pounds` is money in millions, before `sterling`, or near
            // a word about money unless `of` follows (`2 pounds of
            // apples`); a weight otherwise.
            Unit::Weight(Weight::Pound) | Unit::Currency(CurrencyUnit::Generic(Generic::Pound))
                if matches!(ctx.word(last), "pound" | "pounds") =>
            {
                let next = ctx.after().map(|a| ctx.word(a));
                let money = value.abs() >= Dec::from_i64(1_000_000)
                    || next == Some("sterling")
                    || (next != Some("of") && self.near(ctx, MONEY_CUES, 6, 4));
                money == matches!(unit, Unit::Currency(_))
            }
            // A singular `turn` counts one turn or less: `a half turn`,
            // not `2 turn signals`.
            Unit::Angle(Angle::Turn) if ctx.word(last) == "turn" => value <= Dec::ONE,
            Unit::Currency(_) => {
                // A sign glued to a number goes with it: `$10 dollars` is
                // `$10`, `3 USD50` is `USD50`.
                let sign_before = ctx.lexeme(first).kind == Kind::Numeral
                    && ctx.lexeme(first).glued
                    && ctx.before().is_some_and(|p| self.money_sign(ctx, p));
                let sign_after = ctx.after().is_some_and(|a| {
                    ctx.lexeme(a).glued && ctx.lexeme(a).kind == Kind::Numeral && self.money_sign(ctx, last)
                });
                !sign_before && !sign_after
            }
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests;
