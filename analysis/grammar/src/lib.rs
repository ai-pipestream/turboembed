//! A rule grammar for numbers, ordinals and percentages written in words
//! or digits, with exact values (`one hundred and three and two thirds`,
//! `1,674.5 million`, `twenty-first`, `per cent of 210`), for amounts in
//! units and money (`6 ft 2 in`, `$3.5m`), and for dates,
//! times, durations and sets (`tomorrow at 8am`, `from May to Oct 2020`,
//! `for 3 hours`, `every Monday`).
//!
//! - [`engine`] is language-free: a lexer over the text view, a numeral
//!   reader, rules over categories, and a chart parser that ranks the
//!   readings and picks non-overlapping mentions.
//! - [`lang`] holds the languages, each a [`Grammar`]: [`ENGLISH`] for
//!   numbers and [`ENGLISH_TIMES`], which adds the time rules.
//! - [`time`] is the language-free form of a time value, its TIMEX3 text
//!   and its resolution against a reference date-time.
//! - [`unit`] types the units of measure and of money that
//!   [`ENGLISH_UNITS`] reads after a number (`6 ft 2 in`, `$3.5m`).
//! - [`GrammarAnnotator`] and [`TimeAnnotator`] write what a grammar finds
//!   as layers.
//!
//! Values are [`Dec`] decimals, exact for what a text can state and cut
//! at 34 significant digits for a quotient such as `one third`.
//! docs/grammar.md describes the rules and the conformance tests.

pub mod annotator;
pub mod dec;
pub mod engine;
pub mod lang;
pub mod time;
pub mod times;
pub mod unit;
pub mod value;

pub use annotator::{Found, GRAMMAR, GRAMMAR_MONEY, GRAMMAR_NUMBERS, GRAMMAR_PERCENTS, GrammarAnnotator};
pub use dec::Dec;
pub use engine::{Grammar, Mention, Output, Parser};
pub use lang::en::unit::{ENGLISH_UNITS, EnglishUnits};
pub use lang::en::{ENGLISH, ENGLISH_TIMES, English, EnglishTimes};
pub use lang::es::unit::{SPANISH_UNITS, SpanishUnits};
pub use lang::es::{SPANISH, SPANISH_TIMES, Spanish, SpanishTimes};
pub use lang::nl::{DUTCH, DUTCH_TIMES, Dutch, DutchTimes};
pub use times::TimeAnnotator;
pub use unit::{CurrencyUnit, Measure, Unit};
pub use value::{Num, NumKind, Ordinal, Quantity, Relative, Value, flag};
