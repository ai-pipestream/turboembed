//! A rule grammar for numbers, ordinals and percentages written in words
//! or digits, with exact values: `one hundred and three and two thirds`,
//! `1,674.5 million`, `minus 5`, `twenty-first`, `the second to last`,
//! `per cent of 210`.
//!
//! - [`engine`] is language-free: a lexer over the text view, a numeral
//!   reader, rules over categories, and a chart parser that ranks the
//!   readings and picks non-overlapping mentions.
//! - [`lang`] holds the languages, each a [`Grammar`]: [`ENGLISH`].
//! - [`GrammarAnnotator`] writes what a grammar finds as layers.
//!
//! Values are [`Dec`] decimals, exact for what a text can state and cut
//! at 34 significant digits for a quotient such as `one third`.
//! docs/grammar.md describes the rules and the conformance tests.

pub mod annotator;
pub mod dec;
pub mod engine;
pub mod lang;
pub mod time;
pub mod value;

pub use annotator::{Found, GRAMMAR, GRAMMAR_NUMBERS, GRAMMAR_PERCENTS, GrammarAnnotator};
pub use dec::Dec;
pub use engine::{Grammar, Mention, Output, Parser};
pub use lang::en::{ENGLISH, ENGLISH_TIMES, English, EnglishTimes};
pub use value::{Num, NumKind, Ordinal, Relative, Value, flag};
