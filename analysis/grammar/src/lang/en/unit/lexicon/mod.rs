//! The English words for units, after the unit lists of Recognizers-Text
//! and common usage.
//!
//! A unit word's lexicon entry carries its index in [`UNITS`] or
//! [`MONEY`]; a code's its index in [`crate::unit::currency::CODES`]; a
//! country's the index of its first row in [`NATIONS`].

use crate::engine::{Dim, Entry};
use crate::unit::currency::{CODES, code};
use crate::unit::{
    Angle, Area, CurrencyUnit, Generic, Information, Length, Minor, Speed, Temperature, Unit, Volume, Weight,
};

use crate::lang::en::unit::{
    ABBR, AMBIG, BITCASE, CODE, CURW, DEG, DOT, INCHW, LOWER, MCODE, MULT, NATION, NATION_ONLY, ONLY_DOLLAR, PREFIX,
    SIPRE, TIMEW, TSCALE, UWORD,
};

mod measures;
use measures::*;
pub(in crate::lang) use measures::{UNITS, Word};

mod money;
pub(in crate::lang) use money::{MILLION_CODES, MONEY, ambiguous_code};

mod nations;
pub(in crate::lang) use nations::{NATIONS, nation, nation_rows};

mod entries;
pub(in crate::lang) use entries::{entries, scale};
