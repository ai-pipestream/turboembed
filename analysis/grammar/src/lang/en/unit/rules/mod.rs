//! The English unit rules and the functions that build their values.
//!
//! Children arrive in item order with word items left out.

use crate::dec::Dec;
use crate::engine::{Build, Ctx, Dim, Item, Kind, Rule, glued, glued_if, glued_word, node, node_if, opt_word, word};
use crate::lang::en::{NUM, NUMERAL, SCALE};
use crate::unit::currency::{CODES, code};
use crate::unit::{
    Age, Angle, Area, CurrencyUnit, Generic, Information, Length, Minor, Per, Speed, Temperature, Unit, Volume, Weight,
};
use crate::value::{Num, NumKind, Quantity, Value, flag};

use crate::lang::en::unit::lexicon::{MONEY as MONEY_WORDS, UNITS, nation, nation_rows, scale};
use crate::lang::en::unit::{
    ABBR, AGE, AMBIG, BITCASE, CODE, CUR, CURP, CURW, DEG, DOT, INCHW, LOWER, MCODE, MONEY, MULT, NATION, NATION_ONLY,
    ONLY_DOLLAR, PREFIX, QTY, SIPRE, TIMEW, TSCALE, Time, UNIT, UWORD,
};

mod predicates;
pub(crate) use predicates::*;

mod measures;
pub(crate) use measures::*;

mod money;
pub(crate) use money::*;

mod amounts;
pub(crate) use amounts::*;

mod table;
pub(crate) use table::*;
