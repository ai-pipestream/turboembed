//! Dates and periods: `3 de enero del 2019`, `27/11`, `31-dic-1994`,
//! `el día 18`, `viernes 6 de julio`, `el lunes que viene`, `marzo de
//! 2015`, `la semana pasada`, `el primer trimestre de 2017`, `los años
//! noventa`, `el año fiscal 2008`, holidays and seasons. Days come before
//! months in digits: `3/5/2018` is the third of May.

use turbo_analysis_layers::TimeGrain;

use crate::engine::{Ctx, Item, Kind, Rule, glued, glued_word, node, opt_word, word};
use crate::lang::en::time::clock::dig_len;
use crate::lang::en::time::date as en;
use crate::lang::en::time::lexicon::{self as words, rel};
use crate::lang::en::time::{date_of, emit_date, r, term, term_value, year_ref};
use crate::time::spec::{DateSpec, YearKind, YearRef};
use crate::value::Value;

use crate::lang::en::time::lexicon::{BLACK_FRIDAY, EARTH, INDEPENDENCE, PATRICK, VALENTINE, WORKERS};
use crate::lang::es::time::{
    ABBR, AMOUNT, ART, DATE, DAYWORD, DE, DIG, DOM, HOL, HOLIDAY, KIND, MD, MDAY, MONTH, MONTHD, PERIOD, PLURAL, QTR,
    RELW, SEASON, SEP, SHORT, UNIT, WD, WEEKDAY, YEAR,
};
use crate::lang::es::{INT, ORD, TENS};

mod builders;
use builders::*;

mod rules;
pub(super) use rules::RULES;
