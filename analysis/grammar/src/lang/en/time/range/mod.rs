//! Ranges and modified readings: `from 2014 to 2018`, `between 3 and 12
//! of Sept`, `5 to 6pm`, `Monday 8-9am`, `next 3 days`, `within 9
//! months`, `2 weeks starting May 20th`, `before 2018`, `since the end
//! of 1989`, `around 1pm`, `later this week`.

use turbo_analysis_layers::{Modifier, Portion, Relation, TimeGrain};

use crate::engine::{Ctx, Item, Kind, Rule, node, opt_word, word};
use crate::time::TimeValue;
use crate::time::duration::{Duration, MILLI};
use crate::time::spec::{Clock, DateSpec, Edge, Meridiem, MonthRef, Point, Range, Shown, TimeOfDay, YearRef};
use crate::value::Value;

use crate::lang::en::time::date::{month_of, unit_grain};
use crate::lang::en::time::duration::{duration_of, length};
use crate::lang::en::time::lexicon::rel;
use crate::lang::en::time::{
    AMOUNT, BARE, CLOCK, DATE, DPOINT, DT, DTP, DUR, HOL, LRANGE, MDAY, MODED, MONTH, MONTHD, PARTD, PERIOD, RANGE,
    RELW, TO, TPOINT, TRANGE, UNIT, YEAR, any_point, clock_of, emit, r, term, tval, year_ref,
};

mod bounds;
pub(crate) use bounds::*;

mod clocks;
pub(crate) use clocks::*;

mod modifiers;
pub(crate) use modifiers::*;

mod rules;
pub(crate) use rules::*;
