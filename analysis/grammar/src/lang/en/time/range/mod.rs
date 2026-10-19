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
pub(in crate::lang) use bounds::{
    b_days_of, b_edge_point, b_empty, b_month_days, b_point, b_range, b_tpoint, days_of, open_day,
};

mod clocks;
pub(in crate::lang) use clocks::{
    b_date_trange, b_for_from_clock, b_from_clock_for, b_lrange, b_trange, b_trange_date, clock_length, clock_range,
    clocks, day_and_part, stands,
};

mod modifiers;
pub(in crate::lang) use modifiers::{
    align_of, b_after, b_after_around, b_approx, b_before, b_before_around, b_date_moded, b_earlier, b_earlier_in_the,
    b_end, b_end_of_day, b_end_of_unit, b_eoy, b_for, b_later, b_later_in_the, b_less_after, b_mid, b_more_after,
    b_more_before, b_n_next, b_next_dur, b_next_n, b_next_unit, b_since, b_since_after, b_start, b_until, b_within,
    direction, next, this_period,
};

mod rules;
pub(super) use rules::RULES;
