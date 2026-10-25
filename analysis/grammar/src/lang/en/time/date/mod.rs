//! Dates and periods: `May twenty nine`, `22/04`, `2016-11-07`,
//! `Friday`, `next Monday`, `the 18th`, `April 2017`, `this week`, `Q3
//! 2019`, `the 1990s`, `the last week of July`, holidays and seasons.

use turbo_analysis_layers::{Modifier, Portion, TimeGrain};

use crate::engine::{Ctx, Item, Kind, Rule, glued, glued_word, node, opt_word, word};
use crate::time::TimeValue;
use crate::time::spec::{DateSpec, Holiday, MonthRef, Point, Season, YearKind, YearRef};
use crate::value::Value;

use crate::lang::en::time::clock::{dig_len, dig_zero};
use crate::lang::en::time::lexicon::{self, rel};
use crate::lang::en::time::{
    ABBR, AMOUNT, DATE, DAYWORD, DECADEW, DIG, DOM, HOL, HOLIDAY, MD, MDAY, MONTH, MONTHD, ORDINAL, PERIOD, PLURAL,
    QTR, RELW, SEASON, SEP, SHORT, SPELLED, TONIGHT, UNIT, WD, WEEKDAY, YEAR, date_of, emit, emit_date, r, term,
    term_value, year_ref,
};
use crate::lang::en::{INT, NUMERAL, ORD, ORD_SUFFIX};

mod calendar;
pub(in crate::lang) use calendar::{
    b_d_mon_y, b_d_y_mon, b_dmy, b_md_day_month, b_md_dot, b_md_month_day, b_md_month_digits, b_md_slash, b_md_year,
    b_mday_digits, b_mday_ordinal, b_mday_spelled, b_mday_suffix, b_mdy, b_mon_d_y, b_mon_y_d, b_same, b_weekday_date,
    b_weekday_dot, b_y_d_mon, b_y_mon_d, b_year_digits, b_year_pairs, b_year_short, b_year_spelled, b_ymd,
    b_ymd_spaced, b_yyyymmdd, cal, full_year, md, numeric_year, small, spelled, year_digits,
};

mod weekdays;
pub(in crate::lang) use weekdays::{
    b_dayword, b_dom, b_dom_ordinal, b_last_weekday, b_month, b_month_sep_year, b_month_year, b_my_digits, b_nth_month,
    b_nth_week_weekday, b_nth_weekday, b_period_year, b_period_year_spelled, b_rel_month, b_rel_month_day, b_wd,
    b_wd_day, b_wd_of_week, b_wd_rel, b_wd_the_day, b_wd_this_rel, b_week_wd, b_year_month, b_year_word, b_ym_digits,
    day_rel, month_of, nth, weekday, year_of,
};

mod periods;
pub(in crate::lang) use periods::{
    b_century, b_century_decade, b_decade, b_decade_word, b_last_week, b_nth_quarter_of, b_nth_quarter_the_year,
    b_nth_week, b_period_rel, b_period_the, b_qtr, b_quarter, b_quarter_year, b_undefined, b_week_after_next, b_week_n,
    b_weekend, b_workweek, b_year_half, b_year_quarter, b_year_slice, b_years_quarter, quarter, unit_grain,
};

mod named;
pub(in crate::lang) use named::{
    b_calendar_year_glued, b_holiday, b_holiday_near, b_holiday_of_year, b_holiday_weekend, b_holiday_weekend_year,
    b_holiday_year, b_other_year, b_other_year_glued, b_other_year_n, b_rel_calendar_year, b_rel_other_year,
    b_rel_season, b_season, b_season_year, b_sep, b_week_of, hol,
};

mod rules;
pub(super) use rules::RULES;
