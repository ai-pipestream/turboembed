//! The days a [`DateSpec`] names, given the reference date-time: one
//! period, or two when the text leaves the year (or month, or week) open,
//! the last one before the reference and the first one after it, as
//! Recognizers-Text gives them.

use turbo_analysis_layers::{Date, DateTime, TimeGrain};

use super::super::calendar::{
    holiday, iso_week_start, monday, month_end, month_start, month_week_start, nth_weekday, quarter_start,
    weekday_of_week,
};
use super::super::duration::{Duration, Unit};
use super::super::spec::{DateSpec, HolidayOrRel, MonthRef, YearRef};
use super::super::timex::{TimexDate, TimexYear};

/// How far a search for a matching year or month goes.
const SEARCH: i32 = 40;

/// The TIMEX3 value of a period: a date form, or a range for the
/// periods TIMEX3 writes that way (`(2019-07-01,2019-10-01,P3M)`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DateTimex {
    Date(TimexDate),
    Range(TimexDate, TimexDate, Duration),
}

/// A period of days, `start` included and `end` not, or none when the
/// text names a day that does not exist (`Feb 30`) or a period
/// Recognizers-Text leaves unresolved (a season).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dated {
    pub timex: DateTimex,
    pub span: Option<(Date, Date)>,
}

/// The past and the future reading; the same period twice when the text
/// fixes it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub past: Dated,
    pub future: Dated,
}

impl Choice {
    fn one(d: Dated) -> Choice {
        Choice { past: d, future: d }
    }

    /// The distinct readings, past first.
    pub fn readings(&self) -> impl Iterator<Item = (bool, Dated)> {
        let two = self.past != self.future;
        [(false, self.past), (true, self.future)].into_iter().filter(move |&(future, _)| future || two)
    }
}

fn day(d: Date, timex: TimexDate) -> Dated {
    Dated { timex: DateTimex::Date(timex), span: Some((d, d.plus_days(1).unwrap_or(d))) }
}

fn concrete_day(d: Date) -> Dated {
    day(d, TimexDate::day(d))
}

fn unresolved(timex: TimexDate) -> Dated {
    Dated { timex: DateTimex::Date(timex), span: None }
}

fn month_period(year: i32, month: u8, timex_year: TimexYear) -> Option<Dated> {
    Some(Dated {
        timex: DateTimex::Date(TimexDate::month(timex_year, month)),
        span: Some((month_start(year, month)?, month_end(year, month)?)),
    })
}

fn year_period(year: i32) -> Option<Dated> {
    Some(Dated {
        timex: DateTimex::Date(TimexDate::year(year)),
        span: Some((Date::new(year, 1, 1)?, Date::new(year.checked_add(1)?, 1, 1)?)),
    })
}

/// A period TIMEX3 writes as a range of days: `start` for `months`
/// months, or `years` years, with the years written as `year` says.
fn range_period(start: Date, length: Duration, timex_year: Option<TimexYear>) -> Option<Dated> {
    let end = plus(start, &length)?;
    let ymd = |d: Date| TimexDate::Ymd {
        year: match timex_year {
            Some(TimexYear::Short(yy)) => {
                TimexYear::Short((i64::from(yy) + i64::from(d.year()) - i64::from(start.year())).rem_euclid(100) as u8)
            }
            Some(y) => y,
            None => TimexYear::Year(d.year()),
        },
        month: Some(d.month()),
        day: Some(d.day()),
    };
    Some(Dated { timex: DateTimex::Range(ymd(start), ymd(end), length), span: Some((start, end)) })
}

/// `d` moved by a whole-unit duration.
pub fn plus(d: Date, length: &Duration) -> Option<Date> {
    let mut out = d;
    for &(unit, milli) in length.parts() {
        if milli % 1000 != 0 {
            return None;
        }
        let n = i64::from(milli / 1000);
        out = match unit {
            Unit::Year => out.plus_years(n)?,
            Unit::Month => out.plus_months(n)?,
            Unit::Week | Unit::Weekend => out.plus_weeks(n)?,
            Unit::Day | Unit::BusinessDay | Unit::Weekday => out.plus_days(n)?,
            Unit::Hour | Unit::Minute | Unit::Second => return None,
        };
    }
    Some(out)
}

/// The month `n` months from `year`-`month`.
fn month_plus(year: i32, month: u8, n: i32) -> Option<(i32, u8)> {
    let count = i64::from(year) * 12 + i64::from(month) - 1 + i64::from(n);
    let y = i32::try_from(count.div_euclid(12)).ok()?;
    let m = (count.rem_euclid(12) + 1) as u8;
    Some((y, m))
}

/// The years the past and the future readings of a month of an open year
/// fall in: a month before the reference's is this year's in the past and
/// next year's in the future; the reference's own month and later ones
/// are this year's in the future and last year's in the past.
fn month_years(month: u8, now: Date) -> Option<(i32, i32)> {
    let y = now.year();
    if month < now.month() { Some((y, y.checked_add(1)?)) } else { Some((y.checked_sub(1)?, y)) }
}

fn months(year: YearRef, month: MonthRef, now: Date) -> Option<([(i32, u8); 2], TimexYear)> {
    match month {
        MonthRef::Rel(n) => {
            let (y, m) = month_plus(now.year(), now.month(), i32::from(n))?;
            Some(([(y, m), (y, m)], TimexYear::Year(y)))
        }
        MonthRef::Abs(m) => match year.year(now.year()) {
            Some(y) => Some(([(y, m), (y, m)], TimexYear::Year(y))),
            None if year == YearRef::Any => {
                let (p, f) = month_years(m, now)?;
                Some(([(p, m), (f, m)], TimexYear::Any))
            }
            None => None,
        },
    }
}

/// A day of a month and year the text leaves open, whose past reading is
/// the last on or before the reference and whose future reading the
/// first on or after it.
fn month_day(month: u8, d: u8, now: DateTime) -> Choice {
    let timex = TimexDate::Ymd { year: TimexYear::Any, month: Some(month), day: Some(d) };
    let y = now.date().year();
    let at = |y: i32| Date::new(y, month, d);
    let past =
        (0..SEARCH).filter_map(|k| y.checked_sub(k)).find_map(|y| at(y).filter(|&x| DateTime::midnight(x) <= now));
    let future =
        (0..SEARCH).filter_map(|k| y.checked_add(k)).find_map(|y| at(y).filter(|&x| DateTime::midnight(x) >= now));
    match (past, future) {
        (Some(p), Some(f)) => Choice { past: day(p, timex), future: day(f, timex) },
        _ => Choice::one(unresolved(timex)),
    }
}

/// A day of the month with the month open: `the 18th`.
fn day_of_month(d: u8, now: DateTime) -> Choice {
    let timex = TimexDate::Ymd { year: TimexYear::Any, month: None, day: Some(d) };
    let (y, m) = (now.date().year(), now.date().month());
    let at = |k: i32| {
        let (y, m) = month_plus(y, m, k)?;
        Date::new(y, m, d)
    };
    let past = (0..SEARCH).find_map(|k| at(-k).filter(|&x| DateTime::midnight(x) <= now));
    let future = (0..SEARCH).find_map(|k| at(k).filter(|&x| DateTime::midnight(x) >= now));
    match (past, future) {
        (Some(p), Some(f)) => Choice { past: day(p, timex), future: day(f, timex) },
        _ => Choice::one(unresolved(timex)),
    }
}

/// A weekday of an open week: this week's, and a week before or after it
/// for the reading on the other side of the reference day.
fn weekday(wd: u8, now: Date) -> Option<Choice> {
    let this = weekday_of_week(now, wd)?;
    let future = if this >= now { this } else { this.plus_days(7)? };
    let past = if this < now { this } else { this.plus_days(-7)? };
    let timex = TimexDate::Weekday(wd);
    Some(Choice { past: day(past, timex), future: day(future, timex) })
}

/// A weekday on a day of the month: the nearest months whose day `d`
/// falls on `wd`, before and after the reference.
fn weekday_day(wd: u8, d: u8, this_month: bool, now: Date) -> Option<Choice> {
    let (y, m) = (now.year(), now.month());
    if this_month {
        return Date::new(y, m, d).map(|x| Choice::one(concrete_day(x)));
    }
    let at = |k: i32| {
        let (y, m) = month_plus(y, m, k)?;
        Date::new(y, m, d).filter(|x| x.day_of_week() == wd)
    };
    let past = (0..SEARCH).find_map(|k| at(-k).filter(|&x| x <= now))?;
    let future = (0..SEARCH).find_map(|k| at(k).filter(|&x| x >= now))?;
    if past == future {
        return Some(Choice::one(concrete_day(past)));
    }
    let timex = TimexDate::Weekday(wd);
    Some(Choice { past: day(past, timex), future: day(future, timex) })
}

fn cal(year: YearRef, month: u8, d: u8, wd: u8, now: DateTime) -> Option<Choice> {
    let today = now.date();
    if let Some(y) = year.year(today.year()) {
        let ty = TimexYear::Year(y);
        return Some(Choice::one(match (month, d) {
            (0, 0) if wd == 0 => year_period(y)?,
            (0, _) => return None,
            (m, 0) => month_period(y, m, ty)?,
            (m, d) => match Date::new(y, m, d) {
                Some(x) => concrete_day(x),
                None => unresolved(TimexDate::Ymd { year: ty, month: Some(m), day: Some(d) }),
            },
        }));
    }
    if let YearRef::Short(yy) = year {
        let ty = TimexYear::Short(yy);
        let at = |century: i32| -> Option<Dated> {
            let y = century + i32::from(yy);
            Some(match (month, d) {
                (0, _) => return None,
                (m, 0) => month_period(y, m, ty)?,
                (m, d) => day(Date::new(y, m, d)?, TimexDate::Ymd { year: ty, month: Some(m), day: Some(d) }),
            })
        };
        return Some(Choice { past: at(1900)?, future: at(2000)? });
    }
    match (month, d, wd) {
        (m, d, _) if m > 0 && d > 0 => Some(month_day(m, d, now)),
        (m, 0, _) if m > 0 => {
            let (p, f) = month_years(m, today)?;
            Some(Choice { past: month_period(p, m, TimexYear::Any)?, future: month_period(f, m, TimexYear::Any)? })
        }
        (0, d, 0) if d > 0 => Some(day_of_month(d, now)),
        (0, d, wd) if d > 0 => weekday_day(wd, d, false, today),
        (0, 0, wd) if wd > 0 => weekday(wd, today),
        _ => None,
    }
}

/// The period of `grain` holding `now`, moved by `n`.
pub fn relative(grain: TimeGrain, n: i32, now: Date) -> Option<Dated> {
    let n64 = i64::from(n);
    match grain {
        TimeGrain::Day => Some(concrete_day(now.plus_days(n64)?)),
        TimeGrain::Week => {
            let start = monday(now)?.plus_weeks(n64)?;
            let (year, week) = start.iso_week();
            Some(Dated {
                timex: DateTimex::Date(TimexDate::Week { year, week }),
                span: Some((start, start.plus_weeks(1)?)),
            })
        }
        TimeGrain::Weekend => {
            let start = monday(now)?.plus_weeks(n64)?.plus_days(5)?;
            let (year, week) = start.iso_week();
            Some(Dated {
                timex: DateTimex::Date(TimexDate::Weekend { year, week }),
                span: Some((start, start.plus_days(2)?)),
            })
        }
        TimeGrain::Month => {
            let (y, m) = month_plus(now.year(), now.month(), n)?;
            month_period(y, m, TimexYear::Year(y))
        }
        TimeGrain::Quarter => {
            let start = quarter_start(now.year(), now.quarter())?.plus_months(3 * n64)?;
            range_period(start, Duration::units(Unit::Month, 3)?, None)
        }
        TimeGrain::Half => {
            let start = month_start(now.year(), if now.month() <= 6 { 1 } else { 7 })?.plus_months(6 * n64)?;
            range_period(start, Duration::units(Unit::Month, 6)?, None)
        }
        TimeGrain::Year => year_period(now.year().checked_add(n)?),
        TimeGrain::Decade => {
            let year = i32::try_from(i64::from(now.year().div_euclid(10)) * 10 + 10 * n64).ok()?;
            let start = Date::new(year, 1, 1)?;
            range_period(start, Duration::units(Unit::Year, 10)?, None)
        }
        _ => None,
    }
}

/// The readings of `spec`, or none when the spec cannot name a period.
pub fn dates(spec: DateSpec, now: DateTime) -> Option<Choice> {
    let today = now.date();
    let ry = today.year();
    match spec {
        DateSpec::None => None,
        DateSpec::Cal { year, month, day, weekday } => cal(year, month, day, weekday, now),
        DateSpec::WeekdayDay { weekday, day, this_month } => weekday_day(weekday, day, this_month, today),
        DateSpec::Rel { grain, n } => relative(grain, i32::from(n), today).map(Choice::one),
        DateSpec::Fortnight { n } => {
            let start = monday(today)?.plus_weeks(2 * i64::from(n))?;
            range_period(start, Duration::units(Unit::Week, 2)?, None).map(Choice::one)
        }
        DateSpec::WeekdayRel { weekday, weeks } => {
            let d = monday(today)?.plus_weeks(i64::from(weeks))?.plus_days(i64::from(weekday) - 1)?;
            Some(Choice::one(concrete_day(d)))
        }
        DateSpec::WeekdayPast { weekday } => {
            let back = (i64::from(today.day_of_week()) - i64::from(weekday) - 1).rem_euclid(7) + 1;
            Some(Choice::one(concrete_day(today.plus_days(-back)?)))
        }
        DateSpec::Week { year, week } => {
            let y = match year {
                YearRef::Any => ry,
                _ => year.year(ry)?,
            };
            let start = iso_week_start(y, week)?;
            let (year, week) = start.iso_week();
            Some(Choice::one(Dated {
                timex: DateTimex::Date(TimexDate::Week { year, week }),
                span: Some((start, start.plus_weeks(1)?)),
            }))
        }
        DateSpec::MonthWeek { year, month, week } => {
            let (ms, ty) = months(year, month, today)?;
            let at = |(y, m): (i32, u8)| -> Option<Dated> {
                let start = month_week_start(y, m, week)?;
                let timex = TimexDate::MonthWeek { year: ty, month: m, week: if week == 0 { 5 } else { week } };
                Some(Dated { timex: DateTimex::Date(timex), span: Some((start, start.plus_weeks(1)?)) })
            };
            Some(Choice { past: at(ms[0])?, future: at(ms[1])? })
        }
        DateSpec::NthWeekday { year, month, weekday, n } => {
            let (ms, _) = months(year, month, today)?;
            let timex = TimexDate::NthWeekday { month: ms[0].1, weekday, n };
            let at = |(y, m): (i32, u8)| Some(day(nth_weekday(y, m, weekday, n)?, timex));
            Some(Choice { past: at(ms[0])?, future: at(ms[1])? })
        }
        DateSpec::RelMonthDay { months, day } => {
            let (y, m) = month_plus(ry, today.month(), i32::from(months))?;
            Some(Choice::one(match Date::new(y, m, day) {
                Some(d) => concrete_day(d),
                None => unresolved(TimexDate::Ymd { year: TimexYear::Year(y), month: Some(m), day: Some(day) }),
            }))
        }
        DateSpec::WeekOf { month, day } => {
            let c = cal(YearRef::Any, month, day, 0, now)?;
            let week = |d: Dated| -> Option<Dated> {
                let start = monday(d.span?.0)?;
                Some(Dated { timex: d.timex, span: Some((start, start.plus_weeks(1)?)) })
            };
            Some(Choice { past: week(c.past)?, future: week(c.future)? })
        }
        DateSpec::Quarter { year, quarter } => {
            (1..=4).contains(&quarter).then(|| periods(year, (quarter - 1) * 3 + 1, 3, now))?
        }
        DateSpec::Half { year, half } => (1..=2).contains(&half).then(|| periods(year, (half - 1) * 6 + 1, 6, now))?,
        DateSpec::Season { year, season } => {
            Some(Choice::one(unresolved(TimexDate::Season { year: year.year(ry), season })))
        }
        DateSpec::OtherYear { kind, year } => {
            Some(Choice::one(unresolved(TimexDate::OtherYear { kind, year: year.year(ry) })))
        }
        DateSpec::Decade { year } => {
            let ten = Duration::units(Unit::Year, 10)?;
            match year {
                YearRef::Short(yy) => {
                    let base = i32::try_from(i64::from(ry.div_euclid(100)) * 100 + i64::from(yy)).ok()?;
                    let first = Date::new(base, 1, 1)?;
                    let (past, future) = if first.year() > ry {
                        (first.plus_years(-100)?, first)
                    } else {
                        (first, first.plus_years(100)?)
                    };
                    let ty = Some(TimexYear::Short(yy));
                    Some(Choice { past: range_period(past, ten, ty)?, future: range_period(future, ten, ty)? })
                }
                y => range_period(Date::new(y.year(ry)?, 1, 1)?, ten, None).map(Choice::one),
            }
        }
        DateSpec::Century { n } => {
            let start = Date::new((i32::from(n) - 1) * 100, 1, 1)?;
            range_period(start, Duration::units(Unit::Year, 100)?, None).map(Choice::one)
        }
        DateSpec::Holiday { holiday: h, year } => match year.year(ry) {
            Some(y) => Some(Choice::one(concrete_day(holiday(h, y)?))),
            None => {
                let this = holiday(h, ry)?;
                let past = if this <= today { this } else { holiday(h, ry.checked_sub(1)?)? };
                let future = if this >= today { this } else { holiday(h, ry.checked_add(1)?)? };
                let timex = TimexDate::Ymd { year: TimexYear::Any, month: Some(past.month()), day: Some(past.day()) };
                Some(Choice { past: day(past, timex), future: day(future, timex) })
            }
        },
        DateSpec::HolidayNear { holiday: h, future } => {
            let this = holiday(h, ry)?;
            let d = if (future && this > today) || (!future && this < today) {
                this
            } else {
                holiday(h, ry.checked_add(if future { 1 } else { -1 })?)?
            };
            Some(Choice::one(concrete_day(d)))
        }
        DateSpec::Undefined { grain } => {
            let mut d = relative(grain, 0, today)?;
            d.timex = DateTimex::Date(TimexDate::Pattern(match grain {
                TimeGrain::Week => "XXXX-WXX",
                TimeGrain::Weekend => "XXXX-WXX-WE",
                TimeGrain::Month => "XXXX-XX",
                _ => "XXXX",
            }));
            Some(Choice::one(d))
        }
        DateSpec::YearToDate => Some(Choice::one(Dated {
            timex: DateTimex::Date(TimexDate::year(ry)),
            span: Some((Date::new(ry, 1, 1)?, today)),
        })),
        DateSpec::CurrentDate => Some(Choice::one(day(today, TimexDate::Present))),
        DateSpec::WorkWeek { n } => {
            let start = monday(today)?.plus_weeks(i64::from(n))?;
            let (year, week) = start.iso_week();
            Some(Choice::one(Dated {
                timex: DateTimex::Date(TimexDate::Week { year, week }),
                span: Some((start, start.plus_days(5)?)),
            }))
        }
        DateSpec::RestOfWeek => {
            let days = i32::try_from(monday(today)?.plus_days(6)?.epoch_day() - today.epoch_day()).ok()?;
            (days > 0).then(|| range_period(today, Duration::units(Unit::Day, days)?, None))?.map(Choice::one)
        }
        DateSpec::HolidayWeekend { holiday: h, year } => {
            let weekend = |y: i32| -> Option<Dated> {
                let hd = holiday(h, y)?;
                let (sat, end) = if hd.day_of_week() == 1 {
                    (hd.plus_days(-2)?, hd.plus_days(1)?)
                } else {
                    (monday(hd)?.plus_days(5)?, monday(hd)?.plus_days(7)?)
                };
                let (year, week) = sat.iso_week();
                Some(Dated {
                    timex: DateTimex::Date(TimexDate::Weekend { year, week }),
                    span: Some((hd.min(sat), end.max(hd.plus_days(1)?))),
                })
            };
            if let Some(y) = year.year(ry) {
                weekend(y).map(Choice::one)
            } else {
                let this = holiday(h, ry)?;
                let py = if this <= today { ry } else { ry.checked_sub(1)? };
                let fy = if this >= today { ry } else { ry.checked_add(1)? };
                Some(Choice { past: weekend(py)?, future: weekend(fy)? })
            }
        }
        DateSpec::YearSlice { year, unit, count, last } => {
            let y = if year == YearRef::Any { ry } else { year.year(ry)? };
            let n = i32::from(count);
            let start = match (unit, last) {
                (Unit::Week, false) => iso_week_start(y, 1)?,
                (Unit::Week, true) => iso_week_start(y, 0)?.plus_weeks(1 - i64::from(n))?,
                (Unit::Month, false) => Date::new(y, 1, 1)?,
                (Unit::Month, true) => Date::new(y, 13u8.checked_sub(count)?, 1)?,
                (Unit::Day, false) => Date::new(y, 1, 1)?,
                (Unit::Day, true) => Date::new(y.checked_add(1)?, 1, 1)?.plus_days(-i64::from(n))?,
                _ => return None,
            };
            range_period(start, Duration::units(unit, n)?, None).map(Choice::one)
        }
        DateSpec::ShiftedWeekday { base, .. } => match base {
            HolidayOrRel::Today => Some(Choice::one(concrete_day(today))),
            HolidayOrRel::MonthDay { month, day } => Some(month_day(month, day, now)),
            HolidayOrRel::Holiday(h) => dates(DateSpec::Holiday { holiday: h, year: YearRef::Any }, now),
        },
    }
}

/// A run of `length` months from `first_month` of a year that may be
/// open: quarters and halves, written as ranges.
fn periods(year: YearRef, first_month: u8, length: i32, now: DateTime) -> Option<Choice> {
    let dur = Duration::units(Unit::Month, length)?;
    let today = now.date();
    match year.year(today.year()) {
        Some(y) => range_period(month_start(y, first_month)?, dur, None).map(Choice::one),
        None => {
            let y = today.year();
            let contains_or_after = (i32::from(today.month()) - 1) / length <= (i32::from(first_month) - 1) / length;
            let (p, f) = if contains_or_after { (y.checked_sub(1)?, y) } else { (y, y.checked_add(1)?) };
            let at = |y| range_period(month_start(y, first_month)?, dur, Some(TimexYear::Any));
            Some(Choice { past: at(p)?, future: at(f)? })
        }
    }
}
