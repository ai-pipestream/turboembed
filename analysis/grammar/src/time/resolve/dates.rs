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
use super::super::spec::{DateSpec, MonthRef, YearRef};
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
        span: Some((Date::new(year, 1, 1)?, Date::new(year + 1, 1, 1)?)),
    })
}

/// A period TIMEX3 writes as a range of days: `start` for `months`
/// months, or `years` years, with the years written as `year` says.
fn range_period(start: Date, length: Duration, timex_year: Option<TimexYear>) -> Option<Dated> {
    let end = plus(start, &length)?;
    let ymd = |d: Date| TimexDate::Ymd {
        year: match timex_year {
            Some(TimexYear::Short(yy)) => TimexYear::Short(((i32::from(yy) + d.year() - start.year()) % 100) as u8),
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
            Unit::Week => out.plus_weeks(n)?,
            Unit::Weekend => out.plus_weeks(n)?,
            Unit::Day | Unit::BusinessDay | Unit::Weekday => out.plus_days(n)?,
            Unit::Hour | Unit::Minute | Unit::Second => return None,
        };
    }
    Some(out)
}

/// The month `n` months from `year`-`month`.
fn month_plus(year: i32, month: u8, n: i32) -> (i32, u8) {
    let count = year * 12 + i32::from(month) - 1 + n;
    (count.div_euclid(12), (count.rem_euclid(12) + 1) as u8)
}

/// The years the past and the future readings of a month of an open year
/// fall in: a month before the reference's is this year's in the past and
/// next year's in the future; the reference's own month and later ones
/// are this year's in the future and last year's in the past.
fn month_years(month: u8, now: Date) -> (i32, i32) {
    let y = now.year();
    if month < now.month() { (y, y + 1) } else { (y - 1, y) }
}

fn months(year: YearRef, month: MonthRef, now: Date) -> Option<([(i32, u8); 2], TimexYear)> {
    match month {
        MonthRef::Rel(n) => {
            let (y, m) = month_plus(now.year(), now.month(), i32::from(n));
            Some(([(y, m), (y, m)], TimexYear::Year(y)))
        }
        MonthRef::Abs(m) => match year.year(now.year()) {
            Some(y) => Some(([(y, m), (y, m)], TimexYear::Year(y))),
            None if year == YearRef::Any => {
                let (p, f) = month_years(m, now);
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
    let past = (0..SEARCH).map(|k| y - k).find_map(|y| at(y).filter(|&x| DateTime::midnight(x) <= now));
    let future = (0..SEARCH).map(|k| y + k).find_map(|y| at(y).filter(|&x| DateTime::midnight(x) >= now));
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
        let (y, m) = month_plus(y, m, k);
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
        let (y, m) = month_plus(y, m, k);
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
            let (p, f) = month_years(m, today);
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
            let (y, m) = month_plus(now.year(), now.month(), n);
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
        TimeGrain::Year => year_period(now.year() + n),
        TimeGrain::Decade => {
            let start = Date::new(now.year().div_euclid(10) * 10 + 10 * n, 1, 1)?;
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
            let y = year.year(ry).unwrap_or(ry);
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
            let (y, m) = month_plus(ry, today.month(), i32::from(months));
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
        DateSpec::Quarter { year, quarter } => periods(year, (quarter - 1) * 3 + 1, 3, now),
        DateSpec::Half { year, half } => periods(year, (half - 1) * 6 + 1, 6, now),
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
                    let first = Date::new(ry.div_euclid(100) * 100 + i32::from(yy), 1, 1)?;
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
                let past = if this <= today { this } else { holiday(h, ry - 1)? };
                let future = if this >= today { this } else { holiday(h, ry + 1)? };
                let timex = TimexDate::Ymd { year: TimexYear::Any, month: Some(past.month()), day: Some(past.day()) };
                Some(Choice { past: day(past, timex), future: day(future, timex) })
            }
        },
        DateSpec::HolidayNear { holiday: h, future } => {
            let this = holiday(h, ry)?;
            let d = match (future, this > today, this < today) {
                (true, true, _) | (false, _, true) => this,
                (true, false, _) => holiday(h, ry + 1)?,
                (false, _, false) => holiday(h, ry - 1)?,
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
            let (p, f) = if contains_or_after { (y - 1, y) } else { (y, y + 1) };
            let at = |y| range_period(month_start(y, first_month)?, dur, Some(TimexYear::Any));
            Some(Choice { past: at(p)?, future: at(f)? })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::spec::Holiday;

    fn now(s: &str) -> DateTime {
        DateTime::parse(s).unwrap()
    }

    fn show(c: &Choice) -> Vec<String> {
        c.readings()
            .map(|(_, d)| {
                let t = match d.timex {
                    DateTimex::Date(t) => t.to_string(),
                    DateTimex::Range(s, e, l) => format!("({s},{e},{l})"),
                };
                match d.span {
                    Some((s, e)) => format!("{t} {s} {e}"),
                    None => format!("{t} -"),
                }
            })
            .collect()
    }

    fn cal(year: YearRef, month: u8, day: u8, weekday: u8) -> DateSpec {
        DateSpec::Cal { year, month, day, weekday }
    }

    #[test]
    fn open_years_read_both_ways() {
        let n = now("2016-11-07T00:00:00");
        assert_eq!(
            show(&dates(cal(YearRef::Any, 10, 2, 0), n).unwrap()),
            ["XXXX-10-02 2016-10-02 2016-10-03", "XXXX-10-02 2017-10-02 2017-10-03"]
        );
        let n2 = now("2016-11-02T10:00:00");
        assert_eq!(show(&dates(cal(YearRef::Any, 11, 2, 0), n2).unwrap())[1], "XXXX-11-02 2017-11-02 2017-11-03");
        assert_eq!(show(&dates(cal(YearRef::Any, 0, 0, 5), n).unwrap()).len(), 2);
        assert_eq!(show(&dates(cal(YearRef::Any, 0, 0, 1), n).unwrap())[1], "XXXX-WXX-1 2016-11-07 2016-11-08");
        assert_eq!(show(&dates(cal(YearRef::Any, 12, 0, 0), n).unwrap())[0], "XXXX-12 2015-12-01 2016-01-01");
        assert_eq!(show(&dates(cal(YearRef::Any, 2, 30, 0), n).unwrap()), ["XXXX-02-30 -"]);
        let leap = show(&dates(cal(YearRef::Any, 2, 29, 0), now("2017-03-22T00:00:00")).unwrap());
        assert_eq!(leap, ["XXXX-02-29 2016-02-29 2016-03-01", "XXXX-02-29 2020-02-29 2020-03-01"]);
        let the_4th = show(&dates(cal(YearRef::Any, 0, 4, 0), now("2018-08-07T12:00:00")).unwrap());
        assert_eq!(the_4th, ["XXXX-XX-04 2018-08-04 2018-08-05", "XXXX-XX-04 2018-09-04 2018-09-05"]);
        let short = show(&dates(cal(YearRef::Short(30), 12, 12, 0), n).unwrap());
        assert_eq!(short, ["XX30-12-12 1930-12-12 1930-12-13", "XX30-12-12 2030-12-12 2030-12-13"]);
    }

    #[test]
    fn weekdays_on_days_of_the_month() {
        let r = show(
            &dates(DateSpec::WeekdayDay { weekday: 1, day: 21, this_month: false }, now("2019-01-25T12:00:00"))
                .unwrap(),
        );
        assert_eq!(r, ["XXXX-WXX-1 2019-01-21 2019-01-22", "XXXX-WXX-1 2019-10-21 2019-10-22"]);
        let r = show(
            &dates(DateSpec::WeekdayDay { weekday: 1, day: 21, this_month: false }, now("2019-01-21T12:00:00"))
                .unwrap(),
        );
        assert_eq!(r, ["2019-01-21 2019-01-21 2019-01-22"]);
    }

    #[test]
    fn relative_periods() {
        let n = now("2019-07-04T00:00:00");
        let q =
            |n: i16| show(&dates(DateSpec::Rel { grain: TimeGrain::Quarter, n }, now("2019-07-04T00:00:00")).unwrap());
        assert_eq!(q(0), ["(2019-07-01,2019-10-01,P3M) 2019-07-01 2019-10-01"]);
        assert_eq!(q(-1)[0].split(' ').next(), Some("(2019-04-01,2019-07-01,P3M)"));
        assert_eq!(
            show(&dates(DateSpec::Rel { grain: TimeGrain::Week, n: 1 }, n).unwrap())[0],
            "2019-W28 2019-07-08 2019-07-15"
        );
        let we = show(&dates(DateSpec::Rel { grain: TimeGrain::Weekend, n: 0 }, now("2016-11-11T00:00:00")).unwrap());
        assert_eq!(we, ["2016-W45-WE 2016-11-12 2016-11-14"]);
        let decade =
            show(&dates(DateSpec::Rel { grain: TimeGrain::Decade, n: 1 }, now("2018-07-02T00:00:00")).unwrap());
        assert_eq!(decade, ["(2020-01-01,2030-01-01,P10Y) 2020-01-01 2030-01-01"]);
        let past = show(&dates(DateSpec::WeekdayPast { weekday: 3 }, now("2019-01-31T00:00:00")).unwrap());
        assert_eq!(past, ["2019-01-30 2019-01-30 2019-01-31"]);
        let fortnight = show(&dates(DateSpec::Fortnight { n: 0 }, now("2021-05-20T00:00:00")).unwrap());
        assert_eq!(fortnight, ["(2021-05-17,2021-05-31,P2W) 2021-05-17 2021-05-31"]);
        let same = show(&dates(DateSpec::Undefined { grain: TimeGrain::Month }, now("2017-11-08T00:00:00")).unwrap());
        assert_eq!(same, ["XXXX-XX 2017-11-01 2017-12-01"]);
    }

    #[test]
    fn weeks_quarters_decades_and_holidays() {
        let n = now("2016-11-07T00:00:00");
        let lw = DateSpec::MonthWeek { year: YearRef::Any, month: MonthRef::Abs(7), week: 0 };
        assert_eq!(
            show(&dates(lw, n).unwrap()),
            ["XXXX-07-W05 2016-07-25 2016-08-01", "XXXX-07-W05 2017-07-24 2017-07-31"]
        );
        let w1 = DateSpec::Week { year: YearRef::Abs(2015), week: 1 };
        assert_eq!(show(&dates(w1, n).unwrap()), ["2015-W01 2014-12-29 2015-01-05"]);
        let q1 = DateSpec::Quarter { year: YearRef::Any, quarter: 1 };
        let r = show(&dates(q1, now("2019-11-07T00:00:00")).unwrap());
        assert_eq!(r[0], "(XXXX-01-01,XXXX-04-01,P3M) 2019-01-01 2019-04-01");
        assert_eq!(r[1], "(XXXX-01-01,XXXX-04-01,P3M) 2020-01-01 2020-04-01");
        let nineties = show(&dates(DateSpec::Decade { year: YearRef::Short(90) }, now("2019-05-23T00:00:00")).unwrap());
        assert_eq!(nineties[0], "(XX90-01-01,XX00-01-01,P10Y) 1990-01-01 2000-01-01");
        assert_eq!(nineties[1], "(XX90-01-01,XX00-01-01,P10Y) 2090-01-01 2100-01-01");
        let c15 = show(&dates(DateSpec::Century { n: 15 }, n).unwrap());
        assert_eq!(c15, ["(1400-01-01,1500-01-01,P100Y) 1400-01-01 1500-01-01"]);
        let easter_monday = DateSpec::Holiday { holiday: Holiday::Easter { days: 1 }, year: YearRef::Any };
        let r = show(&dates(easter_monday, now("2019-08-05T00:00:00")).unwrap());
        assert_eq!(r, ["XXXX-04-22 2019-04-22 2019-04-23", "XXXX-04-22 2020-04-13 2020-04-14"]);
        let next_easter = DateSpec::HolidayNear { holiday: Holiday::Easter { days: 0 }, future: true };
        assert_eq!(
            show(&dates(next_easter, now("2019-06-12T00:00:00")).unwrap())[0],
            "2020-04-12 2020-04-12 2020-04-13"
        );
        let first_monday = DateSpec::NthWeekday { year: YearRef::Any, month: MonthRef::Rel(1), weekday: 1, n: 1 };
        assert_eq!(
            show(&dates(first_monday, now("2018-08-01T12:00:00")).unwrap())[0],
            "XXXX-09-WXX-1-#1 2018-09-03 2018-09-04"
        );
    }
}
