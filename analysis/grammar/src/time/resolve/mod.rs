//! Resolution: what a [`TimeValue`] names, given a reference date-time.
//!
//! A value resolves to one or more alternatives, as Recognizers-Text
//! lists them: a date the text leaves the year of open reads as the last
//! one before the reference and the first one after it (`May 29`), and a
//! clock time with no half of the day as both halves (`at 5`). Each
//! alternative has its TIMEX3 value and its start and end.

pub mod dates;
mod points;
mod ranges;

use std::fmt;

use turbo_analysis_layers::{Date, DateTime, Modifier, TimeGrain, TimeKind};

use super::duration::Unit;
use super::spec::{DateSpec, Point, Range, Recurrence, TimeOfDay, TimeValue, YearRef};
use super::timex::{Timex, TimexDate, TimexPoint, TimexTime};

/// A resolved instant: a day, a date-time, or a time of day with no date
/// (seconds after midnight).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Instant {
    Date(Date),
    DateTime(DateTime),
    Time(u32),
}

impl Instant {
    /// Local seconds: since 1970-01-01T00:00 for a date or a date-time,
    /// since midnight for a time of day.
    pub fn seconds(self) -> i64 {
        match self {
            Instant::Date(d) => DateTime::midnight(d).local_seconds(),
            Instant::DateTime(t) => t.local_seconds(),
            Instant::Time(s) => i64::from(s),
        }
    }
}

/// Writes the instant as Recognizers-Text does: `2016-11-07`,
/// `2016-11-07 16:12:00`, `16:12:00`.
impl fmt::Display for Instant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Instant::Date(d) => d.fmt(f),
            Instant::DateTime(t) => write!(f, "{} {:02}:{:02}:{:02}", t.date(), t.hour(), t.minute(), t.second()),
            Instant::Time(s) => write!(f, "{:02}:{:02}:{:02}", s / 3600 % 24, s / 60 % 60, s % 60),
        }
    }
}

/// What one alternative resolves to.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Resolution {
    Point(Instant),
    /// A range, open on a side a modifier leaves open (`after 2016`).
    Range {
        start: Option<Instant>,
        end: Option<Instant>,
    },
    Duration {
        millis: i64,
    },
    /// Recognizers-Text's `not resolved`: a set, a season, `Feb 30`.
    NotResolved,
}

/// One reading of a mention.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Alternative {
    pub timex: Timex,
    pub value: Resolution,
}

/// Every reading of a mention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub kind: TimeKind,
    pub grain: TimeGrain,
    pub modifier: Modifier,
    pub alternatives: Vec<Alternative>,
    /// The reading nearest after the reference, in the first half of the
    /// day when the text leaves the half open.
    pub preferred: usize,
}

/// Where a reading lies: an instant, or a range that a modifier may
/// leave open on one side.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Span {
    Point(Instant),
    Range(Option<Instant>, Option<Instant>),
}

/// One reading before it is an [`Alternative`]; no span when the reading
/// is not resolved.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Moment {
    timex: Timex,
    span: Option<Span>,
    /// From the future reading of the date.
    future: bool,
    /// From the first reading of the clock time.
    first: bool,
}

impl Moment {
    fn alternative(&self) -> Alternative {
        let value = match self.span {
            None => Resolution::NotResolved,
            Some(Span::Point(s)) => Resolution::Point(s),
            Some(Span::Range(start, end)) => Resolution::Range { start, end },
        };
        Alternative { timex: self.timex, value }
    }
}

fn finish(kind: TimeKind, grain: TimeGrain, modifier: Modifier, moments: &[Moment]) -> Resolved {
    let mut alternatives: Vec<Alternative> = Vec::with_capacity(moments.len());
    let mut preferred = 0;
    let mut best = (false, false);
    for m in moments {
        let a = m.alternative();
        if let Some(i) = alternatives.iter().position(|x| *x == a) {
            if (m.future, m.first) > best {
                (best, preferred) = ((m.future, m.first), i);
            }
            continue;
        }
        if alternatives.is_empty() || (m.future, m.first) > best {
            (best, preferred) = ((m.future, m.first), alternatives.len());
        }
        alternatives.push(a);
    }
    Resolved { kind, grain, modifier, alternatives, preferred }
}

/// The grain of a duration's finest unit.
pub fn unit_grain(u: Unit) -> TimeGrain {
    match u {
        Unit::Year => TimeGrain::Year,
        Unit::Month => TimeGrain::Month,
        Unit::Week => TimeGrain::Week,
        Unit::Weekend => TimeGrain::Weekend,
        Unit::BusinessDay | Unit::Weekday | Unit::Day => TimeGrain::Day,
        Unit::Hour => TimeGrain::Hour,
        Unit::Minute => TimeGrain::Minute,
        Unit::Second => TimeGrain::Second,
    }
}

/// The TIMEX3 value of a recurrence's point, which names no reference:
/// `XXXX-WXX-1T16`, `T15`, `TMO`, `XXXX-XX-19`.
fn pattern(p: &Point) -> Option<TimexPoint> {
    let date = match p.date {
        DateSpec::None => TimexDate::None,
        DateSpec::Cal { year: YearRef::Any, month: 0, day: 0, weekday } if weekday > 0 => TimexDate::Weekday(weekday),
        DateSpec::Cal { year: YearRef::Any, month: 0, day, weekday: 0 } if day > 0 => {
            TimexDate::Ymd { year: super::timex::TimexYear::Any, month: None, day: Some(day) }
        }
        _ => return None,
    };
    let time = match p.time {
        TimeOfDay::None => TimexTime::None,
        TimeOfDay::Clock(c) => {
            TimexTime::Clock { hour: c.hours().0[0], minute: c.minute, second: c.second, shown: c.shown }
        }
        TimeOfDay::Part(part) => TimexTime::Part(part),
        TimeOfDay::Now => return None,
    };
    Some(TimexPoint { date, time })
}

/// Resolves `value` against the reference `now`, or gives none when the
/// value names nothing a reading can hold.
pub fn resolve(value: &TimeValue, now: DateTime) -> Option<Resolved> {
    match *value {
        TimeValue::Point { point, modifier, clip } => points::resolve(&point, modifier, clip, now),
        TimeValue::Range { range, modifier } => ranges::resolve(&range, modifier, now),
        TimeValue::Duration { length, modifier } => {
            let millis = length.millis();
            if millis < 0 {
                return None;
            }
            let alternative = Alternative { timex: Timex::Duration(length), value: Resolution::Duration { millis } };
            let grain = unit_grain(length.finest()?);
            Some(Resolved { kind: TimeKind::Duration, grain, modifier, alternatives: vec![alternative], preferred: 0 })
        }
        TimeValue::Set(r) => {
            let (timex, grain) = match r {
                Recurrence::Every(d) => (Timex::Duration(d), unit_grain(d.finest()?)),
                Recurrence::At(p) => (Timex::Point(pattern(&p)?), points::grain(&p)?),
            };
            let alternative = Alternative { timex, value: Resolution::NotResolved };
            Some(Resolved {
                kind: TimeKind::Set,
                grain,
                modifier: Modifier::NONE,
                alternatives: vec![alternative],
                preferred: 0,
            })
        }
    }
}

fn date_depends(d: DateSpec) -> bool {
    match d {
        DateSpec::None | DateSpec::Century { .. } => false,
        DateSpec::Cal { year, .. }
        | DateSpec::Week { year, .. }
        | DateSpec::Quarter { year, .. }
        | DateSpec::Half { year, .. }
        | DateSpec::Season { year, .. }
        | DateSpec::Decade { year }
        | DateSpec::Holiday { year, .. }
        | DateSpec::OtherYear { year, .. } => !matches!(year, YearRef::Abs(_)),
        DateSpec::MonthWeek { year, month, .. } | DateSpec::NthWeekday { year, month, .. } => {
            !matches!(year, YearRef::Abs(_)) || matches!(month, super::spec::MonthRef::Rel(_))
        }
        _ => true,
    }
}

fn point_depends(p: &Point) -> bool {
    date_depends(p.date) || p.time == TimeOfDay::Now
}

/// Whether what `value` names depends on the reference: `tomorrow` and
/// `Friday` do, `2016-11-07`, `at 5` and `3 hours` do not.
pub fn depends_on_reference(value: &TimeValue) -> bool {
    match value {
        TimeValue::Point { point, .. } => point_depends(point),
        TimeValue::Range { range, .. } => match range {
            Range::Between { start, end, .. } => point_depends(start) || point_depends(end),
            Range::For { start, .. } => point_depends(start),
            Range::Next { .. } | Range::Within { .. } => true,
        },
        TimeValue::Duration { .. } | TimeValue::Set(_) => false,
    }
}

#[cfg(test)]
mod tests;
