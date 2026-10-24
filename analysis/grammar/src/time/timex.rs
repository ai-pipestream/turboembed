//! TIMEX3 values as Recognizers-Text writes them: `2016-11-07`,
//! `XXXX-WXX-5`, `XXXX-07-W05`, `2016-W45-WE`, `T19:56:30`,
//! `2016-11-07TEV`, `(2016-11-08,2016-11-11,P3D)`, `PT1H30M`,
//! `PRESENT_REF`.

use std::fmt;

use turbo_analysis_layers::{Date, write_year};

use super::duration::Duration;
use super::spec::{Part, Season, Shown, YearKind};

/// The year of a TIMEX3 date.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TimexYear {
    Year(i32),
    /// `XXXX`.
    Any,
    /// Two digits with the century open: `XX30`.
    Short(u8),
}

impl fmt::Display for TimexYear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            TimexYear::Year(y) => write_year(f, y),
            TimexYear::Any => f.write_str("XXXX"),
            TimexYear::Short(yy) => write!(f, "XX{yy:02}"),
        }
    }
}

/// The date part of a TIMEX3 value.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TimexDate {
    #[default]
    None,
    /// `2016-10-16`, `XXXX-10`, `2016`, `XXXX-XX-18` (a day with no
    /// month).
    Ymd { year: TimexYear, month: Option<u8>, day: Option<u8> },
    /// `XXXX-WXX-5`.
    Weekday(u8),
    /// `2018-W27`.
    Week { year: i32, week: u8 },
    /// `2016-12-W05`, `XXXX-07-W03`.
    MonthWeek { year: TimexYear, month: u8, week: u8 },
    /// `XXXX-09-WXX-1-#1`: the first Monday of September.
    NthWeekday { month: u8, weekday: u8, n: u8 },
    /// `2016-W45-WE`.
    Weekend { year: i32, week: u8 },
    /// `2016-SU`, `SU`.
    Season { year: Option<i32>, season: Season },
    /// `FY2018`, `SYXXXX`.
    OtherYear { kind: YearKind, year: Option<i32> },
    /// A fixed form: `XXXX-WXX`, `XXXX-XX`, `XXXX`, `XXXX-WXX-WE`.
    Pattern(&'static str),
    /// `PRESENT_REF`.
    Present,
}

impl TimexDate {
    pub fn day(d: Date) -> TimexDate {
        TimexDate::Ymd { year: TimexYear::Year(d.year()), month: Some(d.month()), day: Some(d.day()) }
    }

    pub fn month(year: TimexYear, month: u8) -> TimexDate {
        TimexDate::Ymd { year, month: Some(month), day: None }
    }

    pub fn year(year: i32) -> TimexDate {
        TimexDate::Ymd { year: TimexYear::Year(year), month: None, day: None }
    }
}

impl fmt::Display for TimexDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            TimexDate::None => Ok(()),
            TimexDate::Ymd { year, month, day } => {
                write!(f, "{year}")?;
                match (month, day) {
                    (Some(m), Some(d)) => write!(f, "-{m:02}-{d:02}"),
                    (Some(m), None) => write!(f, "-{m:02}"),
                    (None, Some(d)) => write!(f, "-XX-{d:02}"),
                    (None, None) => Ok(()),
                }
            }
            TimexDate::Weekday(d) => write!(f, "XXXX-WXX-{d}"),
            TimexDate::Week { year, week } => {
                write_year(f, year)?;
                write!(f, "-W{week:02}")
            }
            TimexDate::MonthWeek { year, month, week } => write!(f, "{year}-{month:02}-W{week:02}"),
            TimexDate::NthWeekday { month, weekday, n } => write!(f, "XXXX-{month:02}-WXX-{weekday}-#{n}"),
            TimexDate::Weekend { year, week } => {
                write_year(f, year)?;
                write!(f, "-W{week:02}-WE")
            }
            TimexDate::Season { year, season } => {
                if let Some(y) = year {
                    write_year(f, y)?;
                    f.write_str("-")?;
                }
                f.write_str(season.code())
            }
            TimexDate::OtherYear { kind, year } => {
                f.write_str(kind.code())?;
                match year {
                    Some(y) => write_year(f, y),
                    None => f.write_str("XXXX"),
                }
            }
            TimexDate::Pattern(p) => f.write_str(p),
            TimexDate::Present => f.write_str("PRESENT_REF"),
        }
    }
}

/// The time part of a TIMEX3 value.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TimexTime {
    #[default]
    None,
    /// `T07`, `T07:30`, `T07:56:30`, with the hour of the day.
    Clock { hour: u8, minute: u8, second: u8, shown: Shown },
    /// `TMO`.
    Part(Part),
}

impl TimexTime {
    /// A time `seconds` after midnight, written to the second.
    pub fn seconds(seconds: u32) -> TimexTime {
        TimexTime::Clock {
            hour: (seconds / 3600 % 24) as u8,
            minute: (seconds / 60 % 60) as u8,
            second: (seconds % 60) as u8,
            shown: Shown::Second,
        }
    }
}

impl fmt::Display for TimexTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            TimexTime::None => Ok(()),
            TimexTime::Clock { hour, minute, second, shown } => {
                write!(f, "T{hour:02}")?;
                if shown >= Shown::Minute {
                    write!(f, ":{minute:02}")?;
                }
                if shown == Shown::Second {
                    write!(f, ":{second:02}")?;
                }
                Ok(())
            }
            TimexTime::Part(p) => write!(f, "T{}", p.code()),
        }
    }
}

/// A TIMEX3 point: a date, a time or both.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TimexPoint {
    pub date: TimexDate,
    pub time: TimexTime,
}

impl fmt::Display for TimexPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.date, self.time)
    }
}

/// A TIMEX3 value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Timex {
    Point(TimexPoint),
    /// `(start,end,duration)`.
    Range(TimexPoint, TimexPoint, Duration),
    /// `(start,end,PXD)`: a date range whose bounds include a calendar
    /// day that does not exist in that year (`2/29-3/1/2019`).
    UnresolvedRange(TimexPoint, TimexPoint),
    Duration(Duration),
}

impl Default for Timex {
    fn default() -> Timex {
        Timex::Point(TimexPoint::default())
    }
}

impl fmt::Display for Timex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Timex::Point(p) => p.fmt(f),
            Timex::Range(s, e, d) => write!(f, "({s},{e},{d})"),
            Timex::UnresolvedRange(s, e) => write!(f, "({s},{e},PXD)"),
            Timex::Duration(d) => d.fmt(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::duration::Unit;

    fn p(date: TimexDate, time: TimexTime) -> String {
        TimexPoint { date, time }.to_string()
    }

    #[test]
    fn dates_and_times() {
        let any = TimexYear::Any;
        assert_eq!(TimexDate::day(Date::parse("2016-11-07").unwrap()).to_string(), "2016-11-07");
        assert_eq!(TimexDate::Ymd { year: any, month: None, day: Some(18) }.to_string(), "XXXX-XX-18");
        assert_eq!(TimexDate::month(any, 5).to_string(), "XXXX-05");
        assert_eq!(
            TimexDate::Ymd { year: TimexYear::Short(30), month: Some(12), day: Some(12) }.to_string(),
            "XX30-12-12"
        );
        assert_eq!(TimexDate::year(2016).to_string(), "2016");
        assert_eq!(TimexDate::Weekday(5).to_string(), "XXXX-WXX-5");
        assert_eq!(TimexDate::Week { year: 2018, week: 3 }.to_string(), "2018-W03");
        assert_eq!(TimexDate::MonthWeek { year: any, month: 7, week: 5 }.to_string(), "XXXX-07-W05");
        assert_eq!(TimexDate::NthWeekday { month: 9, weekday: 1, n: 1 }.to_string(), "XXXX-09-WXX-1-#1");
        assert_eq!(TimexDate::Weekend { year: 2016, week: 45 }.to_string(), "2016-W45-WE");
        assert_eq!(TimexDate::Season { year: Some(2016), season: Season::Summer }.to_string(), "2016-SU");
        assert_eq!(TimexDate::Season { year: None, season: Season::Winter }.to_string(), "WI");
        assert_eq!(TimexDate::OtherYear { kind: YearKind::Fiscal, year: None }.to_string(), "FYXXXX");
        assert_eq!(TimexDate::Present.to_string(), "PRESENT_REF");
        let clock = |shown| TimexTime::Clock { hour: 7, minute: 56, second: 30, shown };
        assert_eq!(p(TimexDate::None, clock(Shown::Hour)), "T07");
        assert_eq!(p(TimexDate::None, clock(Shown::Minute)), "T07:56");
        assert_eq!(p(TimexDate::Weekday(1), clock(Shown::Second)), "XXXX-WXX-1T07:56:30");
        assert_eq!(p(TimexDate::year(2016), TimexTime::Part(Part::Evening)), "2016TEV");
        assert_eq!(TimexTime::seconds(41_400).to_string(), "T11:30:00");
    }

    #[test]
    fn ranges_and_durations() {
        let s = TimexPoint { date: TimexDate::day(Date::parse("2016-11-08").unwrap()), time: TimexTime::None };
        let e = TimexPoint { date: TimexDate::day(Date::parse("2016-11-11").unwrap()), time: TimexTime::None };
        let three = Duration::units(Unit::Day, 3).unwrap();
        assert_eq!(Timex::Range(s, e, three).to_string(), "(2016-11-08,2016-11-11,P3D)");
        assert_eq!(Timex::Duration(three).to_string(), "P3D");
        assert_eq!(Timex::default().to_string(), "");
    }
}
