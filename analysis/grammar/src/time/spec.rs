//! What a time expression states, before it is resolved against a
//! reference date-time: calendar fields that may be missing (`May 29`),
//! periods counted from the reference (`next week`), clock times whose
//! half of the day may be open (`at 5`), durations, ranges and sets.
//!
//! Every type is `Copy` and small, so a chart node carries one. Resolution
//! ([`super::resolve`]) turns a [`TimeValue`] into dates and times.

use turbo_analysis_layers::{Modifier, TimeGrain};

use super::duration::Duration;

/// A year as written.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum YearRef {
    /// Not written: `May 29`.
    #[default]
    Any,
    /// `2016`.
    Abs(i16),
    /// Two digits whose century the text leaves open: `12/12/30`.
    Short(u8),
    /// The reference's year moved by `n`: `this September` (0), `the 9th
    /// month of next year` (1).
    Rel(i16),
}

impl YearRef {
    /// The year, when the text and the reference year fix it.
    pub fn year(self, reference: i32) -> Option<i32> {
        match self {
            YearRef::Abs(y) => Some(i32::from(y)),
            YearRef::Rel(n) => Some(reference + i32::from(n)),
            YearRef::Any | YearRef::Short(_) => None,
        }
    }
}

/// A month as written: `July` or `next month`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MonthRef {
    /// 1 to 12, in the year beside it.
    Abs(u8),
    /// The reference's month moved by `n`.
    Rel(i8),
}

/// Which half of the day a clock time is in.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Meridiem {
    /// Not said: `at 5` is 05:00 or 17:00.
    #[default]
    Unknown,
    Am,
    Pm,
    /// A 24-hour time: `17:30`, `0730` read as written.
    Day,
}

/// How much of a clock time is written, which its TIMEX3 value keeps:
/// `T07`, `T07:30`, `T07:56:30`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Shown {
    #[default]
    Hour,
    Minute,
    Second,
}

/// A time of day as written.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Clock {
    /// 0 to 23 for a 24-hour time, 1 to 12 (or 0) otherwise.
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub shown: Shown,
    pub meridiem: Meridiem,
}

impl Clock {
    pub fn new(hour: u8, minute: u8, second: u8, shown: Shown, meridiem: Meridiem) -> Option<Clock> {
        let max_hour = if matches!(meridiem, Meridiem::Am | Meridiem::Pm) { 12 } else { 24 };
        (hour <= max_hour && minute < 60 && second < 60).then_some(Clock { hour, minute, second, shown, meridiem })
    }

    /// The hours of the day, 0 to 23, the clock reads as: one, or two when
    /// the half of the day is open (`5` is 5 and 17, `12` is 12 and 0).
    pub fn hours(self) -> ([u8; 2], usize) {
        let h = self.hour % 24;
        match self.meridiem {
            Meridiem::Day => ([h, h], 1),
            Meridiem::Am => ([h % 12, 0], 1),
            Meridiem::Pm => ([h % 12 + 12, 0], 1),
            Meridiem::Unknown if (1..=12).contains(&h) => {
                let (a, b) = if h == 12 { (12, 0) } else { (h, h + 12) };
                ([a, b], 2)
            }
            Meridiem::Unknown => ([h, h], 1),
        }
    }

    /// Seconds after midnight at `hour`.
    pub fn seconds_at(self, hour: u8) -> u32 {
        u32::from(hour) * 3600 + u32::from(self.minute) * 60 + u32::from(self.second)
    }

    /// The same clock in the half of the day `m` says.
    pub fn with(self, m: Meridiem) -> Clock {
        if self.meridiem != Meridiem::Unknown || self.hour > 12 {
            return self;
        }
        Clock { meridiem: m, ..self }
    }
}

/// A part of the day, with its TIMEX3 code and hours.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    Morning,
    Afternoon,
    Evening,
    Night,
    Daytime,
    NightTime,
    Breakfast,
    Lunch,
    Dinner,
}

impl Part {
    /// `MO`, `AF`, `EV`, `NI`, `DT`, `NT`, `MEB`, `MEL` or `MED`.
    pub fn code(self) -> &'static str {
        match self {
            Part::Morning => "MO",
            Part::Afternoon => "AF",
            Part::Evening => "EV",
            Part::Night => "NI",
            Part::Daytime => "DT",
            Part::NightTime => "NT",
            Part::Breakfast => "MEB",
            Part::Lunch => "MEL",
            Part::Dinner => "MED",
        }
    }

    /// Its start and end in seconds after midnight, as Recognizers-Text
    /// sets them: the night ends at 23:59:59.
    pub fn seconds(self) -> (u32, u32) {
        let h = |x: u32| x * 3600;
        match self {
            Part::Morning | Part::Breakfast => (h(8), h(12)),
            Part::Afternoon => (h(12), h(16)),
            Part::Evening | Part::Dinner => (h(16), h(20)),
            Part::Night => (h(20), h(24) - 1),
            Part::Daytime => (h(8), h(18)),
            Part::NightTime => (0, h(8)),
            Part::Lunch => (h(11), h(13)),
        }
    }

    /// The half of the day an open clock time in this part is in, with
    /// the hour from which a night time is still the early morning.
    pub fn meridiem(self, hour: u8) -> Meridiem {
        match self {
            Part::Morning | Part::Breakfast | Part::NightTime => Meridiem::Am,
            Part::Night if (1..=4).contains(&hour) => Meridiem::Am,
            Part::Lunch if hour >= 11 => Meridiem::Am,
            _ => Meridiem::Pm,
        }
    }
}

/// The time of day a point names.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TimeOfDay {
    #[default]
    None,
    Clock(Clock),
    Part(Part),
    /// The reference instant itself: `now`.
    Now,
}

/// A season; Recognizers-Text leaves seasons unresolved.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Season {
    Spring,
    Summer,
    Fall,
    Winter,
}

impl Season {
    pub fn code(self) -> &'static str {
        match self {
            Season::Spring => "SP",
            Season::Summer => "SU",
            Season::Fall => "FA",
            Season::Winter => "WI",
        }
    }
}

/// A year that does not start in January, which Recognizers-Text leaves
/// unresolved.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum YearKind {
    Fiscal,
    School,
}

impl YearKind {
    pub fn code(self) -> &'static str {
        match self {
            YearKind::Fiscal => "FY",
            YearKind::School => "SY",
        }
    }
}

/// A holiday whose date follows from the year.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Holiday {
    /// A fixed month and day: Christmas is 12-25.
    Fixed { month: u8, day: u8 },
    /// Easter Sunday moved by `days`: Easter Monday is 1.
    Easter { days: i8 },
    /// The `n`th `weekday` of `month` (0 for the last) moved by `days`:
    /// Thanksgiving is the fourth Thursday of November, Black Friday one
    /// day later.
    Nth { month: u8, weekday: u8, n: u8, days: i8 },
}

/// The date part of a point.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum DateSpec {
    #[default]
    None,
    /// Calendar fields, 0 where the text names none: `2016-10-16`, `May
    /// 29` (any year), `the 18th` (any month), `Friday`, `2016`, `this
    /// September` (the reference's year).
    Cal {
        year: YearRef,
        month: u8,
        day: u8,
        weekday: u8,
    },
    /// A weekday on a day of the month: `Monday 21` is the nearest month
    /// whose 21st is a Monday; with `this_month`, as `Monday the 21st`
    /// says, the 21st of the reference's month.
    WeekdayDay {
        weekday: u8,
        day: u8,
        this_month: bool,
    },
    /// The period of `grain` that holds the reference, moved by `n`:
    /// `today`, `tomorrow`, `this week`, `next month`, `last year`.
    Rel {
        grain: TimeGrain,
        n: i16,
    },
    /// Two weeks from the reference week's Monday, moved by `n` of them.
    Fortnight {
        n: i8,
    },
    /// `weekday` of the week `weeks` from the reference's: `next Monday`.
    WeekdayRel {
        weekday: u8,
        weeks: i8,
    },
    /// The last `weekday` before the reference day: `past Wednesday`.
    WeekdayPast {
        weekday: u8,
    },
    /// ISO week `week` of a year, 0 for its last: `week 31`.
    Week {
        year: YearRef,
        week: u8,
    },
    /// Week `week` of a month, 0 for its last, counting the weeks that
    /// hold the month's Thursdays: `the last week of July`.
    MonthWeek {
        year: YearRef,
        month: MonthRef,
        week: u8,
    },
    /// Day `day` of the reference's month moved by `months`: `the 15th
    /// day of next month`, `the 21st this month`.
    RelMonthDay {
        months: i8,
        day: u8,
    },
    /// The week, Monday to Monday, holding a day of an open year (or of
    /// an open month when `month` is 0), written as that day: `the week
    /// of April 10th`, `w/c Feb 4`.
    WeekOf {
        month: u8,
        day: u8,
    },
    /// The `n`th `weekday` of a month: `the first Monday of next month`.
    NthWeekday {
        year: YearRef,
        month: MonthRef,
        weekday: u8,
        n: u8,
    },
    /// `Q1 2017`, `the first quarter of the year`.
    Quarter {
        year: YearRef,
        quarter: u8,
    },
    /// `2019 H2`.
    Half {
        year: YearRef,
        half: u8,
    },
    Season {
        year: YearRef,
        season: Season,
    },
    /// The decade starting in `year`: `the 1990s`, `the nineties`.
    Decade {
        year: YearRef,
    },
    /// The `n`th century: `the 15th century` is 1400 to 1500.
    Century {
        n: u8,
    },
    Holiday {
        holiday: Holiday,
        year: YearRef,
    },
    /// The holiday's next date after the reference, or with `future`
    /// false its last one before it: `next Easter`, `last Easter`.
    HolidayNear {
        holiday: Holiday,
        future: bool,
    },
    /// `FY2018`, `this school year`.
    OtherYear {
        kind: YearKind,
        year: YearRef,
    },
    /// A period of `grain` the text does not place: `the same week`, `that
    /// weekend`; read as the reference's.
    Undefined {
        grain: TimeGrain,
    },
}

impl DateSpec {
    /// The grain of the dates the spec names.
    pub fn grain(self) -> Option<TimeGrain> {
        Some(match self {
            DateSpec::None => return None,
            DateSpec::Cal { month, day, weekday, .. } => {
                if day > 0 || weekday > 0 {
                    TimeGrain::Day
                } else if month > 0 {
                    TimeGrain::Month
                } else {
                    TimeGrain::Year
                }
            }
            DateSpec::WeekdayDay { .. }
            | DateSpec::WeekdayRel { .. }
            | DateSpec::WeekdayPast { .. }
            | DateSpec::NthWeekday { .. }
            | DateSpec::Holiday { .. }
            | DateSpec::HolidayNear { .. }
            | DateSpec::RelMonthDay { .. } => TimeGrain::Day,
            DateSpec::Rel { grain, .. } | DateSpec::Undefined { grain } => grain,
            DateSpec::Fortnight { .. }
            | DateSpec::Week { .. }
            | DateSpec::MonthWeek { .. }
            | DateSpec::WeekOf { .. } => TimeGrain::Week,
            DateSpec::Quarter { .. } => TimeGrain::Quarter,
            DateSpec::Half { .. } => TimeGrain::Half,
            DateSpec::Season { .. } => TimeGrain::Season,
            DateSpec::Decade { .. } => TimeGrain::Decade,
            DateSpec::Century { .. } => TimeGrain::Century,
            DateSpec::OtherYear { .. } => TimeGrain::Year,
        })
    }

    /// Whether the spec names a single day.
    pub fn is_day(self) -> bool {
        self.grain() == Some(TimeGrain::Day)
    }
}

/// A point, or a period named as one: a date, a time of day, or both,
/// moved by a duration (`3 days after January 12th`, `in 5 minutes`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Point {
    pub date: DateSpec,
    pub time: TimeOfDay,
    pub shift: Duration,
}

impl Point {
    pub fn date(date: DateSpec) -> Point {
        Point { date, ..Point::default() }
    }

    pub fn time(time: TimeOfDay) -> Point {
        Point { time, ..Point::default() }
    }
}

/// Which instant of a period a range bound takes: `from 2007` starts at
/// the start of 2007, `to the end of 2008` ends at its end.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Edge {
    #[default]
    Start,
    Mid,
    End,
}

/// A range of time.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Range {
    /// From one point to another, which ends the range where it starts
    /// unless its edge says otherwise: `2014 to 2018` ends on 2018-01-01.
    Between { start: Point, start_edge: Edge, end: Point, end_edge: Edge },
    /// From a point for a duration: `from 9 for 2.5 hrs`.
    For { start: Point, length: Duration },
    /// The coming or the past `length` from the reference, by the sign of
    /// its amounts: `next 3 days` starts tomorrow, `last 3 months` ends
    /// today, `next hour` starts now, `last 2 quarters` ends at this
    /// quarter's start.
    Next { length: Duration, align: TimeGrain },
    /// From the reference for a duration: `within 9 months`.
    Within { length: Duration },
}

/// A recurrence.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Recurrence {
    /// `weekly`, `every two days`.
    Every(Duration),
    /// `every Monday at 4pm`, `3pm each day`, `mornings`.
    At(Point),
}

/// What a time expression states.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TimeValue {
    /// With `clip`, a portion of the reference's own period stops at the
    /// reference: `later this week` starts no earlier than today.
    Point {
        point: Point,
        modifier: Modifier,
        clip: bool,
    },
    Range {
        range: Range,
        modifier: Modifier,
    },
    Duration {
        length: Duration,
        modifier: Modifier,
    },
    Set(Recurrence),
}

impl TimeValue {
    pub fn point(point: Point) -> TimeValue {
        TimeValue::Point { point, modifier: Modifier::NONE, clip: false }
    }

    pub fn range(range: Range) -> TimeValue {
        TimeValue::Range { range, modifier: Modifier::NONE }
    }

    pub fn modifier(&self) -> Modifier {
        match *self {
            TimeValue::Point { modifier, .. }
            | TimeValue::Range { modifier, .. }
            | TimeValue::Duration { modifier, .. } => modifier,
            TimeValue::Set(_) => Modifier::NONE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clocks_read_as_one_or_two_hours() {
        let c = |h, m| Clock::new(h, 0, 0, Shown::Hour, m).unwrap();
        assert_eq!(c(5, Meridiem::Unknown).hours(), ([5, 17], 2));
        assert_eq!(c(12, Meridiem::Unknown).hours(), ([12, 0], 2));
        assert_eq!(c(0, Meridiem::Unknown).hours().1, 1);
        assert_eq!(c(12, Meridiem::Am).hours().0[0], 0);
        assert_eq!(c(12, Meridiem::Pm).hours().0[0], 12);
        assert_eq!(c(7, Meridiem::Pm).hours().0[0], 19);
        assert_eq!(c(17, Meridiem::Day).hours(), ([17, 17], 1));
        assert_eq!(c(5, Meridiem::Unknown).with(Meridiem::Pm).hours().0[0], 17);
        assert_eq!(c(17, Meridiem::Day).with(Meridiem::Am).hours().0[0], 17);
        assert!(Clock::new(13, 0, 0, Shown::Hour, Meridiem::Pm).is_none());
        assert!(Clock::new(9, 60, 0, Shown::Minute, Meridiem::Day).is_none());
        assert_eq!(c(7, Meridiem::Day).seconds_at(7), 25_200);
    }

    #[test]
    fn parts_and_grains() {
        assert_eq!(Part::Night.seconds(), (72_000, 86_399));
        assert_eq!(Part::Lunch.code(), "MEL");
        assert_eq!(Part::Night.meridiem(3), Meridiem::Am);
        assert_eq!(Part::Night.meridiem(10), Meridiem::Pm);
        assert_eq!(Season::Winter.code(), "WI");
        assert_eq!(YearKind::School.code(), "SY");
        let may = DateSpec::Cal { year: YearRef::Any, month: 5, day: 0, weekday: 0 };
        assert_eq!(may.grain(), Some(TimeGrain::Month));
        assert!(DateSpec::Cal { year: YearRef::Any, month: 0, day: 0, weekday: 5 }.is_day());
        assert_eq!(DateSpec::None.grain(), None);
        assert_eq!(YearRef::Rel(1).year(2018), Some(2019));
        assert_eq!(YearRef::Short(30).year(2018), None);
    }

    #[test]
    fn a_time_value_fits_in_a_chart_node() {
        assert!(std::mem::size_of::<TimeValue>() <= 112, "{}", std::mem::size_of::<TimeValue>());
    }
}
