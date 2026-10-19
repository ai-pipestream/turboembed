//! Dates, times, durations and sets written in a text view, resolved
//! against a reference date-time: `May 29`, `tomorrow at 8am`, `next 3
//! days`, `from 5 to 6pm`, `3.5 years`, `every Monday`.
//!
//! A mention holds its kind, its grain, a modifier (`before 2018`, `end of
//! the month`), the resolved start and end, a duration, whether the text
//! allows more than one reading, and its TIMEX3 value (`2016-11-08T08:00`,
//! `XXXX-WXX-5`, `(2016-11-08,2016-11-11,P3D)`, `P1W`).
//!
//! Start and end are local seconds: for the kinds with a date, seconds
//! since 1970-01-01T00:00 on the text's own clock (no time zone, no leap
//! seconds); for [`TimeKind::Time`] and [`TimeKind::TimeRange`], which
//! name no date, seconds since midnight, so a time range that crosses
//! midnight ends below its start. A point has its start and no end. An
//! open range (`after 2016`) has one of the two. A mention that names a
//! date relative to a reference (`tomorrow`, `next week`) found no
//! reference when it has neither and an empty timex: the kind and grain
//! say what it is. A set (`every Monday`) and a duration have neither.

use std::fmt;

use turbo_analysis::{Ix, Layer, Span, Spans, ViewId};

use crate::{DAY_SECONDS, Strs};

/// What a mention names.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TimeKind {
    /// A day: `May 29`, `tomorrow`.
    Date,
    /// A time of day with no date: `7:56:30 pm`.
    Time,
    /// A date and a time of day: `tomorrow 8:00am`.
    DateTime,
    /// A span of days: `next week`, `2016`, `from 4-23 in next month`.
    DateRange,
    /// A span of the day with no date: `5 to 6pm`, `in the morning`.
    TimeRange,
    /// A span with dates and times: `this evening`, `Monday 8-9am`.
    DateTimeRange,
    /// A length of time: `3 minutes`, `P1DT5H`.
    Duration,
    /// A recurrence: `every Monday`, `weekly`.
    Set,
}

impl TimeKind {
    /// The Recognizers-Text type name, after `datetimeV2.`: `date`,
    /// `time`, `datetime`, `daterange`, `timerange`, `datetimerange`,
    /// `duration` or `set`.
    pub fn name(self) -> &'static str {
        match self {
            TimeKind::Date => "date",
            TimeKind::Time => "time",
            TimeKind::DateTime => "datetime",
            TimeKind::DateRange => "daterange",
            TimeKind::TimeRange => "timerange",
            TimeKind::DateTimeRange => "datetimerange",
            TimeKind::Duration => "duration",
            TimeKind::Set => "set",
        }
    }

    /// Whether the kind names no date, so its start and end are seconds
    /// since midnight.
    pub fn is_time_of_day(self) -> bool {
        matches!(self, TimeKind::Time | TimeKind::TimeRange)
    }
}

/// The finest unit a mention states, from a second to a century.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TimeGrain {
    Second,
    Minute,
    Hour,
    /// A part of the day: morning, afternoon, evening, night.
    PartOfDay,
    Day,
    Week,
    Weekend,
    Month,
    Quarter,
    Half,
    Season,
    Year,
    Decade,
    Century,
}

/// How a modifier bounds the mention: `before 2018` is the time up to its
/// start, `after 2016` the time from its end.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Relation {
    #[default]
    None,
    /// `before`, `by`, `no later than`: up to the mention.
    Before,
    /// `after`, `later than`: from the end of the mention.
    After,
    /// `since`, `starting`, `or later`: from the start of the mention.
    Since,
    /// `until`, `as late as`: up to the end of the mention.
    Until,
    /// A duration of at least: `more than 4 days`.
    More,
    /// A duration of at most: `less than 1 week`.
    Less,
    /// `around`, `circa`.
    Approx,
}

/// Which part of the mention a modifier picks.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Portion {
    #[default]
    Whole,
    /// `early`, `beginning of`.
    Start,
    /// `mid`, `middle of`.
    Mid,
    /// `late`, `end of`.
    End,
    /// `around` before a mention that another relation bounds: `before
    /// around 2pm`.
    Approx,
    /// A period named without saying which: `the same week`, `that
    /// weekend`.
    Undefined,
}

/// A mention's modifier: a relation, a portion, or both (`since the end
/// of 1989`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Modifier {
    pub relation: Relation,
    pub portion: Portion,
}

impl Modifier {
    pub const NONE: Modifier = Modifier { relation: Relation::None, portion: Portion::Whole };

    pub fn relation(relation: Relation) -> Modifier {
        Modifier { relation, portion: Portion::Whole }
    }

    pub fn portion(portion: Portion) -> Modifier {
        Modifier { relation: Relation::None, portion }
    }

    pub fn is_none(self) -> bool {
        self == Modifier::NONE
    }
}

/// Writes the modifier as Recognizers-Text names it: `before`, `mid`,
/// `since-end`, `before-approx`, `ref_undef`, or nothing for none.
impl fmt::Display for Modifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let relation = match self.relation {
            Relation::None => "",
            Relation::Before => "before",
            Relation::After => "after",
            Relation::Since => "since",
            Relation::Until => "until",
            Relation::More => "more",
            Relation::Less => "less",
            Relation::Approx => "approx",
        };
        let portion = match self.portion {
            Portion::Whole => "",
            Portion::Start => "start",
            Portion::Mid => "mid",
            Portion::End => "end",
            Portion::Approx => "approx",
            Portion::Undefined => "ref_undef",
        };
        let dash = if relation.is_empty() || portion.is_empty() { "" } else { "-" };
        write!(f, "{relation}{dash}{portion}")
    }
}

/// Every time mention in a view, one item per mention, in text order and
/// not overlapping. See the module documentation for what the start and
/// end columns hold.
pub struct Times {
    pub spans: Spans<Times>,
    pub kind: Vec<TimeKind>,
    pub grain: Vec<TimeGrain>,
    pub modifier: Vec<Modifier>,
    pub start: Vec<Option<i64>>,
    pub end: Vec<Option<i64>>,
    /// The length in milliseconds, of a duration and of a range that
    /// states one.
    pub duration_ms: Vec<Option<i64>>,
    /// Whether the text reads more than one way (`Friday`, past or next;
    /// `at 5`, morning or afternoon); the columns hold the reading nearest
    /// after the reference.
    pub ambiguous: Vec<bool>,
    /// The TIMEX3 value of that reading.
    pub timex: Strs,
}

/// One mention, as handed to [`Times::push`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TimeMention<'a> {
    pub span: Span,
    pub kind: TimeKind,
    pub grain: TimeGrain,
    pub modifier: Modifier,
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub duration_ms: Option<i64>,
    pub ambiguous: bool,
    pub timex: &'a str,
}

impl Times {
    /// An empty layer over `view`.
    pub fn new(view: ViewId) -> Times {
        Times {
            spans: Spans::new(view),
            kind: Vec::new(),
            grain: Vec::new(),
            modifier: Vec::new(),
            start: Vec::new(),
            end: Vec::new(),
            duration_ms: Vec::new(),
            ambiguous: Vec::new(),
            timex: Strs::default(),
        }
    }

    /// Appends one mention to every column.
    pub fn push(&mut self, m: TimeMention<'_>) -> Ix<Times> {
        self.kind.push(m.kind);
        self.grain.push(m.grain);
        self.modifier.push(m.modifier);
        self.start.push(m.start);
        self.end.push(m.end);
        self.duration_ms.push(m.duration_ms);
        self.ambiguous.push(m.ambiguous);
        self.timex.push(m.timex);
        self.spans.push(m.span)
    }

    /// Removes every mention, keeping the view.
    pub fn clear(&mut self) {
        *self = Times::new(self.spans.view());
    }

    /// Mention `i`, column by column.
    pub fn get(&self, i: Ix<Times>) -> TimeMention<'_> {
        let k = i.get();
        TimeMention {
            span: self.spans.get(i),
            kind: self.kind[k],
            grain: self.grain[k],
            modifier: self.modifier[k],
            start: self.start[k],
            end: self.end[k],
            duration_ms: self.duration_ms[k],
            ambiguous: self.ambiguous[k],
            timex: self.timex.get(k),
        }
    }

    fn check_item(&self, i: usize) -> Result<(), String> {
        let (kind, start, end) = (self.kind[i], self.start[i], self.end[i]);
        if kind.is_time_of_day() {
            for s in [start, end].into_iter().flatten() {
                if !(0..DAY_SECONDS).contains(&s) {
                    return Err(format!("time mention {i} has {s}, not a second of the day"));
                }
            }
        } else if let (Some(s), Some(e)) = (start, end)
            && e < s
        {
            return Err(format!("time mention {i} ends at {e}, before its start {s}"));
        }
        if matches!(kind, TimeKind::Set | TimeKind::Duration) && (start.is_some() || end.is_some()) {
            return Err(format!("time mention {i} is a {} with a start or an end", kind.name()));
        }
        if self.duration_ms[i].is_some_and(|d| d < 0) {
            return Err(format!("time mention {i} has a negative duration"));
        }
        if self.timex.get(i).is_empty() && (start.is_some() || end.is_some()) {
            return Err(format!("time mention {i} is resolved and has no timex"));
        }
        Ok(())
    }
}

impl Layer for Times {
    const ID: &'static str = "turbo:times";
    positional!(Times);

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let columns = [
            self.kind.len(),
            self.grain.len(),
            self.modifier.len(),
            self.start.len(),
            self.end.len(),
            self.duration_ms.len(),
            self.ambiguous.len(),
            self.timex.len(),
        ];
        if columns.iter().any(|&l| l != n) {
            return Err("every column as long as the spans".into());
        }
        self.spans.check(Self::OVERLAP).map_err(|_| "time mentions in text order, not overlapping")?;
        for (i, span) in self.spans.iter() {
            if span.is_empty() {
                return Err(format!("time mention {} is empty at {}", i.get(), span.start));
            }
            self.check_item(i.get())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(start: u32, end: u32, kind: TimeKind, timex: &str) -> TimeMention<'_> {
        TimeMention {
            span: Span::new(start, end),
            kind,
            grain: TimeGrain::Day,
            modifier: Modifier::NONE,
            start: None,
            end: None,
            duration_ms: None,
            ambiguous: false,
            timex,
        }
    }

    #[test]
    fn mentions_round_trip() {
        let mut t = Times::new(ViewId::SOURCE);
        t.push(TimeMention { start: Some(1_478_476_800), ..m(0, 5, TimeKind::Date, "2016-11-07") });
        t.push(TimeMention {
            start: Some(61_200),
            end: Some(64_800),
            duration_ms: Some(3_600_000),
            grain: TimeGrain::Hour,
            ..m(6, 14, TimeKind::TimeRange, "(T17,T18,PT1H)")
        });
        t.push(TimeMention { duration_ms: Some(180_000), ..m(15, 24, TimeKind::Duration, "PT3M") });
        t.push(m(25, 33, TimeKind::Date, ""));
        assert_eq!(t.check(), Ok(()));
        assert_eq!(t.len(), 4);
        assert_eq!(t.get(Ix::new(1)).timex, "(T17,T18,PT1H)");
        assert_eq!(t.get(Ix::new(2)).duration_ms, Some(180_000));
        t.clear();
        assert_eq!((t.len(), t.spans.view()), (0, ViewId::SOURCE));
    }

    #[test]
    fn check_refuses_values_that_do_not_fit_the_kind() {
        let cases = [
            TimeMention { start: Some(90_000), ..m(0, 4, TimeKind::Time, "T25") },
            TimeMention { start: Some(10), end: Some(5), ..m(0, 4, TimeKind::DateRange, "x") },
            TimeMention { start: Some(0), ..m(0, 4, TimeKind::Set, "P1D") },
            TimeMention { duration_ms: Some(-1), ..m(0, 4, TimeKind::Duration, "P1D") },
            TimeMention { start: Some(0), ..m(0, 4, TimeKind::Date, "") },
        ];
        for c in cases {
            let mut t = Times::new(ViewId::SOURCE);
            t.push(c);
            assert!(t.check().is_err(), "{c:?}");
        }
        let mut t = Times::new(ViewId::SOURCE);
        t.push(TimeMention { start: Some(77_400), end: Some(27_000), ..m(0, 4, TimeKind::TimeRange, "x") });
        assert_eq!(t.check(), Ok(()), "a time range may cross midnight");
    }

    #[test]
    fn check_wants_order_columns_and_text() {
        let mut t = Times::new(ViewId::SOURCE);
        t.push(m(4, 6, TimeKind::Date, "x"));
        t.push(m(0, 2, TimeKind::Date, "x"));
        assert!(t.check().unwrap_err().contains("text order"));
        let mut t = Times::new(ViewId::SOURCE);
        t.push(m(4, 6, TimeKind::Date, "x"));
        t.ambiguous.pop();
        assert_eq!(t.check().unwrap_err(), "every column as long as the spans");
        let mut t = Times::new(ViewId::SOURCE);
        t.push(m(4, 4, TimeKind::Date, "x"));
        assert!(t.check().unwrap_err().contains("empty"));
    }

    #[test]
    fn names_and_modifiers() {
        assert_eq!(Times::ID, "turbo:times");
        assert_eq!(TimeKind::DateTimeRange.name(), "datetimerange");
        assert!(TimeKind::TimeRange.is_time_of_day() && !TimeKind::DateTime.is_time_of_day());
        let cases = [
            (Modifier::NONE, ""),
            (Modifier::relation(Relation::Before), "before"),
            (Modifier::portion(Portion::Mid), "mid"),
            (Modifier { relation: Relation::Since, portion: Portion::End }, "since-end"),
            (Modifier { relation: Relation::Before, portion: Portion::Approx }, "before-approx"),
            (Modifier::portion(Portion::Approx), "approx"),
            (Modifier::portion(Portion::Undefined), "ref_undef"),
            (Modifier::relation(Relation::More), "more"),
        ];
        for (modifier, name) in cases {
            assert_eq!(modifier.to_string(), name);
        }
        assert!(Modifier::NONE.is_none() && !Modifier::relation(Relation::Less).is_none());
        assert!(TimeGrain::Day < TimeGrain::Week && TimeGrain::Year < TimeGrain::Century);
    }
}
