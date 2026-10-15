//! Points: dates, times of day and both, moved by a duration, with their
//! modifiers (`before 2018`, `end of the month`, `around 1pm`).

use turbo_analysis_layers::{Date, DateTime, Modifier, Portion, Relation, TimeGrain, TimeKind};

use super::super::duration::Duration;
use super::super::spec::{DateSpec, Point, Shown, TimeOfDay, YearRef};
use super::super::timex::{Timex, TimexDate, TimexPoint, TimexTime};
use super::dates::{DateTimex, Dated, dates, plus};
use super::{Instant, Moment, Resolved, Span, finish};

/// The grain of a point.
pub(super) fn grain(p: &Point) -> Option<TimeGrain> {
    if !p.shift.is_zero() {
        return p.shift.finest().map(super::unit_grain);
    }
    match p.time {
        TimeOfDay::Clock(c) => Some(match c.shown {
            Shown::Hour => TimeGrain::Hour,
            Shown::Minute => TimeGrain::Minute,
            Shown::Second => TimeGrain::Second,
        }),
        TimeOfDay::Part(_) => Some(TimeGrain::PartOfDay),
        TimeOfDay::Now => Some(TimeGrain::Second),
        TimeOfDay::None => p.date.grain(),
    }
}

pub(super) fn point_timex(date: TimexDate, time: TimexTime) -> Timex {
    Timex::Point(TimexPoint { date, time })
}

/// `d` at `seconds` after its midnight, which may be a day or more on.
pub(super) fn at(d: Date, seconds: u32) -> Option<DateTime> {
    DateTime::midnight(d).plus_seconds(i64::from(seconds))
}

/// `t` moved by `length`, calendar units first.
pub(super) fn shift(t: DateTime, length: &Duration) -> Option<DateTime> {
    let mut out = t;
    for &(unit, milli) in length.parts() {
        out = if unit.is_clock() {
            out.plus_seconds(unit.seconds() * i64::from(milli) / 1000)?
        } else {
            let d = plus(out.date(), &Duration::of(unit, milli))?;
            DateTime::at(d, out.second_of_day())?
        };
    }
    Some(out)
}

fn moment(timex: Timex, span: Option<Span>, future: bool, first: bool) -> Moment {
    Moment { timex, span, future, first }
}

/// The readings of a point and its kind, before any modifier.
pub(super) fn moments(p: &Point, now: DateTime) -> Option<(TimeKind, Vec<Moment>)> {
    let mut out = Vec::new();
    let kind = match (p.date, p.time) {
        (DateSpec::None, TimeOfDay::None) => return None,
        (_, TimeOfDay::Now) => {
            let span = Some(Span::Point(Instant::DateTime(now)));
            out.push(moment(point_timex(TimexDate::Present, TimexTime::None), span, true, true));
            TimeKind::DateTime
        }
        (DateSpec::None, TimeOfDay::Clock(c)) => {
            let (hours, n) = c.hours();
            for (k, &h) in hours[..n].iter().enumerate() {
                let time = TimexTime::Clock { hour: h, minute: c.minute, second: c.second, shown: c.shown };
                let span = Some(Span::Point(Instant::Time(c.seconds_at(h))));
                out.push(moment(point_timex(TimexDate::None, time), span, true, k == 0));
            }
            TimeKind::Time
        }
        (DateSpec::None, TimeOfDay::Part(part)) => {
            let (s, e) = part.seconds();
            let span = Some(Span::Range(Some(Instant::Time(s)), Some(Instant::Time(e))));
            out.push(moment(point_timex(TimexDate::None, TimexTime::Part(part)), span, true, true));
            TimeKind::TimeRange
        }
        (date, time) => {
            let day = date.is_day();
            for (future, d) in dates(date, now)?.readings() {
                push_dated(&mut out, d, time, day, future)?;
            }
            match time {
                TimeOfDay::None if day => TimeKind::Date,
                TimeOfDay::None => TimeKind::DateRange,
                TimeOfDay::Clock(_) => TimeKind::DateTime,
                _ => TimeKind::DateTimeRange,
            }
        }
    };
    if p.shift.is_zero() {
        return Some((kind, out));
    }
    // A shift moves the future reading: `2 weeks from January 15`.
    let base = *out.iter().rev().find(|m| m.future)?;
    let clock = p.shift.parts().iter().any(|(u, _)| u.is_clock());
    // A period moves from its start: `in two years since 2011`.
    let from = match base.span? {
        Span::Point(i) | Span::Range(Some(i), _) => i,
        Span::Range(None, _) => return None,
    };
    let moved = match from {
        Instant::Date(d) if !clock => {
            let d = plus(d, &p.shift)?;
            let timex = point_timex(TimexDate::day(d), TimexTime::None);
            return Some((TimeKind::Date, vec![moment(timex, Some(Span::Point(Instant::Date(d))), true, true)]));
        }
        Instant::Date(d) => shift(DateTime::midnight(d), &p.shift)?,
        Instant::DateTime(t) => shift(t, &p.shift)?,
        Instant::Time(_) => return None,
    };
    let timex = point_timex(TimexDate::day(moved.date()), TimexTime::seconds(moved.second_of_day()));
    Some((TimeKind::DateTime, vec![moment(timex, Some(Span::Point(Instant::DateTime(moved))), true, true)]))
}

fn push_dated(out: &mut Vec<Moment>, d: Dated, time: TimeOfDay, day: bool, future: bool) -> Option<()> {
    let date_timex = match d.timex {
        DateTimex::Date(t) => t,
        DateTimex::Range(s, e, l) if time == TimeOfDay::None => {
            let span = d.span.map(|(s, e)| Span::Range(Some(Instant::Date(s)), Some(Instant::Date(e))));
            let point = |date| TimexPoint { date, time: TimexTime::None };
            out.push(moment(Timex::Range(point(s), point(e), l), span, future, true));
            return Some(());
        }
        DateTimex::Range(..) => return None,
    };
    match time {
        TimeOfDay::None => {
            let span = d.span.map(|(s, e)| {
                if day {
                    Span::Point(Instant::Date(s))
                } else {
                    Span::Range(Some(Instant::Date(s)), Some(Instant::Date(e)))
                }
            });
            out.push(moment(point_timex(date_timex, TimexTime::None), span, future, true));
        }
        TimeOfDay::Clock(c) if day => {
            let (hours, n) = c.hours();
            for (k, &h) in hours[..n].iter().enumerate() {
                let time = TimexTime::Clock { hour: h, minute: c.minute, second: c.second, shown: c.shown };
                let span = match d.span {
                    Some((s, _)) => Some(Span::Point(Instant::DateTime(at(s, c.seconds_at(h))?))),
                    None => None,
                };
                out.push(moment(point_timex(date_timex, time), span, future, k == 0));
            }
        }
        TimeOfDay::Part(part) if day => {
            let (ps, pe) = part.seconds();
            let span = match d.span {
                Some((s, _)) => {
                    Some(Span::Range(Some(Instant::DateTime(at(s, ps)?)), Some(Instant::DateTime(at(s, pe)?))))
                }
                None => None,
            };
            out.push(moment(point_timex(date_timex, TimexTime::Part(part)), span, future, true));
        }
        _ => return None,
    }
    Some(())
}

/// Whether the spec counts its year from the reference (`this year`,
/// `end of year`): such a year splits in halves, a named one in thirds.
fn relative_year(spec: DateSpec) -> bool {
    matches!(
        spec,
        DateSpec::Rel { grain: TimeGrain::Year, .. } | DateSpec::Cal { year: YearRef::Rel(_), month: 0, day: 0, .. }
    )
}

fn date_of(i: Instant) -> Option<Date> {
    match i {
        Instant::Date(d) => Some(d),
        Instant::DateTime(t) => Some(t.date()),
        Instant::Time(_) => None,
    }
}

/// An instant `seconds` after `like`'s, of the same kind.
fn moved(like: Instant, seconds: i64) -> Option<Instant> {
    match like {
        Instant::DateTime(t) => t.plus_seconds(seconds).map(Instant::DateTime),
        Instant::Time(s) => u32::try_from(i64::from(s) + seconds).ok().map(Instant::Time),
        Instant::Date(d) => DateTime::midnight(d).plus_seconds(seconds).map(Instant::DateTime),
    }
}

/// The part of the period `s`..`e` a portion picks; with `clip` a portion
/// of the reference's own period stops at the reference.
#[allow(clippy::too_many_arguments)]
fn portion_of(
    s: Instant,
    e: Instant,
    portion: Portion,
    spec: DateSpec,
    grain: TimeGrain,
    clip: bool,
    now: DateTime,
) -> Option<(Instant, Instant)> {
    if grain == TimeGrain::PartOfDay {
        // A part of the day splits in halves.
        let half = (e.seconds() - s.seconds()) / 2;
        return match portion {
            Portion::Start => Some((s, moved(s, half)?)),
            Portion::End => Some((moved(s, half)?, e)),
            _ => None,
        };
    }
    let (ds, de) = (date_of(s)?, date_of(e)?);
    let today = now.date();
    let early = |cut: Date| if clip { cut.min(today) } else { cut };
    let late = |cut: Date| if clip { cut.max(today) } else { cut };
    let (start, end) = match grain {
        TimeGrain::Week => {
            let thursday = ds.plus_days(3)?;
            match portion {
                Portion::Start => (ds, early(thursday)),
                Portion::Mid => (ds.plus_days(1)?, ds.plus_days(4)?),
                Portion::End => (late(thursday), de),
                _ => return None,
            }
        }
        TimeGrain::Month => {
            let at = |d: u8| Date::new(ds.year(), ds.month(), d);
            match portion {
                Portion::Start => (ds, early(at(16)?)),
                Portion::Mid => (at(10)?, at(21)?),
                Portion::End => (late(at(16)?), de),
                _ => return None,
            }
        }
        TimeGrain::Year if relative_year(spec) => {
            let july = Date::new(ds.year(), 7, 1)?;
            match portion {
                Portion::Start => (ds, early(july)),
                Portion::Mid => (Date::new(ds.year(), 5, 1)?, Date::new(ds.year(), 9, 1)?),
                Portion::End => (late(july), de),
                _ => return None,
            }
        }
        // A decade splits by years as a year does by months: `the early
        // eighties` is 1980 to 1983, the middle 1984 to 1986, the late
        // 1987 to 1989.
        TimeGrain::Decade => {
            let at = |n: i32| Date::new(ds.year() + n, 1, 1);
            match portion {
                Portion::Start => (ds, early(at(4)?)),
                Portion::Mid => (at(4)?, at(7)?),
                Portion::End => (late(at(7)?), de),
                _ => return None,
            }
        }
        TimeGrain::Year => {
            let at = |m: u8| Date::new(ds.year(), m, 1);
            match portion {
                Portion::Start => (ds, at(5)?),
                Portion::Mid => (at(5)?, at(9)?),
                Portion::End => (at(9)?, de),
                _ => return None,
            }
        }
        _ => return None,
    };
    Some((Instant::Date(start), Instant::Date(end)))
}

/// A day's portion as date-times: `early in the day` is its first half,
/// `mid today` 10:00 to 14:00, `later in today` its second half.
fn day_portion(d: Date, portion: Portion) -> Option<(Instant, Instant)> {
    let (a, b) = match portion {
        Portion::Start => (0, 12),
        Portion::Mid => (10, 14),
        Portion::End => (12, 24),
        _ => return None,
    };
    Some((Instant::DateTime(at(d, a * 3600)?), Instant::DateTime(at(d, b * 3600)?)))
}

pub(super) fn range_kind(kind: TimeKind) -> TimeKind {
    match kind {
        TimeKind::Date => TimeKind::DateRange,
        TimeKind::Time => TimeKind::TimeRange,
        TimeKind::DateTime => TimeKind::DateTimeRange,
        k => k,
    }
}

/// The end of the day `t` names, to the second: `end of tomorrow`.
fn end_of_day(m: &mut Moment, d: Date) -> Option<()> {
    let t = DateTime::new(d, 23, 59, 59)?;
    if let Timex::Point(p) = m.timex {
        let time = TimexTime::Clock { hour: 23, minute: 59, second: 59, shown: Shown::Second };
        m.timex = Timex::Point(TimexPoint { time, ..p });
    }
    m.span = Some(Span::Point(Instant::DateTime(t)));
    Some(())
}

/// Applies a modifier to the readings of a point; gives the new kind.
pub(super) fn modify(
    kind: TimeKind,
    moments: &mut [Moment],
    modifier: Modifier,
    clip: bool,
    spec: DateSpec,
    grain: TimeGrain,
    now: DateTime,
) -> Option<TimeKind> {
    if modifier.relation == Relation::Approx || modifier.portion == Portion::Undefined || modifier.is_none() {
        return Some(kind);
    }
    let portion = modifier.portion;
    let mut out = kind;
    for m in moments.iter_mut() {
        let Some(span) = m.span else { continue };
        let (s, e) = match span {
            Span::Point(s) => (s, s),
            Span::Range(Some(s), Some(e)) => (s, e),
            Span::Range(..) => return None,
        };
        let (ps, pe) = match portion {
            Portion::Start | Portion::Mid | Portion::End if kind == TimeKind::Date => {
                let d = date_of(s)?;
                if portion == Portion::End && !clip && modifier.relation == Relation::None {
                    end_of_day(m, d)?;
                    out = TimeKind::DateTime;
                    continue;
                }
                out = TimeKind::DateTimeRange;
                day_portion(d, portion)?
            }
            Portion::Start | Portion::Mid | Portion::End => portion_of(s, e, portion, spec, grain, clip, now)?,
            _ => (s, e),
        };
        m.span = Some(match modifier.relation {
            Relation::None => Span::Range(Some(ps), Some(pe)),
            Relation::Before => Span::Range(None, Some(if portion == Portion::End { pe } else { ps })),
            Relation::After => Span::Range(Some(if portion == Portion::Start { ps } else { pe }), None),
            Relation::Since => Span::Range(Some(ps), None),
            Relation::Until => Span::Range(None, Some(pe)),
            _ => return None,
        });
        if modifier.relation != Relation::None {
            out = range_kind(kind);
        }
    }
    Some(out)
}

pub(super) fn resolve(p: &Point, modifier: Modifier, clip: bool, now: DateTime) -> Option<Resolved> {
    let (kind, mut ms) = moments(p, now)?;
    let grain = grain(p)?;
    let modified = modify(kind, &mut ms, modifier, clip, p.date, grain, now)?;
    // Recognizers-Text names no modifier for the end of a day (`end of
    // tomorrow` is 23:59:59) or for a clipped period (`later this week`).
    let end_of_day = kind == TimeKind::Date && modified == TimeKind::DateTime;
    let clipped = clip && matches!(grain, TimeGrain::Week | TimeGrain::Month | TimeGrain::Year);
    let shown = if end_of_day || clipped { Modifier::NONE } else { modifier };
    let mut r = finish(modified, grain, shown, &ms);
    // A decade without its century (`the eighties`) means the last one
    // that starts by the reference far more often than the next.
    if matches!(p.date, DateSpec::Decade { year: YearRef::Short(_) })
        && let Some(i) =
            ms.iter().find(|m| !m.future).and_then(|m| r.alternatives.iter().position(|a| *a == m.alternative()))
    {
        r.preferred = i;
    }
    Some(r)
}
