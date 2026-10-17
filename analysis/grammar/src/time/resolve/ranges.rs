//! Ranges: between two points, from a point for a duration, the coming
//! or past stretch of time, and the time within a duration from the
//! reference.

use turbo_analysis_layers::{Date, DateTime, Modifier, TimeGrain, TimeKind};

use super::super::duration::{Duration, MILLI, Unit};
use super::super::spec::{Clock, DateSpec, Edge, Meridiem, Point, Range, Shown, TimeOfDay, YearRef};
use super::super::timex::{Timex, TimexDate, TimexPoint, TimexTime, TimexYear};
use super::dates::{Choice, DateTimex, Dated, dates, plus, relative};
use super::points::{at, point_timex, shift};
use super::{Instant, Moment, Resolved, Span, finish};

pub(super) fn resolve(r: &Range, modifier: Modifier, now: DateTime) -> Option<Resolved> {
    let (kind, grain, ms) = match *r {
        Range::Between { start, start_edge, end, end_edge } => between(&start, start_edge, &end, end_edge, now)?,
        Range::For { start, length } => for_length(&start, &length, now)?,
        Range::Next { length, align } => next(&length, align, now)?,
        Range::Within { length } => within(&length, now)?,
    };
    Some(finish(kind, grain, modifier, &ms))
}

/// A duration of `seconds` in hours, minutes and seconds: `PT2H30M`.
fn clock_duration(seconds: i64) -> Option<Duration> {
    if seconds < 0 {
        return None;
    }
    let mut d = Duration::ZERO;
    for (unit, size) in [(Unit::Hour, 3600), (Unit::Minute, 60), (Unit::Second, 1)] {
        let n = seconds / size % if size == 3600 { i64::MAX } else { 60 };
        if n > 0 {
            d = d.plus(&Duration::units(unit, i32::try_from(n).ok()?)?)?;
        }
    }
    Some(d)
}

/// Whole days, months or years from `a` to `b`, in the unit the bounds'
/// grains call for.
fn date_duration(a: Date, b: Date, unit: Unit) -> Option<Duration> {
    let n = match unit {
        Unit::Year => b.year().checked_sub(a.year())?,
        Unit::Month => i32::try_from(
            (i64::from(b.year()) - i64::from(a.year())) * 12 + i64::from(b.month()) - i64::from(a.month()),
        )
        .ok()?,
        _ => i32::try_from(b.epoch_day() - a.epoch_day()).ok()?,
    };
    Duration::units(unit, n)
}

fn day_timex(d: Date, open_year: bool) -> TimexPoint {
    let year = if open_year { TimexYear::Any } else { TimexYear::Year(d.year()) };
    TimexPoint { date: TimexDate::Ymd { year, month: Some(d.month()), day: Some(d.day()) }, time: TimexTime::None }
}

/// Whether a reading leaves its year open (`XXXX-05`).
fn open_year(d: &Dated) -> bool {
    let date = match d.timex {
        DateTimex::Date(t) => t,
        DateTimex::Range(s, _, _) => s,
    };
    matches!(date, TimexDate::Ymd { year: TimexYear::Any, .. } | TimexDate::MonthWeek { year: TimexYear::Any, .. })
}

/// The instant of a period an edge takes: its start, its middle (the
/// 16th of a month, July 2 of a year) or its end.
fn edge_of(span: (Date, Date), edge: Edge, grain: TimeGrain) -> Option<Date> {
    let (s, e) = span;
    match edge {
        Edge::Start => Some(s),
        Edge::End => Some(e),
        Edge::Mid => match grain {
            TimeGrain::Month => Date::new(s.year(), s.month(), 16),
            TimeGrain::Year => Date::new(s.year(), 7, 2),
            _ => s.plus_days((e.epoch_day() - s.epoch_day()) / 2),
        },
    }
}

/// The date of a point that names only a date, or `now`'s.
fn date_choice(p: &Point, now: DateTime) -> Option<Choice> {
    match (p.date, p.time) {
        (DateSpec::None, TimeOfDay::Now) => {
            relative(TimeGrain::Day, 0, now.date()).map(|d| Choice { past: d, future: d })
        }
        (date, TimeOfDay::None) if p.shift.is_zero() => dates(date, now),
        _ => None,
    }
}

fn coarse(g: TimeGrain) -> bool {
    matches!(g, TimeGrain::Month | TimeGrain::Quarter | TimeGrain::Half | TimeGrain::Year)
}

fn between(
    start: &Point,
    se: Edge,
    end: &Point,
    ee: Edge,
    now: DateTime,
) -> Option<(TimeKind, TimeGrain, Vec<Moment>)> {
    if let (Some(cs), Some(ce)) = (date_choice(start, now), date_choice(end, now)) {
        let gs = if start.time == TimeOfDay::Now { TimeGrain::Day } else { start.date.grain()? };
        let ge = if end.time == TimeOfDay::Now { TimeGrain::Day } else { end.date.grain()? };
        return date_range(&cs, se, gs, &ce, ee, ge).map(|ms| (TimeKind::DateRange, gs.min(ge), ms));
    }
    clock_range(start, end, now)
}

fn date_range(cs: &Choice, se: Edge, gs: TimeGrain, ce: &Choice, ee: Edge, ge: TimeGrain) -> Option<Vec<Moment>> {
    let unit = if gs == TimeGrain::Year && ge == TimeGrain::Year && se == Edge::Start && ee == Edge::Start {
        Unit::Year
    } else if coarse(gs) && coarse(ge) {
        Unit::Month
    } else {
        Unit::Day
    };
    let bound = |d: &Dated, edge: Edge, g: TimeGrain| edge_of(d.span?, edge, g);
    let (ps, pe) = (bound(&cs.past, se, gs)?, bound(&ce.past, ee, ge)?);
    let (fs, fe) = (bound(&cs.future, se, gs)?, bound(&ce.future, ee, ge)?);
    // A range whose end comes before its start takes the end's later
    // reading in the past and the start's earlier one in the future.
    let past = if pe < ps { (ps, bound(&ce.future, ee, ge)?) } else { (ps, pe) };
    let future = if fe < fs { (bound(&cs.past, se, gs)?, fe) } else { (fs, fe) };
    let (os, oe) = (open_year(&cs.past), open_year(&ce.past));
    let mut out = Vec::with_capacity(2);
    for (is_future, (s, e)) in [(false, past), (true, future)] {
        if e < s {
            return None;
        }
        let timex = Timex::Range(day_timex(s, os), day_timex(e, oe), date_duration(s, e, unit)?);
        let span = Some(Span::Range(Some(Instant::Date(s)), Some(Instant::Date(e))));
        out.push(Moment { timex, span, future: is_future, first: true });
    }
    Some(out)
}

/// Whether a clock says which half of the day it is in.
fn known(c: &Clock) -> bool {
    c.meridiem != Meridiem::Unknown || c.hour > 12 || c.hour == 0
}

/// The readings of a clock range as seconds after midnight, the end on
/// the next day when it comes before the start: the halves of the day an
/// open bound reads in follow from the other bound, and two open bounds
/// read in both halves.
fn clock_pairs(s: &Clock, e: &Clock) -> Vec<(u32, u32, u8, u8)> {
    let secs = |c: &Clock, h: u8| c.seconds_at(h);
    let after = |c: &Clock, from: u32| -> (u32, u8) {
        let (hours, n) = c.hours();
        let mut best: Option<(u32, u8)> = None;
        for &h in &hours[..n] {
            for wrap in [0, 86_400] {
                let t = secs(c, h) + wrap;
                if t > from && best.is_none_or(|(b, _)| t < b) {
                    best = Some((t, h));
                }
            }
        }
        best.unwrap_or((secs(c, hours[0]) + 86_400, hours[0]))
    };
    let (sh, sn) = s.hours();
    match (known(s) || sn == 1, known(e)) {
        (true, true) => {
            let eh = e.hours().0[0];
            let et = secs(e, eh);
            let st = secs(s, sh[0]);
            let et = if et <= st { et + 86_400 } else { et };
            vec![(st, et, sh[0], eh)]
        }
        (false, true) => {
            let eh = e.hours().0[0];
            let et = secs(e, eh);
            let pick =
                sh[..sn].iter().copied().filter(|&h| secs(s, h) < et).max_by_key(|&h| secs(s, h)).unwrap_or(sh[0]);
            let st = secs(s, pick);
            vec![(st, if et <= st { et + 86_400 } else { et }, pick, eh)]
        }
        (true, false) => {
            let st = secs(s, sh[0]);
            let (et, eh) = after(e, st);
            vec![(st, et, sh[0], eh)]
        }
        // `10:30 to 3`: a morning start and an afternoon end.
        (false, false) if s.hour > e.hour && e.hour < 12 && s.shown != Shown::Hour => {
            let (st, eh) = (secs(s, sh[0]), e.hour + 12);
            vec![(st, secs(e, eh), sh[0], eh)]
        }
        (false, false) => sh[..sn]
            .iter()
            .map(|&h| {
                let st = secs(s, h);
                let (et, eh) = after(e, st);
                (st, et, h, eh)
            })
            .collect(),
    }
}

fn clock_of(p: &Point) -> Option<Clock> {
    match p.time {
        TimeOfDay::Clock(c) => Some(c),
        _ => None,
    }
}

/// The next day's form of a date's TIMEX3 value, for a range that ends
/// after midnight.
fn next_day(t: TimexDate, d: Date) -> TimexDate {
    match t {
        TimexDate::Weekday(w) => TimexDate::Weekday(w % 7 + 1),
        TimexDate::Ymd { year: TimexYear::Any, .. } => {
            TimexDate::Ymd { year: TimexYear::Any, month: Some(d.month()), day: Some(d.day()) }
        }
        _ => TimexDate::day(d),
    }
}

fn clock_timex(h: u8, c: &Clock) -> TimexTime {
    TimexTime::Clock { hour: h, minute: c.minute, second: c.second, shown: c.shown }
}

fn clock_range(start: &Point, end: &Point, now: DateTime) -> Option<(TimeKind, TimeGrain, Vec<Moment>)> {
    if !start.shift.is_zero() || !end.shift.is_zero() {
        return now_to_shift(start, end, now);
    }
    let today = || DateSpec::Rel { grain: TimeGrain::Day, n: 0 };
    let now_clock = |p: &Point| {
        (p.time == TimeOfDay::Now).then(|| Clock {
            hour: now.hour(),
            minute: now.minute(),
            second: now.second(),
            shown: Shown::Second,
            meridiem: Meridiem::Day,
        })
    };
    let s = clock_of(start).or_else(|| now_clock(start))?;
    let e = clock_of(end).or_else(|| now_clock(end))?;
    let date = match (start.date, end.date) {
        (DateSpec::None, DateSpec::None) if start.time == TimeOfDay::Now || end.time == TimeOfDay::Now => today(),
        (DateSpec::None, d) | (d, DateSpec::None) => d,
        (a, b) if a == b => a,
        _ => return None,
    };
    let pairs = clock_pairs(&s, &e);
    let grain = super::points::grain(start)?;
    let mut out = Vec::new();
    let length = |st: u32, et: u32| clock_duration(i64::from(et) - i64::from(st));
    if date == DateSpec::None {
        for (k, &(st, et, sh, eh)) in pairs.iter().enumerate() {
            let timex = Timex::Range(
                TimexPoint { date: TimexDate::None, time: clock_timex(sh, &s) },
                TimexPoint { date: TimexDate::None, time: clock_timex(eh, &e) },
                length(st, et)?,
            );
            let span = Some(Span::Range(Some(Instant::Time(st)), Some(Instant::Time(et % 86_400))));
            out.push(Moment { timex, span, future: true, first: k == 0 });
        }
        return Some((TimeKind::TimeRange, grain, out));
    }
    let present = |p: &Point| p.time == TimeOfDay::Now;
    for (future, d) in dates(date, now)?.readings() {
        let DateTimex::Date(dt) = d.timex else { return None };
        let (day, _) = d.span?;
        for (k, &(st, et, sh, eh)) in pairs.iter().enumerate() {
            let (a, b) = (at(day, st)?, at(day, et)?);
            let end_date = if b.date() != day { next_day(dt, b.date()) } else { dt };
            let sp = if present(start) {
                point_timex(TimexDate::Present, TimexTime::None)
            } else {
                point_timex(dt, clock_timex(sh, &s))
            };
            let ep = if present(end) {
                point_timex(TimexDate::Present, TimexTime::None)
            } else {
                point_timex(end_date, clock_timex(eh, &e))
            };
            let (Timex::Point(sp), Timex::Point(ep)) = (sp, ep) else { return None };
            let timex = Timex::Range(sp, ep, length(st, et)?);
            let span = Some(Span::Range(Some(Instant::DateTime(a)), Some(Instant::DateTime(b))));
            out.push(Moment { timex, span, future, first: k == 0 });
        }
    }
    Some((TimeKind::DateTimeRange, grain, out))
}

/// `from now until 3 min later`: now to a shifted now.
fn now_to_shift(start: &Point, end: &Point, now: DateTime) -> Option<(TimeKind, TimeGrain, Vec<Moment>)> {
    if start.time != TimeOfDay::Now || end.time != TimeOfDay::Now || !start.shift.is_zero() || end.shift.millis() < 0 {
        return None;
    }
    let b = shift(now, &end.shift)?;
    if b < now {
        return None;
    }
    let end_timex = TimexPoint { date: TimexDate::day(b.date()), time: TimexTime::seconds(b.second_of_day()) };
    let timex = Timex::Range(TimexPoint { date: TimexDate::Present, time: TimexTime::None }, end_timex, end.shift);
    let span = Some(Span::Range(Some(Instant::DateTime(now)), Some(Instant::DateTime(b))));
    let grain = super::points::grain(end)?;
    Some((TimeKind::DateTimeRange, grain, vec![Moment { timex, span, future: true, first: true }]))
}

fn datetime_timex(t: DateTime) -> TimexPoint {
    TimexPoint { date: TimexDate::day(t.date()), time: TimexTime::seconds(t.second_of_day()) }
}

fn for_length(start: &Point, length: &Duration, now: DateTime) -> Option<(TimeKind, TimeGrain, Vec<Moment>)> {
    if length.millis() < 0 {
        return None;
    }
    let grain = super::points::grain(start)?;
    match (start.date, start.time) {
        (DateSpec::None, TimeOfDay::Clock(c)) => {
            let (hours, n) = c.hours();
            let mut out = Vec::new();
            for (k, &h) in hours[..n].iter().enumerate() {
                let st = c.seconds_at(h);
                let et = i64::from(st).checked_add(length.millis() / 1000)?;
                let et = u32::try_from(et).ok()?;
                let timex = Timex::Range(
                    TimexPoint { date: TimexDate::None, time: clock_timex(h, &c) },
                    TimexPoint { date: TimexDate::None, time: TimexTime::seconds(et) },
                    clock_duration(i64::from(et) - i64::from(st))?,
                );
                let span = Some(Span::Range(Some(Instant::Time(st)), Some(Instant::Time(et % 86_400))));
                out.push(Moment { timex, span, future: true, first: k == 0 });
            }
            Some((TimeKind::TimeRange, grain, out))
        }
        (date, TimeOfDay::Clock(c)) => {
            let (hours, n) = c.hours();
            let mut out = Vec::new();
            for (future, d) in dates(date, now)?.readings() {
                let DateTimex::Date(dt) = d.timex else { return None };
                for (k, &h) in hours[..n].iter().enumerate() {
                    let a = at(d.span?.0, c.seconds_at(h))?;
                    let b = shift(a, length)?;
                    if b < a {
                        return None;
                    }
                    let end_date = if b.date() == a.date() { dt } else { next_day(dt, b.date()) };
                    let timex = Timex::Range(
                        TimexPoint { date: dt, time: clock_timex(h, &c) },
                        TimexPoint { date: end_date, time: TimexTime::seconds(b.second_of_day()) },
                        clock_duration(b.local_seconds() - a.local_seconds())?,
                    );
                    let span = Some(Span::Range(Some(Instant::DateTime(a)), Some(Instant::DateTime(b))));
                    out.push(Moment { timex, span, future, first: k == 0 });
                }
            }
            Some((TimeKind::DateTimeRange, grain, out))
        }
        (date, TimeOfDay::None) if date.is_day() => {
            // A day the text gives no year for counts in the reference's
            // year: `2 weeks starting May 20th`.
            let date = match date {
                DateSpec::Cal { year: YearRef::Any, month, day, weekday } if month > 0 => {
                    DateSpec::Cal { year: YearRef::Rel(0), month, day, weekday }
                }
                d => d,
            };
            let choice = dates(date, now)?;
            let s = choice.future.span?.0;
            let e = plus(s, length)?;
            if e < s {
                return None;
            }
            let timex = Timex::Range(day_timex(s, false), day_timex(e, false), *length);
            let span = Some(Span::Range(Some(Instant::Date(s)), Some(Instant::Date(e))));
            Some((TimeKind::DateRange, TimeGrain::Day, vec![Moment { timex, span, future: true, first: true }]))
        }
        _ => None,
    }
}

/// Working days from `from`, `n` of them forward or back, skipping
/// Saturdays and Sundays: gives the first and the one past the last.
fn business_days(from: Date, n: i32) -> Option<(Date, Date)> {
    let step = if n > 0 { 1 } else { -1 };
    let mut d = from;
    let mut days = Vec::new();
    while days.len() < n.unsigned_abs() as usize {
        d = d.plus_days(step)?;
        if d.day_of_week() <= 5 {
            days.push(d);
        }
    }
    let (first, last) = (*days.iter().min()?, *days.iter().max()?);
    if n > 0 { Some((first, last.plus_days(1)?)) } else { Some((first, from)) }
}

fn next(length: &Duration, align: TimeGrain, now: DateTime) -> Option<(TimeKind, TimeGrain, Vec<Moment>)> {
    let back = length.parts().first()?.1 < 0;
    let forward = if back { length.negated() } else { *length };
    if forward.millis() < 0 {
        return None;
    }
    let grain = super::unit_grain(length.finest()?);
    let one = |timex, span| vec![Moment { timex, span: Some(span), future: true, first: true }];
    if length.parts().iter().any(|(u, _)| u.is_clock()) {
        let (a, b) = if back { (shift(now, length)?, now) } else { (now, shift(now, length)?) };
        if b < a {
            return None;
        }
        let timex = Timex::Range(datetime_timex(a), datetime_timex(b), forward);
        let span = Span::Range(Some(Instant::DateTime(a)), Some(Instant::DateTime(b)));
        return Some((TimeKind::DateTimeRange, grain, one(timex, span)));
    }
    let today = now.date();
    let (s, e) = match (align, forward.parts()) {
        (TimeGrain::Quarter | TimeGrain::Decade, _) => {
            let this = relative(align, 0, today)?.span?.0;
            if back {
                (plus(this, length)?, this)
            } else {
                let s = relative(align, 1, today)?.span?.0;
                (s, plus(s, &forward)?)
            }
        }
        (_, &[(Unit::BusinessDay, n)]) => business_days(today, if back { -n / MILLI } else { n / MILLI })?,
        _ if back => (plus(today, length)?, today),
        _ => {
            let s = today.plus_days(1)?;
            (s, plus(s, &forward)?)
        }
    };
    if e < s {
        return None;
    }
    let timex = Timex::Range(day_timex(s, false), day_timex(e, false), forward);
    let span = Span::Range(Some(Instant::Date(s)), Some(Instant::Date(e)));
    Some((TimeKind::DateRange, grain, one(timex, span)))
}

fn within(length: &Duration, now: DateTime) -> Option<(TimeKind, TimeGrain, Vec<Moment>)> {
    if length.millis() < 0 {
        return None;
    }
    let grain = super::unit_grain(length.finest()?);
    let moment = |timex, span| vec![Moment { timex, span: Some(span), future: true, first: true }];
    if length.parts().iter().any(|(u, _)| u.is_clock()) {
        let b = shift(now, length)?;
        if b < now {
            return None;
        }
        let timex = Timex::Range(datetime_timex(now), datetime_timex(b), *length);
        let span = Span::Range(Some(Instant::DateTime(now)), Some(Instant::DateTime(b)));
        return Some((TimeKind::DateTimeRange, grain, moment(timex, span)));
    }
    let s = now.date();
    let e = plus(s, length)?;
    if e < s {
        return None;
    }
    let timex = Timex::Range(day_timex(s, false), day_timex(e, false), *length);
    let span = Span::Range(Some(Instant::Date(s)), Some(Instant::Date(e)));
    Some((TimeKind::DateRange, grain, moment(timex, span)))
}
