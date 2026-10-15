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

use super::date::{month_of, unit_grain};
use super::duration::{duration_of, length};
use super::lexicon::rel;
use super::{
    AMOUNT, BARE, CLOCK, DATE, DPOINT, DT, DTP, DUR, HOL, LRANGE, MDAY, MODED, MONTH, MONTHD, PARTD, PERIOD, RANGE,
    RELW, TO, TPOINT, TRANGE, UNIT, YEAR, any_point, clock_of, emit, r, term, tval, year_ref,
};

// Connectors.

fn b_empty(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Empty)
}

/// A child that is a point, carried up as it is.
fn b_point(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    any_point(c, &v[0]).map(|_| v[0])
}

/// A clock time or an hour, as a bound of a clock range.
fn b_tpoint(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_of(c, &v[0]).map(|_| v[0])
}

/// A point modified only by a portion, which a range bound reads as its
/// edge: `the end of 2008`.
fn b_edge_point(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (_, m) = any_point(c, &v[0])?;
    (m.relation == Relation::None && matches!(m.portion, Portion::Start | Portion::Mid | Portion::End)).then_some(v[0])
}

// Date ranges.

/// A day of an open month, `the 21st`, which is a time only after `on`.
fn open_day(d: DateSpec) -> bool {
    matches!(d, DateSpec::Cal { month: 0, day: 1.., weekday: 0, .. })
}

/// A range bound that names a date or a period: the point and the edge
/// it takes.
fn date_bound(c: &Ctx<'_>, v: &Value) -> Option<(Point, Edge)> {
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v)? else { return None };
    if !point.shift.is_zero() || modifier.relation != Relation::None {
        return None;
    }
    match (point.date, point.time) {
        (DateSpec::None, TimeOfDay::Now) | (_, TimeOfDay::None) => {}
        _ => return None,
    }
    if matches!(point.date, DateSpec::Undefined { .. }) || open_day(point.date) {
        return None;
    }
    let edge = match modifier.portion {
        Portion::Whole | Portion::Start => Edge::Start,
        Portion::Mid => Edge::Mid,
        Portion::End => Edge::End,
        _ => return None,
    };
    Some((point, edge))
}

/// The start's year, when the text gives it only with the end: `May to
/// Oct 2020`, `Nov-Feb 2017` (the November before).
fn with_end_year(start: DateSpec, end: DateSpec) -> DateSpec {
    let (DateSpec::Cal { year: YearRef::Any, month, day, weekday }, DateSpec::Cal { year, month: em, day: ed, .. }) =
        (start, end)
    else {
        return start;
    };
    if month == 0 || em == 0 {
        return start;
    }
    let before = (month, day) > (em, ed);
    let year = match year {
        YearRef::Abs(y) => YearRef::Abs(if before { y - 1 } else { y }),
        YearRef::Rel(n) => YearRef::Rel(if before { n - 1 } else { n }),
        _ => return start,
    };
    DateSpec::Cal { year, month, day, weekday }
}

/// A weekday that ends a range falls in the start's week: `from next
/// Monday to Friday`.
fn in_start_week(start: DateSpec, end: DateSpec) -> DateSpec {
    match (start, end) {
        (DateSpec::WeekdayRel { weeks, .. }, DateSpec::Cal { year: YearRef::Any, month: 0, day: 0, weekday })
            if weekday > 0 =>
        {
            DateSpec::WeekdayRel { weekday, weeks }
        }
        _ => end,
    }
}

/// A clock time or the present, as a bound of a range with the other:
/// `between now and eight o'clock`.
fn time_bound(c: &Ctx<'_>, v: &Value) -> Option<Point> {
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v)? else { return None };
    let fits = modifier.is_none() && point.shift.is_zero() && point.date == DateSpec::None;
    (fits && matches!(point.time, TimeOfDay::Clock(_) | TimeOfDay::Now)).then_some(point)
}

fn between(c: &Ctx<'_>, a: &Value, b: &Value) -> Option<Value> {
    if let (Some(s), Some(e)) = (time_bound(c, a), time_bound(c, b))
        && (s.time == TimeOfDay::Now) != (e.time == TimeOfDay::Now)
    {
        let range = Range::Between { start: s, start_edge: Edge::Start, end: e, end_edge: Edge::Start };
        return emit(c, TimeValue::range(range));
    }
    let (mut sp, se) = date_bound(c, a)?;
    let (mut ep, ee) = date_bound(c, b)?;
    sp.date = with_end_year(sp.date, ep.date);
    ep.date = in_start_week(sp.date, ep.date);
    let range = Range::Between { start: sp, start_edge: se, end: ep, end_edge: ee };
    emit(c, TimeValue::range(range))
}

fn b_range(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    between(c, &v[0], v.last()?)
}

/// `from 4-23 in next month`, `between 3 and 12 of Sept`: two days of
/// the month that follows.
fn days_of(c: &Ctx<'_>, d1: &Value, d2: &Value, month: &Value) -> Option<Value> {
    let (year, m) = month_of(c, month)?;
    let day = |d: &Value| -> Option<DateSpec> {
        let d = u8::try_from(term(d).0).ok()?;
        Some(match m {
            MonthRef::Abs(month) => DateSpec::Cal { year, month, day: d, weekday: 0 },
            MonthRef::Rel(months) => DateSpec::RelMonthDay { months, day: d },
        })
    };
    let (s, e) = (Point::date(day(d1)?), Point::date(day(d2)?));
    emit(c, TimeValue::range(Range::Between { start: s, start_edge: Edge::Start, end: e, end_edge: Edge::Start }))
}

fn b_days_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    // The connector is a child unless it is `and`.
    days_of(c, &v[0], &v[v.len() - 2], v.last()?)
}

/// `November 18-19`, `april 9th through 17th`, with a year after.
fn b_month_days(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let month = u8::try_from(term(&v[0]).0).ok()?;
    let year = match v.get(4) {
        Some(y) => year_ref(y),
        None => YearRef::Any,
    };
    let day =
        |d: &Value| Some(Point::date(DateSpec::Cal { year, month, day: u8::try_from(term(d).0).ok()?, weekday: 0 }));
    let range = Range::Between { start: day(&v[1])?, start_edge: Edge::Start, end: day(&v[3])?, end_edge: Edge::Start };
    emit(c, TimeValue::range(range))
}

// Clock ranges.

fn known(k: &Clock) -> bool {
    k.meridiem != Meridiem::Unknown
}

fn clock_range(c: &Ctx<'_>, date: DateSpec, s: Clock, e: Clock) -> Option<Value> {
    let p = |k| Point { date, time: TimeOfDay::Clock(k), ..Point::default() };
    emit(c, TimeValue::range(Range::Between { start: p(s), start_edge: Edge::Start, end: p(e), end_edge: Edge::Start }))
}

/// Two clock times: a range by itself when one says its half of the day
/// (`5 to 6pm`); otherwise one that needs a date or a part of the day
/// (`3-8`).
/// Whether the match holds a hyphen inside a spelled number: `seven
/// twenty-seven` is a time, not a range.
fn spelled_hyphen(c: &Ctx<'_>) -> bool {
    (c.start + 1..c.end.saturating_sub(1)).any(|i| {
        let (a, h, b) = (c.lexeme(i - 1), c.lexeme(i), c.lexeme(i + 1));
        c.word(i) == "-" && h.glued && b.glued && a.kind == Kind::Word && b.kind == Kind::Word
    })
}

fn b_trange(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if spelled_hyphen(c) {
        return None;
    }
    let (s, e) = (clock_of(c, &v[0])?, clock_of(c, v.last()?)?);
    stands(&s, &e).then(|| clock_range(c, DateSpec::None, s, e))?
}

fn b_lrange(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if spelled_hyphen(c) {
        return None;
    }
    let (s, e) = (clock_of(c, &v[0])?, clock_of(c, v.last()?)?);
    (!stands(&s, &e)).then(|| clock_range(c, DateSpec::None, s, e))?
}

/// Whether two clock times make a range by themselves: one says its
/// half of the day or has minutes (`9 to 10:30`).
fn stands(s: &Clock, e: &Clock) -> bool {
    known(s) || known(e) || s.shown != Shown::Hour || e.shown != Shown::Hour
}

/// `from 9 for 2.5 hrs`, `for 2.5 hours from 9`.
fn clock_for(c: &Ctx<'_>, clock: &Value, dur: &Value) -> Option<Value> {
    let k = clock_of(c, clock)?;
    let length = duration_of(c, dur)?;
    emit(c, TimeValue::range(Range::For { start: Point::time(TimeOfDay::Clock(k)), length }))
}

fn b_from_clock_for(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_for(c, &v[0], &v[1])
}

fn b_for_from_clock(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_for(c, &v[1], &v[0])
}

/// A clock range from a clock time for a duration.
fn clock_length(c: &Ctx<'_>, v: &Value) -> Option<(Clock, Duration)> {
    match tval(c, v)? {
        TimeValue::Range {
            range: Range::For { start: Point { date: DateSpec::None, time: TimeOfDay::Clock(k), .. }, length },
            ..
        } => Some((k, length)),
        _ => None,
    }
}

/// The clocks of a clock range.
fn clocks(c: &Ctx<'_>, v: &Value) -> Option<(Clock, Clock)> {
    match tval(c, v)? {
        TimeValue::Range {
            range:
                Range::Between {
                    start: Point { date: DateSpec::None, time: TimeOfDay::Clock(s), .. },
                    end: Point { date: DateSpec::None, time: TimeOfDay::Clock(e), .. },
                    ..
                },
            modifier,
        } if modifier.is_none() => Some((s, e)),
        _ => None,
    }
}

/// A day, or a day and a part of it, as a clock range's date.
fn day_and_part(c: &Ctx<'_>, v: &Value) -> Option<(DateSpec, Option<crate::time::spec::Part>)> {
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v)? else { return None };
    if !modifier.is_none() || !point.shift.is_zero() {
        return None;
    }
    match point.time {
        TimeOfDay::None if point.date.is_day() => Some((point.date, None)),
        TimeOfDay::Part(p) => Some((point.date, Some(p))),
        _ => None,
    }
}

fn dated_clocks(c: &Ctx<'_>, day: &Value, range: &Value) -> Option<Value> {
    let (date, part) = day_and_part(c, day)?;
    if let Some((k, length)) = clock_length(c, range) {
        let start = Point { date, time: TimeOfDay::Clock(k), ..Point::default() };
        return emit(c, TimeValue::range(Range::For { start, length }));
    }
    let (s, e) = clocks(c, range)?;
    let (s, e) = match part {
        Some(p) => (s.with(p.meridiem(s.hour)), e.with(p.meridiem(e.hour))),
        None => (s, e),
    };
    clock_range(c, date, s, e)
}

fn b_date_trange(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    dated_clocks(c, &v[0], v.last()?)
}

fn b_trange_date(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    dated_clocks(c, v.last()?, &v[0])
}

// Stretches from the reference.

fn direction(rel_v: &Value) -> Option<bool> {
    match term(rel_v).1 {
        rel::NEXT | rel::COMING | rel::UPCOMING => Some(true),
        rel::LAST | rel::PAST => Some(false),
        _ => None,
    }
}

fn next(c: &Ctx<'_>, forward: bool, d: Duration, align: TimeGrain) -> Option<Value> {
    let length = if forward { d } else { d.negated() };
    emit(c, TimeValue::range(Range::Next { length, align }))
}

fn align_of(unit: &Value, d: &Duration) -> Option<TimeGrain> {
    match unit_grain(unit) {
        Some(g @ (TimeGrain::Quarter | TimeGrain::Decade)) => Some(g),
        _ => d.finest().map(crate::time::resolve::unit_grain),
    }
}

/// `next 3 days`, `last 2 quarters`, `past two quarters`.
fn b_next_n(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let d = length(term(&v[1]).0, &v[2])?;
    next(c, direction(&v[0])?, d, align_of(&v[2], &d)?)
}

/// `5 coming minutes`, `10 previous weeks`.
fn b_n_next(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[2]).1 & super::PLURAL == 0 {
        return None;
    }
    let d = length(term(&v[0]).0, &v[2])?;
    next(c, direction(&v[1])?, d, align_of(&v[2], &d)?)
}

fn b_next_dur(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let d = duration_of(c, &v[1])?;
    next(c, direction(&v[0])?, d, crate::time::resolve::unit_grain(d.finest()?))
}

/// `next hour`, `last minute`.
fn b_next_unit(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let d = length(i64::from(MILLI), &v[1])?;
    if term(&v[1]).1 & super::PLURAL != 0 || !d.parts().iter().all(|(u, _)| u.is_clock()) {
        return None;
    }
    next(c, direction(&v[0])?, d, crate::time::resolve::unit_grain(d.finest()?))
}

fn b_within(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit(c, TimeValue::range(Range::Within { length: duration_of(c, v.last()?)? }))
}

/// `2 weeks starting May 20th`, `for 3 days from today`.
fn b_for(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let length = duration_of(c, &v[0])?;
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v.last()?)? else { return None };
    if !modifier.is_none() || !point.shift.is_zero() {
        return None;
    }
    emit(c, TimeValue::range(Range::For { start: point, length }))
}

// Modifiers.

fn relate(c: &Ctx<'_>, v: &Value, relation: Relation) -> Option<Value> {
    let TimeValue::Point { point, modifier, clip } = tval(c, v)? else { return None };
    if modifier.relation != Relation::None || clip || modifier.portion == Portion::Undefined || open_day(point.date) {
        return None;
    }
    let portion = if relation == Relation::Approx { Portion::Whole } else { modifier.portion };
    if relation == Relation::Approx && modifier.portion != Portion::Whole {
        return None;
    }
    emit(c, TimeValue::Point { point, modifier: Modifier { relation, portion }, clip: false })
}

fn b_before(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, v.last()?, Relation::Before)
}

fn b_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, v.last()?, Relation::After)
}

fn b_since(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, v.last()?, Relation::Since)
}

fn b_until(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, v.last()?, Relation::Until)
}

fn b_since_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, &v[0], Relation::Since)
}

fn b_approx(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, v.last()?, Relation::Approx)
}

/// `before around 2pm`: a bound said loosely.
fn approx_bound(c: &Ctx<'_>, v: &Value, relation: Relation) -> Option<Value> {
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v)? else { return None };
    if !modifier.is_none() {
        return None;
    }
    emit(c, TimeValue::Point { point, modifier: Modifier { relation, portion: Portion::Approx }, clip: false })
}

fn b_before_around(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    approx_bound(c, v.last()?, Relation::Before)
}

fn b_after_around(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    approx_bound(c, v.last()?, Relation::After)
}

/// A date with a bounded clock time: `today before 4pm`, `1/1/2015
/// after 2:00`.
fn b_date_moded(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (date, part) = day_and_part(c, &v[0])?;
    if part.is_some() {
        return None;
    }
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v.last()?)? else { return None };
    if point.date != DateSpec::None || !matches!(point.time, TimeOfDay::Clock(_)) || modifier.relation == Relation::None
    {
        return None;
    }
    emit(c, TimeValue::Point { point: Point { date, ..point }, modifier, clip: false })
}

/// A part of a period: `the end of 2008`, `mid-November`; `later this
/// week`, said of the reference's own period, stops at the reference.
fn portion(c: &Ctx<'_>, v: &Value, portion: Portion, clip: bool) -> Option<Value> {
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v)? else { return None };
    if !modifier.is_none() || !point.shift.is_zero() || point.date == DateSpec::None || point.time != TimeOfDay::None {
        return None;
    }
    // `later in today` is the day's second half.
    let clip = clip
        && (point.date.is_day()
            || matches!(
                point.date,
                DateSpec::Rel { n: 0, grain: TimeGrain::Week | TimeGrain::Month | TimeGrain::Year }
            ));
    emit(c, TimeValue::Point { point, modifier: Modifier::portion(portion), clip })
}

fn b_start(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v.last()?, Portion::Start, false)
}

fn b_mid(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v.last()?, Portion::Mid, false)
}

fn b_end(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v.last()?, Portion::End, false)
}

fn b_earlier(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v.last()?, Portion::Start, true)
}

fn b_later(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v.last()?, Portion::End, true)
}

/// `later in the week`, `earlier in the year`, `end of month`: the
/// reference's period.
fn this_period(c: &Ctx<'_>, unit: &Value, p: Portion, clip: bool) -> Option<Value> {
    if term(unit).1 & super::PLURAL != 0 {
        return None;
    }
    let grain = unit_grain(unit)?;
    let point = Point::date(DateSpec::Rel { grain, n: 0 });
    let clip = clip && matches!(grain, TimeGrain::Week | TimeGrain::Month | TimeGrain::Year);
    emit(c, TimeValue::Point { point, modifier: Modifier::portion(p), clip })
}

fn b_earlier_in_the(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    this_period(c, &v[0], Portion::Start, true)
}

fn b_later_in_the(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    this_period(c, &v[0], Portion::End, true)
}

fn b_end_of_unit(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if unit_grain(&v[0]) == Some(TimeGrain::Day) {
        return None;
    }
    this_period(c, &v[0], Portion::End, false)
}

fn today_portion(c: &Ctx<'_>, p: Portion) -> Option<Value> {
    let point = Point::date(DateSpec::Rel { grain: TimeGrain::Day, n: 0 });
    emit(c, TimeValue::Point { point, modifier: Modifier::portion(p), clip: false })
}

fn b_end_of_day(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    today_portion(c, Portion::End)
}

fn b_eoy(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let point = Point::date(DateSpec::Rel { grain: TimeGrain::Year, n: 0 });
    emit(c, TimeValue::Point { point, modifier: Modifier::portion(Portion::End), clip: false })
}

/// `more than 2 weeks before today`: before the shifted day; `less than
/// 3 days after tomorrow`: the days between.
fn shifted(c: &Ctx<'_>, v: &[Value], more: bool, after: bool) -> Option<Value> {
    let d = duration_of(c, &v[0])?;
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v.last()?)? else { return None };
    if !modifier.is_none() || !point.shift.is_zero() || !point.date.is_day() || point.time != TimeOfDay::None {
        return None;
    }
    let shift = if after { d } else { d.negated() };
    let moved = Point { shift, ..point };
    if more {
        let relation = if after { Relation::After } else { Relation::Before };
        return emit(c, TimeValue::Point { point: moved, modifier: Modifier::relation(relation), clip: false });
    }
    if !after {
        return None;
    }
    emit(c, TimeValue::range(Range::For { start: point, length: d }))
}

fn b_more_before(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    shifted(c, v, true, false)
}

fn b_more_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    shifted(c, v, true, true)
}

fn b_less_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    shifted(c, v, false, true)
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    rr("to", TO, &[word("to")], b_empty),
    rr("to_dash", TO, &[word("-")], b_empty),
    rr("to_en_dash", TO, &[word("\u{2013}")], b_empty),
    rr("to_tilde", TO, &[word("~")], b_empty),
    rr("to_till", TO, &[word("till")], b_empty),
    rr("to_til", TO, &[word("til")], b_empty),
    rr("to_until", TO, &[word("until")], b_empty),
    rr("to_through", TO, &[word("through")], b_empty),
    rr("to_thru", TO, &[word("thru")], b_empty),
    // Bounds.
    rr("dpoint_date", DPOINT, &[node(DATE)], b_point),
    rr("dpoint_month", DPOINT, &[node(MONTHD)], b_point),
    rr("dpoint_period", DPOINT, &[node(PERIOD)], b_point),
    rr("dpoint_hol", DPOINT, &[node(HOL)], b_point),
    rr("dpoint_dt", DPOINT, &[node(DT)], b_point),
    rr("dpoint_dtp", DPOINT, &[node(DTP)], b_point),
    rr("dpoint_clock", DPOINT, &[node(CLOCK)], b_point),
    rr("dpoint_partd", DPOINT, &[node(PARTD)], b_point),
    rr("dpoint_edge", DPOINT, &[node(MODED)], b_edge_point),
    rr("tpoint_clock", TPOINT, &[node(CLOCK)], b_tpoint),
    rr("tpoint_bare", TPOINT, &[node(BARE)], b_tpoint),
    // Date ranges.
    rr("range", RANGE, &[node(DPOINT), node(TO), node(DPOINT)], b_range),
    rr("range_from", RANGE, &[word("from"), node(DPOINT), node(TO), node(DPOINT)], b_range),
    rr("range_between", RANGE, &[word("between"), node(DPOINT), word("and"), node(DPOINT)], b_range),
    rr("range_between_to", RANGE, &[word("between"), node(DPOINT), node(TO), node(DPOINT)], b_range),
    rr("range_days_of", RANGE, &[node(MDAY), node(TO), node(MDAY), word("of"), node(DPOINT)], b_days_of),
    rr("range_days_in", RANGE, &[node(MDAY), node(TO), node(MDAY), word("in"), node(DPOINT)], b_days_of),
    rr(
        "range_from_days_of",
        RANGE,
        &[word("from"), node(MDAY), node(TO), node(MDAY), word("of"), node(DPOINT)],
        b_days_of,
    ),
    rr(
        "range_from_days_in",
        RANGE,
        &[word("from"), node(MDAY), node(TO), node(MDAY), word("in"), node(DPOINT)],
        b_days_of,
    ),
    rr(
        "range_between_days_of",
        RANGE,
        &[
            word("between"),
            opt_word("the"),
            node(MDAY),
            word("and"),
            opt_word("the"),
            node(MDAY),
            word("of"),
            node(DPOINT),
        ],
        b_days_of,
    ),
    rr("range_month_days", RANGE, &[node(MONTH), node(MDAY), node(TO), opt_word("the"), node(MDAY)], b_month_days),
    rr(
        "range_month_days_year",
        RANGE,
        &[node(MONTH), node(MDAY), node(TO), node(MDAY), opt_word(","), node(YEAR)],
        b_month_days,
    ),
    rr(
        "range_from_month_days",
        RANGE,
        &[word("from"), node(MONTH), opt_word("the"), node(MDAY), node(TO), opt_word("the"), node(MDAY)],
        b_month_days,
    ),
    // Clock ranges.
    rr("trange", TRANGE, &[node(TPOINT), node(TO), node(TPOINT)], b_trange),
    rr("trange_from", TRANGE, &[word("from"), node(TPOINT), node(TO), node(TPOINT)], b_trange),
    rr("trange_between", TRANGE, &[word("between"), node(TPOINT), word("and"), node(TPOINT)], b_trange),
    rr("trange_between_to", TRANGE, &[word("between"), node(TPOINT), node(TO), node(TPOINT)], b_trange),
    rr("lrange", LRANGE, &[node(TPOINT), node(TO), node(TPOINT)], b_lrange),
    rr("lrange_from", LRANGE, &[word("from"), node(TPOINT), node(TO), node(TPOINT)], b_lrange),
    rr("lrange_between", LRANGE, &[word("between"), node(TPOINT), word("and"), node(TPOINT)], b_lrange),
    rr("lrange_between_to", LRANGE, &[word("between"), node(TPOINT), node(TO), node(TPOINT)], b_lrange),
    r("range_date_trange", RANGE, 2, &[node(DATE), opt_word(","), opt_word("at"), node(TRANGE)], b_date_trange),
    r("range_date_lrange", RANGE, 2, &[node(DATE), opt_word("at"), node(LRANGE)], b_date_trange),
    rr("range_dtp_trange", RANGE, &[node(DTP), node(TRANGE)], b_date_trange),
    rr("range_dtp_lrange", RANGE, &[node(DTP), node(LRANGE)], b_date_trange),
    rr("range_trange_date", RANGE, &[node(TRANGE), opt_word("on"), opt_word("of"), node(DATE)], b_trange_date),
    rr("range_lrange_date", RANGE, &[node(LRANGE), opt_word("on"), opt_word("of"), node(DATE)], b_trange_date),
    rr("range_trange_dtp", RANGE, &[node(TRANGE), node(DTP)], b_trange_date),
    rr("range_lrange_dtp", RANGE, &[node(LRANGE), node(DTP)], b_trange_date),
    // Stretches from the reference.
    rr("range_next_n", RANGE, &[node(RELW), node(AMOUNT), node(UNIT)], b_next_n),
    rr("range_n_next", RANGE, &[node(AMOUNT), node(RELW), node(UNIT)], b_n_next),
    rr("range_next_dur", RANGE, &[node(RELW), node(DUR)], b_next_dur),
    rr("range_next_unit", RANGE, &[node(RELW), node(UNIT)], b_next_unit),
    rr("range_within", RANGE, &[word("within"), node(DUR)], b_within),
    rr("range_within_next", RANGE, &[word("within"), word("next"), node(DUR)], b_within),
    rr(
        "range_for_starting",
        RANGE,
        &[node(DUR), word("starting"), opt_word("from"), opt_word("on"), node(DPOINT)],
        b_for,
    ),
    rr("range_for_beginning", RANGE, &[node(DUR), word("beginning"), opt_word("on"), node(DPOINT)], b_for),
    rr("range_for_from", RANGE, &[word("for"), node(DUR), opt_word(","), word("from"), node(DPOINT)], b_for),
    rr("trange_from_for", RANGE, &[word("from"), node(TPOINT), word("for"), node(DUR)], b_from_clock_for),
    rr("trange_for_from", RANGE, &[word("for"), node(DUR), word("from"), node(TPOINT)], b_for_from_clock),
    rr("range_date_for", RANGE, &[node(DATE), node(RANGE)], b_date_trange),
    rr("range_less_after", RANGE, &[word("less"), word("than"), node(DUR), word("after"), node(DPOINT)], b_less_after),
    // Bounded and modified points.
    rr("moded_before", MODED, &[word("before"), node(DPOINT)], b_before),
    rr("moded_prior_to", MODED, &[word("prior"), word("to"), node(DPOINT)], b_before),
    rr("moded_earlier_than", MODED, &[word("earlier"), word("than"), node(DPOINT)], b_before),
    rr("moded_by", MODED, &[word("by"), node(DPOINT)], b_before),
    rr("moded_no_later_than", MODED, &[word("no"), word("later"), word("than"), node(DPOINT)], b_before),
    rr("moded_until", MODED, &[word("until"), node(DPOINT)], b_before),
    rr("moded_till", MODED, &[word("till"), node(DPOINT)], b_before),
    rr("moded_after", MODED, &[word("after"), node(DPOINT)], b_after),
    rr("moded_later_than", MODED, &[word("later"), word("than"), node(DPOINT)], b_after),
    rr("moded_greater_than", MODED, &[word("greater"), word("than"), node(DPOINT)], b_after),
    rr("moded_since", MODED, &[word("since"), node(DPOINT)], b_since),
    rr("moded_starting", MODED, &[word("starting"), opt_word("from"), opt_word("on"), node(DPOINT)], b_since),
    rr("moded_beginning", MODED, &[word("beginning"), opt_word("on"), node(DPOINT)], b_since),
    rr("moded_as_early_as", MODED, &[word("as"), word("early"), word("as"), node(DPOINT)], b_since),
    rr("moded_any_time_from", MODED, &[word("any"), word("time"), word("from"), node(DPOINT)], b_since),
    rr("moded_as_late_as", MODED, &[word("as"), word("late"), word("as"), node(DPOINT)], b_until),
    rr("moded_ge", MODED, &[word(">"), opt_word("="), node(DPOINT)], b_since_ge),
    rr("moded_le", MODED, &[word("<"), opt_word("="), node(DPOINT)], b_until_le),
    rr("moded_or_later", MODED, &[node(DPOINT), word("or"), word("later")], b_since_after),
    rr("moded_and_later", MODED, &[node(DPOINT), word("and"), word("later")], b_since_after),
    rr("moded_and_after", MODED, &[node(DPOINT), word("and"), word("after")], b_since_after),
    rr("moded_or_after", MODED, &[node(DPOINT), word("or"), word("after")], b_since_after),
    rr("moded_and_greater", MODED, &[node(DPOINT), word("and"), word("greater")], b_since_after),
    rr("moded_around", MODED, &[word("around"), node(DPOINT)], b_approx),
    rr("moded_circa", MODED, &[word("circa"), node(DPOINT)], b_approx),
    rr("moded_before_around", MODED, &[word("before"), word("around"), node(DPOINT)], b_before_around),
    rr("moded_after_around", MODED, &[word("after"), word("around"), node(DPOINT)], b_after_around),
    rr("moded_date", MODED, &[node(DATE), node(MODED)], b_date_moded),
    rr(
        "moded_more_before",
        MODED,
        &[word("more"), word("than"), node(DUR), word("before"), node(DPOINT)],
        b_more_before,
    ),
    rr("moded_more_after", MODED, &[word("more"), word("than"), node(DUR), word("after"), node(DPOINT)], b_more_after),
    // Portions.
    rr("moded_early", MODED, &[word("early"), opt_word("-"), node(DPOINT)], b_start),
    rr("moded_beginning_of_the", MODED, &[word("the"), word("beginning"), word("of"), node(DPOINT)], b_start),
    rr("moded_beginning_of", MODED, &[word("beginning"), word("of"), node(DPOINT)], b_start),
    rr("moded_start_of_the", MODED, &[word("the"), word("start"), word("of"), node(DPOINT)], b_start),
    rr("moded_start_of", MODED, &[word("start"), word("of"), node(DPOINT)], b_start),
    rr("moded_early_in_the_day", MODED, &[word("early"), word("in"), word("the"), word("day"), node(DPOINT)], b_start),
    rr("moded_mid", MODED, &[word("mid"), opt_word("-"), node(DPOINT)], b_mid),
    rr("moded_middle_of_the", MODED, &[word("the"), word("middle"), word("of"), node(DPOINT)], b_mid),
    rr("moded_middle_of", MODED, &[word("middle"), word("of"), node(DPOINT)], b_mid),
    rr("moded_late", MODED, &[word("late"), opt_word("-"), node(DPOINT)], b_end),
    rr("moded_end_of_the", MODED, &[word("the"), word("end"), word("of"), node(DPOINT)], b_end),
    rr("moded_end_of", MODED, &[word("end"), word("of"), node(DPOINT)], b_end),
    rr("moded_later_in", MODED, &[word("later"), word("in"), node(DPOINT)], b_end),
    rr("moded_earlier", MODED, &[word("earlier"), node(DPOINT)], b_earlier),
    rr("moded_later", MODED, &[word("later"), node(DPOINT)], b_later),
    rr("moded_earlier_in_the", MODED, &[word("earlier"), word("in"), word("the"), node(UNIT)], b_earlier_in_the),
    rr("moded_later_in_the", MODED, &[word("later"), word("in"), word("the"), node(UNIT)], b_later_in_the),
    rr(
        "moded_end_of_day_the",
        MODED,
        &[word("the"), word("end"), word("of"), opt_word("the"), word("day")],
        b_end_of_day,
    ),
    rr("moded_end_of_day", MODED, &[word("end"), word("of"), opt_word("the"), word("day")], b_end_of_day),
    rr("moded_eod", MODED, &[word("eod")], b_end_of_day),
    rr("moded_end_of_unit", MODED, &[word("end"), word("of"), node(UNIT)], b_end_of_unit),
    rr("moded_year_greater_than", MODED, &[word("year"), word("greater"), word("than"), node(DPOINT)], b_after),
    rr("moded_eoy", MODED, &[word("eoy")], b_eoy),
];

fn b_since_ge(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    b_since(v, c)
}

fn b_until_le(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    b_until(v, c)
}
