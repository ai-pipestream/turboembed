use super::*;

// Connectors.

pub(crate) fn b_empty(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Empty)
}

/// A child that is a point, carried up as it is.
pub(crate) fn b_point(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    any_point(c, &v[0]).map(|_| v[0])
}

/// A clock time or an hour, as a bound of a clock range.
pub(crate) fn b_tpoint(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_of(c, &v[0]).map(|_| v[0])
}

/// A point modified only by a portion, which a range bound reads as its
/// edge: `the end of 2008`.
pub(crate) fn b_edge_point(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (_, m) = any_point(c, &v[0])?;
    (m.relation == Relation::None && matches!(m.portion, Portion::Start | Portion::Mid | Portion::End)).then_some(v[0])
}

// Date ranges.

/// A day of an open month, `the 21st`, which is a time only after `on`.
pub(crate) fn open_day(d: DateSpec) -> bool {
    matches!(d, DateSpec::Cal { month: 0, day: 1.., weekday: 0, .. })
}

/// A range bound that names a date or a period: the point and the edge
/// it takes.
pub(crate) fn date_bound(c: &Ctx<'_>, v: &Value) -> Option<(Point, Edge)> {
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
pub(crate) fn with_end_year(start: DateSpec, end: DateSpec) -> DateSpec {
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
pub(crate) fn in_start_week(start: DateSpec, end: DateSpec) -> DateSpec {
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
pub(crate) fn time_bound(c: &Ctx<'_>, v: &Value) -> Option<Point> {
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v)? else { return None };
    let fits = modifier.is_none() && point.shift.is_zero() && point.date == DateSpec::None;
    (fits && matches!(point.time, TimeOfDay::Clock(_) | TimeOfDay::Now)).then_some(point)
}

pub(crate) fn between(c: &Ctx<'_>, a: &Value, b: &Value) -> Option<Value> {
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

pub(crate) fn b_range(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    between(c, &v[0], v.last()?)
}

/// `from 4-23 in next month`, `between 3 and 12 of Sept`: two days of
/// the month that follows.
pub(crate) fn days_of(c: &Ctx<'_>, d1: &Value, d2: &Value, month: &Value) -> Option<Value> {
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

pub(crate) fn b_days_of(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    // The connector is a child unless it is `and`.
    days_of(c, &v[0], &v[v.len() - 2], v.last()?)
}

/// `November 18-19`, `april 9th through 17th`, with a year after.
pub(crate) fn b_month_days(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
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
