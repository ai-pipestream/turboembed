use super::*;

// Clock ranges.

pub(super) fn known(k: &Clock) -> bool {
    k.meridiem != Meridiem::Unknown
}

pub(in crate::lang) fn clock_range(c: &Ctx<'_>, date: DateSpec, s: Clock, e: Clock) -> Option<Value> {
    let p = |k| Point { date, time: TimeOfDay::Clock(k), ..Point::default() };
    emit(c, TimeValue::range(Range::Between { start: p(s), start_edge: Edge::Start, end: p(e), end_edge: Edge::Start }))
}

/// Two clock times: a range by itself when one says its half of the day
/// (`5 to 6pm`); otherwise one that needs a date or a part of the day
/// (`3-8`).
/// Whether the match holds a hyphen inside a spelled number: `seven
/// twenty-seven` is a time, not a range.
pub(super) fn spelled_hyphen(c: &Ctx<'_>) -> bool {
    (c.start + 1..c.end.saturating_sub(1)).any(|i| {
        let (a, h, b) = (c.lexeme(i - 1), c.lexeme(i), c.lexeme(i + 1));
        c.word(i) == "-" && h.glued && b.glued && a.kind == Kind::Word && b.kind == Kind::Word
    })
}

pub(in crate::lang) fn b_trange(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if spelled_hyphen(c) {
        return None;
    }
    let (s, e) = (clock_of(c, &v[0])?, clock_of(c, v.last()?)?);
    stands(&s, &e).then(|| clock_range(c, DateSpec::None, s, e))?
}

pub(in crate::lang) fn b_lrange(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if spelled_hyphen(c) {
        return None;
    }
    let (s, e) = (clock_of(c, &v[0])?, clock_of(c, v.last()?)?);
    (!stands(&s, &e)).then(|| clock_range(c, DateSpec::None, s, e))?
}

/// Whether two clock times make a range by themselves: one says its
/// half of the day or has minutes (`9 to 10:30`).
pub(in crate::lang) fn stands(s: &Clock, e: &Clock) -> bool {
    known(s) || known(e) || s.shown != Shown::Hour || e.shown != Shown::Hour
}

/// `from 9 for 2.5 hrs`, `for 2.5 hours from 9`.
pub(super) fn clock_for(c: &Ctx<'_>, clock: &Value, dur: &Value) -> Option<Value> {
    let k = clock_of(c, clock)?;
    let length = duration_of(c, dur)?;
    emit(c, TimeValue::range(Range::For { start: Point::time(TimeOfDay::Clock(k)), length }))
}

pub(in crate::lang) fn b_from_clock_for(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_for(c, &v[0], &v[1])
}

pub(in crate::lang) fn b_for_from_clock(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_for(c, &v[1], &v[0])
}

/// A clock range from a clock time for a duration.
pub(in crate::lang) fn clock_length(c: &Ctx<'_>, v: &Value) -> Option<(Clock, Duration)> {
    match tval(c, v)? {
        TimeValue::Range {
            range: Range::For { start: Point { date: DateSpec::None, time: TimeOfDay::Clock(k), .. }, length },
            ..
        } => Some((k, length)),
        _ => None,
    }
}

/// The clocks of a clock range.
pub(in crate::lang) fn clocks(c: &Ctx<'_>, v: &Value) -> Option<(Clock, Clock)> {
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
pub(in crate::lang) fn day_and_part(c: &Ctx<'_>, v: &Value) -> Option<(DateSpec, Option<crate::time::spec::Part>)> {
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

pub(super) fn dated_clocks(c: &Ctx<'_>, day: &Value, range: &Value) -> Option<Value> {
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

pub(in crate::lang) fn b_date_trange(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    dated_clocks(c, &v[0], v.last()?)
}

pub(in crate::lang) fn b_trange_date(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    dated_clocks(c, v.last()?, &v[0])
}
