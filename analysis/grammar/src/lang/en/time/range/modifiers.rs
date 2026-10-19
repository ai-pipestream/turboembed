use super::*;

// Stretches from the reference.

pub(crate) fn direction(rel_v: &Value) -> Option<bool> {
    match term(rel_v).1 {
        rel::NEXT | rel::COMING | rel::UPCOMING => Some(true),
        rel::LAST | rel::PAST => Some(false),
        _ => None,
    }
}

pub(crate) fn next(c: &Ctx<'_>, forward: bool, d: Duration, align: TimeGrain) -> Option<Value> {
    let length = if forward { d } else { d.negated() };
    emit(c, TimeValue::range(Range::Next { length, align }))
}

pub(crate) fn align_of(unit: &Value, d: &Duration) -> Option<TimeGrain> {
    match unit_grain(unit) {
        Some(g @ (TimeGrain::Quarter | TimeGrain::Decade)) => Some(g),
        _ => d.finest().map(crate::time::resolve::unit_grain),
    }
}

/// `next 3 days`, `last 2 quarters`, `past two quarters`.
pub(crate) fn b_next_n(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let d = length(term(&v[1]).0, &v[2])?;
    next(c, direction(&v[0])?, d, align_of(&v[2], &d)?)
}

/// `5 coming minutes`, `10 previous weeks`.
pub(crate) fn b_n_next(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[2]).1 & crate::lang::en::time::PLURAL == 0 {
        return None;
    }
    let d = length(term(&v[0]).0, &v[2])?;
    next(c, direction(&v[1])?, d, align_of(&v[2], &d)?)
}

pub(crate) fn b_next_dur(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let d = duration_of(c, &v[1])?;
    next(c, direction(&v[0])?, d, crate::time::resolve::unit_grain(d.finest()?))
}

/// `next hour`, `last minute`.
pub(crate) fn b_next_unit(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let d = length(i64::from(MILLI), &v[1])?;
    if term(&v[1]).1 & crate::lang::en::time::PLURAL != 0 || !d.parts().iter().all(|(u, _)| u.is_clock()) {
        return None;
    }
    next(c, direction(&v[0])?, d, crate::time::resolve::unit_grain(d.finest()?))
}

pub(crate) fn b_within(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit(c, TimeValue::range(Range::Within { length: duration_of(c, v.last()?)? }))
}

/// `2 weeks starting May 20th`, `for 3 days from today`.
pub(crate) fn b_for(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let length = duration_of(c, &v[0])?;
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v.last()?)? else { return None };
    if !modifier.is_none() || !point.shift.is_zero() {
        return None;
    }
    emit(c, TimeValue::range(Range::For { start: point, length }))
}

// Modifiers.

pub(crate) fn relate(c: &Ctx<'_>, v: &Value, relation: Relation) -> Option<Value> {
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

pub(crate) fn b_before(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, v.last()?, Relation::Before)
}

pub(crate) fn b_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, v.last()?, Relation::After)
}

pub(crate) fn b_since(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, v.last()?, Relation::Since)
}

pub(crate) fn b_until(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, v.last()?, Relation::Until)
}

pub(crate) fn b_since_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, &v[0], Relation::Since)
}

pub(crate) fn b_approx(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    relate(c, v.last()?, Relation::Approx)
}

/// `before around 2pm`: a bound said loosely.
pub(crate) fn approx_bound(c: &Ctx<'_>, v: &Value, relation: Relation) -> Option<Value> {
    let TimeValue::Point { point, modifier, clip: false } = tval(c, v)? else { return None };
    if !modifier.is_none() {
        return None;
    }
    emit(c, TimeValue::Point { point, modifier: Modifier { relation, portion: Portion::Approx }, clip: false })
}

pub(crate) fn b_before_around(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    approx_bound(c, v.last()?, Relation::Before)
}

pub(crate) fn b_after_around(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    approx_bound(c, v.last()?, Relation::After)
}

/// A date with a bounded clock time: `today before 4pm`, `1/1/2015
/// after 2:00`.
pub(crate) fn b_date_moded(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
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
pub(crate) fn portion(c: &Ctx<'_>, v: &Value, portion: Portion, clip: bool) -> Option<Value> {
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

pub(crate) fn b_start(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v.last()?, Portion::Start, false)
}

pub(crate) fn b_mid(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v.last()?, Portion::Mid, false)
}

pub(crate) fn b_end(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v.last()?, Portion::End, false)
}

pub(crate) fn b_earlier(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v.last()?, Portion::Start, true)
}

pub(crate) fn b_later(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion(c, v.last()?, Portion::End, true)
}

/// `later in the week`, `earlier in the year`, `end of month`: the
/// reference's period.
pub(crate) fn this_period(c: &Ctx<'_>, unit: &Value, p: Portion, clip: bool) -> Option<Value> {
    if term(unit).1 & crate::lang::en::time::PLURAL != 0 {
        return None;
    }
    let grain = unit_grain(unit)?;
    let point = Point::date(DateSpec::Rel { grain, n: 0 });
    let clip = clip && matches!(grain, TimeGrain::Week | TimeGrain::Month | TimeGrain::Year);
    emit(c, TimeValue::Point { point, modifier: Modifier::portion(p), clip })
}

pub(crate) fn b_earlier_in_the(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    this_period(c, &v[0], Portion::Start, true)
}

pub(crate) fn b_later_in_the(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    this_period(c, &v[0], Portion::End, true)
}

pub(crate) fn b_end_of_unit(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if unit_grain(&v[0]) == Some(TimeGrain::Day) {
        return None;
    }
    this_period(c, &v[0], Portion::End, false)
}

pub(crate) fn today_portion(c: &Ctx<'_>, p: Portion) -> Option<Value> {
    let point = Point::date(DateSpec::Rel { grain: TimeGrain::Day, n: 0 });
    emit(c, TimeValue::Point { point, modifier: Modifier::portion(p), clip: false })
}

pub(crate) fn b_end_of_day(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    today_portion(c, Portion::End)
}

pub(crate) fn b_eoy(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let point = Point::date(DateSpec::Rel { grain: TimeGrain::Year, n: 0 });
    emit(c, TimeValue::Point { point, modifier: Modifier::portion(Portion::End), clip: false })
}

/// `more than 2 weeks before today`: before the shifted day; `less than
/// 3 days after tomorrow`: the days between.
pub(crate) fn shifted(c: &Ctx<'_>, v: &[Value], more: bool, after: bool) -> Option<Value> {
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

pub(crate) fn b_more_before(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    shifted(c, v, true, false)
}

pub(crate) fn b_more_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    shifted(c, v, true, true)
}

pub(crate) fn b_less_after(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    shifted(c, v, false, true)
}
