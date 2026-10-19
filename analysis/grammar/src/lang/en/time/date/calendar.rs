use super::*;

// Years.

pub(crate) fn b_year_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (y, _) = term(&v[0]);
    (dig_len(&v[0]) == 4 && !dig_zero(&v[0]) && (1000..3000).contains(&y)).then(|| term_value(y, 0))?
}

pub(crate) fn b_year_short(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    (dig_len(&v[0]) == 2).then(|| term_value(term(&v[0]).0, SHORT))?
}

pub(crate) fn spelled(v: &Value) -> Option<i64> {
    let n = v.num()?;
    (n.has(crate::value::flag::SPELLED) && !n.has(crate::value::flag::DIGITS) && n.value.is_integer()).then_some(())?;
    n.value.to_i64()
}

pub(crate) fn b_year_spelled(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let y = spelled(&v[0])?;
    (1000..3000).contains(&y).then(|| term_value(y, SPELLED))?
}

/// `nineteen ninety six`: two spelled pairs of digits.
pub(crate) fn b_year_pairs(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (a, b) = (spelled(&v[0])?, spelled(&v[1])?);
    ((10..30).contains(&a) && (10..100).contains(&b)).then(|| term_value(a * 100 + b, SPELLED))?
}

/// A year that stands for itself: four digits, or words after `in`.
pub(crate) fn full_year(v: &Value) -> Option<YearRef> {
    (term(v).1 & SHORT == 0).then(|| year_ref(v))
}

// Days of the month.

pub(crate) fn b_mday_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let d = term(&v[0]).0;
    (dig_len(&v[0]) <= 2 && (1..=31).contains(&d)).then(|| term_value(d, 0))?
}

pub(crate) fn b_mday_ordinal(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let d = v[0].ordinal()?.value;
    (1..=31).contains(&d).then(|| term_value(d, ORDINAL))?
}

/// `2th`, `21th`: digits and any ordinal ending.
pub(crate) fn b_mday_suffix(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let d = term(&v[0]).0;
    (dig_len(&v[0]) <= 2 && (1..=31).contains(&d)).then(|| term_value(d, ORDINAL))?
}

pub(crate) fn b_mday_spelled(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let d = spelled(&v[0])?;
    (1..=31).contains(&d).then(|| term_value(d, SPELLED))?
}

pub(crate) fn day_fits(month: i64, day: i64) -> bool {
    let max = match month {
        2 => 30,
        4 | 6 | 9 | 11 => 30,
        1..=12 => 31,
        _ => 0,
    };
    (1..=max).contains(&day)
}

pub(crate) fn cal(year: YearRef, month: i64, day: i64, weekday: u8) -> Option<DateSpec> {
    (month == 0 || day == 0 || day_fits(month, day)).then_some(())?;
    Some(DateSpec::Cal { year, month: u8::try_from(month).ok()?, day: u8::try_from(day).ok()?, weekday })
}

pub(crate) fn md(c: &Ctx<'_>, month: i64, day: i64) -> Option<Value> {
    emit_date(c, cal(YearRef::Any, month, day, 0)?)
}

/// `one may` is no date: a month with another sense after a day takes
/// the day in digits or as an ordinal.
pub(crate) fn plain_words(month: &Value, day: &Value) -> bool {
    term(month).1 & crate::lang::en::time::AMBIG != 0 && term(day).1 & SPELLED != 0
}

/// `Apr. 1st`; a dot after a full month name ends a sentence.
pub(crate) fn b_md_month_day(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (month, flags) = term(&v[0]);
    if flags & ABBR == 0 && (c.start..c.end).any(|i| c.is_punct(i, '.')) {
        return None;
    }
    md(c, month, term(&v[1]).0)
}

pub(crate) fn b_md_day_month(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if plain_words(&v[1], &v[0]) {
        return None;
    }
    md(c, term(&v[1]).0, term(&v[0]).0)
}

pub(crate) fn b_md_month_digits(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (dig_len(&v[1]) <= 2).then(|| md(c, term(&v[0]).0, term(&v[1]).0))?
}

/// `11/20`, or `22/04` when the first number cannot be a month. After a
/// number or before `of` it is a fraction: `21 1/4`, `3/4 of a point`.
pub(crate) fn b_md_slash(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let fraction =
        c.before().is_some_and(|p| c.lexeme(p).kind == Kind::Numeral) || c.after().is_some_and(|a| c.word(a) == "of");
    if fraction || dig_len(&v[0]) > 2 || dig_len(&v[1]) > 2 {
        return None;
    }
    let (a, b) = (term(&v[0]).0, term(&v[1]).0);
    if a > 12 { md(c, b, a) } else { md(c, a, b) }
}

/// `5.12`, `7.6`: a month and a day written as a decimal.
pub(crate) fn dot_md(raw: &str) -> Option<(i64, i64)> {
    let (m, d) = raw.split_once('.')?;
    if m.is_empty()
        || m.len() > 2
        || d.is_empty()
        || d.len() > 2
        || !(m.bytes().chain(d.bytes())).all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let (m, d) = (m.parse().ok()?, d.parse().ok()?);
    ((1..=12).contains(&m) && day_fits(m, d)).then_some((m, d))
}

pub(crate) fn b_md_dot(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (m, d) = dot_md(c.raw(c.start))?;
    md(c, m, d)
}

pub(crate) fn b_weekday_dot(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (m, d) = dot_md(c.raw(c.end - 1))?;
    md(c, m, d)
}

/// The child that holds the value, the rule's others being words.
pub(crate) fn b_same(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

/// A month and day with the year from the last child.
pub(crate) fn b_md_year(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let DateSpec::Cal { year: YearRef::Any, month, day, weekday } = date_of(c, &v[0])? else { return None };
    // `11/20, 12 of Nov` lists two days.
    let comma = (c.start..c.end).any(|i| c.word(i) == ",");
    if day == 0 || (comma && term(v.last()?).1 & SHORT != 0) {
        return None;
    }
    emit_date(c, DateSpec::Cal { year: year_ref(v.last()?), month, day, weekday })
}

pub(crate) fn numeric_year(v: &Value) -> Option<YearRef> {
    match dig_len(v) {
        4 => Some(YearRef::Abs(i16::try_from(term(v).0).ok()?)),
        2 => Some(year_ref(&Value::Term { value: term(v).0, flags: SHORT })),
        _ => None,
    }
}

/// `5/3/18`, `01/22/2016`, or `13/04/21` when the first number cannot be
/// a month.
pub(crate) fn b_mdy(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) > 2 || dig_len(&v[1]) > 2 {
        return None;
    }
    let (a, b) = (term(&v[0]).0, term(&v[1]).0);
    let (m, d) = if a > 12 { (b, a) } else { (a, b) };
    emit_date(c, cal(numeric_year(&v[2])?, m, d, 0)?)
}

/// `23.04.2022`: day first.
pub(crate) fn b_dmy(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) > 2 || dig_len(&v[1]) > 2 {
        return None;
    }
    let (a, b) = (term(&v[0]).0, term(&v[1]).0);
    let (m, d) = if b > 12 { (a, b) } else { (b, a) };
    emit_date(c, cal(numeric_year(&v[2])?, m, d, 0)?)
}

pub(crate) fn b_ymd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) != 4 || dig_len(&v[1]) > 2 || dig_len(&v[2]) > 2 {
        return None;
    }
    emit_date(c, cal(YearRef::Abs(i16::try_from(term(&v[0]).0).ok()?), term(&v[1]).0, term(&v[2]).0, 0)?)
}

/// `20161016`.
pub(crate) fn b_yyyymmdd(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) != 8 {
        return None;
    }
    let n = term(&v[0]).0;
    let y = n / 10_000;
    if !(1900..2100).contains(&y) {
        return None;
    }
    emit_date(c, cal(YearRef::Abs(y as i16), n / 100 % 100, n % 100, 0)?)
}

pub(crate) fn year_digits(v: &Value) -> Option<YearRef> {
    (dig_len(v) == 4).then(|| YearRef::Abs(term(v).0 as i16))
}

pub(crate) fn small(v: &Value) -> Option<i64> {
    (dig_len(v) <= 2).then(|| term(v).0)
}

// `2019/aug/01`, `05-aug-2016`, `sep-23-2020` and the other orders of a
// year, a month name and a day.
pub(crate) fn b_y_mon_d(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[0])?, term(&v[2]).0, small(&v[4])?, 0)?)
}

pub(crate) fn b_d_mon_y(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[4])?, term(&v[2]).0, small(&v[0])?, 0)?)
}

pub(crate) fn b_mon_d_y(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[4])?, term(&v[0]).0, small(&v[2])?, 0)?)
}

pub(crate) fn b_mon_y_d(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[2])?, term(&v[0]).0, small(&v[4])?, 0)?)
}

pub(crate) fn b_y_d_mon(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[0])?, term(&v[4]).0, small(&v[2])?, 0)?)
}

pub(crate) fn b_d_y_mon(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_date(c, cal(year_digits(&v[2])?, term(&v[4]).0, small(&v[0])?, 0)?)
}

/// `2016 10 16`.
pub(crate) fn b_ymd_spaced(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[1]) != 2 || dig_len(&v[2]) != 2 {
        return None;
    }
    emit_date(c, cal(year_digits(&v[0])?, term(&v[1]).0, term(&v[2]).0, 0)?)
}

/// A weekday before a full date says nothing more: `Tue May 9`.
pub(crate) fn b_weekday_date(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & PLURAL != 0 {
        return None;
    }
    match date_of(c, v.last()?)? {
        DateSpec::Cal { month, day, .. } if month > 0 && day > 0 => Some(*v.last()?),
        _ => None,
    }
}
