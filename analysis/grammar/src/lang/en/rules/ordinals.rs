use super::*;

// Ordinals.

pub(crate) fn b_ordinal_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (value, flags) = term(&v[0]);
    Some(Value::Ordinal(Ordinal { value, flags: flags | SPELLED }))
}

pub(crate) fn b_ordinal_compound(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (t, (u, f)) = (term(&v[0]).0, term(&v[1]));
    Some(Value::Ordinal(Ordinal { value: t + u, flags: (f & PLURAL) | SPELLED }))
}

pub(crate) fn b_ordinal_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (m, (s, f)) = (num(&v[0]), term(&v[1]));
    if m.value >= pow10(s) {
        return None;
    }
    let value = m.value.checked_mul(pow10(s))?.to_i64()?;
    Some(Value::Ordinal(Ordinal { value, flags: (f & PLURAL) | SPELLED }))
}

pub(crate) fn b_ordinal_scale_bare(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (s, f) = term(&v[0]);
    Some(Value::Ordinal(Ordinal { value: pow10(s).to_i64()?, flags: (f & PLURAL) | SPELLED }))
}

pub(crate) fn b_ordinal_sum(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (l, o) = (num(&v[0]), ord(&v[1]));
    if Dec::from_i64(o.value) >= pow10(i64::from(l.grain)) {
        return None;
    }
    let value = l.value.to_i64()?.checked_add(o.value)?;
    Some(Value::Ordinal(Ordinal { value, flags: (o.flags & PLURAL) | SPELLED }))
}

/// `1st`, `22nd`, `13th`: the suffix has to be the one English gives the
/// number.
pub(crate) fn b_ordinal_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let value = num(&v[0]).value.to_i64()?;
    let want = match (value % 100, value % 10) {
        (11..=13, _) => 0,
        (_, d @ 1..=3) => d,
        _ => 0,
    };
    (term(&v[1]).0 == want).then_some(Value::Ordinal(Ordinal { value, flags: DIGITS }))
}

pub(crate) fn relative(offset: i64, to: OrdinalBase) -> Option<Value> {
    Some(Value::Relative(Relative { offset: i32::try_from(offset).ok()?, to }))
}

pub(crate) fn b_relative_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (offset, f) = term(&v[0]);
    relative(offset, if f == TO_END { OrdinalBase::End } else { OrdinalBase::Current })
}

/// `second to last` is the end minus one.
pub(crate) fn b_to_last(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    relative(1 - ord(&v[0]).value, OrdinalBase::End)
}

pub(crate) fn b_end_minus_one(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    relative(-1, OrdinalBase::End)
}

// Percentages.

pub(crate) fn b_percent_after(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Percent(num(&v[0]).value))
}

pub(crate) fn b_percent_before(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Percent(num(v.last().expect("a number")).value))
}
