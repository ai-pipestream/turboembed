use super::*;

// Integers.

pub(crate) fn b_word_int(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(term(&v[0]).0), 0, SPELLED)
}

pub(crate) fn b_tens_unit(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(term(&v[0]).0 + term(&v[1]).0), 0, SPELLED)
}

pub(crate) fn b_numeral_int(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

pub(crate) fn b_hundred(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let m = num(&v[0]);
    int(m.value.checked_mul(pow10(2))?, 2, m.flags | SPELLED)
}

pub(crate) fn b_a_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let s = term(&v[1]).0;
    int(pow10(s), s as i8, SPELLED)
}

pub(crate) fn b_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (m, s) = (num(&v[0]), term(&v[1]).0);
    // A plain digit integer takes any scale (`322 thousand`); otherwise
    // the multiplicand is below the scale, so `two hundred thousand`
    // reads and `two million thousand` does not.
    if !(m.grain == 0 && digits_only(&m)) && m.value >= pow10(s) {
        return None;
    }
    int(m.value.checked_mul(pow10(s))?, s as i8, m.flags | SPELLED)
}

pub(crate) fn b_sum(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (l, r) = (num(&v[0]), num(&v[1]));
    if r.grain >= l.grain || r.value >= pow10(i64::from(l.grain)) || before_multiplier(ctx, l.grain) {
        return None;
    }
    int(l.value.checked_add(r.value)?, r.grain, l.flags | r.flags)
}

pub(crate) fn b_dozen(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let m = num(&v[0]);
    int(m.value.checked_mul(Dec::from_i64(12))?, 0, m.flags | SPELLED)
}

pub(crate) fn b_a_dozen(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(12), 0, SPELLED)
}

/// A number from its parts' sole child, unchanged.
pub(crate) fn b_same(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

pub(crate) fn b_numeral_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (m, s) = (num(&v[0]), term(&v[1]).0);
    let value = m.value.checked_mul(pow10(s))?;
    number(value, NumKind::of(value), DIGITS | SPELLED)
}

pub(crate) fn suffix(v: &[Value], e: i64) -> Option<Value> {
    let value = num(&v[0]).value.checked_mul(pow10(e))?;
    number(value, NumKind::of(value), DIGITS)
}

pub(crate) fn b_neg_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let n = num(&v[1]);
    Some(Value::Num(Num { value: n.value.negated(), flags: n.flags | SPELLED, ..n }))
}

pub(crate) fn b_neg_sign(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if !minus_opens(ctx) {
        return None;
    }
    let n = num(&v[0]);
    Some(Value::Num(Num { value: n.value.negated(), ..n }))
}
