use super::*;

// Fractions.

pub(crate) fn b_den_ordinal(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

pub(crate) fn b_den_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (d, f) = term(&v[0]);
    Some(Value::Ordinal(Ordinal { value: d, flags: f | SPELLED }))
}

pub(crate) fn b_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (n, d) = (num(&v[0]).value, ord(&v[1]));
    // One takes the singular (`one fifth`), anything else the plural.
    if (n == Dec::ONE) == d.has(PLURAL) {
        return None;
    }
    fraction(n.checked_div(Dec::from_i64(d.value))?, SPELLED)
}

pub(crate) fn b_article_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    fraction(Dec::ONE.checked_div(Dec::from_i64(ord(&v[1]).value))?, SPELLED)
}

pub(crate) fn b_half(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    fraction(Dec::ONE.checked_div(Dec::from_i64(2))?, SPELLED)
}

pub(crate) fn b_mixed(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (w, f) = (num(&v[0]), num(&v[1]));
    fraction(w.value.checked_add(f.value)?, w.flags | f.flags)
}

/// `one million and a half`: the fraction is of the scale.
pub(crate) fn b_scale_and_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (w, f) = (num(&v[0]), num(&v[1]));
    fraction(w.value.checked_add(f.value.checked_mul(pow10(i64::from(w.grain)))?)?, w.flags | f.flags)
}

/// `one billion and three quarters million`: a fractional part below the
/// whole's last scale.
pub(crate) fn b_scale_plus_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (w, f) = (num(&v[0]), num(&v[1]));
    if f.value >= pow10(i64::from(w.grain)) {
        return None;
    }
    fraction(w.value.checked_add(f.value)?, w.flags | f.flags)
}

/// `two thirds of a million`, `five and a half million`.
pub(crate) fn b_fraction_of_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (f, s) = (num(&v[0]), term(&v[1]).0);
    if f.value >= pow10(s) {
        return None;
    }
    fraction(f.value.checked_mul(pow10(s))?, f.flags)
}

/// `half million`, `quarter of a billion`.
pub(crate) fn b_word_of_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (d, s) = (term(&v[0]).0, term(&v[1]).0);
    fraction(pow10(s).checked_div(Dec::from_i64(d))?, SPELLED)
}

/// `half a dozen` is six, an integer.
pub(crate) fn of_dozen(value: Dec, flags: u8) -> Option<Value> {
    let value = value.checked_mul(Dec::from_i64(12))?;
    let kind = if value.is_integer() { NumKind::Integer } else { NumKind::Fraction };
    number(value, kind, flags)
}

pub(crate) fn b_fraction_of_dozen(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let f = num(&v[0]);
    of_dozen(f.value, f.flags)
}

pub(crate) fn b_word_of_dozen(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    of_dozen(Dec::ONE.checked_div(Dec::from_i64(term(&v[0]).0))?, SPELLED)
}

pub(crate) fn b_digit_fraction(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if !slash_alone(ctx) {
        return None;
    }
    fraction(num(&v[0]).value.checked_div(num(&v[1]).value)?, DIGITS)
}

pub(crate) fn b_vulgar(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let t = term(&v[0]).0;
    fraction(Dec::from_i64(t / 1000).checked_div(Dec::from_i64(t % 1000))?, DIGITS)
}

pub(crate) fn b_over(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if after_currency(ctx) {
        return None;
    }
    let (n, d) = (num(&v[0]), num(&v[1]));
    fraction(n.value.checked_div(d.value)?, n.flags | d.flags)
}

/// `one in three`, `two out of five`: only a part below the whole reads
/// as a fraction, so `12 in six boxes` is two numbers.
pub(crate) fn b_part_of(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (n, d) = (num(&v[0]), num(&v[1]));
    if n.value >= d.value {
        return None;
    }
    b_over(v, ctx)
}

/// `1 in 5`; but `3 in 2009` is a number and a year.
pub(crate) fn b_one_in(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let d = num(&v[1]);
    let year = digits_only(&d) && d.value >= Dec::from_i64(1000) && d.value < Dec::from_i64(3000);
    if year {
        return None;
    }
    b_part_of(v, ctx)
}

// Spoken decimals: `two point zero three`.

pub(crate) fn digits_value(value: Dec, count: i8) -> Option<Value> {
    Some(Value::Num(Num { value, kind: NumKind::Integer, grain: count, flags: SPELLED }))
}

pub(crate) fn b_digit_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    digits_value(Dec::from_i64(term(&v[0]).0), 1)
}

pub(crate) fn b_more_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let d = num(&v[0]);
    if d.grain >= 30 {
        return None;
    }
    digits_value(d.value.checked_mul(Dec::from_i64(10))?.checked_add(Dec::from_i64(term(&v[1]).0))?, d.grain + 1)
}

pub(crate) fn b_two_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    digits_value(num(&v[0]).value, 2)
}

pub(crate) fn b_point(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (w, d) = (num(&v[0]), num(&v[1]));
    let value = w.value.checked_add(d.value.shift(-i32::from(d.grain))?)?;
    number(value, NumKind::Decimal, SPELLED)
}

// Exponents.

pub(crate) fn b_plus(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Term { value: 1, flags: 0 })
}

pub(crate) fn b_minus(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Term { value: -1, flags: 0 })
}

pub(crate) fn b_sign_plus(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

pub(crate) fn b_sign_minus(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Term { value: -term(&v[0]).0, flags: 0 })
}

/// The exponent written last, with the sign run before it if any.
pub(crate) fn exponent(v: &[Value]) -> Option<i32> {
    let e = num(v.last().expect("an exponent")).value.to_i64()?;
    let sign = if v.len() > 2 { term(&v[v.len() - 2]).0 } else { 1 };
    i32::try_from(e * sign).ok()
}

pub(crate) fn power(value: Dec) -> Option<Value> {
    number(value, NumKind::Power, DIGITS)
}

pub(crate) fn b_ten_power(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    power(num(&v[0]).value.shift(exponent(v)?)?)
}

/// `1.3x10^6`: the ten is a child too, before the optional sign.
pub(crate) fn b_times_ten_power(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let e = exponent(&v[1..])?;
    power(num(&v[0]).value.shift(e)?)
}

pub(crate) fn b_caret(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let e = exponent(v)?;
    if e.unsigned_abs() > 64 {
        return None;
    }
    power(num(&v[0]).value.pow(e)?)
}
