use super::*;

// Quantities.

pub(super) fn b_amount_unit(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value, unit(&v[1]))
}

pub(super) fn b_unit_amount(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[1]).value, unit(&v[0]))
}

/// `$3.5m`, `€2bn`.
pub(super) fn b_unit_amount_multiplier(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[1]).value.checked_mul(pow10(term(&v[2]).0)?)?, unit(&v[0]))
}

/// `75 USD million`.
pub(super) fn b_amount_unit_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value.checked_mul(pow10(term(&v[2]).0)?)?, unit(&v[1]))
}

pub(super) fn million_code(v: &Value) -> Option<Unit> {
    Some(Unit::Currency(CurrencyUnit::Code(code(CODES[term(v).0 as usize].0)?)))
}

/// `30 MUSD`.
pub(super) fn b_amount_million_code(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value.checked_mul(pow10(6)?)?, million_code(&v[1])?)
}

/// `MAUD 60`.
pub(super) fn b_million_code_amount(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[1]).value.checked_mul(pow10(6)?)?, million_code(&v[0])?)
}

/// `13 dollars and 45 cents`, `10 $ 30 cent`, `3 pounds 20 pence`.
pub(super) fn b_major_minor(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (major, n) = (qty(&v[0]), num(&v[1]));
    let Unit::Currency(CurrencyUnit::Minor(m)) = unit(&v[2]) else { return None };
    let Unit::Currency(c) = major.unit() else { return None };
    let (of, e) = m.major();
    let fits = of == c
        || (m == Minor::Cent && c.has_hundredths())
        || (of.currency() == c.currency() && of.currency().is_some());
    if !fits || !n.value.is_integer() || n.value.is_negative() || n.value >= pow10(i64::from(e))? {
        return None;
    }
    quantity(major.value().checked_add(n.value.shift(-i32::from(e))?)?, major.unit())
}

/// `15 dollars 50`, `c $ 100 and fifty`: a bare count of hundredths ends
/// the sentence or a clause, or comes before `more`, `less`, `extra` or
/// `each`, so `15 dollars 3 times` is no 15.03.
pub(super) fn b_major_hundredths(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (major, n) = (qty(&v[0]), num(&v[1]));
    let sign = |i: usize| ctx.punct(i).is_some_and(is_currency_sign);
    // `5 $10` is 5 and $10, not 5.10.
    if (ctx.start + 2..ctx.end).any(|i| ctx.lexeme(i).kind == Kind::Numeral && ctx.lexeme(i).glued && sign(i - 1)) {
        return None;
    }
    let ends = match ctx.after() {
        None => true,
        Some(a) => match ctx.lexeme(a).kind {
            Kind::Punct => !sign(a),
            Kind::Word => matches!(ctx.word(a), "and" | "or" | "but" | "more" | "less" | "extra" | "each"),
            Kind::Numeral => false,
        },
    };
    if !ends {
        return None;
    }
    quantity(major.value().checked_add(n.value.shift(-2)?)?, major.unit())
}

pub(super) fn is_currency_sign(c: char) -> bool {
    turbo_unicode::category::general_category(c) == turbo_unicode::category::GeneralCategory::Sc
}

/// `5 ft 2 in`, `5 feet and 2 inches`.
pub(super) fn b_feet_inches(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (f, n) = (qty(&v[0]), num(&v[1]));
    if n.value >= Dec::from_i64(12) {
        return None;
    }
    quantity(f.value().checked_add(n.value.checked_div(Dec::from_i64(12))?)?, f.unit())
}

/// `6'2"`, `6′2″`: feet and inches with marks.
pub(super) fn b_feet_inch_marks(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (f, n) = (num(&v[0]), num(&v[1]));
    quantity(f.value.checked_add(n.value.checked_div(Dec::from_i64(12))?)?, Unit::Length(Length::Foot))
}

pub(super) fn b_feet_mark(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value, Unit::Length(Length::Foot))
}

/// `2 m dollars`, `4.5 bn euros`.
pub(super) fn b_amount_multiplier_unit(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value.checked_mul(pow10(term(&v[1]).0)?)?, unit(&v[2]))
}

pub(super) fn b_age(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let age = match Time::of(term(&v[1]).0) {
        Time::Day => Age::Day,
        Time::Week => Age::Week,
        Time::Month => Age::Month,
        _ => Age::Year,
    };
    quantity(num(&v[0]).value, Unit::Age(age))
}

pub(super) fn b_age_years(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value, Unit::Age(Age::Year))
}

/// `age 87`, `aged 87`, in lowercase or at the start of a sentence.
pub(super) fn b_age_stated(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let opens = ctx.before().is_none_or(|p| ctx.lexeme(p).kind == Kind::Punct);
    if has_upper(raw(ctx, ctx.start)) && !opens {
        return None;
    }
    quantity(num(&v[0]).value, Unit::Age(Age::Unstated))
}
