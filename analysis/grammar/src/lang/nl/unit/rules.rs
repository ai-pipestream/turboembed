use super::*;
use crate::dec::Dec;
use crate::engine::{Build, Item, glued, glued_word, node, node_if, opt_word, word};
use crate::lang::nl::{NUM, SCALE};
use crate::unit::{Angle, Area, CurrencyUnit, Generic, Information, Per, Rate, Speed, Volume, currency::CODES};
use crate::value::Quantity;

const fn r(name: &'static str, dim: Dim, items: &'static [Item], build: Build) -> Rule {
    Rule { name, dim, items, weight: 0, build }
}
fn quantity(value: Dec, unit: Unit) -> Option<Value> {
    Some(Value::Quantity(Quantity::new(value, unit)))
}
fn currency(v: &Value) -> bool {
    matches!(v.unit(), Some(Unit::Currency(_)))
}
fn degree(v: &Value) -> bool {
    v.unit() == Some(Unit::Temperature(Temperature::Degree))
}
fn length(v: &Value) -> bool {
    matches!(v.unit(), Some(Unit::Length(_)))
}
fn two(v: &Value) -> bool {
    v.num().is_some_and(|n| n.value == Dec::from_i64(2))
}
fn three(v: &Value) -> bool {
    v.num().is_some_and(|n| n.value == Dec::from_i64(3))
}
fn minor(v: &Value) -> bool {
    matches!(v.unit(), Some(Unit::Currency(CurrencyUnit::Minor(_) | CurrencyUnit::Generic(Generic::Cent))))
}
fn whole_money(v: &Value) -> bool {
    matches!(v, Value::Quantity(q) if matches!(q.unit(), Unit::Currency(c) if c.has_hundredths()) && (q.value().is_integer() || cny_tenths(*q)) && !q.value().is_negative())
}
fn amount(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    with_unit(v[0].num()?.value, v[1].unit()?, ctx)
}
fn with_unit(mut value: Dec, unit: Unit, ctx: &Ctx<'_>) -> Option<Value> {
    if matches!(unit, Unit::Age(_)) && value.is_negative() {
        return None;
    }
    let factor = match ctx.word(ctx.end - 1) {
        "ons" => 100,
        "pond" | "ponden" => 500,
        "kwartje" | "kwartjes" => 25,
        "dubbeltje" | "dubbeltjes" => 10,
        "stuiver" | "stuivers" => 5,
        _ => 1,
    };
    value = value.checked_mul(Dec::from_i64(factor))?;
    if ctx.word(ctx.end - 1) == "penni" {
        value = value.shift(-2)?;
    }
    quantity(value, unit)
}
fn prefix_amount(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let w = ctx.word(ctx.start);
    let unit = v[0].unit()?;
    let named = lexicon::WORDS.iter().any(|x| x.word == w && x.unit == unit && x.flags & PREFIX != 0);
    let coded = DUTCH_UNITS.lexicon().lookup(w).iter().any(|e| e.dim == CODE);
    let symbol = matches!(w, "$" | "€" | "£" | "元");
    if !(named || coded || symbol) {
        return None;
    }
    quantity(v[1].num()?.value, unit)
}
fn raw(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (i, _) = v[0].term()?;
    let entry = lexicon::WORDS.get(i as usize)?;
    let mut unit = entry.unit;
    if ctx.raw(ctx.start).ends_with('B') {
        unit = match unit {
            Unit::Information(Information::Megabit) => Unit::Information(Information::Megabyte),
            Unit::Information(Information::Gigabit) => Unit::Information(Information::Gigabyte),
            Unit::Information(Information::Kilobit) => Unit::Information(Information::Kilobyte),
            u => u,
        };
    }
    Some(Value::Unit(unit))
}
fn coded(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let code = CODES.get(v[0].term()?.0 as usize)?.0;
    // Ambiguous short currency codes require capitals.
    if matches!(code, "all" | "cop" | "dem" | "esp" | "las" | "mad" | "pen" | "son" | "try")
        && ctx.raw(ctx.start).chars().any(turbo_unicode::props::lowercase)
    {
        return None;
    }
    Some(Value::Unit(lexicon::coded(code)?))
}
fn square(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let Unit::Length(l) = v[0].unit()? else { return None };
    Some(Value::Unit(Unit::Area(Area::Square(l))))
}
fn cubic(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let Unit::Length(l) = v[0].unit()? else { return None };
    Some(Value::Unit(Unit::Volume(Volume::Cubic(l))))
}
fn scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let u = v[1].unit()?;
    matches!(u, Unit::Temperature(Temperature::Celsius | Temperature::Fahrenheit | Temperature::Kelvin))
        .then_some(Value::Unit(u))
}
fn letter_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Temperature(if v[1].term()?.0 == 0 {
        Temperature::Celsius
    } else {
        Temperature::Fahrenheit
    })))
}
fn speed(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let per = lexicon::TIMES.get(v[1].term()?.0 as usize)?.1;
    let u = match v[0].unit()? {
        Unit::Length(l) if matches!(per, Per::Second | Per::Minute | Per::Hour) => Unit::Speed(Speed::Per(l, per)),
        Unit::Information(i) => Unit::Rate(Rate::Information(i, per)),
        _ => return None,
    };
    Some(Value::Unit(u))
}
fn money_rate(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let Value::Quantity(q) = v[0] else { return None };
    let Unit::Currency(c) = q.unit() else { return None };
    quantity(q.value(), Unit::Rate(Rate::Currency(c, lexicon::TIMES.get(v[1].term()?.0 as usize)?.1)))
}
fn cny_tenths(q: Quantity) -> bool {
    Some(q.unit()) == lexicon::coded("cny") && q.value().shift(1).is_some_and(|n| n.is_integer())
}
fn hundredths(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let Value::Quantity(q) = v[0] else { return None };
    let n = v[1].num()?.value;
    let mut exponent = 2;
    if (ctx.start + 1..ctx.end).any(|i| {
        ctx.lexeme(i).kind == Kind::Numeral
            && ctx.lexeme(i).glued
            && money_token(ctx, i - 1)
            && (ctx.start..i - 1).any(|p| ctx.lexeme(p).kind == Kind::Numeral)
    }) {
        return None;
    }
    if !n.is_integer() || n.is_negative() {
        return None;
    }
    if let Some(u) = v.get(2) {
        let Unit::Currency(c) = q.unit() else { return None };
        match u.unit()? {
            Unit::Currency(CurrencyUnit::Generic(Generic::Cent)) if c.has_hundredths() => {}
            Unit::Currency(CurrencyUnit::Minor(m))
                if m.major().0.currency() == c.currency() && c.currency().is_some() =>
            {
                exponent = m.major().1 as i32;
            }
            _ => return None,
        }
    } else if ctx.after().is_some_and(|i| {
        DUTCH_UNITS.lexicon().lookup(ctx.word(i)).iter().any(|e| matches!(e.dim, RAW | CODE | TIME | SCALE))
            || matches!(ctx.word(i), "keer" | "personen")
    }) {
        return None;
    }
    if n >= Dec::from_i64(1).shift(exponent)? || !q.value().is_integer() && (exponent != 2 || n >= Dec::from_i64(10)) {
        return None;
    }
    quantity(q.value().checked_add(n.shift(-exponent)?)?, q.unit())
}
fn amount_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let exponent = v[2].term()?.0;
    quantity(v[0].num()?.value.shift(exponent as i32)?, v[1].unit()?)
}
pub(super) const RULES: &[Rule] = &[
    r("nl_unit_word", UNIT, &[node(RAW)], raw),
    r("nl_unit_code", UNIT, &[node(CODE)], coded),
    r("nl_unit_code_dollar", UNIT, &[node(CODE), glued_word("$")], coded),
    r("nl_unit_country", UNIT, &[node(COUNTRY), node_if(UNIT, currency)], |v, _| {
        let (_, generic, code) = lexicon::COUNTRIES.get(v[0].term()?.0 as usize)?;
        (v[1].unit()? == Unit::Currency(CurrencyUnit::Generic(*generic)))
            .then(|| Value::Unit(lexicon::coded(code).unwrap()))
    }),
    r("nl_unit_square", UNIT, &[word("vierkante"), node_if(UNIT, length)], square),
    r("nl_unit_square_num", UNIT, &[node_if(UNIT, length), crate::engine::glued_if(NUMERAL, two)], square),
    r("nl_unit_square_symbol", UNIT, &[node_if(UNIT, length), glued_word("²")], square),
    r(
        "nl_unit_square_caret",
        UNIT,
        &[node_if(UNIT, length), glued_word("^"), crate::engine::glued_if(NUMERAL, two)],
        square,
    ),
    r("nl_unit_cubic", UNIT, &[word("kubieke"), node_if(UNIT, length)], cubic),
    r("nl_unit_cubic_num", UNIT, &[node_if(UNIT, length), crate::engine::glued_if(NUMERAL, three)], cubic),
    r("nl_unit_cubic_symbol", UNIT, &[node_if(UNIT, length), glued_word("³")], cubic),
    r("nl_unit_metric", UNIT, &[word("metrische"), node(UNIT)], |v, _| {
        matches!(v[0].unit()?, Unit::Weight(crate::unit::Weight::MetricTon)).then_some(v[0])
    }),
    r("nl_unit_angle", UNIT, &[node_if(UNIT, degree)], |_, _| Some(Value::Unit(Unit::Angle(Angle::Degree)))),
    r("nl_unit_scale", UNIT, &[node_if(UNIT, degree), node(UNIT)], scale),
    r("nl_unit_scale_letter", UNIT, &[node_if(UNIT, degree), node(SCALE_WORD)], letter_scale),
    r("nl_unit_scale_abbr", UNIT, &[node_if(UNIT, degree), word("."), node(SCALE_WORD)], letter_scale),
    r("nl_unit_age_old", UNIT, &[node(UNIT), word("oud")], |v, _| matches!(v[0].unit()?, Unit::Age(_)).then_some(v[0])),
    r("nl_unit_age_life", UNIT, &[node(UNIT), word("leeftijd")], |v, _| {
        matches!(v[0].unit()?, Unit::Age(_)).then_some(v[0])
    }),
    r("nl_unit_rate", UNIT, &[node(UNIT), word("per"), node(TIME)], speed),
    r("nl_unit_rate_slash", UNIT, &[node(UNIT), word("/"), node(TIME)], speed),
    r("nl_unit_amount", QTY, &[node(NUM), node(UNIT)], amount),
    r("nl_unit_half", QTY, &[word("een"), word("halve"), node(UNIT)], |v, ctx| {
        with_unit(Dec::new(5, 1)?, v[0].unit()?, ctx)
    }),
    r("nl_unit_hyphen", QTY, &[node(NUM), glued_word("-"), glued(UNIT)], amount),
    r("nl_unit_money_before", QTY, &[node_if(UNIT, currency), node(NUM)], prefix_amount),
    r("nl_unit_money_scale", QTY, &[node(NUM), node_if(UNIT, currency), node(SCALE)], amount_scale),
    r(
        "nl_unit_money_compound",
        QTY,
        &[node_if(QTY, whole_money), opt_word("en"), node(NUM), node_if(UNIT, minor)],
        hundredths,
    ),
    r("nl_unit_money_bare", QTY, &[node_if(QTY, whole_money), opt_word("en"), node(NUM)], hundredths),
    r("nl_unit_money_rate", QTY, &[node(QTY), word("per"), node(TIME)], money_rate),
    r("nl_unit_money_rate_slash", QTY, &[node(QTY), word("/"), node(TIME)], money_rate),
];
