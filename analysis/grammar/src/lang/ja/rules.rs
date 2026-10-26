use super::*;
use crate::dec::Dec;
use crate::engine::{Build, Item, node, node_if, opt_word, word};
use crate::unit::{CurrencyUnit, Minor, Per, Speed, Temperature, Weight, currency::code};
use crate::value::Quantity;

const fn r(name: &'static str, dim: Dim, items: &'static [Item], build: Build) -> Rule {
    Rule { name, dim, items, weight: 0, build }
}
fn q(value: Dec, unit: Unit) -> Option<Value> {
    Some(Value::Quantity(Quantity::new(value, unit)))
}
fn raw(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let w = lexicon::WORDS.get(v[0].term()?.0 as usize)?;
    let unit = if w.text == "キロ" && ["完売", "出荷", "体重", "重量"].iter().any(|s| ctx.text.contains(s)) {
        Unit::Weight(Weight::Kilogram)
    } else {
        w.unit
    };
    Some(Value::Unit(unit))
}
fn amount(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if matches!(ctx.word(ctx.end - 1), "\\" | "$" | "€" | "£" | "¥" | "￥") {
        return None;
    }
    let value = v[0].num()?.value;
    let unit = v[1].unit()?;
    if matches!(unit, Unit::Age(_)) && value.is_negative() {
        return None;
    }
    q(value, unit)
}
fn prefix(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if !matches!(ctx.word(ctx.start), "$" | "€" | "£" | "¥" | "￥" | "\\" | "us") {
        return None;
    }
    q(v[1].num()?.value, v[0].unit()?)
}
fn temperature(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let u = v[0].unit()?;
    if !matches!(u, Unit::Temperature(Temperature::Celsius | Temperature::Fahrenheit)) {
        return None;
    }
    q(v[1].num()?.value, u)
}
fn coded(s: &str) -> Option<Value> {
    Some(Value::Unit(Unit::Currency(CurrencyUnit::Code(code(s)?))))
}
fn usd(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    coded("usd")
}
fn chf(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    coded("chf")
}
fn pab(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    coded("pab")
}
fn currency(v: &Value) -> bool {
    matches!(v, Value::Quantity(q) if matches!(q.unit(), Unit::Currency(_)))
}
fn minor(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let a = v[0].quantity()?;
    let b = v[1].quantity()?;
    let Unit::Currency(major) = a.unit() else {
        return None;
    };
    let Unit::Currency(CurrencyUnit::Minor(m)) = b.unit() else {
        return None;
    };
    if matches!(major, CurrencyUnit::Minor(_)) {
        return None;
    }
    let (minor_major, exponent) = m.major();
    if !(m == Minor::Cent && major.has_hundredths() || major.currency()?.0 == minor_major.currency()?.0) {
        return None;
    }
    if b.value().is_negative() || b.value() >= Dec::from_i64(if exponent == 1 { 10 } else { 100 }) {
        return None;
    }
    q(a.value().checked_add(b.value().shift(-i32::from(exponent))?)?, a.unit())
}
fn bare_minor(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let a = v[0].quantity()?;
    let Unit::Currency(c) = a.unit() else {
        return None;
    };
    let b = v[1].num()?.value;
    if !c.has_hundredths() || !a.value().is_integer() || b.is_negative() || b >= Dec::from_i64(100) {
        return None;
    }
    q(a.value().checked_add(b.shift(-2)?)?, a.unit())
}
fn repeated(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let unit = v[0].unit()?;
    let qty = *v[1].quantity()?;
    (unit == qty.unit()).then_some(Value::Quantity(qty))
}
fn hourly(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    speed(v, Per::Hour)
}
fn secondly(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    speed(v, Per::Second)
}
fn speed(v: &[Value], per: Per) -> Option<Value> {
    let qty = v[0].quantity()?;
    let Unit::Length(length) = qty.unit() else {
        return None;
    };
    q(qty.value(), Unit::Speed(Speed::Per(length, per)))
}
pub(super) const RULES: &[Rule] = &[
    r("ja_unit", UNIT, &[node(RAW)], raw),
    r("ja_amount", QTY, &[node(cjk::NUM), node(UNIT)], amount),
    r("ja_prefix", QTY, &[node(UNIT), node(cjk::NUM)], prefix),
    r("ja_temperature", QTY, &[node(UNIT), opt_word("の"), node(cjk::NUM), word("度")], temperature),
    r("ja_usd", UNIT, &[word("us"), word("$")], usd),
    r("ja_chf", UNIT, &[word("スイス"), word("・"), word("フラン")], chf),
    r("ja_pab", UNIT, &[word("パナマ"), word("・"), word("バルボア")], pab),
    r("ja_minor", QTY, &[node_if(QTY, currency), opt_word("と"), node_if(QTY, currency)], minor),
    r("ja_bare_minor", QTY, &[node_if(QTY, currency), word("と"), node(cjk::NUM)], bare_minor),
    r("ja_repeated_currency", QTY, &[node(UNIT), node_if(QTY, currency)], repeated),
    r("ja_hour_speed", QTY, &[word("時速"), node(QTY)], hourly),
    r("ja_second_speed", QTY, &[word("毎秒"), node(QTY)], secondly),
];
