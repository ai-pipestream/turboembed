use super::*;
use crate::dec::Dec;
use crate::engine::{Build, Item, node, node_if, opt_word, word};
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
fn minor(v: &Value) -> bool {
    matches!(v.unit(), Some(Unit::Currency(CurrencyUnit::Minor(_))))
}
fn amount(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let n = v[0].num()?.value;
    let u = v[1].unit()?;
    if matches!(u, Unit::Age(_)) && n.is_negative() {
        return None;
    }
    quantity(n, u)
}
fn prefix(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if ctx.before().is_some_and(|i| cjk::parse_number(ctx.raw(i)).is_some()) {
        return None;
    }
    let u = v[0].unit()?;
    lexicon::WORDS.iter().any(|w| w.prefix && w.unit == u && w.word == ctx.word(ctx.start)).then_some(())?;
    quantity(v[1].num()?.value, u)
}
fn same_prefix(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if !matches!(ctx.word(ctx.start), "人民币" | "人民幣") {
        return None;
    }
    let Value::Quantity(q) = v[1] else { return None };
    (v[0].unit()? == q.unit()).then_some(v[1])
}
fn same_suffix(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if !matches!(ctx.word(ctx.end - 1), "人民币" | "人民幣") {
        return None;
    }
    let Value::Quantity(q) = v[0] else { return None };
    (v[1].unit()? == q.unit()).then_some(v[0])
}
fn compound_money(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let Value::Quantity(q) = v[0] else { return None };
    let Unit::Currency(c) = q.unit() else { return None };
    let Unit::Currency(CurrencyUnit::Minor(m)) = v[2].unit()? else { return None };
    let n = v[1].num()?.value;
    let (major, exponent) = m.major();
    let usd_cent = m == Minor::Cent && c == CurrencyUnit::Code(crate::unit::currency::code("usd")?);
    if (!usd_cent && c != major)
        || q.value().is_negative()
        || n.is_negative()
        || !n.is_integer()
        || n >= Dec::from_i64(10).shift(i32::from(exponent) - 1)?
    {
        return None;
    }
    // Numeric value alone loses zero-valued minor places. Inspect their source
    // units as well so `3元0角6角` cannot consume the jiao position twice.
    let repeated_place = (ctx.start..ctx.end - 1).any(|i| {
        lexicon::WORDS.iter().any(|w| {
            w.word == ctx.word(i)
                && matches!(w.unit, Unit::Currency(CurrencyUnit::Minor(prior)) if prior.major().1 >= exponent)
        })
    });
    if repeated_place || !q.value().shift(i32::from(exponent) - 1)?.is_integer() {
        return None;
    }
    quantity(q.value().checked_add(n.shift(-i32::from(exponent))?)?, q.unit())
}
fn half(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let Value::Quantity(q) = v[0] else { return None };
    (!q.value().is_negative() && !matches!(q.unit(), Unit::Currency(_))).then_some(())?;
    quantity(q.value().checked_add(Dec::new(5, 1)?)?, q.unit())
}
fn temperature(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let unit = if ctx.word(ctx.start).starts_with('华') || ctx.word(ctx.start).starts_with('華') {
        Temperature::Fahrenheit
    } else {
        Temperature::Celsius
    };
    quantity(v[0].num()?.value, Unit::Temperature(unit))
}
pub(super) const RULES: &[Rule] = &[
    r("zh_unit", UNIT, &[node(RAW)], |v, _| Some(Value::Unit(lexicon::WORDS.get(v[0].term()?.0 as usize)?.unit))),
    r("zh_amount", QTY, &[node(cjk::NUM), node(UNIT)], amount),
    r("zh_counted_amount", QTY, &[node(cjk::NUM), word("个"), node_if(UNIT, currency)], amount),
    r("zh_counted_amount_traditional", QTY, &[node(cjk::NUM), word("個"), node_if(UNIT, currency)], amount),
    r("zh_prefix", QTY, &[node_if(UNIT, currency), node(cjk::NUM)], prefix),
    r("zh_same_prefix", QTY, &[node_if(UNIT, currency), node(QTY)], same_prefix),
    r("zh_same_suffix", QTY, &[node(QTY), node_if(UNIT, currency)], same_suffix),
    r("zh_minor", QTY, &[node(QTY), opt_word("又"), node(cjk::NUM), node_if(UNIT, minor)], compound_money),
    r("zh_minor_and", QTY, &[node(QTY), word("和"), node(cjk::NUM), node_if(UNIT, minor)], compound_money),
    r("zh_half", QTY, &[node(QTY), word("半")], half),
    r("zh_fahrenheit", QTY, &[word("华氏温度"), node(cjk::NUM), word("度")], temperature),
    r("zh_fahrenheit_traditional", QTY, &[word("華氏溫度"), node(cjk::NUM), word("度")], temperature),
    r("zh_celsius", QTY, &[word("摄氏温度"), node(cjk::NUM), word("度")], temperature),
    r("zh_celsius_traditional", QTY, &[word("攝氏溫度"), node(cjk::NUM), word("度")], temperature),
    r("zh_degree_c", UNIT, &[word("°"), word("c")], |_, _| Some(Value::Unit(Unit::Temperature(Temperature::Celsius)))),
    r("zh_degree_f", UNIT, &[word("°"), word("f")], |_, _| {
        Some(Value::Unit(Unit::Temperature(Temperature::Fahrenheit)))
    }),
    r("zh_degree_k", UNIT, &[word("°"), word("k")], |_, _| Some(Value::Unit(Unit::Temperature(Temperature::Kelvin)))),
];
