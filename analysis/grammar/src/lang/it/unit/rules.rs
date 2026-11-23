use super::*;
use crate::dec::Dec;
use crate::engine::{Build, Item, glued, glued_word, node, node_if, opt_word, word};
use crate::lang::it::{NUM, SCALE};
use crate::unit::{Area, Minor, Rate, Volume};
use lexicon::Words;

const fn r(name: &'static str, dim: Dim, items: &'static [Item], build: Build) -> Rule {
    Rule { name, dim, items, weight: 0, build }
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
    matches!(v.unit(), Some(Unit::Currency(CurrencyUnit::Minor(_))))
}
fn whole_money(v: &Value) -> bool {
    matches!(v, Value::Quantity(q) if matches!(q.unit(), Unit::Currency(c) if c.has_hundredths()) && q.value().is_integer() && !q.value().is_negative())
}
fn amount(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (value, unit) = (v[0].num()?.value, v[1].unit()?);
    if matches!(unit, Unit::Age(_)) && value.is_negative() {
        return None;
    }
    quantity(value, unit)
}
fn prefix_amount(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let w = ctx.word(ctx.start);
    let unit = v[0].unit()?;
    let symbol = matches!(w, "$" | "€" | "£" | "¥");
    let coded = ITALIAN_UNITS.lexicon().lookup(w).iter().any(|e| e.dim == CODE);
    let composite = matches!(unit, Unit::Currency(CurrencyUnit::Code(_)))
        && ctx.start + 1 < ctx.end
        && ctx.word(ctx.start + 1) == "$";
    let named = lexicon::WORDS.iter().any(|x| x.word == w && x.unit == unit && x.flags & PREFIX != 0)
        && ctx.start + 1 < ctx.end
        && !matches!(ctx.word(ctx.start + 1), "un" | "uno" | "una");
    if !(symbol || coded || composite || named) {
        return None;
    }
    quantity(v[1].num()?.value, unit)
}
fn country(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (_, generic, code) = lexicon::COUNTRIES.get(v[1].term()?.0 as usize)?;
    (v[0].unit()? == Unit::Currency(CurrencyUnit::Generic(*generic)))
        .then(|| Value::Unit(code_unit(code).expect("currency code")))
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
fn money_rate(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let Value::Quantity(q) = v[0] else { return None };
    let Unit::Currency(c) = q.unit() else { return None };
    quantity(q.value(), Unit::Rate(Rate::Currency(c, lexicon::TIMES.get(v[1].term()?.0 as usize)?.1)))
}
fn hundredths(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let Value::Quantity(q) = v[0] else { return None };
    let n = v[1].num()?.value;
    if (ctx.start + 1..ctx.end).any(|i| {
        ctx.lexeme(i).kind == Kind::Numeral
            && ctx.lexeme(i).glued
            && money_token(ctx, i - 1)
            && (ctx.start..i - 1).any(|p| ctx.lexeme(p).kind == Kind::Numeral)
    }) {
        return None;
    }
    if !n.is_integer() || n.is_negative() || n >= Dec::from_i64(100) {
        return None;
    }
    if let Some(u) = v.get(2) {
        let Unit::Currency(CurrencyUnit::Minor(m)) = u.unit()? else { return None };
        let Unit::Currency(c) = q.unit() else { return None };
        let fits = match m {
            Minor::Cent => {
                matches!(c, CurrencyUnit::Generic(Generic::Dollar))
                    || c == CurrencyUnit::Code(crate::unit::currency::code("eur")?)
                    || c == CurrencyUnit::Code(crate::unit::currency::code("usd")?)
            }
            Minor::Centavo => {
                matches!(c, CurrencyUnit::Generic(Generic::Dollar | Generic::Peso))
                    || ["usd", "cad", "aud", "nzd", "hkd", "mxn", "eur", "brl"]
                        .iter()
                        .any(|s| code_unit(s) == Some(Unit::Currency(c)))
            }
            _ => m.major().0.currency() == c.currency() && c.currency().is_some(),
        };
        if !fits {
            return None;
        }
    } else if ctx.after().is_some_and(|i| {
        ITALIAN_UNITS.lexicon().lookup(ctx.word(i)).iter().any(|e| matches!(e.dim, RAW | CODE | TIME | SCALE))
            || matches!(ctx.word(i), "volte" | "persone")
    }) {
        return None;
    }
    quantity(q.value().checked_add(n.shift(-2)?)?, q.unit())
}
fn amount_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let exponent = v[2].term()?.0;
    quantity(v[0].num()?.value.shift(exponent as i32)?, v[1].unit()?)
}
fn fixed_currency(code: &str) -> Option<Value> {
    Some(Value::Unit(code_unit(code)?))
}
pub(super) const RULES: &[Rule] = &[
    r("it_unit_quadri", UNIT, &[node_if(UNIT, length), word("quadri")], square),
    r("it_unit_all_ora", UNIT, &[node(UNIT), word("all"), word("'"), node(TIME)], speed::<Words>),
    r("it_unit_all_ora_curly", UNIT, &[node(UNIT), word("all"), word("’"), node(TIME)], speed::<Words>),
    r("it_unit_al_time", UNIT, &[node(UNIT), word("al"), node(TIME)], speed::<Words>),
    r(
        "it_unit_age_context",
        QTY,
        &[node_if(QTY, |v| matches!(v,Value::Quantity(q)if matches!(q.unit(),Unit::Age(_)))), word("di"), word("eta")],
        |v, _| Some(v[0]),
    ),
    r("it_unit_half", QTY, &[word("mezza"), node(UNIT)], |v, _| quantity(Dec::new(5, 1)?, v[0].unit()?)),
    r("it_unit_half_masc", QTY, &[word("mezzo"), node(UNIT)], |v, _| quantity(Dec::new(5, 1)?, v[0].unit()?)),
    r("it_unit_word", UNIT, &[node(RAW)], raw::<Words>),
    r("it_unit_code", UNIT, &[node(CODE)], coded),
    r("it_unit_code_dollar", UNIT, &[node(CODE), glued_word("$")], coded),
    r("it_unit_canadian_sign", UNIT, &[word("c"), word("$")], |_, _| fixed_currency("cad")),
    r("it_unit_hkd_sign", UNIT, &[word("hk"), word("$")], |_, _| fixed_currency("hkd")),
    r("it_unit_us_sign", UNIT, &[word("us"), word("$")], |_, _| fixed_currency("usd")),
    r("it_unit_usd_sign", UNIT, &[word("u"), glued_word("$"), glued_word("d")], |_, _| fixed_currency("usd")),
    r("it_unit_country", UNIT, &[node_if(UNIT, currency), node(COUNTRY)], country),
    r("it_unit_hongkong", UNIT, &[node_if(UNIT, currency), word("di"), word("hong"), word("kong")], |v, _| {
        (v[0].unit()? == Unit::Currency(CurrencyUnit::Generic(Generic::Dollar)))
            .then_some(Value::Unit(code_unit("hkd")?))
    }),
    r("it_unit_dutch_guilder", UNIT, &[node_if(UNIT, currency), word("olandesi")], |v, _| {
        (v[0].unit()? == code_unit("nlg")?).then_some(v[0])
    }),
    r("it_unit_square", UNIT, &[node_if(UNIT, length), word("quadrato")], square),
    r("it_unit_squares", UNIT, &[node_if(UNIT, length), word("quadrati")], square),
    r("it_unit_square_num", UNIT, &[node_if(UNIT, length), crate::engine::glued_if(NUMERAL, two)], square),
    r("it_unit_square_symbol", UNIT, &[node_if(UNIT, length), glued_word("²")], square),
    r("it_unit_cubic", UNIT, &[node_if(UNIT, length), word("cubo")], cubic),
    r("it_unit_cubics", UNIT, &[node_if(UNIT, length), word("cubi")], cubic),
    r("it_unit_cubic_num", UNIT, &[node_if(UNIT, length), crate::engine::glued_if(NUMERAL, three)], cubic),
    r("it_unit_cubic_symbol", UNIT, &[node_if(UNIT, length), glued_word("³")], cubic),
    r("it_unit_metric", UNIT, &[node(UNIT), word("metrica")], |v, _| {
        matches!(v[0].unit()?, Unit::Weight(Weight::MetricTon | Weight::ShortTon))
            .then_some(Value::Unit(Unit::Weight(Weight::MetricTon)))
    }),
    r("it_unit_metrics", UNIT, &[node(UNIT), word("metriche")], |v, _| {
        matches!(v[0].unit()?, Unit::Weight(Weight::MetricTon | Weight::ShortTon))
            .then_some(Value::Unit(Unit::Weight(Weight::MetricTon)))
    }),
    r("it_unit_metric_abbr", UNIT, &[word("t"), glued_word("."), glued_word("metrica")], |_, _| {
        Some(Value::Unit(Unit::Weight(Weight::MetricTon)))
    }),
    r("it_unit_angle", UNIT, &[node_if(UNIT, degree)], |_, _| Some(Value::Unit(Unit::Angle(Angle::Degree)))),
    r("it_unit_scale", UNIT, &[node_if(UNIT, degree), node(UNIT)], scale),
    r("it_unit_scale_letter", UNIT, &[node_if(UNIT, degree), node(SCALE_WORD)], letter_scale),
    r("it_unit_rate", UNIT, &[node(UNIT), word("per"), node(TIME)], speed::<Words>),
    r("it_unit_rate_slash", UNIT, &[node(UNIT), word("/"), node(TIME)], speed::<Words>),
    r("it_unit_amount", QTY, &[node(NUM), opt_word("di"), node(UNIT)], amount),
    r("it_unit_hyphen", QTY, &[node(NUM), glued_word("-"), glued(UNIT)], amount),
    r("it_unit_money_before", QTY, &[node_if(UNIT, currency), node(NUM)], prefix_amount),
    r("it_unit_money_scale", QTY, &[node(NUM), node_if(UNIT, currency), node(SCALE)], amount_scale),
    r(
        "it_unit_money_compound",
        QTY,
        &[node_if(QTY, whole_money), opt_word("e"), node(NUM), node_if(UNIT, minor)],
        hundredths,
    ),
    r("it_unit_money_con", QTY, &[node_if(QTY, whole_money), word("con"), node(NUM), node_if(UNIT, minor)], hundredths),
    r("it_unit_money_bare", QTY, &[node_if(QTY, whole_money), opt_word("e"), node(NUM)], hundredths),
    r("it_unit_money_bare_con", QTY, &[node_if(QTY, whole_money), word("con"), node(NUM)], hundredths),
    r("it_unit_money_rate", QTY, &[node(QTY), word("per"), node(TIME)], money_rate),
    r("it_unit_money_rate_slash", QTY, &[node(QTY), word("/"), node(TIME)], money_rate),
    r("it_unit_compound", QTY, &[node(QTY), opt_word("e"), node(QTY)], compound),
];
