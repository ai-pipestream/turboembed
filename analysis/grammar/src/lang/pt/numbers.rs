use super::*;
use crate::dec::Dec;
use crate::engine::{Build, Entry, Item, glued_if, glued_word, node, node_if, word};
use crate::lang::en::rules as en;
use crate::value::{Num, NumKind, Value, flag};
const fn e(word: &'static str, value: i64) -> Entry {
    Entry { word, dim: WORD, value, flags: 0 }
}
const fn s(word: &'static str, value: i64) -> Entry {
    Entry { word, dim: SCALE, value, flags: 0 }
}
pub(super) const ENTRIES: &[Entry] = &[
    e("zero", 0),
    e("um", 1),
    e("uma", 1),
    e("dois", 2),
    e("duas", 2),
    e("tres", 3),
    e("quatro", 4),
    e("cinco", 5),
    e("seis", 6),
    e("sete", 7),
    e("oito", 8),
    e("nove", 9),
    e("dez", 10),
    e("onze", 11),
    e("doze", 12),
    e("treze", 13),
    e("catorze", 14),
    e("quatorze", 14),
    e("quinze", 15),
    e("dezasseis", 16),
    e("dezesseis", 16),
    e("dezassete", 17),
    e("dezessete", 17),
    e("dezoito", 18),
    e("dezanove", 19),
    e("dezenove", 19),
    e("vinte", 20),
    e("trinta", 30),
    e("quarenta", 40),
    e("cinquenta", 50),
    e("sessenta", 60),
    e("setenta", 70),
    e("oitenta", 80),
    e("noventa", 90),
    e("cem", 100),
    e("cento", 100),
    e("duzentos", 200),
    e("duzentas", 200),
    e("trezentos", 300),
    e("trezentas", 300),
    e("quatrocentos", 400),
    e("quatrocentas", 400),
    e("quinhentos", 500),
    e("quinhentas", 500),
    e("seiscentos", 600),
    e("seiscentas", 600),
    e("setecentos", 700),
    e("setecentas", 700),
    e("oitocentos", 800),
    e("oitocentas", 800),
    e("novecentos", 900),
    e("novecentas", 900),
    s("mil", 3),
    s("milhao", 6),
    s("milhoes", 6),
    s("bilhao", 9),
    s("bilhoes", 9),
    s("biliao", 12),
    s("bilioes", 12),
    s("trilhao", 12),
    s("trilhoes", 12),
];
const fn r(name: &'static str, items: &'static [Item], build: Build) -> Rule {
    Rule { name, dim: NUM, items, weight: 0, build }
}
fn word_number(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let n = v[0].term()?.0;
    let mut num = Num::new(Dec::from_i64(n), NumKind::Integer, flag::SPELLED);
    if ctx.word(ctx.start) == "cem" {
        num.flags |= flag::GRAMMAR;
    }
    num.grain = if n >= 100 {
        2
    } else if n >= 20 && n % 10 == 0 {
        1
    } else {
        0
    };
    Some(Value::Num(num))
}
fn sum(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (a, b) = (*v[0].num()?, *v[1].num()?);
    if a.has(flag::GRAMMAR)
        || a.grain <= b.grain
        || (a.grain >= 6 && en::proper_fraction(&v[1]))
        || a.value.is_negative()
        || b.value <= Dec::from_i64(0)
        || b.value >= Dec::from_i64(1).shift(a.grain as i32)?
    {
        return None;
    }
    let value = a.value.checked_add(b.value)?;
    Some(Value::Num(Num { value, kind: NumKind::of(value), grain: b.grain, flags: a.flags | b.flags }))
}
fn scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let mut n = *v[0].num()?;
    let exponent = v[1].term()?.0;
    if n.grain >= exponent as i8 || n.value.is_negative() {
        return None;
    }
    n.value = n.value.shift(exponent as i32)?;
    n.kind = NumKind::of(n.value);
    n.grain = exponent as i8;
    n.flags = (n.flags & !flag::GRAMMAR) | flag::SPELLED;
    Some(Value::Num(n))
}
// A proper fraction after a million-scale word belongs to that scale.
// Keep its grain so an explicit smaller scale attaches to the fraction
// first: "one billion and half a million" must not multiply the whole.
fn and_fraction(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let grain = v[0].num()?.grain;
    if grain >= 6 && en::proper_fraction(&v[1]) {
        let mut n = *en::b_scale_and_fraction(v, ctx)?.num()?;
        n.grain = grain;
        Some(Value::Num(n))
    } else {
        en::b_mixed(v, ctx)
    }
}
fn large(v: &Value) -> bool {
    v.num().is_some_and(|n| n.grain >= 3)
}
pub(super) const RULES: &[Rule] = &[
    r("pt_num_word", &[node(WORD)], word_number),
    r("pt_num_digits", &[node(NUMERAL)], en::b_same),
    r("pt_num_sum", &[node(NUM), word("e"), node(NUM)], sum),
    r("pt_num_scale_sum", &[node_if(NUM, large), node(NUM)], sum),
    r("pt_num_scale", &[node(NUM), node(SCALE)], scale),
    r("pt_num_thousand", &[word("mil")], |_, _| {
        Some(Value::Num(Num { value: Dec::from_i64(1000), kind: NumKind::Integer, grain: 3, flags: flag::SPELLED }))
    }),
    r("pt_num_half", &[word("meio")], |_, _| {
        Some(Value::Num(Num::new(Dec::new(5, 1)?, NumKind::Fraction, flag::SPELLED)))
    }),
    r("pt_num_halff", &[word("meia")], |_, _| {
        Some(Value::Num(Num::new(Dec::new(5, 1)?, NumKind::Fraction, flag::SPELLED)))
    }),
    r(
        "pt_num_fraction",
        &[node_if(NUMERAL, en::non_negative_integer), glued_word("/"), glued_if(NUMERAL, en::positive_integer)],
        en::b_digit_fraction,
    ),
    r(
        "pt_num_mixed",
        &[node_if(NUMERAL, en::non_negative_integer), node_if(NUM, en::digit_proper_fraction)],
        en::b_mixed,
    ),
    r(
        "pt_num_and_half",
        &[node_if(NUM, en::non_negative_integer), word("e"), node_if(NUM, en::positive_fraction)],
        and_fraction,
    ),
    r("pt_num_negative", &[word("menos"), node_if(NUM, en::non_negative)], |v, _| {
        let mut n = *v[0].num()?;
        n.value = n.value.negated();
        Some(Value::Num(n))
    }),
    r("pt_num_minus", &[word("-"), node_if(NUM, en::digits_non_negative)], en::b_neg_sign),
    r("pt_num_unicode_minus", &[word("\u{2212}"), node_if(NUM, en::digits_non_negative)], en::b_neg_sign),
];
