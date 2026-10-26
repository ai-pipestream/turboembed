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
    e("uno", 1),
    e("due", 2),
    e("tre", 3),
    e("quattro", 4),
    e("cinque", 5),
    e("sei", 6),
    e("sette", 7),
    e("otto", 8),
    e("nove", 9),
    e("dieci", 10),
    e("undici", 11),
    e("dodici", 12),
    e("tredici", 13),
    e("quattordici", 14),
    e("quindici", 15),
    e("sedici", 16),
    e("diciassette", 17),
    e("diciotto", 18),
    e("diciannove", 19),
    e("un", 1),
    e("una", 1),
    e("venti", 20),
    e("ventuno", 21),
    e("ventidue", 22),
    e("ventitre", 23),
    e("ventiquattro", 24),
    e("venticinque", 25),
    e("ventisei", 26),
    e("ventisette", 27),
    e("ventotto", 28),
    e("ventinove", 29),
    e("trenta", 30),
    e("trentuno", 31),
    e("trentadue", 32),
    e("trentatre", 33),
    e("trentaquattro", 34),
    e("trentacinque", 35),
    e("trentasei", 36),
    e("trentasette", 37),
    e("trentotto", 38),
    e("trentanove", 39),
    e("quaranta", 40),
    e("quarantuno", 41),
    e("quarantadue", 42),
    e("quarantatre", 43),
    e("quarantaquattro", 44),
    e("quarantacinque", 45),
    e("quarantasei", 46),
    e("quarantasette", 47),
    e("quarantotto", 48),
    e("quarantanove", 49),
    e("cinquanta", 50),
    e("cinquantuno", 51),
    e("cinquantadue", 52),
    e("cinquantatre", 53),
    e("cinquantaquattro", 54),
    e("cinquantacinque", 55),
    e("cinquantasei", 56),
    e("cinquantasette", 57),
    e("cinquantotto", 58),
    e("cinquantanove", 59),
    e("sessanta", 60),
    e("sessantuno", 61),
    e("sessantadue", 62),
    e("sessantatre", 63),
    e("sessantaquattro", 64),
    e("sessantacinque", 65),
    e("sessantasei", 66),
    e("sessantasette", 67),
    e("sessantotto", 68),
    e("sessantanove", 69),
    e("settanta", 70),
    e("settantuno", 71),
    e("settantadue", 72),
    e("settantatre", 73),
    e("settantaquattro", 74),
    e("settantacinque", 75),
    e("settantasei", 76),
    e("settantasette", 77),
    e("settantotto", 78),
    e("settantanove", 79),
    e("ottanta", 80),
    e("ottantuno", 81),
    e("ottantadue", 82),
    e("ottantatre", 83),
    e("ottantaquattro", 84),
    e("ottantacinque", 85),
    e("ottantasei", 86),
    e("ottantasette", 87),
    e("ottantotto", 88),
    e("ottantanove", 89),
    e("novanta", 90),
    e("novantuno", 91),
    e("novantadue", 92),
    e("novantatre", 93),
    e("novantaquattro", 94),
    e("novantacinque", 95),
    e("novantasei", 96),
    e("novantasette", 97),
    e("novantotto", 98),
    e("novantanove", 99),
    e("cento", 100),
    e("cent", 100),
    e("duecento", 200),
    e("duecent", 200),
    e("trecento", 300),
    e("trecent", 300),
    e("quattrocento", 400),
    e("quattrocent", 400),
    e("cinquecento", 500),
    e("cinquecent", 500),
    e("seicento", 600),
    e("seicent", 600),
    e("settecento", 700),
    e("settecent", 700),
    e("ottocento", 800),
    e("ottocent", 800),
    e("novecento", 900),
    e("novecent", 900),
    s("mille", 3),
    s("mila", 3),
    s("milione", 6),
    s("milioni", 6),
    s("miliardo", 9),
    s("miliardi", 9),
    s("bilione", 12),
    s("bilioni", 12),
    s("trilione", 18),
    s("trilioni", 18),
];
const fn r(name: &'static str, items: &'static [Item], build: Build) -> Rule {
    Rule { name, dim: NUM, items, weight: 0, build }
}
fn word_number(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if ctx.word(ctx.start).ends_with("cent")
        && !ctx.after().is_some_and(|i| {
            ctx.lexeme(i).glued
                && matches!(
                    ctx.word(i),
                    "otto"
                        | "ottanta"
                        | "ottantuno"
                        | "ottantadue"
                        | "ottantatre"
                        | "ottantaquattro"
                        | "ottantacinque"
                        | "ottantasei"
                        | "ottantasette"
                        | "ottantotto"
                        | "ottantanove"
                )
        })
    {
        return None;
    }
    let n = v[0].term()?.0;
    let mut num = Num::new(Dec::from_i64(n), NumKind::Integer, flag::SPELLED);
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
    if a.grain <= b.grain
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
fn large(v: &Value) -> bool {
    v.num().is_some_and(|n| n.grain >= 2)
}
pub(super) const RULES: &[Rule] = &[
    r("it_num_word", &[node(WORD)], word_number),
    r("it_num_digits", &[node(NUMERAL)], en::b_same),
    r("it_num_sum", &[node(NUM), word("e"), node(NUM)], sum),
    r("it_num_scale_sum", &[node_if(NUM, large), node(NUM)], sum),
    r("it_num_scale", &[node(NUM), node(SCALE)], scale),
    r("it_num_thousand", &[word("mille")], |_, _| {
        Some(Value::Num(Num { value: Dec::from_i64(1000), kind: NumKind::Integer, grain: 3, flags: flag::SPELLED }))
    }),
    r("it_num_half", &[word("mezzo")], |_, _| {
        Some(Value::Num(Num::new(Dec::new(5, 1)?, NumKind::Fraction, flag::SPELLED)))
    }),
    r("it_num_halff", &[word("mezza")], |_, _| {
        Some(Value::Num(Num::new(Dec::new(5, 1)?, NumKind::Fraction, flag::SPELLED)))
    }),
    r(
        "it_num_fraction",
        &[node_if(NUMERAL, en::non_negative_integer), glued_word("/"), glued_if(NUMERAL, en::positive_integer)],
        en::b_digit_fraction,
    ),
    r(
        "it_num_mixed",
        &[node_if(NUMERAL, en::non_negative_integer), node_if(NUM, en::digit_proper_fraction)],
        en::b_mixed,
    ),
    r(
        "it_num_and_half",
        &[node_if(NUM, en::non_negative_integer), word("e"), node_if(NUM, en::positive_fraction)],
        en::b_mixed,
    ),
    r("it_num_negative", &[word("meno"), node_if(NUM, en::non_negative)], |v, _| {
        let mut n = *v[0].num()?;
        n.value = n.value.negated();
        Some(Value::Num(n))
    }),
    r("it_num_minus", &[word("-"), node_if(NUM, en::digits_non_negative)], en::b_neg_sign),
    r("it_num_unicode_minus", &[word("\u{2212}"), node_if(NUM, en::digits_non_negative)], en::b_neg_sign),
];
