//! The English unit rules and the functions that build their values.
//!
//! Children arrive in item order with word items left out.

use crate::dec::Dec;
use crate::engine::{Build, Ctx, Dim, Item, Kind, Rule, glued, glued_if, glued_word, node, node_if, opt_word, word};
use crate::lang::en::{NUM, NUMERAL, SCALE};
use crate::unit::currency::{CODES, code};
use crate::unit::{
    Age, Angle, Area, CurrencyUnit, Generic, Information, Length, Minor, Per, Speed, Temperature, Unit, Volume, Weight,
};
use crate::value::{Num, NumKind, Quantity, Value, flag};

use super::lexicon::{MONEY as MONEY_WORDS, UNITS, nation, nation_rows, scale};
use super::{
    ABBR, AGE, AMBIG, BITCASE, CODE, CUR, CURP, CURW, DEG, DOT, INCHW, LOWER, MCODE, MONEY, MULT, NATION, NATION_ONLY,
    ONLY_DOLLAR, PREFIX, QTY, SIPRE, TIMEW, TSCALE, Time, UNIT, UWORD,
};

// Reading children; a rule's items fix each child's category.

fn term(v: &Value) -> (i64, u8) {
    v.term().expect("a lexicon child")
}

fn unit(v: &Value) -> Unit {
    v.unit().expect("a unit child")
}

fn num(v: &Value) -> Num {
    *v.num().expect("a number child")
}

fn qty(v: &Value) -> Quantity {
    *v.quantity().expect("a quantity child")
}

fn quantity(value: Dec, unit: Unit) -> Option<Value> {
    Some(Value::Quantity(Quantity::new(value, unit)))
}

fn pow10(e: i64) -> Option<Dec> {
    Dec::ONE.shift(i32::try_from(e).ok()?)
}

/// Lexeme `i` as written.
fn raw<'a>(ctx: &Ctx<'a>, i: usize) -> &'a str {
    ctx.lexed.raw(ctx.text, i)
}

fn has_upper(s: &str) -> bool {
    s.chars().any(char::is_uppercase)
}

fn has_lower(s: &str) -> bool {
    s.chars().any(char::is_lowercase)
}

// Tests on children.

fn is_unit(v: &Value, f: impl Fn(Unit) -> bool) -> bool {
    v.unit().is_some_and(f)
}

fn length(v: &Value) -> bool {
    is_unit(v, |u| matches!(u, Unit::Length(_)))
}

fn ton(v: &Value) -> bool {
    is_unit(v, |u| u == Unit::Weight(Weight::Ton))
}

fn fluid(v: &Value) -> bool {
    is_unit(v, |u| matches!(u, Unit::Weight(Weight::Ounce | Weight::Dram) | Unit::Volume(Volume::Pint)))
}

fn inch(v: &Value) -> bool {
    is_unit(v, |u| u == Unit::Length(Length::Inch))
}

fn dotted(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & DOT != 0)
}

fn spelled_out(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & ABBR == 0)
}

fn quote_mark(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| t == 1)
}

fn double_prime(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| t == 2)
}

fn inch_mark(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| t > 0)
}

/// A whole number of feet a person or a thing is tall: `6` in `6'2"`.
fn feet_count(v: &Value) -> bool {
    v.num().is_some_and(|n| n.kind == NumKind::Integer && n.value > Dec::ZERO && n.value < Dec::from_i64(100))
}

/// A whole number of inches below a foot.
fn inch_count(v: &Value) -> bool {
    v.num().is_some_and(|n| n.kind == NumKind::Integer && !n.value.is_negative() && n.value < Dec::from_i64(12))
}

fn per_time(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| Time::of(t) <= Time::Hour)
}

fn age_time(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| Time::of(t) >= Time::Day)
}

fn is_year(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| Time::of(t) == Time::Year)
}

fn prefix_money(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & PREFIX != 0 && f & NATION_ONLY == 0)
}

fn prefix_code(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & AMBIG == 0)
}

fn is_n(v: &Value, n: i64) -> bool {
    v.num().is_some_and(|x| x.has(flag::DIGITS) && x.value == Dec::from_i64(n))
}

fn two(v: &Value) -> bool {
    is_n(v, 2)
}

fn three(v: &Value) -> bool {
    is_n(v, 3)
}

fn non_negative(v: &Value) -> bool {
    v.num().is_some_and(|n| !n.value.is_negative())
}

fn positive(v: &Value) -> bool {
    v.num().is_some_and(|n| n.value > Dec::ZERO)
}

/// A whole number of hundredths or less: `50` in `15 dollars 50`.
fn hundredths(v: &Value) -> bool {
    v.num().is_some_and(|n| n.kind == NumKind::Integer && !n.value.is_negative() && n.value < Dec::from_i64(100))
}

fn feet(v: &Value) -> bool {
    v.quantity().is_some_and(|q| q.unit() == Unit::Length(Length::Foot))
}

/// Money in a major unit, a whole amount: `15 dollars`, not `$4.25`.
fn whole_major(v: &Value) -> bool {
    v.quantity().is_some_and(|q| {
        q.value().is_integer()
            && !q.value().is_negative()
            && matches!(q.unit(), Unit::Currency(c) if c.has_hundredths())
    })
}

fn minor_unit(v: &Value) -> bool {
    is_unit(v, |u| matches!(u, Unit::Currency(CurrencyUnit::Minor(_))))
}

// Units.

fn info_parts(i: Information) -> (u8, bool) {
    use Information::*;
    match i {
        Bit => (0, false),
        Kilobit => (3, false),
        Megabit => (6, false),
        Gigabit => (9, false),
        Terabit => (12, false),
        Petabit => (15, false),
        Byte => (0, true),
        Kilobyte => (3, true),
        Megabyte => (6, true),
        Gigabyte => (9, true),
        Terabyte => (12, true),
        Petabyte => (15, true),
    }
}

fn b_unit_word(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (i, f) = term(&v[0]);
    let written = raw(ctx, ctx.start);
    if f & LOWER != 0 && has_upper(written) {
        return None;
    }
    let mut u = UNITS[i as usize].unit;
    if f & BITCASE != 0
        && let Unit::Information(x) = u
    {
        u = Unit::Information(Information::of(info_parts(x).0, written.ends_with('B'))?);
    }
    Some(Value::Unit(u))
}

/// `kilo meters`, `mega bytes`, `micro grams`.
fn b_prefixed(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let p = term(&v[0]).0;
    let u = match UNITS[term(&v[1]).0 as usize].unit {
        Unit::Length(Length::Meter) => Unit::Length(match p {
            3 => Length::Kilometer,
            2 => Length::Hectometer,
            1 => Length::Decameter,
            -1 => Length::Decimeter,
            -2 => Length::Centimeter,
            -3 => Length::Millimeter,
            -6 => Length::Micrometer,
            -9 => Length::Nanometer,
            -12 => Length::Picometer,
            _ => return None,
        }),
        Unit::Information(x) if info_parts(x).0 == 0 => {
            Unit::Information(Information::of(u8::try_from(p).ok()?, info_parts(x).1)?)
        }
        Unit::Weight(Weight::Gram) => Unit::Weight(match p {
            3 => Weight::Kilogram,
            -3 => Weight::Milligram,
            -6 => Weight::Microgram,
            _ => return None,
        }),
        Unit::Volume(Volume::Liter) => Unit::Volume(match p {
            2 => Volume::Hectoliter,
            1 => Volume::Decaliter,
            -1 => Volume::Deciliter,
            -2 => Volume::Centiliter,
            -3 => Volume::Milliliter,
            _ => return None,
        }),
        _ => return None,
    };
    Some(Value::Unit(u))
}

fn the_length(v: &[Value]) -> Length {
    match v.iter().find_map(Value::unit) {
        Some(Unit::Length(l)) => l,
        _ => unreachable!("a length child"),
    }
}

fn b_square(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Area(Area::Square(the_length(v)))))
}

fn b_cubic(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Volume(Volume::Cubic(the_length(v)))))
}

fn per(t: i64) -> Per {
    match Time::of(t) {
        Time::Millisecond => Per::Millisecond,
        Time::Second => Per::Second,
        Time::Minute => Per::Minute,
        _ => Per::Hour,
    }
}

fn b_speed(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Speed(Speed::Per(the_length(v), per(term(&v[1]).0)))))
}

/// `miles an hour`.
fn b_per_hour(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Speed(Speed::Per(the_length(v), Per::Hour))))
}

fn b_metric(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Weight(Weight::MetricTon)))
}

fn b_long_ton(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Weight(Weight::LongTon)))
}

fn b_short_ton(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Weight(Weight::ShortTon)))
}

fn b_fluid(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Volume(match unit(&v[0]) {
        Unit::Weight(Weight::Ounce) => Volume::FluidOunce,
        Unit::Weight(Weight::Dram) => Volume::FluidDram,
        _ => Volume::Pint,
    })))
}

fn b_light_year(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Length(Length::LightYear)))
}

fn b_degree_temperature(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Temperature(Temperature::Degree)))
}

fn b_degree_angle(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Angle(Angle::Degree)))
}

/// `degrees celsius`, `°F`, `deg c`.
fn b_degree_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Temperature(scale(term(&v[1]).0))))
}

fn b_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Temperature(scale(term(&v[0]).0))))
}

/// `"` after a number is an inch outside a quotation: when the quotes
/// before it are paired, it cannot close one.
fn b_inch_mark(_: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let at = ctx.lexeme(ctx.start).start as usize;
    let quotes = ctx.text.as_bytes()[..at].iter().filter(|&&b| b == b'"').count();
    (quotes % 2 == 0).then_some(Value::Unit(Unit::Length(Length::Inch)))
}

// Currencies.

fn currency(c: CurrencyUnit) -> Option<Value> {
    Some(Value::Unit(Unit::Currency(c)))
}

fn coded(lower: &str) -> Option<Value> {
    currency(CurrencyUnit::Code(code(lower)?))
}

fn b_money_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (i, f) = term(&v[0]);
    if f & NATION_ONLY != 0 {
        return None;
    }
    Some(Value::Unit(MONEY_WORDS[i as usize].unit))
}

/// A code; one that is also an English word only in capitals.
fn b_code(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (i, f) = term(&v[0]);
    if f & AMBIG != 0 && has_lower(raw(ctx, ctx.start)) {
        return None;
    }
    coded(CODES[i as usize].0)
}

/// A country word followed by the name of its currency.
fn b_nation_money(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let rows = nation_rows(term(&v[0]).0 as usize);
    let named = MONEY_WORDS[term(&v[1]).0 as usize].unit;
    rows.iter().filter(|r| r.flags & ONLY_DOLLAR == 0).find_map(|r| {
        let c = code(r.code)?;
        let fits = match (r.generic, named) {
            (Some(g), Unit::Currency(CurrencyUnit::Generic(h))) => g == h,
            (None, Unit::Currency(CurrencyUnit::Code(x))) => x == c,
            _ => false,
        };
        fits.then_some(Value::Unit(Unit::Currency(CurrencyUnit::Code(c))))
    })
}

/// Words before which a lowercase `us` is the country: `a us $ 150
/// million`, not `give us $5`.
const US_CUES: &[&str] = &["a", "an", "the", "s", "its", "his", "her", "their", "our", "your", "my"];

/// A country before `$`: `C$`, `hk $`, `australian $`.
fn b_nation_dollar(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let rows = nation_rows(term(&v[0]).0 as usize);
    let r = rows.iter().find(|r| r.generic == Some(Generic::Dollar))?;
    let first = ctx.lexeme(ctx.start);
    let short = r.flags & ONLY_DOLLAR != 0 || r.word == "us";
    // A short form glued to a word before it, or after an apostrophe, is
    // no country: `it's $5`, `music ' s $ 500 million man`.
    let after_apostrophe = ctx.before().is_some_and(|p| matches!(ctx.punct(p), Some('\'' | '’')));
    if short && (after_apostrophe || first.glued && ctx.before().is_some_and(|p| ctx.lexeme(p).kind != Kind::Punct)) {
        return None;
    }
    if r.word == "us"
        && has_lower(raw(ctx, ctx.start))
        && ctx.before().is_some_and(|p| ctx.lexeme(p).kind == Kind::Word && !US_CUES.contains(&ctx.word(p)))
    {
        return None;
    }
    coded(r.code)
}

fn nation_term(word: &str) -> Option<Value> {
    Some(Value::Term { value: nation(word)? as i64, flags: 0 })
}

/// A code glued to `$` is a currency in any case: `cad$ 15`.
fn b_code_dollar(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    coded(CODES[term(&v[0]).0 as usize].0)
}

// Quantities.

fn b_amount_unit(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value, unit(&v[1]))
}

fn b_unit_amount(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[1]).value, unit(&v[0]))
}

/// `$3.5m`, `€2bn`.
fn b_unit_amount_multiplier(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[1]).value.checked_mul(pow10(term(&v[2]).0)?)?, unit(&v[0]))
}

/// `75 USD million`.
fn b_amount_unit_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value.checked_mul(pow10(term(&v[2]).0)?)?, unit(&v[1]))
}

fn million_code(v: &Value) -> Option<Unit> {
    Some(Unit::Currency(CurrencyUnit::Code(code(CODES[term(v).0 as usize].0)?)))
}

/// `30 MUSD`.
fn b_amount_million_code(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value.checked_mul(pow10(6)?)?, million_code(&v[1])?)
}

/// `MAUD 60`.
fn b_million_code_amount(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[1]).value.checked_mul(pow10(6)?)?, million_code(&v[0])?)
}

/// `13 dollars and 45 cents`, `10 $ 30 cent`, `3 pounds 20 pence`.
fn b_major_minor(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
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
fn b_major_hundredths(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (major, n) = (qty(&v[0]), num(&v[1]));
    let ends = match ctx.after() {
        None => true,
        Some(a) => match ctx.lexeme(a).kind {
            Kind::Punct => ctx.punct(a).is_some_and(|c| {
                turbo_unicode::category::general_category(c) != turbo_unicode::category::GeneralCategory::Sc
            }),
            Kind::Word => matches!(ctx.word(a), "and" | "or" | "but" | "more" | "less" | "extra" | "each"),
            Kind::Numeral => false,
        },
    };
    if !ends {
        return None;
    }
    quantity(major.value().checked_add(n.value.shift(-2)?)?, major.unit())
}

/// `5 ft 2 in`, `5 feet and 2 inches`.
fn b_feet_inches(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (f, n) = (qty(&v[0]), num(&v[1]));
    if n.value >= Dec::from_i64(12) {
        return None;
    }
    quantity(f.value().checked_add(n.value.checked_div(Dec::from_i64(12))?)?, f.unit())
}

/// `6'2"`, `6′2″`: feet and inches with marks.
fn b_feet_inch_marks(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (f, n) = (num(&v[0]), num(&v[1]));
    quantity(f.value.checked_add(n.value.checked_div(Dec::from_i64(12))?)?, Unit::Length(Length::Foot))
}

fn b_feet_mark(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value, Unit::Length(Length::Foot))
}

/// `2 m dollars`, `4.5 bn euros`.
fn b_amount_multiplier_unit(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value.checked_mul(pow10(term(&v[1]).0)?)?, unit(&v[2]))
}

fn b_age(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let age = match Time::of(term(&v[1]).0) {
        Time::Day => Age::Day,
        Time::Week => Age::Week,
        Time::Month => Age::Month,
        _ => Age::Year,
    };
    quantity(num(&v[0]).value, Unit::Age(age))
}

fn b_age_years(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    quantity(num(&v[0]).value, Unit::Age(Age::Year))
}

/// `age 87`, `aged 87`, in lowercase or at the start of a sentence.
fn b_age_stated(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let opens = ctx.before().is_none_or(|p| ctx.lexeme(p).kind == Kind::Punct);
    if has_upper(raw(ctx, ctx.start)) && !opens {
        return None;
    }
    quantity(num(&v[0]).value, Unit::Age(Age::Unstated))
}

const fn r(name: &'static str, dim: Dim, weight: i16, items: &'static [Item], build: Build) -> Rule {
    Rule { name, dim, items, weight, build }
}

/// An optional word glued to what comes before.
const fn opt_glued_word(text: &'static str) -> Item {
    Item::Word { text, glued: true, optional: true, exact_case: false }
}

const OPT_DOT: Item = opt_glued_word(".");

/// Every unit rule, after the number grammar's.
pub static RULES: &[Rule] = &[
    // Units of measure.
    r("unit_word", UNIT, 0, &[node(UWORD)], b_unit_word),
    r("unit_word_dot", UNIT, 0, &[node_if(UWORD, dotted), glued_word(".")], b_unit_word),
    r("unit_prefixed", UNIT, 0, &[node(SIPRE), node_if(UWORD, spelled_out)], b_prefixed),
    r("unit_square", UNIT, 0, &[word("square"), node_if(UNIT, length)], b_square),
    r("unit_sq", UNIT, 0, &[word("sq"), OPT_DOT, node_if(UNIT, length)], b_square),
    r("unit_squared", UNIT, 0, &[node_if(UNIT, length), glued_if(NUMERAL, two)], b_square),
    r("unit_squared_sign", UNIT, 0, &[node_if(UNIT, length), glued_word("²")], b_square),
    r("unit_squared_caret", UNIT, 0, &[node_if(UNIT, length), glued_word("^"), glued_if(NUMERAL, two)], b_square),
    r("unit_cubic", UNIT, 0, &[word("cubic"), node_if(UNIT, length)], b_cubic),
    r("unit_cubed", UNIT, 0, &[node_if(UNIT, length), glued_if(NUMERAL, three)], b_cubic),
    r("unit_cubed_sign", UNIT, 0, &[node_if(UNIT, length), glued_word("³")], b_cubic),
    r("unit_per", UNIT, 0, &[node_if(UNIT, length), word("per"), node_if(TIMEW, per_time)], b_speed),
    r("unit_slash", UNIT, 0, &[node_if(UNIT, length), word("/"), node_if(TIMEW, per_time)], b_speed),
    r("unit_an_hour", UNIT, 0, &[node_if(UNIT, length), word("an"), word("hour")], b_per_hour),
    r("unit_a_hour", UNIT, 0, &[node_if(UNIT, length), word("a"), word("hour")], b_per_hour),
    r("unit_metric_ton", UNIT, 0, &[word("metric"), node_if(UNIT, ton)], b_metric),
    r("unit_long_ton", UNIT, 0, &[word("long"), node_if(UNIT, ton)], b_long_ton),
    r("unit_short_ton", UNIT, 0, &[word("short"), node_if(UNIT, ton)], b_short_ton),
    r("unit_fluid", UNIT, 0, &[word("fluid"), node_if(UNIT, fluid)], b_fluid),
    r("unit_fl", UNIT, 0, &[word("fl"), OPT_DOT, node_if(UNIT, fluid)], b_fluid),
    r("unit_light_year", UNIT, 0, &[word("light"), node_if(TIMEW, is_year)], b_light_year),
    r("unit_light_year_hyphen", UNIT, 0, &[word("light"), glued_word("-"), glued_if(TIMEW, is_year)], b_light_year),
    r("unit_degree_temperature", UNIT, 0, &[node(DEG)], b_degree_temperature),
    r("unit_degree_angle", UNIT, 0, &[node(DEG)], b_degree_angle),
    r("unit_degree_dot_temperature", UNIT, 0, &[node_if(DEG, dotted), glued_word(".")], b_degree_temperature),
    r("unit_degree_dot_angle", UNIT, 0, &[node_if(DEG, dotted), glued_word(".")], b_degree_angle),
    r("unit_degree_scale", UNIT, 0, &[node(DEG), node(TSCALE)], b_degree_scale),
    r("unit_scale", UNIT, 0, &[node_if(TSCALE, spelled_out)], b_scale),
    r("unit_inch_mark", UNIT, 0, &[node_if(INCHW, quote_mark)], b_inch_mark),
    r("unit_double_prime", UNIT, 0, &[node_if(INCHW, double_prime)], |_, _| {
        Some(Value::Unit(Unit::Length(Length::Inch)))
    }),
    // Currencies.
    r("cur_word", CUR, 0, &[node(CURW)], b_money_word),
    r("cur_sign", CURP, 0, &[node_if(CURW, prefix_money)], b_money_word),
    r("cur_code", CUR, 0, &[node(CODE)], b_code),
    r("cur_code_before", CURP, 0, &[node_if(CODE, prefix_code)], b_code),
    r("cur_code_dollar", CURP, 0, &[node(CODE), glued_word("$")], b_code_dollar),
    r("cur_nation", CUR, 0, &[node(NATION), node(CURW)], b_nation_money),
    r("cur_nation_dollar", CURP, 0, &[node(NATION), word("$")], b_nation_dollar),
    r("cur_before", CUR, 0, &[node(CURP)], |v, _| Some(v[0])),
    r("nation_hong_kong", NATION, 0, &[word("hong"), word("kong")], |_, _| nation_term("hong kong")),
    r("nation_new_zealand", NATION, 0, &[word("new"), word("zealand")], |_, _| nation_term("new zealand")),
    r("nation_new_taiwan", NATION, 0, &[word("new"), word("taiwan")], |_, _| nation_term("new taiwan")),
    r("nation_united_states", NATION, 0, &[word("united"), word("states")], |_, _| nation_term("united states")),
    r("nation_us_dots", NATION, 0, &[word("u"), glued_word("."), glued_word("s"), OPT_DOT], |_, _| nation_term("u.s.")),
    r("nation_south_african", NATION, 0, &[word("south"), word("african")], |_, _| nation_term("south african")),
    r("nation_south_korean", NATION, 0, &[word("south"), word("korean")], |_, _| nation_term("south korean")),
    r("nation_north_korean", NATION, 0, &[word("north"), word("korean")], |_, _| nation_term("north korean")),
    r("nation_sri_lankan", NATION, 0, &[word("sri"), word("lankan")], |_, _| nation_term("sri lankan")),
    r("nation_trinidad", NATION, 0, &[word("trinidad"), word("and"), word("tobago")], |_, _| {
        nation_term("trinidad and tobago")
    }),
    r("nation_cayman", NATION, 0, &[word("cayman"), word("islands")], |_, _| nation_term("cayman islands")),
    r("nation_solomon", NATION, 0, &[word("solomon"), word("islands")], |_, _| nation_term("solomon islands")),
    // Measured quantities.
    r("qty", QTY, 0, &[node(NUM), node(UNIT)], b_amount_unit),
    r("qty_hyphen", QTY, 0, &[node(NUM), glued_word("-"), glued(UNIT)], b_amount_unit),
    r("qty_parens", QTY, 0, &[node(NUM), word("("), glued(UNIT), glued_word(")")], b_amount_unit),
    r("qty_brackets", QTY, 0, &[node(NUM), word("["), glued(UNIT), glued_word("]")], b_amount_unit),
    r("qty_braces", QTY, 0, &[node(NUM), word("{"), glued(UNIT), glued_word("}")], b_amount_unit),
    r("qty_angles", QTY, 0, &[node(NUM), word("<"), glued(UNIT), glued_word(">")], b_amount_unit),
    r(
        "qty_feet_inches",
        QTY,
        1,
        &[node_if(QTY, feet), opt_word("and"), node_if(NUM, non_negative), node(INCHW)],
        b_feet_inches,
    ),
    r(
        "qty_feet_inch_unit",
        QTY,
        1,
        &[node_if(QTY, feet), opt_word("and"), node_if(NUM, non_negative), node_if(UNIT, inch)],
        b_feet_inches,
    ),
    r(
        "qty_feet_inch_marks",
        QTY,
        1,
        &[node_if(NUMERAL, feet_count), glued_word("'"), glued_if(NUMERAL, inch_count), glued_if(INCHW, inch_mark)],
        b_feet_inch_marks,
    ),
    r(
        "qty_feet_inch_primes",
        QTY,
        1,
        &[node_if(NUMERAL, feet_count), glued_word("′"), glued_if(NUMERAL, inch_count), opt_glued_word("″")],
        b_feet_inch_marks,
    ),
    r(
        "qty_feet_apostrophe_inches",
        QTY,
        0,
        &[node_if(NUMERAL, feet_count), glued_word("'"), glued_if(NUMERAL, inch_count)],
        b_feet_inch_marks,
    ),
    r("qty_feet_prime", QTY, 0, &[node(NUM), glued_word("′")], b_feet_mark),
    // Money.
    r("money_after", MONEY, 0, &[node(NUM), node(CUR)], b_amount_unit),
    r("money_hyphen", MONEY, 0, &[node(NUM), glued_word("-"), glued(CUR)], b_amount_unit),
    r("money_before", MONEY, 0, &[node(CURP), node(NUM)], b_unit_amount),
    r("money_multiplier", MONEY, 0, &[node(CURP), node(NUMERAL), glued(MULT)], b_unit_amount_multiplier),
    r("money_multiplier_after", MONEY, 0, &[node(NUMERAL), node(MULT), node(CUR)], b_amount_multiplier_unit),
    r("money_scale_after", MONEY, 0, &[node(NUM), node(CUR), node(SCALE)], b_amount_unit_scale),
    r("money_million_code", MONEY, 0, &[node(NUM), node(MCODE)], b_amount_million_code),
    r("money_million_code_before", MONEY, 0, &[node(MCODE), node(NUM)], b_million_code_amount),
    r(
        "money_major_minor",
        MONEY,
        1,
        &[node_if(MONEY, whole_major), opt_word("and"), node(NUM), node_if(CUR, minor_unit)],
        b_major_minor,
    ),
    r(
        "money_major_hundredths",
        MONEY,
        0,
        &[node_if(MONEY, whole_major), opt_word("and"), node_if(NUM, hundredths)],
        b_major_hundredths,
    ),
    // Ages.
    r("age_old", AGE, 0, &[node_if(NUM, positive), node_if(TIMEW, age_time), word("old")], b_age),
    r("age_of_age", AGE, 0, &[node_if(NUM, positive), node_if(TIMEW, age_time), word("of"), word("age")], b_age),
    r(
        "age_hyphens",
        AGE,
        0,
        &[node_if(NUM, positive), glued_word("-"), glued_if(TIMEW, age_time), glued_word("-"), glued_word("old")],
        b_age,
    ),
    r("age_yo", AGE, 0, &[node_if(NUM, positive), word("yo")], b_age_years),
    r("age_stated", AGE, 0, &[word("age"), node_if(NUM, positive)], b_age_stated),
    r("age_aged", AGE, 0, &[word("aged"), node_if(NUM, positive)], b_age_stated),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_names_are_unique() {
        let mut names: Vec<&str> = RULES.iter().map(|r| r.name).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n);
    }
}
