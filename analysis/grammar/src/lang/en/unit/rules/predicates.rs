use super::*;

// Reading children; a rule's items fix each child's category.

pub(super) fn term(v: &Value) -> (i64, u8) {
    v.term().expect("a lexicon child")
}

pub(super) fn unit(v: &Value) -> Unit {
    v.unit().expect("a unit child")
}

pub(super) fn num(v: &Value) -> Num {
    *v.num().expect("a number child")
}

pub(super) fn qty(v: &Value) -> Quantity {
    *v.quantity().expect("a quantity child")
}

pub(super) fn quantity(value: Dec, unit: Unit) -> Option<Value> {
    Some(Value::Quantity(Quantity::new(value, unit)))
}

pub(super) fn pow10(e: i64) -> Option<Dec> {
    Dec::ONE.shift(i32::try_from(e).ok()?)
}

/// Lexeme `i` as written.
pub(super) fn raw<'a>(ctx: &Ctx<'a>, i: usize) -> &'a str {
    ctx.lexed.raw(ctx.text, i)
}

pub(super) fn has_upper(s: &str) -> bool {
    s.chars().any(char::is_uppercase)
}

pub(super) fn has_lower(s: &str) -> bool {
    s.chars().any(char::is_lowercase)
}

// Tests on children.

pub(super) fn is_unit(v: &Value, f: impl Fn(Unit) -> bool) -> bool {
    v.unit().is_some_and(f)
}

pub(super) fn length(v: &Value) -> bool {
    is_unit(v, |u| matches!(u, Unit::Length(_)))
}

pub(super) fn ton(v: &Value) -> bool {
    is_unit(v, |u| matches!(u, Unit::Weight(Weight::ShortTon | Weight::MetricTon)))
}

pub(super) fn mile(v: &Value) -> bool {
    is_unit(v, |u| u == Unit::Length(Length::Mile))
}

/// A US customary measure that has an imperial namesake: `gallon`,
/// `quart`, `pint`, `fluid ounce`, `ton`.
pub(super) fn customary(v: &Value) -> bool {
    is_unit(v, |u| {
        matches!(
            u,
            Unit::Volume(Volume::Gallon | Volume::Quart | Volume::Pint | Volume::FluidOunce)
                | Unit::Weight(Weight::ShortTon)
        )
    })
}

pub(super) fn fluid(v: &Value) -> bool {
    is_unit(v, |u| matches!(u, Unit::Weight(Weight::Ounce | Weight::Dram) | Unit::Volume(Volume::Pint)))
}

pub(super) fn inch(v: &Value) -> bool {
    is_unit(v, |u| u == Unit::Length(Length::Inch))
}

pub(super) fn dotted(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & DOT != 0)
}

pub(super) fn spelled_out(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & ABBR == 0)
}

pub(super) fn quote_mark(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| t == 1)
}

pub(super) fn double_prime(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| t == 2)
}

pub(super) fn inch_mark(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| t > 0)
}

/// A whole number of feet a person or a thing is tall: `6` in `6'2"`.
pub(super) fn feet_count(v: &Value) -> bool {
    v.num().is_some_and(|n| n.kind == NumKind::Integer && n.value > Dec::ZERO && n.value < Dec::from_i64(100))
}

/// A whole number of inches below a foot.
pub(super) fn inch_count(v: &Value) -> bool {
    v.num().is_some_and(|n| n.kind == NumKind::Integer && !n.value.is_negative() && n.value < Dec::from_i64(12))
}

pub(super) fn per_time(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| Time::of(t) <= Time::Hour)
}

pub(super) fn age_time(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| Time::of(t) >= Time::Day)
}

pub(super) fn is_year(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| Time::of(t) == Time::Year)
}

pub(super) fn prefix_money(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & PREFIX != 0 && f & NATION_ONLY == 0)
}

pub(super) fn prefix_code(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & AMBIG == 0)
}

pub(super) fn is_n(v: &Value, n: i64) -> bool {
    v.num().is_some_and(|x| x.has(flag::DIGITS) && x.value == Dec::from_i64(n))
}

pub(super) fn two(v: &Value) -> bool {
    is_n(v, 2)
}

pub(super) fn three(v: &Value) -> bool {
    is_n(v, 3)
}

pub(super) fn non_negative(v: &Value) -> bool {
    v.num().is_some_and(|n| !n.value.is_negative())
}

pub(super) fn positive(v: &Value) -> bool {
    v.num().is_some_and(|n| n.value > Dec::ZERO)
}

/// A whole number of hundredths or less: `50` in `15 dollars 50`.
pub(super) fn hundredths(v: &Value) -> bool {
    v.num().is_some_and(|n| n.kind == NumKind::Integer && !n.value.is_negative() && n.value < Dec::from_i64(100))
}

pub(super) fn feet(v: &Value) -> bool {
    v.quantity().is_some_and(|q| q.unit() == Unit::Length(Length::Foot))
}

/// Money in a major unit, a whole amount or a yuan amount with tenths
/// (`one kuai and five mao`), so a minor unit (`cents`, `fen`) can follow.
pub(super) fn whole_major(v: &Value) -> bool {
    v.quantity().is_some_and(|q| {
        !q.value().is_negative()
            && matches!(q.unit(), Unit::Currency(c) if c.has_hundredths())
            && (q.value().is_integer()
                || (q.unit() == Unit::Currency(CurrencyUnit::Code(code("cny").expect("cny")))
                    && q.value().shift(1).is_some_and(|x| x.is_integer())))
    })
}

/// A yuan amount with a non-zero tenths digit (`one kuai and five mao`),
/// before a bare fen digit (`one kuai and five mao five`).
pub(super) fn cny_tenths(v: &Value) -> bool {
    v.quantity().is_some_and(|q| {
        !q.value().is_negative()
            && !q.value().is_integer()
            && q.unit() == Unit::Currency(CurrencyUnit::Code(code("cny").expect("cny")))
            && q.value().shift(1).is_some_and(|x| x.is_integer())
    })
}

/// A single non-zero digit (`1..=9`), as the fen after `mao` (`five mao five`).
pub(super) fn single_digit(v: &Value) -> bool {
    v.num().is_some_and(|n| n.kind == NumKind::Integer && n.value > Dec::ZERO && n.value < Dec::from_i64(10))
}

pub(super) fn minor_unit(v: &Value) -> bool {
    is_unit(v, |u| matches!(u, Unit::Currency(CurrencyUnit::Minor(_))))
}

// Units.

pub(super) fn info_parts(i: Information) -> (u8, bool) {
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
