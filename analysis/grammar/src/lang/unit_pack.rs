//! What the Spanish, Italian, Dutch and Portuguese unit grammars share:
//! the dims of their unit rules, the word type of their lexicons, and the
//! rule bodies that read nothing of a language but its word and time
//! tables ([`UnitWords`]).

use crate::dec::Dec;
use crate::engine::{Ctx, Dim, Entry};
use crate::unit::{
    CurrencyUnit, Generic, Information, Length, Per, Rate, Speed, Temperature, Unit, Volume, Weight,
    currency::{CODES, code},
};
use crate::value::{Quantity, Value};

/// The number of dims of each number grammar these unit grammars extend;
/// each language module checks its own against it.
pub(crate) const NUMBER_DIMS: u16 = 25;
pub(crate) const RAW: Dim = Dim(NUMBER_DIMS);
pub(crate) const UNIT: Dim = Dim(NUMBER_DIMS + 1);
pub(crate) const QTY: Dim = Dim(NUMBER_DIMS + 2);
pub(crate) const CODE: Dim = Dim(NUMBER_DIMS + 3);
pub(crate) const SCALE_WORD: Dim = Dim(NUMBER_DIMS + 4);
pub(crate) const TIME: Dim = Dim(NUMBER_DIMS + 5);
pub(crate) const COUNTRY: Dim = Dim(NUMBER_DIMS + 6);
/// The dims of a unit grammar: its number grammar's and the seven above.
pub(crate) const UNIT_DIMS: u16 = NUMBER_DIMS + 7;
/// A unit word that stands alone as a quantity.
pub(crate) const BARE: u8 = 1;
/// A unit word that may come before its amount.
pub(crate) const PREFIX: u8 = 2;

/// A unit word of a language's lexicon.
pub(crate) struct Word {
    pub word: &'static str,
    pub unit: Unit,
    pub flags: u8,
}
pub(crate) const fn w(word: &'static str, unit: Unit, flags: u8) -> Word {
    Word { word, unit, flags }
}
pub(crate) const fn l(word: &'static str, unit: Length) -> Word {
    w(word, Unit::Length(unit), 0)
}
pub(crate) const fn v(word: &'static str, unit: Volume) -> Word {
    w(word, Unit::Volume(unit), 0)
}
pub(crate) const fn g(word: &'static str, unit: Weight) -> Word {
    w(word, Unit::Weight(unit), 0)
}
pub(crate) const fn t(word: &'static str, unit: Temperature, flags: u8) -> Word {
    w(word, Unit::Temperature(unit), flags)
}

/// A language's unit tables.
pub(crate) trait UnitWords {
    /// The unit words; a [`RAW`] entry's value is its index here.
    const WORDS: &'static [Word];
    /// The words for a unit of time after `per`; a [`TIME`] entry's value
    /// is its index here.
    const TIMES: &'static [(&'static str, Per)];
    /// The country words after a generic currency, with that currency and
    /// the code they make of it.
    const COUNTRIES: &'static [(&'static str, Generic, &'static str)];
}

/// The lexicon entries of `L`'s unit tables, the currency codes and the
/// scale letters `c` and `f`.
pub(crate) fn entries<L: UnitWords>() -> Vec<Entry> {
    let e = |word, dim, value, flags| Entry { word, dim, value, flags };
    let mut out: Vec<_> = L::WORDS.iter().enumerate().map(|(i, w)| e(w.word, RAW, i as i64, w.flags)).collect();
    out.extend(CODES.iter().enumerate().map(|(i, (c, _))| e(c, CODE, i as i64, PREFIX)));
    out.extend(L::TIMES.iter().enumerate().map(|(i, (w, _))| e(w, TIME, i as i64, 0)));
    out.extend(L::COUNTRIES.iter().enumerate().map(|(i, (w, _, _))| e(w, COUNTRY, i as i64, 0)));
    out.extend([("c", 0), ("f", 1)].map(|(w, i)| e(w, SCALE_WORD, i, 0)));
    out
}

/// The currency of the ISO code `s`.
pub(crate) fn code_unit(s: &str) -> Option<Unit> {
    Some(Unit::Currency(CurrencyUnit::Code(code(s)?)))
}

pub(crate) fn quantity(value: Dec, unit: Unit) -> Option<Value> {
    Some(Value::Quantity(Quantity::new(value, unit)))
}

/// A unit word, read as bytes when written with a capital `B` (`MB`), and
/// `o`, `oc`, `of` and `ok` only after `º`.
pub(crate) fn raw<L: UnitWords>(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (i, _) = v[0].term()?;
    let entry = L::WORDS.get(i as usize)?;
    if matches!(entry.word, "o" | "oc" | "of" | "ok") && !ctx.raw(ctx.start).starts_with('º') {
        return None;
    }
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

/// A currency code. Codes that are also common words are codes only in
/// capitals.
pub(crate) fn coded(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let code = CODES.get(v[0].term()?.0 as usize)?.0;
    if matches!(code, "all" | "cop" | "dem" | "esp" | "las" | "mad" | "pen" | "son" | "try")
        && ctx.raw(ctx.start).chars().any(turbo_unicode::props::lowercase)
    {
        return None;
    }
    Some(Value::Unit(code_unit(code)?))
}

/// A length per second, minute or hour, or an amount of information per
/// any unit of time.
pub(crate) fn speed<L: UnitWords>(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let per = L::TIMES.get(v[1].term()?.0 as usize)?.1;
    let u = match v[0].unit()? {
        Unit::Length(l) if matches!(per, Per::Second | Per::Minute | Per::Hour) => Unit::Speed(Speed::Per(l, per)),
        Unit::Information(i) => Unit::Rate(Rate::Information(i, per)),
        _ => return None,
    };
    Some(Value::Unit(u))
}

/// Feet and inches, or pounds and ounces, as one amount of the first.
pub(crate) fn compound(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (Value::Quantity(a), Value::Quantity(b)) = (v[0], v[1]) else { return None };
    let divisor = match (a.unit(), b.unit()) {
        (Unit::Length(Length::Foot), Unit::Length(Length::Inch)) => 12,
        (Unit::Weight(Weight::Pound), Unit::Weight(Weight::Ounce)) => 16,
        _ => return None,
    };
    if a.value().is_negative() || b.value().is_negative() || b.value() >= Dec::from_i64(divisor) {
        return None;
    }
    quantity(a.value().checked_add(b.value().checked_div(Dec::from_i64(divisor))?)?, a.unit())
}

/// A number and the unit after it; an age is never negative.
pub(crate) fn amount(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (value, unit) = (v[0].num()?.value, v[1].unit()?);
    if matches!(unit, Unit::Age(_)) && value.is_negative() {
        return None;
    }
    quantity(value, unit)
}

/// A generic currency and a country word of `L` that names it: the
/// country's own currency (`dólar` `canadiense`).
pub(crate) fn country<L: UnitWords>(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (_, generic, code) = L::COUNTRIES.get(v[1].term()?.0 as usize)?;
    (v[0].unit()? == Unit::Currency(CurrencyUnit::Generic(*generic)))
        .then(|| Value::Unit(code_unit(code).expect("currency code")))
}

/// A degree and the temperature scale after it.
pub(crate) fn scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let u = v[1].unit()?;
    matches!(u, Unit::Temperature(Temperature::Celsius | Temperature::Fahrenheit | Temperature::Kelvin))
        .then_some(Value::Unit(u))
}

/// A degree and the scale letter after it: `c` Celsius, `f` Fahrenheit.
pub(crate) fn letter_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let scale = if v[1].term()?.0 == 0 { Temperature::Celsius } else { Temperature::Fahrenheit };
    Some(Value::Unit(Unit::Temperature(scale)))
}

/// An amount of money per a unit of time of `L`.
pub(crate) fn money_rate<L: UnitWords>(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let Value::Quantity(q) = v[0] else { return None };
    let Unit::Currency(c) = q.unit() else { return None };
    quantity(q.value(), Unit::Rate(Rate::Currency(c, L::TIMES.get(v[1].term()?.0 as usize)?.1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Two;
    impl UnitWords for Two {
        const WORDS: &'static [Word] = &[l("km", Length::Kilometer), g("lb", Weight::Pound)];
        const TIMES: &'static [(&'static str, Per)] = &[("h", Per::Hour)];
        const COUNTRIES: &'static [(&'static str, Generic, &'static str)] = &[("x", Generic::Dollar, "usd")];
    }

    #[test]
    fn entries_index_each_table_by_its_own_dim() {
        let all = entries::<Two>();
        let of = |dim: Dim| all.iter().filter(|e| e.dim == dim).map(|e| (e.word, e.value)).collect::<Vec<_>>();
        assert_eq!(of(RAW), [("km", 0), ("lb", 1)]);
        assert_eq!(of(TIME), [("h", 0)]);
        assert_eq!(of(COUNTRY), [("x", 0)]);
        assert_eq!(of(SCALE_WORD), [("c", 0), ("f", 1)]);
        assert_eq!(of(CODE).len(), CODES.len());
        assert!(all.iter().filter(|e| e.dim == CODE).all(|e| e.flags == PREFIX));
        assert_eq!(all.len(), 2 + 1 + 1 + 2 + CODES.len());
    }

    #[test]
    fn the_unit_dims_follow_the_number_dims_without_a_gap() {
        let dims = [RAW, UNIT, QTY, CODE, SCALE_WORD, TIME, COUNTRY].map(|d| d.0);
        assert_eq!(dims, std::array::from_fn(|i| NUMBER_DIMS + i as u16));
        assert_eq!(UNIT_DIMS, NUMBER_DIMS + dims.len() as u16);
        assert_eq!(code_unit("usd"), Some(Unit::Currency(CurrencyUnit::Code(code("usd").unwrap()))));
        assert_eq!(code_unit("zzz"), None);
    }
}
