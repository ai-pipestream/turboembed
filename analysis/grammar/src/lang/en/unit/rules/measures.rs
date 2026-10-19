use super::*;

pub(super) fn b_unit_word(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
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
pub(super) fn b_prefixed(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
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

pub(super) fn the_length(v: &[Value]) -> Length {
    match v.iter().find_map(Value::unit) {
        Some(Unit::Length(l)) => l,
        _ => unreachable!("a length child"),
    }
}

pub(super) fn b_square(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Area(Area::Square(the_length(v)))))
}

pub(super) fn b_cubic(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Volume(Volume::Cubic(the_length(v)))))
}

pub(super) fn per(t: i64) -> Per {
    match Time::of(t) {
        Time::Millisecond => Per::Millisecond,
        Time::Second => Per::Second,
        Time::Minute => Per::Minute,
        _ => Per::Hour,
    }
}

pub(super) fn b_speed(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Speed(Speed::Per(the_length(v), per(term(&v[1]).0)))))
}

/// `miles an hour`.
pub(super) fn b_per_hour(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Speed(Speed::Per(the_length(v), Per::Hour))))
}

pub(super) fn b_metric(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Weight(Weight::MetricTon)))
}

pub(super) fn b_long_ton(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Weight(Weight::LongTon)))
}

pub(super) fn b_short_ton(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Weight(Weight::ShortTon)))
}

pub(super) fn b_nautical(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Length(Length::NauticalMile)))
}

/// `imperial gallons`, `UK pints`, `British tons`.
pub(super) fn b_imperial(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(match unit(&v[0]) {
        Unit::Volume(Volume::Gallon) => Unit::Volume(Volume::ImperialGallon),
        Unit::Volume(Volume::Quart) => Unit::Volume(Volume::ImperialQuart),
        Unit::Volume(Volume::Pint) => Unit::Volume(Volume::ImperialPint),
        Unit::Volume(Volume::FluidOunce) => Unit::Volume(Volume::ImperialFluidOunce),
        _ => Unit::Weight(Weight::LongTon),
    }))
}

/// `US gallons`, `US tons`: the unit an unqualified name means, with the
/// country written in capitals so `give us pints` is not read.
pub(super) fn b_us(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    (ctx.raw(ctx.start) == "US").then(|| Value::Unit(unit(&v[0])))
}

pub(super) fn b_fluid(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Volume(match unit(&v[0]) {
        Unit::Weight(Weight::Ounce) => Volume::FluidOunce,
        Unit::Weight(Weight::Dram) => Volume::FluidDram,
        _ => Volume::Pint,
    })))
}

pub(super) fn b_light_year(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Length(Length::LightYear)))
}

pub(super) fn b_degree_temperature(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Temperature(Temperature::Degree)))
}

pub(super) fn b_degree_angle(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Angle(Angle::Degree)))
}

/// `degrees celsius`, `°F`, `deg c`.
pub(super) fn b_degree_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Temperature(scale(term(&v[1]).0))))
}

pub(super) fn b_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Unit(Unit::Temperature(scale(term(&v[0]).0))))
}

/// `"` after a number is an inch outside a quotation: when the quotes
/// before it are paired, it cannot close one.
pub(super) fn b_inch_mark(_: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let at = ctx.lexeme(ctx.start).start as usize;
    let quotes = ctx.text.as_bytes()[..at].iter().filter(|&&b| b == b'"').count();
    (quotes % 2 == 0).then_some(Value::Unit(Unit::Length(Length::Inch)))
}
