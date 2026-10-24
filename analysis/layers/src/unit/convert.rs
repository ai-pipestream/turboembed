//! Shared unit symbol table and exact conversion factors to SI / stated base
//! units.
//!
//! Base units per dimension / measure:
//! - [`Length`] -> [`Length::Meter`] (`m`)
//! - [`Area`] -> `Area::Square(Length::Meter)` (`m²`)
//! - [`Volume`] -> `Volume::Cubic(Length::Meter)` (`m³`)
//! - [`Weight`] -> [`Weight::Kilogram`] (`kg`)
//! - [`Speed`] -> `Speed::Per(Length::Meter, Per::Second)` (`m/s`)
//! - [`Information`] -> [`Information::Byte`] (`B`)
//! - [`Angle`] -> [`Angle::Radian`] (`rad`)
//! - [`DurationUnit`] -> [`DurationUnit::Second`] (`s`)
//! - [`Frequency`] -> [`Frequency::Hertz`] (`Hz`)
//! - [`Power`] -> [`Power::Watt`] (`W`)
//! - [`Energy`] -> [`Energy::Joule`] (`J`)
//! - [`Temperature`] -> [`Temperature::Kelvin`] (`K`), affine conversion
//! - [`Age`] -> [`Age::Year`] (`years old`)

use crate::dec::Dec;

use super::{
    Age, Angle, Area, DurationUnit, Energy, Frequency, Information, Length, Per, Power, Speed, Temperature, Unit,
    Volume, Weight,
};

/// The units OpenNLP's quantity extractor knows unless a caller gives its own
/// set: common measurement tokens from the shared [`Unit`] table, matched
/// exactly and case-sensitively. Ambiguous English words such as `in` are left
/// out on purpose.
pub const DEFAULT_UNITS: [&str; 37] = [
    "km", "m", "cm", "mm", "nm", "mi", "ft", "yd", //
    "kg", "g", "mg", "t", "lb", "lbs", "oz", //
    "L", "mL", "ml", //
    "ms", "ns", "min", "hr", //
    "KB", "MB", "GB", "TB", "PB", //
    "Hz", "kHz", "MHz", "GHz", //
    "kW", "MW", "GW", "kWh", //
    "mph", "kph",
];

fn dec(mantissa: i128, scale: i32) -> Dec {
    Dec::new(mantissa, scale).expect("a valid unit conversion constant")
}

impl Length {
    /// SI base unit of length: the meter (`m`).
    pub const BASE: Length = Length::Meter;

    /// Exact conversion factor from `self` to [`Length::Meter`].
    pub fn factor(self) -> Dec {
        match self {
            Length::Kilometer => dec(1_000, 0),
            Length::Hectometer => dec(100, 0),
            Length::Decameter => dec(10, 0),
            Length::Meter => Dec::ONE,
            Length::Decimeter => dec(1, 1),
            Length::Centimeter => dec(1, 2),
            Length::Millimeter => dec(1, 3),
            Length::Micrometer => dec(1, 6),
            Length::Nanometer => dec(1, 9),
            Length::Picometer => dec(1, 12),
            Length::Inch => dec(254, 4),
            Length::Foot => dec(3048, 4),
            Length::Yard => dec(9144, 4),
            Length::Mile => dec(1_609_344, 3),
            Length::NauticalMile => dec(1852, 0),
            Length::LightYear => dec(9_460_730_472_580_800, 0),
        }
    }

    pub fn from_symbol(s: &str) -> Option<Length> {
        Some(match s {
            "km" => Length::Kilometer,
            "hm" => Length::Hectometer,
            "dam" => Length::Decameter,
            "m" => Length::Meter,
            "dm" => Length::Decimeter,
            "cm" => Length::Centimeter,
            "mm" => Length::Millimeter,
            "µm" | "μm" => Length::Micrometer,
            "nm" => Length::Nanometer,
            "pm" => Length::Picometer,
            "mi" => Length::Mile,
            "nmi" => Length::NauticalMile,
            "yd" => Length::Yard,
            "ft" => Length::Foot,
            "in" => Length::Inch,
            "ly" => Length::LightYear,
            _ => return None,
        })
    }
}

impl Area {
    /// SI base unit of area: the square meter (`m²`).
    pub const BASE: Area = Area::Square(Length::Meter);

    /// Exact conversion factor from `self` to square meters (`m²`).
    pub fn factor(self) -> Dec {
        match self {
            Area::Square(l) => {
                let f = l.factor();
                f.checked_mul(f).expect("square of length factor fits in Dec")
            }
            // 1 international acre = 4840 yd² = 4046.8564224 m².
            Area::Acre => dec(40_468_564_224, 7),
        }
    }
}

impl Volume {
    /// SI base unit of volume: the cubic meter (`m³`).
    pub const BASE: Volume = Volume::Cubic(Length::Meter);

    /// Exact conversion factor from `self` to cubic meters (`m³`).
    pub fn factor(self) -> Dec {
        match self {
            Volume::Cubic(l) => {
                let f = l.factor();
                f.checked_mul(f).and_then(|sq| sq.checked_mul(f)).expect("cube of length factor fits in Dec")
            }
            Volume::Hectoliter => dec(1, 1),
            Volume::Decaliter => dec(1, 2),
            Volume::Liter => dec(1, 3),
            Volume::Deciliter => dec(1, 4),
            Volume::Centiliter => dec(1, 5),
            Volume::Milliliter => dec(1, 6),
            // US liquid measures (exact derivatives of 1 US gal = 231 in³ = 0.003785411784 m³).
            Volume::Gallon => dec(3_785_411_784, 12),
            Volume::Quart => dec(946_352_946, 12),
            Volume::Pint => dec(473_176_473, 12),
            Volume::Cup => dec(2_365_882_365, 13),
            Volume::Gill => dec(11_829_411_825, 14),
            Volume::FluidOunce => dec(295_735_295_625, 16),
            Volume::Tablespoon => dec(1_478_676_478_125, 17),
            Volume::Teaspoon => dec(492_892_159_375, 17),
            Volume::FluidDram => dec(36_966_911_953_125, 19),
            Volume::Pinch => dec(308_057_599_609_375, 21),
            Volume::Minim => dec(61_611_519_921_875, 21),
            Volume::Barrel => dec(158_987_294_928, 12),
            Volume::Hogshead => dec(238_480_942_392, 12),
            // 1 cord = 128 ft³ = 3.624556363776 m³.
            Volume::Cord => dec(3_624_556_363_776, 12),
            // 1 US bushel = 2150.42 in³ = 0.03523907016688 m³; 1 peck = 1/4 bushel.
            Volume::Bushel => dec(3_523_907_016_688, 14),
            Volume::Peck => dec(880_976_754_172, 14),
            // Imperial measures (exact derivatives of 1 imp gal = 4.54609 L = 0.00454609 m³).
            Volume::ImperialGallon => dec(454_609, 8),
            Volume::ImperialQuart => dec(11_365_225, 10),
            Volume::ImperialPint => dec(56_826_125, 11),
            Volume::ImperialFluidOunce => dec(284_130_625, 13),
        }
    }
}

impl Weight {
    /// SI base unit of mass/weight: the kilogram (`kg`).
    pub const BASE: Weight = Weight::Kilogram;

    /// Exact conversion factor from `self` to [`Weight::Kilogram`].
    pub fn factor(self) -> Dec {
        match self {
            Weight::Kilogram => Dec::ONE,
            Weight::Gram => dec(1, 3),
            Weight::Milligram => dec(1, 6),
            Weight::Microgram => dec(1, 9),
            Weight::MetricTon => dec(1_000, 0),
            // Exact international avoirdupois pound = 0.45359237 kg.
            Weight::Pound => dec(45_359_237, 8),
            Weight::Ounce => dec(28_349_523_125, 12),
            Weight::Dram => dec(17_718_451_953_125, 16),
            Weight::Grain => dec(6_479_891, 11),
            Weight::Pennyweight => dec(155_517_384, 11),
            Weight::Stone => dec(635_029_318, 8),
            Weight::ShortTon => dec(90_718_474, 5),
            Weight::LongTon => dec(10_160_469_088, 7),
        }
    }
}

impl Per {
    /// Exact number of seconds in `self`.
    pub fn seconds(self) -> Dec {
        match self {
            Per::Millisecond => dec(1, 3),
            Per::Second => Dec::ONE,
            Per::Minute => dec(60, 0),
            Per::Hour => dec(3600, 0),
        }
    }

    pub fn from_symbol(s: &str) -> Option<Per> {
        Some(match s {
            "ms" => Per::Millisecond,
            "s" => Per::Second,
            "min" => Per::Minute,
            "h" | "hr" => Per::Hour,
            _ => return None,
        })
    }
}

impl Speed {
    /// SI base unit of speed: meters per second (`m/s`).
    pub const BASE: Speed = Speed::Per(Length::Meter, Per::Second);

    /// Conversion factor from `self` to `m/s`.
    pub fn factor(self) -> Dec {
        match self {
            Speed::Per(l, p) => l.factor().checked_div(p.seconds()).expect("non-zero time denominator"),
            Speed::Knot => dec(1852, 0).checked_div(dec(3600, 0)).expect("non-zero time denominator"),
        }
    }

    /// Converts `value` in `self` to `m/s`, multiplying by the length factor
    /// before dividing by the time denominator so exact rational products
    /// (such as `36 km/h = 10 m/s`) suffer no intermediate rounding.
    pub fn to_base(self, value: Dec) -> Option<Dec> {
        match self {
            Speed::Per(l, p) => value.checked_mul(l.factor())?.checked_div(p.seconds()),
            Speed::Knot => value.checked_mul(dec(1852, 0))?.checked_div(dec(3600, 0)),
        }
    }
}

impl Information {
    /// Stated base unit of information: the byte (`B`).
    pub const BASE: Information = Information::Byte;

    /// Exact conversion factor from `self` to [`Information::Byte`].
    pub fn factor(self) -> Dec {
        match self {
            Information::Byte => Dec::ONE,
            Information::Kilobyte => dec(1_000, 0),
            Information::Megabyte => dec(1_000_000, 0),
            Information::Gigabyte => dec(1_000_000_000, 0),
            Information::Terabyte => dec(1_000_000_000_000, 0),
            Information::Petabyte => dec(1_000_000_000_000_000, 0),
            Information::Bit => dec(125, 3),
            Information::Kilobit => dec(125, 0),
            Information::Megabit => dec(125_000, 0),
            Information::Gigabit => dec(125_000_000, 0),
            Information::Terabit => dec(125_000_000_000, 0),
            Information::Petabit => dec(125_000_000_000_000, 0),
        }
    }
}

impl Angle {
    /// SI base unit of angle: the radian (`rad`).
    pub const BASE: Angle = Angle::Radian;

    /// Conversion factor from `self` to [`Angle::Radian`].
    pub fn factor(self) -> Dec {
        const PI: i128 = 31_415_926_535_897_932_384_626_433_832_795_029;
        match self {
            Angle::Radian => Dec::ONE,
            Angle::Degree => dec(PI, 34).checked_div(dec(180, 0)).expect("non-zero divisor"),
            Angle::Turn => dec(PI * 2, 34),
        }
    }
}

impl DurationUnit {
    /// SI base unit of duration: the second (`s`).
    pub const BASE: DurationUnit = DurationUnit::Second;

    /// Exact conversion factor from `self` to [`DurationUnit::Second`].
    pub fn factor(self) -> Dec {
        match self {
            DurationUnit::Nanosecond => dec(1, 9),
            DurationUnit::Microsecond => dec(1, 6),
            DurationUnit::Millisecond => dec(1, 3),
            DurationUnit::Second => Dec::ONE,
            DurationUnit::Minute => dec(60, 0),
            DurationUnit::Hour => dec(3600, 0),
            DurationUnit::Day => dec(86_400, 0),
        }
    }
}

impl Frequency {
    /// SI base unit of frequency: the hertz (`Hz`).
    pub const BASE: Frequency = Frequency::Hertz;

    /// Exact conversion factor from `self` to [`Frequency::Hertz`].
    pub fn factor(self) -> Dec {
        match self {
            Frequency::Hertz => Dec::ONE,
            Frequency::Kilohertz => dec(1_000, 0),
            Frequency::Megahertz => dec(1_000_000, 0),
            Frequency::Gigahertz => dec(1_000_000_000, 0),
        }
    }
}

impl Power {
    /// SI base unit of power: the watt (`W`).
    pub const BASE: Power = Power::Watt;

    /// Exact conversion factor from `self` to [`Power::Watt`].
    pub fn factor(self) -> Dec {
        match self {
            Power::Watt => Dec::ONE,
            Power::Kilowatt => dec(1_000, 0),
            Power::Megawatt => dec(1_000_000, 0),
            Power::Gigawatt => dec(1_000_000_000, 0),
        }
    }
}

impl Energy {
    /// SI base unit of energy: the joule (`J`).
    pub const BASE: Energy = Energy::Joule;

    /// Exact conversion factor from `self` to [`Energy::Joule`].
    pub fn factor(self) -> Dec {
        match self {
            Energy::Joule => Dec::ONE,
            Energy::Kilojoule => dec(1_000, 0),
            Energy::WattHour => dec(3_600, 0),
            Energy::KilowattHour => dec(3_600_000, 0),
        }
    }
}

impl Temperature {
    /// SI base unit of temperature: the kelvin (`K`).
    pub const BASE: Temperature = Temperature::Kelvin;

    /// Affine conversion of `value` in `self` to [`Temperature::Kelvin`];
    /// returns `None` for [`Temperature::Degree`] (unstated scale).
    pub fn to_base(self, value: Dec) -> Option<Dec> {
        match self {
            Temperature::Kelvin => Some(value),
            Temperature::Celsius => value.checked_add(dec(27315, 2)),
            // (T + 459.67) * 5 / 9 = (5 * T + 2298.35) / 9
            Temperature::Fahrenheit => {
                value.checked_mul(dec(5, 0))?.checked_add(dec(229_835, 2))?.checked_div(dec(9, 0))
            }
            Temperature::Rankine => value.checked_mul(dec(5, 0))?.checked_div(dec(9, 0)),
            // 373.15 - T * 2 / 3 = (1119.45 - 2 * T) / 3
            Temperature::Delisle => dec(111_945, 2).checked_sub(value.checked_mul(dec(2, 0))?)?.checked_div(dec(3, 0)),
            Temperature::Degree => None,
        }
    }
}

impl Age {
    /// Stated base unit of age: [`Age::Year`] (`years old`).
    pub const BASE: Age = Age::Year;

    pub fn to_base(self, value: Dec) -> Option<Dec> {
        match self {
            Age::Year => Some(value),
            Age::Month => value.checked_div(dec(12, 0)),
            Age::Week => value.checked_mul(dec(7, 0))?.checked_div(dec(365, 0)),
            Age::Day => value.checked_div(dec(365, 0)),
            Age::Unstated => None,
        }
    }
}

impl Unit {
    /// The SI or stated base [`Unit`] for `self`; `None` for currency or
    /// unstated temperature/age.
    pub fn base_unit(self) -> Option<Unit> {
        Some(match self {
            Unit::Length(_) => Unit::Length(Length::BASE),
            Unit::Area(_) => Unit::Area(Area::BASE),
            Unit::Volume(_) => Unit::Volume(Volume::BASE),
            Unit::Weight(_) => Unit::Weight(Weight::BASE),
            Unit::Speed(_) => Unit::Speed(Speed::BASE),
            Unit::Information(_) => Unit::Information(Information::BASE),
            Unit::Angle(_) => Unit::Angle(Angle::BASE),
            Unit::Duration(_) => Unit::Duration(DurationUnit::BASE),
            Unit::Frequency(_) => Unit::Frequency(Frequency::BASE),
            Unit::Power(_) => Unit::Power(Power::BASE),
            Unit::Energy(_) => Unit::Energy(Energy::BASE),
            Unit::Temperature(Temperature::Degree) | Unit::Age(Age::Unstated) | Unit::Currency(_) => return None,
            Unit::Temperature(_) => Unit::Temperature(Temperature::BASE),
            Unit::Age(_) => Unit::Age(Age::BASE),
        })
    }

    /// Linear multiplicative factor to [`Unit::base_unit`]; `None` for affine
    /// temperature scales (`Celsius`, `Fahrenheit`, `Delisle`), unstated
    /// scales, and currency.
    pub fn conversion_factor(self) -> Option<Dec> {
        Some(match self {
            Unit::Length(l) => l.factor(),
            Unit::Area(a) => a.factor(),
            Unit::Volume(v) => v.factor(),
            Unit::Weight(w) => w.factor(),
            Unit::Speed(s) => s.factor(),
            Unit::Information(i) => i.factor(),
            Unit::Angle(a) => a.factor(),
            Unit::Duration(d) => d.factor(),
            Unit::Frequency(f) => f.factor(),
            Unit::Power(p) => p.factor(),
            Unit::Energy(e) => e.factor(),
            Unit::Temperature(Temperature::Kelvin) => Dec::ONE,
            Unit::Temperature(Temperature::Rankine) => dec(5, 0).checked_div(dec(9, 0))?,
            Unit::Age(a) => a.to_base(Dec::ONE)?,
            Unit::Temperature(_) | Unit::Currency(_) => return None,
        })
    }

    /// Normalizes `value` in `self` to its base [`Unit`] (linear or affine).
    pub fn normalize(self, value: Dec) -> Option<(Dec, Unit)> {
        let base = self.base_unit()?;
        let converted = match self {
            Unit::Speed(s) => s.to_base(value)?,
            Unit::Temperature(t) => t.to_base(value)?,
            Unit::Age(a) => a.to_base(value)?,
            _ => value.checked_mul(self.conversion_factor()?)?,
        };
        Some((converted, base))
    }
}
