//! Units of measure and of money, typed, with no strings for what a unit
//! is: a [`Unit`] is one of the enums below, grouped by what it measures.
//!
//! [`fmt::Display`] writes the English name Recognizers-Text gives a unit
//! (`Kilometer per hour`, `Square meter`, `C`, `United States dollar`),
//! which the conformance tests compare. [`Unit::write_symbol`] writes the
//! short form a layer stores (`km/h`, `m²`, `°C`, `years old`).
//!
//! [`convert`] holds the shared unit symbol table ([`DEFAULT_UNITS`]),
//! symbol lookup ([`Unit::from_symbol`]), base unit mapping
//! ([`Unit::base_unit`]), exact conversion factors ([`Unit::conversion_factor`]),
//! and quantity normalization ([`Unit::normalize`], including affine
//! temperature conversion to Kelvin).

mod convert;
pub mod currency;
mod kinds;
mod physical;

use std::fmt;

pub use convert::DEFAULT_UNITS;
pub use currency::{CurrencyUnit, Generic, Minor};
pub use kinds::*;
pub use physical::*;

/// What a dimension measures: the Recognizers-Text subtype (plus duration,
/// frequency, power, and energy for the shared numeric unit table).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dimension {
    Length,
    Area,
    Volume,
    Weight,
    Speed,
    Information,
    Angle,
    Duration,
    Frequency,
    Power,
    Energy,
}

impl Dimension {
    /// The subtype name Recognizers-Text uses: `Length`, `Weight`.
    pub fn name(self) -> &'static str {
        match self {
            Dimension::Length => "Length",
            Dimension::Area => "Area",
            Dimension::Volume => "Volume",
            Dimension::Weight => "Weight",
            Dimension::Speed => "Speed",
            Dimension::Information => "Information",
            Dimension::Angle => "Angle",
            Dimension::Duration => "Duration",
            Dimension::Frequency => "Frequency",
            Dimension::Power => "Power",
            Dimension::Energy => "Energy",
        }
    }
}

/// The four kinds of Recognizers-Text's number-with-unit models.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Measure {
    Currency,
    Dimension,
    Temperature,
    Age,
}

impl Measure {
    /// The type name Recognizers-Text uses: `currency`, `dimension`,
    /// `temperature`, `age`.
    pub fn name(self) -> &'static str {
        match self {
            Measure::Currency => "currency",
            Measure::Dimension => "dimension",
            Measure::Temperature => "temperature",
            Measure::Age => "age",
        }
    }
}

/// A unit of measure or of money.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Unit {
    Length(Length),
    Area(Area),
    Volume(Volume),
    Weight(Weight),
    Speed(Speed),
    Information(Information),
    Angle(Angle),
    Duration(DurationUnit),
    Frequency(Frequency),
    Power(Power),
    Energy(Energy),
    Temperature(Temperature),
    Age(Age),
    Currency(CurrencyUnit),
}

impl Unit {
    pub fn measure(self) -> Measure {
        match self {
            Unit::Temperature(_) => Measure::Temperature,
            Unit::Age(_) => Measure::Age,
            Unit::Currency(_) => Measure::Currency,
            _ => Measure::Dimension,
        }
    }

    /// What a dimension measures; `None` for a temperature, an age or
    /// money.
    pub fn dimension(self) -> Option<Dimension> {
        Some(match self {
            Unit::Length(_) => Dimension::Length,
            Unit::Area(_) => Dimension::Area,
            Unit::Volume(_) => Dimension::Volume,
            Unit::Weight(_) => Dimension::Weight,
            Unit::Speed(_) => Dimension::Speed,
            Unit::Information(_) => Dimension::Information,
            Unit::Angle(_) => Dimension::Angle,
            Unit::Duration(_) => Dimension::Duration,
            Unit::Frequency(_) => Dimension::Frequency,
            Unit::Power(_) => Dimension::Power,
            Unit::Energy(_) => Dimension::Energy,
            _ => return None,
        })
    }

    pub fn currency(self) -> Option<CurrencyUnit> {
        match self {
            Unit::Currency(c) => Some(c),
            _ => None,
        }
    }

    /// Appends the unit's short form: `km`, `m²`, `km/h`, `°C`, `MB`,
    /// `years old`, a currency's code or name.
    pub fn write_symbol(self, out: &mut String) {
        let s = match self {
            Unit::Length(l) => l.symbol(),
            Unit::Area(Area::Square(l)) => {
                out.push_str(l.symbol());
                "²"
            }
            Unit::Area(Area::Acre) => "ac",
            Unit::Volume(Volume::Cubic(l)) => {
                out.push_str(l.symbol());
                "³"
            }
            Unit::Volume(v) => v.symbol(),
            Unit::Weight(w) => w.symbol(),
            Unit::Speed(Speed::Per(l, p)) => {
                out.push_str(l.symbol());
                out.push('/');
                p.symbol()
            }
            Unit::Speed(Speed::Knot) => "kn",
            Unit::Information(i) => i.symbol(),
            Unit::Angle(Angle::Degree) => "°",
            Unit::Angle(Angle::Radian) => "rad",
            Unit::Angle(Angle::Turn) => "tr",
            Unit::Duration(d) => d.symbol(),
            Unit::Frequency(f) => f.symbol(),
            Unit::Power(p) => p.symbol(),
            Unit::Energy(e) => e.symbol(),
            Unit::Temperature(t) => match t {
                Temperature::Celsius => "°C",
                Temperature::Fahrenheit => "°F",
                Temperature::Kelvin => "K",
                Temperature::Rankine => "°R",
                Temperature::Delisle => "°De",
                Temperature::Degree => "deg",
            },
            Unit::Age(a) => match a {
                Age::Day => "days old",
                Age::Week => "weeks old",
                Age::Month => "months old",
                Age::Year => "years old",
                Age::Unstated => "age",
            },
            Unit::Currency(c) => {
                c.write_symbol(out);
                ""
            }
        };
        out.push_str(s);
    }

    /// Returns the unit's short symbol as a `String`.
    pub fn symbol(self) -> String {
        let mut s = String::new();
        self.write_symbol(&mut s);
        s
    }
}

/// Writes `name` with its first letter in lowercase: `Square foot`.
fn lower_first(f: &mut fmt::Formatter<'_>, name: &str) -> fmt::Result {
    let mut chars = name.chars();
    if let Some(c) = chars.next() {
        write!(f, "{}", c.to_ascii_lowercase())?;
    }
    f.write_str(chars.as_str())
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Unit::Length(l) => f.write_str(l.name()),
            Unit::Area(Area::Square(l)) => {
                f.write_str("Square ")?;
                lower_first(f, l.name())
            }
            Unit::Area(Area::Acre) => f.write_str("Acre"),
            Unit::Volume(Volume::Cubic(l)) => {
                f.write_str("Cubic ")?;
                lower_first(f, l.name())
            }
            Unit::Volume(v) => f.write_str(v.name()),
            Unit::Weight(w) => f.write_str(w.name()),
            Unit::Speed(Speed::Per(l, p)) => write!(f, "{} per {}", l.name(), p.name()),
            Unit::Speed(Speed::Knot) => f.write_str("Knot"),
            Unit::Information(i) => f.write_str(i.name()),
            Unit::Angle(a) => f.write_str(match a {
                Angle::Degree => "Degree",
                Angle::Radian => "Radian",
                Angle::Turn => "Turn",
            }),
            Unit::Duration(d) => f.write_str(d.name()),
            Unit::Frequency(freq) => f.write_str(freq.name()),
            Unit::Power(p) => f.write_str(p.name()),
            Unit::Energy(e) => f.write_str(e.name()),
            Unit::Temperature(t) => f.write_str(match t {
                Temperature::Celsius => "C",
                Temperature::Fahrenheit => "F",
                Temperature::Kelvin => "K",
                Temperature::Rankine => "R",
                Temperature::Delisle => "D",
                Temperature::Degree => "Degree",
            }),
            Unit::Age(a) => f.write_str(match a {
                Age::Day => "Day",
                Age::Week => "Week",
                Age::Month => "Month",
                Age::Year => "Year",
                Age::Unstated => "Age",
            }),
            Unit::Currency(c) => fmt::Display::fmt(&c, f),
        }
    }
}

impl Unit {
    /// Looks up a [`Unit`] by its short symbol or token in the shared unit
    /// table (`km`, `m²`, `cm³`, `km/h`, `mph`, `lbs`, `L`, `°C`, `C`, etc.).
    pub fn from_symbol(symbol: &str) -> Option<Unit> {
        if let Some(prefix) = symbol.strip_suffix('²') {
            return Length::from_symbol(prefix).map(|l| Unit::Area(Area::Square(l)));
        }
        if let Some(prefix) = symbol.strip_suffix('³') {
            return Length::from_symbol(prefix).map(|l| Unit::Volume(Volume::Cubic(l)));
        }
        if let Some((num, den)) = symbol.split_once('/') {
            return Some(Unit::Speed(Speed::Per(Length::from_symbol(num)?, Per::from_symbol(den)?)));
        }
        if let Some(l) = Length::from_symbol(symbol) {
            return Some(Unit::Length(l));
        }
        Some(match symbol {
            "ac" => Unit::Area(Area::Acre),
            "hl" => Unit::Volume(Volume::Hectoliter),
            "dal" => Unit::Volume(Volume::Decaliter),
            "l" | "L" => Unit::Volume(Volume::Liter),
            "dl" => Unit::Volume(Volume::Deciliter),
            "cl" => Unit::Volume(Volume::Centiliter),
            "ml" | "mL" => Unit::Volume(Volume::Milliliter),
            "fl oz" => Unit::Volume(Volume::FluidOunce),
            "tsp" => Unit::Volume(Volume::Teaspoon),
            "tbsp" => Unit::Volume(Volume::Tablespoon),
            "pt" => Unit::Volume(Volume::Pint),
            "qt" => Unit::Volume(Volume::Quart),
            "cup" => Unit::Volume(Volume::Cup),
            "gi" => Unit::Volume(Volume::Gill),
            "pinch" => Unit::Volume(Volume::Pinch),
            "fl dr" => Unit::Volume(Volume::FluidDram),
            "bbl" => Unit::Volume(Volume::Barrel),
            "minim" => Unit::Volume(Volume::Minim),
            "cord" => Unit::Volume(Volume::Cord),
            "pk" => Unit::Volume(Volume::Peck),
            "bu" => Unit::Volume(Volume::Bushel),
            "hhd" => Unit::Volume(Volume::Hogshead),
            "gal" => Unit::Volume(Volume::Gallon),
            "imp gal" => Unit::Volume(Volume::ImperialGallon),
            "imp qt" => Unit::Volume(Volume::ImperialQuart),
            "imp pt" => Unit::Volume(Volume::ImperialPint),
            "imp fl oz" => Unit::Volume(Volume::ImperialFluidOunce),
            "kg" => Unit::Weight(Weight::Kilogram),
            "g" => Unit::Weight(Weight::Gram),
            "mg" => Unit::Weight(Weight::Milligram),
            "µg" | "μg" | "mcg" => Unit::Weight(Weight::Microgram),
            "t" => Unit::Weight(Weight::MetricTon),
            "lb" | "lbs" => Unit::Weight(Weight::Pound),
            "oz" => Unit::Weight(Weight::Ounce),
            "gr" => Unit::Weight(Weight::Grain),
            "dwt" => Unit::Weight(Weight::Pennyweight),
            "long tn" => Unit::Weight(Weight::LongTon),
            "sh tn" => Unit::Weight(Weight::ShortTon),
            "st" => Unit::Weight(Weight::Stone),
            "dr" => Unit::Weight(Weight::Dram),
            "mph" => Unit::Speed(Speed::Per(Length::Mile, Per::Hour)),
            "kph" | "kmph" => Unit::Speed(Speed::Per(Length::Kilometer, Per::Hour)),
            "kn" => Unit::Speed(Speed::Knot),
            "bit" => Unit::Information(Information::Bit),
            "kbit" => Unit::Information(Information::Kilobit),
            "Mbit" => Unit::Information(Information::Megabit),
            "Gbit" => Unit::Information(Information::Gigabit),
            "Tbit" => Unit::Information(Information::Terabit),
            "Pbit" => Unit::Information(Information::Petabit),
            "B" => Unit::Information(Information::Byte),
            "kB" | "KB" => Unit::Information(Information::Kilobyte),
            "MB" => Unit::Information(Information::Megabyte),
            "GB" => Unit::Information(Information::Gigabyte),
            "TB" => Unit::Information(Information::Terabyte),
            "PB" => Unit::Information(Information::Petabyte),
            "°" => Unit::Angle(Angle::Degree),
            "rad" => Unit::Angle(Angle::Radian),
            "tr" => Unit::Angle(Angle::Turn),
            "ns" => Unit::Duration(DurationUnit::Nanosecond),
            "µs" | "μs" => Unit::Duration(DurationUnit::Microsecond),
            "ms" => Unit::Duration(DurationUnit::Millisecond),
            "s" => Unit::Duration(DurationUnit::Second),
            "min" => Unit::Duration(DurationUnit::Minute),
            "hr" | "h" => Unit::Duration(DurationUnit::Hour),
            "d" => Unit::Duration(DurationUnit::Day),
            "Hz" => Unit::Frequency(Frequency::Hertz),
            "kHz" => Unit::Frequency(Frequency::Kilohertz),
            "MHz" => Unit::Frequency(Frequency::Megahertz),
            "GHz" => Unit::Frequency(Frequency::Gigahertz),
            "W" => Unit::Power(Power::Watt),
            "kW" => Unit::Power(Power::Kilowatt),
            "MW" => Unit::Power(Power::Megawatt),
            "GW" => Unit::Power(Power::Gigawatt),
            "J" => Unit::Energy(Energy::Joule),
            "kJ" => Unit::Energy(Energy::Kilojoule),
            "Wh" => Unit::Energy(Energy::WattHour),
            "kWh" => Unit::Energy(Energy::KilowattHour),
            "°C" | "C" => Unit::Temperature(Temperature::Celsius),
            "°F" | "F" => Unit::Temperature(Temperature::Fahrenheit),
            "K" => Unit::Temperature(Temperature::Kelvin),
            "°R" | "R" => Unit::Temperature(Temperature::Rankine),
            "°De" | "De" => Unit::Temperature(Temperature::Delisle),
            "deg" => Unit::Temperature(Temperature::Degree),
            "days old" => Unit::Age(Age::Day),
            "weeks old" => Unit::Age(Age::Week),
            "months old" => Unit::Age(Age::Month),
            "years old" => Unit::Age(Age::Year),
            "age" => Unit::Age(Age::Unstated),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests;
