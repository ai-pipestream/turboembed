//! Units of measure and of money, typed, with no strings for what a unit
//! is: a [`Unit`] is one of the enums below, grouped by what it measures.
//!
//! [`fmt::Display`] writes the English name Recognizers-Text gives a unit
//! (`Kilometer per hour`, `Square meter`, `C`, `United States dollar`),
//! which the conformance tests compare. [`Unit::write_symbol`] writes the
//! short form a layer stores (`km/h`, `m²`, `°C`, `years old`).

pub mod currency;

use std::fmt;

pub use currency::{CurrencyUnit, Generic, Minor};

/// A length.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Length {
    Kilometer,
    Hectometer,
    Decameter,
    Meter,
    Decimeter,
    Centimeter,
    Millimeter,
    Micrometer,
    Nanometer,
    Picometer,
    Mile,
    Yard,
    Foot,
    Inch,
    LightYear,
}

impl Length {
    fn name(self) -> &'static str {
        match self {
            Length::Kilometer => "Kilometer",
            Length::Hectometer => "Hectometer",
            Length::Decameter => "Decameter",
            Length::Meter => "Meter",
            Length::Decimeter => "Decimeter",
            Length::Centimeter => "Centimeter",
            Length::Millimeter => "Millimeter",
            Length::Micrometer => "Micrometer",
            Length::Nanometer => "Nanometer",
            Length::Picometer => "Picometer",
            Length::Mile => "Mile",
            Length::Yard => "Yard",
            Length::Foot => "Foot",
            Length::Inch => "Inch",
            Length::LightYear => "Light year",
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Length::Kilometer => "km",
            Length::Hectometer => "hm",
            Length::Decameter => "dam",
            Length::Meter => "m",
            Length::Decimeter => "dm",
            Length::Centimeter => "cm",
            Length::Millimeter => "mm",
            Length::Micrometer => "µm",
            Length::Nanometer => "nm",
            Length::Picometer => "pm",
            Length::Mile => "mi",
            Length::Yard => "yd",
            Length::Foot => "ft",
            Length::Inch => "in",
            Length::LightYear => "ly",
        }
    }
}

/// An area: a square of a length, or an acre. A hectare is a square
/// hectometer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Area {
    Square(Length),
    Acre,
}

/// A volume.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Volume {
    Cubic(Length),
    Hectoliter,
    Decaliter,
    Liter,
    Deciliter,
    Centiliter,
    Milliliter,
    FluidOunce,
    Teaspoon,
    Tablespoon,
    Pint,
    Quart,
    Cup,
    Gill,
    Pinch,
    FluidDram,
    Barrel,
    Minim,
    Cord,
    Peck,
    Bushel,
    Hogshead,
    Gallon,
}

impl Volume {
    fn name(self) -> &'static str {
        match self {
            Volume::Cubic(_) => "",
            Volume::Hectoliter => "Hectoliter",
            Volume::Decaliter => "Decaliter",
            Volume::Liter => "Liter",
            Volume::Deciliter => "Deciliter",
            Volume::Centiliter => "Centiliter",
            Volume::Milliliter => "Milliliter",
            Volume::FluidOunce => "Fluid ounce",
            Volume::Teaspoon => "Teaspoon",
            Volume::Tablespoon => "Tablespoon",
            Volume::Pint => "Pint",
            Volume::Quart => "Quart",
            Volume::Cup => "Cup",
            Volume::Gill => "Gill",
            Volume::Pinch => "Pinch",
            Volume::FluidDram => "Fluid Dram",
            Volume::Barrel => "Barrel",
            Volume::Minim => "Minim",
            Volume::Cord => "Cord",
            Volume::Peck => "Peck",
            Volume::Bushel => "Bushel",
            Volume::Hogshead => "Hogshead",
            Volume::Gallon => "Gallon",
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Volume::Cubic(_) => "",
            Volume::Hectoliter => "hl",
            Volume::Decaliter => "dal",
            Volume::Liter => "l",
            Volume::Deciliter => "dl",
            Volume::Centiliter => "cl",
            Volume::Milliliter => "ml",
            Volume::FluidOunce => "fl oz",
            Volume::Teaspoon => "tsp",
            Volume::Tablespoon => "tbsp",
            Volume::Pint => "pt",
            Volume::Quart => "qt",
            Volume::Cup => "cup",
            Volume::Gill => "gi",
            Volume::Pinch => "pinch",
            Volume::FluidDram => "fl dr",
            Volume::Barrel => "bbl",
            Volume::Minim => "minim",
            Volume::Cord => "cord",
            Volume::Peck => "pk",
            Volume::Bushel => "bu",
            Volume::Hogshead => "hhd",
            Volume::Gallon => "gal",
        }
    }
}

/// A weight (a mass, as English speaks of it).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Weight {
    Kilogram,
    Gram,
    Milligram,
    Microgram,
    MetricTon,
    Ton,
    Pound,
    Ounce,
    Grain,
    Pennyweight,
    LongTon,
    ShortTon,
    Stone,
    Dram,
}

impl Weight {
    fn name(self) -> &'static str {
        match self {
            Weight::Kilogram => "Kilogram",
            Weight::Gram => "Gram",
            Weight::Milligram => "Milligram",
            Weight::Microgram => "Microgram",
            Weight::MetricTon => "Metric ton",
            Weight::Ton => "Ton",
            Weight::Pound => "Pound",
            Weight::Ounce => "Ounce",
            Weight::Grain => "Grain",
            Weight::Pennyweight => "Pennyweight",
            Weight::LongTon => "Long ton (British)",
            Weight::ShortTon => "Short ton (US)",
            Weight::Stone => "Stone",
            Weight::Dram => "Dram",
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Weight::Kilogram => "kg",
            Weight::Gram => "g",
            Weight::Milligram => "mg",
            Weight::Microgram => "µg",
            Weight::MetricTon => "t",
            Weight::Ton => "ton",
            Weight::Pound => "lb",
            Weight::Ounce => "oz",
            Weight::Grain => "gr",
            Weight::Pennyweight => "dwt",
            Weight::LongTon => "long ton",
            Weight::ShortTon => "short ton",
            Weight::Stone => "st",
            Weight::Dram => "dr",
        }
    }
}

/// The time a speed is measured over.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Per {
    Millisecond,
    Second,
    Minute,
    Hour,
}

impl Per {
    fn name(self) -> &'static str {
        match self {
            Per::Millisecond => "millisecond",
            Per::Second => "second",
            Per::Minute => "minute",
            Per::Hour => "hour",
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Per::Millisecond => "ms",
            Per::Second => "s",
            Per::Minute => "min",
            Per::Hour => "h",
        }
    }
}

/// A speed: a length over a time, or a knot.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Speed {
    Per(Length, Per),
    Knot,
}

/// An amount of information: bits or bytes with a decimal prefix.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Information {
    Bit,
    Kilobit,
    Megabit,
    Gigabit,
    Terabit,
    Petabit,
    Byte,
    Kilobyte,
    Megabyte,
    Gigabyte,
    Terabyte,
    Petabyte,
}

impl Information {
    /// The unit of `bytes` (or bits) with the decimal prefix of ten to the
    /// `power`: 0, 3, 6, 9, 12 or 15.
    pub fn of(power: u8, bytes: bool) -> Option<Information> {
        use Information::*;
        Some(match (power, bytes) {
            (0, false) => Bit,
            (3, false) => Kilobit,
            (6, false) => Megabit,
            (9, false) => Gigabit,
            (12, false) => Terabit,
            (15, false) => Petabit,
            (0, true) => Byte,
            (3, true) => Kilobyte,
            (6, true) => Megabyte,
            (9, true) => Gigabyte,
            (12, true) => Terabyte,
            (15, true) => Petabyte,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        use Information::*;
        match self {
            Bit => "Bit",
            Kilobit => "Kilobit",
            Megabit => "Megabit",
            Gigabit => "Gigabit",
            Terabit => "Terabit",
            Petabit => "Petabit",
            Byte => "Byte",
            Kilobyte => "Kilobyte",
            Megabyte => "Megabyte",
            Gigabyte => "Gigabyte",
            Terabyte => "Terabyte",
            Petabyte => "Petabyte",
        }
    }

    fn symbol(self) -> &'static str {
        use Information::*;
        match self {
            Bit => "bit",
            Kilobit => "kbit",
            Megabit => "Mbit",
            Gigabit => "Gbit",
            Terabit => "Tbit",
            Petabit => "Pbit",
            Byte => "B",
            Kilobyte => "kB",
            Megabyte => "MB",
            Gigabyte => "GB",
            Terabyte => "TB",
            Petabyte => "PB",
        }
    }
}

/// An angle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Angle {
    Degree,
    Radian,
    Turn,
}

/// A temperature scale, or `Degree` when the text names none (`98
/// degrees`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Temperature {
    Celsius,
    Fahrenheit,
    Kelvin,
    Rankine,
    Delisle,
    Degree,
}

/// What an age is counted in, or `Unstated` (`age 87`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Age {
    Day,
    Week,
    Month,
    Year,
    Unstated,
}

/// What a dimension measures: the Recognizers-Text subtype.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dimension {
    Length,
    Area,
    Volume,
    Weight,
    Speed,
    Information,
    Angle,
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

#[cfg(test)]
mod tests {
    use turbo_analysis_layers::Currency;

    use super::*;

    fn symbol(u: Unit) -> String {
        let mut s = String::new();
        u.write_symbol(&mut s);
        s
    }

    #[test]
    fn names_follow_recognizers_text() {
        assert_eq!(Unit::Speed(Speed::Per(Length::Kilometer, Per::Hour)).to_string(), "Kilometer per hour");
        assert_eq!(Unit::Area(Area::Square(Length::Meter)).to_string(), "Square meter");
        assert_eq!(Unit::Volume(Volume::Cubic(Length::Foot)).to_string(), "Cubic foot");
        assert_eq!(Unit::Area(Area::Square(Length::LightYear)).to_string(), "Square light year");
        assert_eq!(Unit::Temperature(Temperature::Celsius).to_string(), "C");
        assert_eq!(Unit::Age(Age::Unstated).to_string(), "Age");
        assert_eq!(Unit::Weight(Weight::MetricTon).to_string(), "Metric ton");
        assert_eq!(Unit::Information(Information::Megabyte).to_string(), "Megabyte");
        assert_eq!(Unit::Angle(Angle::Turn).to_string(), "Turn");
        assert_eq!(Unit::Speed(Speed::Knot).to_string(), "Knot");
        let usd = Currency::parse("USD").unwrap();
        assert_eq!(Unit::Currency(CurrencyUnit::Code(usd)).to_string(), "United States dollar");
    }

    #[test]
    fn symbols_are_short_forms() {
        assert_eq!(symbol(Unit::Speed(Speed::Per(Length::Kilometer, Per::Hour))), "km/h");
        assert_eq!(symbol(Unit::Area(Area::Square(Length::Meter))), "m²");
        assert_eq!(symbol(Unit::Volume(Volume::Cubic(Length::Centimeter))), "cm³");
        assert_eq!(symbol(Unit::Temperature(Temperature::Fahrenheit)), "°F");
        assert_eq!(symbol(Unit::Age(Age::Year)), "years old");
        assert_eq!(symbol(Unit::Information(Information::Gigabyte)), "GB");
        assert_eq!(symbol(Unit::Currency(CurrencyUnit::Generic(Generic::Dollar))), "dollar");
    }

    #[test]
    fn kinds_and_subtypes() {
        assert_eq!(Unit::Weight(Weight::Pound).measure(), Measure::Dimension);
        assert_eq!(Unit::Weight(Weight::Pound).dimension(), Some(Dimension::Weight));
        assert_eq!(Unit::Temperature(Temperature::Degree).dimension(), None);
        assert_eq!(Unit::Age(Age::Day).measure().name(), "age");
        assert_eq!(Dimension::Information.name(), "Information");
        assert_eq!(Information::of(9, true), Some(Information::Gigabyte));
        assert_eq!(Information::of(9, false), Some(Information::Gigabit));
        assert_eq!(Information::of(2, true), None);
        let pound = Unit::Currency(CurrencyUnit::Generic(Generic::Pound));
        assert_eq!(pound.measure(), Measure::Currency);
        assert!(pound.currency().is_some());
        assert_eq!(Unit::Length(Length::Inch).currency(), None);
    }

    #[test]
    fn a_unit_is_small() {
        assert!(std::mem::size_of::<Unit>() <= 6);
    }
}
