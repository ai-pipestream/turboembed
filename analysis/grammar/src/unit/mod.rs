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
mod kinds;
mod physical;
pub use kinds::*;
pub use physical::*;

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
    Rate(Rate),
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
            Unit::Currency(_) | Unit::Rate(Rate::Currency(..)) => Measure::Currency,
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
            Unit::Rate(Rate::Information(..)) => Dimension::Information,
            Unit::Information(_) => Dimension::Information,
            Unit::Angle(_) => Dimension::Angle,
            _ => return None,
        })
    }

    pub fn currency(self) -> Option<CurrencyUnit> {
        match self {
            Unit::Currency(c) => Some(c),
            Unit::Rate(Rate::Currency(c, _)) => Some(c),
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
            Unit::Rate(Rate::Information(i, p)) => {
                out.push_str(i.symbol());
                out.push('/');
                p.symbol()
            }
            Unit::Rate(Rate::Currency(c, p)) => {
                c.write_symbol(out);
                out.push('/');
                p.symbol()
            }
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
            Unit::Rate(Rate::Information(i, p)) => write!(f, "{} per {}", i.name(), p.name()),
            Unit::Rate(Rate::Currency(c, p)) => write!(f, "{c} per {}", p.name()),
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
mod tests;
