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
    NauticalMile,
    Yard,
    Foot,
    Inch,
    LightYear,
}

impl Length {
    pub(super) fn name(self) -> &'static str {
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
            Length::NauticalMile => "Nautical mile",
            Length::Yard => "Yard",
            Length::Foot => "Foot",
            Length::Inch => "Inch",
            Length::LightYear => "Light year",
        }
    }

    pub(super) fn symbol(self) -> &'static str {
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
            Length::NauticalMile => "nmi",
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

/// A volume. `Gallon`, `Quart`, `Pint` and `FluidOunce` are the US
/// measures, which an unqualified name is taken to mean; the imperial ones
/// are their own units.
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
    ImperialGallon,
    ImperialQuart,
    ImperialPint,
    ImperialFluidOunce,
}

impl Volume {
    pub(super) fn name(self) -> &'static str {
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
            Volume::ImperialGallon => "Imperial gallon",
            Volume::ImperialQuart => "Imperial quart",
            Volume::ImperialPint => "Imperial pint",
            Volume::ImperialFluidOunce => "Imperial fluid ounce",
        }
    }

    pub(super) fn symbol(self) -> &'static str {
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
            Volume::ImperialGallon => "imp gal",
            Volume::ImperialQuart => "imp qt",
            Volume::ImperialPint => "imp pt",
            Volume::ImperialFluidOunce => "imp fl oz",
        }
    }
}

/// A weight (a mass, as English speaks of it). A metric ton (a tonne) is
/// 1000 kg, a short ton 2000 lb and a long ton 2240 lb.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Weight {
    Kilogram,
    Gram,
    Milligram,
    Microgram,
    MetricTon,
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
    pub(super) fn name(self) -> &'static str {
        match self {
            Weight::Kilogram => "Kilogram",
            Weight::Gram => "Gram",
            Weight::Milligram => "Milligram",
            Weight::Microgram => "Microgram",
            Weight::MetricTon => "Metric ton",
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

    pub(super) fn symbol(self) -> &'static str {
        match self {
            Weight::Kilogram => "kg",
            Weight::Gram => "g",
            Weight::Milligram => "mg",
            Weight::Microgram => "µg",
            Weight::MetricTon => "t",
            Weight::Pound => "lb",
            Weight::Ounce => "oz",
            Weight::Grain => "gr",
            Weight::Pennyweight => "dwt",
            Weight::LongTon => "long tn",
            Weight::ShortTon => "sh tn",
            Weight::Stone => "st",
            Weight::Dram => "dr",
        }
    }
}

use super::currency::CurrencyUnit;
use super::kinds::Information;

/// The time a speed or another rate is measured over.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Per {
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

/// A rate other than a speed: information or money over a time
/// (`10 Mb/s`, `$5 per hour`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Rate {
    Information(Information, Per),
    Currency(CurrencyUnit, Per),
}

impl Per {
    pub(super) fn name(self) -> &'static str {
        match self {
            Per::Millisecond => "millisecond",
            Per::Second => "second",
            Per::Minute => "minute",
            Per::Hour => "hour",
            Per::Day => "day",
            Per::Week => "week",
            Per::Month => "month",
            Per::Year => "year",
        }
    }

    pub(super) fn symbol(self) -> &'static str {
        match self {
            Per::Millisecond => "ms",
            Per::Second => "s",
            Per::Minute => "min",
            Per::Hour => "h",
            Per::Day => "d",
            Per::Week => "wk",
            Per::Month => "mo",
            Per::Year => "yr",
        }
    }
}

/// A speed: a length over a time, or a knot.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Speed {
    Per(Length, Per),
    Knot,
}
