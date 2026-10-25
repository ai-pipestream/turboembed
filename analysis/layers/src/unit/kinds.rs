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

    pub(super) fn name(self) -> &'static str {
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

    pub(super) fn symbol(self) -> &'static str {
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

/// A duration unit for measured time quantities (`500 ms`, `12 hr`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DurationUnit {
    Nanosecond,
    Microsecond,
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
}

impl DurationUnit {
    pub(super) fn name(self) -> &'static str {
        match self {
            DurationUnit::Nanosecond => "Nanosecond",
            DurationUnit::Microsecond => "Microsecond",
            DurationUnit::Millisecond => "Millisecond",
            DurationUnit::Second => "Second",
            DurationUnit::Minute => "Minute",
            DurationUnit::Hour => "Hour",
            DurationUnit::Day => "Day",
        }
    }

    pub(super) fn symbol(self) -> &'static str {
        match self {
            DurationUnit::Nanosecond => "ns",
            DurationUnit::Microsecond => "µs",
            DurationUnit::Millisecond => "ms",
            DurationUnit::Second => "s",
            DurationUnit::Minute => "min",
            DurationUnit::Hour => "hr",
            DurationUnit::Day => "d",
        }
    }
}

/// A frequency unit (`Hz`, `kHz`, `MHz`, `GHz`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Frequency {
    Hertz,
    Kilohertz,
    Megahertz,
    Gigahertz,
}

impl Frequency {
    pub(super) fn name(self) -> &'static str {
        match self {
            Frequency::Hertz => "Hertz",
            Frequency::Kilohertz => "Kilohertz",
            Frequency::Megahertz => "Megahertz",
            Frequency::Gigahertz => "Gigahertz",
        }
    }

    pub(super) fn symbol(self) -> &'static str {
        match self {
            Frequency::Hertz => "Hz",
            Frequency::Kilohertz => "kHz",
            Frequency::Megahertz => "MHz",
            Frequency::Gigahertz => "GHz",
        }
    }
}

/// A power unit (`W`, `kW`, `MW`, `GW`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Power {
    Watt,
    Kilowatt,
    Megawatt,
    Gigawatt,
}

impl Power {
    pub(super) fn name(self) -> &'static str {
        match self {
            Power::Watt => "Watt",
            Power::Kilowatt => "Kilowatt",
            Power::Megawatt => "Megawatt",
            Power::Gigawatt => "Gigawatt",
        }
    }

    pub(super) fn symbol(self) -> &'static str {
        match self {
            Power::Watt => "W",
            Power::Kilowatt => "kW",
            Power::Megawatt => "MW",
            Power::Gigawatt => "GW",
        }
    }
}

/// An energy unit (`J`, `kJ`, `Wh`, `kWh`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Energy {
    Joule,
    Kilojoule,
    WattHour,
    KilowattHour,
}

impl Energy {
    pub(super) fn name(self) -> &'static str {
        match self {
            Energy::Joule => "Joule",
            Energy::Kilojoule => "Kilojoule",
            Energy::WattHour => "Watt-hour",
            Energy::KilowattHour => "Kilowatt-hour",
        }
    }

    pub(super) fn symbol(self) -> &'static str {
        match self {
            Energy::Joule => "J",
            Energy::Kilojoule => "kJ",
            Energy::WattHour => "Wh",
            Energy::KilowattHour => "kWh",
        }
    }
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
