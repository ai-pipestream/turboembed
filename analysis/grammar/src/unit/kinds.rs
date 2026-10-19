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
