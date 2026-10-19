//! Lengths of time as a text states them: `3 hours 30 minutes`, `3.5
//! years`, `two weekends`, each unit kept as written so the TIMEX3 value
//! reads `PT3H30M`, `P3.5Y`, `P2WE`.

use std::fmt;

/// A unit a duration counts in.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Unit {
    Year,
    Month,
    Week,
    /// A Saturday and a Sunday: two days.
    Weekend,
    /// A working day, Monday to Friday.
    BusinessDay,
    /// A day of the working week, as a set repeats: `every weekday`.
    Weekday,
    Day,
    Hour,
    Minute,
    #[default]
    Second,
}

impl Unit {
    /// Its length in seconds, with a year of 365 days and a month of 30
    /// as Recognizers-Text counts them.
    pub fn seconds(self) -> i64 {
        match self {
            Unit::Year => 365 * 86_400,
            Unit::Month => 30 * 86_400,
            Unit::Week => 7 * 86_400,
            Unit::Weekend => 2 * 86_400,
            Unit::BusinessDay | Unit::Weekday | Unit::Day => 86_400,
            Unit::Hour => 3_600,
            Unit::Minute => 60,
            Unit::Second => 1,
        }
    }

    /// Whether TIMEX3 writes the unit after the `T`.
    pub fn is_clock(self) -> bool {
        matches!(self, Unit::Hour | Unit::Minute | Unit::Second)
    }

    fn letters(self) -> &'static str {
        match self {
            Unit::Year => "Y",
            Unit::Month => "M",
            Unit::Week => "W",
            Unit::Weekend => "WE",
            Unit::BusinessDay => "BD",
            Unit::Weekday => "WD",
            Unit::Day => "D",
            Unit::Hour => "H",
            Unit::Minute => "M",
            Unit::Second => "S",
        }
    }
}

/// Thousandths of a unit in one.
pub const MILLI: i32 = 1000;

/// Up to three amounts of distinct units, largest unit first, each in
/// thousandths (`3.5 years` is 3500 of [`Unit::Year`]). A negative amount
/// counts back: `2 days ago` shifts by minus two days.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Duration {
    parts: [(Unit, i32); 3],
    len: u8,
}

impl Duration {
    pub const ZERO: Duration = Duration { parts: [(Unit::Second, 0); 3], len: 0 };

    /// `milli` thousandths of `unit`.
    pub fn of(unit: Unit, milli: i32) -> Duration {
        Duration { parts: [(unit, milli), (Unit::Second, 0), (Unit::Second, 0)], len: 1 }
    }

    /// `n` whole units.
    pub fn units(unit: Unit, n: i32) -> Option<Duration> {
        Some(Duration::of(unit, n.checked_mul(MILLI)?))
    }

    pub fn parts(&self) -> &[(Unit, i32)] {
        &self.parts[..usize::from(self.len)]
    }

    pub fn is_zero(&self) -> bool {
        self.len == 0
    }

    /// The sum of two durations whose units do not repeat and which hold
    /// three parts at most between them, largest unit first: `1 hour and
    /// 30 minutes`.
    pub fn plus(&self, other: &Duration) -> Option<Duration> {
        let mut out = *self;
        for &(u, n) in other.parts() {
            if out.parts().iter().any(|&(v, _)| v == u) || out.len == 3 {
                return None;
            }
            out.parts[usize::from(out.len)] = (u, n);
            out.len += 1;
        }
        out.parts[..usize::from(out.len)].sort_by_key(|&(u, _)| u);
        Some(out)
    }

    /// The same length counted back.
    pub fn negated(&self) -> Duration {
        let mut out = *self;
        for p in &mut out.parts[..usize::from(out.len)] {
            p.1 = p.1.saturating_neg();
        }
        out
    }

    /// Every amount times `k`.
    pub fn times(&self, k: i32) -> Option<Duration> {
        let mut out = *self;
        for p in &mut out.parts[..usize::from(out.len)] {
            p.1 = p.1.checked_mul(k)?;
        }
        Some(out)
    }

    /// Its length in milliseconds, months and years counted as
    /// [`Unit::seconds`] counts them.
    pub fn millis(&self) -> i64 {
        self.parts().iter().fold(0i64, |acc, &(u, n)| acc.saturating_add(u.seconds().saturating_mul(i64::from(n))))
    }

    /// Whether every amount is a whole number of its unit.
    pub fn is_whole(&self) -> bool {
        self.parts().iter().all(|&(_, n)| n % MILLI == 0)
    }

    /// The finest unit, if any.
    pub fn finest(&self) -> Option<Unit> {
        self.parts().last().map(|&(u, _)| u)
    }
}

/// Writes a thousandths amount: `3`, `3.5`, `0.25`.
fn write_milli(f: &mut fmt::Formatter<'_>, n: i32) -> fmt::Result {
    let (whole, frac) = (n.unsigned_abs() / 1000, n.unsigned_abs() % 1000);
    if n < 0 {
        f.write_str("-")?;
    }
    if frac == 0 {
        return write!(f, "{whole}");
    }
    let digits = format!("{frac:03}");
    write!(f, "{whole}.{}", digits.trim_end_matches('0'))
}

/// Writes the TIMEX3 duration: `P3Y`, `PT1H30M`, `P1DT5H`, `P3.5Y`,
/// `P2WE`; `PT0S` for none.
impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("P")?;
        if self.is_zero() {
            return f.write_str("T0S");
        }
        let mut in_time = false;
        for &(u, n) in self.parts() {
            if u.is_clock() && !in_time {
                f.write_str("T")?;
                in_time = true;
            }
            write_milli(f, n)?;
            f.write_str(u.letters())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(unit: Unit, n: i32) -> Duration {
        Duration::units(unit, n).unwrap()
    }

    #[test]
    fn durations_are_written_as_timex() {
        assert_eq!(d(Unit::Hour, 3).to_string(), "PT3H");
        assert_eq!(Duration::of(Unit::Year, 3500).to_string(), "P3.5Y");
        assert_eq!(Duration::of(Unit::Second, 123_450).to_string(), "PT123.45S");
        assert_eq!(d(Unit::Hour, 1).plus(&d(Unit::Minute, 30)).unwrap().to_string(), "PT1H30M");
        assert_eq!(d(Unit::Hour, 5).plus(&d(Unit::Day, 1)).unwrap().to_string(), "P1DT5H");
        let p3m2w2d = d(Unit::Month, 3).plus(&d(Unit::Week, 2)).unwrap().plus(&d(Unit::Day, 2)).unwrap();
        assert_eq!(p3m2w2d.to_string(), "P3M2W2D");
        assert_eq!(d(Unit::Weekend, 3).to_string(), "P3WE");
        assert_eq!(d(Unit::BusinessDay, 4).to_string(), "P4BD");
        assert_eq!(Duration::of(Unit::Year, 250).to_string(), "P0.25Y");
        assert_eq!(Duration::ZERO.to_string(), "PT0S");
    }

    #[test]
    fn lengths_and_arithmetic() {
        assert_eq!(Duration::of(Unit::Year, 3500).millis() / 1000, 110_376_000);
        assert_eq!(d(Unit::Weekend, 3).millis() / 1000, 518_400);
        assert_eq!(Duration::of(Unit::Second, 123_450).millis(), 123_450);
        assert!(d(Unit::Day, 1).plus(&d(Unit::Day, 2)).is_none());
        let three = d(Unit::Year, 1).plus(&d(Unit::Month, 1)).unwrap().plus(&d(Unit::Day, 1)).unwrap();
        assert!(three.plus(&d(Unit::Hour, 1)).is_none());
        assert_eq!(d(Unit::Day, 2).negated().parts(), &[(Unit::Day, -2000)]);
        assert_eq!(Duration::of(Unit::Day, i32::MIN).negated().parts(), &[(Unit::Day, i32::MAX)]);
        assert_eq!(d(Unit::Week, 2).times(3).unwrap(), d(Unit::Week, 6));
        assert!(d(Unit::Day, 2).is_whole() && !Duration::of(Unit::Hour, 1500).is_whole());
        assert_eq!(three.finest(), Some(Unit::Day));
        assert_eq!(Duration::ZERO.finest(), None);
    }
}
