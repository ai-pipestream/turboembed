//! Clock times (`07:30`, `at 5`, `7:56:30 pm`) and parts of the day
//! (`morning`, `evening`, `lunch`).

/// Which half of the day a clock time is in.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Meridiem {
    /// Not said: `at 5` is 05:00 or 17:00.
    #[default]
    Unknown,
    Am,
    Pm,
    /// A 24-hour time: `17:30`, `0730` read as written.
    Day,
}

/// How much of a clock time is written, which its TIMEX3 value keeps:
/// `T07`, `T07:30`, `T07:56:30`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Shown {
    #[default]
    Hour,
    Minute,
    Second,
}

/// A time of day as written.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Clock {
    /// 0 to 23 for a 24-hour time, 1 to 12 (or 0) otherwise.
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub shown: Shown,
    pub meridiem: Meridiem,
}

impl Clock {
    pub fn new(hour: u8, minute: u8, second: u8, shown: Shown, meridiem: Meridiem) -> Option<Clock> {
        let max_hour = if matches!(meridiem, Meridiem::Am | Meridiem::Pm) { 12 } else { 24 };
        (hour <= max_hour && minute < 60 && second < 60).then_some(Clock { hour, minute, second, shown, meridiem })
    }

    /// The hours of the day, 0 to 23, the clock reads as: one, or two when
    /// the half of the day is open (`5` is 5 and 17, `12` is 12 and 0).
    pub fn hours(self) -> ([u8; 2], usize) {
        let h = self.hour % 24;
        match self.meridiem {
            Meridiem::Day => ([h, h], 1),
            Meridiem::Am => ([h % 12, 0], 1),
            Meridiem::Pm => ([h % 12 + 12, 0], 1),
            Meridiem::Unknown if (1..=12).contains(&h) => {
                let (a, b) = if h == 12 { (12, 0) } else { (h, h + 12) };
                ([a, b], 2)
            }
            Meridiem::Unknown => ([h, h], 1),
        }
    }

    /// Seconds after midnight at `hour`.
    pub fn seconds_at(self, hour: u8) -> u32 {
        u32::from(hour) * 3600 + u32::from(self.minute) * 60 + u32::from(self.second)
    }

    /// The same clock in the half of the day `m` says.
    pub fn with(self, m: Meridiem) -> Clock {
        if self.meridiem != Meridiem::Unknown || self.hour > 12 {
            return self;
        }
        Clock { meridiem: m, ..self }
    }
}

/// A part of the day, with its TIMEX3 code and hours.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    Dawn,
    Morning,
    Afternoon,
    Evening,
    Night,
    Daytime,
    NightTime,
    Breakfast,
    Lunch,
    Dinner,
}

impl Part {
    /// `DA`, `MO`, `AF`, `EV`, `NI`, `DT`, `NT`, `MEB`, `MEL` or `MED`.
    pub fn code(self) -> &'static str {
        match self {
            Part::Dawn => "DA",
            Part::Morning => "MO",
            Part::Afternoon => "AF",
            Part::Evening => "EV",
            Part::Night => "NI",
            Part::Daytime => "DT",
            Part::NightTime => "NT",
            Part::Breakfast => "MEB",
            Part::Lunch => "MEL",
            Part::Dinner => "MED",
        }
    }

    /// Its start and end in seconds after midnight, as Recognizers-Text
    /// sets them: the night ends at 23:59:59.
    pub fn seconds(self) -> (u32, u32) {
        let h = |x: u32| x * 3600;
        match self {
            Part::Dawn => (h(4), h(8)),
            Part::Morning | Part::Breakfast => (h(8), h(12)),
            Part::Afternoon => (h(12), h(16)),
            Part::Evening | Part::Dinner => (h(16), h(20)),
            Part::Night => (h(20), h(24) - 1),
            Part::Daytime => (h(8), h(18)),
            Part::NightTime => (0, h(8)),
            Part::Lunch => (h(11), h(13)),
        }
    }

    /// The half of the day an open clock time in this part is in, with
    /// the hour from which a night time is still the early morning.
    pub fn meridiem(self, hour: u8) -> Meridiem {
        match self {
            Part::Dawn | Part::Morning | Part::Breakfast | Part::NightTime => Meridiem::Am,
            Part::Night if (1..=4).contains(&hour) => Meridiem::Am,
            Part::Lunch if hour >= 11 => Meridiem::Am,
            _ => Meridiem::Pm,
        }
    }
}

/// The time of day a point names.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TimeOfDay {
    #[default]
    None,
    Clock(Clock),
    Part(Part),
    /// The reference instant itself: `now`.
    Now,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clocks_read_as_one_or_two_hours() {
        let c = |h, m| Clock::new(h, 0, 0, Shown::Hour, m).unwrap();
        assert_eq!(c(5, Meridiem::Unknown).hours(), ([5, 17], 2));
        assert_eq!(c(12, Meridiem::Unknown).hours(), ([12, 0], 2));
        assert_eq!(c(0, Meridiem::Unknown).hours().1, 1);
        assert_eq!(c(12, Meridiem::Am).hours().0[0], 0);
        assert_eq!(c(12, Meridiem::Pm).hours().0[0], 12);
        assert_eq!(c(7, Meridiem::Pm).hours().0[0], 19);
        assert_eq!(c(17, Meridiem::Day).hours(), ([17, 17], 1));
        assert_eq!(c(5, Meridiem::Unknown).with(Meridiem::Pm).hours().0[0], 17);
        assert_eq!(c(17, Meridiem::Day).with(Meridiem::Am).hours().0[0], 17);
        assert!(Clock::new(13, 0, 0, Shown::Hour, Meridiem::Pm).is_none());
        assert!(Clock::new(9, 60, 0, Shown::Minute, Meridiem::Day).is_none());
        assert_eq!(c(7, Meridiem::Day).seconds_at(7), 25_200);
    }
}
