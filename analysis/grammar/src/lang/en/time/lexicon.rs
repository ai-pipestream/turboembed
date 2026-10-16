//! The English words of the time grammar: months, weekdays, units, parts
//! of the day, relative words, seasons, holidays and set words.

use crate::engine::{Dim, Entry};
use crate::time::spec::Holiday;

use super::{
    ABBR, AMBIG, DAYWORD, DECADEW, HOLIDAY, MONTH, PART, PLURAL, RELW, SEASON, SETWORD, TONIGHT, UNIT, WEEKDAY,
};

const fn e(word: &'static str, dim: Dim, value: i64, flags: u8) -> Entry {
    Entry { word, dim, value, flags }
}

/// What a `RELW` word means, in its flags.
pub(crate) mod rel {
    pub const THIS: u8 = 1;
    pub const NEXT: u8 = 2;
    pub const LAST: u8 = 3;
    pub const PAST: u8 = 4;
    pub const COMING: u8 = 5;
    pub const UPCOMING: u8 = 6;
}

/// A unit word's value: the [`crate::time::Unit`] index, and how many of
/// it the word stands for (a decade is ten years) times 256.
pub(crate) const fn unit(u: u8, times: i64) -> i64 {
    u as i64 | times << 8
}

// Unit indexes, in the order of `crate::time::Unit`.
pub(crate) const YEAR: u8 = 0;
pub(crate) const MONTH_U: u8 = 1;
pub(crate) const WEEK: u8 = 2;
pub(crate) const WEEKEND: u8 = 3;
pub(crate) const BUSINESS_DAY: u8 = 4;
pub(crate) const WEEKDAY_U: u8 = 5;
pub(crate) const DAY: u8 = 6;
pub(crate) const HOUR: u8 = 7;
pub(crate) const MINUTE: u8 = 8;
pub(crate) const SECOND: u8 = 9;

/// A unit word that is also a part of the day: `two nights`.
pub(crate) const NIGHT: u8 = 1 << 6;

/// Holidays a single word names, by `HOLIDAY` value.
pub(crate) const HOLIDAYS: &[Holiday] = &[
    Holiday::Fixed { month: 12, day: 25 },
    Holiday::Easter { days: 0 },
    Holiday::Fixed { month: 10, day: 31 },
    Holiday::Nth { month: 11, weekday: 4, n: 4, days: 0 },
    Holiday::Fixed { month: 6, day: 19 },
    Holiday::Fixed { month: 12, day: 24 },
    Holiday::Fixed { month: 1, day: 1 },
    Holiday::Fixed { month: 7, day: 4 },
    Holiday::Nth { month: 9, weekday: 1, n: 1, days: 0 },
    Holiday::Nth { month: 11, weekday: 4, n: 4, days: 1 },
    Holiday::Fixed { month: 4, day: 22 },
    Holiday::Easter { days: 1 },
    Holiday::Fixed { month: 3, day: 17 },
    Holiday::Fixed { month: 5, day: 1 },
    Holiday::Nth { month: 5, weekday: 1, n: 0, days: 0 },
    Holiday::Fixed { month: 2, day: 14 },
    Holiday::Easter { days: -2 },
    Holiday::Nth { month: 5, weekday: 7, n: 2, days: 0 },
    Holiday::Nth { month: 6, weekday: 7, n: 3, days: 0 },
    Holiday::Fixed { month: 12, day: 31 },
    Holiday::Fixed { month: 11, day: 11 },
];

// Indexes into `HOLIDAYS` for the names the rules spell out.
pub(crate) const CHRISTMAS: i64 = 0;
pub(crate) const EASTER: i64 = 1;
pub(crate) const JUNETEENTH: i64 = 4;
pub(crate) const CHRISTMAS_EVE: i64 = 5;
pub(crate) const NEW_YEAR: i64 = 6;
pub(crate) const INDEPENDENCE: i64 = 7;
pub(crate) const LABOR: i64 = 8;
pub(crate) const BLACK_FRIDAY: i64 = 9;
pub(crate) const EARTH: i64 = 10;
pub(crate) const EASTER_MONDAY: i64 = 11;
pub(crate) const PATRICK: i64 = 12;
pub(crate) const WORKERS: i64 = 13;
pub(crate) const MEMORIAL: i64 = 14;
pub(crate) const VALENTINE: i64 = 15;
pub(crate) const GOOD_FRIDAY: i64 = 16;
pub(crate) const MOTHERS: i64 = 17;
pub(crate) const FATHERS: i64 = 18;
pub(crate) const NEW_YEARS_EVE: i64 = 19;
pub(crate) const VETERANS: i64 = 20;

/// A set word's value: a unit index and an amount in thousandths, as
/// `amount << 8 | unit`.
pub(crate) const fn every(u: u8, milli: i64) -> i64 {
    milli << 8 | u as i64
}

pub const ENTRIES: &[Entry] = &[
    e("january", MONTH, 1, 0),
    e("jan", MONTH, 1, ABBR),
    e("february", MONTH, 2, 0),
    e("feb", MONTH, 2, ABBR),
    e("march", MONTH, 3, AMBIG),
    e("mar", MONTH, 3, ABBR),
    e("april", MONTH, 4, 0),
    e("apr", MONTH, 4, ABBR),
    e("may", MONTH, 5, AMBIG),
    e("june", MONTH, 6, 0),
    e("jun", MONTH, 6, ABBR),
    e("july", MONTH, 7, 0),
    e("jul", MONTH, 7, ABBR),
    e("august", MONTH, 8, 0),
    e("aug", MONTH, 8, ABBR),
    e("september", MONTH, 9, 0),
    e("sep", MONTH, 9, ABBR),
    e("sept", MONTH, 9, ABBR),
    e("october", MONTH, 10, 0),
    e("oct", MONTH, 10, ABBR),
    e("november", MONTH, 11, 0),
    e("nov", MONTH, 11, ABBR),
    e("december", MONTH, 12, 0),
    e("dec", MONTH, 12, ABBR),
    e("monday", WEEKDAY, 1, 0),
    e("mon", WEEKDAY, 1, ABBR),
    e("mondays", WEEKDAY, 1, PLURAL),
    e("tuesday", WEEKDAY, 2, 0),
    e("tue", WEEKDAY, 2, ABBR),
    e("tues", WEEKDAY, 2, ABBR),
    e("tuesdays", WEEKDAY, 2, PLURAL),
    e("wednesday", WEEKDAY, 3, 0),
    e("wed", WEEKDAY, 3, ABBR),
    e("wednesdays", WEEKDAY, 3, PLURAL),
    e("thursday", WEEKDAY, 4, 0),
    e("thu", WEEKDAY, 4, ABBR),
    e("thur", WEEKDAY, 4, ABBR),
    e("thurs", WEEKDAY, 4, ABBR),
    e("thursdays", WEEKDAY, 4, PLURAL),
    e("friday", WEEKDAY, 5, 0),
    e("fri", WEEKDAY, 5, ABBR),
    e("fridays", WEEKDAY, 5, PLURAL),
    e("saturday", WEEKDAY, 6, 0),
    e("sat", WEEKDAY, 6, ABBR),
    e("saturdays", WEEKDAY, 6, PLURAL),
    e("sunday", WEEKDAY, 7, 0),
    e("sun", WEEKDAY, 7, ABBR | AMBIG),
    e("sundays", WEEKDAY, 7, PLURAL),
    e("year", UNIT, unit(YEAR, 1), 0),
    e("years", UNIT, unit(YEAR, 1), PLURAL),
    e("yr", UNIT, unit(YEAR, 1), 0),
    e("yrs", UNIT, unit(YEAR, 1), PLURAL),
    e("decade", UNIT, unit(YEAR, 10), 0),
    e("decades", UNIT, unit(YEAR, 10), PLURAL),
    e("month", UNIT, unit(MONTH_U, 1), 0),
    e("months", UNIT, unit(MONTH_U, 1), PLURAL),
    e("quarter", UNIT, unit(MONTH_U, 3), 0),
    e("quarters", UNIT, unit(MONTH_U, 3), PLURAL),
    e("week", UNIT, unit(WEEK, 1), 0),
    e("weeks", UNIT, unit(WEEK, 1), PLURAL),
    e("wk", UNIT, unit(WEEK, 1), 0),
    e("wks", UNIT, unit(WEEK, 1), PLURAL),
    e("fortnight", UNIT, unit(WEEK, 2), 0),
    e("fortnights", UNIT, unit(WEEK, 2), PLURAL),
    e("weekend", UNIT, unit(WEEKEND, 1), 0),
    e("weekends", UNIT, unit(WEEKEND, 1), PLURAL),
    e("weekday", UNIT, unit(WEEKDAY_U, 1), 0),
    e("weekdays", UNIT, unit(WEEKDAY_U, 1), PLURAL),
    e("day", UNIT, unit(DAY, 1), 0),
    e("days", UNIT, unit(DAY, 1), PLURAL),
    e("night", UNIT, unit(DAY, 1), NIGHT),
    e("nights", UNIT, unit(DAY, 1), NIGHT | PLURAL),
    e("hour", UNIT, unit(HOUR, 1), 0),
    e("hours", UNIT, unit(HOUR, 1), PLURAL),
    e("hr", UNIT, unit(HOUR, 1), 0),
    e("hrs", UNIT, unit(HOUR, 1), PLURAL),
    e("minute", UNIT, unit(MINUTE, 1), 0),
    e("minutes", UNIT, unit(MINUTE, 1), PLURAL),
    e("min", UNIT, unit(MINUTE, 1), 0),
    e("mins", UNIT, unit(MINUTE, 1), PLURAL),
    e("second", UNIT, unit(SECOND, 1), 0),
    e("seconds", UNIT, unit(SECOND, 1), PLURAL),
    e("sec", UNIT, unit(SECOND, 1), 0),
    e("secs", UNIT, unit(SECOND, 1), PLURAL),
    // Parts of the day, by `Part` index.
    e("morning", PART, 0, 0),
    e("mornings", PART, 0, PLURAL),
    e("afternoon", PART, 1, 0),
    e("afternoons", PART, 1, PLURAL),
    e("evening", PART, 2, 0),
    e("evenings", PART, 2, PLURAL),
    e("night", PART, 3, 0),
    e("nights", PART, 3, PLURAL),
    e("daytime", PART, 4, 0),
    e("nighttime", PART, 5, 0),
    e("breakfast", PART, 6, 0),
    e("lunch", PART, 7, 0),
    e("lunchtime", PART, 7, 0),
    e("dinner", PART, 8, 0),
    e("dinnertime", PART, 8, 0),
    e("today", DAYWORD, 0, 0),
    e("otd", DAYWORD, 0, 0),
    e("tomorrow", DAYWORD, 1, 0),
    e("tmrw", DAYWORD, 1, 0),
    e("tmr", DAYWORD, 1, 0),
    e("yesterday", DAYWORD, -1, 0),
    e("tonight", DAYWORD, 0, TONIGHT),
    e("tonite", DAYWORD, 0, TONIGHT),
    e("this", RELW, 0, rel::THIS),
    e("current", RELW, 0, rel::THIS),
    e("next", RELW, 1, rel::NEXT),
    e("following", RELW, 1, rel::NEXT),
    e("coming", RELW, 1, rel::COMING),
    e("upcoming", RELW, 1, rel::UPCOMING),
    e("last", RELW, -1, rel::LAST),
    e("previous", RELW, -1, rel::LAST),
    e("prior", RELW, -1, rel::LAST),
    e("past", RELW, -1, rel::PAST),
    e("spring", SEASON, 0, 0),
    e("summer", SEASON, 1, 0),
    e("fall", SEASON, 2, AMBIG),
    e("autumn", SEASON, 2, 0),
    e("winter", SEASON, 3, 0),
    e("christmas", HOLIDAY, CHRISTMAS, 0),
    e("xmas", HOLIDAY, CHRISTMAS, 0),
    e("easter", HOLIDAY, EASTER, 0),
    e("halloween", HOLIDAY, 2, 0),
    e("thanksgiving", HOLIDAY, 3, 0),
    e("juneteenth", HOLIDAY, JUNETEENTH, 0),
    e("weekly", SETWORD, every(WEEK, 1000), 0),
    e("daily", SETWORD, every(DAY, 1000), 0),
    e("everyday", SETWORD, every(DAY, 1000), 0),
    e("nightly", SETWORD, every(DAY, 1000), 0),
    e("hourly", SETWORD, every(HOUR, 1000), 0),
    e("monthly", SETWORD, every(MONTH_U, 1000), 0),
    e("quarterly", SETWORD, every(MONTH_U, 3000), 0),
    e("annually", SETWORD, every(YEAR, 1000), 0),
    e("annual", SETWORD, every(YEAR, 1000), 0),
    e("yearly", SETWORD, every(YEAR, 1000), 0),
    e("semiannual", SETWORD, every(YEAR, 500), 0),
    e("semiannually", SETWORD, every(YEAR, 500), 0),
    e("biannual", SETWORD, every(YEAR, 500), 0),
    e("bimonthly", SETWORD, every(MONTH_U, 2000), 0),
    e("biweekly", SETWORD, every(WEEK, 2000), 0),
    e("fortnightly", SETWORD, every(WEEK, 2000), 0),
    e("twenties", DECADEW, 20, 0),
    e("thirties", DECADEW, 30, 0),
    e("forties", DECADEW, 40, 0),
    e("fifties", DECADEW, 50, 0),
    e("sixties", DECADEW, 60, 0),
    e("seventies", DECADEW, 70, 0),
    e("eighties", DECADEW, 80, 0),
    e("nineties", DECADEW, 90, 0),
];
