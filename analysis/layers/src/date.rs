//! A calendar date in the proleptic Gregorian calendar, as Java's
//! `LocalDate` holds one, with its ISO-8601 reading and writing.

use std::fmt;

/// A date between `-999999999-01-01` and `+999999999-12-31`, the range of
/// `LocalDate`. Dates order by time.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Date {
    year: i32,
    month: u8,
    day: u8,
}

/// The years `LocalDate` holds.
const MIN_YEAR: i32 = -999_999_999;
const MAX_YEAR: i32 = 999_999_999;

impl Date {
    /// `LocalDate.MIN`.
    pub const MIN: Date = Date { year: MIN_YEAR, month: 1, day: 1 };
    /// `LocalDate.MAX`.
    pub const MAX: Date = Date { year: MAX_YEAR, month: 12, day: 31 };

    /// The date, when the year is in range and the day exists in that
    /// month: `LocalDate.of`.
    pub fn new(year: i32, month: u8, day: u8) -> Option<Date> {
        let valid = (MIN_YEAR..=MAX_YEAR).contains(&year)
            && (1..=12).contains(&month)
            && day >= 1
            && day <= days_in_month(year, month);
        valid.then_some(Date { year, month, day })
    }

    pub fn year(self) -> i32 {
        self.year
    }

    pub fn month(self) -> u8 {
        self.month
    }

    pub fn day(self) -> u8 {
        self.day
    }

    /// Reads a date as `LocalDate.parse` does with its default formatter,
    /// `ISO_LOCAL_DATE`, which is strict: `yyyy-MM-dd` in ASCII digits,
    /// where a year of more than four digits takes a `+`, a year before
    /// year 0 takes a `-`, a year of four digits takes no `+` and `-0000`
    /// is refused; the month and day are two digits each, and the date
    /// must exist.
    pub fn parse(s: &str) -> Option<Date> {
        let b = s.as_bytes();
        let (sign, digits_start) = match b.first() {
            Some(b'+') => (Some(false), 1),
            Some(b'-') => (Some(true), 1),
            _ => (None, 0),
        };
        let year_digits = b[digits_start..].iter().take(10).take_while(|c| c.is_ascii_digit()).count();
        if year_digits < 4 {
            return None;
        }
        let year_end = digits_start + year_digits;
        let magnitude = b[digits_start..year_end].iter().fold(0i64, |v, &d| v * 10 + i64::from(d - b'0'));
        let year = match sign {
            Some(true) if magnitude == 0 => return None,
            Some(true) => -magnitude,
            Some(false) if year_digits <= 4 => return None,
            None if year_digits > 4 => return None,
            _ => magnitude,
        };
        let rest = &b[year_end..];
        if rest.len() != 6 || rest[0] != b'-' || rest[3] != b'-' {
            return None;
        }
        let two = |at: usize| -> Option<u8> {
            let (h, l) = (rest[at], rest[at + 1]);
            (h.is_ascii_digit() && l.is_ascii_digit()).then(|| (h - b'0') * 10 + (l - b'0'))
        };
        Date::new(i32::try_from(year).ok()?, two(1)?, two(4)?)
    }

    /// Days since 1970-01-01: `LocalDate.toEpochDay`.
    pub fn epoch_day(self) -> i64 {
        // Howard Hinnant's days_from_civil, with March as the first month
        // so that a leap day ends its year.
        let y = i64::from(self.year) - i64::from(self.month <= 2);
        let era = y.div_euclid(400);
        let yoe = y.rem_euclid(400);
        let m = i64::from(self.month);
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(self.day) - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    /// The date `epoch_day` days after 1970-01-01, when it is in range:
    /// `LocalDate.ofEpochDay`.
    pub fn from_epoch_day(epoch_day: i64) -> Option<Date> {
        if epoch_day < Date::MIN.epoch_day() || epoch_day > Date::MAX.epoch_day() {
            return None;
        }
        let z = epoch_day + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = (doy - (153 * mp + 2) / 5 + 1) as u8;
        let month = if mp < 10 { mp + 3 } else { mp - 9 } as u8;
        let year = yoe + era * 400 + i64::from(month <= 2);
        Some(Date { year: year as i32, month, day })
    }

    /// The date `days` later, or earlier for a negative count, when it is
    /// in range: `LocalDate.plusDays`.
    pub fn plus_days(self, days: i64) -> Option<Date> {
        Date::from_epoch_day(self.epoch_day().checked_add(days)?)
    }

    /// `LocalDate.plusWeeks`: seven days per week.
    pub fn plus_weeks(self, weeks: i64) -> Option<Date> {
        self.plus_days(weeks.checked_mul(7)?)
    }

    /// The date `months` later, or earlier, when its year is in range:
    /// `LocalDate.plusMonths`. A day past the end of the new month becomes
    /// its last day, so 2024-01-31 plus one month is 2024-02-29.
    pub fn plus_months(self, months: i64) -> Option<Date> {
        let count = i64::from(self.year) * 12 + i64::from(self.month - 1);
        let shifted = count.checked_add(months)?;
        let year = i32::try_from(shifted.div_euclid(12)).ok()?;
        let month = shifted.rem_euclid(12) as u8 + 1;
        Date::previous_valid(year, month, self.day)
    }

    /// The date `years` later, or earlier, when its year is in range:
    /// `LocalDate.plusYears`. A leap day in a year without one becomes
    /// February 28.
    pub fn plus_years(self, years: i64) -> Option<Date> {
        let year = i32::try_from(i64::from(self.year).checked_add(years)?).ok()?;
        Date::previous_valid(year, self.month, self.day)
    }

    /// `LocalDate.resolvePreviousValid`: the day, or the month's last day
    /// when the month is shorter.
    fn previous_valid(year: i32, month: u8, day: u8) -> Option<Date> {
        if !(MIN_YEAR..=MAX_YEAR).contains(&year) {
            return None;
        }
        Some(Date { year, month, day: day.min(days_in_month(year, month)) })
    }

    /// The day of the week, 1 for Monday to 7 for Sunday:
    /// `getDayOfWeek().getValue()`.
    pub fn day_of_week(self) -> u8 {
        (self.epoch_day() + 3).rem_euclid(7) as u8 + 1
    }

    /// The day of the year, from 1: `getDayOfYear()`.
    pub fn day_of_year(self) -> u16 {
        let before: u16 = (1..self.month).map(|m| u16::from(days_in_month(self.year, m))).sum();
        before + u16::from(self.day)
    }

    /// The quarter of the year, 1 to 4: `IsoFields.QUARTER_OF_YEAR`.
    pub fn quarter(self) -> u8 {
        (self.month - 1) / 3 + 1
    }

    /// The ISO 8601 week-based year and the week in it, 1 to 53:
    /// `IsoFields.WEEK_BASED_YEAR` and `WEEK_OF_WEEK_BASED_YEAR`. Week 1
    /// is the week, Monday to Sunday, that holds the year's first
    /// Thursday, so 2021-01-01 is in week 53 of 2020 and 2020-12-31 in
    /// week 53 of 2020. The week-based year of a day near the ends of the
    /// `LocalDate` range can be one outside it.
    pub fn iso_week(self) -> (i32, u8) {
        // Days from the Monday of the year's first week to this day. The
        // first week's Monday is between December 29 and January 4.
        let doy = i32::from(self.day_of_year());
        let dow = i32::from(self.day_of_week());
        let week = (doy - dow + 10).div_euclid(7);
        if week < 1 {
            return (self.year - 1, weeks_in_year(self.year - 1));
        }
        if week > i32::from(weeks_in_year(self.year)) {
            return (self.year + 1, 1);
        }
        (self.year, week as u8)
    }
}

/// Writes the date as `LocalDate.toString` does: `2026-07-14`, `-0001-01-01`
/// for a year before year 0 and `+10000-01-01` past year 9999.
impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_year(f, self.year)?;
        write!(f, "-{:02}-{:02}", self.month, self.day)
    }
}

/// Writes a year as `java.time` does, with at least four digits: a `-`
/// before a year below zero (`-0001`) and a `+` past 9999 (`+10000`).
/// This is `LocalDate.toString` and a formatter's `appendValue(field, 4,
/// 10, SignStyle.EXCEEDS_PAD)`.
pub fn write_year(f: &mut impl fmt::Write, year: i32) -> fmt::Result {
    if year < 0 {
        write!(f, "-{:04}", year.unsigned_abs())
    } else if year > 9999 {
        write!(f, "+{year}")
    } else {
        write!(f, "{year:04}")
    }
}

fn is_leap(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// How many ISO weeks a week-based year has: 53 when it starts on a
/// Thursday, or on a Wednesday in a leap year, 52 otherwise.
fn weeks_in_year(year: i32) -> u8 {
    // January 1 is day 1, so its day of the week follows from any date.
    let jan1 = Date { year, month: 1, day: 1 };
    let dow = (jan1.epoch_day() + 3).rem_euclid(7) + 1;
    if dow == 4 || (dow == 3 && is_leap(year)) { 53 } else { 52 }
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap_or_else(|| panic!("{s:?}"))
    }

    #[test]
    fn iso_dates() {
        assert_eq!(d("2026-07-14"), Date::new(2026, 7, 14).unwrap());
        assert_eq!(d("2024-02-29").day(), 29);
        assert_eq!(d("0000-01-01").year(), 0);
        assert_eq!(d("-0001-12-31").year(), -1);
        assert_eq!(d("+10000-01-01").year(), 10000);
        assert_eq!(d("+00002-01-01").year(), 2);
        assert_eq!(d("+999999999-12-31"), Date::MAX);
        assert_eq!(d("-999999999-01-01"), Date::MIN);
        for bad in [
            "",
            "yesterday",
            "2026-7-14",
            "2026-07-1",
            "2026-07-140",
            "26-07-14",
            "+2026-07-14",
            "20260-07-14",
            "-0000-01-01",
            "2023-02-29",
            "2026-13-01",
            "2026-00-01",
            "2026-04-31",
            "2026-07-00",
            "2026/07/14",
            " 2026-07-14",
            "2026-07-14 ",
            "+1000000000-01-01",
            "-1000000000-01-01",
            "2026-+7-14",
            "\u{0662}026-07-14",
        ] {
            assert_eq!(Date::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn written_as_local_date_writes_them() {
        for s in ["2026-07-14", "0000-01-01", "-0001-12-31", "0999-01-01", "+10000-01-01", "-12345-06-30"] {
            assert_eq!(d(s).to_string(), s);
        }
        assert_eq!(Date::MAX.to_string(), "+999999999-12-31");
        assert_eq!(Date::MIN.to_string(), "-999999999-01-01");
    }

    #[test]
    fn epoch_days() {
        assert_eq!(d("1970-01-01").epoch_day(), 0);
        assert_eq!(d("2026-07-14").epoch_day(), 20648);
        assert_eq!(d("1969-12-31").epoch_day(), -1);
        assert_eq!(d("2000-03-01").epoch_day(), 11017);
        // LocalDate.MIN.toEpochDay() and LocalDate.MAX.toEpochDay().
        assert_eq!(Date::MIN.epoch_day(), -365243219162);
        assert_eq!(Date::MAX.epoch_day(), 365241780471);
        for s in ["2024-02-29", "2023-03-01", "-0400-02-29", "1600-12-31"] {
            assert_eq!(Date::from_epoch_day(d(s).epoch_day()), Some(d(s)));
        }
        assert_eq!(Date::MAX.plus_days(1), None);
        assert_eq!(Date::MIN.plus_days(-1), None);
        assert_eq!(Date::MAX.plus_days(-7).unwrap().to_string(), "+999999999-12-24");
        assert_eq!(d("2026-07-10").plus_days(7), Some(d("2026-07-17")));
        assert!(d("2026-07-10") < d("2026-07-11") && d("-0001-12-31") < d("0000-01-01"));
    }

    #[test]
    fn calendar_fields() {
        // 1970-01-01 was a Thursday.
        assert_eq!(d("1970-01-01").day_of_week(), 4);
        assert_eq!(d("2026-07-13").day_of_week(), 1);
        assert_eq!(d("2026-07-19").day_of_week(), 7);
        assert_eq!(d("2024-12-31").day_of_year(), 366);
        assert_eq!(d("2026-03-01").day_of_year(), 60);
        assert_eq!((d("2026-03-31").quarter(), d("2026-04-01").quarter(), d("2026-12-31").quarter()), (1, 2, 4));
        assert_eq!(d("2021-01-01").iso_week(), (2020, 53));
        assert_eq!(d("2018-12-31").iso_week(), (2019, 1));
        assert_eq!(d("2026-07-17").iso_week(), (2026, 29));
        // ISO_WEEK_DATE writes LocalDate.MIN as -999999999-W01-1 and
        // LocalDate.MAX as +999999999-W52-5.
        assert_eq!(Date::MIN.iso_week(), (-999_999_999, 1));
        assert_eq!(Date::MAX.iso_week(), (999_999_999, 52));
    }

    #[test]
    fn months_and_years_clamp_the_day_and_stay_in_range() {
        assert_eq!(d("2024-01-31").plus_months(1), Some(d("2024-02-29")));
        assert_eq!(d("2024-03-31").plus_months(-1), Some(d("2024-02-29")));
        assert_eq!(d("2024-02-29").plus_years(-1), Some(d("2023-02-28")));
        assert_eq!(d("2026-07-14").plus_months(-19), Some(d("2024-12-14")));
        assert_eq!(d("2026-07-14").plus_weeks(-2), Some(d("2026-06-30")));
        assert_eq!(Date::MAX.plus_months(1), None);
        assert_eq!(Date::MIN.plus_months(-1), None);
        assert_eq!(Date::MAX.plus_years(1), None);
        assert_eq!(Date::MAX.plus_months(0), Some(Date::MAX));
        assert_eq!(Date::MIN.plus_weeks(-1), None);
        assert_eq!(d("2026-07-14").plus_months(i64::MAX), None);
        assert_eq!(d("2026-07-14").plus_weeks(i64::MAX), None);
    }

    #[test]
    fn years_are_written_with_four_digits_or_a_sign() {
        let mut s = String::new();
        for y in [0, 7, 999, 2026, 9999, 10000, -1, -10000] {
            write_year(&mut s, y).unwrap();
            s.push(' ');
        }
        assert_eq!(s, "0000 0007 0999 2026 9999 +10000 -0001 -10000 ");
    }
}
