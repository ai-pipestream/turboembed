//! Calendar arithmetic over [`Date`]: weeks, weeks of a month, the nth
//! weekday of a month, Easter and the holidays built on them.

use turbo_analysis_layers::Date;

use super::spec::Holiday;

/// The Monday of the week (Monday to Sunday) holding `d`.
pub fn monday(d: Date) -> Option<Date> {
    d.plus_days(1 - i64::from(d.day_of_week()))
}

/// The first day of `month` of `year`.
pub fn month_start(year: i32, month: u8) -> Option<Date> {
    Date::new(year, month, 1)
}

/// The first day of the month after `month` of `year`.
pub fn month_end(year: i32, month: u8) -> Option<Date> {
    month_start(year, month)?.plus_months(1)
}

/// The day `weekday` (1 for Monday) of the week holding `d`.
pub fn weekday_of_week(d: Date, weekday: u8) -> Option<Date> {
    monday(d)?.plus_days(i64::from(weekday) - 1)
}

/// The Monday of ISO week `week` of `year`, or of its last week for 0.
pub fn iso_week_start(year: i32, week: u8) -> Option<Date> {
    let week = if week == 0 { Date::new(year, 12, 28)?.iso_week().1 } else { week };
    // January 4 is always in week 1.
    let first = monday(Date::new(year, 1, 4)?)?;
    let start = first.plus_weeks(i64::from(week) - 1)?;
    (start.iso_week() == (year, week)).then_some(start)
}

/// The `n`th `weekday` of a month, or its last for 0.
pub fn nth_weekday(year: i32, month: u8, weekday: u8, n: u8) -> Option<Date> {
    if n == 0 {
        let last = month_end(year, month)?.plus_days(-1)?;
        let back = (i64::from(last.day_of_week()) - i64::from(weekday)).rem_euclid(7);
        return last.plus_days(-back);
    }
    let first = month_start(year, month)?;
    let ahead = (i64::from(weekday) - i64::from(first.day_of_week())).rem_euclid(7);
    let d = first.plus_days(ahead + 7 * (i64::from(n) - 1))?;
    (d.month() == month).then_some(d)
}

/// The Monday of week `week` of a month (0 for its last), counting the
/// weeks that hold the month's Thursdays, as ISO weeks belong to the year
/// that holds their Thursday.
pub fn month_week_start(year: i32, month: u8, week: u8) -> Option<Date> {
    monday(nth_weekday(year, month, 4, week)?)
}

/// Easter Sunday of `year` in the Gregorian calendar (the anonymous
/// algorithm, also known as Meeus/Jones/Butcher).
pub fn easter(year: i32) -> Option<Date> {
    let a = year.rem_euclid(19);
    let b = year.div_euclid(100);
    let c = year.rem_euclid(100);
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15).rem_euclid(30);
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k).rem_euclid(7);
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = (h + l - 7 * m + 114) % 31 + 1;
    Date::new(year, month as u8, day as u8)
}

/// The holiday's date in `year`.
pub fn holiday(h: Holiday, year: i32) -> Option<Date> {
    match h {
        Holiday::Fixed { month, day } => Date::new(year, month, day),
        Holiday::Easter { days } => easter(year)?.plus_days(i64::from(days)),
        Holiday::Nth { month, weekday, n, days } => nth_weekday(year, month, weekday, n)?.plus_days(i64::from(days)),
    }
}

/// The first day of quarter `q` (1 to 4) of `year`.
pub fn quarter_start(year: i32, q: u8) -> Option<Date> {
    if !(1..=4).contains(&q) {
        return None;
    }
    month_start(year, (q - 1) * 3 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    #[test]
    fn weeks() {
        assert_eq!(monday(d("2016-11-13")), Some(d("2016-11-07")));
        assert_eq!(monday(d("2016-11-07")), Some(d("2016-11-07")));
        assert_eq!(weekday_of_week(d("2016-11-07"), 5), Some(d("2016-11-11")));
        assert_eq!(iso_week_start(2019, 1), Some(d("2018-12-31")));
        assert_eq!(iso_week_start(2015, 1), Some(d("2014-12-29")));
        assert_eq!(iso_week_start(2016, 0), Some(d("2016-12-26")));
        assert_eq!(iso_week_start(2020, 53), Some(d("2020-12-28")));
        assert_eq!(iso_week_start(2019, 53), None);
        assert_eq!(month_week_start(2017, 7, 0), Some(d("2017-07-24")));
        assert_eq!(month_week_start(2016, 7, 0), Some(d("2016-07-25")));
        assert_eq!(month_week_start(2015, 1, 1), Some(d("2014-12-29")));
        assert_eq!(month_week_start(2019, 1, 3), Some(d("2019-01-14")));
        assert_eq!(month_week_start(2021, 9, 0), Some(d("2021-09-27")));
    }

    #[test]
    fn nth_weekdays_and_holidays() {
        assert_eq!(nth_weekday(2018, 9, 1, 1), Some(d("2018-09-03")));
        assert_eq!(nth_weekday(2018, 9, 1, 5), None);
        assert_eq!(nth_weekday(2016, 5, 1, 0), Some(d("2016-05-30")));
        for (y, s) in [(2018, "2018-04-01"), (2019, "2019-04-21"), (2020, "2020-04-12"), (2016, "2016-03-27")] {
            assert_eq!(easter(y), Some(d(s)));
        }
        let thanksgiving = Holiday::Nth { month: 11, weekday: 4, n: 4, days: 0 };
        assert_eq!(holiday(thanksgiving, 2023), Some(d("2023-11-23")));
        let black_friday = Holiday::Nth { month: 11, weekday: 4, n: 4, days: 1 };
        assert_eq!(holiday(black_friday, 2010), Some(d("2010-11-26")));
        assert_eq!(holiday(Holiday::Easter { days: 1 }, 2019), Some(d("2019-04-22")));
        assert_eq!(holiday(Holiday::Fixed { month: 2, day: 30 }, 2019), None);
        assert_eq!(quarter_start(2019, 3), Some(d("2019-07-01")));
        assert_eq!(quarter_start(2019, 0), None);
        assert_eq!(quarter_start(2019, 5), None);
        assert_eq!(quarter_start(2019, u8::MAX), None);
        assert_eq!(month_end(2016, 12), Some(d("2017-01-01")));
    }
}
