//! A date written relative to a reference date: `3 months ago`, `in 2
//! weeks`, `today`.
//!
//! The count is the whole number of the largest unit that fits, rounded
//! toward the reference, so the phrase never says more than the date did:
//! a day under a week away is counted in days, under 31 days in weeks,
//! under a year in calendar months and from a year on in years. A month
//! mention (`July 2026`) is counted in months, or in years from twelve
//! months on. The phrases are ones the temporal scanner reads back with
//! the same reference date.

use turbo_analysis_layers::Date;

use super::form::Stated;

/// Whole calendar months from `a` to `b`, `a` not after `b`: a month is
/// counted once its day is reached, and the last day of a shorter month
/// counts as reaching any later day.
fn months_between(a: Date, b: Date) -> i64 {
    let whole = (i64::from(b.year()) - i64::from(a.year())) * 12 + i64::from(b.month()) - i64::from(a.month());
    let reached = b.day() >= a.day() || b.plus_days(1).is_some_and(|n| n.month() != b.month());
    if reached { whole } else { whole - 1 }
}

fn count(out: &mut String, n: i64, unit: &str, past: bool) {
    let plural = if n == 1 { "" } else { "s" };
    if past {
        out.push_str(&format!("{n} {unit}{plural} ago"));
    } else {
        out.push_str(&format!("in {n} {unit}{plural}"));
    }
}

/// Appends `was` relative to `reference` to `out`.
pub(super) fn write_relative(was: Stated, reference: Date, out: &mut String) {
    match was {
        Stated::Day(d) => {
            let days = reference.epoch_day() - d.epoch_day();
            let past = days > 0;
            let n = days.abs();
            match n {
                0 => out.push_str("today"),
                1 if past => out.push_str("yesterday"),
                1 => out.push_str("tomorrow"),
                2..=6 => count(out, n, "day", past),
                7..=30 => count(out, n / 7, "week", past),
                _ => {
                    let (a, b) = if past { (d, reference) } else { (reference, d) };
                    let months = months_between(a, b);
                    if months < 12 { count(out, months, "month", past) } else { count(out, months / 12, "year", past) }
                }
            }
        }
        Stated::Month { year, month } => {
            let months = (i64::from(reference.year()) * 12 + i64::from(reference.month()))
                - (i64::from(year) * 12 + i64::from(month));
            let (past, n) = (months > 0, months.abs());
            match n {
                0 => out.push_str("this month"),
                1..=11 => count(out, n, "month", past),
                _ => count(out, n / 12, "year", past),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u8, day: u8) -> Date {
        Date::new(y, m, day).unwrap()
    }

    fn rel(was: Stated, reference: Date) -> String {
        let mut out = String::new();
        write_relative(was, reference, &mut out);
        out
    }

    #[test]
    fn days_weeks_months_and_years() {
        let r = d(2026, 7, 14);
        let at = |y, m, day| rel(Stated::Day(d(y, m, day)), r);
        assert_eq!(at(2026, 7, 14), "today");
        assert_eq!(at(2026, 7, 13), "yesterday");
        assert_eq!(at(2026, 7, 15), "tomorrow");
        assert_eq!(at(2026, 7, 11), "3 days ago");
        assert_eq!(at(2026, 7, 20), "in 6 days");
        assert_eq!(at(2026, 7, 7), "1 week ago");
        assert_eq!(at(2026, 6, 14), "4 weeks ago", "30 days");
        assert_eq!(at(2026, 6, 13), "1 month ago", "31 days");
        assert_eq!(at(2026, 4, 15), "2 months ago", "the third month is not reached");
        assert_eq!(at(2025, 7, 15), "11 months ago");
        assert_eq!(at(2025, 7, 14), "1 year ago");
        assert_eq!(at(2028, 7, 13), "in 1 year");
        assert_eq!(at(1952, 7, 4), "74 years ago");
    }

    #[test]
    fn month_ends_and_leap_days_count_as_reached() {
        // From January 31 the end of February is a whole month.
        assert_eq!(months_between(d(2024, 1, 31), d(2024, 2, 29)), 1);
        assert_eq!(months_between(d(2023, 1, 31), d(2023, 2, 28)), 1);
        assert_eq!(months_between(d(2024, 1, 31), d(2024, 2, 28)), 0);
        // A leap day is a year old on the next February 28.
        assert_eq!(months_between(d(2024, 2, 29), d(2025, 2, 28)), 12);
        assert_eq!(rel(Stated::Day(d(2024, 2, 29)), d(2025, 2, 28)), "1 year ago");
        assert_eq!(rel(Stated::Day(d(2024, 2, 29)), d(2025, 2, 27)), "11 months ago");
    }

    #[test]
    fn a_month_mention_counts_months() {
        let r = d(2026, 7, 14);
        let at = |year, month| rel(Stated::Month { year, month }, r);
        assert_eq!(at(2026, 7), "this month");
        assert_eq!(at(2026, 6), "1 month ago");
        assert_eq!(at(2026, 9), "in 2 months");
        assert_eq!(at(2025, 7), "1 year ago");
        assert_eq!(at(2024, 8), "1 year ago", "23 months");
    }
}
