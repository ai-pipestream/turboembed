//! The clock suffix of an ISO 8601 timestamp after a date
//! (`T09:30:00.125+05:30`). OpenNLP checks it and reports only the date,
//! so the check decides whether the date before it is a mention at all.

use crate::scan::{boundary_after, byte_at};

/// `CursorTemporalExtractor.timeOfDayAt`: whether a complete time of day
/// starts at byte `start`. That is `T` or `t`; two digits of hour, `:`
/// and two of minute; optionally `:` and two digits of second; optionally
/// a `.` or `,` fraction of the last field; then optionally `Z`, `z` or a
/// `+hh:mm` or `-hh:mm` offset; and a word boundary that is not a further
/// clock field, fraction or offset.
///
/// Hour 24 needs zero minutes, seconds and fraction. Second 60 is taken
/// without a leap second table. Offset hours go to 23 and minutes to 59.
/// Every byte compared is ASCII, so a byte offset steps as the Java's
/// UTF-16 index does.
pub(super) fn time_of_day_at(b: &[u8], text: &str, start: usize) -> bool {
    if !matches!(byte_at(b, start), b'T' | b't') || byte_at(b, start + 3) != b':' {
        return false;
    }
    let (Some(hour), Some(minute)) = (two_digits(b, start + 1), two_digits(b, start + 4)) else {
        return false;
    };
    if hour > 24 || minute >= 60 {
        return false;
    }
    let mut i = start + 6;
    let mut second = 0;
    if byte_at(b, i) == b':' && byte_at(b, i + 1).is_ascii_digit() {
        match two_digits(b, i + 1) {
            Some(s) if s <= 60 => second = s,
            _ => return false,
        }
        i += 3;
    }
    let mut nonzero_fraction = false;
    if matches!(byte_at(b, i), b'.' | b',') && byte_at(b, i + 1).is_ascii_digit() {
        i += 1;
        while byte_at(b, i).is_ascii_digit() {
            nonzero_fraction |= b[i] != b'0';
            i += 1;
        }
    }
    if hour == 24 && (minute != 0 || second != 0 || nonzero_fraction) {
        return false;
    }
    match byte_at(b, i) {
        b'Z' | b'z' => i += 1,
        b'+' | b'-' => {
            let (Some(h), Some(m)) = (two_digits(b, i + 1), two_digits(b, i + 4)) else {
                return false;
            };
            if h >= 24 || byte_at(b, i + 3) != b':' || m >= 60 {
                return false;
            }
            i += 6;
        }
        _ => {}
    }
    time_boundary_after(b, text, i)
}

/// `CursorTemporalExtractor.timeBoundaryAfter`: after any run of `:`, `.`
/// and `,`, no sign and a word boundary, so `T09:30.` ends a sentence
/// while `T09:30:00:01`, `T09:30:00.1.2` and `T09:30+05:30+01:00` are no
/// time of day.
fn time_boundary_after(b: &[u8], text: &str, start: usize) -> bool {
    let mut i = start;
    while matches!(byte_at(b, i), b':' | b'.' | b',') {
        i += 1;
    }
    !matches!(byte_at(b, i), b'+' | b'-') && boundary_after(text, i)
}

/// `CursorTemporalExtractor.digits` with a width of two: two ASCII digits
/// at `start` and no third.
pub(super) fn two_digits(b: &[u8], start: usize) -> Option<u32> {
    digits(b, start, 2)
}

/// `CursorTemporalExtractor.digits`: exactly `width` ASCII digits at
/// `start`, not followed by another digit.
pub(super) fn digits(b: &[u8], start: usize, width: usize) -> Option<u32> {
    let mut value = 0;
    for i in start..start + width {
        let d = byte_at(b, i);
        if !d.is_ascii_digit() {
            return None;
        }
        value = value * 10 + u32::from(d - b'0');
    }
    (!byte_at(b, start + width).is_ascii_digit()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time(suffix: &str) -> bool {
        time_of_day_at(suffix.as_bytes(), suffix, 0)
    }

    #[test]
    fn complete_suffixes() {
        for ok in ["T00:00", "T24:00", "T24:00:00.000Z", "t09:30z", "T09:30,5", "T12:00-00:00", "T09:30.", "T09:30)"] {
            assert!(time(ok), "{ok}");
        }
        for bad in ["T24:01", "T09:30:1", "T09:30+05", "T09:30+05:30Z", "T09:30:00.1x", "T9:30", "X09:30", "T09:30-"] {
            assert!(!time(bad), "{bad}");
        }
    }

    #[test]
    fn digits_need_the_exact_width() {
        assert_eq!(digits(b"2026-", 0, 4), Some(2026));
        assert_eq!(digits(b"20261", 0, 4), None);
        assert_eq!(digits(b"202", 0, 4), None);
        assert_eq!(two_digits(b"7", 0), None);
    }
}
