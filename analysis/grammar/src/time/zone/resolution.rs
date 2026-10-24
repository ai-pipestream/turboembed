use super::Zone;
use crate::time::{Instant, Resolution, Resolved, Timex, calendar::nth_weekday};
use std::fmt::{self, Write};
use turbo_analysis_layers::DateTime;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Offsets {
    pub start: i16,
    pub end: i16,
}
/// Offset in minutes east of UTC. Explicit abbreviations are fixed.
/// Cities use the EU rule for 1996..=2099 and refuse skipped or repeated
/// wall-clock hours. A date is required for a city, even for a time-only value.
pub fn offset_for(zone: Zone, local: Option<DateTime>) -> Option<i16> {
    let base = match zone {
        Zone::Unresolved => return None,
        Zone::Fixed(minutes) => return (-840..=840).contains(&minutes).then_some(minutes),
        Zone::CentralEurope => 60,
        Zone::London => 0,
    };
    let local = local?;
    let date = local.date();
    if !(1996..=2099).contains(&date.year()) {
        return None;
    }
    let spring = nth_weekday(date.year(), 3, 7, 0)?;
    let autumn = nth_weekday(date.year(), 10, 7, 0)?;
    let seconds = i64::from(local.hour()) * 3600 + i64::from(local.minute()) * 60 + i64::from(local.second());
    // EU transitions occur at 01:00 UTC. In local time the missing
    // spring and repeated autumn hour both begin at 01:00 + base.
    let transition = 3600 + i64::from(base) * 60;
    if (date == spring || date == autumn) && (transition..transition + 3600).contains(&seconds) {
        return None;
    }
    let summer = (date > spring || date == spring && seconds >= transition + 3600)
        && (date < autumn || date == autumn && seconds < transition);
    Some(base + if summer { 60 } else { 0 })
}
fn local(instant: Instant, reference: Option<DateTime>) -> Option<DateTime> {
    match instant {
        Instant::DateTime(d) => Some(d),
        Instant::Time(s) => DateTime::new(reference?.date(), (s / 3600) as u8, (s / 60 % 60) as u8, (s % 60) as u8),
        Instant::Date(_) => None,
    }
}
impl Offsets {
    pub fn for_resolution(r: &Resolved, zone: Zone, reference: Option<DateTime>) -> Option<Self> {
        let a = r.alternatives.get(r.preferred)?;
        let resolve = |i: Instant| offset_for(zone, local(i, reference));
        match a.value {
            Resolution::Point(i) if !matches!(i, Instant::Date(_)) => {
                let start = resolve(i)?;
                Some(Self { start, end: start })
            }
            Resolution::Range { start, end } => {
                let start = start.or(end).and_then(resolve)?;
                let end = end.map(resolve).unwrap_or(Some(start))?;
                Some(Self { start, end })
            }
            _ => None,
        }
    }
}
fn suffix(out: &mut impl Write, offset: i16) -> fmt::Result {
    if offset == 0 {
        out.write_char('Z')
    } else {
        let n = offset.unsigned_abs();
        write!(out, "{}{:02}:{:02}", if offset < 0 { '-' } else { '+' }, n / 60, n % 60)
    }
}
/// Write an explicit offset on each timed endpoint. Duration is a local
/// wall-clock duration, as in the unzoned grammar; it is not elapsed UTC time.
pub fn write_timex(out: &mut impl Write, timex: Timex, offsets: Offsets) -> fmt::Result {
    match timex {
        Timex::Point(p) => {
            write!(out, "{p}")?;
            suffix(out, offsets.start)
        }
        Timex::Range(a, b, d) => {
            write!(out, "({a}")?;
            suffix(out, offsets.start)?;
            write!(out, ",{b}")?;
            suffix(out, offsets.end)?;
            write!(out, ",{d})")
        }
        Timex::Duration(d) => write!(out, "{d}"),
    }
}
