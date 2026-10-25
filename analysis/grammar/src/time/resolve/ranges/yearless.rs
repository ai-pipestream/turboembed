//! Pair annual date bounds around their shared leap-day occurrence.

use super::*;

fn month_day(d: &Dated) -> Option<(u8, u8)> {
    match d.timex {
        DateTimex::Date(TimexDate::Ymd { year: TimexYear::Any, month: Some(m), day: Some(d) }) => Some((m, d)),
        _ => None,
    }
}

/// Anchor both bounds to each occurrence of February 29. The other
/// endpoint belongs to that year, or the adjacent year for a range
/// crossing December. Independent searches can otherwise pair dates
/// separated by several years. Apply edges after pairing the raw days,
/// so an inclusive February 29 end still belongs to its leap year.
pub(super) fn pairs(cs: &Choice, se: Edge, ce: &Choice, ee: Edge) -> Option<[(Date, Date); 2]> {
    let (sm, em) = (month_day(&cs.past)?, month_day(&ce.past)?);
    if sm != (2, 29) && em != (2, 29) {
        return None;
    }
    let anchor = if sm == (2, 29) { cs } else { ce };
    let pair = |d: &Dated| {
        let anchor_year = d.span?.0.year();
        let crosses_year = i32::from(em < sm);
        let sy = if sm == (2, 29) { anchor_year } else { anchor_year.checked_sub(crosses_year)? };
        let ey = sy.checked_add(crosses_year)?;
        let (s, e) = (Date::new(sy, sm.0, sm.1)?, Date::new(ey, em.0, em.1)?);
        Some((edge_of((s, s.plus_days(1)?), se, TimeGrain::Day)?, edge_of((e, e.plus_days(1)?), ee, TimeGrain::Day)?))
    };
    Some([pair(&anchor.past)?, pair(&anchor.future)?])
}
