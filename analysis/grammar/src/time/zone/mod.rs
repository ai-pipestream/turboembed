//! Explicit offsets and a bounded inventory of zone names. Regional
//! abbreviations require caller context; city names use dated EU rules.
mod resolution;
use crate::engine::{Lexed, NumeralStyle};
pub use resolution::{Offsets, offset_for, write_timex};
use turbo_unicode::whitespace::WhitespaceMode;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Region {
    #[default]
    Unspecified,
    NorthAmerica,
    China,
    India,
    Ireland,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zone {
    /// Recognized abbreviation without sufficient regional context.
    Unresolved,
    /// Minutes east of UTC, limited to [-14:00, +14:00].
    Fixed(i16),
    /// Madrid and Amsterdam under the EU rule, 1996 onward.
    CentralEurope,
    /// London under the EU rule, 1996 onward.
    London,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZoneMatch {
    pub start: usize,
    pub end: usize,
    pub zone: Zone,
}

/// A zone at `start`, allowing whitespace but no intervening words.
/// UTC/GMT numeric forms consume the complete offset or refuse it.
pub fn at(text: &str, start: usize, region: Region) -> Option<ZoneMatch> {
    let rest = text.get(start..)?;
    let skip = rest
        .chars()
        .take_while(|&c| WhitespaceMode::Unicode.is_whitespace(c as u32))
        .map(char::len_utf8)
        .sum::<usize>();
    let start = start + skip;
    let tail = &text[start..];
    // Every supported name is short. Bound scanning work even when a
    // time mention is followed by a large document.
    let rest = &tail[..tail.char_indices().nth(128).map_or(tail.len(), |(i, _)| i)];
    let mut lex = Lexed::default();
    lex.lex(
        rest,
        &NumeralStyle { mixed_separators: false, space_groups: false, indian_groups: false, decimal_comma: false },
    );
    if lex.is_empty() || lex.lexemes[0].start != 0 {
        return None;
    }
    let first = lex.word(0);
    if rest.as_bytes().get(lex.lexemes[0].end as usize) == Some(&b'_') {
        return None;
    }
    if first == "utc" || first == "gmt" {
        let after = lex.lexemes[0].end as usize;
        let (minutes, length) = numeric(&rest[after..])?;
        return Some(ZoneMatch { start, end: start + after + length, zone: Zone::Fixed(minutes) });
    }
    let region_offset = match (first, region) {
        ("cst", Region::NorthAmerica) => Some(-360),
        ("pt", Region::NorthAmerica) => Some(-480),
        ("cst", Region::China) => Some(480),
        ("ist", Region::India) => Some(330),
        ("ist", Region::Ireland) => Some(60),
        _ => None,
    };
    if let Some(minutes) = region_offset {
        return Some(ZoneMatch { start, end: start + lex.lexemes[0].end as usize, zone: Zone::Fixed(minutes) });
    }
    if matches!(first, "cst" | "ist" | "pt") {
        return Some(ZoneMatch { start, end: start + lex.lexemes[0].end as usize, zone: Zone::Unresolved });
    }
    for &(phrase, zone) in NAMES {
        let words: Vec<_> = phrase.split(' ').collect();
        if words.len() > lex.len() {
            continue;
        }
        if words.iter().enumerate().all(|(i, w)| {
            lex.word(i) == *w
                && (i == 0
                    || rest[lex.lexemes[i - 1].end as usize..lex.lexemes[i].start as usize]
                        .chars()
                        .all(|c| WhitespaceMode::Unicode.is_whitespace(c as u32)))
        }) {
            return Some(ZoneMatch { start, end: start + lex.lexemes[words.len() - 1].end as usize, zone });
        }
    }
    None
}
fn numeric(s: &str) -> Option<(i16, usize)> {
    if !s.starts_with(['+', '-']) {
        return Some((0, 0));
    }
    let b = s.as_bytes();
    let sign = if b[0] == b'-' { -1 } else { 1 };
    let mut end = 1;
    while b.get(end).is_some_and(u8::is_ascii_digit) {
        end += 1;
    }
    let count = end - 1;
    let (hours, mut minutes) = match count {
        1 | 2 => (s[1..end].parse::<i16>().ok()?, 0),
        4 => (s[1..3].parse().ok()?, s[3..5].parse().ok()?),
        _ => return None,
    };
    if b.get(end) == Some(&b':') {
        if count > 2 || !b.get(end + 1..end + 3)?.iter().all(u8::is_ascii_digit) {
            return None;
        }
        minutes = s[end + 1..end + 3].parse().ok()?;
        end += 3;
    }
    if hours > 14
        || minutes > 59
        || (hours == 14 && minutes != 0)
        || s[end..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == ':' || c == '_' || c == '+' || c == '-')
    {
        return None;
    }
    if b.get(end) == Some(&b'.') && b.get(end + 1).is_some_and(u8::is_ascii_digit) {
        return None;
    }
    Some((sign * (hours * 60 + minutes), end))
}
const NAMES: &[(&str, Zone)] = &[
    ("eastern standard time", Zone::Fixed(-300)),
    ("eastern daylight time", Zone::Fixed(-240)),
    ("pacific standard time", Zone::Fixed(-480)),
    ("pacific daylight time", Zone::Fixed(-420)),
    ("central european summer time", Zone::Fixed(120)),
    ("central european time", Zone::Fixed(60)),
    ("hora de madrid", Zone::CentralEurope),
    ("hora de londres", Zone::London),
    ("amsterdamse tijd", Zone::CentralEurope),
    ("tijd in amsterdam", Zone::CentralEurope),
    ("londense tijd", Zone::London),
    ("madrid time", Zone::CentralEurope),
    ("london time", Zone::London),
    ("pacific timezone", Zone::Fixed(-480)),
    ("zona horaria del pacífico", Zone::Fixed(-480)),
    ("pacifische tijdzone", Zone::Fixed(-480)),
    ("mountain standard time", Zone::Fixed(-420)),
    ("mountain daylight time", Zone::Fixed(-360)),
    ("mountain timezone", Zone::Fixed(-420)),
    ("mountain time", Zone::Fixed(-420)),
    ("pacific time", Zone::Fixed(-480)),
    ("eastern time", Zone::Fixed(-300)),
    ("mountain tijdzone", Zone::Fixed(-420)),
    ("est", Zone::Fixed(-300)),
    ("edt", Zone::Fixed(-240)),
    ("pst", Zone::Fixed(-480)),
    ("pdt", Zone::Fixed(-420)),
    ("mst", Zone::Fixed(-420)),
    ("mdt", Zone::Fixed(-360)),
    ("cdt", Zone::Fixed(-300)),
    ("cet", Zone::Fixed(60)),
    ("cest", Zone::Fixed(120)),
    ("eet", Zone::Fixed(120)),
    ("eest", Zone::Fixed(180)),
];

// Only values with an explicit clock can consume a zone suffix.
pub(crate) fn has_clock(value: crate::time::TimeValue) -> bool {
    use crate::time::spec::{Range, TimeOfDay, TimeValue};
    let clock = |p: crate::time::spec::Point| matches!(p.time, TimeOfDay::Clock(_));
    match value {
        TimeValue::Point { point, .. } => clock(point),
        TimeValue::Range { range: Range::Between { start, end, .. }, .. } => clock(start) || clock(end),
        TimeValue::Range { range: Range::For { start, .. }, .. } => clock(start),
        _ => false,
    }
}
