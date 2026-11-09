//! United States street addresses, `UsAddressPiiExtractor`: a house
//! number, an optional direction, one to four capitalized name words or
//! ordinals, a street suffix from USPS Publication 28 Appendix C1
//! (`Street`, `St`, `Avenue`, `Ave`), an optional direction after it, and
//! an optional unit (`Apt 4B`, `Suite 200`, `# 12`); and post office
//! boxes (`PO Box 123`, `P.O. Box 123`, `Post Office Box 123`). The
//! mention runs from the house number, or the box words, to the last
//! piece found, so a city, state and ZIP after a comma stay outside it.

use turbo_analysis_layers::{PiiEvidence, PiiKind};

use crate::hits::Hits;
use crate::tables::street_suffixes::STREET_SUFFIXES;
use crate::text::{after, letter_or_digit, on_end, on_number_start, on_word_start};

const MAX_HOUSE_DIGITS: usize = 6;
const MAX_NAME_WORDS: usize = 4;
const MAX_UNIT_CHARS: usize = 6;
const MAX_BOX_DIGITS: usize = 10;

const DIRECTIONS: [&str; 16] = [
    "N",
    "S",
    "E",
    "W",
    "NE",
    "NW",
    "SE",
    "SW",
    "North",
    "South",
    "East",
    "West",
    "Northeast",
    "Northwest",
    "Southeast",
    "Southwest",
];

const UNITS: [&str; 16] = [
    "Apartment",
    "Apt",
    "Building",
    "Bldg",
    "Department",
    "Dept",
    "Floor",
    "Fl",
    "Lot",
    "Room",
    "Rm",
    "Space",
    "Spc",
    "Suite",
    "Ste",
    "Unit",
];

/// Suffixes from [`STREET_SUFFIXES`] that name a thoroughfare or postal
/// abbreviation rather than a general English noun (`Park`, `Point`,
/// `Summit`, `Center`, `Hill`, `Plaza`), sorted for binary search.
const STREET_ONLY_SUFFIXES: [&str; 56] = [
    "ALLEY",
    "ALY",
    "AVE",
    "AVENUE",
    "BLVD",
    "BOULEVARD",
    "BYP",
    "BYPASS",
    "CIR",
    "COURT",
    "CSWY",
    "CT",
    "CTR",
    "CV",
    "CYN",
    "DR",
    "DRIVE",
    "EXPRESSWAY",
    "EXPY",
    "FREEWAY",
    "FWY",
    "HIGHWAY",
    "HWY",
    "JCT",
    "LANE",
    "LN",
    "LOOP",
    "MOTORWAY",
    "MTWY",
    "PARKWAY",
    "PKWY",
    "PL",
    "PLZ",
    "RD",
    "RDG",
    "ROAD",
    "RTE",
    "RUE",
    "SKWY",
    "SQ",
    "ST",
    "STRA",
    "STRAVENUE",
    "STREET",
    "TER",
    "TERRACE",
    "THROUGHWAY",
    "TPKE",
    "TRCE",
    "TRFY",
    "TRL",
    "TRWY",
    "TURNPIKE",
    "WAY",
    "XING",
    "XRD",
];

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let found = if c.is_ascii_digit() && on_number_start(text, i) {
            street(text, i)
        } else if matches!(c, b'P' | b'p') && on_word_start(text, i) {
            po_box(text, i)
        } else {
            None
        };
        match found {
            Some(end) => {
                hits.add_text(text, i, end, PiiKind::StreetAddress, PiiEvidence::Shape);
                i = end;
            }
            None => i += 1,
        }
    }
}

/// One word at `at`: a run of letters, digits, `'` and `-` that starts
/// and ends with a letter or digit. Answers its end.
fn word_end(text: &str, at: usize) -> Option<usize> {
    let mut last = at;
    for (o, ch) in text[at..].char_indices() {
        if letter_or_digit(ch) {
            last = at + o + ch.len_utf8();
        } else if !(matches!(ch, '\'' | '\u{2019}' | '-') && o > 0) {
            break;
        }
    }
    (last > at).then_some(last)
}

/// The byte after one horizontal space (`' '`, `'\t'`, `U+00A0` or
/// `U+202F`) at `at`, never a line break, so an address does not run on
/// into the next line.
fn space(b: &[u8], at: usize) -> Option<usize> {
    match b.get(at..) {
        Some([b' ' | b'\t', ..]) => Some(at + 1),
        Some([0xC2, 0xA0, ..]) => Some(at + 2),
        Some([0xE2, 0x80, 0xAF, ..]) => Some(at + 3),
        _ => None,
    }
}

fn is_ordinal(w: &str) -> bool {
    let digits = w.bytes().take_while(u8::is_ascii_digit).count();
    let tail = &w[digits..];
    (1..=3).contains(&digits) && ["st", "nd", "rd", "th"].iter().any(|t| tail.eq_ignore_ascii_case(t))
}

fn capitalized(w: &str) -> bool {
    w.chars().next().is_some_and(char::is_uppercase)
}

fn in_table(w: &str, table: &[&str]) -> bool {
    if !capitalized(w) || w.len() > 10 || !w.is_ascii() {
        return false;
    }
    let mut up = [0u8; 10];
    for (u, c) in up.iter_mut().zip(w.bytes()) {
        *u = c.to_ascii_uppercase();
    }
    let up = std::str::from_utf8(&up[..w.len()]).unwrap_or("");
    table.binary_search(&up).is_ok()
}

fn is_suffix(w: &str) -> bool {
    in_table(w, &STREET_SUFFIXES)
}

fn is_street_only_suffix(w: &str) -> bool {
    in_table(w, &STREET_ONLY_SUFFIXES)
}

fn is_direction(w: &str) -> bool {
    DIRECTIONS.contains(&w)
}

fn comma_city(text: &str, end: usize) -> bool {
    let b = text.as_bytes();
    b.get(end) == Some(&b',')
        && space(b, end + 1).and_then(|s| word_end(text, s).map(|e| &text[s..e])).is_some_and(capitalized)
}

/// The end of a street address whose house number starts at `start`.
fn street(text: &str, start: usize) -> Option<usize> {
    let b = text.as_bytes();
    let digits = b[start..].iter().take(MAX_HOUSE_DIGITS + 1).take_while(|c| c.is_ascii_digit()).count();
    if !(1..=MAX_HOUSE_DIGITS).contains(&digits) || b[start] == b'0' {
        return None;
    }
    let mut p = start + digits;
    // One letter may follow the number (`221B`).
    if b.get(p).is_some_and(u8::is_ascii_uppercase) && on_end(text, p + 1) {
        p += 1;
    }
    if !on_end(text, p) {
        return None;
    }
    // The words after the number, each after one space: a direction may
    // lead, then names and ordinals; the last suffix with a name before
    // it ends the street.
    let mut words = [(0usize, 0usize); MAX_NAME_WORDS + 2];
    let mut n_words = 0;
    let mut at = p;
    while n_words < words.len() {
        let Some(s) = space(b, at) else { break };
        let Some(e) = word_end(text, s) else { break };
        let w = &text[s..e];
        if !(capitalized(w) || is_ordinal(w)) {
            break;
        }
        words[n_words] = (s, e);
        n_words += 1;
        at = e;
    }
    let words = &words[..n_words];
    let suffix_from = |first: usize| {
        let stop = words.len().min(first + 1 + MAX_NAME_WORDS);
        (first + 1..stop).rev().find(|&k| is_suffix(&text[words[k].0..words[k].1]))
    };
    let leads = words.first().is_some_and(|&(s, e)| is_direction(&text[s..e]));
    let suffix = leads.then(|| suffix_from(1)).flatten().or_else(|| suffix_from(0))?;
    let suffix_word = &text[words[suffix].0..words[suffix].1];
    let mut end = words[suffix].1;
    if let Some(e) = post_direction(text, end) {
        end = e;
    }
    if let Some(e) = unit(text, end) {
        end = e;
    } else if end < words[words.len() - 1].1 {
        // A city may follow a street suffix without a comma. Only a known
        // title or an ambiguous noun suffix is enough to reject that boundary.
        let word = |i: usize| &text[words[i].0..words[i].1];
        let journal = suffix == 1
            && word(0).eq_ignore_ascii_case("Wall")
            && (suffix_word.eq_ignore_ascii_case("Street") || suffix_word.eq_ignore_ascii_case("St"))
            && word(2).eq_ignore_ascii_case("Journal");
        if !is_street_only_suffix(suffix_word) || journal {
            return None;
        }
    }
    let year = digits == 4 && p == start + 4 && matches!(b[start..start + 2], [b'1', b'9'] | [b'2', b'0']);
    let year_safe = is_street_only_suffix(suffix_word) && !suffix_word.eq_ignore_ascii_case("Court");
    if year && end == words[suffix].1 && !year_safe && !comma_city(text, end) {
        return None;
    }
    Some(end)
}

/// The end of a direction after the suffix ending at `end`: `Main St
/// NW`, or after a `.` or a comma a quadrant alone (`L St., NW`), since
/// a word after them more often starts the next sentence or the city.
fn post_direction(text: &str, end: usize) -> Option<usize> {
    let b = text.as_bytes();
    let mut p = end;
    if b.get(p) == Some(&b'.') {
        p += 1;
    }
    if b.get(p) == Some(&b',') {
        p += 1;
    }
    let s = space(b, p)?;
    let e = word_end(text, s)?;
    let w = &text[s..e];
    let quadrant = matches!(w, "NE" | "NW" | "SE" | "SW");
    (is_direction(w) && (quadrant || p == end) && on_end(text, e)).then_some(e)
}

/// The end of a unit after the street at `end`: an optional comma, a
/// designator from [`UNITS`] with an optional `.`, or `#`, then the unit's
/// letters and digits.
fn unit(text: &str, end: usize) -> Option<usize> {
    let b = text.as_bytes();
    let mut p = end;
    if b.get(p) == Some(&b',') {
        p += 1;
    }
    p = space(b, p)?;
    if b.get(p) == Some(&b'#') {
        p += 1;
        if let Some(q) = space(b, p) {
            p = q;
        }
    } else {
        let e = word_end(text, p)?;
        let w = &text[p..e];
        if !UNITS.iter().any(|u| u.eq_ignore_ascii_case(w)) || !capitalized(w) {
            return None;
        }
        p = e;
        if b.get(p) == Some(&b'.') {
            p += 1;
        }
        p = space(b, p)?;
        if b.get(p) == Some(&b'#') {
            p += 1;
        }
    }
    let n = b[p..].iter().take(MAX_UNIT_CHARS + 1).take_while(|c| c.is_ascii_alphanumeric() || **c == b'-').count();
    let stop = p + n;
    let ok = (1..=MAX_UNIT_CHARS).contains(&n) && b[stop - 1] != b'-' && on_end(text, stop);
    ok.then_some(stop)
}

/// The end of a post office box at `start`: `PO`, `P.O.`, `P. O.`,
/// `P O` or `Post Office`, then `Box` and its number, in any case.
fn po_box(text: &str, start: usize) -> Option<usize> {
    let b = text.as_bytes();
    let lower = |at: usize| b.get(at).map(u8::to_ascii_lowercase);
    let mut p = start;
    if b.len() >= p + 4
        && b[p..p + 4].eq_ignore_ascii_case(b"post")
        && let Some(q) = space(b, p + 4)
        && b.len() >= q + 6
        && b[q..q + 6].eq_ignore_ascii_case(b"office")
    {
        p = q + 6;
    } else {
        p += 1;
        if lower(p) == Some(b'.') {
            p += 1;
        }
        if let Some(q) = space(b, p)
            && lower(q) == Some(b'o')
        {
            p = q;
        }
        if lower(p) != Some(b'o') {
            return None;
        }
        p += 1;
        if lower(p) == Some(b'.') {
            p += 1;
        }
    }
    p = space(b, p)?;
    if !(b.len() >= p + 3 && b[p..p + 3].eq_ignore_ascii_case(b"box")) {
        return None;
    }
    p = space(b, p + 3)?;
    let n = b[p..].iter().take(MAX_BOX_DIGITS + 1).take_while(|c| c.is_ascii_digit()).count();
    let stop = p + n;
    let ok = (1..=MAX_BOX_DIGITS).contains(&n) && on_end(text, stop) && after(text, stop) != Some('-');
    ok.then_some(stop)
}

#[cfg(test)]
mod tests {
    use super::{STREET_ONLY_SUFFIXES, STREET_SUFFIXES};

    #[test]
    fn street_only_suffixes_are_sorted_and_in_street_suffixes() {
        assert!(STREET_ONLY_SUFFIXES.windows(2).all(|w| w[0] < w[1]));
        for &s in &STREET_ONLY_SUFFIXES {
            assert!(STREET_SUFFIXES.binary_search(&s).is_ok(), "{s}");
        }
    }
}
