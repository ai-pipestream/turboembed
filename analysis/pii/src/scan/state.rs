//! United States states, `UsStatePiiExtractor`: the name of a state, the
//! District of Columbia or an inhabited territory (`Illinois`, `New York`,
//! `Puerto Rico`), as written in the table or all in capitals, at word
//! boundaries, with single spaces between its words. `Washington, D.C.`
//! and `Washington DC` are the District, `New York City` is a city, not
//! the state, and a name before a county word that makes a county of the
//! list (`Washington County`) is the county's. A two-letter USPS code (`IL`) is a common word or
//! abbreviation too (`IN`, `OR`, `ME`, `OK`), so it is a mention only in
//! capitals and only after a state label (`State: IL`) or before a ZIP
//! code that ends an address: after a comma or a city of that state
//! (`Springfield, IL 62704`, `Springfield IL 62704`), or alone on a line
//! with its ZIP code, not after another word (`Order IN 12345`). The normalized form is the USPS code.

use turbo_analysis_layers::{PiiEvidence, PiiKind};

use super::city::city_before;
use super::national::on_number_end;
use super::us_labelled::STATES;
use crate::hits::Hits;
use crate::tables::us_counties::{SUFFIXES, county};
use crate::tables::us_states::US_STATES;
use crate::text::{on_end, on_name_end, on_name_start};
use crate::window::find_any_label;

/// The labels a code is read after, longest first.
const LABELS: [&[u8]; 2] = [b"state of residence", b"state"];
/// What after `Washington` makes it the District.
const DISTRICT: [&[u8]; 4] = [b", D.C.", b" D.C.", b", DC", b" DC"];

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_uppercase() || !on_name_start(text, i) {
            i += 1;
            continue;
        }
        let found = name_at(text, i).or_else(|| code_at(text, i).map(|end| (end, &text[i..end])));
        match found {
            Some((end, code)) => {
                hits.add(i, end, PiiKind::State, PiiEvidence::Shape, |s| s.push_str(code));
                i = end;
            }
            None => i += 1,
        }
    }
}

/// The end and code of the state name at `i`.
pub(super) fn name_at(text: &str, i: usize) -> Option<(usize, &'static str)> {
    let b = text.as_bytes();
    let (name, code) = US_STATES.iter().find(|(name, _)| {
        let n = name.as_bytes();
        b.get(i..i + n.len())
            .is_some_and(|w| w == n || (w.eq_ignore_ascii_case(n) && !w.iter().any(u8::is_ascii_lowercase)))
            && on_name_end(text, i + n.len())
    })?;
    let end = i + name.len();
    if *code == "WA"
        && let Some(d) = DISTRICT.iter().find(|d| b.get(end..end + d.len()).is_some_and(|w| w == **d))
        && on_name_end(text, end + d.len())
    {
        return Some((end + d.len(), "DC"));
    }
    if *code == "NY" && b.get(end..end + 5).is_some_and(|w| w.eq_ignore_ascii_case(b" city")) && on_end(text, end + 5) {
        return None;
    }
    // `Washington County` is a county, and the county scanner reads it;
    // a suffix word that makes no county (`the Texas area`, `Illinois
    // County`) leaves the state. Here, as for `New York City`, only a
    // letter or digit after the word makes it another word, so
    // `Washington County_x` is no state either.
    if b.get(end) == Some(&b' ')
        && SUFFIXES.iter().any(|s| {
            let at = end + 1;
            b.get(at..at + s.len()).is_some_and(|w| w.eq_ignore_ascii_case(s.as_bytes()))
                && on_end(text, at + s.len())
                && county(&text[i..at + s.len()]).is_some()
        })
    {
        return None;
    }
    Some((end, code))
}

/// The end of a two-capital USPS code at `i` that a state label is
/// before or a ZIP code after.
fn code_at(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    let code = b.get(i..i + 2)?;
    if !code.iter().all(u8::is_ascii_uppercase)
        || !on_name_end(text, i + 2)
        || !STATES.iter().any(|s| s.as_slice() == code)
    {
        return None;
    }
    let zip = b.get(i + 2) == Some(&b' ') && zip_end(text, i + 3).is_some_and(|end| address_like(text, i, end));
    (zip || find_any_label(text, i, &LABELS, false).is_some()).then_some(i + 2)
}

/// Whether a code at `code` before the ZIP code ending at `end` reads as
/// the end of an address, by what comes before it: a comma
/// (`Springfield, IL 62704`), a city of that state and one space
/// (`Springfield IL 62704 is`), or only blanks back to the start of a line
/// when only blanks follow the ZIP code to the line's end (`IL 62704`). A
/// word that is no city there (`Order IN 12345`, `PART OR 90210`), or a
/// word after a line's code and number (`IN 12345 units`), makes the
/// number a count.
fn address_like(text: &str, code: usize, end: usize) -> bool {
    let before = &text.as_bytes()[..code];
    if before.ends_with(b", ") {
        return true;
    }
    // Only the blanks directly before the code and after the ZIP code are
    // walked, so each byte is looked at once however many codes a line
    // holds.
    let blanks = before.iter().rev().take_while(|&&c| c == b' ' || c == b'\t').count();
    if matches!(before.len().checked_sub(blanks + 1).map(|p| before[p]), None | Some(b'\n')) {
        return line_ends_after(text, end);
    }
    before.len() >= 2
        && before.ends_with(b" ")
        && before[before.len() - 2] != b' '
        && city_before(text, code - 1, &text[code..code + 2])
}

/// Whether only blanks follow `end` to the end of its line or the text.
fn line_ends_after(text: &str, end: usize) -> bool {
    text[end..]
        .chars()
        .find(|&c| !matches!(c, ' ' | '\t' | '\r' | '\u{a0}'))
        .is_none_or(|c| matches!(c, '\n' | '\u{b}' | '\u{c}' | '\u{85}' | '\u{2028}' | '\u{2029}'))
}

/// The end of a whole ZIP or ZIP+4 token at `at` (`62704`, `62704-1234`),
/// with no letter, digit, `-` or `_` joined to it and not the start of a
/// decimal, grouped or split number (`12345abc`, `12345.67`, `12345,678`
/// and `12345/6` are none).
fn zip_end(text: &str, at: usize) -> Option<usize> {
    let b = text.as_bytes();
    let digits = |from: usize, n: usize| b.get(from..from + n).is_some_and(|d| d.iter().all(u8::is_ascii_digit));
    if !digits(at, 5) {
        return None;
    }
    let end = if b.get(at + 5) == Some(&b'-') && digits(at + 6, 4) { at + 10 } else { at + 5 };
    (on_number_end(text, end, b".,-/") && !matches!(b.get(end), Some(b'-' | b'_'))).then_some(end)
}
