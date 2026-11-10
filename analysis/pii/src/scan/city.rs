//! Cities, `CityPiiExtractor`: a populated place of the bundled
//! gazetteer (`turbo_geo::BundledGazetteer`, Natural Earth's populated
//! places, public domain) in one of two settings, since many city names
//! are common words or given names too (`Mobile`, `Phoenix`, `Jackson`):
//!
//! - directly before a comma, a space and a US state's name or USPS code
//!   (`Springfield, IL`, `Mobile, Alabama`), when the gazetteer holds a
//!   place of that name in that state;
//! - directly after a city label (`city of Phoenix`, `City: Mobile`,
//!   `place of birth: Lyon`), as any place of that name.
//!
//! A bare name with neither is no mention. The name has at most
//! [`MAX_WORDS`] words with single spaces between them, its first and last
//! word start with a capital (`Rio de Janeiro`), no word character touches
//! either end ([`name_char`]), and it is matched by the gazetteer's own
//! rule (case, accents and punctuation folded). The longest name wins. The
//! normalized form is the gazetteer's name of the place.

use turbo_analysis_layers::{PiiEvidence, PiiKind};
use turbo_geo::{BundledGazetteer, Entry, Gazetteer};

use super::state::name_at;
use crate::hits::Hits;
use crate::tables::us_states::US_STATES;
use crate::text::{name_char, on_end, on_name_end, on_name_start, on_word_start, whitespace};

/// The most words a city name holds.
const MAX_WORDS: usize = 5;
/// The labels a city is read after, longest first, and whether each needs
/// a colon (`City: Mobile`), as `the city Council` and `town Hall` name
/// none.
const LABELS: [(&[u8], bool); 8] = [
    (b"city of residence", false),
    (b"place of birth", false),
    (b"birthplace", true),
    (b"city/town", true),
    (b"city of", false),
    (b"town of", false),
    (b"city", true),
    (b"town", true),
];

/// Finds the cities from their anchors, so text with no state after a
/// comma and no city label costs one pass over its bytes: a name that
/// ends at a comma before a state, and a name that starts after a label.
pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut found: Vec<(usize, usize, &'static str)> = Vec::new();
    for i in 0..b.len() {
        match b[i] {
            b',' => {
                if let Some(code) = state_after(text, i) {
                    found.extend(before_state(text, i, code));
                }
            }
            b'c' | b'C' | b't' | b'T' | b'p' | b'P' | b'b' | b'B' => {
                let head = b.get(i..i + 4).map(|h| [h[0] | 0x20, h[1] | 0x20, h[2] | 0x20, h[3] | 0x20]);
                if !matches!(&head, Some(h) if matches!(h, b"city" | b"town" | b"plac" | b"birt")) {
                    continue;
                }
                if let Some(at) = after_label(text, i) {
                    found.extend(after_label_name(text, at));
                }
            }
            _ => {}
        }
    }
    found.sort_unstable_by_key(|&(start, end, _)| (start, usize::MAX - end));
    let mut last = 0;
    for (start, end, name) in found {
        if start >= last {
            hits.add(start, end, PiiKind::City, PiiEvidence::Shape, |s| s.push_str(name));
            last = end;
        }
    }
}

/// Whether a capital letter starts at `i`.
fn capital_at(text: &str, i: usize) -> bool {
    text.get(i..).and_then(|t| t.chars().next()).is_some_and(char::is_uppercase)
}

/// The longest city name in the state with USPS code `code` that ends at
/// the comma at `comma`.
fn before_state(text: &str, comma: usize, code: &str) -> Option<(usize, usize, &'static str)> {
    let mut starts = [0; MAX_WORDS];
    let n = word_starts_before(text, comma, &mut starts);
    starts[..n].iter().rev().find_map(|&start| {
        let entry = lookup(text, start, comma)?
            .into_iter()
            .find(|e| e.country_code() == Some("US") && e.containment().any(|a| is_state(a, code)))?;
        Some((start, comma, entry.name()))
    })
}

/// The longest city name that starts at `at`, after a label: any place of
/// that name, or one in the state after it when one follows.
fn after_label_name(text: &str, at: usize) -> Option<(usize, usize, &'static str)> {
    let mut ends = [0; MAX_WORDS];
    let n = word_ends(text, at, &mut ends);
    ends[..n].iter().rev().find_map(|&end| {
        let found = lookup(text, at, end)?;
        let state = state_after(text, end);
        let in_state = |e: &&Entry<'static>| {
            state.is_some_and(|code| e.country_code() == Some("US") && e.containment().any(|a| is_state(a, code)))
        };
        let entry = found.iter().find(in_state).or(found.first())?;
        Some((at, end, entry.name()))
    })
}

/// The gazetteer's places named `text[start..end]`, when it is written
/// like a name with no word character touching either end.
fn lookup(text: &str, start: usize, end: usize) -> Option<Vec<Entry<'static>>> {
    let written = &text[start..end];
    if !on_name_start(text, start) || !on_name_end(text, end) || !capitals_like_a_name(written) {
        return None;
    }
    let found = BundledGazetteer::get().lookup(written);
    (!found.is_empty()).then_some(found)
}

/// Where a city name may start after the city label at `i`: past the
/// spaces, colons, `#` and `=` after it, a colon among them when the label
/// needs one.
fn after_label(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    if !on_word_start(text, i) {
        return None;
    }
    let (label, colon) = LABELS
        .iter()
        .find(|(l, _)| b.get(i..i + l.len()).is_some_and(|w| w.eq_ignore_ascii_case(l)) && on_end(text, i + l.len()))?;
    let mut at = i + label.len();
    let mut saw_colon = false;
    while let Some(c) = text[at..].chars().next() {
        if !(whitespace(c) || matches!(c, ':' | '#' | '=')) {
            break;
        }
        saw_colon |= c == ':';
        at += c.len_utf8();
    }
    (at > i + label.len() && (saw_colon || !colon) && capital_at(text, at)).then_some(at)
}

/// The starts of the words before `end`, nearest first, with single spaces
/// between them; a word runs back over word characters and the marks
/// inside a name (`St. Louis`, `O'Fallon`, `Winston-Salem`).
fn word_starts_before(text: &str, end: usize, starts: &mut [usize]) -> usize {
    let mut n = 0;
    let mut at = end;
    while n < starts.len() {
        let mut start = text[..at]
            .char_indices()
            .rev()
            .find(|&(_, c)| !(name_char(c) || matches!(c, '\'' | '\u{2019}' | '.' | '-')))
            .map_or(0, |(p, c)| p + c.len_utf8());
        let Some(lead) = text[start..at].find(|c: char| c.is_alphanumeric()) else { break };
        start += lead;
        if !capital_at(text, start) {
            break;
        }
        starts[n] = start;
        n += 1;
        let b = text.as_bytes();
        if start < 2 || b[start - 1] != b' ' || b[start - 2] == b' ' {
            break;
        }
        at = start - 1;
    }
    n
}

/// The ends of the first `ends.len()` words from `i`, nearest first, with
/// single spaces between them; a word runs over word characters and the
/// marks inside a name (`St. Louis`, `O'Fallon`, `Winston-Salem`), and
/// ends before trailing ones (`Springfield.`).
fn word_ends(text: &str, i: usize, ends: &mut [usize]) -> usize {
    let mut n = 0;
    let mut at = i;
    while n < ends.len() {
        let run = text[at..]
            .char_indices()
            .find(|&(_, c)| !(name_char(c) || matches!(c, '\'' | '\u{2019}' | '.' | '-')))
            .map_or(text.len(), |(p, _)| at + p);
        let end = text[at..run].trim_end_matches(|c: char| !name_char(c)).len() + at;
        if end == at || !on_name_end(text, end) {
            break;
        }
        ends[n] = end;
        n += 1;
        if text.as_bytes().get(run) != Some(&b' ') || !on_name_start(text, run + 1) {
            break;
        }
        at = run + 1;
    }
    n
}

/// Whether the first and last words of `written` start with a capital, as
/// a name does whose inner words may be small (`Rio de Janeiro`).
fn capitals_like_a_name(written: &str) -> bool {
    let first = written.split(' ').next().is_some_and(|w| w.chars().next().is_some_and(char::is_uppercase));
    let last = written.rsplit(' ').next().is_some_and(|w| w.chars().next().is_some_and(char::is_uppercase));
    first && last
}

/// The USPS code of the state named directly after `end` by a comma and a
/// space (`, Illinois`, `, IL`).
fn state_after(text: &str, end: usize) -> Option<&'static str> {
    let b = text.as_bytes();
    if b.get(end..end + 2) != Some(b", ") || !b.get(end + 2).is_some_and(u8::is_ascii_uppercase) {
        return None;
    }
    let at = end + 2;
    if let Some((_, code)) = name_at(text, at) {
        return Some(code);
    }
    let code = b.get(at..at + 2)?;
    if !on_name_end(text, at + 2) {
        return None;
    }
    US_STATES.iter().find(|(_, c)| c.as_bytes() == code).map(|(_, c)| *c)
}

/// Whether the gazetteer's admin-1 name `admin` is a name of the state
/// with USPS code `code`.
fn is_state(admin: &str, code: &str) -> bool {
    US_STATES.iter().any(|(name, c)| *c == code && name.eq_ignore_ascii_case(admin))
}
