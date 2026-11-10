//! Cities, `CityPiiExtractor`: a populated place of the bundled
//! gazetteer (`turbo_geo::BundledGazetteer`, Natural Earth's populated
//! places, public domain) in one of two settings, since many city names
//! are common words or given names too (`Mobile`, `Phoenix`, `Jackson`):
//!
//! - directly before a comma, a space and a US state's or territory's
//!   name or USPS code (`Springfield, IL`, `Mobile, Alabama`,
//!   `San Juan, PR`), when the gazetteer holds a place of that name there;
//! - directly after a city label (`city of Phoenix`, `City: Mobile`,
//!   `place of birth: Lyon`), as any place of that name when no state
//!   follows (unless another capitalized word follows it, `City: San Foo`),
//!   or in the state after it when a state name or code follows (`City:
//!   Springfield, IL`, `City: Mobile Alabama`, `City: Springfield IL
//!   62704`), with no fallback to a record outside that state, so a
//!   mismatched state (`City: Mobile, IL`) is refused at the cost of
//!   smaller US homonyms missing from the 1:10m table under that state
//!   (`City: Dublin, OH`, `City: Rome, NY`).
//!
//! A bare name with neither is no mention. The name has at most
//! [`MAX_WORDS`] words with single spaces between them, its first and last
//! word start with a capital (`Rio de Janeiro`), no word character touches
//! either end ([`name_char`]), and it is matched by the gazetteer's own
//! rule (case, accents and punctuation folded). The longest name wins. The
//! normalized form is the gazetteer's name of the place.

use turbo_analysis_layers::{PiiEvidence, PiiKind};
use turbo_geo::{BundledGazetteer, Entry, Gazetteer};

use super::state::{name_at, zip_end};
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

/// The territories, which the gazetteer files under their own country
/// codes, the same as their USPS codes.
const TERRITORIES: [&str; 5] = ["AS", "GU", "MP", "PR", "VI"];
/// How many bytes the scan tests for anchors at once.
const LANES: usize = 32;

/// Finds the cities from their anchors, so text with no state after a
/// comma and no city label costs one pass that tests [`LANES`] bytes at a
/// time: a comma, a space and a capital (the name ends at the comma
/// before a state), and the colon or the `of` of a city label (the name
/// starts after it).
pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut found: Vec<(usize, usize, &'static str)> = Vec::new();
    let mut at = 0;
    while at < b.len() {
        // A whole lane is tested with no branch, so the test vectorizes;
        // the lanes of most text hold no anchor.
        if let Some(w) = b.get(at..at + LANES + 2).and_then(|w| <&[u8; LANES + 2]>::try_from(w).ok())
            && (0..LANES).fold(0u8, |any, j| any | u8::from(anchor(w[j], w[j + 1], w[j + 2]))) == 0
        {
            at += LANES;
            continue;
        }
        let byte = |i: usize| b.get(i).copied().unwrap_or(0);
        for i in at..(at + LANES).min(b.len()) {
            if anchor(byte(i), byte(i + 1), byte(i + 2)) {
                at_anchor(text, i, &mut found);
            }
        }
        at += LANES;
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

/// Whether the bytes `c0`, `c1`, `c2` start an anchor: `, ` and a capital,
/// a colon, or `of` in either case.
#[inline(always)]
fn anchor(c0: u8, c1: u8, c2: u8) -> bool {
    (c0 == b',') & (c1 == b' ') & c2.is_ascii_uppercase() | (c0 == b':') | (c0 | 0x20 == b'o') & (c1 | 0x20 == b'f')
}

/// The cities that the anchor at `i` starts or ends.
fn at_anchor(text: &str, i: usize, found: &mut Vec<(usize, usize, &'static str)>) {
    let mut after = |start: Option<usize>| {
        if let Some(name) = start.and_then(|start| after_label(text, start)) {
            found.extend(after_label_name(text, name));
        }
    };
    match text.as_bytes()[i] {
        b',' => {
            if let Some(code) = state_after(text, i) {
                found.extend(before_state(text, i, code));
            }
        }
        b':' => {
            // Only the first colon of a run of separators looks back, and
            // only as far as the run goes, so each byte is walked once
            // (`:::` costs no more than `:`).
            let mut end = i;
            for (at, c) in text[..i].char_indices().rev() {
                if c == ':' {
                    return;
                }
                if !(whitespace(c) || matches!(c, '#' | '=')) {
                    break;
                }
                end = at;
            }
            let last = end.checked_sub(1).map(|e| text.as_bytes()[e] | 0x20);
            for (label, _) in LABELS.iter().filter(|(l, _)| l.last().copied() == last) {
                after(end.checked_sub(label.len()));
            }
        }
        _ => {
            for len in [b"city of".len(), b"place of".len()] {
                after((i + 2).checked_sub(len));
            }
        }
    }
}

/// Whether `entry` is a place in the state or territory with USPS code
/// `code`.
fn in_state(entry: &Entry<'static>, code: &str) -> bool {
    match entry.country_code() {
        Some("US") => entry.containment().any(|a| is_state(a, code)),
        Some(country) => country == code && TERRITORIES.contains(&code),
        None => false,
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
        let entry = lookup(text, start, comma)?.into_iter().find(|e| in_state(e, code))?;
        Some((start, comma, entry.name()))
    })
}

/// The longest city name that starts at `at`, after a label: any place of
/// that name when no state follows, or one in the state after it when one
/// follows (`City: Springfield, IL`, `City: Springfield IL 62704`, `City:
/// Mobile Alabama`), with no fallback to a place outside that state
/// (`City: Mobile, IL` is no mention). A name that another capitalized
/// word follows is part of a longer one (`City: San Foo`), so it is no
/// mention.
fn after_label_name(text: &str, at: usize) -> Option<(usize, usize, &'static str)> {
    let mut ends = [0; MAX_WORDS];
    let n = word_ends(text, at, &mut ends);
    ends[..n].iter().rev().find_map(|&end| {
        let state = state_after(text, end);
        if state.is_none() && text.as_bytes().get(end) == Some(&b' ') && capital_at(text, end + 1) {
            return None;
        }
        let found = lookup(text, at, end)?;
        let entry = match state {
            Some(code) => found.into_iter().find(|e| in_state(e, code))?,
            None => found.into_iter().next()?,
        };
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
        // The last word starts with a capital; an inner one may be small
        // (`Truth or Consequences`, `Fond du Lac`), and `lookup` holds each
        // candidate's first word to a capital.
        if n == 0 && !capital_at(text, start) {
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

/// The USPS code of the state named directly after `end`: after `, ` by
/// name or USPS code (`, Illinois`, `, IL`), or after ` ` by name or by a
/// capital USPS code before a ZIP code (` Alabama`, ` IL 62704`).
fn state_after(text: &str, end: usize) -> Option<&'static str> {
    let b = text.as_bytes();
    let (comma, at) = if b.get(end..end + 2) == Some(b", ") {
        (true, end + 2)
    } else if b.get(end) == Some(&b' ') {
        (false, end + 1)
    } else {
        return None;
    };
    if !b.get(at).is_some_and(u8::is_ascii_uppercase) {
        return None;
    }
    if let Some((_, code)) = name_at(text, at) {
        return Some(code);
    }
    let code = b.get(at..at + 2)?;
    if !on_name_end(text, at + 2) {
        return None;
    }
    if !comma && !(b.get(at + 2) == Some(&b' ') && zip_end(text, at + 3).is_some()) {
        return None;
    }
    US_STATES.iter().find(|(_, c)| c.as_bytes() == code).map(|(_, c)| *c)
}

/// Whether the gazetteer's admin-1 name `admin` is a name of the state
/// with USPS code `code`.
fn is_state(admin: &str, code: &str) -> bool {
    US_STATES.iter().any(|(name, c)| *c == code && name.eq_ignore_ascii_case(admin))
}
