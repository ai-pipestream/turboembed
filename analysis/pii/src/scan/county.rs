//! United States counties, `UsCountyPiiExtractor`: a county or county
//! equivalent of the Census Bureau's list with its own last word
//! (`Cook County`, `Orleans Parish`, `Matanuska-Susitna Borough`,
//! `Bethel Census Area`, `Mayagüez Municipio`), at word boundaries with
//! single spaces between its words. Each word is read as the table
//! writes it or all in capitals, the last also in small letters
//! (`Cook county`), and a right single quotation mark reads as an
//! apostrophe (`Prince George’s County`). A name is compared in
//! normalization form C, so a decomposed spelling (`Doña Ana County` with
//! `n` + U+0303) reads as the precomposed one. The longest name ending at
//! a suffix wins. The normalized form is the table's spelling.

use std::borrow::Cow;

use turbo_analysis_layers::{PiiEvidence, PiiKind};
use turbo_unicode::normalize::{Form, is_normalized, normalize};

use crate::hits::Hits;
use crate::tables::us_counties::{MAX_WORDS, SUFFIXES, county};
use crate::text::{name_char, on_name_end, on_name_start, on_word_start};

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let Some(end) = suffix_at(text, i) else {
            i += 1;
            continue;
        };
        if let Some((start, name)) = name_before(text, i, end) {
            hits.add(start, end, PiiKind::County, PiiEvidence::Shape, |s| s.push_str(name));
        }
        i = end;
    }
}

/// The end of a county suffix word at `i`, in any case.
fn suffix_at(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    if !matches!(b[i] | 0x20, b'c' | b'p' | b'b' | b'a' | b'm') || !on_word_start(text, i) {
        return None;
    }
    SUFFIXES.iter().find_map(|s| {
        let end = i + s.len();
        (b.get(i..end).is_some_and(|w| w.eq_ignore_ascii_case(s.as_bytes())) && on_name_end(text, end)).then_some(end)
    })
}

/// The longest county name ending at `end` whose last word starts at
/// `suffix`: its start and the table's spelling.
fn name_before(text: &str, suffix: usize, end: usize) -> Option<(usize, &'static str)> {
    // The starts of the words before the suffix, nearest first.
    let mut starts = [0; MAX_WORDS - 1];
    let mut n = 0;
    let mut at = suffix;
    while n < starts.len() && at >= 2 && text.as_bytes()[at - 1] == b' ' {
        let word_end = at - 1;
        // A word runs back over letters, digits, combining marks, `_` and
        // the marks inside a name (`St.`, `O'Brien`, `Miami-Dade`); anything
        // else ends it (`County:Cook County`, a no-break space). So
        // `x_Cook County` and `e\u{301}Cook County` hold no name, as
        // `\u{e9}Cook County` holds none.
        let mut start = text[..word_end]
            .char_indices()
            .rev()
            .find(|&(_, c)| !(name_char(c) || matches!(c, '\'' | '\u{2019}' | '.' | '-')))
            .map_or(0, |(p, c)| p + c.len_utf8());
        // Leading punctuation (`(Cook County`) is not part of the name.
        let Some(lead) = text[start..word_end].find(|c: char| c.is_alphanumeric()) else { break };
        start += lead;
        starts[n] = start;
        n += 1;
        at = start;
    }
    // Every name of the table starts with a capital, and so does its word
    // before the suffix but for `and` (`Juneau City and Borough`). A word
    // is read as the table writes it or in capitals, so text that breaks
    // this (`In the area`, `the county`) is no name, with no search.
    let nearest = *starts[..n].first()?;
    if !capital_at(text, nearest) && !text.as_bytes()[nearest..suffix - 1].eq_ignore_ascii_case(b"and") {
        return None;
    }
    starts[..n].iter().rev().filter(|&&start| capital_at(text, start)).find_map(|&start| {
        let written = nfc(&text[start..end]);
        let name = county(&written)?;
        (on_name_start(text, start) && cased_like(&written, name)).then_some((start, name))
    })
}

/// `written` in normalization form C, as the table is written, so a
/// decomposed spelling (`n` + U+0303 for `ñ`) finds its name. The mention
/// keeps the bytes of the source; text already in NFC is not copied.
fn nfc(written: &str) -> Cow<'_, str> {
    if written.is_ascii() || is_normalized(Form::Nfc, written) {
        return Cow::Borrowed(written);
    }
    let mut out = String::new();
    normalize(Form::Nfc, written, &mut out);
    Cow::Owned(out)
}

/// Whether each word of `written` is the word of `name` as the table
/// writes it or in capitals, the last also in small letters.
fn cased_like(written: &str, name: &str) -> bool {
    let words = name.split(' ').count();
    written.split(' ').zip(name.split(' ')).enumerate().all(|(k, (w, n))| {
        let w = w.replace('\u{2019}', "'");
        w == n || w == n.to_uppercase() || (k + 1 == words && w == n.to_lowercase())
    })
}

/// Whether a capital letter starts at `i`.
fn capital_at(text: &str, i: usize) -> bool {
    text[i..].chars().next().is_some_and(char::is_uppercase)
}
