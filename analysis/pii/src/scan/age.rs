//! A person's age in years, `AgePiiExtractor`: a number from 0 to 125
//! before `years old`, `year old`, `yrs old` or `yr old` (the pieces
//! joined by spaces or hyphens, `34-year-old`), before `y/o` or `y.o.`,
//! or after `age`, `age:`, `age of` or `aged`, in any case. The mention is
//! the number alone and its normalized value has no leading zeros, so a
//! rewrite can act on the number and keep the words.

use turbo_analysis_layers::{PiiEvidence, PiiKind};

use crate::hits::Hits;
use crate::text::{letter_or_digit, on_end, on_number_start, on_word_start};

const MAX_DIGITS: usize = 3;
const MAX_AGE: u32 = 125;
const YEARS: [&[u8]; 4] = [b"years", b"year", b"yrs", b"yr"];
/// Units after a number that is then not an age in years (`aged 3 months`).
const NOT_YEARS: [&[u8]; 8] = [b"months", b"month", b"weeks", b"week", b"days", b"day", b"hours", b"%"];

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let found = if b[i].is_ascii_digit() && on_number_start(text, i) {
            number(text, i).filter(|&(_, end)| before_years(text, end))
        } else if matches!(b[i], b'a' | b'A') && on_word_start(text, i) {
            after_age(text, i)
        } else {
            None
        };
        match found {
            Some((start, end)) => {
                hits.add(start, end, PiiKind::Age, PiiEvidence::Shape, |s| {
                    let digits = text[start..end].trim_start_matches('0');
                    s.push_str(if digits.is_empty() { "0" } else { digits });
                });
                i = end;
            }
            None => i += 1,
        }
    }
}

/// The digits of an age starting at `start`, ending on a boundary and not
/// inside a decimal or a grouped number.
fn number(text: &str, start: usize) -> Option<(usize, usize)> {
    let b = text.as_bytes();
    let n = b[start..].iter().take(MAX_DIGITS + 1).take_while(|c| c.is_ascii_digit()).count();
    let end = start + n;
    if !(1..=MAX_DIGITS).contains(&n) || !on_end(text, end) {
        return None;
    }
    if matches!(b.get(end), Some(b'.' | b',')) && b.get(end + 1).is_some_and(u8::is_ascii_digit) {
        return None;
    }
    let value: u32 = text[start..end].parse().ok()?;
    (value <= MAX_AGE).then_some((start, end))
}

/// The word in `words` that `b` holds at `at`, ending on a boundary.
fn word_at(text: &str, at: usize, words: &[&[u8]]) -> Option<usize> {
    let b = text.as_bytes();
    let w = words.iter().find(|w| b.len() >= at + w.len() && b[at..at + w.len()].eq_ignore_ascii_case(w))?;
    let end = at + w.len();
    (*w == b"%" || on_end(text, end)).then_some(end)
}

/// One space or hyphen at `at`.
fn joint(b: &[u8], at: usize) -> Option<usize> {
    matches!(b.get(at), Some(b' ' | b'-')).then_some(at + 1)
}

/// Whether the number ending at `end` is followed by `years old` (any of
/// [`YEARS`]) or `y/o`.
fn before_years(text: &str, end: usize) -> bool {
    let b = text.as_bytes();
    let Some(p) = joint(b, end) else { return false };
    if let Some(q) = word_at(text, p, &YEARS)
        && let Some(r) = joint(b, q)
        && word_at(text, r, &[b"old"]).is_some()
    {
        return true;
    }
    let p = if b[end] == b' ' { p } else { return false };
    let slash = b.len() >= p + 3 && b[p..p + 3].eq_ignore_ascii_case(b"y/o");
    let dots = b.len() >= p + 4 && b[p..p + 4].eq_ignore_ascii_case(b"y.o.");
    let stop = p + if dots { 4 } else { 3 };
    (slash && on_end(text, stop)) || (dots && !text[stop..].starts_with(|c: char| letter_or_digit(c)))
}

/// The age after `age`, `age:`, `age of` or `aged` at `start`.
fn after_age(text: &str, start: usize) -> Option<(usize, usize)> {
    let b = text.as_bytes();
    let mut p = word_at(text, start, &[b"aged", b"age"])?;
    let aged = p - start == 4;
    if !aged && b.get(p) == Some(&b':') {
        p += 1;
    }
    p = if b.get(p) == Some(&b' ') { p + 1 } else { return None };
    if !aged && let Some(q) = word_at(text, p, &[b"of"]) {
        p = if b.get(q) == Some(&b' ') { q + 1 } else { return None };
    }
    if !b.get(p).is_some_and(u8::is_ascii_digit) {
        return None;
    }
    let (s, e) = number(text, p)?;
    // A unit other than years after it: not an age in years.
    let unit =
        joint(b, e).and_then(|q| word_at(text, q, &NOT_YEARS)).is_some() || word_at(text, e, &NOT_YEARS).is_some();
    (!unit).then_some((s, e))
}
