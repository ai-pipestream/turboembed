//! Bare city names in running prose (#2097): a gazetteer name with no
//! state after it and no label before it (`lunch in Rouen`, `the Boston
//! area`, `a Pasadena native`), read only where it is unlikely to be a
//! word, a person or part of a longer name:
//!
//! - it is not a common English word or a common given or family name
//!   ([`city_common_words`], `Mobile`, `Reading`, `Jackson`), not a US
//!   state's name (`Virginia`, `New York`, which the state scanner reads),
//!   not a country's English name (`Luxembourg`, `Kuwait`, `Djibouti`),
//!   and a one-word name has four letters or more (`Van`, `San`, `Po`);
//! - the gazetteer holds a place of that name with at least
//!   [`MIN_POPULATION`] people, whose name is the normalized form;
//! - no state follows it, as the anchored readings decide those
//!   (`Mobile, IL` stays refused);
//! - no capitalized word follows it after a space, its possessive `'s`
//!   included (`The San Francisco Chronicle`, `Orleans Parish`, `Prince
//!   George's County`, `New York City`, `Alice Waters`);
//! - the word before it, after one space, is not capitalized, or is a
//!   word that starts a sentence before a place ([`LEAD_WORDS`], `In
//!   Rouen`, `The Boston area`), so `Harry Denton` is a person, and is not
//!   `of` after such a word (`University of Bristol`);
//! - no verb of speech follows it ([`SPEECH_VERBS`], `Madison said`), as
//!   a person's name does;
//! - it is not joined to an address, a path or a word (`example.com/Paris`,
//!   `Paris.example`, `Grand-Rouen`).
//!
//! A possessive name ends before its `'s` (`San Francisco's`).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::OnceLock;

use turbo_geo::{BUNDLED_TABLE, Folder};

use super::city::{capital_at, capitals_like_a_name, lookup, state_after, word_ends};
use crate::tables::city_common_words;
use crate::tables::us_states::US_STATES;
use crate::text::{name_char, on_name_end};

/// The capitalized words that may stand right before a bare city name:
/// the prepositions, articles, conjunctions and determiners a sentence
/// starts with before a place (`In Paris`, `From London`).
const LEAD_WORDS: [&str; 40] = [
    "a", "across", "after", "an", "and", "around", "as", "at", "before", "between", "both", "but", "by", "during",
    "every", "from", "her", "his", "in", "inside", "into", "its", "like", "my", "near", "of", "on", "or", "our",
    "outside", "since", "that", "the", "their", "this", "through", "to", "toward", "via", "your",
];

/// The fewest people a place has for its name to be read bare: smaller
/// places' names are more often words of other languages, surnames or
/// the country names the Antarctic stations carry (`Ende`, `Nord`,
/// `Australia` for Casey Station).
const MIN_POPULATION: u64 = 100_000;

/// The most words a city name holds, as the anchored readings count them.
const MAX_WORDS: usize = 5;

/// Adds each bare city name of `text` to `found`.
pub(super) fn scan(text: &str, found: &mut Vec<(usize, usize, &'static str)>) {
    let b = text.as_bytes();
    let Names { prefixes, starts } = names();
    let mut folder = Folder::new();
    let mut key = String::new();
    let mut at = 0;
    while let Some(p) = b[at..].iter().position(|&c| c.is_ascii_uppercase() || c >= 0xC0) {
        super::step(p + 1);
        let start = at + p;
        at = start + 1;
        if trigram(&b[start..]).is_some_and(|t| starts[t / 64] & 1 << (t % 64) == 0)
            || !capital_at(text, start)
            || !free_start(text, start)
        {
            continue;
        }
        // The first word alone decides most capitals, so the rest of the
        // words are walked only when it starts a name. A word of ASCII
        // letters and digits that a space or a plain mark ends folds to
        // itself in lower case, with no walk.
        let run = b[start..].iter().take_while(|c| c.is_ascii_alphanumeric()).count();
        let mut ends = [0; MAX_WORDS];
        if b.get(start + run).is_none_or(|&c| c.is_ascii() && !matches!(c, b'\'' | b'.' | b'-' | b'_')) {
            key.clear();
            key.extend(b[start..start + run].iter().map(|&c| char::from(c.to_ascii_lowercase())));
        } else {
            if word_ends(text, start, &mut ends[..1]) == 0 {
                continue;
            }
            folder.fold_into(&text[start..possessive_stem(text, start, ends[0])], &mut key);
        }
        if !prefixes.contains_key(key.as_str()) {
            continue;
        }
        let n = word_ends(text, start, &mut ends);
        // The ends of the names the words from `start` spell, shortest
        // first, read word by word while they still start a name.
        let mut full = [0; MAX_WORDS];
        let mut m = 0;
        for &end in &ends[..n] {
            let end = possessive_stem(text, start, end);
            super::step(end - start);
            folder.fold_into(&text[start..end], &mut key);
            match prefixes.get(key.as_str()) {
                None => break,
                Some(&is_name) => {
                    if is_name {
                        full[m] = end;
                        m += 1;
                    }
                }
            }
        }
        // The longest name written like one, as `lookup` would take it;
        // the gazetteer is asked only for one the context accepts.
        let Some(&end) =
            full[..m].iter().rev().find(|&&end| on_name_end(text, end) && capitals_like_a_name(&text[start..end]))
        else {
            continue;
        };
        if !accepted(text, start, end) {
            continue;
        }
        if let Some(entry) =
            lookup(text, start, end).and_then(|f| f.into_iter().find(|e| e.population() >= MIN_POPULATION))
        {
            found.push((start, end, entry.name()));
            at = end;
        }
    }
}

/// The names read bare and the first letters they start with.
struct Names {
    /// The match keys of the names (and alternate names) of the places
    /// with at least [`MIN_POPULATION`] people, and every run of their
    /// first words, each mapped to whether it is a whole name read bare
    /// (none [`left_out`]): a name is looked up in the gazetteer only once
    /// its words spell one.
    prefixes: HashMap<String, bool, BuildHasherDefault<WordHasher>>,
    /// A bit per [`trigram`] of those names as the table writes them and
    /// of their keys, so a capitalized word that starts none costs one
    /// probe, with no fold.
    starts: Box<[u64]>,
}

/// A multiply-rotate hash of a short key (the Fx hash), several times
/// cheaper than the default SipHash on one word; the keys are the
/// gazetteer's, never a caller's, so no flooding defence is needed.
#[derive(Default)]
struct WordHasher(u64);

impl Hasher for WordHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 = (self.0.rotate_left(5) ^ u64::from_le_bytes(word)).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }

    fn write_u8(&mut self, byte: u8) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(byte)).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// The first three bytes of `bytes` as one of 27 * 27 * 27 numbers, a
/// letter in either case as itself and any other ASCII byte or the end as
/// a 27th symbol; none when a byte is not ASCII, as its fold may differ.
fn trigram(bytes: &[u8]) -> Option<usize> {
    let mut t = 0;
    for i in 0..3 {
        let c = bytes.get(i).copied().unwrap_or(b' ');
        if !c.is_ascii() {
            return None;
        }
        let symbol = if c.is_ascii_alphabetic() { usize::from((c | 0x20) - b'a') } else { 26 };
        t = t * 27 + symbol;
    }
    Some(t)
}

fn names() -> &'static Names {
    static NAMES: OnceLock<Names> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut folder = Folder::new();
        let mut prefixes = HashMap::default();
        let states: Vec<String> = US_STATES.iter().map(|(name, _)| folder.fold(name)).collect();
        let mut starts = vec![0u64; (27 * 27 * 27usize).div_ceil(64)].into_boxed_slice();
        let mut mark = |bytes: &[u8]| {
            if let Some(t) = trigram(bytes) {
                starts[t / 64] |= 1 << (t % 64);
            }
        };
        for row in BUNDLED_TABLE.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = row.split(';').collect();
            if f.len() < 9 || f[8].parse::<u64>().map_or(true, |people| people < MIN_POPULATION) {
                continue;
            }
            for name in std::iter::once(f[2]).chain(f[3].split('|')).filter(|n| !n.is_empty()) {
                let key = folder.fold(name);
                mark(name.as_bytes());
                mark(key.as_bytes());
                let mut prefix = String::new();
                for word in key.split(' ') {
                    if !prefix.is_empty() {
                        prefix.push(' ');
                    }
                    prefix.push_str(word);
                    prefixes.entry(prefix.clone()).or_insert(false);
                }
                let is_name = !left_out(&key, &states);
                *prefixes.entry(key).or_insert(false) |= is_name;
            }
        }
        Names { prefixes, starts }
    })
}

/// Whether the match key `key` names no bare city: a US state's name
/// (`virginia`, `new york`), a country's English name (`kuwait`), or one
/// word that is a [`city_common_words`] word or has fewer than four
/// letters (`van`, `po`).
fn left_out(key: &str, states: &[String]) -> bool {
    if states.iter().any(|s| s == key) || turbo_geo::english_country(key).is_some() {
        return true;
    }
    !key.contains(' ') && (key.chars().count() < 4 || city_common_words::contains(key))
}

/// Whether the name `text[start..end]` that [`Names::prefixes`] holds is read
/// bare.
fn accepted(text: &str, start: usize, end: usize) -> bool {
    state_after(text, end).is_none()
        && free_end(text, end)
        && !after_name_word(text, start)
        && !before_speech(text, end)
}

/// The verbs of speech and address a person's name stands before and a
/// city's seldom does (`Madison said`, `Paris told her`).
const SPEECH_VERBS: [&str; 12] =
    ["added", "asked", "replied", "said", "says", "smiled", "nodded", "told", "tells", "wrote", "writes", "laughed"];

/// Whether one of [`SPEECH_VERBS`] follows the name at `end` after one
/// space, so the name is a person's (`Columbus said`, `Salem wrote`).
fn before_speech(text: &str, end: usize) -> bool {
    let Some(rest) = text[end..].strip_prefix(' ') else {
        return false;
    };
    let word = &rest[..rest.find(|c: char| !c.is_ascii_alphabetic()).unwrap_or(rest.len())];
    SPEECH_VERBS.contains(&word)
}

/// Whether nothing joins the start of a name at `start` to a word, a
/// path or an address before it (`-Salem`, `/Paris`, `.Paris`, `'s`).
fn free_start(text: &str, start: usize) -> bool {
    !text[..start].chars().next_back().is_some_and(|c| {
        name_char(c) || matches!(c, '\'' | '\u{2019}' | '-' | '.' | '/' | '@' | '#' | '&' | '\\' | '=' | '+')
    })
}

/// Whether the name ending at `end` stands alone: past its possessive
/// `'s`, no space and capitalized word follows (`San Francisco
/// Chronicle`), and no path, address or domain goes on (`Paris/`,
/// `Paris@`, `Paris.example`).
fn free_end(text: &str, end: usize) -> bool {
    let rest = &text[end..];
    let rest = rest.strip_prefix("'s").or_else(|| rest.strip_prefix("\u{2019}s")).unwrap_or(rest);
    let at = text.len() - rest.len();
    let mut chars = rest.chars();
    match chars.next() {
        Some(' ') => !capital_at(text, at + 1),
        Some('/' | '@' | '#' | '\\' | '_') => false,
        Some('.') => !chars.next().is_some_and(name_char),
        _ => true,
    }
}

/// Whether the word before `start`, after one space, is capitalized and
/// not a [`LEAD_WORDS`] word, so the name is part of a longer one or a
/// person's (`Harry Denton`, `Alex Van`, `Mr. Lincoln`), or is `of` after
/// such a word (`University of Bristol`, `Bank of Rome`).
fn after_name_word(text: &str, start: usize) -> bool {
    let name_word = |w: Option<(usize, &str)>| {
        w.is_some_and(|(at, w)| capital_at(text, at) && !LEAD_WORDS.iter().any(|l| l.eq_ignore_ascii_case(w)))
    };
    match word_before(text, start) {
        Some((at, "of")) => name_word(word_before(text, at)),
        w => name_word(w),
    }
}

/// The start and text of the word that ends one space before `at`, its
/// inner marks included (`Mr.`, `O'Neil`); the word is empty after two
/// spaces or a mark.
fn word_before(text: &str, at: usize) -> Option<(usize, &str)> {
    let before = text[..at].strip_suffix(' ')?;
    let start = before
        .char_indices()
        .rev()
        .find(|&(_, c)| {
            super::step(1);
            !(name_char(c) || matches!(c, '\'' | '\u{2019}' | '.' | '-'))
        })
        .map_or(0, |(p, c)| p + c.len_utf8());
    Some((start, &before[start..]))
}

/// `end`, or the end before a possessive `'s` that ends the word there.
fn possessive_stem(text: &str, start: usize, end: usize) -> usize {
    if end <= start {
        return end;
    }
    let word = &text[start..end];
    word.strip_suffix("'s").or_else(|| word.strip_suffix("\u{2019}s")).map_or(end, |w| start + w.len())
}
