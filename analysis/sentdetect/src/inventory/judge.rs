//! The verdict on one full stop: the token around it, the entries it
//! matches and the word after it.

use super::index::{Entry, keys, probe};
use super::{AbbreviationInventory, ContextRule, Verdict};

/// The longest token, in bytes, the inventory looks at; a longer run
/// without whitespace is no abbreviation, and the bound keeps a text
/// without whitespace linear.
pub(super) const MAX_TOKEN: usize = 64;

/// Opening marks stripped from the start of a token.
const OPENERS: &[char] = &['"', '\'', '`', '(', '[', '{', '\u{AB}', '\u{2018}', '\u{201C}'];
/// Closing marks and punctuation stripped from the end of a token.
const CLOSERS: &[char] = &['"', '\'', ')', ']', '}', '\u{BB}', '\u{2019}', '\u{201D}', ',', ';', ':', '!', '?'];

/// Which ASCII bytes are among `marks`.
const fn ascii_table(marks: &[char]) -> [bool; 128] {
    let mut table = [false; 128];
    let mut i = 0;
    while i < marks.len() {
        if marks[i].is_ascii() {
            table[marks[i] as usize] = true;
        }
        i += 1;
    }
    table
}

const ASCII_OPENERS: [bool; 128] = ascii_table(OPENERS);
const ASCII_CLOSERS: [bool; 128] = ascii_table(CLOSERS);

/// Whether the ASCII byte `b` is whitespace as `char::is_whitespace`
/// has it.
fn ascii_space(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | 0x0B | 0x0C | b'\r' | b' ')
}

/// A whitespace-separated token: the bounds of the run, and its start
/// and end without the marks around it.
#[derive(Copy, Clone, Debug)]
struct Token {
    raw_start: usize,
    start: usize,
    end: usize,
    raw_end: usize,
}

/// The token holding the byte `at`, which is not whitespace. ASCII is
/// read byte by byte; only a byte from 0x80 up is decoded.
fn token_at(text: &str, at: usize) -> Option<Token> {
    let b = text.as_bytes();
    let mut start = at;
    while start > 0 {
        let c = b[start - 1];
        let i = if c < 0x80 {
            if ascii_space(c) {
                break;
            }
            start - 1
        } else {
            let (i, ch) = text[..start].char_indices().next_back()?;
            if ch.is_whitespace() {
                break;
            }
            i
        };
        start = i;
        if at - start > MAX_TOKEN {
            return None;
        }
    }
    let mut raw_end = at;
    while raw_end < b.len() {
        let c = b[raw_end];
        raw_end += if c < 0x80 {
            if ascii_space(c) {
                break;
            }
            1
        } else {
            let ch = text[raw_end..].chars().next()?;
            if ch.is_whitespace() {
                break;
            }
            ch.len_utf8()
        };
        if raw_end - start > MAX_TOKEN {
            return None;
        }
    }
    let mut s = start;
    while s < raw_end {
        let c = b[s];
        s += if c < 0x80 {
            if !ASCII_OPENERS[c as usize] {
                break;
            }
            1
        } else {
            match text[s..raw_end].chars().next() {
                Some(ch) if OPENERS.contains(&ch) => ch.len_utf8(),
                _ => break,
            }
        };
    }
    let mut e = raw_end;
    while e > s {
        let c = b[e - 1];
        e -= if c < 0x80 {
            if !ASCII_CLOSERS[c as usize] {
                break;
            }
            1
        } else {
            match text[s..e].chars().next_back() {
                Some(ch) if CLOSERS.contains(&ch) => ch.len_utf8(),
                _ => break,
            }
        };
    }
    Some(Token { raw_start: start, start: s, end: e, raw_end })
}

/// The token before byte `at`, past the whitespace there.
fn prev_token(text: &str, at: usize) -> Option<Token> {
    let (i, _) = text[..at].char_indices().rev().find(|(_, c)| !c.is_whitespace())?;
    if at - i > MAX_TOKEN {
        return None;
    }
    token_at(text, i)
}

/// The token after byte `at`, past the whitespace there.
fn next_token(text: &str, at: usize) -> Option<Token> {
    let (i, _) = text[at..].char_indices().find(|(_, c)| !c.is_whitespace())?;
    if i > MAX_TOKEN {
        return None;
    }
    token_at(text, at + i)
}

/// `token` with its first letter's ASCII case flipped, in `buf`.
fn fold_first<'b>(token: &str, buf: &'b mut [u8; MAX_TOKEN]) -> Option<&'b str> {
    let first = *token.as_bytes().first()?;
    if !first.is_ascii_alphabetic() || token.len() > MAX_TOKEN {
        return None;
    }
    let out = &mut buf[..token.len()];
    out.copy_from_slice(token.as_bytes());
    out[0] = if first.is_ascii_uppercase() { first.to_ascii_lowercase() } else { first.to_ascii_uppercase() };
    std::str::from_utf8(out).ok()
}

/// Whether `word` (the text after the stop and its whitespace and opening
/// marks) starts with a number: digits, maybe after `#` or `§`, or a
/// Roman numeral in canonical form, other than a lone `I` or a single
/// lower case letter.
fn starts_number(word: &str) -> bool {
    let w = word.trim_start_matches(['#', '\u{A7}']);
    if w.starts_with(|c: char| c.is_ascii_digit()) {
        return true;
    }
    let len = w.find(|c: char| !c.is_ascii_alphabetic()).unwrap_or(w.len());
    let roman = &w[..len];
    if w[len..].starts_with(|c: char| c.is_alphanumeric()) || roman.is_empty() {
        return false;
    }
    let upper = roman.bytes().all(|b| b.is_ascii_uppercase());
    let lower = roman.bytes().all(|b| b.is_ascii_lowercase());
    if !(upper || lower) || roman == "I" || (lower && roman.len() == 1) {
        return false;
    }
    roman_value(roman).is_some_and(|v| spells_roman(v, roman))
}

const ROMAN: [(&str, u32); 13] = [
    ("M", 1000),
    ("CM", 900),
    ("D", 500),
    ("CD", 400),
    ("C", 100),
    ("XC", 90),
    ("L", 50),
    ("XL", 40),
    ("X", 10),
    ("IX", 9),
    ("V", 5),
    ("IV", 4),
    ("I", 1),
];

fn roman_digit(b: u8) -> Option<i64> {
    match b.to_ascii_uppercase() {
        b'I' => Some(1),
        b'V' => Some(5),
        b'X' => Some(10),
        b'L' => Some(50),
        b'C' => Some(100),
        b'D' => Some(500),
        b'M' => Some(1000),
        _ => None,
    }
}

fn roman_value(s: &str) -> Option<u32> {
    let b = s.as_bytes();
    let mut total = 0;
    for (i, &d) in b.iter().enumerate() {
        let v = roman_digit(d)?;
        if b.get(i + 1).and_then(|&n| roman_digit(n)).is_some_and(|n| n > v) { total -= v } else { total += v }
    }
    u32::try_from(total).ok().filter(|&v| (1..4000).contains(&v))
}

/// Whether `s` is `v` in canonical Roman numerals, in either case.
fn spells_roman(mut v: u32, s: &str) -> bool {
    let s = s.as_bytes();
    let mut at = 0;
    for (sym, n) in ROMAN {
        while v >= n {
            let sym = sym.as_bytes();
            if !s.get(at..at + sym.len()).is_some_and(|w| w.eq_ignore_ascii_case(sym)) {
                return false;
            }
            at += sym.len();
            v -= n;
        }
    }
    at == s.len()
}

/// The verdict of a rule on what follows an abbreviation's final stop,
/// `rest` being the text after it.
fn apply(rule: ContextRule, rest: &str) -> Verdict {
    match rest.chars().next() {
        None => return Verdict::Neutral,
        Some(',' | ';' | ':') => return Verdict::Suppress,
        Some(_) => {}
    }
    let word = rest.trim_start_matches(|c: char| c.is_whitespace() || OPENERS.contains(&c));
    let Some(c) = word.chars().next() else { return Verdict::Neutral };
    let suppress = match rule {
        ContextRule::Prepositive => c.is_alphanumeric(),
        ContextRule::Numeric => starts_number(word),
        ContextRule::Trailing => c.is_lowercase() || c.is_ascii_digit(),
    };
    if suppress { Verdict::Suppress } else { Verdict::Neutral }
}

impl AbbreviationInventory {
    /// The verdict on the full stop at byte `period` of `text`; anything
    /// but a full stop there is [`Verdict::Neutral`].
    pub fn judge(&self, text: &str, period: usize) -> Verdict {
        if self.is_empty() || text.as_bytes().get(period) != Some(&b'.') {
            return Verdict::Neutral;
        }
        let Some(tok) = token_at(text, period) else { return Verdict::Neutral };
        if period < tok.start || period >= tok.end {
            return Verdict::Neutral;
        }
        let full = &text[tok.start..tok.end];
        let last = period + 1 == tok.end;
        // A word ending in a lower case letter and longer than every entry,
        // part and never-abbreviation word that can end so has no verdict.
        if last
            && full.len() >= 2
            && full.as_bytes()[full.len() - 2].is_ascii_lowercase()
            && self.index().too_long_lower(full.len())
        {
            return Verdict::Neutral;
        }
        // Most words are no entry and no part of one, which the filter
        // tells without a search.
        let (entry_key, word_key) = keys(full);
        let known = self.index().may_match(full.len(), entry_key);
        if !last {
            // A stop inside a one-token entry ends nothing.
            return if known && self.lookup(full).is_some() { Verdict::Suppress } else { Verdict::Neutral };
        }
        let verdict = if !known {
            Verdict::Neutral
        } else {
            probe(0, 1);
            match self.lookup(full) {
                Some(rule) => apply(rule, &text[period + 1..]),
                None => self.multi(text, tok).unwrap_or(Verdict::Neutral),
            }
        };
        if verdict != Verdict::Neutral {
            return verdict;
        }
        if self.is_word(&full[..full.len() - 1], word_key) { Verdict::Allow } else { Verdict::Neutral }
    }

    /// The rule of a one-token entry `token`, as written or with its first
    /// letter in the other case.
    pub(super) fn lookup(&self, token: &str) -> Option<ContextRule> {
        let mut buf = [0; MAX_TOKEN];
        self.index().single(token).or_else(|| fold_first(token, &mut buf).and_then(|t| self.index().single(t)))
    }

    pub(super) fn excluded(&self, text: &str) -> bool {
        self.exclusions.binary_search_by(|t| (**t).cmp(text)).is_ok()
    }

    /// The verdict of the first multi-token entry the token at `tok` is a
    /// part of, with the tokens around it.
    fn multi(&self, text: &str, tok: Token) -> Option<Verdict> {
        let full = &text[tok.start..tok.end];
        let mut buf = [0; MAX_TOKEN];
        let folded = fold_first(full, &mut buf);
        let same = |entry: &str, word: &str, first: bool, folded: Option<&str>| {
            entry == word || (first && folded == Some(entry))
        };
        for Entry { text: entry, rule, k, n } in self.index().entries(full, folded) {
            probe(1, 0);
            // The parts before the token, nearest first, against the
            // tokens before it.
            let mut at = tok.raw_start;
            let back = entry.rsplit(' ').skip(n - k).enumerate().all(|(i, want)| {
                let Some(p) = prev_token(text, at) else { return false };
                at = p.raw_start;
                let mut b = [0; MAX_TOKEN];
                let word = &text[p.start..p.end];
                same(want, word, i + 1 == k, fold_first(word, &mut b))
            });
            let mut at = tok.raw_end;
            let forward = back
                && entry.split(' ').skip(k + 1).all(|want| {
                    let Some(t) = next_token(text, at) else { return false };
                    at = t.raw_end;
                    &text[t.start..t.end] == want
                });
            if forward {
                return Some(if k + 1 == n { apply(rule, &text[tok.end..]) } else { Verdict::Suppress });
            }
        }
        None
    }

    /// Whether `word`, of [`word_key`](super::index::word_key)
    /// `word_key`, is never an abbreviation.
    fn is_word(&self, word: &str, word_key: usize) -> bool {
        let Some(bundle) = self.bundle.filter(|_| !self.classes.is_empty()) else { return false };
        if word.len() > 8 || word.contains('.') || !self.index().may_be_word(word_key) {
            return false;
        }
        let mut buf = [0; 8];
        let lower = &mut buf[..word.len()];
        lower.copy_from_slice(word.as_bytes());
        lower.make_ascii_lowercase();
        // Folding ASCII letters keeps the bytes UTF-8.
        let Ok(lower) = std::str::from_utf8(lower) else { return false };
        !self.excluded(lower) && bundle.penultimates.binary_search(&lower).is_ok()
    }
}
