//! The verdict on one full stop: the token around it, the entries it
//! matches and the word after it.

use super::{AbbreviationClass, AbbreviationInventory, ContextRule, Verdict};

/// The longest token, in bytes, the inventory looks at; a longer run
/// without whitespace is no abbreviation, and the bound keeps a text
/// without whitespace linear.
pub(super) const MAX_TOKEN: usize = 64;

/// Opening marks stripped from the start of a token.
const OPENERS: &[char] = &['"', '\'', '`', '(', '[', '{', '\u{AB}', '\u{2018}', '\u{201C}'];
/// Closing marks and punctuation stripped from the end of a token.
const CLOSERS: &[char] = &['"', '\'', ')', ']', '}', '\u{BB}', '\u{2019}', '\u{201D}', ',', ';', ':', '!', '?'];

/// A whitespace-separated token: the bounds of the run, and its start
/// and end without the marks around it.
#[derive(Copy, Clone, Debug)]
struct Token {
    raw_start: usize,
    start: usize,
    end: usize,
    raw_end: usize,
}

/// The token holding the byte `at`, which is not whitespace.
fn token_at(text: &str, at: usize) -> Option<Token> {
    let mut start = at;
    for (i, c) in text[..at].char_indices().rev() {
        if c.is_whitespace() {
            break;
        }
        start = i;
        if at - start > MAX_TOKEN {
            return None;
        }
    }
    let mut raw_end = at;
    for (i, c) in text[at..].char_indices() {
        if c.is_whitespace() {
            break;
        }
        raw_end = at + i + c.len_utf8();
        if raw_end - start > MAX_TOKEN {
            return None;
        }
    }
    let mut s = start;
    while let Some(c) = text[s..raw_end].chars().next().filter(|c| OPENERS.contains(c)) {
        s += c.len_utf8();
    }
    let mut e = raw_end;
    while let Some(c) = text[s..e].chars().next_back().filter(|c| CLOSERS.contains(c)) {
        e -= c.len_utf8();
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
    roman_value(roman).is_some_and(|v| to_roman(v).eq_ignore_ascii_case(roman))
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

fn roman_value(s: &str) -> Option<u32> {
    let digit = |b: u8| match b.to_ascii_uppercase() {
        b'I' => Some(1),
        b'V' => Some(5),
        b'X' => Some(10),
        b'L' => Some(50),
        b'C' => Some(100),
        b'D' => Some(500),
        b'M' => Some(1000),
        _ => None,
    };
    let values: Option<Vec<u32>> = s.bytes().map(digit).collect();
    let values = values?;
    let mut total = 0;
    for (i, &v) in values.iter().enumerate() {
        if values.get(i + 1).is_some_and(|&n| n > v) { total -= v as i64 } else { total += v as i64 }
    }
    u32::try_from(total).ok().filter(|&v| (1..4000).contains(&v))
}

fn to_roman(mut v: u32) -> String {
    let mut out = String::new();
    for (sym, n) in ROMAN {
        while v >= n {
            out.push_str(sym);
            v -= n;
        }
    }
    out
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
        if period + 1 != tok.end {
            // A stop inside a one-token entry ends nothing.
            return if self.lookup(full).is_some() { Verdict::Suppress } else { Verdict::Neutral };
        }
        let verdict = match self.lookup(full) {
            Some(rule) => apply(rule, &text[period + 1..]),
            None => self.multi(text, tok).unwrap_or(Verdict::Neutral),
        };
        if verdict != Verdict::Neutral {
            return verdict;
        }
        if self.is_word(&full[..full.len() - 1]) { Verdict::Allow } else { Verdict::Neutral }
    }

    /// The rule of a one-token entry `token`, as written or with its first
    /// letter in the other case.
    pub(super) fn lookup(&self, token: &str) -> Option<ContextRule> {
        let mut buf = [0; MAX_TOKEN];
        self.find(token).or_else(|| fold_first(token, &mut buf).and_then(|t| self.find(t)))
    }

    fn excluded(&self, text: &str) -> bool {
        self.exclusions.binary_search_by(|t| (**t).cmp(text)).is_ok()
    }

    fn find(&self, token: &str) -> Option<ContextRule> {
        if self.excluded(token) {
            return None;
        }
        if let Ok(i) = self.additions.binary_search_by(|(t, _)| (**t).cmp(token)) {
            return Some(self.additions[i].1);
        }
        let bundle = self.bundle?;
        AbbreviationClass::ALL.into_iter().filter(|c| self.classes.contains(*c)).find_map(|c| {
            let table = bundle.classes[c as usize];
            table.binary_search_by(|a| a.text.cmp(token)).ok().map(|i| table[i].rule)
        })
    }

    /// The multi-token entries, additions first.
    fn multis(&self) -> impl Iterator<Item = (&str, ContextRule)> + '_ {
        let added = self.additions.iter().map(|(t, r)| (&**t, *r));
        let bundled = self.bundled().map(|(_, a)| (a.text, a.rule));
        added.chain(bundled).filter(|(t, _)| t.contains(' ') && !self.excluded(t))
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
        for (entry, rule) in self.multis() {
            let n = entry.split(' ').count();
            for (k, part) in entry.split(' ').enumerate() {
                if !same(part, full, k == 0, folded) {
                    continue;
                }
                let mut at = tok.raw_start;
                let before = entry.split(' ').take(k).collect::<Vec<_>>();
                let back = before.iter().enumerate().rev().all(|(j, want)| {
                    let Some(p) = prev_token(text, at) else { return false };
                    at = p.raw_start;
                    let mut b = [0; MAX_TOKEN];
                    let word = &text[p.start..p.end];
                    same(want, word, j == 0, fold_first(word, &mut b))
                });
                let mut at = tok.raw_end;
                let forward = entry.split(' ').skip(k + 1).all(|want| {
                    let Some(t) = next_token(text, at) else { return false };
                    at = t.raw_end;
                    &text[t.start..t.end] == want
                });
                if back && forward {
                    let last = k + 1 == n;
                    return Some(if last { apply(rule, &text[tok.end..]) } else { Verdict::Suppress });
                }
            }
        }
        None
    }

    /// Whether `word` is never an abbreviation.
    fn is_word(&self, word: &str) -> bool {
        let Some(bundle) = self.bundle.filter(|_| !self.classes.is_empty()) else { return false };
        if word.len() > 8 || word.contains('.') {
            return false;
        }
        let lower = word.to_ascii_lowercase();
        !self.excluded(&lower) && bundle.penultimates.binary_search(&lower.as_str()).is_ok()
    }
}
