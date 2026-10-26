//! The patterns that cut a byte-level BPE's text into words, matched by
//! hand as upstream's regular expressions match them: GPT-2's, and the
//! Split pattern Llama-3 and Qwen2 put before their ByteLevel step.

use crate::classes::{LETTERS, NUMBERS};
use crate::manifest::BpePattern;

/// Llama-3's Split pattern, as its tokenizer file writes it.
pub const LLAMA3: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Qwen2's: Llama-3's with one digit at a time.
pub const QWEN2: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// `word(start, end)` for each match of `p`'s pattern in `s`. Only
/// GPT-2's matcher is made for each caller's `word`: the other two take
/// it as `dyn`, which costs them nothing measurable, where a copy of each
/// beside GPT-2's in every caller made GPT-2's path about 8% slower.
pub(crate) fn words(p: BpePattern, s: &str, mut word: impl FnMut(usize, usize)) {
    match p {
        BpePattern::Gpt2 => split(s, word),
        BpePattern::Llama3 => split_digits(s, 3, &mut word as &mut dyn FnMut(usize, usize)),
        BpePattern::Qwen2 => split_digits(s, 1, &mut word as &mut dyn FnMut(usize, usize)),
    }
}

/// What the pattern sees a character as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Class {
    Space,
    Letter,
    Number,
    Other,
}

pub(super) fn class(c: char) -> Class {
    if c.is_ascii() {
        return match c {
            'a'..='z' | 'A'..='Z' => Class::Letter,
            '0'..='9' => Class::Number,
            '\t' | '\n' | '\x0b' | '\x0c' | '\r' | ' ' => Class::Space,
            _ => Class::Other,
        };
    }
    let within = |runs: &[(u32, u32)]| {
        let u = c as u32;
        runs.binary_search_by(|&(a, b)| {
            if b < u {
                std::cmp::Ordering::Less
            } else if a > u {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
    };
    if c.is_whitespace() {
        Class::Space
    } else if within(LETTERS) {
        Class::Letter
    } else if within(NUMBERS) {
        Class::Number
    } else {
        Class::Other
    }
}

/// GPT-2's pattern, `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
/// ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, as upstream matches it: `word(start,
/// end)` for each match, which together cover the text. Public so that a
/// trainer cuts its corpus into the words this encoder merges.
pub fn split(s: &str, mut word: impl FnMut(usize, usize)) {
    let b = s.as_bytes();
    let at = |i: usize| s[i..].chars().next().expect("inside the text");
    // The end of the run of `class` from i.
    let run = |mut i: usize, k: Class| {
        while i < b.len() {
            let c = at(i);
            if class(c) != k {
                break;
            }
            i += c.len_utf8();
        }
        i
    };
    let mut i = 0;
    while i < b.len() {
        let c = at(i);
        // A contraction.
        if c == '\'' {
            let rest = &b[i + 1..];
            let len = if rest.starts_with(b"re") || rest.starts_with(b"ve") || rest.starts_with(b"ll") {
                2
            } else if matches!(rest.first(), Some(b's' | b't' | b'm' | b'd')) {
                1
            } else {
                0
            };
            if len > 0 {
                word(i, i + 1 + len);
                i += 1 + len;
                continue;
            }
        }
        // An optional space, then a run of letters, numbers or others.
        let j = if c == ' ' { i + 1 } else { i };
        if j < b.len() {
            let k = class(at(j));
            if k != Class::Space {
                let end = run(j, k);
                word(i, end);
                i = end;
                continue;
            }
        }
        // Whitespace: up to the last before a character that is not,
        // unless that leaves none.
        let end = run(i, Class::Space);
        if end < b.len() {
            let last = s[..end].chars().next_back().expect("a run of one at least").len_utf8();
            if end - last > i {
                word(i, end - last);
                i = end - last;
                continue;
            }
        }
        word(i, end);
        i = end;
    }
}

/// The end of the run of `k` from `i`.
fn run(s: &str, mut i: usize, k: Class) -> usize {
    while let Some(c) = s[i..].chars().next() {
        if class(c) != k {
            break;
        }
        i += c.len_utf8();
    }
    i
}

/// The bytes after an apostrophe that make a contraction of
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)`, or 0. Case folding makes U+017F LATIN
/// SMALL LETTER LONG S an s.
fn contraction(rest: &str) -> usize {
    let mut chars = rest.chars();
    let fold = |c: Option<char>| match c {
        Some('\u{17f}') => 's',
        Some(c) => c.to_ascii_lowercase(),
        None => '\0',
    };
    let first = chars.next();
    let two = (fold(first), fold(chars.next()));
    match two {
        ('r', 'e') | ('v', 'e') | ('l', 'l') => 2,
        ('s' | 't' | 'm' | 'd', _) => first.map_or(0, char::len_utf8),
        _ => 0,
    }
}

/// Llama-3's pattern with runs of up to `digits` numbers (Qwen2's with
/// one): `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|
/// \p{N}{1,digits}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
/// as upstream matches it: the first alternative that matches at each
/// place, each as long as it can be.
pub fn split_digits(s: &str, digits: usize, mut word: impl FnMut(usize, usize)) {
    let b = s.as_bytes();
    let newline = |c: char| c == '\r' || c == '\n';
    let mut i = 0;
    while i < b.len() {
        let c = s[i..].chars().next().expect("inside the text");
        let k = class(c);
        let mut end = i;
        if c == '\'' {
            let n = contraction(&s[i + 1..]);
            if n > 0 {
                end = i + 1 + n;
            }
        }
        if end == i && k == Class::Letter {
            end = run(s, i, Class::Letter);
        }
        // One character that is no letter, number or line break, then
        // letters.
        if end == i && k != Class::Number && !newline(c) {
            let j = i + c.len_utf8();
            if s[j..].chars().next().is_some_and(|d| class(d) == Class::Letter) {
                end = run(s, j, Class::Letter);
            }
        }
        if end == i && k == Class::Number {
            end = s[i..]
                .char_indices()
                .take_while(|&(_, d)| class(d) == Class::Number)
                .take(digits)
                .last()
                .map_or(i, |(at, d)| i + at + d.len_utf8());
        }
        // An optional space, others, then line breaks.
        if end == i {
            let j = if c == ' ' { i + 1 } else { i };
            if s[j..].chars().next().is_some_and(|d| class(d) == Class::Other) {
                end = run(s, j, Class::Other);
                while end < b.len() && matches!(b[end], b'\r' | b'\n') {
                    end += 1;
                }
            }
        }
        if end == i {
            // Whitespace: up to its last line break; else up to the last
            // before a character that is not, unless that leaves none.
            let ws = run(s, i, Class::Space);
            end = match s[i..ws].rfind(['\r', '\n']) {
                Some(at) => i + at + 1,
                None if ws < b.len() => {
                    let last = s[..ws].chars().next_back().expect("a run of one at least").len_utf8();
                    if ws - last > i { ws - last } else { ws }
                }
                None => ws,
            };
        }
        word(i, end);
        i = end;
    }
}
