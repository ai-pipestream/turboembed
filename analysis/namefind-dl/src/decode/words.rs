//! The CoNLL-style raw-text words `Decoding::Word` labels.
//! A run of text with no space in it is one
//! word, except that:
//!
//! - brackets, quotes, commas, colons, semicolons and question or
//!   exclamation marks are words of their own, including inside a run;
//!   commas and colons between numeric pre-tokens stay in the number;
//! - a full stop after it is a word of its own (`Macron.`), unless it ends
//!   an abbreviation: a word that already holds a full stop (`U.N.`), a
//!   single letter or digit (`C.`, `1.`), up to three digits (`12.`), or a
//!   common English title, company or month abbreviation (`Mr.`, `Inc.`);
//! - repeated trailing full stops are punctuation, including after names;
//! - a closing `'s` is a word of its own (`Merkel's` is `Merkel`, `'s`);
//! - each CJK ideograph is a word, as BERT's normalizer spaces them.
//!
//! Hyphens, slashes, ampersands and full stops inside a run stay in it
//! (`U.S.-based`, `AT&T`). The words are unions of the tokenizer's own
//! pre-tokens (`turbo::tokenizer::pre_words`), so no piece crosses one.

use turbo::tokenizer::pre_words;

/// Abbreviations whose full stop is part of the word.
const ABBREVIATIONS: &[&str] = &[
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Mt", "Ft", "Gen", "Gov", "Sen", "Rep", "Rev", "Hon", "Lt",
    "Col", "Capt", "Sgt", "Cpl", "Adm", "Maj", "Inc", "Co", "Corp", "Ltd", "Bros", "Plc", "Ave", "Blvd", "Rd", "No",
    "Nos", "Vol", "vs", "etc", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
];

const OPEN: &[char] = &['(', '[', '{', '<', '"', '`', '\u{201c}', '\u{2018}', '\u{ab}', '\u{bf}', '\u{a1}'];
const CLOSE: &[char] = &[')', ']', '}', '>', '"', ',', ';', ':', '!', '?', '\u{201d}', '\u{bb}', '\u{2026}'];

fn is_apostrophe(c: char) -> bool {
    c == '\'' || c == '\u{2019}'
}

/// A CJK ideograph, as BERT's `_is_chinese_char` reads one.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

/// The byte ranges of the words of `text`, in order.
pub fn name_words(text: &str) -> Vec<(usize, usize)> {
    let pre = pre_words(text);
    let mut out = Vec::with_capacity(pre.len());
    let mut a = 0;
    while a < pre.len() {
        // A run of pre-tokens with nothing between them.
        let mut b = a + 1;
        while b < pre.len() && pre[b].0 == pre[b - 1].1 {
            b += 1;
        }
        split_run(text, &pre[a..b], &mut out);
        a = b;
    }
    out
}

fn split_run(text: &str, els: &[(usize, usize)], out: &mut Vec<(usize, usize)>) {
    let digits = |k: usize| text[els[k].0..els[k].1].bytes().all(|b| b.is_ascii_digit());
    let mut from = 0;
    for (k, &(a, b)) in els.iter().enumerate() {
        let mut chars = text[a..b].chars();
        let Some(c) = chars.next() else { continue };
        if chars.next().is_some() {
            continue;
        }
        let numeric = matches!(c, ',' | ':') && k > 0 && k + 1 < els.len() && digits(k - 1) && digits(k + 1);
        if (OPEN.contains(&c) || CLOSE.contains(&c)) && !numeric {
            split_word(text, &els[from..k], out);
            out.push((a, b));
            from = k + 1;
        }
    }
    split_word(text, &els[from..], out);
}

fn split_word(text: &str, els: &[(usize, usize)], out: &mut Vec<(usize, usize)>) {
    if els.is_empty() {
        return;
    }
    let s = |k: usize| &text[els[k].0..els[k].1];
    let one = |k: usize| {
        let mut cs = s(k).chars();
        cs.next().filter(|_| cs.next().is_none())
    };
    let (mut lo, mut hi) = (0, els.len());
    while lo < hi {
        let Some(c) = one(lo) else { break };
        let clitic = is_apostrophe(c) && lo + 1 < hi && is_clitic(s(lo + 1));
        if !(OPEN.contains(&c) || is_apostrophe(c) && !clitic) {
            break;
        }
        out.push(els[lo]);
        lo += 1;
    }
    let mut tail = Vec::new();
    while hi > lo {
        let Some(c) = one(hi - 1) else { break };
        let peel = match c {
            '.' => !keeps_stop(lo < hi - 1, || &text[els[lo].0..els[hi - 2].1]),
            c => CLOSE.contains(&c) || is_apostrophe(c),
        };
        if !peel {
            break;
        }
        tail.push(els[hi - 1]);
        hi -= 1;
    }
    let middle = &els[lo..hi];
    if middle.iter().any(|&(x, y)| text[x..y].chars().any(is_cjk)) {
        out.extend_from_slice(middle);
    } else if middle.len() >= 3 && one(hi - 2).is_some_and(is_apostrophe) && matches!(s(hi - 1), "s" | "S") {
        out.push((els[lo].0, els[hi - 3].1));
        out.push((els[hi - 2].0, els[hi - 1].1));
    } else if !middle.is_empty() {
        out.push((els[lo].0, els[hi - 1].1));
    }
    out.extend(tail.into_iter().rev());
}

/// Whether the word after an apostrophe is a clitic it belongs to
/// (`'s`, `'re`, `'90s`).
fn is_clitic(w: &str) -> bool {
    matches!(w.to_ascii_lowercase().as_str(), "s" | "d" | "m" | "t" | "re" | "ve" | "ll")
        || w.starts_with(|c: char| c.is_ascii_digit())
}

/// Whether a full stop after `stem` is part of the word.
fn keeps_stop<'a>(any: bool, stem: impl FnOnce() -> &'a str) -> bool {
    if !any {
        return false;
    }
    let stem = stem();
    let mut cs = stem.chars();
    let single = cs.next().is_some_and(char::is_alphanumeric) && cs.next().is_none();
    // A repeated trailing stop is punctuation, not an abbreviation.
    (stem.contains('.') && !stem.ends_with('.'))
        || single
        || (stem.len() <= 3 && stem.bytes().all(|b| b.is_ascii_digit()))
        || ABBREVIATIONS.contains(&stem)
}
