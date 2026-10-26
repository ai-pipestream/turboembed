//! Unicode normalization, aligned offsets, and character categories.

use super::*;

/// BertNormalizer, in upstream's order: clean, split CJK, strip accents,
/// lowercase. Every step works on one character at a time except
/// stripping accents, whose NFD reorders a run of combining marks; an
/// ASCII character is never one, so each run of non-ASCII text goes
/// through the steps on its own and ASCII text through a table.
pub(super) fn normalize(n: &Normalizer, ascii: &[u8; 128], text: &str, s: &mut String) {
    s.clear();
    s.reserve(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            let m = ascii[b as usize];
            if m != DROP {
                s.push(m as char);
            }
            i += 1;
            continue;
        }
        // An ASCII byte is never inside a UTF-8 sequence, so the run ends
        // on a character boundary.
        let start = i;
        while i < bytes.len() && bytes[i] >= 0x80 {
            i += 1;
        }
        normalize_run(n, &text[start..i], s);
    }
}

/// The largest normalized buffer a thread keeps between texts.
pub(super) const KEPT_NORMALIZED: usize = 64 << 10;

/// normalize() into a new String.
pub(super) fn normalized(n: &Normalizer, ascii: &[u8; 128], text: &str) -> String {
    let mut s = String::new();
    normalize(n, ascii, text, &mut s);
    s
}

thread_local! {
    /// plain_ids' normalized text.
    pub(super) static NORMALIZED: Cell<String> = const { Cell::new(String::new()) };
    /// The source of each of its bytes, when spans are kept.
    pub(super) static SOURCES: Cell<Vec<Span>> = const { Cell::new(Vec::new()) };
}

/// What normalize() makes of each ASCII character: no ASCII character is
/// CJK or decomposes, its lowercase is its ASCII lowercase, and its
/// control characters are those under 0x20 and 0x7f.
pub(super) const DROP: u8 = 0xff;

pub(super) fn ascii_map(n: &Normalizer) -> [u8; 128] {
    let mut m = [0u8; 128];
    for (b, out) in m.iter_mut().enumerate() {
        let b = b as u8;
        *out = if n.clean_text && matches!(b, b'\t' | b'\n' | b'\r') {
            b' '
        } else if n.clean_text && (b < 0x20 || b == 0x7f) {
            DROP
        } else if n.lowercase {
            b.to_ascii_lowercase()
        } else {
            b
        };
    }
    m
}

/// The normalizer's steps on text, one character at a time, appended to
/// `out`.
pub(super) fn normalize_run(n: &Normalizer, text: &str, out: &mut String) {
    let clean = n.clean_text;
    let cleaned = text
        .chars()
        .filter(move |&c| !(clean && (c == '\0' || c == '\u{fffd}' || is_control(c))))
        .map(move |c| if clean && is_whitespace(c) { ' ' } else { c });
    let split_cjk = n.split_cjk;
    let split = cleaned.flat_map(move |c| {
        let cjk = split_cjk && is_cjk(c);
        [cjk.then_some(' '), Some(c), cjk.then_some(' ')].into_iter().flatten()
    });
    if n.strip_accents {
        lowercased(n, aligned_chars(Form::Nfd, split).map(|(c, _)| c).filter(|&c| !category(c, MARK_NONSPACING)), out);
    } else {
        lowercased(n, split, out);
    }
}

pub(super) fn lowercased(n: &Normalizer, chars: impl Iterator<Item = char>, out: &mut String) {
    if n.lowercase {
        out.extend(chars.flat_map(char::to_lowercase));
    } else {
        out.extend(chars);
    }
}

/// normalize(), and the source of each byte it writes: the span in the
/// caller's text of the character it came from, `base` being where
/// `text` starts there. Upstream's alignments, step by step: a character
/// a step writes in place of one, or beside it (a space around CJK, the
/// marks NFD splits off, a lowercase longer than one character), comes
/// from that character.
pub(super) fn normalize_aligned(
    n: &Normalizer,
    ascii: &[u8; 128],
    text: &str,
    base: usize,
    s: &mut String,
    sources: &mut Vec<Span>,
) {
    s.clear();
    sources.clear();
    s.reserve(text.len());
    sources.reserve(text.len());
    // ASCII through the table and each run of other text on its own, as
    // normalize() does.
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            let m = ascii[b as usize];
            if m != DROP {
                s.push(m as char);
                let at = (base + i) as u32;
                sources.push([at, at + 1]);
            }
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i] >= 0x80 {
            i += 1;
        }
        normalize_run_aligned(n, &text[start..i], base + start, s, sources);
    }
}

/// normalize_run(), and the source of each byte it writes.
pub(super) fn normalize_run_aligned(n: &Normalizer, text: &str, base: usize, s: &mut String, sources: &mut Vec<Span>) {
    let clean = n.clean_text;
    let mut chars: Vec<(char, Span)> = text
        .char_indices()
        .filter(|&(_, c)| !(clean && (c == '\0' || c == '\u{fffd}' || is_control(c))))
        .map(|(i, c)| {
            let at = (base + i) as u32;
            (if clean && is_whitespace(c) { ' ' } else { c }, [at, at + c.len_utf8() as u32])
        })
        .collect();
    if n.split_cjk && chars.iter().any(|&(c, _)| is_cjk(c)) {
        let mut spaced = Vec::with_capacity(chars.len() + 8);
        for (c, at) in chars {
            if is_cjk(c) {
                spaced.extend([(' ', at), (c, at), (' ', at)]);
            } else {
                spaced.push((c, at));
            }
        }
        chars = spaced;
    }
    if n.strip_accents && chars.iter().any(|&(c, _)| !c.is_ascii()) {
        // NFD writes a character's decomposition in its place, the first
        // character with change 0 and the rest with 1, and reorders runs
        // of combining marks: a character of change 0 takes the next
        // source in order, the others the last one taken.
        let mut stripped = Vec::with_capacity(chars.len());
        let mut next = 0;
        let mut last = chars.first().map_or([base as u32; 2], |c| c.1);
        for (c, change) in aligned_chars(Form::Nfd, chars.iter().map(|&(c, _)| c)) {
            if change <= 0
                && let Some(&(_, at)) = chars.get(next)
            {
                last = at;
                next += 1 + change.unsigned_abs();
            }
            if !category(c, MARK_NONSPACING) {
                stripped.push((c, last));
            }
        }
        chars = stripped;
    }
    for (c, at) in chars {
        if n.lowercase {
            for l in c.to_lowercase() {
                s.push(l);
                sources.extend(std::iter::repeat_n(at, l.len_utf8()));
            }
        } else {
            s.push(c);
            sources.extend(std::iter::repeat_n(at, c.len_utf8()));
        }
    }
}

/// BertPreTokenizer: split on whitespace, and each punctuation character
/// is a word of its own.
pub(super) fn pre_tokenize<'a>(s: &'a str, mut word: impl FnMut(&'a str)) {
    for chunk in s.split(char::is_whitespace) {
        let mut begin = 0;
        for (i, c) in chunk.char_indices() {
            if c.is_ascii_punctuation() || (!c.is_ascii() && category(c, PUNCTUATION)) {
                if begin < i {
                    word(&chunk[begin..i]);
                }
                word(&chunk[i..i + c.len_utf8()]);
                begin = i + c.len_utf8();
            }
        }
        if begin < chunk.len() {
            word(&chunk[begin..]);
        }
    }
}

/// The words a BERT tokenizer cuts `text` into before WordPiece, as byte
/// ranges: `BertPreTokenizer`'s, whitespace parting them and each
/// punctuation character a word of its own, with each CJK ideograph one
/// too, as `BertNormalizer`'s handle_chinese_chars spaces it. No piece of
/// such a tokenizer crosses one.
pub fn pre_words(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        let alone = c.is_ascii_punctuation() || (!c.is_ascii() && category(c, PUNCTUATION)) || is_cjk(c);
        if is_whitespace(c) || alone {
            if let Some(s) = start.take() {
                out.push((s, i));
            }
            if alone {
                out.push((i, i + c.len_utf8()));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        out.push((s, text.len()));
    }
    out
}

/// A character of regex's Unicode \\w, which upstream's single_word
/// reads: alphabetic, a mark, a decimal digit, a connector, or a joiner.
pub(super) fn is_word(c: char) -> bool {
    let category = turbo_unicode::category::general_category(c);
    turbo_unicode::props::alphabetic(c)
        || category.is_mark()
        || matches!(category, turbo_unicode::GeneralCategory::Nd | turbo_unicode::GeneralCategory::Pc)
        || matches!(c, '\u{200c}' | '\u{200d}')
}

pub(super) fn is_whitespace(c: char) -> bool {
    turbo_unicode::props::white_space(c)
}

pub(super) fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && category(c, OTHER)
}

pub(super) const OTHER: u8 = 1;
pub(super) const MARK_NONSPACING: u8 = 2;
pub(super) const PUNCTUATION: u8 = 4;
pub(crate) const LETTER: u8 = 8;
pub(crate) const NUMBER: u8 = 16;

/// Whether `c` is in the categories `bits` names (Unicode's C, Mn, P, L
/// and N): from a table of the Basic Multilingual Plane, made once,
/// rather than a binary search per character.
pub(crate) fn category(c: char, bits: u8) -> bool {
    pub(super) static BMP: std::sync::OnceLock<Box<[u8; 0x10000]>> = std::sync::OnceLock::new();
    let Ok(at) = u16::try_from(c as u32) else {
        return categories(c) & bits != 0;
    };
    let table = BMP.get_or_init(|| {
        let mut t = Box::new([0u8; 0x10000]);
        for (i, f) in t.iter_mut().enumerate() {
            // A surrogate is no char, and in category C.
            *f = char::from_u32(i as u32).map_or(OTHER, categories);
        }
        t
    });
    table[at as usize] & bits != 0
}

/// `c`'s bits: C, Mn, P, L and N from turbo-unicode.
pub(super) fn categories(c: char) -> u8 {
    let category = turbo_unicode::category::general_category(c);
    (category.is_other() as u8)
        | ((matches!(category, turbo_unicode::GeneralCategory::Mn) as u8) << 1)
        | ((category.is_punctuation() as u8) << 2)
        | ((category.is_letter() as u8) << 3)
        | ((category.is_number() as u8) << 4)
}

pub(super) fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

pub(super) fn preview(text: &str) -> String {
    let short: String = text.chars().take(32).collect();
    if short.len() < text.len() { format!("{short:?}...") } else { format!("{short:?}") }
}
