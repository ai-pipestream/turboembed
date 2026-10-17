//! Model of how the JVM's `WordpieceEncoder` normalizes text and tracks
//! source byte offsets, and how the port's expected spans derive from it.

use turbo_goldens::expect::{only_stripped, without_trailing_marks};
use turbo_unicode::GeneralCategory;
use turbo_unicode::category::general_category;
use turbo_unicode::java::to_lower_char;
use turbo_unicode::normalize::{Form, is_normalized, normalize};

use super::Piece;

/// `WordpieceEncoder`'s longest word, in normalized code points, that is
/// segmented rather than made unknown; the bundle's `max_input_chars_per_word`.
pub(super) const MAX_WORD_CODE_POINTS: usize = 100;

/// The strings upstream `tokenizers` matches as special tokens before
/// normalizing, which `WordpieceEncoder` reads as text.
pub(super) const SPECIAL_TOKENS: [&str; 5] = ["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"];

/// The ideographs the reference BERT counts as CJK and upstream does not.
pub(super) fn jvm_only_cjk(c: char) -> bool {
    ('\u{2B820}'..='\u{2B91F}').contains(&c)
}

/// One code point of the JVM's mapped text with its source byte span.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Unit {
    pub(super) c: char,
    pub(super) start: u32,
    pub(super) end: u32,
}

/// Which of the documented differences the mapping went through.
#[derive(Default)]
pub(super) struct Used {
    /// A stripped mark gave its end to the character before it.
    pub(super) stripped: bool,
    /// The runs canonical reordering made every character of span whole.
    pub(super) reordered: Vec<[u32; 2]>,
    /// A capital sigma was lowercased to final sigma.
    pub(super) sigma: bool,
}

/// `BertNormalization.isControl`: a `C*` category but for tab, line feed
/// and carriage return.
fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r')
        && matches!(
            general_category(c),
            GeneralCategory::Cc | GeneralCategory::Cf | GeneralCategory::Cs | GeneralCategory::Co | GeneralCategory::Cn
        )
}

/// `BertNormalization.isWhitespace`: space, tab, line feed, carriage
/// return and the space separators, the no-break spaces among them.
fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r') || general_category(c) == GeneralCategory::Zs
}

fn is_line_or_paragraph_separator(c: char) -> bool {
    matches!(general_category(c), GeneralCategory::Zl | GeneralCategory::Zp)
}

/// `BertNormalization.isPunctuation`: ASCII symbols and every punctuation
/// category.
fn is_punctuation(c: char) -> bool {
    matches!(c, '!'..='/' | ':'..='@' | '['..='`' | '{'..='~')
        || matches!(
            general_category(c),
            GeneralCategory::Pc
                | GeneralCategory::Pd
                | GeneralCategory::Ps
                | GeneralCategory::Pe
                | GeneralCategory::Pi
                | GeneralCategory::Pf
                | GeneralCategory::Po
        )
}

/// `BertNormalization.isCjk`: the reference BERT's ideograph ranges.
fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// `WordpieceEncoder.isCaseIgnorable`: the categories and the Unicode
/// 15.0 word break values the `Case_Ignorable` property is made of.
fn case_ignorable(c: char) -> bool {
    matches!(
        general_category(c),
        GeneralCategory::Mn | GeneralCategory::Me | GeneralCategory::Cf | GeneralCategory::Lm | GeneralCategory::Sk
    ) || matches!(
        c as u32,
        0x0027
            | 0x002E
            | 0x003A
            | 0x00B7
            | 0x0387
            | 0x055F
            | 0x05F4
            | 0x2018
            | 0x2019
            | 0x2024
            | 0x2027
            | 0xFE13
            | 0xFE52
            | 0xFE55
            | 0xFF07
            | 0xFF0E
            | 0xFF1A
    )
}

/// `WordpieceEncoder.isCased`: upper, lower or title case.
fn cased(c: char) -> bool {
    c.is_uppercase() || c.is_lowercase() || general_category(c) == GeneralCategory::Lt
}

/// `cleanAndIsolateCjk`: controls dropped, whitespace and separators made
/// spaces, every ideograph between two empty spaces.
fn clean_and_isolate_cjk(text: &str) -> Vec<Unit> {
    let mut out = Vec::with_capacity(text.len());
    for (i, c) in text.char_indices() {
        let (start, end) = (i as u32, (i + c.len_utf8()) as u32);
        if c == '\0' || c == '\u{FFFD}' || is_control(c) {
            continue;
        }
        if is_whitespace(c) || is_line_or_paragraph_separator(c) {
            out.push(Unit { c: ' ', start, end });
        } else if is_cjk(c) {
            out.push(Unit { c: ' ', start, end: start });
            out.push(Unit { c, start, end });
            out.push(Unit { c: ' ', start: end, end });
        } else {
            out.push(Unit { c, start, end });
        }
    }
    out
}

/// `isFinalSigma`: the capital sigma at `index` of the run `run` follows
/// a cased character and no cased character follows it, case ignorable
/// characters skipped on both sides and the search kept to the run.
fn final_sigma(run: &[Unit], index: usize) -> bool {
    let follows_cased = run[..index].iter().rev().find(|u| !case_ignorable(u.c)).is_some_and(|u| cased(u.c));
    follows_cased && !run[index + 1..].iter().find(|u| !case_ignorable(u.c)).is_some_and(|u| cased(u.c))
}

/// `lowerCaseRun`: `Character.toLowerCase` of each code point, final sigma
/// where the sigma is word final.
fn lowercase_run(run: &[Unit], used: &mut Used, sigmas: &mut Vec<u32>) -> Vec<Unit> {
    run.iter()
        .enumerate()
        .map(|(i, u)| {
            let c = if u.c == '\u{03A3}' && final_sigma(run, i) {
                used.sigma = true;
                sigmas.push(u.start);
                '\u{03C2}'
            } else {
                to_lower_char(u.c)
            };
            Unit { c, ..*u }
        })
        .collect()
}

fn nfd(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    normalize(Form::Nfd, text, &mut out);
    out
}

/// `appendWithoutAccents`: every nonspacing mark is left out, and its end
/// given to the character before it in the same run.
fn append_without_accents(units: &[Unit], out: &mut Vec<Unit>, used: &mut Used) {
    let base = out.len();
    for u in units {
        if general_category(u.c) == GeneralCategory::Mn {
            if out.len() > base {
                used.stripped = true;
                let last = out.len() - 1;
                out[last].end = u.end;
            }
        } else {
            out.push(*u);
        }
    }
}

/// `transformRun`: lowercase, then NFD and accent stripping, with each
/// decomposed character keeping its source character's span unless the
/// run's NFD is not its characters' NFDs in turn, when every character
/// spans the run.
fn transform_run(run: &[Unit], out: &mut Vec<Unit>, used: &mut Used, sigmas: &mut Vec<u32>) {
    let lower = lowercase_run(run, used, sigmas);
    let text: String = lower.iter().map(|u| u.c).collect();
    if is_normalized(Form::Nfd, &text) {
        append_without_accents(&lower, out, used);
        return;
    }
    let mut by_code_point = Vec::with_capacity(lower.len());
    for u in &lower {
        for c in nfd(u.c.encode_utf8(&mut [0; 4])).chars() {
            by_code_point.push(Unit { c, ..*u });
        }
    }
    let whole = nfd(&text);
    if whole.chars().eq(by_code_point.iter().map(|u| u.c)) {
        append_without_accents(&by_code_point, out, used);
    } else {
        let (start, end) = (run[0].start, run[run.len() - 1].end);
        used.reordered.push([start, end]);
        out.extend(whole.chars().filter(|&c| general_category(c) != GeneralCategory::Mn).map(|c| Unit {
            c,
            start,
            end,
        }));
    }
}

/// `lowerCaseAndStripAccents`: every run between spaces transformed.
fn lowercase_and_strip_accents(units: &[Unit], used: &mut Used, sigmas: &mut Vec<u32>) -> Vec<Unit> {
    let mut out = Vec::with_capacity(units.len());
    let mut from = 0;
    while from < units.len() {
        if units[from].c == ' ' {
            out.push(units[from]);
            from += 1;
            continue;
        }
        let to = units[from..].iter().position(|u| u.c == ' ').map_or(units.len(), |i| from + i);
        transform_run(&units[from..to], &mut out, used, sigmas);
        from = to;
    }
    out
}

/// `isolatePunctuation`: an empty space on either side of every
/// punctuation character.
fn isolate_punctuation(units: &[Unit]) -> Vec<Unit> {
    let mut out = Vec::with_capacity(units.len());
    for u in units {
        if is_punctuation(u.c) {
            out.push(Unit { c: ' ', start: u.start, end: u.start });
            out.push(*u);
            out.push(Unit { c: ' ', start: u.end, end: u.end });
        } else {
            out.push(*u);
        }
    }
    out
}

/// `WordpieceEncoder`'s mapped text for `text`.
pub(super) struct Mapped {
    pub(super) units: Vec<Unit>,
    /// `text` with every capital sigma the JVM lowercased to final sigma
    /// written as final sigma, the same length.
    pub(super) as_jvm: String,
    pub(super) used: Used,
}

pub(super) fn map(text: &str) -> Mapped {
    let mut used = Used::default();
    let mut sigmas = Vec::new();
    let units = clean_and_isolate_cjk(text);
    let units = lowercase_and_strip_accents(&units, &mut used, &mut sigmas);
    let units = isolate_punctuation(&units);
    let mut as_jvm = text.to_string();
    for at in sigmas {
        let at = at as usize;
        // Both sigmas are two bytes, so the text keeps its offsets.
        as_jvm.replace_range(at..at + '\u{03A3}'.len_utf8(), "\u{03C2}");
    }
    Mapped { units, as_jvm, used }
}

/// The special token strings in `text`, left to right, each with the
/// port's id for it.
pub(super) fn special_occurrences(text: &str, special: &[(&str, u32)]) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    for (token, id) in special {
        for (at, _) in text.match_indices(token) {
            out.push((at as u32, (at + token.len()) as u32, *id));
        }
    }
    out.sort_unstable();
    out
}

/// What the port's span of a piece must be, given the JVM's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PortSpan {
    /// This span exactly.
    Exact([u32; 2]),
    /// A non-empty span inside this one: the port keeps each character's
    /// own span where the JVM gave the whole run to every piece.
    Within([u32; 2]),
}

/// The source spans of the modelled words (runs of units between spaces)
/// that hold a capital sigma.
pub(super) fn sigma_words(text: &str, units: &[Unit]) -> Vec<[u32; 2]> {
    let mut words = Vec::new();
    let mut from = 0;
    while from < units.len() {
        if units[from].c == ' ' {
            from += 1;
            continue;
        }
        let to = units[from..].iter().position(|u| u.c == ' ').map_or(units.len(), |i| from + i);
        let span = [units[from].start, units[to - 1].end];
        if text[span[0] as usize..span[1] as usize].contains('\u{3A3}') {
            words.push(span);
        }
        from = to;
    }
    words
}

/// The port's expected span for each JVM piece, as `analysis/embed/tests/goldens.rs`
/// documents the differences: a piece in a run canonical reordering
/// crossed lies inside the run; otherwise the port's span is the JVM's
/// with the stripped marks trimmed off its end, as `without_trailing_marks`
/// trims them, which leaves a final sigma's span and a plain span alone.
pub(super) fn expected_port_spans(text: &str, mapped: &Mapped, jvm: &[Piece]) -> Vec<PortSpan> {
    jvm.iter()
        .map(|&(_, span)| {
            if mapped.used.reordered.contains(&span) {
                PortSpan::Within(span)
            } else {
                PortSpan::Exact(without_trailing_marks(text, span))
            }
        })
        .collect()
}

/// Whether the port's pieces have the expected spans, one for one. The
/// pieces of a reordered run must lie inside it, follow each other with
/// nothing but stripped marks and dropped characters between them (or
/// repeat the span before them, as the pieces of one decomposed character
/// do: a Hangul syllable is two or three jamo to the port, each piece with
/// the syllable's span), and reach the run's end but for such characters;
/// no piece ends in one, as the port ends a piece before the marks it
/// strips.
pub(super) fn port_spans_fit(text: &str, expected: &[PortSpan], rust: &[Piece]) -> bool {
    port_span_misfit(text, expected, rust).is_none()
}

/// The index of the first piece that does not fit as [`port_spans_fit`]
/// asks, `rust.len()` when the port has too few pieces, None when all fit.
pub(super) fn port_span_misfit(text: &str, expected: &[PortSpan], rust: &[Piece]) -> Option<usize> {
    let gap_is_stripped = |from: u32, to: u32| from <= to && only_stripped(text, [from, to]);
    // The piece before, inside a run.
    let mut previous: Option<[u32; 2]> = None;
    for (i, (e, &(_, [a, b]))) in expected.iter().zip(rust).enumerate() {
        if !text.is_char_boundary(a as usize) || !text.is_char_boundary(b as usize) {
            return Some(i);
        }
        match *e {
            PortSpan::Exact(span) => {
                if span != [a, b] || previous.is_some() {
                    return Some(i);
                }
            }
            PortSpan::Within([s, end]) => {
                let first = i == 0 || expected[i - 1] != *e;
                let follows = match (first, previous) {
                    (true, None) => gap_is_stripped(s, a),
                    (false, Some([p, q])) => gap_is_stripped(q, a) || [a, b] == [p, q],
                    _ => false,
                };
                if !follows || a >= b || b > end || without_trailing_marks(text, [a, b]) != [a, b] {
                    return Some(i);
                }
                let last = expected.get(i + 1) != Some(e);
                if last && !gap_is_stripped(b, end) {
                    return Some(i);
                }
                previous = if last { None } else { Some([a, b]) };
            }
        }
    }
    (expected.len() != rust.len()).then_some(expected.len().min(rust.len()))
}
