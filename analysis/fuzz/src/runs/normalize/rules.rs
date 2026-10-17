//! Per-normalizer rules recognizing the documented differences from OpenNLP
//! in `docs/normalizers.md`.

use turbo_goldens::expect::Disagreement;
use turbo_normalize::tables::{is_horizontal_whitespace, is_line_break, is_whitespace};
use turbo_normalize::{EnglishContractions, normalize};
use turbo_unicode::category::general_category;
use turbo_unicode::confusables::{prototype, skeleton};
use turbo_unicode::java::is_letter;
use turbo_unicode::normalize::{Form, normalize as unicode_normalize};

use super::super::confusables_17::CHANGES_17;

/// `golden` with each character `f` maps written as its replacement.
fn fold(golden: &str, f: impl Fn(char) -> Option<&'static str>) -> String {
    let mut out = String::with_capacity(golden.len());
    for c in golden.chars() {
        match f(c) {
            Some(with) => out.push_str(with),
            None => out.push(c),
        }
    }
    out
}

/// The port's result is the JVM's with the addition `f` applied to the
/// characters the source holds; the source must hold at least one.
fn addition(text: &str, d: &Disagreement, f: impl Fn(char) -> Option<&'static str>) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    d.source(text).chars().any(|c| f(c).is_some()) && rust == fold(golden, &f)
}

pub(super) fn quote_rule(text: &str, d: &Disagreement) -> bool {
    addition(text, d, |c| (c == '\u{2E42}').then_some("\""))
}

pub(super) fn bullet_rule(text: &str, d: &Disagreement) -> bool {
    addition(text, d, |c| matches!(c, '\u{204C}' | '\u{204D}' | '\u{25D8}' | '\u{29BE}' | '\u{29BF}').then_some(" "))
}

pub(super) fn ellipsis_rule(text: &str, d: &Disagreement) -> bool {
    addition(text, d, |c| match c {
        '\u{2024}' => Some("."),
        '\u{FE19}' => Some("..."),
        '\u{FE30}' => Some(".."),
        _ => None,
    })
}

pub(super) fn invisible_rule(text: &str, d: &Disagreement) -> bool {
    addition(text, d, |c| ('\u{206A}'..='\u{206F}').contains(&c).then_some(""))
}

/// In `default-chain`, `Invisible` runs before `Whitespace::collapse()`, so
/// deleting `U+206A..=U+206F` between whitespace characters lets the chain
/// collapse the newly adjacent spaces into one space (or trim a leading or
/// trailing space when the disagreement reaches the edge of the text).
pub(super) fn chain_rule(text: &str, d: &Disagreement) -> bool {
    let map = |c: char| match c {
        '\u{206A}'..='\u{206F}' => Some(""),
        '\u{2E42}' => Some("\""),
        _ => None,
    };
    let Some((rust, golden)) = d.results() else { return false };
    if !d.source(text).chars().any(|c| map(c).is_some()) {
        return false;
    }
    let folded = fold(golden, map);
    if rust == folded {
        return true;
    }
    let mut collapsed = String::with_capacity(folded.len());
    for c in folded.chars() {
        if !(c == ' ' && collapsed.ends_with(' ')) {
            collapsed.push(c);
        }
    }
    let mut slice = collapsed.as_str();
    if text[..d.start].chars().all(is_whitespace) {
        slice = slice.strip_prefix(' ').unwrap_or(slice);
    }
    if text[d.end..].chars().all(is_whitespace) {
        slice = slice.strip_suffix(' ').unwrap_or(slice);
    }
    rust == slice
}

/// The region is a hyphen, a line break and indentation that the port
/// deleted and the JVM kept, after a letter with one or more combining
/// marks and before a letter.
pub(super) fn dehyphenation_rule(text: &str, d: &Disagreement) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    let source = d.source(text);
    if !rust.is_empty() || golden != source {
        return false;
    }
    let mut chars = source.chars();
    if !matches!(chars.next(), Some('-' | '\u{00AD}' | '\u{2010}')) {
        return false;
    }
    let rest = chars.as_str();
    let Some(first) = rest.chars().next().filter(|&c| is_line_break(c)) else { return false };
    let mut after = &rest[first.len_utf8()..];
    if first == '\r' {
        after = after.strip_prefix('\n').unwrap_or(after);
    }
    if !after.chars().all(is_horizontal_whitespace) {
        return false;
    }
    let mut before = text[..d.start].chars().rev().peekable();
    let mut marks = 0;
    while before.peek().is_some_and(|&c| general_category(c).is_mark()) {
        before.next();
        marks += 1;
    }
    marks > 0 && before.next().is_some_and(is_letter) && text[d.end..].chars().next().is_some_and(is_letter)
}

/// The word of ASCII letters and apostrophes around the region.
fn word_around(text: &str, start: usize, end: usize) -> (usize, usize) {
    let in_word = |c: char| c.is_ascii_alphabetic() || matches!(c, '\'' | '\u{2019}' | '\u{02BC}' | '\u{FF07}');
    let before = text[..start].chars().rev().take_while(|&c| in_word(c)).map(char::len_utf8).sum::<usize>();
    let after = text[end..].chars().take_while(|&c| in_word(c)).map(char::len_utf8).sum::<usize>();
    (start - before, end + after)
}

/// The region is inside a contraction the port expands and OpenNLP does
/// not: `y'all`, or one of the added subjects before 're, 've or 'll. The
/// JVM left the region alone and the port's text is its own expansion of
/// the word.
pub(super) fn contraction_rule(text: &str, d: &Disagreement) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    if golden != d.source(text) {
        return false;
    }
    let (ws, we) = word_around(text, d.start, d.end);
    let word = &text[ws..we];
    let lower = word.to_ascii_lowercase();
    let apostrophe =
        |s: &str| s.chars().next().is_some_and(|c| matches!(c, '\'' | '\u{2019}' | '\u{02BC}' | '\u{FF07}'));
    let added = |stems: &[&str], suffix: &str| {
        lower.strip_suffix(suffix).is_some_and(|s| {
            let (stem, apo) = s.split_at(s.len() - s.chars().last().map_or(0, char::len_utf8));
            apostrophe(apo) && stems.contains(&stem)
        })
    };
    let expands_here_only =
        lower.len() == 5 && lower.starts_with('y') && lower.ends_with("all") && apostrophe(&lower[1..])
            || added(&["here", "these", "those"], "re")
            || added(&["may", "there", "these", "those", "what", "where"], "ve")
            || added(&["here", "this", "these", "those", "which"], "ll");
    if !expands_here_only {
        return false;
    }
    let (expanded, _) = normalize(&EnglishContractions, word);
    expanded == format!("{}{rust}{}", &text[ws..d.start], &text[d.end..we])
}

/// `SymbolWords`' ten symbols and their words.
fn symbol_word(c: char) -> Option<&'static str> {
    Some(match c {
        '&' => "and",
        '+' => "plus",
        '@' => "at",
        '%' => "percent",
        '\u{00A7}' => "section",
        '\u{00B6}' => "paragraph",
        '\u{00B0}' => "degree",
        '\u{00A9}' => "copyright",
        '\u{00AE}' => "registered",
        '\u{2122}' => "trademark",
        _ => return None,
    })
}

/// The region is one symbol with whitespace or the text's edge on both
/// sides, not the whole text (which OpenNLP maps too), left by the JVM
/// and written as its word here.
pub(super) fn symbol_rule(text: &str, d: &Disagreement) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    let source = d.source(text);
    let mut chars = source.chars();
    let (Some(c), None) = (chars.next(), chars.next()) else { return false };
    let Some(word) = symbol_word(c) else { return false };
    let bounded = text[..d.start].chars().next_back().is_none_or(is_whitespace)
        && text[d.end..].chars().next().is_none_or(is_whitespace);
    bounded && source != text && golden == source && rust == word
}

/// The region holds an e-mail address with a non-ASCII letter, digit or
/// mark, which `BoundedUrls` takes as part of the address and OpenNLP's
/// ASCII scan does not. Where the two results differ, the JVM kept the
/// address (whole, or what is left of it on either side of the ASCII
/// part it replaced by a space) and the port wrote nothing or one space
/// in its place: what the JVM kept is one or two stretches without
/// whitespace, around at most one space, holding a non-ASCII letter,
/// digit or mark.
///
/// What the JVM kept is the source's own text, in order, and each kept
/// stretch is in a whitespace-delimited token of the source that holds an
/// `@`: the address the port matched (the region can hold other tokens
/// too, a URL both sides replaced next to the address among them).
pub(super) fn bounded_url_rule(text: &str, d: &Disagreement) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    let source = d.source(text);
    if !source.contains('@') {
        return false;
    }
    let prefix = rust.char_indices().zip(golden.chars()).take_while(|((_, a), b)| a == b).count();
    let prefix = rust.char_indices().nth(prefix).map_or(rust.len(), |(i, _)| i);
    let (r_rest, g_rest) = (&rust[prefix..], &golden[prefix..]);
    let suffix: usize =
        r_rest.chars().rev().zip(g_rest.chars().rev()).take_while(|(a, b)| a == b).map(|(a, _)| a.len_utf8()).sum();
    let (removed, kept) = (&r_rest[..r_rest.len() - suffix], &g_rest[..g_rest.len() - suffix]);
    let international = |c: char| !c.is_ascii() && (c.is_alphanumeric() || general_category(c).is_mark());
    let parts: Vec<&str> = kept.split(' ').collect();
    if !parts.iter().all(|part| !part.is_empty() && !part.chars().any(char::is_whitespace)) {
        return false;
    }
    // Each kept stretch is in the source, after the one before it, in a
    // token with an `@`. If an earlier URL in the same disagreement region
    // happens to contain the same substring, keep searching forward until a
    // match inside an `@`-bearing token is found.
    let mut from = 0;
    let in_source = parts.iter().all(|part| {
        let mut search = from;
        while let Some(i) = source[search..].find(part) {
            let at = search + i;
            let token_start = source[..at]
                .rfind(char::is_whitespace)
                .map_or(0, |w| w + source[w..].chars().next().map_or(1, char::len_utf8));
            let token_end = source[at..].find(char::is_whitespace).map_or(source.len(), |w| at + w);
            if source[token_start..token_end].contains('@') {
                from = at + part.len();
                return true;
            }
            search = at + part.chars().next().map_or(1, char::len_utf8);
        }
        false
    });
    (removed.is_empty() || removed == " ") && parts.len() <= 2 && in_source && kept.chars().any(international)
}

/// The skeleton of `text` by confusables.txt 17.0.0, the data OpenNLP
/// bundles: NFD, each character replaced by its 17.0.0 prototype
/// (`confusables_17::CHANGES_17` where that version differs from 18.0.0,
/// `turbo_unicode`'s prototype otherwise), NFD again.
pub(super) fn skeleton_17(text: &str) -> String {
    let mut decomposed = String::with_capacity(text.len());
    unicode_normalize(Form::Nfd, text, &mut decomposed);
    let mut mapped = String::with_capacity(decomposed.len());
    for c in decomposed.chars() {
        let changed = CHANGES_17.binary_search_by_key(&(c as u32), |&(cp, _)| cp).ok().map(|i| CHANGES_17[i].1);
        match changed.unwrap_or_else(|| prototype(c)) {
            Some(target) => mapped.push_str(target),
            None => mapped.push(c),
        }
    }
    let mut out = String::with_capacity(mapped.len());
    unicode_normalize(Form::Nfd, &mapped, &mut out);
    out
}

/// The port's text is the Unicode 18.0 skeleton of the source and the
/// JVM's its 17.0 skeleton: the two data versions, nothing else.
pub(super) fn confusable_rule(text: &str, d: &Disagreement) -> bool {
    let Some((rust, golden)) = d.results() else { return false };
    let mut of_source = String::new();
    skeleton(d.source(text), &mut of_source);
    rust == of_source && golden == skeleton_17(d.source(text))
}
