//! `BoundedSocialMedia`: hashtags, mentions, retweet markers and emoticons
//! found by Unicode properties and word boundaries.

use turbo_analysis::{Annotator, Error, Refs, View, ViewBuilder};
use turbo_normalize_layers::{PlaceholderKind, Placeholders};
use turbo_unicode::category::general_category;
use turbo_unicode::java::is_letter_or_digit;

use super::shared::{emoticon_end, laughter_at, shrink_laughter};
use crate::Normalizer;
use crate::placeholder::{self, Sink, span};

/// Replaces hashtags, mentions, retweet markers and emoticons with one
/// U+0020 each, and shrinks laughter, finding each by Unicode letter,
/// digit and mark properties rather than by ASCII whitespace.
///
/// A word character here is a letter or digit
/// (`java.lang.Character.isLetterOrDigit`), a combining mark (M*), or `_`.
///
/// - A hashtag is `#` (or the fullwidth U+FF03) not preceded by a word
///   character or `&`, followed by a run of word characters, ZWNJ and ZWJ
///   that contains at least one letter or mark: `#café`, `#日本`,
///   `#100DaysOfCode`. `#1`, `#2024` and the `&#39;` of an HTML entity are
///   not hashtags. The hashtag ends with the run, so `#win!` becomes `" !"`.
/// - A mention is `@` (or U+FF20) not preceded by a word character or one
///   of `! @ # $ % & *`, followed by a run of word characters: `@ana`,
///   `@José_1`. A fediverse handle continues with `@` and a domain of two
///   or more labels (`@ana@social.example`); any other `@` right after the
///   run means it is no mention. `a@b.com` is not a mention.
/// - A retweet marker is a run of `rt` units, ASCII case-insensitively,
///   each followed by a space or a colon, not preceded by a word
///   character. A last `rt` followed directly by a mention (`RT@ana`) is a
///   marker too, and the mention is replaced on its own.
/// - An emoticon is eyes (`:`, `;`, `x`, `X`), an optional `-` nose and a
///   mouth (`( ) d D o O p P`) that is not followed by a word character,
///   and whose `x` or `X` eyes are not preceded by one: `:)`, `xD`, `;-P`.
///   `expo`, `taxi`, `xoxo` and `:people` are left alone.
/// - Laughter shrinks as in [`SocialMedia`](super::SocialMedia).
///
/// OpenNLP has no counterpart. `SocialMediaCharSequenceNormalizer` is the
/// reference for what to find; the differences are listed in
/// docs/normalizers.md.
///
/// As an `Annotator` it reads a `View` and writes the normalized `View`
/// and a `Placeholders` layer with a `Hashtag`, `Mention`,
/// `RetweetMarker` or `Emoticon` row per replacement.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct BoundedSocialMedia;

const ZWNJ: char = '\u{200C}';
const ZWJ: char = '\u{200D}';

fn is_mark(c: char) -> bool {
    !c.is_ascii() && general_category(c).is_mark()
}

fn is_word(c: char) -> bool {
    c == '_' || is_letter_or_digit(c) || is_mark(c)
}

fn before(text: &str, i: usize) -> Option<char> {
    text[..i].chars().next_back()
}

/// The end of the run from byte `from` of characters `keep` accepts.
fn run_end(text: &str, from: usize, keep: impl Fn(char) -> bool) -> usize {
    text[from..].char_indices().find(|&(_, c)| !keep(c)).map_or(text.len(), |(k, _)| from + k)
}

/// The hashtag whose `#` (of `width` bytes) is at byte `i`.
fn hashtag_end(text: &str, i: usize, width: usize) -> Option<usize> {
    if before(text, i).is_some_and(|c| is_word(c) || c == '&') {
        return None;
    }
    let body = i + width;
    let end = run_end(text, body, |c| is_word(c) || c == ZWNJ || c == ZWJ);
    let lettered = text[body..end].chars().any(|c| general_category(c).is_letter() || is_mark(c));
    (end > body && lettered).then_some(end)
}

/// The end of a domain of two or more labels of letters, digits, marks and
/// `-`, starting at byte `from`; a trailing dot is left out.
fn domain_end(text: &str, from: usize) -> Option<usize> {
    let label = |c: char| c == '-' || is_letter_or_digit(c) || is_mark(c);
    let mut end = run_end(text, from, label);
    if end == from {
        return None;
    }
    let mut labels = 1;
    while text[end..].starts_with('.') {
        let next = run_end(text, end + 1, label);
        if next == end + 1 {
            break;
        }
        end = next;
        labels += 1;
    }
    (labels >= 2).then_some(end)
}

/// The mention whose `@` (of `width` bytes) is at byte `i`.
fn mention_end(text: &str, i: usize, width: usize) -> Option<usize> {
    if before(text, i).is_some_and(|c| is_word(c) || matches!(c, '!' | '@' | '#' | '$' | '%' | '&' | '*')) {
        return None;
    }
    mention_rest(text, i + width)
}

/// The end of a mention whose body starts at byte `body`, whatever precedes
/// its `@`.
fn mention_rest(text: &str, body: usize) -> Option<usize> {
    let end = run_end(text, body, is_word);
    if end == body {
        return None;
    }
    if !text[end..].starts_with('@') {
        return Some(end);
    }
    domain_end(text, end + 1)
}

/// The tag starting at byte `i`: its end and kind.
fn tag_at(text: &str, i: usize) -> Option<(usize, PlaceholderKind)> {
    let c = text[i..].chars().next()?;
    match c {
        '#' | '\u{FF03}' => hashtag_end(text, i, c.len_utf8()).map(|end| (end, PlaceholderKind::Hashtag)),
        '@' | '\u{FF20}' => mention_end(text, i, c.len_utf8()).map(|end| (end, PlaceholderKind::Mention)),
        _ => None,
    }
}

/// The retweet marker at byte `i`, which holds `r` or `R`: its end, and the
/// end of a mention that follows its last `rt` directly.
fn retweet_at(text: &str, i: usize) -> Option<(usize, Option<usize>)> {
    if before(text, i).is_some_and(is_word) {
        return None;
    }
    let bytes = text.as_bytes();
    let mut end = i;
    while end + 2 <= bytes.len() && matches!(bytes[end], b'r' | b'R') && matches!(bytes[end + 1], b't' | b'T') {
        match bytes.get(end + 2) {
            Some(b' ' | b':') => end += 3,
            Some(b'@') => match mention_rest(text, end + 3) {
                Some(mention) => return Some((end + 2, Some(mention))),
                None => break,
            },
            _ => break,
        }
    }
    (end > i).then_some((end, None))
}

/// The emoticon at byte `i`, which holds its eyes.
fn bounded_emoticon_end(text: &str, i: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if matches!(bytes[i], b'x' | b'X') && before(text, i).is_some_and(is_word) {
        return None;
    }
    let end = emoticon_end(bytes, i)?;
    (!text[end..].chars().next().is_some_and(is_word)).then_some(end)
}

impl BoundedSocialMedia {
    /// Writes `text` with hashtags, mentions, retweet markers and emoticons
    /// replaced by one space each and laughter shrunk, and reports each
    /// placeholder to `sink`.
    pub fn scan(text: &str, out: &mut ViewBuilder<'_>, sink: &mut Sink<'_>) {
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let found = match bytes[i] {
                b'#' | b'@' | 0xEF => tag_at(text, i),
                b'r' | b'R' => match retweet_at(text, i) {
                    Some((end, Some(mention))) => {
                        out.keep(i);
                        out.replace(end, " ");
                        out.replace(mention, " ");
                        sink(span(i, end), PlaceholderKind::RetweetMarker);
                        sink(span(end, mention), PlaceholderKind::Mention);
                        i = mention;
                        continue;
                    }
                    Some((end, None)) => Some((end, PlaceholderKind::RetweetMarker)),
                    None => None,
                },
                b':' | b';' | b'x' | b'X' => bounded_emoticon_end(text, i).map(|end| (end, PlaceholderKind::Emoticon)),
                b'h' | b'H' | b'j' | b'J' => {
                    match laughter_at(bytes, i) {
                        Ok((end, consonant, vowel)) => {
                            shrink_laughter(bytes, i, end, [consonant, vowel, consonant, vowel], out);
                            i = end;
                        }
                        Err(next) => i = next,
                    }
                    continue;
                }
                _ => None,
            };
            match found {
                Some((end, kind)) => {
                    out.keep(i);
                    out.replace(end, " ");
                    sink(span(i, end), kind);
                    i = end;
                }
                None => i += 1,
            }
        }
    }
}

impl Normalizer for BoundedSocialMedia {
    const ID: &'static str = "turbo:normalize:bounded-social-media";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        BoundedSocialMedia::scan(text, out, &mut placeholder::discard);
    }
}

impl Annotator for BoundedSocialMedia {
    const ID: &'static str = <BoundedSocialMedia as Normalizer>::ID;
    type Reads = (View,);
    type Writes = (View, Placeholders);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View, Placeholders), Error> {
        Ok(placeholder::annotate(src, BoundedSocialMedia::scan))
    }
}
