//! `SocialMedia`: OpenNLP's `SocialMediaCharSequenceNormalizer`.

use turbo_analysis::{Annotator, Error, Refs, View, ViewBuilder};
use turbo_normalize_layers::{PlaceholderKind, Placeholders};
use turbo_unicode::category::{GeneralCategory, general_category};

use super::shared::{emoticon_end, laughter_at, shrink_laughter};
use crate::Normalizer;
use crate::placeholder::{self, Sink, span};

/// Replaces hashtags, handles, retweet markers and ASCII emoticons with
/// one U+0020 each, and shrinks repeated laughter.
///
/// - A hashtag or handle is `#` or `@` followed by a character that is
///   not ASCII whitespace (U+0009..U+000D, U+0020), up to the next ASCII
///   whitespace. It swallows everything else, further `#` and `@`, emoji
///   and no-break spaces included: `#a#b` is one hashtag.
/// - A retweet marker is a run of units `rt` followed by a space or a
///   colon, ASCII case-insensitively (`RT RT `, `rt:rt:`). It must start on
///   a word boundary in the sense of the JDK regular expression engine:
///   the character before it is not an ASCII letter, digit or `_`, and is
///   not a non-spacing mark (Mn) that sits on a letter or digit.
/// - An emoticon is eyes (`:`, `;`, `x`, `X`), an optional `-` nose and a
///   mouth (`( ) d D o O p P`). It matches inside words too: `expo`
///   becomes `"e o"`.
/// - Laughter is a run of `h`/`j`, a run of vowels (`a e i o u`), and at
///   least one more run of the last consonant and run of the last vowel,
///   ASCII case-insensitively. It shrinks to that consonant and vowel
///   twice, as written in the text: `hahaha` becomes `haha`, `hAHa`
///   becomes `hAhA`.
///
/// OpenNLP counterpart: `SocialMediaCharSequenceNormalizer`, which runs
/// four passes in the order above, each over the previous pass's output.
/// This normalizer runs one pass and produces the same text; see
/// [`SocialMedia::scan`] for why.
///
/// As an `Annotator` it reads a `View` and writes the normalized `View`
/// and a `Placeholders` layer with a `Hashtag`, `Mention`,
/// `RetweetMarker` or `Emoticon` row per replacement. Laughter is not a
/// placeholder and is not reported.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SocialMedia;

/// ASCII whitespace as OpenNLP's `AsciiChars.WHITESPACE`: U+0009..U+000D
/// and U+0020.
fn is_ascii_space(b: u8) -> bool {
    matches!(b, b'\t'..=b'\r' | b' ')
}

/// The end of the hashtag or handle at byte `i`, which holds `#` or `@`.
fn tag_end(bytes: &[u8], i: usize) -> Option<usize> {
    if !bytes.get(i + 1).is_some_and(|&b| !is_ascii_space(b)) {
        return None;
    }
    let rest = &bytes[i + 2..];
    Some(i + 2 + rest.iter().position(|&b| is_ascii_space(b)).unwrap_or(rest.len()))
}

fn tag_kind(b: u8) -> PlaceholderKind {
    if b == b'#' { PlaceholderKind::Hashtag } else { PlaceholderKind::Mention }
}

/// Whether the character before byte `i` blocks a word boundary there, as
/// the JDK regular expression engine decides `\b`: an ASCII word
/// character blocks it, and so does a non-spacing mark whose base
/// character is a letter or digit. The engine walks back over the marks
/// one UTF-16 unit at a time and stops at the low surrogate of any
/// supplementary character, so a supplementary character in that walk
/// (mark or base) never blocks.
fn boundary_blocked(text: &str, i: usize) -> bool {
    let mut before = text[..i].chars().rev();
    let Some(c) = before.next() else { return false };
    if c.is_ascii_alphanumeric() || c == '_' {
        return true;
    }
    if general_category(c) != GeneralCategory::Mn {
        return false;
    }
    for c in std::iter::once(c).chain(before) {
        if !c.is_ascii() && u32::from(c) > 0xFFFF {
            return false;
        }
        if turbo_unicode::java::is_letter_or_digit(c) {
            return true;
        }
        if general_category(c) != GeneralCategory::Mn {
            return false;
        }
    }
    false
}

/// What a retweet marker at a given position covers.
struct Retweet {
    /// The end of the `rt` units.
    end: usize,
    /// A hashtag or handle right after the last `rt`, with its end. In
    /// OpenNLP's passes the tag first becomes a space, and that space then
    /// ends the last unit, so the tag is replaced together with the
    /// marker.
    tag: Option<usize>,
}

/// The retweet marker at byte `i`, which holds `r` or `R`.
fn retweet_at(bytes: &[u8], i: usize) -> Option<Retweet> {
    let mut end = i;
    while end + 2 < bytes.len() && matches!(bytes[end], b'r' | b'R') && matches!(bytes[end + 1], b't' | b'T') {
        match bytes[end + 2] {
            b' ' | b':' => end += 3,
            b'#' | b'@' => match tag_end(bytes, end + 2) {
                Some(tag) => return Some(Retweet { end: end + 2, tag: Some(tag) }),
                None => break,
            },
            _ => break,
        }
    }
    (end > i).then_some(Retweet { end, tag: None })
}

impl SocialMedia {
    /// Writes `text` with hashtags, handles, retweet markers and emoticons
    /// replaced by one space each and laughter shrunk, and reports each
    /// placeholder to `sink`.
    ///
    /// OpenNLP runs four passes (tags, retweet markers, emoticons,
    /// laughter); this is one, trying them in that order at each position.
    /// The results are the same because the patterns barely touch:
    ///
    /// - Each pattern starts with characters no other pattern starts with
    ///   (`#@`, `rR`, `:;xX`, `hHjJ`), and none of the later three can
    ///   contain a character another pattern starts with, so at any
    ///   position at most one pattern can begin and a match never swallows
    ///   the start of an earlier pass's match. The space a replacement
    ///   leaves is not part of any emoticon or laughter.
    /// - A hashtag ends at ASCII whitespace or at the end of the text, so
    ///   no later pass ever sees the space it became as the character
    ///   before a retweet marker; the word-boundary check reads the
    ///   original text, which is what the retweet pass sees there.
    /// - The one interaction: the space a hashtag or handle becomes can end
    ///   an `rt` unit. `rt#tag` is `"rt "` after the first pass and one
    ///   space after the second. `retweet_at` accepts a tag in that place
    ///   and the whole stretch becomes one space (reported as a retweet
    ///   marker and a hashtag or mention).
    ///
    /// A randomized test compares this scan with a four-pass reference.
    pub fn scan(text: &str, out: &mut ViewBuilder<'_>, sink: &mut Sink<'_>) {
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            let found = match b {
                b'#' | b'@' => tag_end(bytes, i).map(|end| (end, tag_kind(b))),
                b'r' | b'R' if !boundary_blocked(text, i) => match retweet_at(bytes, i) {
                    Some(Retweet { end, tag: Some(tag) }) => {
                        out.keep(i);
                        out.replace(tag, " ");
                        sink(span(i, end), PlaceholderKind::RetweetMarker);
                        sink(span(end, tag), tag_kind(bytes[end]));
                        i = tag;
                        continue;
                    }
                    Some(Retweet { end, tag: None }) => Some((end, PlaceholderKind::RetweetMarker)),
                    None => None,
                },
                b':' | b';' | b'x' | b'X' => emoticon_end(bytes, i).map(|end| (end, PlaceholderKind::Emoticon)),
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

impl Normalizer for SocialMedia {
    const ID: &'static str = "turbo:normalize:social-media";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        SocialMedia::scan(text, out, &mut placeholder::discard);
    }
}

impl Annotator for SocialMedia {
    const ID: &'static str = <SocialMedia as Normalizer>::ID;
    type Reads = (View,);
    type Writes = (View, Placeholders);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View, Placeholders), Error> {
        Ok(placeholder::annotate(src, SocialMedia::scan))
    }
}
