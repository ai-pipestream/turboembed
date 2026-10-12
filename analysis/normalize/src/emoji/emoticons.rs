//! `EmojiToEmoticon` and `EmoticonToEmoji`: OpenNLP's `EmojiEmoticons`
//! folds over the table of emoji-emoticons.txt.

use std::cmp::Reverse;
use std::sync::LazyLock;

use turbo_analysis::ViewBuilder;

use super::EMOJI_SELECTOR;
use super::data::{EMOTICON_FOLDS, EmoticonFold, Fold};
use crate::Normalizer;
use crate::tables::is_whitespace;

const ZWJ: char = '\u{200D}';
const TEXT_SELECTOR: char = '\u{FE0E}';

/// Folds the pictographs of emoji-emoticons.txt to their ASCII emoticons:
/// `🙂` and `😊` to `:)`, `❤️` and `💙` to `<3`, `😢` to `:'(`.
///
/// At each position the longest source that matches wins, so `❤️`
/// (U+2764 U+FE0F) folds as one unit. A match is skipped when the
/// character before it is U+200D or the character after it (past one
/// U+FE0F, which the fold takes along) is U+200D or U+FE0E, so a heart
/// inside a ZWJ sequence such as `❤️‍🔥` and a heart that asks for text
/// presentation stay as they are. A pictograph glued to a word folds:
/// `great😀` becomes `great:D`.
///
/// OpenNLP counterpart: `EmojiToEmoticonCharSequenceNormalizer`. No
/// differences. Each fold is one replacement, so `:'(` maps back to the
/// pictograph and its U+FE0F, and adjacent folds map to their own
/// sources.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EmojiToEmoticon;

/// Folds the ASCII emoticons of emoji-emoticons.txt to pictographs: `:)`
/// and `:-)` to `🙂`, `<3` to `❤️`, `</3` to `💔`.
///
/// An emoticon folds only when the text edge or a `White_Space` character
/// (the table of [`crate::tables::WHITESPACE`], no-break spaces included)
/// is on both sides of it, since the same characters occur in ordinary
/// text (`1:3`, `https://`, `nice:)`). At each position the longest
/// source wins, so `:-)` is one unit. The short form of each emoticon is
/// canonical, so a round trip through both directions turns `:-)` into
/// `:)`.
///
/// OpenNLP counterpart: `EmoticonToEmojiCharSequenceNormalizer`. No
/// differences.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EmoticonToEmoji;

impl Normalizer for EmojiToEmoticon {
    const ID: &'static str = "turbo:normalize:emoji-to-emoticon";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        substitute(text, out, &EMOJI_TO_EMOTICON, emoji_match);
    }
}

impl Normalizer for EmoticonToEmoji {
    const ID: &'static str = "turbo:normalize:emoticon-to-emoji";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        substitute(text, out, &EMOTICON_TO_EMOJI, emoticon_match);
    }
}

/// The rows of one fold direction, sorted by the first character of their
/// source and, for each first character, longest source first, so the
/// first candidate that matches is the longest match. Length is counted in
/// UTF-16 units and the sort is stable, as OpenNLP sorts its candidate
/// lists, so ties keep the file's order.
struct Direction {
    rows: Vec<(char, &'static EmoticonFold)>,
}

impl Direction {
    fn new(fold: Fold) -> Direction {
        let mut rows: Vec<(char, &'static EmoticonFold)> = EMOTICON_FOLDS
            .iter()
            .filter(|r| r.fold == fold)
            .map(|r| (r.source.chars().next().expect("a source is never empty"), r))
            .collect();
        rows.sort_by_key(|&(first, r)| (first, Reverse(r.source.encode_utf16().count())));
        Direction { rows }
    }

    /// The rows whose source starts with `c`, longest first.
    fn candidates(&self, c: char) -> &[(char, &'static EmoticonFold)] {
        let start = self.rows.partition_point(|&(first, _)| first < c);
        let end = start + self.rows[start..].partition_point(|&(first, _)| first == c);
        &self.rows[start..end]
    }
}

static EMOJI_TO_EMOTICON: LazyLock<Direction> = LazyLock::new(|| Direction::new(Fold::Emoji));
static EMOTICON_TO_EMOJI: LazyLock<Direction> = LazyLock::new(|| Direction::new(Fold::Emoticon));

/// Decides whether a candidate whose source matches at `start..end` folds,
/// and returns the end of the stretch it replaces.
type MatchRule = fn(text: &str, start: usize, end: usize) -> Option<usize>;

/// Copies `text` into `out`, replacing at each position the longest source
/// of `direction` that `rule` accepts, each with its own replacement.
fn substitute(text: &str, out: &mut ViewBuilder<'_>, direction: &Direction, rule: MatchRule) {
    let (Some(&(min, _)), Some(&(max, _))) = (direction.rows.first(), direction.rows.last()) else { return };
    let mut i = 0;
    while let Some(c) = text[i..].chars().next() {
        if (min..=max).contains(&c)
            && let Some((target, end)) = fold_at(text, i, direction.candidates(c), rule)
        {
            out.keep(i);
            out.replace(end, target);
            i = end;
        } else {
            i += c.len_utf8();
        }
    }
}

/// The first candidate whose source is at `i` and that `rule` accepts:
/// its target and the end of what it replaces.
fn fold_at(
    text: &str,
    i: usize,
    candidates: &[(char, &'static EmoticonFold)],
    rule: MatchRule,
) -> Option<(&'static str, usize)> {
    candidates.iter().find_map(|&(_, row)| {
        let end = i + row.source.len();
        if !text[i..].starts_with(row.source) {
            return None;
        }
        rule(text, i, end).map(|end| (row.target, end))
    })
}

fn char_before(text: &str, i: usize) -> Option<char> {
    text[..i].chars().next_back()
}

fn char_at(text: &str, i: usize) -> Option<char> {
    text[i..].chars().next()
}

/// The pictograph direction: not after U+200D; one following U+FE0F is
/// taken along; not followed by U+200D or U+FE0E.
fn emoji_match(text: &str, start: usize, mut end: usize) -> Option<usize> {
    if char_before(text, start) == Some(ZWJ) {
        return None;
    }
    if char_at(text, end) == Some(EMOJI_SELECTOR) {
        end += EMOJI_SELECTOR.len_utf8();
    }
    match char_at(text, end) {
        Some(ZWJ | TEXT_SELECTOR) => None,
        _ => Some(end),
    }
}

/// The emoticon direction: the text edge or whitespace on both sides.
fn emoticon_match(text: &str, start: usize, end: usize) -> Option<usize> {
    let before = char_before(text, start).is_none_or(is_whitespace);
    let after = char_at(text, end).is_none_or(is_whitespace);
    (before && after).then_some(end)
}
