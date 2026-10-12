//! `EmojiRemove`: emoji become spaces.

use turbo_analysis::{Annotator, Error, Refs, View, ViewBuilder};
use turbo_normalize_layers::{PlaceholderKind, Placeholders};
use turbo_unicode::emoji::scan;

use super::is_emoji_presentation;
use crate::placeholder::{self, Sink, span};
use crate::{Normalizer, hash_config};

/// What [`EmojiRemove`] treats as an emoji.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum EmojiRemoveMode {
    /// Whole emoji sequences as `turbo_unicode::emoji::scan` finds them
    /// (UTS #51 possible emoji), shown as emoji by the rule of
    /// [`super::is_emoji_presentation`].
    #[default]
    Sequences,
    /// Every supplementary-plane code point (U+10000 and above), as
    /// OpenNLP's `EmojiCharSequenceNormalizer` does outside its legacy
    /// mode.
    Supplementary,
}

/// Replaces emoji with U+0020.
///
/// In [`EmojiRemoveMode::Sequences`] (the default) each emoji sequence is
/// one unit: a ZWJ family, a flag, a keycap, a skin-toned hand, a
/// subdivision flag and a character with its U+FE0F all become one space,
/// and sequences with no text between them become one space together.
/// Text-default characters without U+FE0F (`©`, `™`, `↔`, a bare `❤`) are
/// left alone. A U+FE0E after a character keeps it as text, and a joiner
/// that does not join two emoji stays in the text.
///
/// In [`EmojiRemoveMode::Supplementary`] every run of supplementary-plane
/// code points becomes one space, emoji or not: `😀`, but also `𝐀`
/// (U+1D400) and `𠀀` (U+20000). BMP emoji such as `☺` or `❤` and the
/// U+FE0F and U+200D inside a sequence stay, so a ZWJ sequence becomes
/// spaces around its joiners. This is the output of OpenNLP's
/// `EmojiCharSequenceNormalizer`, which the language detector models were
/// trained with.
///
/// As an `Annotator` it reads a `View` and writes the normalized `View`
/// and a `Placeholders` layer with an `Emoji` row per sequence (in
/// `Sequences` mode) or per run (in `Supplementary` mode). In `Sequences`
/// mode each sequence is written with its own replacement: the first of a
/// run of adjacent sequences becomes the space and the others become
/// nothing, so each placeholder's span is exactly one sequence.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EmojiRemove {
    pub mode: EmojiRemoveMode,
}

impl EmojiRemove {
    /// Removes whole emoji sequences.
    pub fn sequences() -> EmojiRemove {
        EmojiRemove { mode: EmojiRemoveMode::Sequences }
    }

    /// Removes supplementary-plane code points, as OpenNLP does.
    pub fn supplementary() -> EmojiRemove {
        EmojiRemove { mode: EmojiRemoveMode::Supplementary }
    }

    /// Writes `text` with its emoji replaced and reports each replaced
    /// stretch to `sink`.
    pub fn scan(&self, text: &str, out: &mut ViewBuilder<'_>, sink: &mut Sink<'_>) {
        match self.mode {
            EmojiRemoveMode::Sequences => remove_sequences(text, out, sink),
            EmojiRemoveMode::Supplementary => remove_supplementary(text, out, sink),
        }
    }
}

fn remove_sequences(text: &str, out: &mut ViewBuilder<'_>, sink: &mut Sink<'_>) {
    // The end of the last sequence removed, to tell whether the next one
    // follows it directly.
    let mut last_end = None;
    for (range, kind) in scan(text) {
        if !is_emoji_presentation(&text[range.clone()], kind) {
            continue;
        }
        if last_end == Some(range.start) {
            out.replace(range.end, "");
        } else {
            out.keep(range.start);
            out.replace(range.end, " ");
        }
        sink(span(range.start, range.end), PlaceholderKind::Emoji);
        last_end = Some(range.end);
    }
}

/// A supplementary-plane code point is the only one whose UTF-8 form is
/// four bytes, and those start with a byte of 0xF0 or more, so the scan
/// reads lead bytes and steps over whole characters.
fn remove_supplementary(text: &str, out: &mut ViewBuilder<'_>, sink: &mut Sink<'_>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] < 0xF0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i] >= 0xF0 {
            i += 4;
        }
        out.keep(start);
        out.replace(i, " ");
        sink(span(start, i), PlaceholderKind::Emoji);
    }
}

impl Normalizer for EmojiRemove {
    const ID: &'static str = "turbo:normalize:emoji-remove";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        self.scan(text, out, &mut placeholder::discard);
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.mode)
    }
}

impl Annotator for EmojiRemove {
    const ID: &'static str = <EmojiRemove as Normalizer>::ID;
    type Reads = (View,);
    type Writes = (View, Placeholders);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View, Placeholders), Error> {
        Ok(placeholder::annotate(src, |text, out, sink| self.scan(text, out, sink)))
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.mode)
    }
}
