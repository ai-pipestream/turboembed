//! Emoji: removing them, folding them to and from ASCII emoticons, and
//! annotating them.
//!
//! - [`EmojiRemove`] replaces emoji with spaces, either whole UTS #51
//!   sequences ([`EmojiRemove::sequences`]) or every supplementary-plane
//!   code point as OpenNLP's `EmojiCharSequenceNormalizer` does
//!   ([`EmojiRemove::supplementary`]).
//! - [`EmojiToEmoticon`] and [`EmoticonToEmoji`] port OpenNLP's
//!   `EmojiEmoticons` folds over the table of emoji-emoticons.txt.
//! - [`EmojiAnnotate`] writes the `turbo:emoji` layer, one row per emoji
//!   sequence; [`opennlp_annotation`] and [`features`] give OpenNLP's
//!   bundled annotations and the feature strings of its `EmojiAnnotator`.
//!
//! Which sequences count as emoji is decided once, by
//! [`is_emoji_presentation`], for both the remover and the annotator.

mod annotate;
pub mod data;
mod emoticons;
mod remove;

pub use annotate::{
    EmojiAnnotate, EmojiEntityType, OpenNlpAnnotation, category_constant, features, opennlp_annotation,
};
pub use emoticons::{EmojiToEmoticon, EmoticonToEmoji};
pub use remove::{EmojiRemove, EmojiRemoveMode};

use turbo_normalize_layers::EmojiKind;
use turbo_unicode::emoji::EmojiKindHint;
use turbo_unicode::props::emoji_presentation;

/// U+FE0F VARIATION SELECTOR-16, which asks for emoji presentation.
pub(crate) const EMOJI_SELECTOR: char = '\u{FE0F}';

/// Whether a sequence found by `turbo_unicode::emoji::scan` is shown as an
/// emoji, following the presentation rules of UTS #51 (section 4).
///
/// Every keycap, flag, modifier, tag and ZWJ sequence is. A single
/// character (a `Basic` sequence) is when it has the Emoji_Presentation
/// property, or when U+FE0F follows it. The characters this leaves out
/// are the text-default emoji written without U+FE0F: U+00A9 ©, U+00AE ®,
/// U+2122 ™, U+2194 ↔, U+2639 ☹, U+2764 ❤ and the rest of the 207
/// characters of Unicode 18.0 that have the Emoji property but not
/// Emoji_Presentation (the keycap bases `0`-`9`, `#` and `*` aside).
/// They have the Emoji property, so the scanner reports them, but they are
/// displayed as text unless U+FE0F asks otherwise, and in running text
/// they are usually symbols (`© 2024`, `A ↔ B`). No character
/// property separates © from ☹, so this one rule decides for all of them.
pub fn is_emoji_presentation(sequence: &str, kind: EmojiKindHint) -> bool {
    if kind != EmojiKindHint::Basic {
        return true;
    }
    let mut chars = sequence.chars();
    chars.next().is_some_and(emoji_presentation) || chars.next() == Some(EMOJI_SELECTOR)
}

/// The layer's kind for a scanner's kind. The two enums have the same
/// members; they live in different crates so that the layer crate does not
/// depend on the Unicode tables.
pub(crate) fn layer_kind(kind: EmojiKindHint) -> EmojiKind {
    match kind {
        EmojiKindHint::Basic => EmojiKind::Basic,
        EmojiKindHint::Keycap => EmojiKind::Keycap,
        EmojiKindHint::Flag => EmojiKind::Flag,
        EmojiKindHint::Modifier => EmojiKind::Modifier,
        EmojiKindHint::Tag => EmojiKind::Tag,
        EmojiKindHint::Zwj => EmojiKind::Zwj,
    }
}
