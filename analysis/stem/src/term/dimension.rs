//! Normalization [`Dimension`]s and their default character-level normalizers.

use std::fmt;
use std::sync::LazyLock;

use turbo_normalize::{
    AccentFold, CaseFold, ConfusableSkeleton, Dashes, DynNormalizer, EmojiToEmoticon, UnicodeForm, Whitespace,
};

/// Total number of [`Dimension`] variants.
pub(crate) const NUM_DIMENSIONS: usize = 12;

/// A layer of the [`crate::Term`] normalization stack, in increasing order of
/// aggressiveness (`opennlp.tools.util.normalizer.Dimension`).
///
/// The declaration order is the canonical application order, because the
/// transforms do not commute. [`Dimension::Original`] is the source token text
/// and is always present. [`Dimension::Stem`] and [`Dimension::Lemma`] are
/// token-level and have no default normalizer; they require a [`crate::Stemmer`] or
/// [`crate::TermLemmatizer`] on the analyzer ([`Dimension::Lemma`] also requires a
/// part-of-speech tag).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Dimension {
    /// The original token text, the source of truth (`ORIGINAL`).
    Original = 0,
    /// Unicode canonical composition (`NFC`); lossless under canonical equivalence.
    Nfc = 1,
    /// Unicode compatibility composition (`NFKC`); lossy (for example superscripts to digits).
    Nfkc = 2,
    /// Unicode whitespace folded to ASCII spaces (`WHITESPACE`).
    Whitespace = 3,
    /// Unicode dashes folded to the ASCII hyphen-minus (`DASH`).
    Dash = 4,
    /// Case folding (`CASE_FOLD`); `Locale.ROOT` lowercasing by default.
    CaseFold = 5,
    /// Unicode full case folding (`FULL_CASE_FOLD`, UTS #21); expanding (`ß` to `ss`, ligatures).
    FullCaseFold = 6,
    /// Diacritic and accent folding (`ACCENT_FOLD`).
    AccentFold = 7,
    /// Emoji folded to ASCII emoticons where a mapping exists (`EMOJI_FOLD`).
    EmojiFold = 8,
    /// Confusable (homoglyph) skeleton folding per UTS #39 (`CONFUSABLE_FOLD`).
    ConfusableFold = 9,
    /// Stemming through a configured [`crate::Stemmer`] (`STEM`).
    Stem = 10,
    /// Lemmatization through a configured [`crate::TermLemmatizer`] (`LEMMA`).
    Lemma = 11,
}

static DEFAULT_NFC: LazyLock<UnicodeForm> = LazyLock::new(UnicodeForm::nfc);
static DEFAULT_NFKC: LazyLock<UnicodeForm> = LazyLock::new(UnicodeForm::nfkc);
static DEFAULT_WHITESPACE: LazyLock<Whitespace> = LazyLock::new(Whitespace::collapse);
static DEFAULT_DASH: LazyLock<Dashes> = LazyLock::new(Dashes::new);
static DEFAULT_CASE_FOLD: LazyLock<CaseFold> = LazyLock::new(CaseFold::lowercase);
static DEFAULT_FULL_CASE_FOLD: LazyLock<CaseFold> = LazyLock::new(CaseFold::full);
static DEFAULT_ACCENT_FOLD: LazyLock<AccentFold> = LazyLock::new(AccentFold::new);
static DEFAULT_EMOJI_FOLD: EmojiToEmoticon = EmojiToEmoticon;
static DEFAULT_CONFUSABLE_FOLD: ConfusableSkeleton = ConfusableSkeleton;

impl Dimension {
    /// All twelve dimensions in canonical pipeline order.
    pub const ALL: [Dimension; NUM_DIMENSIONS] = [
        Dimension::Original,
        Dimension::Nfc,
        Dimension::Nfkc,
        Dimension::Whitespace,
        Dimension::Dash,
        Dimension::CaseFold,
        Dimension::FullCaseFold,
        Dimension::AccentFold,
        Dimension::EmojiFold,
        Dimension::ConfusableFold,
        Dimension::Stem,
        Dimension::Lemma,
    ];

    /// Returns whether this dimension is a character-level normalizer
    /// ([`Dimension::Nfc`] through [`Dimension::ConfusableFold`]).
    pub const fn is_character_level(self) -> bool {
        matches!(
            self,
            Dimension::Nfc
                | Dimension::Nfkc
                | Dimension::Whitespace
                | Dimension::Dash
                | Dimension::CaseFold
                | Dimension::FullCaseFold
                | Dimension::AccentFold
                | Dimension::EmojiFold
                | Dimension::ConfusableFold
        )
    }

    /// Returns the default character-level normalizer for this dimension, or
    /// `None` for [`Dimension::Original`], [`Dimension::Stem`], and
    /// [`Dimension::Lemma`] (`Dimension.defaultNormalizer()`).
    pub fn default_normalizer(self) -> Option<&'static dyn DynNormalizer> {
        match self {
            Dimension::Original | Dimension::Stem | Dimension::Lemma => None,
            Dimension::Nfc => Some(&*DEFAULT_NFC),
            Dimension::Nfkc => Some(&*DEFAULT_NFKC),
            Dimension::Whitespace => Some(&*DEFAULT_WHITESPACE),
            Dimension::Dash => Some(&*DEFAULT_DASH),
            Dimension::CaseFold => Some(&*DEFAULT_CASE_FOLD),
            Dimension::FullCaseFold => Some(&*DEFAULT_FULL_CASE_FOLD),
            Dimension::AccentFold => Some(&*DEFAULT_ACCENT_FOLD),
            Dimension::EmojiFold => Some(&DEFAULT_EMOJI_FOLD),
            Dimension::ConfusableFold => Some(&DEFAULT_CONFUSABLE_FOLD),
        }
    }

    pub(crate) const fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Dimension::Original => "ORIGINAL",
            Dimension::Nfc => "NFC",
            Dimension::Nfkc => "NFKC",
            Dimension::Whitespace => "WHITESPACE",
            Dimension::Dash => "DASH",
            Dimension::CaseFold => "CASE_FOLD",
            Dimension::FullCaseFold => "FULL_CASE_FOLD",
            Dimension::AccentFold => "ACCENT_FOLD",
            Dimension::EmojiFold => "EMOJI_FOLD",
            Dimension::ConfusableFold => "CONFUSABLE_FOLD",
            Dimension::Stem => "STEM",
            Dimension::Lemma => "LEMMA",
        };
        f.write_str(name)
    }
}
