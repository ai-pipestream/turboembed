//! Emoji sequences: the RGI set, CLDR names, groups and subgroups from
//! emoji-test.txt, flag decoding, and [`scan`], which finds emoji in text.

mod flag;
mod scan;

use std::cmp::Ordering;

use crate::tables::emoji::{GROUPS, NAME_KEYS, NAMES, RGI, SUBGROUP_IDS, SUBGROUPS};

pub use flag::region;
pub use scan::{EmojiKindHint, Scan, scan};

const PRESENTATION_SELECTOR: char = '\u{FE0F}';

/// The group of an emoji in emoji-test.txt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EmojiGroup {
    /// Smileys & Emotion.
    SmileysAndEmotion,
    /// People & Body.
    PeopleAndBody,
    /// Component: skin tones and hair styles.
    Component,
    /// Animals & Nature.
    AnimalsAndNature,
    /// Food & Drink.
    FoodAndDrink,
    /// Travel & Places.
    TravelAndPlaces,
    /// Activities.
    Activities,
    /// Objects.
    Objects,
    /// Symbols.
    Symbols,
    /// Flags.
    Flags,
}

impl EmojiGroup {
    /// Every group, in the order of emoji-test.txt.
    pub const ALL: [EmojiGroup; 10] = [
        EmojiGroup::SmileysAndEmotion,
        EmojiGroup::PeopleAndBody,
        EmojiGroup::Component,
        EmojiGroup::AnimalsAndNature,
        EmojiGroup::FoodAndDrink,
        EmojiGroup::TravelAndPlaces,
        EmojiGroup::Activities,
        EmojiGroup::Objects,
        EmojiGroup::Symbols,
        EmojiGroup::Flags,
    ];

    /// The group's name in emoji-test.txt, for example `"Smileys & Emotion"`.
    pub fn name(self) -> &'static str {
        match self {
            EmojiGroup::SmileysAndEmotion => "Smileys & Emotion",
            EmojiGroup::PeopleAndBody => "People & Body",
            EmojiGroup::Component => "Component",
            EmojiGroup::AnimalsAndNature => "Animals & Nature",
            EmojiGroup::FoodAndDrink => "Food & Drink",
            EmojiGroup::TravelAndPlaces => "Travel & Places",
            EmojiGroup::Activities => "Activities",
            EmojiGroup::Objects => "Objects",
            EmojiGroup::Symbols => "Symbols",
            EmojiGroup::Flags => "Flags",
        }
    }
}

/// Whether `sequence` is exactly one RGI emoji: an element of the
/// RGI_Emoji set of UTS #51, which is every sequence listed in
/// emoji-sequences.txt and emoji-zwj-sequences.txt. The comparison is
/// exact, so a sequence that lacks a U+FE0F the RGI form has is not RGI.
pub fn is_rgi(sequence: &str) -> bool {
    RGI.binary_search(&sequence).is_ok()
}

/// The CLDR short name of `sequence` from emoji-test.txt, for example
/// `"grinning face"`. U+FE0F is ignored, so the minimally-qualified and
/// unqualified forms of an emoji have the name of its fully-qualified form.
pub fn name(sequence: &str) -> Option<&'static str> {
    position(sequence).map(|i| NAMES[i])
}

/// The emoji-test.txt group of `sequence`, ignoring U+FE0F.
pub fn group(sequence: &str) -> Option<EmojiGroup> {
    position(sequence).map(|i| EmojiGroup::ALL[GROUPS[i] as usize])
}

/// The emoji-test.txt subgroup of `sequence`, for example
/// `"face-smiling"`, ignoring U+FE0F.
pub fn subgroup(sequence: &str) -> Option<&'static str> {
    position(sequence).map(|i| SUBGROUPS[SUBGROUP_IDS[i] as usize])
}

/// The position of `sequence` in `NAME_KEYS`, whose keys have no U+FE0F.
/// The search compares without building a copy of `sequence`.
fn position(sequence: &str) -> Option<usize> {
    NAME_KEYS.binary_search_by(|key| compare_without_selector(key, sequence)).ok()
}

/// Compares `key` with `sequence` with every U+FE0F of `sequence` left
/// out. Comparing characters gives the same order as comparing UTF-8 bytes,
/// the order of `NAME_KEYS`.
fn compare_without_selector(key: &str, sequence: &str) -> Ordering {
    key.chars().cmp(sequence.chars().filter(|&c| c != PRESENTATION_SELECTOR))
}
