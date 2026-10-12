//! `EmojiAnnotate`, which writes the `turbo:emoji` layer, and the lookups
//! that reproduce OpenNLP's `EmojiAnnotations` and `EmojiAnnotator`.

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_normalize_layers::Emoji;
use turbo_unicode::emoji::{EmojiGroup, name, region, scan};

use super::data::ANNOTATIONS;
use super::{EMOJI_SELECTOR, is_emoji_presentation, layer_kind};
use crate::hash_config;

/// Finds the emoji of a view and writes one `turbo:emoji` row per
/// sequence: its span in the view, its kind, its CLDR short name from the
/// Unicode 18.0 emoji-test.txt (empty when it has none) and the region a
/// flag or subdivision flag spells (empty for other emoji).
///
/// The sequences are those of `turbo_unicode::emoji::scan`. By default
/// they are filtered by [`super::is_emoji_presentation`], the rule
/// `EmojiRemove` uses, so `©`, `™` and a bare `❤` are not rows;
/// [`EmojiAnnotate::with_text_default`] keeps them, for callers that want
/// every character with the Emoji property.
///
/// OpenNLP counterpart: `EmojiAnnotator`, which annotates one token at a
/// time; this annotator finds the sequences itself, so it needs no
/// tokenizer and a token that holds an emoji and other text still yields
/// the emoji. OpenNLP's bundled facts and feature strings are
/// [`opennlp_annotation`] and [`features`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EmojiAnnotate {
    /// Also report text-default characters written without U+FE0F.
    pub text_default: bool,
}

impl EmojiAnnotate {
    /// Reports the sequences shown as emoji.
    pub fn new() -> EmojiAnnotate {
        EmojiAnnotate { text_default: false }
    }

    /// Reports every sequence the scanner finds, text-default characters
    /// included.
    pub fn with_text_default() -> EmojiAnnotate {
        EmojiAnnotate { text_default: true }
    }

    /// The emoji of `text`, as a layer over the view `view`.
    pub fn find(&self, text: &str, view: turbo_analysis::ViewId) -> Emoji {
        let mut found = Emoji::new(view);
        for (range, kind) in scan(text) {
            let sequence = &text[range.clone()];
            if !self.text_default && !is_emoji_presentation(sequence, kind) {
                continue;
            }
            found.spans.push(Span::new(range.start as u32, range.end as u32));
            found.kind.push(layer_kind(kind));
            found.name.push(name(sequence).unwrap_or(""));
            found.region.push(region(sequence).as_deref().unwrap_or(""));
        }
        found
    }
}

impl Annotator for EmojiAnnotate {
    const ID: &'static str = "turbo:normalize:emoji-annotate";
    type Reads = (View,);
    type Writes = (Emoji,);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(Emoji,), Error> {
        Ok((self.find(src.text(), src.id()),))
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.text_default)
    }
}

/// OpenNLP's `EmojiEntityType`: a coarse entity type assigned from the
/// emoji-test.txt subgroup.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EmojiEntityType {
    Face,
    Heart,
    Animal,
    Food,
    Vehicle,
    Landmark,
    Flag,
}

impl EmojiEntityType {
    /// The Java constant's name, as OpenNLP's feature strings write it.
    pub fn constant(self) -> &'static str {
        match self {
            EmojiEntityType::Face => "FACE",
            EmojiEntityType::Heart => "HEART",
            EmojiEntityType::Animal => "ANIMAL",
            EmojiEntityType::Food => "FOOD",
            EmojiEntityType::Vehicle => "VEHICLE",
            EmojiEntityType::Landmark => "LANDMARK",
            EmojiEntityType::Flag => "FLAG",
        }
    }
}

/// The name of OpenNLP's `EmojiCategory` constant for an emoji-test.txt
/// group. The two enums have the same ten members in the same order.
pub fn category_constant(group: EmojiGroup) -> &'static str {
    match group {
        EmojiGroup::SmileysAndEmotion => "SMILEYS_AND_EMOTION",
        EmojiGroup::PeopleAndBody => "PEOPLE_AND_BODY",
        EmojiGroup::Component => "COMPONENT",
        EmojiGroup::AnimalsAndNature => "ANIMALS_AND_NATURE",
        EmojiGroup::FoodAndDrink => "FOOD_AND_DRINK",
        EmojiGroup::TravelAndPlaces => "TRAVEL_AND_PLACES",
        EmojiGroup::Activities => "ACTIVITIES",
        EmojiGroup::Objects => "OBJECTS",
        EmojiGroup::Symbols => "SYMBOLS",
        EmojiGroup::Flags => "FLAGS",
    }
}

/// One symbol of OpenNLP's emoji-annotations.txt with its four
/// attributes. `symbol` has no U+FE0F, as the file's keys do not.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OpenNlpAnnotation {
    pub symbol: &'static str,
    /// The CLDR short name, as emoji-test.txt 17.0 printed it.
    pub name: &'static str,
    /// OpenNLP's coarse polarity, -2 to 2.
    pub sentiment: i8,
    /// The note of the sentiment row.
    pub sentiment_note: &'static str,
    pub entity_type: EmojiEntityType,
    /// The emoji-test.txt subgroup the entity type was assigned from.
    pub subgroup: &'static str,
    /// The emoji-test.txt group (OpenNLP's `EmojiCategory`).
    pub category: EmojiGroup,
}

/// OpenNLP's bundled annotation of `sequence`, as
/// `EmojiAnnotations.lookup` finds it: every U+FE0F is ignored, so `❤️`
/// and `❤` are the same symbol. Flags have no bundled annotation.
pub fn opennlp_annotation(sequence: &str) -> Option<&'static OpenNlpAnnotation> {
    let first = sequence.chars().find(|&c| c != EMOJI_SELECTOR)?;
    ANNOTATIONS.iter().find(|a| a.symbol.starts_with(first) && a.symbol.chars().eq(without_selector(sequence)))
}

fn without_selector(s: &str) -> impl Iterator<Item = char> + '_ {
    s.chars().filter(|&c| c != EMOJI_SELECTOR)
}

/// Appends the feature strings OpenNLP's `EmojiAnnotator.collectFeatures`
/// produces for a token, in its order: `emojiSentiment=2`,
/// `emojiType=FACE`, `emojiCategory=SMILEYS_AND_EMOTION`,
/// `emojiRegion=GB`.
///
/// A token whose first character is ASCII has none. The first three come
/// from the bundled annotation; a flag (two regional indicators, or
/// U+1F3F4 with tag characters and U+E007F) has no bundled annotation and
/// gives `emojiType=FLAG`, `emojiCategory=FLAGS` and its region.
pub fn features(token: &str, out: &mut Vec<String>) {
    if token.chars().next().is_none_or(|c| c.is_ascii()) {
        return;
    }
    let bundled = opennlp_annotation(token);
    let iso_region = region(token);
    if let Some(a) = bundled {
        out.push(format!("emojiSentiment={}", a.sentiment));
        out.push(format!("emojiType={}", a.entity_type.constant()));
        out.push(format!("emojiCategory={}", category_constant(a.category)));
    } else if iso_region.is_some() {
        out.push(format!("emojiType={}", EmojiEntityType::Flag.constant()));
        out.push(format!("emojiCategory={}", category_constant(EmojiGroup::Flags)));
    }
    if let Some(r) = iso_region {
        out.push(format!("emojiRegion={r}"));
    }
}
