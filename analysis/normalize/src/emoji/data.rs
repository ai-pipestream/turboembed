//! The data of OpenNLP's emoji normalizers and annotator, as Rust
//! constants.
//!
//! [`EMOTICON_FOLDS`] is emoji-emoticons.txt and [`ANNOTATIONS`] is
//! emoji-annotations.txt, both from Apache OpenNLP
//! (`opennlp-core/opennlp-runtime/src/main/resources/opennlp/tools/util/normalizer/`),
//! licensed under the Apache License, Version 2.0, and authored by the
//! OpenNLP project. The rows are in the files' order. The tests in
//! `analysis/normalize/tests/emoji.rs` hold the audits OpenNLP's
//! `EmojiEmoticonsTest` and `EmojiAnnotationsTest` run on the files, and
//! check every name, group and subgroup against the Unicode 18.0
//! emoji-test.txt.

use turbo_unicode::emoji::EmojiGroup;

use super::annotate::{EmojiEntityType, OpenNlpAnnotation};

/// Which way a row of emoji-emoticons.txt folds (its `fold_type` column).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Fold {
    /// `EMOJI`: a pictograph to its ASCII emoticon.
    Emoji,
    /// `EMOTICON`: an ASCII emoticon to its pictograph.
    Emoticon,
}

/// One row of emoji-emoticons.txt:
/// `source ; target ; fold_type ; standard ; unicode_version ; notes`.
/// The file writes `source` and `target` as hexadecimal code points; here
/// they are the strings those code points spell.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EmoticonFold {
    pub source: &'static str,
    pub target: &'static str,
    pub fold: Fold,
    /// `UNSPECIFIED` on every row: the mappings are OpenNLP's own.
    pub standard: &'static str,
    pub unicode_version: &'static str,
    pub note: &'static str,
}

const fn row(
    source: &'static str,
    target: &'static str,
    fold: Fold,
    standard: &'static str,
    unicode_version: &'static str,
    note: &'static str,
) -> EmoticonFold {
    EmoticonFold { source, target, fold, standard, unicode_version, note }
}

/// emoji-emoticons.txt: 35 `EMOJI` rows, then 26 `EMOTICON` rows.
pub const EMOTICON_FOLDS: &[EmoticonFold] = &[
    row("\u{1F642}", ":)", Fold::Emoji, "UNSPECIFIED", "17.0.0", "SLIGHTLY SMILING FACE -> :)"),
    row("\u{1F60A}", ":)", Fold::Emoji, "UNSPECIFIED", "17.0.0", "SMILING FACE WITH SMILING EYES -> :)"),
    row("\u{263A}\u{FE0F}", ":)", Fold::Emoji, "UNSPECIFIED", "17.0.0", "WHITE SMILING FACE, emoji presentation -> :)"),
    row("\u{263A}", ":)", Fold::Emoji, "UNSPECIFIED", "17.0.0", "WHITE SMILING FACE -> :)"),
    row("\u{1F600}", ":D", Fold::Emoji, "UNSPECIFIED", "17.0.0", "GRINNING FACE -> :D"),
    row("\u{1F601}", ":D", Fold::Emoji, "UNSPECIFIED", "17.0.0", "GRINNING FACE WITH SMILING EYES -> :D"),
    row("\u{1F603}", ":D", Fold::Emoji, "UNSPECIFIED", "17.0.0", "SMILING FACE WITH OPEN MOUTH -> :D"),
    row("\u{1F604}", ":D", Fold::Emoji, "UNSPECIFIED", "17.0.0", "SMILING FACE WITH OPEN MOUTH AND SMILING EYES -> :D"),
    row("\u{1F641}", ":(", Fold::Emoji, "UNSPECIFIED", "17.0.0", "SLIGHTLY FROWNING FACE -> :("),
    row(
        "\u{2639}\u{FE0F}",
        ":(",
        Fold::Emoji,
        "UNSPECIFIED",
        "17.0.0",
        "WHITE FROWNING FACE, emoji presentation -> :(",
    ),
    row("\u{2639}", ":(", Fold::Emoji, "UNSPECIFIED", "17.0.0", "WHITE FROWNING FACE -> :("),
    row("\u{1F609}", ";)", Fold::Emoji, "UNSPECIFIED", "17.0.0", "WINKING FACE -> ;)"),
    row("\u{1F61B}", ":P", Fold::Emoji, "UNSPECIFIED", "17.0.0", "FACE WITH STUCK-OUT TONGUE -> :P"),
    row("\u{1F61C}", ":P", Fold::Emoji, "UNSPECIFIED", "17.0.0", "FACE WITH STUCK-OUT TONGUE AND WINKING EYE -> :P"),
    row(
        "\u{1F61D}",
        ":P",
        Fold::Emoji,
        "UNSPECIFIED",
        "17.0.0",
        "FACE WITH STUCK-OUT TONGUE AND TIGHTLY-CLOSED EYES -> :P",
    ),
    row("\u{1F62E}", ":O", Fold::Emoji, "UNSPECIFIED", "17.0.0", "FACE WITH OPEN MOUTH -> :O"),
    row("\u{1F632}", ":O", Fold::Emoji, "UNSPECIFIED", "17.0.0", "ASTONISHED FACE -> :O"),
    row("\u{1F622}", ":'(", Fold::Emoji, "UNSPECIFIED", "17.0.0", "CRYING FACE -> :'("),
    row("\u{1F62D}", ":'(", Fold::Emoji, "UNSPECIFIED", "17.0.0", "LOUDLY CRYING FACE -> :'("),
    row("\u{1F610}", ":|", Fold::Emoji, "UNSPECIFIED", "17.0.0", "NEUTRAL FACE -> :|"),
    row("\u{1F611}", ":|", Fold::Emoji, "UNSPECIFIED", "17.0.0", "EXPRESSIONLESS FACE -> :|"),
    row("\u{1F615}", ":/", Fold::Emoji, "UNSPECIFIED", "17.0.0", "CONFUSED FACE -> :/"),
    row("\u{1F617}", ":*", Fold::Emoji, "UNSPECIFIED", "17.0.0", "KISSING FACE -> :*"),
    row("\u{1F618}", ":*", Fold::Emoji, "UNSPECIFIED", "17.0.0", "FACE THROWING A KISS -> :*"),
    row("\u{2764}\u{FE0F}", "<3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "HEAVY BLACK HEART, emoji presentation -> <3"),
    row("\u{2764}", "<3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "HEAVY BLACK HEART -> <3"),
    row("\u{1F499}", "<3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "BLUE HEART -> <3"),
    row("\u{1F49A}", "<3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "GREEN HEART -> <3"),
    row("\u{1F49B}", "<3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "YELLOW HEART -> <3"),
    row("\u{1F49C}", "<3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "PURPLE HEART -> <3"),
    row("\u{1F5A4}", "<3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "BLACK HEART -> <3"),
    row("\u{1F90D}", "<3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "WHITE HEART -> <3"),
    row("\u{1F90E}", "<3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "BROWN HEART -> <3"),
    row("\u{1F9E1}", "<3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "ORANGE HEART -> <3"),
    row("\u{1F494}", "</3", Fold::Emoji, "UNSPECIFIED", "17.0.0", "BROKEN HEART -> </3"),
    row(":)", "\u{1F642}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":) -> SLIGHTLY SMILING FACE"),
    row(":-)", "\u{1F642}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":-) -> SLIGHTLY SMILING FACE"),
    row("=)", "\u{1F642}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", "=) -> SLIGHTLY SMILING FACE"),
    row(":D", "\u{1F600}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":D -> GRINNING FACE"),
    row(":-D", "\u{1F600}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":-D -> GRINNING FACE"),
    row(":(", "\u{1F641}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":( -> SLIGHTLY FROWNING FACE"),
    row(":-(", "\u{1F641}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":-( -> SLIGHTLY FROWNING FACE"),
    row(";)", "\u{1F609}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ";) -> WINKING FACE"),
    row(";-)", "\u{1F609}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ";-) -> WINKING FACE"),
    row(":P", "\u{1F61B}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":P -> FACE WITH STUCK-OUT TONGUE"),
    row(":-P", "\u{1F61B}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":-P -> FACE WITH STUCK-OUT TONGUE"),
    row(":p", "\u{1F61B}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":p -> FACE WITH STUCK-OUT TONGUE"),
    row(":-p", "\u{1F61B}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":-p -> FACE WITH STUCK-OUT TONGUE"),
    row(":O", "\u{1F62E}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":O -> FACE WITH OPEN MOUTH"),
    row(":-O", "\u{1F62E}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":-O -> FACE WITH OPEN MOUTH"),
    row(":o", "\u{1F62E}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":o -> FACE WITH OPEN MOUTH"),
    row(":-o", "\u{1F62E}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":-o -> FACE WITH OPEN MOUTH"),
    row(":'(", "\u{1F622}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":'( -> CRYING FACE"),
    row(":|", "\u{1F610}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":| -> NEUTRAL FACE"),
    row(":-|", "\u{1F610}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":-| -> NEUTRAL FACE"),
    row(":/", "\u{1F615}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":/ -> CONFUSED FACE"),
    row(":-/", "\u{1F615}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":-/ -> CONFUSED FACE"),
    row(":*", "\u{1F617}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":* -> KISSING FACE"),
    row(":-*", "\u{1F617}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", ":-* -> KISSING FACE"),
    row(
        "<3",
        "\u{2764}\u{FE0F}",
        Fold::Emoticon,
        "UNSPECIFIED",
        "17.0.0",
        "<3 -> HEAVY BLACK HEART, emoji presentation",
    ),
    row("</3", "\u{1F494}", Fold::Emoticon, "UNSPECIFIED", "17.0.0", "</3 -> BROKEN HEART"),
];

/// The `source` column of the `name` rows of emoji-annotations.txt.
pub const NAME_SOURCE: &str = "CLDR:annotation";
/// The `notes` column of the `name` rows.
pub const NAME_NOTE: &str = "emoji-test.txt v17.0 short name";
/// The `source` column of the `sentiment` rows: a judgment of the OpenNLP
/// project, not third-party data.
pub const SENTIMENT_SOURCE: &str = "UNSPECIFIED";
/// The `source` column of the `entityType` and `category` rows.
pub const EMOJI_TEST_SOURCE: &str = "UCD:emoji-test";

const fn ann(
    symbol: &'static str,
    name: &'static str,
    sentiment: i8,
    sentiment_note: &'static str,
    entity_type: EmojiEntityType,
    subgroup: &'static str,
    category: EmojiGroup,
) -> OpenNlpAnnotation {
    OpenNlpAnnotation { symbol, name, sentiment, sentiment_note, entity_type, subgroup, category }
}

/// emoji-annotations.txt, one entry per symbol. The file has four rows per
/// symbol (`name`, `sentiment`, `entityType`, `category`), each with a
/// source and a note. The sources are the same on every row of an
/// attribute ([`NAME_SOURCE`], [`SENTIMENT_SOURCE`], [`EMOJI_TEST_SOURCE`]),
/// as are the notes of the name rows ([`NAME_NOTE`]); the note of an
/// `entityType` row is `subgroup <subgroup>` and of a `category` row
/// `group <group name>`, so an entry keeps the subgroup and the sentiment
/// note, and the rest follows from these constants.
pub const ANNOTATIONS: &[OpenNlpAnnotation] = &[
    ann(
        "\u{1F600}",
        "grinning face",
        2,
        "project judgment",
        EmojiEntityType::Face,
        "face-smiling",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F603}",
        "grinning face with big eyes",
        2,
        "project judgment",
        EmojiEntityType::Face,
        "face-smiling",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F604}",
        "grinning face with smiling eyes",
        2,
        "project judgment",
        EmojiEntityType::Face,
        "face-smiling",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F601}",
        "beaming face with smiling eyes",
        2,
        "project judgment",
        EmojiEntityType::Face,
        "face-smiling",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F642}",
        "slightly smiling face",
        1,
        "project judgment",
        EmojiEntityType::Face,
        "face-smiling",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F609}",
        "winking face",
        1,
        "project judgment",
        EmojiEntityType::Face,
        "face-smiling",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F60A}",
        "smiling face with smiling eyes",
        2,
        "project judgment",
        EmojiEntityType::Face,
        "face-smiling",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{263A}",
        "smiling face",
        1,
        "project judgment",
        EmojiEntityType::Face,
        "face-affection",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F617}",
        "kissing face",
        1,
        "project judgment",
        EmojiEntityType::Face,
        "face-affection",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F618}",
        "face blowing a kiss",
        2,
        "project judgment",
        EmojiEntityType::Face,
        "face-affection",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F61B}",
        "face with tongue",
        1,
        "project judgment",
        EmojiEntityType::Face,
        "face-tongue",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F61C}",
        "winking face with tongue",
        1,
        "project judgment",
        EmojiEntityType::Face,
        "face-tongue",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F61D}",
        "squinting face with tongue",
        1,
        "project judgment",
        EmojiEntityType::Face,
        "face-tongue",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F610}",
        "neutral face",
        0,
        "project judgment",
        EmojiEntityType::Face,
        "face-neutral-skeptical",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F611}",
        "expressionless face",
        0,
        "project judgment",
        EmojiEntityType::Face,
        "face-neutral-skeptical",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F615}",
        "confused face",
        -1,
        "project judgment",
        EmojiEntityType::Face,
        "face-concerned",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F641}",
        "slightly frowning face",
        -1,
        "project judgment",
        EmojiEntityType::Face,
        "face-concerned",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{2639}",
        "frowning face",
        -1,
        "project judgment",
        EmojiEntityType::Face,
        "face-concerned",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F62E}",
        "face with open mouth",
        0,
        "project judgment",
        EmojiEntityType::Face,
        "face-concerned",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F632}",
        "astonished face",
        0,
        "project judgment",
        EmojiEntityType::Face,
        "face-concerned",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F622}",
        "crying face",
        -2,
        "project judgment",
        EmojiEntityType::Face,
        "face-concerned",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F62D}",
        "loudly crying face",
        -2,
        "project judgment",
        EmojiEntityType::Face,
        "face-concerned",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann("\u{2764}", "red heart", 2, "project judgment", EmojiEntityType::Heart, "heart", EmojiGroup::SmileysAndEmotion),
    ann(
        "\u{1F9E1}",
        "orange heart",
        2,
        "project judgment",
        EmojiEntityType::Heart,
        "heart",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F49B}",
        "yellow heart",
        2,
        "project judgment",
        EmojiEntityType::Heart,
        "heart",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F49A}",
        "green heart",
        2,
        "project judgment",
        EmojiEntityType::Heart,
        "heart",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F499}",
        "blue heart",
        2,
        "project judgment",
        EmojiEntityType::Heart,
        "heart",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F49C}",
        "purple heart",
        2,
        "project judgment",
        EmojiEntityType::Heart,
        "heart",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F90E}",
        "brown heart",
        2,
        "project judgment",
        EmojiEntityType::Heart,
        "heart",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F5A4}",
        "black heart",
        0,
        "project judgment, polarity ambiguous in running text",
        EmojiEntityType::Heart,
        "heart",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F90D}",
        "white heart",
        2,
        "project judgment",
        EmojiEntityType::Heart,
        "heart",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F494}",
        "broken heart",
        -2,
        "project judgment",
        EmojiEntityType::Heart,
        "heart",
        EmojiGroup::SmileysAndEmotion,
    ),
    ann(
        "\u{1F436}",
        "dog face",
        0,
        "project judgment",
        EmojiEntityType::Animal,
        "animal-mammal",
        EmojiGroup::AnimalsAndNature,
    ),
    ann(
        "\u{1F431}",
        "cat face",
        0,
        "project judgment",
        EmojiEntityType::Animal,
        "animal-mammal",
        EmojiGroup::AnimalsAndNature,
    ),
    ann("\u{1F355}", "pizza", 0, "project judgment", EmojiEntityType::Food, "food-prepared", EmojiGroup::FoodAndDrink),
    ann("\u{1F34E}", "red apple", 0, "project judgment", EmojiEntityType::Food, "food-fruit", EmojiGroup::FoodAndDrink),
    ann(
        "\u{1F697}",
        "automobile",
        0,
        "project judgment",
        EmojiEntityType::Vehicle,
        "transport-ground",
        EmojiGroup::TravelAndPlaces,
    ),
    ann(
        "\u{2708}",
        "airplane",
        0,
        "project judgment",
        EmojiEntityType::Vehicle,
        "transport-air",
        EmojiGroup::TravelAndPlaces,
    ),
    ann(
        "\u{1F5FD}",
        "Statue of Liberty",
        0,
        "project judgment",
        EmojiEntityType::Landmark,
        "place-building",
        EmojiGroup::TravelAndPlaces,
    ),
    ann(
        "\u{1F5FC}",
        "Tokyo tower",
        0,
        "project judgment",
        EmojiEntityType::Landmark,
        "place-building",
        EmojiGroup::TravelAndPlaces,
    ),
];
