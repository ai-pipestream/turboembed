//! The emoji features, the presentation rule, and `EmojiAnnotate`.

use super::*;

// ---------------------------------------------------------------------------
// Features (EmojiAnnotatorTest, EmojiFlagsTest, EmojiAnnotatorUsageExampleTest)
// ---------------------------------------------------------------------------

fn feats(token: &str) -> Vec<String> {
    let mut out = Vec::new();
    features(token, &mut out);
    out
}

#[test]
fn features_of_a_bundled_symbol() {
    let expected = ["emojiSentiment=2", "emojiType=HEART", "emojiCategory=SMILEYS_AND_EMOTION"];
    assert_eq!(feats(&cp(&[0x2764, 0xFE0F])), expected);
    assert_eq!(feats(&cp(&[0x2764])), expected);
    assert_eq!(feats(&cp(&[0x1F494])), ["emojiSentiment=-2", "emojiType=HEART", "emojiCategory=SMILEYS_AND_EMOTION"]);
    assert_eq!(feats(&cp(&[0x2708])), ["emojiSentiment=0", "emojiType=VEHICLE", "emojiCategory=TRAVEL_AND_PLACES"]);
}

#[test]
fn features_of_a_flag_are_derived() {
    assert_eq!(feats(&flag("DE")), ["emojiType=FLAG", "emojiCategory=FLAGS", "emojiRegion=DE"]);
    for (code, region) in [("US", "US"), ("JP", "JP"), ("UN", "UN"), ("EU", "EU"), ("XX", "XX")] {
        assert_eq!(feats(&flag(code)).last().unwrap(), &format!("emojiRegion={region}"));
    }
    for (tags, region) in [("gbeng", "GB-ENG"), ("gbsct", "GB-SCT"), ("gbwls", "GB-WLS"), ("fr75", "FR-75")] {
        assert_eq!(feats(&tag_flag(tags)), ["emojiType=FLAG", "emojiCategory=FLAGS", &format!("emojiRegion={region}")]);
    }
}

/// Tokens OpenNLP's annotator returns nothing for: words, an emoticon, a
/// lone regional indicator, an unannotated pictograph, and the malformed
/// flags EmojiFlagsTest lists (the lenient decoding gives no region).
#[test]
fn features_are_total_over_arbitrary_tokens() {
    let black = cp(&[0x1F3F4]);
    let cancel = cp(&[0xE007F]);
    let tags = |s: &str| tag_flag(s).trim_start_matches(black.as_str()).trim_end_matches(cancel.as_str()).to_string();
    let tokens = [
        "word".to_string(),
        String::new(),
        ":-)".to_string(),
        flag("D"),
        cp(&[0x1F9F8]),
        flag("DEF"),
        flag("DEFR"),
        format!("{}E", flag("D")),
        format!("{}!", flag("DE")),
        format!("A{}", flag("DE")),
        black.clone(),
        cp(&[0x1F3F4, 0x200D, 0x2620, 0xFE0F]),
        format!("{black}{}", tags("gbeng")),
        format!("{black}{cancel}"),
        tag_flag("gb"),
        tag_flag("1beng"),
        format!("{black}{}e{}{cancel}", tags("gb"), tags("ng")),
        format!("{}x", tag_flag("gbeng")),
    ];
    for t in &tokens {
        assert_eq!(feats(t), Vec::<String>::new(), "{t:?}");
    }
}

#[test]
fn feature_strings_use_the_java_constant_names() {
    let all: Vec<&str> = EmojiGroup::ALL.iter().map(|&g| category_constant(g)).collect();
    assert_eq!(
        all,
        [
            "SMILEYS_AND_EMOTION",
            "PEOPLE_AND_BODY",
            "COMPONENT",
            "ANIMALS_AND_NATURE",
            "FOOD_AND_DRINK",
            "TRAVEL_AND_PLACES",
            "ACTIVITIES",
            "OBJECTS",
            "SYMBOLS",
            "FLAGS"
        ]
    );
}

/// EmojiAnnotatorUsageExampleTest: "Berlin 🇩🇪 ❤️" gives the region DE, the
/// name "red heart", FLAG then HEART, FLAGS then SMILEYS_AND_EMOTION and
/// the sentiment 2. Here the sequences come from the annotator's layer.
#[test]
fn usage_example_annotates_flag_and_heart() {
    let text = format!("Berlin {} {}", flag("DE"), cp(&[0x2764, 0xFE0F]));
    let layer = EmojiAnnotate::new().find(&text, turbo_analysis::ViewId::SOURCE);
    let sequences: Vec<&str> = layer.spans.iter().map(|(_, s)| &text[s.range()]).collect();
    assert_eq!(sequences, [flag("DE"), cp(&[0x2764, 0xFE0F])]);
    let mut all = Vec::new();
    for s in &sequences {
        features(s, &mut all);
    }
    assert_eq!(
        all,
        [
            "emojiType=FLAG",
            "emojiCategory=FLAGS",
            "emojiRegion=DE",
            "emojiSentiment=2",
            "emojiType=HEART",
            "emojiCategory=SMILEYS_AND_EMOTION"
        ]
    );
    assert_eq!(layer.region.iter().collect::<Vec<_>>(), ["DE", ""]);
    assert_eq!(layer.name.iter().collect::<Vec<_>>(), ["flag: Germany", "red heart"]);
    assert!(feats("Berlin").is_empty());
}

// ---------------------------------------------------------------------------
// The presentation rule
// ---------------------------------------------------------------------------

fn basic(s: &str) -> bool {
    is_emoji_presentation(s, EmojiKindHint::Basic)
}

#[test]
fn text_default_characters_need_the_emoji_selector() {
    for c in ['\u{A9}', '\u{AE}', '\u{2122}', '\u{2194}', '\u{2639}', '\u{263A}', '\u{2764}', '\u{2708}'] {
        assert!(!basic(&c.to_string()), "{c:?} alone");
        assert!(basic(&format!("{c}\u{FE0F}")), "{c:?} with U+FE0F");
    }
    for c in ['\u{1F600}', '\u{231A}', '\u{1F1E9}', '\u{1F3FB}'] {
        assert!(basic(&c.to_string()), "{c:?}");
    }
    // Sequences of the other kinds are emoji whatever their first character.
    for kind in
        [EmojiKindHint::Keycap, EmojiKindHint::Flag, EmojiKindHint::Modifier, EmojiKindHint::Tag, EmojiKindHint::Zwj]
    {
        assert!(is_emoji_presentation("\u{261D}\u{1F3FB}", kind));
    }
}

// ---------------------------------------------------------------------------
// EmojiAnnotate
// ---------------------------------------------------------------------------

#[test]
fn annotator_reports_kind_name_and_region() {
    let family = cp(&[0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467]);
    let text = format!(
        "{} {} {} {} {} {} \u{A9} {}",
        cp(&[0x1F600]),
        cp(&[0x31, 0xFE0F, 0x20E3]),
        flag("JP"),
        cp(&[0x1F44B, 0x1F3FF]),
        tag_flag("gbsct"),
        family,
        cp(&[0x1F3F4, 0x200D, 0x2620, 0xFE0F]),
    );
    let layer = EmojiAnnotate::new().find(&text, turbo_analysis::ViewId::SOURCE);
    layer.check().unwrap();
    let kinds: Vec<EmojiKind> = layer.kind.clone();
    assert_eq!(
        kinds,
        [
            EmojiKind::Basic,
            EmojiKind::Keycap,
            EmojiKind::Flag,
            EmojiKind::Modifier,
            EmojiKind::Tag,
            EmojiKind::Zwj,
            EmojiKind::Zwj
        ]
    );
    assert_eq!(
        layer.name.iter().collect::<Vec<_>>(),
        [
            "grinning face",
            "keycap: 1",
            "flag: Japan",
            "waving hand: dark skin tone",
            "flag: Scotland",
            "family: man, woman, girl",
            "pirate flag"
        ]
    );
    assert_eq!(layer.region.iter().collect::<Vec<_>>(), ["", "", "JP", "", "GB-SCT", "", ""]);
    // © is reported only when text-default characters are asked for.
    let all = EmojiAnnotate::with_text_default().find(&text, turbo_analysis::ViewId::SOURCE);
    assert_eq!(all.spans.len(), 8);
    assert_eq!(all.name.get(6), "copyright");
}

#[test]
fn annotator_names_sequences_outside_the_rgi_set_with_an_empty_name() {
    // A ZWJ sequence Unicode does not list, and an unassigned flag.
    let text = format!("{} {}", cp(&[0x1F680, 0x200D, 0x1F355]), flag("XX"));
    let layer = EmojiAnnotate::new().find(&text, turbo_analysis::ViewId::SOURCE);
    assert_eq!(layer.name.iter().collect::<Vec<_>>(), ["", ""]);
    assert_eq!(layer.region.iter().collect::<Vec<_>>(), ["", "XX"]);
    assert_eq!(layer.kind, [EmojiKind::Zwj, EmojiKind::Flag]);
}

/// The layer in a pipeline: emoji found on the source, removed into a new
/// view, and every row read back and mapped through the views.
#[test]
fn annotator_and_remover_in_a_pipeline() {
    let analyzer = Analyzer::builder()
        .step(Bound::new(EmojiAnnotate::new()))
        .step(Bound::new(EmojiRemove::sequences()).write(0, "no-emoji").write(1, "emoji"))
        .build()
        .unwrap();
    let src = format!("Berlin {} is {}{}!", flag("DE"), cp(&[0x2764, 0xFE0F]), cp(&[0x1F600]));
    let mut docs = vec![Document::new(src.as_str())];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let doc = &docs[0];
    let emoji = doc.get::<Emoji>().unwrap();
    let texts: Vec<&str> =
        emoji.spans.iter().map(|(_, s)| &src[doc.to_source(emoji.spans.view(), s).range()]).collect();
    assert_eq!(texts, [flag("DE"), cp(&[0x2764, 0xFE0F]), cp(&[0x1F600])]);
    assert_eq!(emoji.region.get(0), "DE");
    assert_eq!(emoji.name.get(1), "red heart");
    let view = doc.get_key(&Key::<View>::named("no-emoji")).unwrap();
    assert_eq!(view.text(), "Berlin   is  !");
    let placeholders = doc.get_key(&Key::<Placeholders>::named("emoji")).unwrap();
    let removed: Vec<&str> = placeholders.spans.iter().map(|(_, s)| &src[s.range()]).collect();
    assert_eq!(removed, texts);
    // Every emoji of the layer maps forward into the space it became.
    for (_, s) in emoji.spans.iter() {
        let child = view.alignment().span_to_child(s);
        assert!(view.text()[..child.end as usize].ends_with(' '), "{s:?}");
    }
}
