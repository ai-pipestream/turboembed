//! The emoji normalizers and annotator: the expectations of OpenNLP's
//! EmojiToEmoticonCharSequenceNormalizerTest,
//! EmoticonToEmojiCharSequenceNormalizerTest, EmojiEmoticonsTest,
//! EmojiAnnotationsTest, EmojiAnnotatorTest, EmojiFlagsTest and
//! EmojiAnnotatorUsageExampleTest; every fully-qualified row of the
//! Unicode 18.0 emoji-test.txt through `EmojiRemove`; and the alignment of
//! every edit.

mod scanners_support;

use std::collections::HashSet;

use scanners_support::{Rng, check_alignment, run};
use turbo_analysis::{Analyzer, Annotator, Bound, Document, Key, Layer, Refs, Span, View};
use turbo_normalize::emoji::data::{
    ANNOTATIONS, EMOJI_TEST_SOURCE, EMOTICON_FOLDS, Fold, NAME_NOTE, NAME_SOURCE, SENTIMENT_SOURCE,
};
use turbo_normalize::emoji::{EmojiEntityType, category_constant, features, is_emoji_presentation, opennlp_annotation};
use turbo_normalize::{
    Chain, EmojiAnnotate, EmojiRemove, EmojiToEmoticon, EmoticonToEmoji, Normalizer, Whitespace, apply, normalize,
};
use turbo_normalize_layers::{Emoji, EmojiKind, PlaceholderKind, Placeholders};
use turbo_unicode::emoji::{EmojiGroup, EmojiKindHint, scan};

/// The string the code points spell.
fn cp(points: &[u32]) -> String {
    points.iter().map(|&p| char::from_u32(p).expect("a scalar value")).collect()
}

/// U+1F3F4, the tag characters spelling `ascii`, and U+E007F.
fn tag_flag(ascii: &str) -> String {
    let mut s = cp(&[0x1F3F4]);
    s.extend(ascii.chars().map(|c| char::from_u32(0xE0000 + c as u32).unwrap()));
    s.push('\u{E007F}');
    s
}

/// Regional indicators spelling `letters`.
fn flag(letters: &str) -> String {
    letters.chars().map(|c| char::from_u32(0x1F1E6 + (c as u32 - 'A' as u32)).unwrap()).collect()
}

const EMOJI_TEST: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../unicode/ucd/emoji-test.txt"));

/// The code points of every row of emoji-test.txt with `status`.
fn emoji_test_rows(status: &str) -> Vec<String> {
    let mut rows = Vec::new();
    for line in EMOJI_TEST.lines() {
        let Some((points, rest)) = line.split_once(';') else { continue };
        if line.starts_with('#') || rest.split('#').next().unwrap().trim() != status {
            continue;
        }
        let points: Vec<u32> = points.split_whitespace().map(|h| u32::from_str_radix(h, 16).unwrap()).collect();
        rows.push(cp(&points));
    }
    rows
}

// ---------------------------------------------------------------------------
// EmojiToEmoticon (EmojiToEmoticonCharSequenceNormalizerTest)
// ---------------------------------------------------------------------------

fn to_emoticon(s: &str) -> String {
    run(&EmojiToEmoticon, s)
}

#[test]
fn folds_supplementary_emoji_to_emoticon() {
    assert_eq!(to_emoticon(&cp(&[0x1F642])), ":)");
    assert_eq!(to_emoticon(&cp(&[0x1F600])), ":D");
    assert_eq!(to_emoticon(&cp(&[0x1F622])), ":'(");
}

#[test]
fn folds_bmp_pictograph_to_emoticon() {
    assert_eq!(to_emoticon(&cp(&[0x263A])), ":)");
    assert_eq!(to_emoticon(&cp(&[0x2764])), "<3");
}

#[test]
fn emoji_presentation_sequence_folds_as_one_unit() {
    assert_eq!(to_emoticon(&cp(&[0x2764, 0xFE0F])), "<3");
    assert_eq!(to_emoticon(&cp(&[0x263A, 0xFE0F])), ":)");
}

#[test]
fn many_to_one_variants_converge_on_the_canonical_emoticon() {
    assert_eq!(to_emoticon(&cp(&[0x1F603, 0x1F604])), ":D:D");
    assert_eq!(to_emoticon(&cp(&[0x1F499, 0x1F9E1])), "<3<3");
}

#[test]
fn unmapped_content_passes_through() {
    let rocket = cp(&[0x1F680]);
    let s = format!("go {rocket} now");
    assert_eq!(to_emoticon(&s), s);
    assert_eq!(to_emoticon("plain text"), "plain text");
    assert_eq!(to_emoticon(""), "");
    assert!(matches!(apply(&EmojiToEmoticon, "plain text"), std::borrow::Cow::Borrowed(_)));
}

#[test]
fn folds_inside_running_text_without_boundaries() {
    assert_eq!(to_emoticon(&format!("great{} news", cp(&[0x1F600]))), "great:D news");
}

#[test]
fn alignment_maps_expansion_back_to_the_pictograph() {
    // "x :'( y": the emoticon (output 2..5) maps back to the four bytes of
    // U+1F622 (OpenNLP: two UTF-16 units).
    let (out, a) = normalize(&EmojiToEmoticon, &format!("x {} y", cp(&[0x1F622])));
    assert_eq!(out, "x :'( y");
    assert_eq!(a.span_to_parent(Span::new(2, 5)), Span::new(2, 6));
}

#[test]
fn adjacent_folds_each_map_to_their_own_source() {
    let (out, a) = normalize(&EmojiToEmoticon, &cp(&[0x1F642, 0x1F600]));
    assert_eq!(out, ":):D");
    // Each fold is its own replacement. While the view builder merges
    // touching edits, the two folds map back to both pictographs together.
    assert_exact_or_merged(a.span_to_parent(Span::new(0, 2)), Span::new(0, 4), Span::new(0, 8));
    assert_exact_or_merged(a.span_to_parent(Span::new(2, 4)), Span::new(4, 8), Span::new(0, 8));
}

#[test]
fn emoji_to_emoticon_composes_into_a_chain() {
    // Collapse the double space, then fold the pictograph: ":)" maps back
    // through both stages to the pictograph. `compose` merges stretches
    // that touch, so the collapsed space and the fold, which touch, come
    // back as one stretch.
    let chain = Chain::new().then(Whitespace::collapse()).then(EmojiToEmoticon);
    let src = format!("hi  {}", cp(&[0x1F642]));
    let (out, a) = normalize(&chain, &src);
    check_alignment(&src, &out, &a);
    assert_eq!(out, "hi :)");
    assert_exact_or_merged(a.span_to_parent(Span::new(3, 5)), Span::new(4, 8), Span::new(2, 8));
}

#[test]
fn zwj_sequence_is_not_corrupted() {
    let heart_on_fire = cp(&[0x2764, 0xFE0F, 0x200D, 0x1F525]);
    assert_eq!(to_emoticon(&heart_on_fire), heart_on_fire);
    let couple = cp(&[0x1F469, 0x200D, 0x2764, 0xFE0F, 0x200D, 0x1F48B, 0x200D, 0x1F468]);
    assert_eq!(to_emoticon(&couple), couple);
}

#[test]
fn zwj_sequence_next_to_a_foldable_emoji_only_folds_the_standalone_one() {
    let heart_on_fire = cp(&[0x2764, 0xFE0F, 0x200D, 0x1F525]);
    let input = format!("{} {heart_on_fire}", cp(&[0x1F642]));
    assert_eq!(to_emoticon(&input), format!(":) {heart_on_fire}"));
}

#[test]
fn text_presentation_selector_suppresses_the_fold() {
    let text_smiley = cp(&[0x263A, 0xFE0E]);
    assert_eq!(to_emoticon(&text_smiley), text_smiley);
}

#[test]
fn trailing_emoji_presentation_selector_is_absorbed_for_every_mapped_pictograph() {
    assert_eq!(to_emoticon(&cp(&[0x1F642, 0xFE0F])), ":)");
}

#[test]
fn absorbed_selector_maps_back_to_the_whole_sequence() {
    let (out, a) = normalize(&EmojiToEmoticon, &format!("{}!", cp(&[0x1F642, 0xFE0F])));
    assert_eq!(out, ":)!");
    // U+1F642 is four bytes and U+FE0F three (OpenNLP: three UTF-16 units).
    assert_eq!(a.span_to_parent(Span::new(0, 2)), Span::new(0, 7));
}

/// Each output emoticon maps back to the source it replaced, or, while the
/// view builder merges touching edits, to the run of adjacent folds.
#[test]
fn each_fold_maps_back_to_its_source() {
    let src = format!("{}{}{}", cp(&[0x1F642]), cp(&[0x2764, 0xFE0F]), cp(&[0x1F622]));
    let (out, a) = normalize(&EmojiToEmoticon, &src);
    assert_eq!(out, ":)<3:'(");
    let whole = Span::new(0, src.len() as u32);
    assert_exact_or_merged(a.span_to_parent(Span::new(0, 2)), Span::new(0, 4), whole);
    assert_exact_or_merged(a.span_to_parent(Span::new(2, 4)), Span::new(4, 10), whole);
    assert_exact_or_merged(a.span_to_parent(Span::new(4, 7)), Span::new(10, 14), whole);
    // Separated by text, each fold is an edit of its own either way.
    let (_, a) = normalize(&EmojiToEmoticon, &format!("{} {}", cp(&[0x1F642]), cp(&[0x1F622])));
    assert_eq!(a.span_to_parent(Span::new(3, 6)), Span::new(5, 9));
}

// ---------------------------------------------------------------------------
// EmoticonToEmoji (EmoticonToEmojiCharSequenceNormalizerTest)
// ---------------------------------------------------------------------------

fn to_emoji(s: &str) -> String {
    run(&EmoticonToEmoji, s)
}

#[test]
fn folds_delimited_emoticons() {
    assert_eq!(to_emoji(":)"), cp(&[0x1F642]));
    assert_eq!(to_emoji("a :( b"), format!("a {} b", cp(&[0x1F641])));
    assert_eq!(to_emoji("love <3"), format!("love {}", cp(&[0x2764, 0xFE0F])));
}

#[test]
fn longest_match_wins_at_a_position() {
    assert_eq!(to_emoji("hi :-)"), format!("hi {}", cp(&[0x1F642])));
    assert_eq!(to_emoji("</3"), cp(&[0x1F494]));
}

#[test]
fn embedded_sequences_are_left_alone() {
    assert_eq!(to_emoji("see https://example.org now"), "see https://example.org now");
    assert_eq!(to_emoji("nice:)"), "nice:)");
    assert_eq!(to_emoji("a ratio of 1:3 today"), "a ratio of 1:3 today");
}

#[test]
fn trailing_punctuation_blocks_the_fold_by_design() {
    assert_eq!(to_emoji("fun :)."), "fun :).");
    assert_eq!(to_emoji(":):("), ":):(");
}

#[test]
fn unicode_white_space_delimits() {
    // White_Space, not Java's isWhitespace: the no-break space delimits.
    assert_eq!(to_emoji("a\u{A0}:)"), format!("a\u{A0}{}", cp(&[0x1F642])));
    assert_eq!(to_emoji(":)\u{3000}x"), format!("{}\u{3000}x", cp(&[0x1F642])));
    // U+200B is not White_Space.
    assert_eq!(to_emoji("a\u{200B}:)"), "a\u{200B}:)");
}

#[test]
fn alignment_maps_contraction_back_to_the_emoticon() {
    let (out, a) = normalize(&EmoticonToEmoji, "a :-) b");
    assert_eq!(out, format!("a {} b", cp(&[0x1F642])));
    assert_eq!(a.span_to_parent(Span::new(2, 6)), Span::new(2, 5));
}

#[test]
fn emoticon_to_emoji_composes_into_a_chain() {
    let chain = Chain::new().then(Whitespace::collapse()).then(EmoticonToEmoji);
    let (out, a) = normalize(&chain, "ok   :-)");
    check_alignment("ok   :-)", &out, &a);
    assert_eq!(out, format!("ok {}", cp(&[0x1F642])));
    // As above, the whitespace edit and the fold touch and compose merges them.
    assert_exact_or_merged(a.span_to_parent(Span::new(3, 7)), Span::new(5, 8), Span::new(2, 8));
}

#[test]
fn round_trip_through_both_directions_converges_on_canonical_forms() {
    let folded = to_emoji("well :-) then");
    assert_eq!(to_emoticon(&folded), "well :) then");
}

#[test]
fn adjacent_emoticons_separated_by_spaces_each_fold() {
    let out = to_emoji(":) :-D <3 ;-)");
    assert_eq!(out, cp(&[0x1F642, 0x20, 0x1F600, 0x20, 0x2764, 0xFE0F, 0x20, 0x1F609]));
}

// ---------------------------------------------------------------------------
// The table (EmojiEmoticonsTest)
// ---------------------------------------------------------------------------

fn sources(fold: Fold) -> Vec<&'static str> {
    EMOTICON_FOLDS.iter().filter(|r| r.fold == fold).map(|r| r.source).collect()
}

#[test]
fn directions_close_over_each_other() {
    let emoticon_sources = sources(Fold::Emoticon);
    let emoji_sources = sources(Fold::Emoji);
    for row in EMOTICON_FOLDS {
        let other = if row.fold == Fold::Emoji { &emoticon_sources } else { &emoji_sources };
        assert!(other.contains(&row.target), "{row:?} has no row in the other direction");
    }
}

#[test]
fn emoticon_sources_are_ascii_and_emoji_sources_are_not() {
    for s in sources(Fold::Emoticon) {
        assert!(s.bytes().all(|b| (0x21..0x7F).contains(&b)), "{s:?}");
    }
    for s in sources(Fold::Emoji) {
        assert!(!s.starts_with(|c: char| c.is_ascii()), "{s:?}");
    }
}

#[test]
fn bundled_row_counts_are_audited() {
    for (fold, count) in [(Fold::Emoji, 35), (Fold::Emoticon, 26)] {
        let all = sources(fold);
        let distinct: HashSet<&str> = all.iter().copied().collect();
        assert_eq!(all.len(), count);
        assert_eq!(distinct.len(), count, "no source twice in one direction");
    }
    for row in EMOTICON_FOLDS {
        assert_eq!((row.standard, row.unicode_version), ("UNSPECIFIED", "17.0.0"));
        assert!(!row.note.is_empty() && !row.target.is_empty());
    }
}

/// Each source on its own folds to its own target: the longest match wins
/// (`❤️` is not `<3` and a stray U+FE0F), in both directions.
#[test]
fn every_row_folds_its_own_source() {
    for row in EMOTICON_FOLDS {
        let out = match row.fold {
            Fold::Emoji => to_emoticon(row.source),
            Fold::Emoticon => to_emoji(row.source),
        };
        assert_eq!(out, row.target, "{row:?}");
    }
}

/// Longest first by UTF-16 units (OpenNLP's measure) and by UTF-8 bytes
/// give the same order for every pair of sources that share a first
/// character, so the port's choice of measure changes nothing.
#[test]
fn candidate_order_does_not_depend_on_the_encoding() {
    for a in EMOTICON_FOLDS {
        for b in EMOTICON_FOLDS {
            if a.fold == b.fold && a.source.chars().next() == b.source.chars().next() {
                let utf16 = a.source.encode_utf16().count().cmp(&b.source.encode_utf16().count());
                assert_eq!(utf16, a.source.len().cmp(&b.source.len()), "{a:?} {b:?}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The annotations (EmojiAnnotationsTest)
// ---------------------------------------------------------------------------

#[test]
fn bundled_record_counts_are_audited() {
    assert_eq!(ANNOTATIONS.len(), 40);
    let distinct: HashSet<&str> = ANNOTATIONS.iter().map(|a| a.symbol).collect();
    assert_eq!(distinct.len(), 40);
}

#[test]
fn bundled_values_are_within_their_domains() {
    for a in ANNOTATIONS {
        assert!((-2..=2).contains(&a.sentiment), "{a:?}");
        assert!(a.name.bytes().all(|b| (0x20..0x7F).contains(&b)), "{a:?}");
        assert!(!a.symbol.starts_with(|c: char| c.is_ascii()), "{a:?}");
        assert!(!a.symbol.contains('\u{FE0F}'), "{a:?}");
        assert!(a.sentiment_note.starts_with("project judgment"), "{a:?}");
        assert_ne!(a.entity_type, EmojiEntityType::Flag, "flags are derived, never bundled");
    }
    assert_eq!(
        (NAME_SOURCE, NAME_NOTE, SENTIMENT_SOURCE, EMOJI_TEST_SOURCE),
        ("CLDR:annotation", "emoji-test.txt v17.0 short name", "UNSPECIFIED", "UCD:emoji-test")
    );
}

#[test]
fn every_emoji_fold_source_is_annotated() {
    for s in sources(Fold::Emoji) {
        let key = s.replace('\u{FE0F}', "");
        assert!(ANNOTATIONS.iter().any(|a| a.symbol == key), "{s:?}");
    }
}

/// The names, groups and subgroups the file took from emoji-test.txt 17.0
/// are still those of the Unicode 18.0 emoji-test.txt.
#[test]
fn bundled_facts_match_the_unicode_data() {
    for a in ANNOTATIONS {
        assert_eq!(turbo_unicode::emoji::name(a.symbol), Some(a.name), "{a:?}");
        assert_eq!(turbo_unicode::emoji::group(a.symbol), Some(a.category), "{a:?}");
        assert_eq!(turbo_unicode::emoji::subgroup(a.symbol), Some(a.subgroup), "{a:?}");
    }
}

#[test]
fn lookup_resolves_a_typed_record() {
    let a = opennlp_annotation(&cp(&[0x1F642])).unwrap();
    assert_eq!(a.symbol, cp(&[0x1F642]));
    assert_eq!(a.name, "slightly smiling face");
    assert_eq!(a.sentiment, 1);
    assert_eq!(a.entity_type, EmojiEntityType::Face);
    assert_eq!(a.category, EmojiGroup::SmileysAndEmotion);
}

#[test]
fn lookup_strips_the_presentation_selector() {
    let with = opennlp_annotation(&cp(&[0x2764, 0xFE0F])).unwrap();
    let bare = opennlp_annotation(&cp(&[0x2764])).unwrap();
    assert_eq!(with, bare);
    assert_eq!(with.name, "red heart");
    // Every U+FE0F goes, wherever it is, as stripPresentationSelector does.
    assert_eq!(opennlp_annotation(&cp(&[0xFE0F, 0x2764, 0xFE0F, 0xFE0F])), Some(bare));
}

#[test]
fn lookup_of_unannotated_text_is_empty() {
    for s in ["word", ":-)", "", "\u{FE0F}", "\u{FE0F}\u{FE0F}", &flag("DE"), &cp(&[0x1F642, 0x1F642])] {
        assert_eq!(opennlp_annotation(s), None, "{s:?}");
    }
}

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
// EmojiRemove
// ---------------------------------------------------------------------------

/// Runs `EmojiRemove` as an annotator, checks the view's alignment, the
/// layer's invariants and where each placeholder landed, and returns the
/// output and the placeholders' source texts.
///
/// A placeholder lands on its space, or, for a sequence that directly
/// follows another one, on nothing right after the run's space (with
/// edits that merge it lands on the space too).
fn removed(n: EmojiRemove, src: &str) -> (String, Vec<String>) {
    let doc = Document::new(src);
    let view = doc.get::<View>().unwrap();
    let input: Refs<'_, EmojiRemove> = (view,);
    let (out_view, found) = n.annotate(input).unwrap();
    found.check().unwrap();
    let out = out_view.text().to_string();
    let a = out_view.alignment();
    check_alignment(src, &out, a);
    assert_eq!(apply(&n, src), out, "the normalizer and the annotator agree");
    let mut previous_end = None;
    let mut rows = Vec::new();
    for (ix, s) in found.spans.iter() {
        assert_eq!(found.kind[ix.get()], PlaceholderKind::Emoji);
        let child = a.span_to_child(s);
        let landed = &out[child.range()];
        let follows = previous_end == Some(s.start);
        assert!(
            landed == " " || (follows && landed.is_empty() && out[..child.start as usize].ends_with(' ')),
            "{:?} landed on {landed:?} in {out:?}",
            &src[s.range()]
        );
        previous_end = Some(s.end);
        rows.push(src[s.range()].to_string());
    }
    (out, rows)
}

fn sequences(src: &str) -> (String, Vec<String>) {
    removed(EmojiRemove::sequences(), src)
}

#[test]
fn every_fully_qualified_emoji_becomes_one_space() {
    let rows = emoji_test_rows("fully-qualified");
    assert!(rows.len() > 3500, "{} rows", rows.len());
    for row in &rows {
        assert_eq!(sequences(row), (" ".to_string(), vec![row.clone()]), "{row:?}");
        let (out, found) = sequences(&format!("a{row}b"));
        assert_eq!((out.as_str(), found.as_slice()), ("a b", std::slice::from_ref(row)), "{row:?}");
    }
}

#[test]
fn components_and_qualified_variants_are_whole_sequences() {
    for status in ["component", "minimally-qualified"] {
        for row in emoji_test_rows(status) {
            assert_eq!(sequences(&row), (" ".to_string(), vec![row.clone()]), "{status} {row:?}");
        }
    }
}

#[test]
fn zwj_families_skin_tones_keycaps_and_flags_are_one_space_each() {
    let family = cp(&[0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467, 0x200D, 0x1F466]);
    let toned = cp(&[0x1F44D, 0x1F3FD]);
    let toned_couple = cp(&[0x1F9D1, 0x1F3FB, 0x200D, 0x2764, 0xFE0F, 0x200D, 0x1F9D1, 0x1F3FF]);
    let keycap = cp(&[0x31, 0xFE0F, 0x20E3]);
    let keycap_bare = cp(&[0x23, 0x20E3]);
    let england = tag_flag("gbeng");
    for seq in [&family, &toned, &toned_couple, &keycap, &keycap_bare, &flag("FR"), &england] {
        let (out, found) = sequences(&format!("x {seq} y"));
        assert_eq!(out, "x   y", "{seq:?}");
        assert_eq!(found, std::slice::from_ref(seq));
    }
    // Digits, `#` and `*` without U+20E3 are text.
    assert_eq!(sequences("#1 *2"), ("#1 *2".to_string(), vec![]));
}

#[test]
fn adjacent_sequences_become_one_space_with_a_placeholder_each() {
    let a = cp(&[0x1F600]);
    let b = flag("DE");
    let c = cp(&[0x1F44D, 0x1F3FD]);
    let (out, found) = sequences(&format!("hi {a}{b}{c}!"));
    assert_eq!(out, "hi  !");
    assert_eq!(found, [a.clone(), b.clone(), c.clone()]);
    // Three regional indicators: a flag and a lone indicator, adjacent.
    let (out, found) = sequences(&flag("DEF"));
    assert_eq!(out, " ");
    assert_eq!(found, [flag("DE"), flag("F")]);
    // Text between keeps them apart.
    assert_eq!(sequences(&format!("{a}-{a}")).0, " - ");
}

#[test]
fn each_adjacent_sequence_maps_back_to_itself() {
    let (a, b) = (cp(&[0x1F600]), cp(&[0x1F680]));
    let src = format!("{a}{b}");
    let (out, al) = normalize(&EmojiRemove::sequences(), &src);
    assert_eq!(out, " ");
    // The run maps forward to the space. Backward, the space maps to the
    // first sequence, or to the whole run when the builder merges the two
    // touching edits.
    assert_eq!(al.span_to_child(Span::new(0, 8)), Span::new(0, 1));
    assert!([Span::new(0, 4), Span::new(0, 8)].contains(&al.span_to_parent(Span::new(0, 1))));
    assert!([Span::new(0, 1), Span::new(1, 1)].contains(&al.span_to_child(Span::new(4, 8))));
}

#[test]
fn text_default_symbols_with_and_without_the_selector() {
    // © and ™ alone stay; with U+FE0F they are emoji.
    let s = "\u{A9} 2026 Acme\u{2122}, \u{2764} NY, A \u{2194} B, \u{2639}";
    assert_eq!(sequences(s), (s.to_string(), vec![]));
    let (out, found) = sequences("\u{A9}\u{FE0F} \u{2764}\u{FE0F}!");
    assert_eq!(out, "   !");
    assert_eq!(found, ["\u{A9}\u{FE0F}", "\u{2764}\u{FE0F}"]);
    // U+FE0E asks for text: the character stays even when it is emoji by
    // default.
    let watch = "\u{231A}\u{FE0E}";
    assert_eq!(sequences(watch), (watch.to_string(), vec![]));
    assert_eq!(sequences("\u{231A}").0, " ");
    // A modifier sequence on a text-default base is an emoji.
    assert_eq!(sequences("\u{261D}\u{1F3FB}").0, " ");
}

#[test]
fn stray_joiners_and_selectors_stay() {
    let (out, _) = sequences("a\u{200D}\u{1F600}\u{200D}b");
    assert_eq!(out, "a\u{200D} \u{200D}b");
    let (out, _) = sequences("\u{1F600}\u{FE0F}\u{FE0F}");
    assert_eq!(out, " \u{FE0F}");
}

#[test]
fn removing_sequences_is_idempotent_and_finds_scanner_sequences() {
    let pool = [
        "a",
        " ",
        "\u{A9}",
        "\u{2122}",
        "\u{FE0F}",
        "\u{FE0E}",
        "\u{200D}",
        "\u{20E3}",
        "1",
        "#",
        "\u{1F600}",
        "\u{2764}",
        "\u{1F3FD}",
        "\u{1F1E9}",
        "\u{1F1EA}",
        "\u{1F3F4}",
        "\u{E0067}",
        "\u{E007F}",
        "\u{1F468}",
        "\u{1D400}",
        "\u{E9}",
        "\u{301}",
    ];
    let mut rng = Rng::new(0x05EE_DE10);
    for _ in 0..20_000 {
        let src = rng.input(&pool);
        let (out, found) = sequences(&src);
        assert_eq!(apply(&EmojiRemove::sequences(), &out), out, "{src:?}");
        let expected: Vec<String> = scan(&src)
            .filter(|(r, k)| is_emoji_presentation(&src[r.clone()], *k))
            .map(|(r, _)| src[r].to_string())
            .collect();
        assert_eq!(found, expected, "{src:?}");
    }
}

/// OpenNLP's `EmojiCharSequenceNormalizer` outside its legacy mode, one
/// code point at a time.
fn supplementary_reference(s: &str) -> String {
    let mut out = String::new();
    let mut in_run = false;
    for c in s.chars() {
        if u32::from(c) > 0xFFFF {
            if !in_run {
                out.push(' ');
            }
            in_run = true;
        } else {
            out.push(c);
            in_run = false;
        }
    }
    out
}

#[test]
fn supplementary_mode_matches_opennlp() {
    let n = EmojiRemove::supplementary();
    for (src, expected) in [
        ("plain", "plain"),
        ("\u{1F600}", " "),
        ("a\u{1F600}\u{1F680}b", "a b"),
        ("\u{263A}\u{2764}\u{FE0F}", "\u{263A}\u{2764}\u{FE0F}"),
        ("x\u{1D400}y\u{20000}", "x y "),
        ("\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}", " \u{200D} \u{200D} "),
        ("\u{1F44D}\u{1F3FD}", " "),
        ("\u{A9} - \u{1F1E9}\u{1F1EA}", "\u{A9} -  "),
    ] {
        let (out, _) = removed(n, src);
        assert_eq!(out, expected, "{src:?}");
        assert_eq!(out, supplementary_reference(src));
    }
    let pool = ["a", " ", "-", "\u{263A}", "\u{FE0F}", "\u{200D}", "\u{1F600}", "\u{1D400}", "\u{E9}", "\u{10FFFF}"];
    let mut rng = Rng::new(17);
    for _ in 0..20_000 {
        let src = rng.input(&pool);
        let (out, found) = removed(n, &src);
        assert_eq!(out, supplementary_reference(&src), "{src:?}");
        assert!(found.iter().all(|f| f.chars().all(|c| u32::from(c) > 0xFFFF)));
    }
}

#[test]
fn supplementary_mode_records_one_placeholder_per_run() {
    let (out, found) = removed(EmojiRemove::supplementary(), "a\u{1F600}\u{1F680} b\u{1D400}");
    assert_eq!(out, "a  b ");
    assert_eq!(found, ["\u{1F600}\u{1F680}", "\u{1D400}"]);
}

#[test]
fn modes_have_distinct_configurations() {
    assert_eq!(EmojiRemove::default(), EmojiRemove::sequences());
    assert_ne!(
        <EmojiRemove as Normalizer>::config_hash(&EmojiRemove::sequences()),
        <EmojiRemove as Normalizer>::config_hash(&EmojiRemove::supplementary())
    );
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

/// `got` is the exact span OpenNLP's alignment gives, or the wider span
/// the alignment gives when touching edits are merged (`ViewBuilder`
/// merges touching replacements, and `compose` merges touching stretches).
fn assert_exact_or_merged(got: Span, exact: Span, merged: Span) {
    assert!(got == exact || got == merged, "{got:?} is neither {exact:?} nor {merged:?}");
}
