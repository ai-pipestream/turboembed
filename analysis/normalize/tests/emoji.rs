//! The emoji normalizers and annotator: the expectations of OpenNLP's
//! EmojiToEmoticonCharSequenceNormalizerTest,
//! EmoticonToEmojiCharSequenceNormalizerTest, EmojiEmoticonsTest,
//! EmojiAnnotationsTest, EmojiAnnotatorTest, EmojiFlagsTest and
//! EmojiAnnotatorUsageExampleTest; every fully-qualified row of the
//! Unicode 18.0 emoji-test.txt through `EmojiRemove`; and the alignment of
//! every edit.

#[path = "emoji/annotator.rs"]
mod annotator;
#[path = "emoji/remove.rs"]
mod remove;
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

/// `got` is the exact span OpenNLP's alignment gives, or the wider span
/// the alignment gives when touching edits are merged (`ViewBuilder`
/// merges touching replacements, and `compose` merges touching stretches).
fn assert_exact_or_merged(got: Span, exact: Span, merged: Span) {
    assert!(got == exact || got == merged, "{got:?} is neither {exact:?} nor {merged:?}");
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
