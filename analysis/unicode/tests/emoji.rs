//! Emoji sequences against emoji-test.txt and the scanning rules of the
//! `emoji` module documentation.

use turbo_unicode::emoji::{EmojiGroup, EmojiKindHint, group, is_rgi, name, region, scan, subgroup};
use turbo_unicode::props;

mod common;

struct Row {
    text: String,
    status: String,
    name: String,
    group: String,
    subgroup: String,
}

fn rows() -> Vec<Row> {
    let mut rows = Vec::new();
    let (mut group, mut subgroup) = (String::new(), String::new());
    for line in common::read("emoji-test.txt").lines() {
        if let Some(g) = line.strip_prefix("# group: ") {
            group = g.to_owned();
        } else if let Some(s) = line.strip_prefix("# subgroup: ") {
            subgroup = s.to_owned();
        } else if !line.starts_with('#') && !line.trim().is_empty() {
            let (data, comment) = line.split_once('#').expect("a comment");
            let (cps, status) = data.split_once(';').expect("a status");
            let text = common::string(cps);
            // The comment is `<emoji> E<version> <name>`.
            let after_emoji = comment.trim_start().strip_prefix(text.as_str()).expect("the emoji").trim_start();
            let name = after_emoji.split_once(' ').expect("a version and a name").1.to_owned();
            rows.push(Row {
                text,
                status: status.trim().to_owned(),
                name,
                group: group.clone(),
                subgroup: subgroup.clone(),
            });
        }
    }
    rows
}

fn found(text: &str) -> Vec<(std::ops::Range<usize>, EmojiKindHint)> {
    scan(text).collect()
}

#[test]
fn every_row_of_emoji_test_is_one_sequence() {
    let rows = rows();
    assert!(rows.len() > 5000, "the whole file was read");
    for row in &rows {
        let sequences = found(&row.text);
        assert_eq!(sequences.len(), 1, "{} ({}) is one sequence: {sequences:?}", row.name, row.status);
        assert_eq!(sequences[0].0, 0..row.text.len(), "{} ({}) is covered exactly", row.name, row.status);
    }
}

#[test]
fn qualified_rows_are_rgi_and_others_are_not() {
    for row in rows() {
        let rgi = matches!(row.status.as_str(), "fully-qualified" | "component");
        assert_eq!(is_rgi(&row.text), rgi, "{} ({})", row.name, row.status);
    }
}

#[test]
fn every_row_has_its_name_group_and_subgroup() {
    for row in rows() {
        // Minimally-qualified and unqualified rows differ from their
        // fully-qualified row only by U+FE0F, which the lookup ignores.
        assert_eq!(name(&row.text), Some(row.name.as_str()), "{:?} ({})", row.text, row.status);
        assert_eq!(group(&row.text).map(EmojiGroup::name), Some(row.group.as_str()), "{}", row.name);
        assert_eq!(subgroup(&row.text), Some(row.subgroup.as_str()), "{}", row.name);
    }
    assert_eq!(name("\u{1F600}"), Some("grinning face"));
    assert_eq!(group("\u{1F600}"), Some(EmojiGroup::SmileysAndEmotion));
    assert_eq!(subgroup("\u{1F600}"), Some("face-smiling"));
    assert_eq!(name("\u{2639}"), name("\u{2639}\u{FE0F}"));
    assert_eq!(name("a"), None);
    assert_eq!(name(""), None);
}

#[test]
fn rows_have_the_kind_of_their_shape() {
    for row in rows() {
        let kind = found(&row.text)[0].1;
        let expected = if row.text.contains('\u{200D}') {
            EmojiKindHint::Zwj
        } else if row.subgroup == "country-flag" {
            EmojiKindHint::Flag
        } else if row.subgroup == "subdivision-flag" {
            EmojiKindHint::Tag
        } else if row.text.contains('\u{20E3}') {
            EmojiKindHint::Keycap
        } else if row.text.chars().skip(1).any(props::emoji_modifier) {
            EmojiKindHint::Modifier
        } else {
            EmojiKindHint::Basic
        };
        assert_eq!(kind, expected, "{} ({})", row.name, row.status);
    }
}

#[test]
fn every_rgi_sequence_of_the_data_files_is_found_and_rgi() {
    for file in ["emoji-sequences.txt", "emoji-zwj-sequences.txt"] {
        for fields in common::records(&common::read(file)) {
            let texts: Vec<String> = if fields[0].contains("..") {
                common::range(&fields[0]).map(|cp| char::from_u32(cp).unwrap().to_string()).collect()
            } else {
                vec![common::string(&fields[0])]
            };
            for text in texts {
                assert!(is_rgi(&text), "{text:?} of {file}");
                assert_eq!(found(&text), vec![(0..text.len(), found(&text)[0].1)], "{text:?} of {file}");
            }
        }
    }
}

#[test]
fn text_default_emoji_are_reported_and_text_presentation_is_not() {
    for text in ["\u{A9}", "\u{2122}", "\u{2639}", "\u{263A}"] {
        assert_eq!(found(text), vec![(0..text.len(), EmojiKindHint::Basic)], "{text:?}");
    }
    assert_eq!(found("\u{2639}\u{FE0F}"), vec![(0..6, EmojiKindHint::Basic)]);
    assert_eq!(found("\u{2639}\u{FE0E}"), vec![], "text presentation selector");
    assert_eq!(found("\u{1F600}\u{FE0E}"), vec![]);
    assert_eq!(found("(c) 2026 \u{A9} Acme"), vec![(9..11, EmojiKindHint::Basic)]);
}

#[test]
fn extended_pictographic_without_the_emoji_property_needs_the_selector() {
    let c = common::all_chars()
        .find(|&c| props::extended_pictographic(c) && !props::emoji(c))
        .expect("such characters exist");
    let alone = c.to_string();
    assert_eq!(found(&alone), vec![], "U+{:04X} alone", c as u32);
    let selected = format!("{c}\u{FE0F}");
    assert_eq!(found(&selected), vec![(0..selected.len(), EmojiKindHint::Basic)]);
}

#[test]
fn keycaps_need_the_combining_keycap() {
    assert_eq!(found("1"), vec![]);
    assert_eq!(found("#1*"), vec![]);
    assert_eq!(found("1\u{FE0F}"), vec![]);
    assert_eq!(found("1\u{20E3}"), vec![(0..4, EmojiKindHint::Keycap)]);
    assert_eq!(found("x#\u{FE0F}\u{20E3}y"), vec![(1..8, EmojiKindHint::Keycap)]);
    assert_eq!(found("\u{20E3}"), vec![], "a lone combining keycap");
}

#[test]
fn flags_pair_regional_indicators_from_the_left() {
    let ri = |letter: char| char::from_u32(0x1F1E6 + (letter as u32 - 'A' as u32)).unwrap();
    let three: String = [ri('D'), ri('E'), ri('F')].iter().collect();
    assert_eq!(found(&three), vec![(0..8, EmojiKindHint::Flag), (8..12, EmojiKindHint::Basic)]);
    let four: String = [ri('D'), ri('E'), ri('F'), ri('R')].iter().collect();
    assert_eq!(found(&four), vec![(0..8, EmojiKindHint::Flag), (8..16, EmojiKindHint::Flag)]);
}

#[test]
fn joiners_and_tags_need_what_follows_them() {
    // A joiner with no element after it is not part of the sequence.
    assert_eq!(found("\u{1F468}\u{200D}"), vec![(0..4, EmojiKindHint::Basic)]);
    assert_eq!(found("\u{1F468}\u{200D}x"), vec![(0..4, EmojiKindHint::Basic)]);
    let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
    assert_eq!(found(family), vec![(0..family.len(), EmojiKindHint::Zwj)]);
    // A tag run without CANCEL TAG is not part of the element.
    let open = "\u{1F3F4}\u{E0067}\u{E0062}";
    assert_eq!(found(open), vec![(0..4, EmojiKindHint::Basic)]);
    let england = "\u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F}";
    assert_eq!(found(england), vec![(0..england.len(), EmojiKindHint::Tag)]);
    let thumbs = "\u{1F44D}\u{1F3FD}";
    assert_eq!(found(thumbs), vec![(0..8, EmojiKindHint::Modifier)]);
}

#[test]
fn sequences_are_found_inside_text_at_byte_offsets() {
    let text = "ok \u{1F44D} then \u{1F1E9}\u{1F1EA}, done";
    assert_eq!(found(text), vec![(3..7, EmojiKindHint::Basic), (13..21, EmojiKindHint::Flag)]);
    assert_eq!(found("plain ascii text"), vec![]);
    assert_eq!(found(""), vec![]);
}

// Region decoding, ported from OpenNLP's EmojiFlagsTest. The JDK method
// throws on malformed flags; `region` returns None, like EmojiFlags.isFlag.

fn ri(letters: &str) -> String {
    letters.chars().map(|l| char::from_u32(0x1F1E6 + (l as u32 - 'A' as u32)).unwrap()).collect()
}

fn tags(ascii: &str) -> String {
    ascii.chars().map(|c| char::from_u32(0xE0000 + c as u32).unwrap()).collect()
}

const BLACK_FLAG: &str = "\u{1F3F4}";
const CANCEL: &str = "\u{E007F}";

#[test]
fn regional_indicator_pairs_decode_to_iso_3166_alpha_2() {
    for code in ["DE", "US", "JP", "UN", "EU", "XX"] {
        assert_eq!(region(&ri(code)).as_deref(), Some(code));
    }
}

#[test]
fn subdivision_tag_sequences_decode_to_iso_3166_2() {
    for (tag, code) in [("gbeng", "GB-ENG"), ("gbsct", "GB-SCT"), ("gbwls", "GB-WLS"), ("fr75", "FR-75")] {
        assert_eq!(region(&format!("{BLACK_FLAG}{}{CANCEL}", tags(tag))).as_deref(), Some(code));
    }
}

#[test]
fn non_flags_and_malformed_flags_have_no_region() {
    let malformed = [
        String::new(),
        "word".to_owned(),
        "\u{1F642}".to_owned(),
        BLACK_FLAG.to_owned(),
        "\u{1F3F4}\u{200D}\u{2620}\u{FE0F}".to_owned(),
        format!("A{}", ri("DE")),
        ri("D"),
        ri("DEF"),
        ri("DEFR"),
        format!("{}E", ri("D")),
        format!("{}!", ri("DE")),
        format!("{BLACK_FLAG}{}", tags("gbeng")),
        format!("{BLACK_FLAG}{CANCEL}"),
        format!("{BLACK_FLAG}{}{CANCEL}", tags("gb")),
        format!("{BLACK_FLAG}{}{CANCEL}", tags("1beng")),
        format!("{BLACK_FLAG}{}e{}{CANCEL}", tags("gb"), tags("ng")),
        format!("{BLACK_FLAG}{}{CANCEL}x", tags("gbeng")),
        format!("{BLACK_FLAG}{}{CANCEL}", tags("GBENG")),
    ];
    for text in malformed {
        assert_eq!(region(&text), None, "{text:?}");
    }
}
