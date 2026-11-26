//! `EmojiRemove`.

use super::*;

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
