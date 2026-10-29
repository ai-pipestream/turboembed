use turbo_unicode::confusables::prototype;

use super::super::confusables_17::CHANGES_17;
use super::rules::skeleton_17;
use super::*;

fn e(start: usize, end: usize, with: &str) -> ViewEdit {
    ViewEdit { start, end, with: with.into() }
}

/// The outcome of the run `name` on `text` against the JVM edits
/// `golden`, the exact comparison taken as failed.
fn outcome(name: &str, text: &str, golden: &[ViewEdit]) -> Outcome {
    let (normalize, edits, documented) = table(name).unwrap();
    let run = NormalizerRun { name: name.into(), normalize, edits, documented };
    let doc = run.annotate(text, None).unwrap();
    run.explain(&doc, golden, "differs".into())
}

fn documented(name: &str, text: &str, golden: &[ViewEdit]) {
    let out = outcome(name, text, golden);
    assert!(matches!(out, Outcome::Documented(_)), "{name} on {text:?}: {out}");
}

fn divergence(name: &str, text: &str, golden: &[ViewEdit]) {
    let out = outcome(name, text, golden);
    assert!(matches!(out, Outcome::Divergence(_)), "{name} on {text:?}: {out}");
}

/// Explains one golden of a fuzz run: `TURBO_FUZZ_GOLDEN` names the
/// golden file, `TURBO_FUZZ_INPUT` its input text and `TURBO_FUZZ_RUN`
/// the normalizer run.
#[test]
#[ignore = "a diagnostic over files named by the environment"]
fn explain_one_golden_from_the_environment() {
    let name = std::env::var("TURBO_FUZZ_RUN").unwrap();
    let (normalize, edits, documented) = table(&name).unwrap();
    let run = NormalizerRun { name: name.clone(), normalize, edits, documented };
    let golden = crate::read_golden(std::path::Path::new(&std::env::var("TURBO_FUZZ_GOLDEN").unwrap())).unwrap();
    let text = std::fs::read_to_string(std::env::var("TURBO_FUZZ_INPUT").unwrap()).unwrap();
    let doc = run.annotate(&text, Some(&golden)).unwrap();
    eprintln!("port: {:?}", view_edits(&doc));
    eprintln!("jvm:  {:?}", golden_edits(&golden));
    eprintln!("{}", run.compare(&doc, &golden));
}

#[test]
fn the_character_additions_are_documented_and_other_characters_are_not() {
    // U+2E42 folded here only; the JVM folds U+201C as the port does.
    documented("quote", "\u{2E42}Wort\u{201C}", &[e(7, 10, "\"")]);
    divergence("quote", "\u{2E42}Wort\u{201C}", &[e(7, 10, "'")]);
    divergence("quote", "\u{2E42}Wort\u{201C}", &[]);
    documented("bullet", "\u{204C} a \u{2022} b", &[e(6, 9, " ")]);
    divergence("bullet", "\u{204C} a \u{2022} b", &[]);
    documented("ellipsis", "a\u{2024}b\u{2026}", &[e(5, 8, "...")]);
    divergence("ellipsis", "a\u{2024}b\u{2026}", &[e(5, 8, "..")]);
    documented("invisible", "a\u{206A}b\u{200B}", &[e(5, 8, "")]);
    divergence("invisible", "a\u{206A}b\u{200B}", &[]);
    documented("default-chain", "\u{2E42}A\u{206A}", &[e(3, 4, "a")]);
    divergence("default-chain", "\u{2E42}A\u{206A}", &[e(3, 4, "A")]);
    // U+206A between spaces or at text boundaries in `default-chain`:
    // deleting U+206A before Whitespace::collapse merges or trims the
    // surrounding spaces where the JVM keeps `" \u{206A} "`, `"\u{206A} "`,
    // or `" \u{206A}"`.
    documented("default-chain", "a \u{206A} b", &[]);
    documented("default-chain", "\u{206A} a", &[]);
    documented("default-chain", "a \u{206A}", &[]);
    divergence("default-chain", "a \u{206A} b", &[e(1, 6, "")]);
}

#[test]
fn chain_only_adjusts_spaces_exposed_by_deleted_characters() {
    for c in (0x206a..=0x206f).filter_map(char::from_u32) {
        for text in [
            format!("a {c}  b"),
            format!("a  {c} b"),
            format!("{c}  a"),
            format!("a  {c}"),
            format!(" {c} a"),
            format!("a {c} "),
            format!(" {c} "),
        ] {
            divergence("default-chain", &text, &[]);
        }
        for text in [format!("a {c} b"), format!("{c} a"), format!("a {c}"), format!("a {c} {c} b")] {
            documented("default-chain", &text, &[]);
        }
        let text = format!("a {c}  b");
        documented("default-chain", &text, &[e(5, 7, " ")]);
    }
    divergence("default-chain", "a  \u{2e42}b", &[]);
}

#[test]
fn a_join_after_combining_marks_is_documented_and_a_missing_join_is_not() {
    let text = "cafe\u{301}-\n  teria and well-\nknown";
    let well = text.find("well-").unwrap() + 4;
    documented("dehyphenation", text, &[e(well, well + 2, "")]);
    divergence("dehyphenation", text, &[]);
    // A letter right before the hyphen: both join, so a golden without
    // the join is a divergence, and so is a join of another kind.
    divergence("dehyphenation", "cafe-\n  teria", &[]);
    divergence("dehyphenation", text, &[e(well, well + 2, " ")]);
}

#[test]
fn the_added_contractions_are_documented_and_the_shared_ones_are_not() {
    documented("english-contraction", "here're y'all", &[]);
    documented("english-contraction", "THESE\u{2019}LL do", &[]);
    divergence("english-contraction", "they're here", &[]);
    divergence("english-contraction", "here're y'all", &[e(0, 7, "we are")]);
}

#[test]
fn a_symbol_in_running_text_is_documented_and_a_lone_one_is_not() {
    documented("symbol-joiner", "R&D & co \u{00A9} x", &[]);
    divergence("symbol-joiner", "&", &[]);
    divergence("symbol-joiner", "a & b", &[e(2, 3, "et")]);
}

#[test]
fn an_international_address_is_documented_and_an_ascii_one_is_not() {
    documented("bounded-url", "mail Jos\u{E9}@example.com now", &[]);
    documented("bounded-url", "mail \u{E9}a@b.co now", &[e(7, 13, " ")]);
    divergence("bounded-url", "mail a@b.co now", &[]);
    divergence("bounded-url", "mail Jos\u{E9}@example.com now", &[e(5, 9, "Jos ")]);
    // Text the JVM kept that is not the source's, or out of its order.
    divergence("bounded-url", "mail Jos\u{E9}@example.com now", &[e(5, 8, "Zos")]);
    divergence("bounded-url", "see first\u{5B8}+tag@a.b.c\u{20DD} now", &[e(4, 24, "c\u{20DD} first\u{5B8}")]);
    // A URL both replace right before the address, one region for the
    // diff; and a kept stretch in a token without an `@`.
    let text = "needs https://s\u{E48}.d.ex/ x.y.z@sub.d\u{338}omain.example b";
    documented("bounded-url", text, &[e(6, 36, "   ")]);
    divergence("bounded-url", text, &[e(6, 36, "   "), e(49, 50, " \u{E9}")]);
    // An earlier URL in the same disagreement region shares a substring
    // with the internationalized local-part that the JVM keeps.
    let text = "a https://jos\u{E9}.example/ jos\u{E9}@b.co z";
    documented("bounded-url", text, &[e(2, 35, "  jos\u{E9}")]);
    // The ASCII part replaced in the middle, the Greek kept by the JVM.
    let text = "x brien@example.com...\u{3AC}\u{3BD} y";
    documented("bounded-url", text, &[e(2, 22, " ")]);
    divergence("bounded-url", text, &[e(2, 22, "  ")]);
    // A URL both replace, then an address the JVM's scan leaves at a
    // combining mark inside the domain.
    let text = "a https://h.example/ x.y@sub.d\u{338}omain.example b";
    documented("bounded-url", text, &[e(2, 20, " "), e(21, 30, " ")]);
    divergence("bounded-url", text, &[e(2, 20, " "), e(21, 30, "  ")]);
    // Marks on both sides of the ASCII part: the JVM keeps both ends.
    let text = "see first\u{5B8}+tag@a.b.c\u{20DD} now";
    documented("bounded-url", text, &[e(11, 20, " ")]);
    divergence("bounded-url", text, &[e(11, 20, "  ")]);
    divergence("bounded-url", text, &[e(11, 23, " ")]);
}

#[test]
fn neighbouring_deletions_the_jvm_lists_as_one_are_documented() {
    // Two invisible characters side by side: two edits here, one gap
    // in OpenNLP's alignment.
    documented("invisible", "a\u{200E}\u{FEFF}b", &[e(1, 7, "")]);
    divergence("invisible", "a\u{200E}\u{FEFF}b", &[e(1, 4, ""), e(4, 7, "x")]);
    assert_eq!(
        joined_deletions(&[e(1, 4, ""), e(4, 7, ""), e(8, 9, ""), e(9, 10, "y")]),
        [e(1, 7, ""), e(8, 9, ""), e(9, 10, "y")]
    );
}

#[test]
fn the_jvms_skeleton_by_the_older_data_is_documented() {
    // 18.0 maps U+030B to two acute accents and U+3099 to U+FF9E; 17.0
    // left both. U+309A goes to U+030A in both.
    // NFD puts the kana marks (class 8) before the accent (class 230).
    assert_eq!(skeleton_17("o\u{30B}\u{3099}\u{309A}"), "o\u{3099}\u{30A}\u{30B}");
    documented("confusable-skeleton", "o\u{30B}", &[]);
    documented("confusable-skeleton", "a\u{3099}\u{309A}", &[e(4, 7, "\u{30A}")]);
    divergence("confusable-skeleton", "o\u{30B}", &[e(0, 3, "x")]);
    // A character both versions map the same way: the JVM must map it.
    documented("confusable-skeleton", "\u{43E}\u{30B}", &[e(0, 2, "o")]);
    divergence("confusable-skeleton", "\u{43E}\u{30B}", &[]);
    // The table is sorted for the binary search, and holds the
    // differences only.
    assert!(CHANGES_17.windows(2).all(|w| w[0].0 < w[1].0));
    assert!(CHANGES_17.iter().all(|&(cp, t)| prototype(char::from_u32(cp).unwrap()) != t));
}
