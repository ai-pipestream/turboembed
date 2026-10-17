//! Unit tests for the WordPiece differential run and its JVM offset model.

use super::model::{PortSpan, Unit, expected_port_spans, map, port_spans_fit, special_occurrences};
use super::*;

fn text_of(units: &[Unit]) -> String {
    units.iter().map(|u| u.c).collect()
}

fn spans_of(units: &[Unit]) -> Vec<[u32; 2]> {
    units.iter().map(|u| [u.start, u.end]).collect()
}

#[test]
fn final_sigma_follows_the_jvm_rule_within_a_run() {
    assert_eq!(map("ΟΔΥΣΣΕΥΣ ΣΟΦΟΣ ΣΑΣ Σ ΑΣ. λόγος ΛΌΓΟΣ\n").as_jvm, "ΟΔΥΣΣΕΥς ΣΟΦΟς ΣΑς Σ Ας. λόγος ΛΌΓΟς\n");
    // A mark after the sigma is skipped; a letter after it is not, and
    // the search stops at the run's end, so a word after a space does
    // not count.
    assert_eq!(map("ΟΣ\u{0E48} ΟΣa ΟΣ b").as_jvm, "Ος\u{0E48} ΟΣa Ος b");
    assert!(map("ΟΣ b").used.sigma);
    assert!(!map("σ Σ").used.sigma);
}

#[test]
fn whitespace_is_a_space_and_controls_are_dropped() {
    // The no-break space is whitespace; the vertical tab is a control
    // and gone, so the characters around it are one word.
    let m = map("Cafe\u{301} \u{a0}x\u{b}y");
    assert_eq!(text_of(&m.units), "cafe  xy");
    assert_eq!(spans_of(&m.units), [[0, 1], [1, 2], [2, 3], [3, 6], [6, 7], [7, 9], [9, 10], [11, 12]]);
    assert!(m.used.stripped && m.used.reordered.is_empty());
    // A mark at the start of a run is dropped without a span.
    assert_eq!(spans_of(&map("\u{301}a").units), [[2, 3]]);
}

#[test]
fn ideographs_and_punctuation_are_isolated() {
    let m = map("a.b生x");
    assert_eq!(text_of(&m.units), "a . b 生 x");
    assert_eq!(spans_of(&m.units), [[0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 6], [6, 6], [6, 7]]);
}

#[test]
fn reordered_marks_give_the_run_to_every_character() {
    let m = map("Vie\u{302}\u{323}t x");
    assert_eq!(text_of(&m.units), "viet x");
    assert_eq!(spans_of(&m.units), [[0, 8], [0, 8], [0, 8], [0, 8], [8, 9], [9, 10]]);
    assert_eq!(m.used.reordered, [[0, 8]]);
}

#[test]
fn the_ports_spans_are_derived_from_the_jvms_piece_by_piece() {
    // A stripped mark: the JVM's piece runs over it, the port's ends
    // before it. The id does not matter to the spans.
    let text = "Cafe\u{301} x";
    let m = map(text);
    let jvm = [(1, [0, 6]), (2, [7, 8])];
    let expected = expected_port_spans(text, &m, &jvm);
    assert_eq!(expected, [PortSpan::Exact([0, 4]), PortSpan::Exact([7, 8])]);
    assert!(port_spans_fit(text, &expected, &[(1, [0, 4]), (2, [7, 8])]));
    // A fabricated wrong port span: the mark kept, a piece short by a
    // character, a piece too many or too few.
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 6]), (2, [7, 8])]));
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 3]), (2, [7, 8])]));
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 4])]));
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 4]), (2, [7, 8]), (3, [8, 8])]));
    // A reordered run: the JVM gives every piece the run, the port's
    // pieces lie inside it, and one outside the run does not fit.
    let text = "Vie\u{302}\u{323}t x";
    let m = map(text);
    let jvm = [(1, [0, 8]), (2, [0, 8]), (3, [9, 10])];
    let expected = expected_port_spans(text, &m, &jvm);
    assert_eq!(expected, [PortSpan::Within([0, 8]), PortSpan::Within([0, 8]), PortSpan::Exact([9, 10])]);
    assert!(port_spans_fit(text, &expected, &[(1, [0, 2]), (2, [2, 8]), (3, [9, 10])]));
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 2]), (2, [2, 9]), (3, [9, 10])]));
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 0]), (2, [2, 8]), (3, [9, 10])]));
    // Inside the run but with a gap, not from its start, or not to its
    // end: the pieces must be contiguous and cover the run.
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 2]), (2, [3, 8]), (3, [9, 10])]));
    assert!(!port_spans_fit(text, &expected, &[(1, [1, 2]), (2, [2, 8]), (3, [9, 10])]));
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 2]), (2, [2, 7]), (3, [9, 10])]));
    // A run whose marks are all stripped: the port's piece is the base
    // character alone, and a piece that ends in a mark does not fit.
    let text = "a\u{308}\u{323} x";
    let m = map(text);
    let jvm = [(1, [0, 5]), (2, [6, 7])];
    let expected = expected_port_spans(text, &m, &jvm);
    assert_eq!(expected, [PortSpan::Within([0, 5]), PortSpan::Exact([6, 7])]);
    assert!(port_spans_fit(text, &expected, &[(1, [0, 1]), (2, [6, 7])]));
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 3]), (2, [6, 7])]));
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 5]), (2, [6, 7])]));
    // A Hangul syllable in a run is two or three pieces with one span.
    let text = "\u{D55C}\u{302}\u{323}\u{AD6D} x";
    let m = map(text);
    let jvm = [(1, [0, 10]), (2, [0, 10]), (3, [0, 10]), (4, [0, 10]), (5, [0, 10]), (6, [11, 12])];
    let expected = expected_port_spans(text, &m, &jvm);
    assert_eq!(expected[0], PortSpan::Within([0, 10]));
    let fits = [(1, [0, 3]), (2, [0, 3]), (3, [0, 3]), (4, [7, 10]), (5, [7, 10]), (6, [11, 12])];
    assert!(port_spans_fit(text, &expected, &fits));
    // The repeated span must be the one right before it.
    let gap = [(1, [0, 3]), (2, [0, 3]), (3, [7, 10]), (4, [0, 3]), (5, [7, 10]), (6, [11, 12])];
    assert!(!port_spans_fit(text, &expected, &gap));
    // Two reordered runs in a row, each covered.
    let text = "Vie\u{302}\u{323}t Vie\u{302}\u{323}t";
    let m = map(text);
    let jvm = [(1, [0, 8]), (2, [0, 8]), (1, [9, 17]), (2, [9, 17])];
    let expected = expected_port_spans(text, &m, &jvm);
    assert!(port_spans_fit(text, &expected, &[(1, [0, 3]), (2, [3, 8]), (1, [9, 12]), (2, [12, 17])]));
    assert!(!port_spans_fit(text, &expected, &[(1, [0, 3]), (2, [3, 7]), (1, [9, 12]), (2, [12, 17])]));
    // A final sigma leaves the spans alone.
    let text = "ΟΔΥΣΣΕΥΣ";
    let m = map(text);
    assert_eq!(expected_port_spans(text, &m, &[(1, [0, 16])]), [PortSpan::Exact([0, 16])]);
}

/// The checked in golden over combining marks is a documented
/// difference; the same golden against the port's pieces with one span
/// changed is a divergence, whichever way the span is wrong.
#[test]
fn a_fabricated_wrong_port_span_is_a_divergence() {
    let wp = WordPiece::new(&Resources::from_env()).unwrap_or_else(|e| panic!("{e}"));
    let data = turbo_goldens::data_dir();
    let path = data.join("subwords").join(VOCABULARY).join("subwords").join("accents-combining.golden");
    let golden = crate::read_golden(&path).unwrap_or_else(|e| panic!("{e}"));
    let text = golden.input_text(&data).unwrap();
    let doc = wp.annotate(&text, Some(&golden)).unwrap();
    assert!(matches!(wp.compare(&doc, &golden), Outcome::Documented(_)));
    let jvm = jvm_pieces(&golden).unwrap();
    let rust = rust_pieces(&doc);
    assert!(wp.documented(&text, &jvm, &rust).is_ok());
    // The port's piece given the JVM's span over the stripped mark.
    let i = rust.iter().zip(&jvm).position(|(r, j)| r.1 != j.1).expect("a piece over a mark");
    let mut wrong = rust.clone();
    wrong[i].1 = jvm[i].1;
    let reason = wp.documented(&text, &jvm, &wrong).unwrap_err();
    assert!(reason.contains(&format!("the port's span of piece {i}")), "{reason}");
    // A piece a byte short, a piece missing, a piece too many.
    let mut wrong = rust.clone();
    wrong[0].1[1] -= 1;
    assert!(wp.documented(&text, &jvm, &wrong).is_err());
    assert!(wp.documented(&text, &jvm, &rust[1..]).is_err());
    let mut wrong = rust.clone();
    wrong.push(rust[rust.len() - 1]);
    assert!(wp.documented(&text, &jvm, &wrong).is_err());
}

/// The checked in final sigma golden is a documented difference; the
/// port's own pieces on the text are still checked, so one of them
/// changed beyond a sigma's id is a divergence.
#[test]
fn the_ports_pieces_are_checked_when_final_sigma_differs() {
    let wp = WordPiece::new(&Resources::from_env()).unwrap_or_else(|e| panic!("{e}"));
    let data = turbo_goldens::data_dir();
    let path = data.join("subwords").join(VOCABULARY).join("subwords").join("final-sigma.golden");
    let golden = crate::read_golden(&path).unwrap_or_else(|e| panic!("{e}"));
    let text = golden.input_text(&data).unwrap();
    assert_ne!(map(&text).as_jvm, text, "the input has a final sigma");
    let doc = wp.annotate(&text, Some(&golden)).unwrap();
    let jvm = jvm_pieces(&golden).unwrap();
    let rust = rust_pieces(&doc);
    wp.documented(&text, &jvm, &rust).unwrap_or_else(|e| panic!("{e}"));
    // The id of a piece in a word without a sigma ("λόγος"), its span,
    // and in the first word, which has sigmas, a piece a byte short, a
    // character short and a piece missing.
    let i = rust.iter().position(|&(_, [a, b])| text[a as usize..b as usize].starts_with('λ')).unwrap();
    let mut wrong = rust.clone();
    wrong[i].0 += 1;
    let reason = wp.documented(&text, &jvm, &wrong).unwrap_err();
    assert!(reason.contains("outside the words with a capital sigma"), "{reason}");
    let mut wrong = rust.clone();
    wrong[i].1[1] -= 2;
    assert!(wp.documented(&text, &jvm, &wrong).unwrap_err().contains("outside the words"));
    let mut wrong = rust.clone();
    wrong[0].1[1] -= 1;
    assert!(wp.documented(&text, &jvm, &wrong).unwrap_err().contains("do not cover the word"));
    let mut wrong = rust.clone();
    wrong[0].1[1] -= 2;
    assert!(wp.documented(&text, &jvm, &wrong).unwrap_err().contains("do not cover the word"));
    assert!(wp.documented(&text, &jvm, &rust[1..]).unwrap_err().contains("do not cover the word"));
}

#[test]
fn cjk_ideographs_compose_with_special_tokens_and_stripped_marks() {
    let wp = WordPiece::new(&Resources::from_env()).unwrap_or_else(|e| panic!("{e}"));
    // Combine `[CLS]`, a stripped mark on `"Cafe\u{301}"`, and two CJK Extension E
    // ideographs `"\u{2B820}\u{2B821}"` that upstream keeps as one unknown word
    // while the JVM isolates into two unknown pieces.
    let text = "[CLS] Cafe\u{301} \u{2B820}\u{2B821}";
    let doc = wp.annotate(text, None).unwrap();
    let rust = rust_pieces(&doc);
    let cls_doc = wp.annotate("[CLS]".to_ascii_lowercase().as_str(), None).unwrap();
    let mut jvm = rust_pieces(&cls_doc);
    let cafe_id = rust[1].0;
    let unk = wp.unknown;
    jvm.extend_from_slice(&[(cafe_id, [6, 12]), (unk, [13, 17]), (unk, [17, 21])]);
    let why = wp.documented(text, &jvm, &rust).unwrap_or_else(|e| panic!("{e}"));
    assert!(why.contains("special_token_strings_are_special_tokens_here_only"), "{why}");
    assert!(why.contains("a_stripped_mark_is_in_the_jvm_span_and_not_in_ours"), "{why}");
    assert!(why.contains("the_ideographs_from_u2b820_are_words_on_the_jvm_only"), "{why}");
}

/// Explains one golden of a fuzz run: `TURBO_FUZZ_GOLDEN` names the
/// golden file and `TURBO_FUZZ_INPUT` its input text.
#[test]
#[ignore = "a diagnostic over files named by the environment"]
fn explain_one_golden_from_the_environment() {
    let wp = WordPiece::new(&Resources::from_env()).unwrap_or_else(|e| panic!("{e}"));
    let golden = crate::read_golden(std::path::Path::new(&std::env::var("TURBO_FUZZ_GOLDEN").unwrap())).unwrap();
    let text = std::fs::read_to_string(std::env::var("TURBO_FUZZ_INPUT").unwrap()).unwrap();
    let doc = wp.annotate(&text, Some(&golden)).unwrap();
    let jvm = jvm_pieces(&golden).unwrap();
    eprintln!("{:?}", wp.documented(&text, &jvm, &rust_pieces(&doc)));
}

#[test]
fn special_token_strings_are_found_with_their_ids() {
    let special = [("[CLS]", 101), ("[SEP]", 102)];
    assert_eq!(special_occurrences("a [SEP] b [CLS]", &special), [(2, 7, 102), (10, 15, 101)]);
    assert!(special_occurrences("a [sep] b", &special).is_empty());
}
