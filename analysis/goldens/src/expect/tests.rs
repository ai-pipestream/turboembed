use super::*;
use crate::format::{
    AnnotatorRef, Column, ColumnData, Generator, GoldenAnchor, GoldenLayer, Header, Input, Jdk, OpenNlp, UnicodeVersion,
};
use turbo_analysis_layers::Strs;

fn span(utf8: (u32, u32), utf16: (u32, u32), cut: bool, prob: f64) -> JvmSpan {
    JvmSpan { utf8, utf16, cut, prob: Some(prob), kind: None }
}

#[test]
fn halves_of_a_pair_join_and_their_probabilities_multiply() {
    // "'😀." as TokenizerME splits it: ' | high | low | .
    let text = "'😀.";
    let spans = [
        span((0, 1), (0, 1), false, 0.9),
        span((1, 5), (1, 2), true, 0.5),
        span((1, 5), (2, 3), true, 0.5),
        span((5, 6), (3, 4), false, 0.8),
    ];
    assert_eq!(joined_me_spans(text, &spans).unwrap(), [(0, 1, 0.9), (1, 5, 0.25), (5, 6, 0.8)]);
    let whole = whole_character_spans(text, &spans).unwrap();
    assert_eq!(whole.iter().map(|s| s.utf8).collect::<Vec<_>>(), [(0, 1), (1, 5), (5, 6)]);
    assert_eq!(whole[1].utf16, (1, 3));
    assert!(!whole[1].cut);
    assert!(joined_me_spans(text, &spans[..2]).is_err());
    assert!(whole_character_spans(text, &spans[..2]).is_err());
}

#[test]
fn jvm_spans_scopes_word_types_to_tokens_and_rejects_mismatched_auxiliary_rows() {
    let header = Header {
        generator: Generator { name: "g".into(), version: 1 },
        opennlp: OpenNlp { commit: [0; 20], version: "v".into() },
        jdk: Jdk { version: "21".into(), vm_version: "21+1".into(), vendor: "V".into() },
        unicode: UnicodeVersion { major: 15, minor: 0 },
        input: Input { set: "s".into(), id: "i".into(), utf8_len: 5, utf16_len: 5, sha256: [0; 32] },
        models: Vec::new(),
        annotator: AnnotatorRef { class: "C".into(), method: "m".into() },
        config: Vec::new(),
        error: None,
    };
    let mut kinds = Strs::default();
    kinds.push("alpha");
    let golden = Golden {
        header,
        layers: vec![
            GoldenLayer {
                id: "opennlp:sentences".into(),
                anchor: GoldenAnchor::Text,
                columns: vec![
                    Column { name: "start".into(), data: ColumnData::U32(vec![0, 3]) },
                    Column { name: "end".into(), data: ColumnData::U32(vec![2, 5]) },
                ],
            },
            GoldenLayer {
                id: "opennlp:tokens".into(),
                anchor: GoldenAnchor::Text,
                columns: vec![
                    Column { name: "start".into(), data: ColumnData::U32(vec![0, 3]) },
                    Column { name: "end".into(), data: ColumnData::U32(vec![2, 5]) },
                ],
            },
            GoldenLayer {
                id: "turbo:word-types".into(),
                anchor: GoldenAnchor::Aligned("opennlp:tokens".into()),
                columns: vec![Column { name: "kind".into(), data: ColumnData::Str(kinds) }],
            },
        ],
    };
    // Sentences ignore turbo:word-types even when turbo:word-types has fewer rows.
    let sents = jvm_spans(&golden, "ab cd", "opennlp:sentences").unwrap();
    assert_eq!(sents.len(), 2);
    assert!(sents.iter().all(|s| s.kind.is_none()));
    // Tokens reject mismatched turbo:word-types row counts instead of panicking.
    assert!(jvm_token_spans(&golden, "ab cd").is_err());
}

#[test]
fn newline_tokens_join_the_plain_tokens_in_order() {
    let text = "a\r\n b\n";
    assert_eq!(simple_newline_spans(text, &[(0, 1), (4, 5)]), [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]);
}

#[test]
fn the_jvms_keep_new_lines_spans_are_modelled_with_their_stale_start() {
    let jvm = |t: &str| jvm_simple_newline_spans(t, WhitespaceMode::Unicode);
    // A line break right after a token: its own span. After a space:
    // the span starting at the last token's end, which is the space,
    // and the next break one further on.
    assert_eq!(jvm("ab\ncd"), Ok(vec![(0, 2), (2, 3), (3, 5)]));
    assert_eq!(jvm("ab \n\ncd"), Ok(vec![(0, 2), (2, 3), (3, 4), (5, 7)]));
    // A break before any token: a span from -1, refused.
    assert_eq!(jvm("\nab"), Err("java.lang.IllegalArgumentException".into()));
    assert_eq!(jvm(" \n"), Err("java.lang.IllegalArgumentException".into()));
    // CR LF: two breaks, the second's span stale by one.
    assert_eq!(jvm("a\r\nb"), Ok(vec![(0, 1), (1, 2), (2, 3), (3, 4)]));
    // A character outside the BMP is two units of class other that
    // differ, so two tokens, in UTF-16 units.
    assert_eq!(jvm("\u{1F600}\n"), Ok(vec![(0, 1), (1, 2), (2, 3)]));
    // Where the port's expectation and the JVM agree.
    let text = "ab\ncd";
    assert_eq!(simple_newline_spans(text, &[(0, 2), (3, 5)]), [(0, 2), (2, 3), (3, 5)]);
    assert_eq!(jvm("x1,,.").unwrap(), [(0, 1), (1, 2), (2, 4), (4, 5)]);
}

#[test]
fn trailing_marks_leave_a_piece() {
    let text = "Cafe\u{301} x\u{301}\u{301}";
    assert_eq!(without_trailing_marks(text, [0, 6]), [0, 4]);
    assert_eq!(without_trailing_marks(text, [7, 12]), [7, 8]);
    assert_eq!(without_trailing_marks(text, [8, 12]), [8, 12]);
    // A dropped control between the letter and the mark is covered by
    // the JVM's span and left out with the mark; alone it is kept.
    let text = "x\u{FEFF}\u{5B8}y\0 z\u{1B}";
    assert_eq!(without_trailing_marks(text, [0, 6]), [0, 1]);
    assert_eq!(without_trailing_marks(text, [6, 8]), [6, 7]);
    assert_eq!(without_trailing_marks(text, [1, 6]), [1, 6]);
    assert_eq!(without_trailing_marks(text, [9, 11]), [9, 10]);
}

#[test]
fn edits_of_another_granularity_are_equivalent_when_the_text_agrees() {
    let text = "aXYb";
    let e = |start, end, with: &str| ViewEdit { start, end, with: with.into() };
    let one = [e(1, 3, "xy")];
    let two = [e(1, 2, "x"), e(2, 3, "y")];
    assert!(view_edits_equivalent(text, &one, &two));
    assert!(!view_edits_equivalent(text, &one, &[e(1, 3, "xz")]));
    assert!(!view_edits_equivalent(text, &one, &[]));
    assert_eq!(apply_view_edits(text, &two).unwrap(), "axyb");
    assert!(apply_view_edits(text, &[e(2, 1, "")]).is_err());
    // A diff over repeated characters: one list deletes the first x and
    // inserts one before the last, the other changes only the marks.
    // Region by region they differ; over both regions they agree.
    let text = "\u{301}\u{323}xxxx\u{308}\u{301}\u{323}";
    let rust = [e(0, 4, "\u{323}\u{301}"), e(7, 14, "\u{1e8d}\u{323}\u{301}")];
    let golden = [e(0, 5, "\u{323}\u{301}"), e(8, 10, "\u{1e8d}"), e(10, 12, "\u{323}"), e(12, 14, "\u{301}")];
    assert!(view_edits_equivalent(text, &rust, &golden));
    assert_eq!(apply_view_edits(text, &rust), apply_view_edits(text, &golden));
    // Not when the texts differ after all.
    assert!(!view_edits_equivalent(text, &rust, &[e(0, 5, "\u{323}\u{301}"), e(8, 10, "\u{1e8d}")]));
}

#[test]
fn an_edit_attributed_to_another_place_is_not_equivalent() {
    let e = |start, end, with: &str| ViewEdit { start, end, with: with.into() };
    // Both delete one "ab" of three, so the outputs agree, but the
    // regions do not and the source between them is not one
    // repeated character.
    let text = "ababab";
    let rust = [e(0, 2, "")];
    let golden = [e(4, 6, "")];
    assert_eq!(apply_view_edits(text, &rust), apply_view_edits(text, &golden));
    assert!(!view_edits_equivalent(text, &rust, &golden));
    // A change in one region and the same change elsewhere.
    let text = "aXbYc";
    let d = view_edit_disagreements(text, &[e(1, 2, "x")], &[e(3, 4, "x")]);
    assert_eq!(d.len(), 2);
    assert_eq!((d[0].start, d[0].end, d[0].results()), (1, 2, Some(("x", "X"))));
    assert_eq!((d[1].start, d[1].end, d[1].source(text)), (3, 4, "Y"));
    // The same two regions, equivalent within each.
    assert!(view_edits_equivalent(text, &[e(1, 2, "x"), e(3, 4, "y")], &[e(1, 2, "x"), e(3, 4, "y")]));
}

#[test]
fn a_deletion_at_either_end_of_a_repeated_run_is_equivalent() {
    let e = |start, end, with: &str| ViewEdit { start, end, with: with.into() };
    // "6/98/2635 4" with the two spaces of a date collapsed: the diff
    // may drop the first or the last of three.
    let text = "a   b";
    let rust = [e(1, 2, ""), e(4, 5, "B")];
    let golden = [e(3, 4, ""), e(4, 5, "B")];
    assert_eq!(apply_view_edits(text, &rust), apply_view_edits(text, &golden));
    assert!(view_edits_equivalent(text, &rust, &golden));
    // Not over a run of two different characters.
    let text = "a xx b";
    assert!(!view_edits_equivalent(text, &[e(1, 2, "")], &[e(4, 5, "")]));
}
