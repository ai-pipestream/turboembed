//! The whitespace normalizer in its three modes. The expectations are
//! those of OpenNLP's WhitespaceCharSequenceNormalizer,
//! LineBreakPreservingWhitespaceCharSequenceNormalizer and
//! ParagraphPreservingWhitespaceCharSequenceNormalizer tests
//! (UnicodeCharSequenceNormalizerTest, the two *PreservingTest classes and
//! AlignedNormalizerPipelineTest), with OpenNLP's UTF-16 offsets written
//! as UTF-8 byte offsets.

mod folds_support;

use folds_support::{back, check_alignment, covered, forward, run_checked};
use turbo_normalize::tables::{LINE_BREAKS, WHITESPACE, is_whitespace};
use turbo_normalize::{Normalizer, Whitespace, WhitespaceMode, apply, normalize};

/// An edit the whitespace normalizer may make: a run of whitespace
/// becomes nothing, one space or one line feed.
fn whitespace_edit(parent: &str, child: &str) -> bool {
    !parent.is_empty() && parent.chars().all(is_whitespace) && matches!(child, "" | " " | "\n")
}

fn collapse(s: &str) -> String {
    run_checked(&Whitespace::collapse(), s, whitespace_edit)
}

fn lines(s: &str) -> String {
    run_checked(&Whitespace::keep_line_breaks(), s, whitespace_edit)
}

fn paragraphs(s: &str) -> String {
    run_checked(&Whitespace::keep_paragraphs(), s, whitespace_edit)
}

fn all_modes() -> [Whitespace; 3] {
    [Whitespace::collapse(), Whitespace::keep_line_breaks(), Whitespace::keep_paragraphs()]
}

#[test]
fn ids_and_configurations() {
    assert_eq!(Whitespace::ID, "turbo:normalize:whitespace");
    assert_eq!(Whitespace::default().mode(), WhitespaceMode::Collapse);
    let hashes: Vec<u64> = all_modes().iter().map(|w| w.config_hash()).collect();
    assert_ne!(hashes[0], hashes[1]);
    assert_ne!(hashes[1], hashes[2]);
    assert_ne!(hashes[0], hashes[2]);
    assert_eq!(
        Whitespace::keep_paragraphs().config_hash(),
        Whitespace::new(WhitespaceMode::KeepParagraphs).config_hash()
    );
}

// UnicodeCharSequenceNormalizerTest.testWhitespaceCollapsesUnicodeRunsAndTrims
#[test]
fn collapse_folds_unicode_runs_and_trims() {
    assert_eq!(collapse("  a\u{A0}\u{A0}b\u{3000}  "), "a b");
}

// AlignedNormalizerPipelineTest.whitespaceCollapseAndTrimMapsSpanBackToOriginal
#[test]
fn collapse_maps_a_word_back_through_collapse_and_trim() {
    let src = "  hello   world  ";
    let (out, a) = normalize(&Whitespace::collapse(), src);
    assert_eq!(out, "hello world");
    assert_eq!(back(&a, 6, 11).0 as usize, src.find("world").unwrap());
    assert_eq!(covered(src, &a, 6, 11), "world");
}

// AlignedNormalizerPipelineTest.roundTripOfAFullySpanningMatchReturnsTheWholeOriginal
#[test]
fn the_whole_output_covers_the_source_from_first_to_last_kept_character() {
    let src = "  the   quick  ";
    let (out, a) = normalize(&Whitespace::collapse(), src);
    assert_eq!(out, "the quick");
    assert_eq!(covered(src, &a, 0, out.len()), "the   quick");
}

// AlignedNormalizerPipelineTest.whitespaceNormalizerCollapsesAllWhitespaceToEmptyWithAValidSpan
#[test]
fn all_whitespace_becomes_empty() {
    for w in all_modes() {
        let (out, a) = normalize(&w, "   ");
        assert_eq!(out, "");
        check_alignment("   ", &out, &a, whitespace_edit);
    }
}

// The second half of the OpenNLP test above: the empty match at the start
// of the empty output maps back to a valid empty span. The alignment in
// turbo-analysis maps an empty span that sits on a deletion with its start
// after its end (the start moves past the deletion, the end does not), and
// `Span::new` panics, so this fails until `Alignment` handles that case.
#[test]
#[ignore = "turbo-analysis Alignment maps an empty span on a deletion to start > end"]
fn empty_output_span_at_a_deletion() {
    let (_, a) = normalize(&Whitespace::collapse(), "   ");
    let (s, e) = back(&a, 0, 0);
    assert!(s <= e);
    let (_, a) = normalize(&Whitespace::collapse(), "a   ");
    let (s, e) = back(&a, 1, 1);
    assert!(s <= e);
}

// AlignedNormalizerPipelineTest.lineBreakPreservingCollapsesHorizontalRunsButKeepsBreaks
#[test]
fn collapse_flattens_a_blank_line_that_keep_line_breaks_keeps() {
    let src = "Hello   world\n\n\tfoo  bar";
    assert_eq!(lines(src), "Hello world\nfoo bar");
    assert_eq!(collapse(src), "Hello world foo bar");
    let (_, a) = normalize(&Whitespace::keep_line_breaks(), src);
    assert_eq!(back(&a, 16, 19).0 as usize, src.find("bar").unwrap());
    assert_eq!(covered(src, &a, 16, 19), "bar");
    assert_eq!(covered(src, &a, 11, 12), "\n\n\t");
}

// LineBreakPreservingWhitespaceCharSequenceNormalizerTest
#[test]
fn keep_line_breaks_matches_opennlp() {
    assert_eq!(lines("a \t  b\u{A0}\u{3000}c"), "a b c");
    assert_eq!(lines("a \n b"), "a\nb");
    assert_eq!(lines("one\n\n\ntwo"), "one\ntwo");
    assert_eq!(lines("a\r\nb"), "a\nb");
    assert_eq!(lines("a\u{2028}b"), "a\nb");
    assert_eq!(lines("a \u{2029} b"), "a\nb");
    assert_eq!(lines("  a  "), "a");
    assert_eq!(lines("\n a \n"), "a");
    assert_eq!(lines(" \n "), "");
    assert_eq!(lines(""), "");
    // AlignedNormalizerPipelineTest.lineBreakPreservingNormalizesCrLfAndUnicodeSeparators
    assert_eq!(lines("a\n\n\n\nb"), "a\nb");
    assert_eq!(lines("x\u{2028}y"), "x\ny");
    assert_eq!(lines("p\u{2029}q"), "p\nq");
    assert_eq!(lines("a  b \n c"), "a b\nc");
    // A line feed between two words is already its own replacement.
    assert_eq!(lines("a\nb"), "a\nb");
}

// LineBreakPreservingWhitespaceCharSequenceNormalizerTest.alignmentMapsThroughCollapseAndTrim
// and the ParagraphPreserving test of the same name.
#[test]
fn spans_map_through_the_collapse_and_the_trim() {
    for w in [Whitespace::keep_line_breaks(), Whitespace::keep_paragraphs()] {
        let (out, a) = normalize(&w, "  x\n\ny  ");
        assert_eq!(out, "x\ny");
        assert_eq!(back(&a, 0, 1), (2, 3));
        assert_eq!(back(&a, 2, 3), (5, 6));
    }
}

// AlignedNormalizerPipelineTest.lineBreakPreservingTrimsLeadingAndTrailingBreaks
#[test]
fn leading_and_trailing_breaks_are_trimmed() {
    let src = "\n\nHello\n\n";
    let (out, a) = normalize(&Whitespace::keep_line_breaks(), src);
    assert_eq!(out, "Hello");
    assert_eq!(covered(src, &a, 0, 5), "Hello");
    assert_eq!(back(&a, 0, 5).0, 2);
}

// ParagraphPreservingWhitespaceCharSequenceNormalizerTest
#[test]
fn keep_paragraphs_matches_opennlp() {
    assert_eq!(paragraphs("a \t  b\u{A0}\u{3000}c"), "a b c");
    assert_eq!(paragraphs("a \n b"), "a b");
    assert_eq!(paragraphs("on the\nbank"), "on the bank");
    assert_eq!(paragraphs("a\n\nb"), "a\nb");
    assert_eq!(paragraphs("a\n\n\n\nb"), "a\nb");
    assert_eq!(paragraphs("Hello   world\n\n\tfoo  bar"), "Hello world\nfoo bar");
    assert_eq!(paragraphs("a\r\nb"), "a b");
    assert_eq!(paragraphs("a\r\n\r\nb"), "a\nb");
    assert_eq!(paragraphs("  a  "), "a");
    assert_eq!(paragraphs("\n a \n"), "a");
    assert_eq!(paragraphs(" \n "), "");
    assert_eq!(paragraphs(""), "");
    let src = "  one \t two\r\n\r\nthree \u{2028} four  ";
    assert_eq!(paragraphs(src), "one two\nthree four");
}

// ParagraphPreservingWhitespaceCharSequenceNormalizerTest, the two
// parameterized tests over every UnicodeWhitespace line break.
#[test]
fn every_line_break_unwraps_alone_and_keeps_a_paragraph_when_doubled() {
    for &br in LINE_BREAKS {
        if br == '\r' {
            assert_eq!(paragraphs("a\rb"), "a b");
            assert_eq!(paragraphs("a\r\nb"), "a b");
            assert_eq!(paragraphs("a\r\n\r\nb"), "a\nb");
        } else {
            assert_eq!(paragraphs(&format!("a{br}b")), "a b", "{br:?}");
            assert_eq!(paragraphs(&format!("a{br}{br}b")), "a\nb", "{br:?}");
        }
        assert_eq!(lines(&format!("a{br}b")), "a\nb", "{br:?}");
        // Two carriage returns are two breaks; only CR LF counts as one.
        assert_eq!(paragraphs("a\r\rb"), "a\nb");
    }
}

// ParagraphPreservingWhitespaceCharSequenceNormalizerTest.gutenbergHardWrapUnwrapsButKeepsParagraphBreak
#[test]
fn a_hard_wrapped_text_unwraps_but_keeps_its_paragraph_break() {
    let src = "Alice was beginning to get very tired of sitting by her sister on the\n\
bank, and of having nothing to do: once or twice she had peeped into\n\
the book her sister was reading, but it had no pictures or\n\
conversations in it, \"and what is the use of a book,\" thought Alice\n\
\"without pictures or conversations?\"\n\
\n\
So she was considering in her own mind (as well as she could, for the";
    assert_eq!(
        paragraphs(src),
        "Alice was beginning to get very tired of sitting by her sister on the bank, \
and of having nothing to do: once or twice she had peeped into \
the book her sister was reading, but it had no pictures or \
conversations in it, \"and what is the use of a book,\" thought Alice \
\"without pictures or conversations?\"\n\
So she was considering in her own mind (as well as she could, for the"
    );
}

// AlignedNormalizerPipelineTest.paragraphPreservingUnwrapsHardWrapsButKeepsBlankLines
#[test]
fn keep_paragraphs_maps_the_newline_back_to_its_whole_run() {
    let src = "Hello   world\n\n\tfoo  bar";
    let (out, a) = normalize(&Whitespace::keep_paragraphs(), src);
    assert_eq!(out, "Hello world\nfoo bar");
    assert_eq!(back(&a, 16, 19).0 as usize, src.find("bar").unwrap());
    assert_eq!(covered(src, &a, 16, 19), "bar");
    assert_eq!(covered(src, &a, 11, 12), "\n\n\t");
    // A source span inside the run maps forward to the one character
    // that stands for the run.
    assert_eq!(forward(&a, 14, 15), (11, 12));
}

// LineBreakPreserving / ParagraphPreserving alignedNormalizedMatchesNormalize
#[test]
fn the_same_input_in_all_three_modes() {
    let src = "  one \t two\r\n\r\nthree \u{2028} four  ";
    assert_eq!(collapse(src), "one two three four");
    assert_eq!(lines(src), "one two\nthree\nfour");
    assert_eq!(paragraphs(src), "one two\nthree four");
}

#[test]
fn every_table_member_is_whitespace_and_nothing_else_is() {
    for &w in WHITESPACE {
        assert_eq!(collapse(&format!("a{w}b")), "a b", "{w:?}");
        assert_eq!(collapse(&format!("{w}a{w}")), "a", "{w:?}");
    }
    // Lookalikes OpenNLP lists but does not treat as whitespace.
    for c in ['\u{180E}', '\u{200B}', '\u{200C}', '\u{200D}', '\u{2060}', '\u{FEFF}', '\u{1C}', '\u{1F}'] {
        let s = format!("a{c}b");
        assert_eq!(collapse(&s), s, "{c:?}");
    }
}

#[test]
fn text_with_single_spaces_is_returned_without_edits() {
    let src = "one two three";
    for w in all_modes() {
        let (_, a) = normalize(&w, src);
        assert!(a.edits().is_empty());
        assert!(matches!(apply(&w, src), std::borrow::Cow::Borrowed(_)));
    }
    // A single tab between words is a run that changes.
    let (out, a) = normalize(&Whitespace::collapse(), "a\tb");
    assert_eq!(out, "a b");
    assert_eq!(a.edits().len(), 1);
}

#[test]
fn each_run_is_one_edit_covering_its_whole_extent() {
    let src = " \u{3000}a \t\u{A0}b\n\n";
    let (out, a) = normalize(&Whitespace::collapse(), src);
    assert_eq!(out, "a b");
    let edits: Vec<(u32, u32, u32, u32)> =
        a.edits().iter().map(|e| (e.parent.start, e.parent.end, e.child.start, e.child.end)).collect();
    // The leading run (4 bytes) is deleted, the inner run (4 bytes)
    // becomes one space, the trailing run (2 bytes) is deleted.
    assert_eq!(edits, [(0, 4, 0, 0), (5, 9, 1, 2), (10, 12, 3, 3)]);
}

#[test]
fn whitespace_next_to_multibyte_characters() {
    assert_eq!(collapse("\u{1F600}\u{2003}\u{2003}\u{10400}"), "\u{1F600} \u{10400}");
    assert_eq!(lines("\u{E9}\u{85}\u{E9}"), "\u{E9}\n\u{E9}");
    assert_eq!(paragraphs("\u{4E2D}\u{85}\u{85}\u{6587}"), "\u{4E2D}\n\u{6587}");
}

#[test]
fn keep_paragraphs_run_twice_joins_the_paragraphs() {
    // The paragraph boundary is written as one "\n", the same as OpenNLP,
    // so a second run sees a single line break and unwraps it.
    let once = paragraphs("one\n\ntwo");
    assert_eq!(once, "one\ntwo");
    assert_eq!(paragraphs(&once), "one two");
}
